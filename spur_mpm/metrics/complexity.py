"""What a reconstruction costs: arithmetic, parameters and measured wall time.

Every arm in this project is scored on the same data with the same budget, so
"arm X is better" is only usable if the reader also knows what X costs. Two
kinds of number are needed and they must not be confused:

* **analytical cost** -- multiplies-accumulates (MACs) and parameters, counted
  from the module graph by forward hooks. Deterministic, hardware-free, and the
  only fair way to compare architectures: it is what a reader can recompute.
* **measured cost** -- latency and peak memory on a specific device. This is
  what decides whether a 200-plane stack can be reconstructed in a usable time,
  and it depends on the device, so every call records ``device`` next to the
  number. A CPU latency is reported for completeness and must never be quoted
  as the inference time of the model we deploy.

Why the 2-D view matters for the volume arithmetic: the network is applied
slice-wise, so the per-volume cost is *exactly* ``n_slices * per-slice`` -- no
cross-slice term, no approximation. A true 3-D convolution of the same stack
would pay a ``k`` times larger kernel fan-in per axis (``k^3`` instead of
``k^2``) and would need the whole volume resident; that comparison is stated in
:func:`throughput_for_volume` rather than measured, because measuring it would
require an architecture this project does not have.

Counted conventions, stated so the numbers are comparable with the literature:
a MAC is one multiply-add (``flops = 2 * macs``); the bias add and the
normalisation's elementwise cost are not counted as MACs, except batch-norm
which is charged 2 MAC/element as it is one multiply and one add per element;
attention is charged its projections plus the ``2 * L_q * L_k * E`` score /
weighted-sum products (the softmax is elementwise and free at this precision).

Nothing here imports fvcore/thop/ptflops: they are not installed and must not
become a dependency of the project, and the hook set below covers every layer
this backbone actually uses.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from spur_mpm.models.registry import count_parameters

__all__ = ["count_macs", "measure_latency", "throughput_for_volume",
           "complexity_report"]

_MIB = float(1 << 20)


# ---------------------------------------------------------------------------
# per-module MAC rules
# ---------------------------------------------------------------------------

def _conv2d_macs(module: nn.Conv2d, inputs, output) -> int:
    """One multiply-add per (output element, input channel in the group, tap)."""
    taps = module.kernel_size[0] * module.kernel_size[1]
    return int(output.numel()) * (module.in_channels // module.groups) * taps


def _conv_transpose2d_macs(module: nn.ConvTranspose2d, inputs, output) -> int:
    """Counted from the input side: every input element hits ``taps`` outputs.

    Counting from the produced output instead would credit the zeros that the
    stride scatters into the map, which is not arithmetic the GPU performs.
    """
    x = inputs[0]
    taps = module.kernel_size[0] * module.kernel_size[1]
    return int(x.numel()) * (module.out_channels // module.groups) * taps


def _linear_macs(module: nn.Linear, inputs, output) -> int:
    return int(output.numel()) * module.in_features


def _batchnorm_macs(module: nn.modules.batchnorm._BatchNorm, inputs, output) -> int:
    """2 MAC per element: one scale, one shift (the statistics are a pass of
    their own and are not counted as MACs)."""
    return 2 * int(output.numel())


def _seq_len(x: torch.Tensor, batch_first: bool) -> int:
    if x.dim() == 2:                      # unbatched [L, E]
        return int(x.shape[0])
    return int(x.shape[1]) if batch_first else int(x.shape[0])


def _attention_macs(module: nn.MultiheadAttention, inputs, output) -> int:
    """Projections + score/weighted-sum products, batched dimension irrelevant."""
    q, k, v = inputs[0], inputs[1], inputs[2]
    if isinstance(q, tuple):              # callers may pass one (q, k, v) tuple
        q, k, v = q
    e = int(module.embed_dim)
    lq = _seq_len(q, bool(module.batch_first))
    lk = _seq_len(k, bool(module.batch_first))
    lv = _seq_len(v, bool(module.batch_first))
    proj = lq * e * e + lk * int(module.kdim) * e + lv * int(module.vdim) * e
    out_proj = lq * e * e
    scores = 2 * lq * lk * e              # Q.K^T then the weighted sum with V
    return int(proj + out_proj + scores)


# (module class -> (reported name, mac rule)); first match wins, so a module is
# charged once even when a class has subclasses.
_MAC_RULES: tuple[tuple[type, str, object], ...] = (
    (nn.Conv2d, "Conv2d", _conv2d_macs),
    (nn.ConvTranspose2d, "ConvTranspose2d", _conv_transpose2d_macs),
    (nn.Linear, "Linear", _linear_macs),
    (nn.MultiheadAttention, "MultiheadAttention", _attention_macs),
    (nn.modules.batchnorm._BatchNorm, "BatchNorm", _batchnorm_macs),
)


def _rule_for(module: nn.Module):
    for cls, name, rule in _MAC_RULES:
        if isinstance(module, cls):
            return name, rule
    return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_device(device) -> torch.device:
    """A usable device: a ``cuda`` request degrades to cpu when there is none.

    Benchmark scripts must run (and say so) on a laptop, not crash, because the
    MACs and parameters are device-independent and still useful there.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _sig(value: float, digits: int = 4) -> float:
    """Round to ``digits`` significant digits (a report number, not a tensor)."""
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return float(value)
    if value == 0:
        return 0.0
    return float(f"{float(value):.{digits}g}")


def _to_device(example, device: torch.device):
    if isinstance(example, torch.Tensor):
        return example.to(device)
    if isinstance(example, (list, tuple)):
        return type(example)(_to_device(e, device) for e in example)
    if isinstance(example, dict):
        return {k: _to_device(v, device) for k, v in example.items()}
    return example


def _dummy(shape, device: torch.device) -> torch.Tensor:
    return torch.zeros(tuple(int(s) for s in shape), device=device)


# ---------------------------------------------------------------------------
# analytical cost
# ---------------------------------------------------------------------------

def count_macs(model: nn.Module, input_shape, *, device="cpu") -> dict:
    """Count MACs and parameters of one forward pass of shape ``input_shape``.

    ``input_shape`` is what the model's ``forward`` expects -- for this project
    ``(B, F*Z, H, W)``, the arm's feature stack flattened to channels (see
    ``ModelInputs.stack``). One forward pass is run with forward hooks on the
    layers that do real arithmetic, and every hook is removed afterwards: a
    leftover hook would silently double the next count.

    Parameters come from :func:`spur_mpm.models.registry.count_parameters`, so
    this module cannot drift from the number the run log records.

    Returns ``{"macs", "flops", "params", "by_module", "input_shape"}`` where
    ``by_module`` maps a layer class name to its own MAC total -- what tells a
    reader whether the backbone or the fusion stem is the expensive part.
    """
    dev = _resolve_device(device)
    shape = tuple(int(s) for s in input_shape)
    totals: dict[str, int] = {}
    handles = []

    for module in model.modules():
        rule = _rule_for(module)
        if rule is None:
            continue
        name, fn = rule

        def hook(mod, inputs, output, _name=name, _fn=fn):
            totals[_name] = totals.get(_name, 0) + max(0, int(_fn(mod, inputs, output)))

        handles.append(module.register_forward_hook(hook))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(_dummy(shape, dev))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    macs = int(sum(totals.values()))
    return {"macs": macs, "flops": 2 * macs,
            "params": int(count_parameters(model)),
            "by_module": dict(sorted(totals.items())),
            "input_shape": shape}


# ---------------------------------------------------------------------------
# measured cost
# ---------------------------------------------------------------------------

def measure_latency(model: nn.Module, example_input, *, device="cpu", warmup=3,
                    iters=10, amp=False, cudnn_benchmark=None) -> dict:
    """Wall time of one forward pass, as a distribution rather than a mean.

    A mean latency is the one statistic that hides the behaviour that matters
    (a few slow iterations when the GPU clocks drop or a workspace is
    reallocated), so median / p10 / p90 are returned and the caller decides
    which to quote. ``torch.inference_mode`` and ``model.eval()`` are used so
    autograd bookkeeping is not measured; the model is moved to ``device`` and
    left in the train/eval mode it arrived in.

    CUDA is synchronised before and after every iteration -- without that the
    kernel launch is timed, not the kernel. ``peak_memory_mb`` is the extra
    peak allocation of the measured iterations over what was already resident
    (weights and input), and is ``None`` on cpu where the figure would be
    meaningless.

    ``cudnn_benchmark`` is applied for the measurement and restored afterwards,
    so a benchmark cannot silently change the timing of everything after it.
    """
    dev = _resolve_device(device)
    model.to(dev)
    example = _to_device(example_input, dev)
    use_amp = bool(amp) and dev.type == "cuda"
    was_training = model.training
    prev_benchmark = torch.backends.cudnn.benchmark
    if cudnn_benchmark is not None and dev.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    def step() -> None:
        with torch.autocast(dev.type, enabled=use_amp):
            model(example)

    times: list[float] = []
    baseline = 0
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(max(0, int(warmup))):
                step()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                baseline = int(torch.cuda.memory_allocated(dev))
                torch.cuda.reset_peak_memory_stats(dev)
            for _ in range(max(1, int(iters))):
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                step()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                times.append((time.perf_counter() - t0) * 1000.0)
            if dev.type == "cuda":
                peak = int(torch.cuda.max_memory_allocated(dev))
                peak_mb = max(0, peak - baseline) / _MIB
            else:
                peak_mb = None
    finally:
        model.train(was_training)
        if cudnn_benchmark is not None and dev.type == "cuda":
            torch.backends.cudnn.benchmark = prev_benchmark

    arr = np.asarray(times, dtype=float)
    p10, median, p90 = (float(v) for v in np.percentile(arr, [10, 50, 90]))
    return {"device": str(dev),
            "n_iters": int(arr.size),
            "warmup": int(warmup),
            "amp": use_amp,
            "median_ms": median,
            "mean_ms": float(arr.mean()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p10_ms": p10,
            "p90_ms": p90,
            "fps": (1000.0 / median) if median > 0 else float("inf"),
            "peak_memory_mb": peak_mb}


# ---------------------------------------------------------------------------
# volume arithmetic
# ---------------------------------------------------------------------------

def throughput_for_volume(model: nn.Module, volume_shape, *, in_channels,
                          patch=256, device="cuda", warmup=3, iters=10,
                          amp=False) -> dict:
    """Extrapolate a measured patch latency to a whole volume.

    ``volume_shape`` is ``(Z, H, W)`` for the project's data (a Dryad stack is
    about ``(200, 512, 512)``), or ``(C, Z, H, W)`` if the channel count is
    worth stating explicitly. The network is 2-D and sees one ``patch x patch``
    tile at a time, so the arithmetic is exact and has no hidden factor:

        patches_per_slice = ceil(H/patch) * ceil(W/patch)
        per-slice time    = patches_per_slice * measured patch latency
        per-volume time   = Z * per-slice time

    There is no cross-slice term to model because there is no cross-slice
    computation: neighbouring planes enter as *channels* of the same 2-D
    convolution. A true 3-D convolution over the same stack would be a
    different cost class -- ``k^3`` taps per voxel instead of ``k^2`` (e.g. 27
    instead of 9 at ``k=3``), i.e. roughly ``k`` times the arithmetic per voxel,
    plus the whole volume resident in memory rather than one plane -- which is
    precisely why the slice-wise carrier was chosen. This function reports the
    exact product for the 2-D model only; nothing here estimates a 3-D network.

    The extrapolation assumes the tiles are processed one at a time
    (``batch = 1``), which is how the evaluation pipeline runs: batching tiles
    would raise throughput on a GPU that is not saturated and is not what the
    reported per-volume time should assume.
    """
    dev = _resolve_device(device)
    dims = tuple(int(d) for d in volume_shape)
    if len(dims) == 4:
        n_z, h, w = dims[1], dims[2], dims[3]
        if dims[0] != int(in_channels):
            raise ValueError(f"volume_shape says {dims[0]} channels but "
                             f"in_channels={in_channels}")
    elif len(dims) == 3:
        n_z, h, w = dims
    else:
        raise ValueError(f"volume_shape must be (Z, H, W) or (C, Z, H, W), "
                         f"got {volume_shape}")

    per_axis = [int(math.ceil(size / int(patch))) for size in (h, w)]
    patches_per_slice = per_axis[0] * per_axis[1]
    example = _dummy((1, int(in_channels), patch, patch), dev)
    latency = measure_latency(model, example, device=dev, warmup=warmup,
                              iters=iters, amp=amp)

    patch_ms = latency["median_ms"]
    slice_ms = patch_ms * patches_per_slice
    volume_s = slice_ms * n_z / 1000.0
    note = ("CPU measurement: useful for relative comparison and for the "
            "arithmetic, but it is NOT the inference time of the deployed model"
            if dev.type == "cpu" else
            f"measured on {dev}; tiles are timed one at a time, matching the "
            "evaluation pipeline")
    return {"device": latency["device"],
            "amp": latency["amp"],
            "in_channels": int(in_channels),
            "patch": int(patch),
            "volume_shape": (n_z, h, w),
            "n_slices": n_z,
            "patches_per_slice": patches_per_slice,
            "patch_latency_ms": patch_ms,
            "slice_ms": slice_ms,
            "volume_s": volume_s,
            "slices_per_s": (1000.0 / slice_ms) if slice_ms > 0 else float("inf"),
            "peak_memory_mb": latency["peak_memory_mb"],
            "latency": latency,
            "note": note}


# ---------------------------------------------------------------------------
# one report per arm
# ---------------------------------------------------------------------------

def complexity_report(model: nn.Module, in_channels, *, patch=256, n_slices=200,
                      device="cuda", warmup=3, iters=10, amp=False) -> dict:
    """Everything a complexity table needs, for one arm, in one dict.

    The reference volume is ``n_slices`` slices of ``patch x patch`` -- the
    patch the model is actually trained and evaluated on, not a whole 512x512
    plane. Use :func:`throughput_for_volume` with the true volume shape when
    the tiling matters (a 512x512 plane is four 256x256 tiles).

    Floats are rounded to 4 significant digits: these are report numbers, and
    carrying 17 digits of a latency makes the table harder to read without
    making it more true. Integers (MACs, parameters) stay exact.
    """
    dev = _resolve_device(device)
    model.to(dev)
    shape = (1, int(in_channels), int(patch), int(patch))
    cost = count_macs(model, shape, device=dev)
    tput = throughput_for_volume(model, (int(n_slices), int(patch), int(patch)),
                                 in_channels=int(in_channels), patch=int(patch),
                                 device=dev, warmup=warmup, iters=iters, amp=amp)
    slice_ms = tput["slice_ms"]
    return {"params": cost["params"],
            "params_m": _sig(cost["params"] / 1e6),
            "in_channels": int(in_channels),
            "patch": int(patch),
            "n_slices": int(n_slices),
            "input_shape": cost["input_shape"],
            "macs_per_slice": cost["macs"],
            "flops_per_slice": cost["flops"],
            "macs_per_volume": cost["macs"] * int(n_slices),
            "flops_per_volume": cost["flops"] * int(n_slices),
            "macs_per_slice_g": _sig(cost["macs"] / 1e9),
            "macs_per_volume_t": _sig(cost["macs"] * int(n_slices) / 1e12),
            "by_module": cost["by_module"],
            "device": tput["device"],
            "is_cpu": tput["device"] == "cpu",
            "amp": tput["amp"],
            "latency_ms_per_slice": _sig(slice_ms),
            "sec_per_slice": _sig(slice_ms / 1000.0),
            "sec_per_volume": _sig(tput["volume_s"]),
            "peak_memory_mb": (None if tput["peak_memory_mb"] is None
                               else _sig(tput["peak_memory_mb"])),
            "latency": {"median_ms": _sig(tput["patch_latency_ms"]),
                        "p10_ms": _sig(tput["latency"]["p10_ms"]),
                        "p90_ms": _sig(tput["latency"]["p90_ms"]),
                        "mean_ms": _sig(tput["latency"]["mean_ms"]),
                        "fps": _sig(tput["latency"]["fps"]),
                        "n_iters": tput["latency"]["n_iters"],
                        "device": tput["latency"]["device"]},
            "note": tput["note"]}

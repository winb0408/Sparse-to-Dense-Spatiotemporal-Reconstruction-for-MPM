"""Cost table for every arm: parameters, MACs and measured wall time.

    python -m spur_mpm.scripts.benchmark --markdown \
        --arms S0 S1 S2 S1g SS1 SS1c --patch 256 --n-slices 200

A reconstruction result is only half a result: the reader also has to know what
the arm costs, and "the network is bigger" is not a cost statement. This script
produces, in one run, the four numbers a complexity table needs -- parameters,
MACs per slice, MACs per volume, and seconds per volume -- for the same arms,
the same patch and the same backbone width the experiments used (taken from
``configs/base.yaml``, never from a constant repeated here, so the table cannot
drift from the training budget).

Two traps this script is built around:

* **A0 / A1 are not networks.** They are bilinear interpolation and a
  fill-first Wiener solve. There is no forward pass to time and no parameter to
  count, so they are reported with ``params = 0`` and ``latency = "n/a"`` rather
  than being dropped: the reader must see the parameter-free floor in the same
  table as the arms it is the floor *for*.
* **A latency without a device is not a number.** On cpu the measurement is
  recorded and flagged as such, because cpu milliseconds say nothing about the
  deployed model; and a machine with no free GPU must still produce the MACs and
  the parameters instead of crashing.

GPU selection: the GPUs are shared, so the device is probed with ``nvidia-smi``
before anything is allocated and the least-used free one is picked. A card that
is already carrying another user's job is not a benchmarking device -- timing on
it would measure their workload, and evicting them is not ours to do.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from spur_mpm.common.config import load_config
from spur_mpm.metrics.complexity import complexity_report
from spur_mpm.models.registry import ARMS, build_model, get_arm

__all__ = ["benchmark_arm", "benchmark_all", "format_markdown", "free_gpus",
           "main"]

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"

# GPUs below this usage are considered idle. A card that merely holds a leftover
# context (a few hundred MiB) is still usable; one with a real job on it is not.
GPU_IDLE_MIB = 1024
# Cards that belong to someone else and are never to be taken, whatever their
# reported usage: index -> reason. Recorded here rather than only in a chat
# message, so the script cannot quietly start using one.
RESERVED_GPUS = {3: "in use by another user"}


def _nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def free_gpus() -> list[int]:
    """Indices of GPUs with no real job on them, cheapest (lowest) first."""
    out = []
    for line in _nvidia_smi().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(re.sub(r"[^0-9-]", "", parts[0]))
            used = int(re.sub(r"[^0-9]", "", parts[1]) or -1)
        except ValueError:
            continue
        if index in RESERVED_GPUS or used < 0 or used > GPU_IDLE_MIB:
            continue
        out.append(index)
    return sorted(out)


def input_channels(arm, context_depth: int) -> int:
    """Channels the network sees: ``F * Z``, doubled for the dual arm.

    ``ComplementaryDual`` is handed both branches' stacks concatenated, so it is
    twice as wide at the input while each branch stays a single arm -- the
    input-shape arithmetic has to know this or the MAC count is wrong by 2x.
    """
    return len(arm.features) * int(context_depth) * (2 if arm.dual else 1)


def benchmark_arm(arm_name: str, *, patch: int, n_slices: int, device: str,
                  model_cfg: dict, context_depth: int, warmup: int, iters: int,
                  amp: bool) -> dict:
    """One table row. Never raises for an arm that has no network."""
    arm = get_arm(arm_name)
    row = {"arm": arm.name, "features": list(arm.features),
           "supervision": arm.supervision, "learned": bool(arm.needs_training),
           "in_channels": None, "params": 0, "params_m": 0.0,
           "macs_per_slice": None, "macs_per_volume": None,
           "latency_ms_per_slice": None, "sec_per_volume": None,
           "device": device, "note": None}
    if not arm.needs_training:
        # Baselines: interpolation / Wiener. Reported, not built.
        row["note"] = (f"no network ({arm.warm_start}): the parameter-free "
                       "floor, MACs and latency are not applicable")
        return row

    model = build_model(arm, context_depth,
                        base=int(model_cfg["base"]),
                        window=int(model_cfg["window"]),
                        depths=tuple(model_cfg["depths"]),
                        heads=tuple(model_cfg["heads"]))
    report = complexity_report(model, input_channels(arm, context_depth),
                               patch=patch, n_slices=n_slices, device=device,
                               warmup=warmup, iters=iters, amp=amp)
    row.update({"in_channels": report["in_channels"],
                "params": report["params"],
                "params_m": report["params_m"],
                "macs_per_slice": report["macs_per_slice"],
                "macs_per_volume": report["macs_per_volume"],
                "latency_ms_per_slice": report["latency_ms_per_slice"],
                "sec_per_volume": report["sec_per_volume"],
                "peak_memory_mb": report["peak_memory_mb"],
                "device": report["device"],
                "by_module": report["by_module"],
                "latency": report["latency"],
                "note": report["note"]})
    return row


def benchmark_all(arms, *, patch: int, n_slices: int, device: str,
                  model_cfg: dict, context_depth: int, warmup: int, iters: int,
                  amp: bool) -> dict:
    """Benchmark each requested arm; an unbuildable arm becomes a note, not a
    traceback, so one broken arm cannot cost the whole table."""
    rows = []
    for name in arms:
        try:
            rows.append(benchmark_arm(name, patch=patch, n_slices=n_slices,
                                      device=device, model_cfg=model_cfg,
                                      context_depth=context_depth, warmup=warmup,
                                      iters=iters, amp=amp))
        except Exception as exc:                      # noqa: BLE001 - reported
            rows.append({"arm": str(name), "learned": None, "params": 0,
                         "params_m": 0.0, "macs_per_slice": None,
                         "macs_per_volume": None,
                         "latency_ms_per_slice": None, "sec_per_volume": None,
                         "device": device, "skipped": True,
                         "note": f"could not build/benchmark: {exc}"})
    return {"generated": datetime.now().isoformat(timespec="seconds"),
            "device": device,
            "on_cpu": str(device).startswith("cpu"),
            "device_note": ("latency measured on cpu: NOT the inference time of "
                            "the deployed model" if str(device).startswith("cpu")
                            else f"latency measured on {device}"),
            "patch": int(patch), "n_slices": int(n_slices),
            "context_depth": int(context_depth), "model": dict(model_cfg),
            "arms": rows}


def _fmt(value, spec: str = ".4g") -> str:
    return "n/a" if value is None else format(value, spec)


def format_markdown(payload: dict) -> str:
    """The complexity table, plus the caveat that makes it readable."""
    dev = payload["device"]
    lines = ["| arm | params(M) | MACs/slice(G) | MACs/volume(T) "
             "| latency ms/slice | s/volume |",
             "|---|---|---|---|---|---|"]
    for row in payload["arms"]:
        macs_slice = row.get("macs_per_slice")
        macs_vol = row.get("macs_per_volume")
        lines.append(
            f"| {row['arm']} | {_fmt(row.get('params_m'), '.3f')} "
            f"| {_fmt(None if macs_slice is None else macs_slice / 1e9)} "
            f"| {_fmt(None if macs_vol is None else macs_vol / 1e12)} "
            f"| {_fmt(row.get('latency_ms_per_slice'), '.3f')} "
            f"| {_fmt(row.get('sec_per_volume'), '.3f')} |")
    lines += ["",
              f"Device: `{dev}`. Tiles of {payload['patch']}x{payload['patch']} "
              f"px, {payload['n_slices']} per volume, so MACs/volume = "
              f"{payload['n_slices']} x MACs/slice: the network is 2-D and "
              "applied slice-wise.",
              ""]
    if payload["on_cpu"]:
        lines.append(f"**{payload['device_note']}.** Parameters and MACs are "
                     "exact; the millisecond columns are for relative "
                     "comparison only.")
        lines.append("")
    caveats = [r for r in payload["arms"]
               if r.get("note") and r.get("learned") is not True]
    for row in caveats:
        lines.append(f"- `{row['arm']}`: {row['note']}")
    return "\n".join(lines)


def default_arms() -> list[str]:
    """Every supervised arm in the registry, in registry order."""
    return [a.name for a in ARMS.values()
            if a.needs_training and a.supervision == "dense"]


def _resolve_device(requested: str | None) -> str:
    """Honour ``--device``; otherwise take a free GPU, else fall back to cpu.

    A bare index (``--device 2``) is accepted like everywhere else in the repo
    and means ``cuda:2``. It used to be passed through verbatim, and torch then
    rejected it with "Invalid device string: '2'" *per arm*, which the table
    reported as "could not build/benchmark" for every network.
    """
    if requested:
        return f"cuda:{requested}" if str(requested).isdigit() else str(requested)
    if not torch.cuda.is_available():
        print("[benchmark] no CUDA device visible: benchmarking on cpu "
              "(latency will NOT be the deployment inference time)")
        return "cpu"
    usable = free_gpus()
    if not usable:
        print("[benchmark] no free GPU on this host: benchmarking on cpu "
              "(latency will NOT be the deployment inference time)")
        return "cpu"
    index = usable[0]
    # CUDA_VISIBLE_DEVICES must be set before the first allocation, and makes
    # the chosen card rank 0 for everything downstream.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    print(f"[benchmark] using physical GPU {index} "
          f"(free set {usable}; reserved {sorted(RESERVED_GPUS)}); "
          "in this process it is cuda:0")
    return "cuda:0"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=None,
                   help="arm names (default: every supervised arm in the registry)")
    p.add_argument("--patch", type=int, default=256, help="tile side, px")
    p.add_argument("--n-slices", type=int, default=200,
                   help="slices per volume (a Dryad stack is ~200)")
    p.add_argument("--device", default=None,
                   help="cpu | cuda | cuda:N (default: a free GPU, else cpu)")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON path (default runs/bench/<name>.json)")
    p.add_argument("--name", default=None, help="stem for the default --out")
    p.add_argument("--markdown", action="store_true",
                   help="print the table to stdout as well")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--amp", action="store_true",
                   help="time under autocast (cuda only)")
    p.add_argument("--base-config", type=Path, default=BASE_CONFIG,
                   help="config the backbone width/patch budget is read from")
    args = p.parse_args(argv)

    cfg = load_config(args.base_config)
    model_cfg = {"base": int(cfg["model"]["base"]),
                 "window": int(cfg["model"]["window"]),
                 "depths": [int(d) for d in cfg["model"]["depths"]],
                 "heads": [int(h) for h in cfg["model"]["heads"]]}
    context_depth = int(cfg["data"].get("context_slices", 1) or 1)

    arms = args.arms or default_arms()
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        p.error(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")
    device = _resolve_device(args.device)

    payload = benchmark_all(arms, patch=args.patch, n_slices=args.n_slices,
                            device=device, model_cfg=model_cfg,
                            context_depth=context_depth, warmup=args.warmup,
                            iters=args.iters, amp=args.amp)

    name = args.name or (f"bench_{str(device).replace(':', '')}"
                         f"_p{args.patch}_z{args.n_slices}")
    out_path = args.out or Path("runs") / "bench" / f"{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload["out"] = str(out_path)
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    table = format_markdown(payload)
    if args.markdown:
        print(table)
    print(f"wrote {out_path} ({len(payload['arms'])} arms, device "
          f"{payload['device']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

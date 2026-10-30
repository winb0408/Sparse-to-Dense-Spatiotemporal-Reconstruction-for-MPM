"""The sparse-acquisition ladder: dense data in, measured/unmeasured support out.

Every reconstruction number this project reports is defined on a *virtual*
acquisition (plan §3.2): a densely acquired stack is the reference, and a
sampling protocol decides which positions the scanner would have measured. This
module is the single place that turns one into the other, and — because the whole
claim rests on the construction being right — it also ships the checks that let a
reader confirm it without reading the reconstruction code:

    Y = M . D     (no optics, no noise),   exactly, voxel for voxel

with ``D`` the calibrated dense reference and ``M`` the support. Consequences
that are asserted rather than trusted:

* ``M = 0`` positions are exactly ``0.0`` -- they were never measured, and a
  non-zero value there would hand the model the answer it is asked to infer;
* ``M = 1`` positions equal the dense value exactly when noise is off, and are
  perturbed only there when noise is on;
* a kept row is kept **across all of X** (that is what makes line skipping a real
  scanning scheme), a kept plane across all of (H, W);
* the lattice is anchored at global index 0 with a declared phase, so a patch
  cropped anywhere in the volume sits on the same lattice as the whole stack.

Two dimensionalities, one rule
------------------------------
2-D data (one optical plane, or a TYX movie) can only have **Y-direction**
sparsity: rows are scanned or they are not. 3-D data adds **Z** (whole planes
skipped) and **Y+Z** (both). Depth-axis modes on a 2-D array are rejected rather
than silently reinterpreted.

Sparsity is not speed-up
------------------------
``scan_line_fraction`` is the honest cost model for point scanning: X is the fast
axis, so a row that is scanned at all costs a full row of dwell time. For the
``y``/``z``/``yz`` protocols that fraction equals the measured-voxel fraction
(X is never subsampled); for ``xy`` it does not, which is exactly why ``xy`` is a
stress test and not a protocol we claim a speed-up for.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from spur_mpm.physics.acquisition import (
    AcquisitionConfig,
    NoiseConfig,
    Observation,
    observe,
)
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec, kept_planes, kept_rows, make_mask

__all__ = [
    "LEVELS_2D", "LEVELS_3D", "levels_for", "measured_mask", "build_sparse",
    "build_ladder", "verify_construction", "acquisition_cost", "write_entry",
    "read_manifest", "LADDER_NOTE",
]

# --- the ladder -------------------------------------------------------------
# Fractions in brackets are the ideal measured-voxel fraction 1/ky, 1/kz, 1/(ky*kz).
# The rungs are chosen so the three families can be compared at a *matched*
# budget: 25 % exists in all three (y_k4, z_k4, yz_k2x2) and 12.5 % in two
# (y_k8, yz_k2x4). Comparing Y-skipping against Z-skipping at different measured
# fractions would answer a different question than the one being asked.
LEVELS_2D: dict[str, SamplingSpec] = {
    "y_k2": SamplingSpec(mode="y", ky=2),      # 50.0 %
    "y_k3": SamplingSpec(mode="y", ky=3),      # 33.3 %
    "y_k4": SamplingSpec(mode="y", ky=4),      # 25.0 %
    "y_k8": SamplingSpec(mode="y", ky=8),      # 12.5 %
}

LEVELS_3D: dict[str, SamplingSpec] = {
    "y_k2": SamplingSpec(mode="y", ky=2),          # 50.0 %
    "y_k3": SamplingSpec(mode="y", ky=3),          # 33.3 %
    "y_k4": SamplingSpec(mode="y", ky=4),          # 25.0 %
    "y_k8": SamplingSpec(mode="y", ky=8),          # 12.5 %
    "z_k2": SamplingSpec(mode="z", kz=2),          # 50.0 %
    "z_k3": SamplingSpec(mode="z", kz=3),          # 33.3 %
    "z_k4": SamplingSpec(mode="z", kz=4),          # 25.0 %
    "yz_k2x2": SamplingSpec(mode="yz", ky=2, kz=2),  # 25.0 %
    "yz_k2x4": SamplingSpec(mode="yz", ky=2, kz=4),  # 12.5 %
}

LADDER_NOTE = (
    "virtual_sparse: the dense stack is the reference, the mask is the protocol. "
    "No measured speed-up, dose reduction or hardware scan is implied."
)


def levels_for(ndim: int) -> dict[str, SamplingSpec]:
    """Levels that make sense for an ``ndim``-dimensional image (2 or 3)."""
    if int(ndim) == 2:
        return dict(LEVELS_2D)
    if int(ndim) == 3:
        return dict(LEVELS_3D)
    raise ValueError(f"only 2-D and 3-D data exist here, got ndim={ndim}")


def measured_mask(shape: tuple[int, ...], spec: SamplingSpec,
                  phase: tuple[int, ...] = (0, 0, 0)) -> np.ndarray:
    """Boolean support ``M`` for ``shape`` (``[H,W]`` or ``[Z,H,W]``)."""
    return make_mask(spec, tuple(shape), phase).numpy()


# --- construction -----------------------------------------------------------

def build_sparse(dense: np.ndarray, spec: SamplingSpec, *,
                 psf: PSFConfig | None = None, noise: NoiseConfig | None = None,
                 phase: tuple[int, ...] = (0, 0, 0)) -> dict:
    """Realise one virtual acquisition of ``dense`` under ``spec``.

    ``dense`` is the calibrated reference on ``[0, 1]`` (float, any ndim in
    {2, 3}). Returns the observation plus the bookkeeping a reader needs to check
    it: the support, the measured count/fraction, the ideal fraction the protocol
    promises, and the scan-line cost.

    Nothing is interpolated and nothing is filled in: the output ``y`` is the
    measurement, zeros elsewhere. Filling the gaps here would make every
    downstream "reconstruction" score partly a copy.
    """
    arr = np.asarray(dense, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected [H,W] or [Z,H,W], got {arr.shape}")
    obs: Observation = observe(
        torch.from_numpy(arr),
        AcquisitionConfig(sampling=spec, psf=psf or PSFConfig(),
                          noise=noise or NoiseConfig(), phase=tuple(phase)),
    )
    y = obs.y.numpy()
    m = obs.mask.numpy()
    cost = acquisition_cost(spec, arr.shape, phase=phase)
    return {
        "y": y,
        "mask": m,
        "shape": tuple(arr.shape),
        "ndim": arr.ndim,
        "level": spec.tag,
        "spec": spec.to_dict(),
        "phase": tuple(phase),
        "n_total": int(m.size),
        "n_measured": int(m.sum()),
        "measured_fraction": float(m.mean()),
        "ideal_fraction": float(spec.ideal_fraction()),
        **cost,
    }


def build_ladder(dense: np.ndarray, *, psf: PSFConfig | None = None,
                 noise: NoiseConfig | None = None,
                 phase: tuple[int, ...] = (0, 0, 0),
                 levels: dict[str, SamplingSpec] | None = None) -> dict[str, dict]:
    """Every level for one image, keyed by level name."""
    table = levels if levels is not None else levels_for(np.ndim(dense))
    return {name: build_sparse(dense, spec, psf=psf, noise=noise, phase=phase)
            for name, spec in table.items()}


def acquisition_cost(spec: SamplingSpec, shape: tuple[int, ...],
                     phase: tuple[int, ...] = (0, 0, 0)) -> dict:
    """Kept rows / planes and the honest scan-time fraction.

    ``scan_line_fraction`` counts (Z, Y) line pairs that contain at least one
    measured voxel, divided by all line pairs. X is the fast axis, so a row that
    is scanned at all costs a full row: this is the cost model of point scanning,
    not a voxel count.
    """
    m = make_mask(spec, tuple(shape), phase).numpy()
    if m.ndim == 2:
        lines = m.any(axis=-1)
        return {"scan_line_fraction": float(lines.mean()),
                "n_scan_lines": int(lines.sum()),
                "n_scan_lines_total": int(lines.size)}
    lines = m.any(axis=-1)                      # [Z, H]
    rows = kept_rows(spec, int(shape[-2]), int(phase[-2]) if len(phase) >= 2 else 0)
    planes = kept_planes(spec, int(shape[0]), int(phase[0]))
    return {"scan_line_fraction": float(lines.mean()),
            "n_scan_lines": int(lines.sum()),
            "n_scan_lines_total": int(lines.size),
            "kept_row_pattern": _lattice_pattern(rows),
            "kept_plane_pattern": _lattice_pattern(planes)}


def _lattice_pattern(keep: torch.Tensor) -> dict | None:
    """Compact description of a periodic boolean vector: ``start/step/count``."""
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    if idx.numel() == 0:
        return None
    if idx.numel() == 1:
        return {"indices": [int(idx[0])]}
    step = int(idx[1] - idx[0])
    regular = bool(torch.equal(idx, idx[0] + step * torch.arange(idx.numel())))
    if not regular:
        return {"indices": [int(i) for i in idx]}
    return {"start": int(idx[0]), "step": step, "count": int(idx.numel())}


# --- verification -----------------------------------------------------------

def verify_construction(dense: np.ndarray, y: np.ndarray, mask: np.ndarray,
                        spec: SamplingSpec, *, phase: tuple[int, ...] = (0, 0, 0),
                        noisy: bool = False, atol: float = 0.0) -> dict:
    """Check one construction against the definition ``Y = M . D``.

    Returns a dict of named checks (all booleans) plus the numbers they were
    computed from, so a failing check can be read, not just noticed. ``ok`` is the
    conjunction of them.

    With ``noisy=False`` the measurement must equal the reference **exactly** on
    the support (``atol=0``), which is the strongest statement available and the
    one that catches a fill or a resample creeping in. With ``noisy=True`` only
    the structural checks still hold -- noise is allowed to move measured values
    -- and the size of that movement is reported as ``noise_delta_max`` instead
    of being asserted away.
    """
    # float32, not float64: these are full volumes (202 x 512 x 512 x 4 B = 211 MB)
    # and an upcast would double the footprint of every check for no extra truth.
    d = np.asarray(dense, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32)
    m = np.asarray(mask).astype(bool)

    checks: dict[str, bool] = {}
    checks["shape_matches"] = d.shape == yy.shape == m.shape
    checks["mask_is_boolean"] = m.dtype == bool
    checks["unmeasured_is_exactly_zero"] = bool(
        np.all(yy[~m] == 0.0)) if m.any() else bool(np.all(yy == 0.0))
    if noisy:
        checks["measured_equals_dense"] = None       # not a property under noise
    else:
        checks["measured_equals_dense"] = bool(
            np.allclose(yy[m], d[m], atol=atol, rtol=0.0))

    # Geometry: a kept row is kept across the whole fast axis (that is what makes
    # line skipping a scanning scheme), and in ``z`` mode a kept plane is kept
    # across the whole field. In ``yz`` a kept plane is *not* full-field by
    # design, so only the row statement applies -- the factorisation itself is
    # already covered by ``lattice_matches_spec``.
    row_axis_full = m.all(axis=-1) == m.any(axis=-1)
    checks["kept_rows_are_full_width"] = (
        bool(row_axis_full.all()) if spec.mode in ("y", "yz") else True)
    if m.ndim == 3 and spec.mode == "z":
        plane_axis_full = m.all(axis=(1, 2)) == m.any(axis=(1, 2))
        checks["kept_planes_are_full_field"] = bool(plane_axis_full.all())
    else:
        checks["kept_planes_are_full_field"] = True

    checks["lattice_matches_spec"] = bool(
        np.array_equal(m, make_mask(spec, d.shape, phase).numpy()))

    measured = float(m.mean())
    ideal = float(spec.ideal_fraction())
    # Discretisation is real: a k=3 lattice over 64 rows keeps 22 lines, not
    # 21.33. The count may be off by at most one line per subsampled axis -- and
    # that statement is only meaningful when X is dense, so ``xy`` is reported
    # but not asserted.
    if spec.mode in ("y", "z", "yz"):
        n_lines = m.sum() / m.shape[-1]
        total_lines = m.size / m.shape[-1]
        # One line of slack per subsampled axis, expressed in scan lines: a
        # row-lattice error of one row repeats on every kept plane, so it costs
        # ``D`` lines, not one.
        depth = m.shape[0] if m.ndim == 3 else 1
        height = m.shape[-2]
        slack = {"y": depth, "z": height, "yz": depth + height}[spec.mode]
        checks["line_count_within_discretisation"] = bool(
            abs(n_lines - ideal * total_lines) <= slack)
        fraction_note = f"kept scan lines vs ideal, ±{slack} line(s) allowed"
    else:
        checks["line_count_within_discretisation"] = None
        fraction_note = "not asserted for the xy stress test (X is not dense)"

    out = dict(checks)
    out.update({
        "measured_fraction": measured,
        "ideal_fraction": ideal,
        "n_measured": int(m.sum()),
        "fraction_note": fraction_note,
        "noise_delta_max": float(np.abs(yy[m] - d[m]).max()) if (noisy and m.any())
        else 0.0,
    })
    out["ok"] = all(bool(v) for v in checks.values() if v is not None)
    return out


# --- persistence ------------------------------------------------------------

def write_entry(out_dir: str | Path, sample_id: str, level: str, built: dict,
                *, write_y: bool = False) -> dict:
    """Write one ``(sample, level)`` construction and return its manifest entry.

    The mask is stored bit-packed (``np.packbits``): it is 1 bit per voxel of
    ground truth about the protocol, and it is what a reader should load to check
    the construction themselves. ``Y`` is *not* stored by default because it is
    exactly ``dense * mask``; ask for it with ``write_y`` when a downstream tool
    wants the array without re-running the builder.
    """
    out_dir = Path(out_dir)
    mask_path = out_dir / "masks" / level / f"{sample_id}.npz"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(np.asarray(built["mask"], dtype=np.uint8).ravel())
    payload = {"packed": packed, "shape": np.asarray(built["shape"], dtype=np.int64)}
    if write_y:
        payload["y"] = np.asarray(built["y"], dtype=np.float16)
    np.savez_compressed(mask_path, **payload)

    entry = {
        "sample_id": sample_id, "level": level, "shape": list(built["shape"]),
        "n_total": built["n_total"], "n_measured": built["n_measured"],
        "measured_fraction": built["measured_fraction"],
        "ideal_fraction": built["ideal_fraction"],
        "scan_line_fraction": built["scan_line_fraction"],
        "n_scan_lines": built["n_scan_lines"],
        "n_scan_lines_total": built["n_scan_lines_total"],
        "kept_row_pattern": built.get("kept_row_pattern"),
        "kept_plane_pattern": built.get("kept_plane_pattern"),
        "spec": built["spec"], "phase": list(built["phase"]),
        "mask_path": str(mask_path),
        "mask_sha1": hashlib.sha1(packed.tobytes()).hexdigest(),
        "y_stored": bool(write_y),
    }
    return entry


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_mask(entry: dict) -> np.ndarray:
    """Unpack the support written by :func:`write_entry` back to bool ``shape``."""
    with np.load(entry["mask_path"]) as z:
        packed = z["packed"]
        shape = tuple(int(v) for v in z["shape"])
    n = int(np.prod(shape))
    return np.unpackbits(packed)[:n].reshape(shape).astype(bool)

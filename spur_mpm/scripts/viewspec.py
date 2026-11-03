"""Frozen view geometry for the qualitative figures.

Why this module exists
----------------------
The panel coordinates used to be re-chosen per figure from the *measured
support mask* (``_view_spec(ref, mask)``). That is fine inside one sparse
protocol and wrong across protocols: the mask picks a different "best covered"
row for ``y_k4`` than for ``z_k4``, so two figures that are supposed to be a
cross-protocol comparison silently look at different rows, and the labels
"XZ given / XZ gap" describe a Y-axis sampling scheme even when the protocol
skips *planes*.

The rules implemented here:

1. **A view spec is built from the dense reference alone.** Same sample, same
   ROI, same cuts for every protocol and every arm. It is JSON-serialisable so
   it can be written once and re-read (``--view-spec``), instead of being
   recomputed (and accidentally re-chosen) per run.
2. **Protocol-dependent panels are derived from the mask and labelled with the
   actual coordinates**, never with a generic "given"/"gap" that may be false
   for this protocol. Whether a panel is a *measured plane*, a *skipped plane*,
   a *measured row* or a *skipped row* is decided by asking the mask.
3. **The dense reference never depends on the protocol.** ``check_targets``
   asserts that the ``target`` arrays of two protocols are numerically
   identical -- if they are not, the data path is broken, not the figure.

Nothing here touches torch: it is pure numpy so the invariants can be tested
without loading a volume cache or a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# A spec is a plain dict so it can be written to JSON as-is. These are the
# fields a reader needs to reproduce a crop; anything else is derived.
SPEC_KEYS = ("sample_id", "shape", "voxel_um", "z_view", "y_view", "x_view",
             "roi", "mip_z", "zoom")


def _round(v: float, nd: int = 6) -> float:
    return float(round(float(v), nd))


def build_view_spec(ref: np.ndarray, *, sample_id: str = "",
                    voxel_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
                    zoom: int = 96, z_view: int | None = None,
                    y_view: int | None = None, x_view: int | None = None,
                    roi: tuple[int, int, int] | None = None) -> dict:
    """Fix the cuts for one sample, from the dense reference only.

    ``z_view`` / ``y_view`` / ``x_view`` default to the middle of the volume.
    The zoom box is centred on the busiest patch of the primary plane -- that
    choice is data-driven but *protocol-independent*, which is the property the
    old implementation lacked: it searched the mask, so every sparsity level got
    its own crop and the figures could not be compared.
    """
    ref = np.asarray(ref)
    if ref.ndim != 3:
        raise ValueError(f"reference must be (Z, Y, X), got shape {ref.shape}")
    z, h, w = ref.shape
    zc = int(0.5 * z) if z_view is None else int(z_view)
    yc = h // 2 if y_view is None else int(y_view)
    xc = w // 2 if x_view is None else int(x_view)
    # Check the named cuts before using them: an explicit coordinate outside the
    # volume would otherwise fail deep inside the ROI search with an IndexError.
    if not (0 <= zc < z and 0 <= yc < h and 0 <= xc < w):
        raise ValueError(f"view cuts (z={zc}, y={yc}, x={xc}) outside the "
                         f"volume {(z, h, w)}")
    n = int(min(zoom, h, w))
    if roi is None:
        from scipy.ndimage import uniform_filter
        plane = np.nan_to_num(ref[zc], nan=0.0)
        sm = uniform_filter(plane, size=max(n // 3, 3), mode="nearest")
        fy, fx = np.unravel_index(int(np.argmax(sm)), sm.shape)
        y0 = int(np.clip(fy - n // 2, 0, max(h - n, 0)))
        x0 = int(np.clip(fx - n // 2, 0, max(w - n, 0)))
    else:
        y0, x0, n = int(roi[0]), int(roi[1]), int(roi[2])
    return validate_view_spec({
        "sample_id": sample_id,
        "shape": [int(z), int(h), int(w)],
        "voxel_um": [_round(v) for v in voxel_um],
        "z_view": zc, "y_view": yc, "x_view": xc,
        "roi": [y0, x0, n],
        "mip_z": [0, int(z)],
        "zoom": n,
    }, (z, h, w))


def validate_view_spec(spec: dict, shape: tuple[int, int, int]) -> dict:
    """Check the cuts against the volume they will be applied to.

    A spec that does not fit is a hard error: silently clipping it is how a
    comparison figure ends up showing a different physical ROI per protocol.
    """
    z, h, w = (int(s) for s in shape)
    if list(spec.get("shape", [])) != [z, h, w]:
        raise ValueError(f"view spec says shape {spec.get('shape')} but the "
                         f"volume is {(z, h, w)}")
    y0, x0, n = (int(v) for v in spec["roi"])
    if not (0 <= int(spec["z_view"]) < z and 0 <= int(spec["y_view"]) < h
            and 0 <= int(spec["x_view"]) < w):
        raise ValueError(f"view spec cuts outside the volume: {spec}")
    if not (0 <= y0 <= h - n and 0 <= x0 <= w - n and n > 0):
        raise ValueError(f"view spec ROI {spec['roi']} does not fit {(h, w)}")
    mz0, mz1 = (int(v) for v in spec["mip_z"])
    if not (0 <= mz0 < mz1 <= z):
        raise ValueError(f"view spec MIP range {spec['mip_z']} outside {z}")
    return spec


def fingerprint(files: dict, cfg: dict) -> dict:
    """What the figure was derived from: the files' identity, and the config.

    A view spec without its provenance is not reproducible -- the same cuts on a
    rebuilt cache are a different experiment.  The hash is over the merged
    config, so any setting that changes the input (sampling, PSF, noise,
    normalisation) changes the fingerprint.
    """
    out: dict[str, dict | None] = {}
    for name, p in files.items():
        p = Path(p)
        if not p.exists():
            out[name] = None
            continue
        st = p.stat()
        out[name] = {"path": str(p), "bytes": int(st.st_size),
                     "mtime": int(st.st_mtime)}
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return {"files": out,
            "config_sha256": hashlib.sha256(blob).hexdigest()}


def projection_note(columns: list[dict] | None) -> str:
    """What a figure projects, said from the panels it actually draws.

    This used to be a constant -- "none (single XY/XZ/YZ slices; mip_z kept for
    a future projection)" -- written when the layout had no projection panel.
    The layout then gained a MIP-z column and the constant stayed, so every
    ``view_spec.json`` described a figure without a projection while the PNG had
    one. A note that is typed once and reused is a claim nothing checks; this one
    is derived from the column list, so it cannot drift from the panels again.
    """
    if not columns:
        return "not recorded: no column list was given to the view spec"
    axes = [str(c.get("axis")) for c in columns]
    slices = [{"xy": "XY", "xz": "XZ", "yz": "YZ", "zoom": "Zoom"}[a]
              for a in axes if a in ("xy", "xz", "yz", "zoom")]
    single = ", ".join(dict.fromkeys(slices)) or "none"
    if "mipz" in axes:
        return (f"MIP-z over the recorded z range (one fixed column); {single} "
                "are single slices")
    return f"no projection: every panel is a single slice ({single})"


def annotate(spec: dict, *, windows: dict | None = None,
             provenance: dict | None = None,
             columns: list[dict] | None = None) -> dict:
    """Return a copy of ``spec`` carrying the windows and provenance it was used with.

    Kept inside the spec (rather than in a sidecar next to it) because §4.4.1 D
    asks for one file per sample that says *both* where the cuts are and what
    display window every protocol must reuse.
    """
    out = dict(spec)
    if windows is not None:
        out["windows"] = {k: [float(v[0]), float(v[1])] for k, v in windows.items()}
    if provenance is not None:
        out["fingerprint"] = provenance
    # Re-derived rather than trusted: a frozen spec written by an older version
    # carries the older claim, and that claim is what the reader checks against
    # the picture.
    out["projection"] = projection_note(columns)
    return out


def save_view_spec(spec: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2))
    return path


def load_view_spec(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# mask semantics: which axes were sampled, and how
# --------------------------------------------------------------------------

def mask_geometry(mask: np.ndarray) -> dict:
    """Describe a measured-support mask as a lattice, measured from the mask.

    Extended from the original helper: it now also says whether the *y* axis was
    subsampled at all, because the report has to distinguish "rows were kept and
    skipped" (y / yz protocols) from "every row of a measured plane was taken"
    (z protocol). Every downstream label is derived from these booleans instead
    of from the protocol name.
    """
    m = np.asarray(mask).astype(bool)
    z, h, w = m.shape
    rows = m.all(axis=2).any(axis=0)                 # kept end-to-end somewhere
    kept = np.flatnonzero(rows)
    step = int(np.diff(kept).min()) if kept.size > 1 else 1
    per_plane = m.reshape(z, -1).mean(axis=1)
    planes = m.any(axis=(1, 2))
    # A row counts as "dense" when it is fully measured in every plane that was
    # visited. Asking the whole volume instead would make a Z-only mask look as
    # if it subsampled rows, because the skipped planes measure nothing at all --
    # that is exactly the mislabel ("XZ given/gap" under z_k4) to avoid.
    visited = m[planes]
    dense_rows = (visited.all(axis=2).all(axis=0) if len(visited)
                  else np.zeros(h, dtype=bool))
    return {
        "measured_fraction": _round(m.mean(), 4),
        "kept_y_rows": int(rows.sum()),
        "y_period": step,
        "x_dense": bool(kept.size and m[:, kept, :].all()),
        "y_dense": bool(dense_rows.all()),
        "z_dense": bool(planes.all()),
        "z_planes_measured": int(planes.sum()),
        "z_planes_total": int(z),
        "row_pattern": "".join("1" if r else "0" for r in rows[:4 * step]),
        "per_plane_fraction": [round(float(p), 4) for p in per_plane],
    }


def _nearest(cands: np.ndarray, target: int) -> int:
    return int(cands[np.argmin(np.abs(cands - target))])


def sampling_semantics(mask: np.ndarray, sampling: dict) -> dict:
    """Does the mask on screen actually correspond to the protocol it is filed under?

    The report is only worth its JSON if this is checked rather than asserted:
    a figure labelled ``z_k4`` whose mask happens to subsample rows instead is
    exactly the failure §4.4.1 exists to prevent, and the label is what a human
    reads. Two independent facts are compared -- the measured fraction against
    the protocol's ideal, and the dense/subsampled axis flags against the
    protocol's own ``ky`` / ``kz``.
    """
    geom = mask_geometry(mask)
    frac = float(geom["measured_fraction"])
    ideal = sampling.get("ideal_fraction")
    ideal = float(ideal) if ideal is not None else None
    ky, kz = int(sampling.get("ky", 1) or 1), int(sampling.get("kz", 1) or 1)
    # A grid lattice is exact up to the boundary: the last row/plane of a period
    # may be kept or dropped, so the fraction can miss by at most one row (1/H)
    # and one plane (1/Z). Anything larger is a mask that is not this protocol.
    tol = 0.0
    if ideal is not None:
        _, hgt, _ = np.asarray(mask).shape
        tol = 2.0 / max(1, hgt) + 2.0 / max(1, int(geom["z_planes_total"]))
    return {
        "measured_fraction": frac,
        "ideal_fraction": ideal,
        "fraction_tolerance": round(tol, 4),
        "fraction_ok": (ideal is None
                        or abs(frac - ideal) <= tol),
        "z_dense": geom["z_dense"], "y_dense": geom["y_dense"],
        "expected_z_dense": kz == 1, "expected_y_dense": ky == 1,
        "axes_ok": (geom["z_dense"] == (kz == 1)
                    and geom["y_dense"] == (ky == 1)),
        "sampling": dict(sampling),
    }


def window_stability(spec: dict, ref: np.ndarray, mask: np.ndarray) -> dict:
    """Do the frozen display windows still equal the ones this reference yields?

    Deriving them twice is the cheap way to prove the frozen values are the
    reference's own: if a spec were frozen from a different volume, an older
    cache or a different normalisation, the two sets would differ.
    """
    fixed = [c for c in protocol_columns(spec, mask) if c["role"] == "fixed"]
    now = ref_windows(ref, fixed, spec)
    frozen = {k: [float(v[0]), float(v[1])]
              for k, v in (spec.get("windows") or {}).items()}
    diff = 0.0
    missing = []
    for key, (lo, hi) in now.items():
        if key not in frozen:
            missing.append(key)
            continue
        diff = max(diff, abs(float(lo) - frozen[key][0]),
                   abs(float(hi) - frozen[key][1]))
    for key in frozen:
        if key not in now:
            missing.append(key)
    return {"window_diff": float(diff), "windows_missing": missing,
            "windows_ok": bool(not missing and diff == 0.0),
            "windows": frozen}


def protocol_columns(spec: dict, mask: np.ndarray) -> list[dict]:
    """The panels worth showing for *this* protocol, with honest coordinates.

    Six fixed panels come first, identical for every protocol and every arm (that
    is what makes two figures comparable). Diagnostic panels follow, and they
    exist only for the axes this mask actually subsamples:

    * planes skipped -> an entirely measured plane and an entirely skipped one;
    * rows skipped   -> a row the input was given and a row it never saw.

    A Y-only mask gets the row diagnostics and no plane diagnostics (claiming a
    "skipped plane" there would be false -- every plane is measured), and a
    Z-only mask gets the opposite. YZ gets both.
    """
    z, h, w = (int(v) for v in spec["shape"])
    geom = mask_geometry(mask)
    m = np.asarray(mask).astype(bool)

    cols: list[dict] = [
        {"key": "XY primary", "axis": "xy", "coord": int(spec["z_view"]),
         "role": "fixed", "note": f"z={spec['z_view']}"},
        {"key": "XZ", "axis": "xz", "coord": int(spec["y_view"]),
         "role": "fixed", "note": f"y={spec['y_view']}"},
        {"key": "YZ", "axis": "yz", "coord": int(spec["x_view"]),
         "role": "fixed", "note": f"x={spec['x_view']}"},
        {"key": "Zoom", "axis": "zoom", "coord": int(spec["z_view"]),
         "role": "fixed", "note": f"z={spec['z_view']}"},
        {"key": "MIP-z", "axis": "mipz", "coord": 0,
         "role": "fixed", "note": f"z {spec['mip_z'][0]}-{spec['mip_z'][1]}"},
    ]

    if not geom["z_dense"]:
        per_plane = m.reshape(z, -1).mean(axis=1)
        meas = np.flatnonzero(per_plane >= per_plane.max() - 1e-9)
        gap = np.flatnonzero(per_plane <= per_plane.min() + 1e-9)
        zm = _nearest(meas, int(spec["z_view"]))
        zg = _nearest(gap, int(spec["z_view"]))
        # If even the best plane is partial (yz protocol) the label says so: it
        # is a plane that was *visited*, not a plane that was fully measured.
        frac = float(per_plane[zm])
        cols.append({"key": "XY measured plane", "axis": "xy", "coord": zm,
                     "role": "diagnostic",
                     "note": f"z={zm}, {100 * frac:.0f}% of voxels measured"})
        cols.append({"key": "XY skipped plane", "axis": "xy", "coord": zg,
                     "role": "diagnostic",
                     "note": f"z={zg}, {100 * float(per_plane[zg]):.0f}% measured"})

    if not geom["y_dense"]:
        # Row coverage measured *within the planes that were visited*: a row of a
        # skipped plane is unmeasured for a different reason (the whole plane is
        # gone), and mixing the two would make every row look half measured.
        visited = m.any(axis=2)
        cov = np.zeros(h)
        denom = max(int(visited.sum()), 1)
        for yi in range(h):
            cov[yi] = (m[:, yi, :] & visited).sum() / (denom * w)
        want = np.flatnonzero(cov >= cov.max() - 1e-9)
        lack = np.flatnonzero(cov <= cov.min() + 1e-9)
        ym = _nearest(want, int(spec["y_view"]))
        yg = _nearest(lack, int(spec["y_view"]))
        cols.append({"key": "XZ measured row", "axis": "xz", "coord": ym,
                     "role": "diagnostic",
                     "note": f"y={ym}, {100 * float(cov[ym]):.0f}% measured"})
        cols.append({"key": "XZ skipped row", "axis": "xz", "coord": yg,
                     "role": "diagnostic",
                     "note": f"y={yg}, {100 * float(cov[yg]):.0f}% measured"})
    return cols


def take(vol: np.ndarray, col: dict, spec: dict) -> np.ndarray:
    """Cut one 2-D panel out of a (Z, Y, X) volume."""
    z, y, x = col["coord"], col["coord"], col["coord"]
    axis = col["axis"]
    if axis == "xy":
        return vol[z]
    if axis == "xz":
        return vol[:, y, :]
    if axis == "yz":
        return vol[:, :, x]
    if axis == "zoom":
        y0, x0, n = (int(v) for v in spec["roi"])
        return vol[z, y0:y0 + n, x0:x0 + n]
    if axis == "mipz":
        z0, z1 = (int(v) for v in spec["mip_z"])
        return _nanmax(vol[z0:z1], axis=0)
    raise ValueError(f"unknown view axis {axis!r}")


def _nanmax(vol: np.ndarray, axis: int) -> np.ndarray:
    """Max that keeps "nothing here" as NaN instead of hiding it in a zero."""
    ok = np.isfinite(vol)
    out = np.where(ok, vol, -np.inf).max(axis=axis)
    return np.where(ok.any(axis=axis), out, np.nan)


def window(img: np.ndarray) -> tuple[float, float]:
    """Robust display window: 0.5th-99.5th percentile of the finite voxels."""
    v = np.asarray(img)[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, (0.5, 99.5))
    return float(lo), float(max(hi, lo + 1e-6))


def ref_windows(ref: np.ndarray, cols: list[dict], spec: dict) -> dict:
    """One display window per panel, taken from the dense reference.

    Computed once from the reference and reused by every protocol and every arm,
    so a dim reconstruction reads as dim instead of being re-normalised into a
    plausible picture. Different panels get different windows because depth
    genuinely changes brightness; the same panel never does.
    """
    return {c["key"]: window(take(ref, c, spec)) for c in cols}


# --------------------------------------------------------------------------
# consistency: the things a comparison figure must not silently break
# --------------------------------------------------------------------------

def check_group(vols: list[dict], spec: dict) -> dict:
    """Everything that must hold inside one figure (one protocol, several arms).

    Returns a dict of maxima; the caller decides whether a non-finite or
    non-zero entry is fatal. ``target_diff`` must be exactly 0 -- the dense
    reference is not allowed to depend on the sparse protocol or on the arm.
    """
    if not vols:
        raise ValueError("no volumes")
    ref = np.asarray(vols[0]["target"], dtype=np.float64)
    target_diff = 0.0
    for v in vols[1:]:
        t = np.asarray(v["target"], dtype=np.float64)
        if t.shape != ref.shape:
            return {"target_diff": float("inf"), "shape_mismatch": True}
        target_diff = max(target_diff, float(np.max(np.abs(t - ref))))
    view_diff = 0.0
    views0 = {c["key"]: take(ref, c, spec)
              for c in protocol_columns(spec, vols[0]["in_mask"])}
    for v in vols[1:]:
        cols = protocol_columns(spec, v["in_mask"])
        for c in cols:
            a = views0.get(c["key"])
            b = take(np.asarray(v["target"], dtype=np.float64), c, spec)
            if a is None or a.shape != b.shape:
                return {"target_diff": target_diff, "view_mismatch": True}
            view_diff = max(view_diff, float(np.nanmax(np.abs(a - b)))
                            if np.isfinite(b).any() else 0.0)
    # Same input for every arm: the arms may only differ by their weights.
    input_diff = 0.0
    for v in vols[1:]:
        a = np.asarray(vols[0]["y_in"], dtype=np.float64)
        b = np.asarray(v["y_in"], dtype=np.float64)
        input_diff = max(input_diff, float(np.max(np.abs(a - b))))
        if not np.array_equal(np.asarray(vols[0]["in_mask"]),
                              np.asarray(v["in_mask"])):
            input_diff = float("inf")
    return {"target_diff": target_diff, "view_diff": view_diff,
            "input_diff": input_diff, "shape_mismatch": False,
            "view_mismatch": False}


def check_protocols(groups: list[dict], spec: dict) -> dict:
    """Everything that must hold *across* protocols in a comparison figure.

    ``groups`` is a list of ``{"level": str, "vols": [...]}``. The reference of
    every protocol must be the same array, and the fixed panels taken from it
    must be identical -- if they are not, the dense cache, the normalisation or
    the sample id is inconsistent and no figure may be published.
    """
    if len(groups) < 2:
        raise ValueError("a cross-protocol comparison needs at least two groups")
    ref0 = np.asarray(groups[0]["vols"][0]["target"], dtype=np.float64)
    target_diff = 0.0
    view_diff = 0.0
    per_group = []
    for g in groups:
        rep = check_group(g["vols"], spec)
        per_group.append({"level": g["level"], **rep})
        t = np.asarray(g["vols"][0]["target"], dtype=np.float64)
        if t.shape != ref0.shape:
            target_diff = float("inf")
            continue
        target_diff = max(target_diff, float(np.max(np.abs(t - ref0))))
        cols = [c for c in protocol_columns(spec, g["vols"][0]["in_mask"])
                if c["role"] == "fixed"]
        for c in cols:
            a, b = take(ref0, c, spec), take(t, c, spec)
            view_diff = max(view_diff, float(np.nanmax(np.abs(a - b))))
    return {"target_diff": target_diff, "fixed_view_diff": view_diff,
            "groups": per_group}

"""Figure 5: where a reconstruction of microvasculature breaks, not just how often.

A volume-average clDice is the natural headline for sparse-to-dense vessel
reconstruction, and it is also the number that hides the failure that matters.
Two reconstructions can post the same clDice while one drops thin capillary
segments and the other invents a shortcut between two vessels. In the mean the
two mistakes cancel into one mediocre score; clinically they are opposites --
a lost branch is missing perfusion, an invented bridge is a false anastomosis --
and no single number tells them apart. They also live in different places: lost
structure is a property of the reference (skeleton voxels the prediction never
came near), invented structure is a property of the prediction (skeleton voxels
the reference never had).

So this script draws the difference instead of averaging it. Per test volume:

* reference mask MIP with its skeleton, and reconstruction mask MIP with its
  skeleton, over the z range that actually contains scored planes;
* a *break map* -- reference-skeleton voxels farther than one voxel from any
  predicted-skeleton voxel -- painted on the reference MIP;
* a *false-connection map* -- predicted-skeleton voxels farther than one voxel
  from any reference-skeleton voxel -- painted on the prediction MIP;
* a local crop around the densest cluster of reference bifurcations, reference
  and reconstruction side by side in the same window, so a branch point can be
  judged by eye rather than by a count;
* the per-volume numbers *and their coverage*: Dice, clDice, skeleton F1,
  components, junctions and length are read from ``structure_summary.json``,
  while the scored-plane share is printed beside them, because on this cohort
  28-66 % of the planes are gated out and a clDice without that share invites
  the reader to assume it describes the whole stack;
* clDice per depth slab, because the same average hides whether the failure is
  uniform or concentrated in the deepest planes.

The global numbers are never recomputed on a crop: dice / clDice / skelF1 and
the counts come from the summary file unchanged. The break and false-connection
maps are the only quantities computed here, they are computed on the whole
exported (post-gating) volume, and the panel titles state that extent. The crop
is chosen from the *reference* and frozen in a small JSON next to the figure, so
re-rendering cannot silently look at a different branch point -- the same
failure mode that ``scripts/viewspec.py`` exists to prevent for the sparse
protocol figures.

Skeletonisation is the only expensive step, so both skeletons are cached as
packed bits under ``<out>/cache`` and reused unless ``--rebuild-cache`` is
given. Everything else is a projection or a nearest-neighbour query on a few
ten thousand points.

    python -m spur_mpm.scripts.structure_figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")               # headless: every figure is written to disk

from spur_mpm.metrics.skeleton import skeleton

__all__ = [
    "TOLERANCE_VOX",
    "load_masks",
    "scored_z_range",
    "mip",
    "break_map",
    "false_map",
    "break_false_stats",
    "choose_crop",
    "load_or_make_crop",
    "prepare_panels",
    "stats_from_summary",
    "verify_skeleton_counts",
    "load_or_compute_skeletons",
    "render_volume",
    "render_cross_volume",
    "main",
]

# Same tolerance as ``structure_summary.json`` (skeleton_overlap default): one
# voxel, Euclidean, in *voxel* units. Matching the summary is the point -- a
# break map drawn at a stricter tolerance would look worse than the number it is
# supposed to explain.
TOLERANCE_VOX = 1.0

CACHE_VERSION = 1
DEFAULT_STRUCTURE_DIRS = (
    Path("runs/E021_ladder/E021_S1_y_k2/structure"),
    Path("runs/E021_ladder/E021_A1_y_k2/structure"),
)
DEFAULT_OUT = Path("runs/E021_ladder/figures/structure")

# Display palette. Colours are named once so the panels and the footer legend
# cannot drift apart.
SKELETON_COLOR = "#facc15"          # yellow
BREAK_COLOR = "#ef4444"             # red: reference structure that was lost
FALSE_COLOR = "#22d3ee"             # cyan: structure the prediction invented


# ---------------------------------------------------------------------------
# helpers: pure numpy, testable on tiny synthetic volumes
# ---------------------------------------------------------------------------

def load_masks(structure_dir, arm: str, sample: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the exported reference / prediction masks as booleans.

    Both masks were written by ``structure_report.py`` from the same
    binarisation rule, which is the only reason a voxelwise comparison is
    meaningful. Nothing here re-thresholds them.
    """
    import tifffile

    d = Path(structure_dir)
    ref = tifffile.imread(str(d / f"{arm}_{sample}_ref_mask.tif"))
    pred = tifffile.imread(str(d / f"{arm}_{sample}_pred_mask.tif"))
    ref, pred = np.asarray(ref, bool), np.asarray(pred, bool)
    if ref.shape != pred.shape:
        raise ValueError(f"{sample}: ref {ref.shape} and pred {pred.shape} differ")
    if ref.ndim != 3:
        raise ValueError(f"{sample}: expected a (Z, Y, X) mask, got {ref.shape}")
    return ref, pred


def scored_z_range(ref: np.ndarray, pred: np.ndarray) -> tuple[int, int]:
    """Half-open index range spanning every plane that has any foreground.

    Gated-out planes (reference fills > 15 % of the plane) have both masks
    emptied, so they contribute nothing to a MIP; trimming the empty planes at
    the top and bottom keeps the panel label honest without pretending to know
    *why* an interior plane is empty. A pair with nothing in it returns (0, 0).
    """
    r = np.asarray(ref, bool)
    p = np.asarray(pred, bool)
    if r.shape != p.shape:
        raise ValueError(f"ref {r.shape} and pred {p.shape} differ")
    occupied = np.flatnonzero(r.reshape(r.shape[0], -1).any(axis=1)
                              | p.reshape(p.shape[0], -1).any(axis=1))
    if occupied.size == 0:
        return (0, 0)
    return (int(occupied[0]), int(occupied[-1]) + 1)


def mip(mask: np.ndarray, z_range: tuple[int, int] | None = None) -> np.ndarray:
    """Maximum projection along z, optionally restricted to ``z_range``."""
    m = np.asarray(mask, bool)
    if m.ndim != 3:
        raise ValueError(f"expected (Z, Y, X), got {m.shape}")
    lo, hi = (0, m.shape[0]) if z_range is None else (int(z_range[0]), int(z_range[1]))
    lo, hi = max(lo, 0), min(hi, m.shape[0])
    if hi <= lo:
        return np.zeros(m.shape[1:], bool)
    return m[lo:hi].any(axis=0)


def _nearest_distance(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Euclidean distance from every ``src`` voxel to the nearest ``dst`` voxel.

    This is the per-voxel form of ``_matched_fraction`` in
    ``metrics/skeleton.py``: same metric (Euclidean voxel units, no sampling),
    but returning the map instead of its mean. A KD-tree is used because the
    skeletons hold ~10^5 points, where the exact query is milliseconds and a
    full-volume distance transform would allocate hundreds of megabytes for a
    number the query already gives exactly.
    """
    if len(dst) == 0:
        return np.full(len(src), np.inf)
    return cKDTree(dst).query(src, k=1, workers=-1)[0]


def break_map(ref_skel: np.ndarray, pred_skel: np.ndarray,
              tolerance_vox: float = TOLERANCE_VOX) -> np.ndarray:
    """Reference-skeleton voxels with no predicted skeleton within tolerance.

    These are the places where the reconstruction lost structure. An empty
    predicted skeleton makes every reference voxel a break (the same convention
    ``skeleton_overlap`` uses: one empty side scores zero recall); an empty
    reference skeleton has nothing to lose and returns an empty map. The empty
    case is why :func:`break_false_stats` reports NaN fractions rather than 0:
    "nothing to match" must never read as "nothing lost".
    """
    ref = np.asarray(ref_skel, bool)
    pred = np.asarray(pred_skel, bool)
    if ref.shape != pred.shape:
        raise ValueError(f"ref skeleton {ref.shape} and pred {pred.shape} differ")
    out = np.zeros(ref.shape, bool)
    if not ref.any():
        return out
    pts = np.argwhere(ref)
    dist = _nearest_distance(pts, np.argwhere(pred))
    lost = dist > float(tolerance_vox) + 1e-9
    out[tuple(pts[lost].T)] = True
    return out


def false_map(pred_skel: np.ndarray, ref_skel: np.ndarray,
              tolerance_vox: float = TOLERANCE_VOX) -> np.ndarray:
    """Predicted-skeleton voxels with no reference skeleton within tolerance.

    The mirror image of :func:`break_map`: structure the reconstruction invented
    (a bridged gap, a spurious side branch), judged against the reference. An
    empty reference skeleton makes every predicted voxel an invention.
    """
    pred = np.asarray(pred_skel, bool)
    ref = np.asarray(ref_skel, bool)
    if pred.shape != ref.shape:
        raise ValueError(f"pred skeleton {pred.shape} and ref {ref.shape} differ")
    out = np.zeros(pred.shape, bool)
    if not pred.any():
        return out
    pts = np.argwhere(pred)
    dist = _nearest_distance(pts, np.argwhere(ref))
    invented = dist > float(tolerance_vox) + 1e-9
    out[tuple(pts[invented].T)] = True
    return out


def _fraction(num: int, den: int) -> float:
    """num / den, or NaN when the denominator is zero (no voxels to match)."""
    return float(num) / float(den) if den else float("nan")


def break_false_stats(ref_skel: np.ndarray, pred_skel: np.ndarray,
                      tolerance_vox: float = TOLERANCE_VOX) -> dict:
    """Counts and fractions of lost / invented skeleton voxels.

    The fractions are NaN when their denominator is empty, so an all-empty mask
    pair cannot silently score 0 % lost *and* 0 % invented, which would read as
    a perfect reconstruction of nothing.
    """
    ref = np.asarray(ref_skel, bool)
    pred = np.asarray(pred_skel, bool)
    brk = break_map(ref, pred, tolerance_vox)
    fal = false_map(pred, ref, tolerance_vox)
    n_ref, n_pred = int(ref.sum()), int(pred.sum())
    return {
        "tolerance_vox": float(tolerance_vox),
        "ref_skel_voxels": n_ref,
        "pred_skel_voxels": n_pred,
        "break_voxels": int(brk.sum()),
        "false_voxels": int(fal.sum()),
        "break_fraction": _fraction(int(brk.sum()), n_ref),
        "false_fraction": _fraction(int(fal.sum()), n_pred),
    }


def _junction_voxels(thin: np.ndarray) -> np.ndarray:
    """Skeleton voxels with >= 3 skeleton neighbours.

    Deliberately the *voxel* rule from ``skeleton_stats`` (which then clusters
    them before reporting a junction count). The crop only needs a density, so
    the un-clustered map is the right input, and using the same neighbour
    definition keeps it consistent with the count in the stats panel.
    """
    t = np.asarray(thin, bool)
    kernel = np.ones((3,) * t.ndim, np.uint8)
    counts = ndi.convolve(t.astype(np.uint8), kernel, mode="constant", cval=0)
    return t & (counts.astype(np.int16) - 1 >= 3)


def _box_sums_2d(a: np.ndarray, n: int) -> np.ndarray:
    """Sum of every n x n window, shape (h - n + 1, w - n + 1)."""
    ii = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
    return ii[n:, n:] - ii[:-n, n:] - ii[n:, :-n] + ii[:-n, :-n]


def choose_crop(ref_mask: np.ndarray, ref_skel: np.ndarray, *,
                size: int = 96, slab: int = 12) -> dict:
    """Pick a 3-D box around the reference's densest bifurcation cluster.

    The choice is made from the **reference only**, so the reconstruction cannot
    move the window to a region it happens to get right. In order:

    1. project the reference-skeleton junction voxels onto XY and take the
       ``size`` x ``size`` window with the most of them (top-left positions only,
       first maximum in row-major order -- ties resolve deterministically);
    2. inside that window, take the ``slab``-plane z run with the most junction
       voxels. Without junctions, centre the slab on the reference foreground.

    Every index is clipped to the volume, so the returned box always fits. The
    result is a plain JSON-serialisable dict; :func:`load_or_make_crop` freezes
    it so a re-render cannot choose a different branch point.
    """
    ref = np.asarray(ref_mask, bool)
    skel = np.asarray(ref_skel, bool)
    if ref.shape != skel.shape:
        raise ValueError(f"mask {ref.shape} and skeleton {skel.shape} differ")
    z, h, w = ref.shape
    n = int(min(size, h, w))
    if n <= 0:
        raise ValueError(f"crop size {size} does not fit a {(h, w)} plane")

    junctions = _junction_voxels(skel)
    # Densest XY window of junctions, falling back to the mask itself when the
    # reference has no bifurcation at all (a straight capillary, or empty).
    source = junctions.any(axis=0) if junctions.any() else ref.any(axis=0)
    scores = _box_sums_2d(source.astype(np.int64), n)
    y0, x0 = (int(v) for v in np.unravel_index(int(np.argmax(scores)), scores.shape))

    nz = int(min(slab, z))
    sub = junctions[:, y0:y0 + n, x0:x0 + n]
    if sub.any() and z > nz:
        per_z = sub.reshape(z, -1).sum(axis=1)
        z0 = int(np.argmax(np.convolve(per_z, np.ones(nz, np.int64), mode="valid")))
    elif sub.any():
        z0 = 0
    else:
        per_z = ref[:, y0:y0 + n, x0:x0 + n].reshape(z, -1).sum(axis=1)
        occupied = np.flatnonzero(per_z)
        middle = int(occupied[len(occupied) // 2]) if occupied.size else 0
        z0 = int(np.clip(middle - nz // 2, 0, max(z - nz, 0)))
    z0 = int(np.clip(z0, 0, max(z - nz, 0)))

    return {
        "sample_id": None,          # filled by the caller; shape is the check
        "shape": [int(z), int(h), int(w)],
        "z0": z0, "y0": y0, "x0": x0,
        "nz": nz, "ny": n, "nx": n,
        "crop_size": n,
        "slab": nz,
        "junction_voxels_in_crop": int(sub.sum()),
        "chosen_from": "reference skeleton junction density",
        "note": ("frozen crop: re-rendering reuses these indices even if the "
                 "mask changes; delete the JSON to re-choose"),
    }


def _validate_crop(crop: dict, shape: tuple[int, int, int]) -> dict:
    """Check a frozen crop against the volume it will be applied to."""
    if [int(v) for v in crop.get("shape", [])] != [int(s) for s in shape]:
        raise ValueError(f"crop shape {crop.get('shape')} does not match the "
                         f"volume {shape}: the frozen crop belongs to another "
                         f"volume, delete it to re-choose")
    z, h, w = (int(s) for s in shape)
    for start, span, extent in (("z0", "nz", z), ("y0", "ny", h),
                                ("x0", "nx", w)):
        a, n = int(crop[start]), int(crop[span])
        if n <= 0 or a < 0 or a + n > extent:
            raise ValueError(f"frozen crop {crop} does not fit {(z, h, w)}")
    return crop


def load_or_make_crop(path, ref_mask: np.ndarray, ref_skel: np.ndarray, *,
                      size: int = 96, slab: int = 12,
                      sample_id: str | None = None) -> dict:
    """Return the frozen crop, creating and writing it on the first render.

    Once written, the crop is returned untouched -- changing the mask does not
    move it. That is what makes two renders of the same volume comparable, and
    it is why the file sits next to the figure instead of in a temp directory.
    """
    p = Path(path)
    if p.exists():
        crop = json.loads(p.read_text())
        return _validate_crop(crop, tuple(np.asarray(ref_mask).shape))
    crop = choose_crop(ref_mask, ref_skel, size=size, slab=slab)
    crop["sample_id"] = sample_id or p.stem
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(crop, indent=2) + "\n")
    return crop


def prepare_panels(ref: np.ndarray, pred: np.ndarray, ref_skel: np.ndarray,
                   pred_skel: np.ndarray, crop: dict) -> dict:
    """Everything the renderer draws, as 2-D arrays plus one stats dict.

    Split from plotting so a synthetic tube can exercise the break / false maps
    and the frozen crop without a summary file, a TIFF, or matplotlib.
    """
    r = np.asarray(ref, bool)
    p = np.asarray(pred, bool)
    rs = np.asarray(ref_skel, bool)
    ps = np.asarray(pred_skel, bool)
    for name, arr in (("ref", r), ("pred", p), ("ref_skel", rs),
                      ("pred_skel", ps)):
        if arr.shape != r.shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {r.shape}")
    _validate_crop(crop, r.shape)

    zr = scored_z_range(r, p)
    brk = break_map(rs, ps)
    fal = false_map(ps, rs)

    z0, y0, x0 = int(crop["z0"]), int(crop["y0"]), int(crop["x0"])
    nz, ny, nx = int(crop["nz"]), int(crop["ny"]), int(crop["nx"])
    box = (slice(z0, z0 + nz), slice(y0, y0 + ny), slice(x0, x0 + nx))

    return {
        "z_range": zr,
        "ref_mip": mip(r, zr),
        "pred_mip": mip(p, zr),
        "ref_skel_mip": mip(rs, zr),
        "pred_skel_mip": mip(ps, zr),
        "break_mip": mip(brk, zr),
        "false_mip": mip(fal, zr),
        "ref_crop_mip": mip(r[box]),
        "pred_crop_mip": mip(p[box]),
        "ref_crop_skel_mip": mip(rs[box]),
        "pred_crop_skel_mip": mip(ps[box]),
        "stats": break_false_stats(rs, ps),
    }


def stats_from_summary(summary: dict, sample: str, n_planes: int) -> dict:
    """The per-volume display numbers, read from ``structure_summary.json``.

    Only formatting happens here. Length is converted from micrometres to
    millimetres for the panel, and the scored-plane share is derived from the
    volume's own plane count and the recorded ``n_planes_gated`` -- a count of
    gated planes alone invites the reader to mistake it for an index, and a
    clDice without its coverage invites them to mistake it for the whole stack.
    """
    try:
        rec = next(v for v in summary["volumes"] if v["sample_id"] == sample)
    except (KeyError, StopIteration) as exc:
        raise KeyError(f"{sample} is not in the structure summary") from exc
    n_planes = int(n_planes)
    gated = int(rec.get("n_planes_gated") or 0)
    scored = n_planes - gated
    if scored < 0:
        raise ValueError(f"{sample}: {gated} gated planes of {n_planes}")
    return {
        "sample_id": sample,
        "arm": summary.get("arm"),
        "levels": summary.get("levels"),
        "ky": summary.get("ky"),
        "kz": summary.get("kz"),
        "dice": float(rec["dice"]),
        "cl_dice": float(rec["cl_dice"]),
        "f1": float(rec["f1"]),
        "tolerance_vox": float(rec.get("tolerance_vox", TOLERANCE_VOX)),
        "components_ref": int(rec["ref_n_components"]),
        "components_pred": int(rec["pred_n_components"]),
        "junctions_ref": int(rec["ref_n_junctions"]),
        "junctions_pred": int(rec["pred_n_junctions"]),
        "length_um_ref": float(rec["ref_total_length_um"]),
        "length_um_pred": float(rec["pred_total_length_um"]),
        "n_planes": n_planes,
        "n_planes_gated": gated,
        "planes_scored": scored,
        "planes_share": scored / n_planes if n_planes else float("nan"),
        "threshold": rec.get("threshold"),
        "threshold_mode": rec.get("threshold_mode"),
        "segment_method": rec.get("segment_method"),
        "depth_profile": list(rec.get("depth_profile") or []),
        # Kept only so the renderer can prove the masks on screen are the ones
        # the summary was computed from.
        "summary_ref_skel_voxels": int(rec["ref_n_skeleton_voxels"]),
        "summary_pred_skel_voxels": int(rec["pred_n_skeleton_voxels"]),
    }


def verify_skeleton_counts(stats: dict, ref_skel: np.ndarray,
                           pred_skel: np.ndarray) -> None:
    """Fail loudly if the skeletons differ from the ones the summary scored.

    A figure quoting summary numbers next to masks that have since been rebuilt
    would be a lie that no reader can detect, so the mismatch is an error rather
    than a warning.
    """
    got = (int(np.asarray(ref_skel).sum()), int(np.asarray(pred_skel).sum()))
    want = (stats["summary_ref_skel_voxels"], stats["summary_pred_skel_voxels"])
    if got != want:
        raise RuntimeError(
            f"{stats['sample_id']}: skeleton voxel counts {got} do not match "
            f"structure_summary.json {want}; the masks on disk are not the ones "
            f"the quoted numbers came from -- re-run structure_report.py")


def _packed(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, bool).ravel())


def _unpacked(packed: np.ndarray, shape) -> np.ndarray:
    n = int(np.prod(shape))
    return np.unpackbits(np.asarray(packed))[:n].reshape(tuple(shape)).astype(bool)


def load_or_compute_skeletons(ref: np.ndarray, pred: np.ndarray, path,
                              rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return both skeletons, cached as packed bits so a re-render is cheap.

    The cache is keyed by file name (arm + sample) in the caller, validated
    against the mask shape here, and carries a version so an old cache from a
    different skeleton convention is ignored instead of silently reused.
    """
    p = Path(path)
    shape = tuple(np.asarray(ref).shape)
    if p.exists() and not rebuild:
        with np.load(p) as z:
            if int(z["version"]) == CACHE_VERSION and \
                    tuple(int(v) for v in z["shape"]) == shape:
                return _unpacked(z["ref"], shape), _unpacked(z["pred"], shape)
    rs = skeleton(np.asarray(ref, bool))
    ps = skeleton(np.asarray(pred, bool))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, version=np.array(CACHE_VERSION),
                        shape=np.asarray(shape, np.int64),
                        ref=_packed(rs), pred=_packed(ps))
    return np.asarray(rs, bool), np.asarray(ps, bool)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _compose(base: np.ndarray, layers, gain: float = 0.78,
             floor: float = 0.06) -> np.ndarray:
    """Gray base image with coloured binary layers painted on top."""
    from matplotlib.colors import to_rgb

    h, w = np.asarray(base).shape
    img = np.empty((h, w, 3), np.float32)
    img[:] = (floor + gain * np.asarray(base, np.float32))[..., None]
    for mask, color in layers:
        img[np.asarray(mask, bool)] = np.asarray(to_rgb(color), np.float32)
    return img


def _draw(ax, img: np.ndarray, title: str = "", ticks: bool = False) -> None:
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=8.2, pad=3.0)
    if ticks:
        ax.tick_params(labelsize=5.5, length=2, pad=1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _fmt_mm(um: float) -> str:
    return f"{um / 1000.0:,.1f}"


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100.0 * x:.1f} %"


def _depth_panel(ax, profile: list, cl_dice: float) -> None:
    """clDice per z slab; grey bands are slabs with no scored plane."""
    if not profile:
        ax.text(0.5, 0.5, "no depth profile in summary", ha="center",
                va="center", fontsize=7, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    for slab in profile:
        lo, hi = float(slab["z_lo_um"]), float(slab["z_hi_um"])
        value = float(slab.get("cl_dice", float("nan")))
        if np.isfinite(value):
            ax.barh(0.5 * (lo + hi), value, height=hi - lo, color="#2563eb")
        else:
            # Shade the band rather than drawing a bar: a full-length grey bar
            # would read as clDice = 1.0, the opposite of "nothing was scored".
            ax.axhspan(lo, hi, color="#e5e7eb", zorder=0, lw=0.0)
    ax.axvline(cl_dice, color="#111111", lw=0.9, ls="--")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("clDice on the slab", fontsize=6.5)
    ax.set_ylabel("depth from first plane (um)", fontsize=6.5)
    ax.tick_params(labelsize=5.8, length=2, pad=1)
    ax.grid(alpha=0.25, axis="x")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("clDice by z slab (dashed: volume clDice; grey: no scored "
                 "plane)", fontsize=7.6, pad=3.0)


def render_volume(panels: dict, stats: dict, crop: dict, out_path, *,
                  sample: str, arm: str, voxel_um=(5.0, 0.98, 0.98),
                  dpi: int = 180) -> Path:
    """One volume: masks, skeletons, break / false maps, crop, stats, depth."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    zr = panels["z_range"]
    z0, y0, x0 = int(crop["z0"]), int(crop["y0"]), int(crop["x0"])
    nz, ny, nx = int(crop["nz"]), int(crop["ny"]), int(crop["nx"])
    sc = panels["stats"]
    n_planes = stats["n_planes"]
    scored = stats["planes_scored"]
    scored_txt = (f"{scored}/{n_planes} ({100.0 * stats['planes_share']:.0f} %)")

    fig = plt.figure(figsize=(10.6, 9.6), dpi=dpi)
    gs = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.0, 1.35),
                          height_ratios=(1.0, 1.0, 1.0),
                          left=0.035, right=0.975, bottom=0.065, top=0.90,
                          wspace=0.10, hspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_p = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    ax_l = fig.add_subplot(gs[2, 2])

    zoom = (f"MIP z {zr[0]}-{zr[1] - 1}" if zr[1] > zr[0] else "empty volume")

    # (a) reference mask + its skeleton, (b) prediction + its skeleton.
    _draw(ax_a, _compose(panels["ref_mip"], [(panels["ref_skel_mip"],
                                             SKELETON_COLOR)]),
          f"reference mask + skeleton ({zoom})")
    _draw(ax_b, _compose(panels["pred_mip"], [(panels["pred_skel_mip"],
                                              SKELETON_COLOR)]),
          f"reconstruction mask + skeleton ({zoom})")

    # (c) lost reference structure, (d) invented predicted structure. The mask
    # underneath is dimmed so the coloured voxels, not the vessel tree, set the
    # visual weight of the panel.
    _draw(ax_c, _compose(panels["ref_mip"], [(panels["break_mip"],
                                             BREAK_COLOR)], gain=0.42),
          f"break map: lost reference skeleton\n"
          f"{sc['break_voxels']} vox = {_fmt_pct(sc['break_fraction'])} of ref "
          f"skeleton")
    _draw(ax_d, _compose(panels["pred_mip"], [(panels["false_mip"],
                                              FALSE_COLOR)], gain=0.42),
          f"false connections: invented predicted skeleton\n"
          f"{sc['false_voxels']} vox = {_fmt_pct(sc['false_fraction'])} of pred "
          f"skeleton")

    # (e)/(f) the frozen bifurcation crop, reference and prediction.
    _draw(ax_e, _compose(panels["ref_crop_mip"],
                         [(panels["ref_crop_skel_mip"], SKELETON_COLOR)]),
          f"reference crop + skeleton\nz {z0}-{z0 + nz - 1}, "
          f"y {y0}-{y0 + ny - 1}, x {x0}-{x0 + nx - 1}", ticks=True)
    _draw(ax_f, _compose(panels["pred_crop_mip"],
                         [(panels["pred_crop_skel_mip"], SKELETON_COLOR)]),
          "reconstruction, same crop and window", ticks=True)
    ax_f.set_xlabel("x (voxels)", fontsize=6.5)
    ax_e.set_xlabel("x (voxels)", fontsize=6.5)

    # (g) locator: where the crop sits in the reference MIP (origin upper left,
    # exactly as imshow draws it, so no axis flip can misplace the box).
    _draw(ax_l, _compose(panels["ref_mip"], [], gain=0.55),
          "crop location on the reference MIP")
    ax_l.add_patch(Rectangle((x0, y0), nx, ny, fill=False,
                             edgecolor=SKELETON_COLOR, lw=1.1))
    ax_l.text(0.02, 0.02,
              f"crop chosen from the reference:\ndenest junction window "
              f"({crop['junction_voxels_in_crop']} junction voxels)\n"
              f"frozen in crop_{arm}_{sample}.json",
              transform=ax_l.transAxes, fontsize=6.0, va="bottom", ha="left",
              color="white",
              bbox=dict(facecolor="black", alpha=0.55, lw=0.0, pad=2.0))

    # (h) numbers. The summary block and the computed block are separated on
    # purpose: a reader must be able to tell which numbers are global and which
    # were measured for this figure, and on what extent.
    lines = [
        f"arm {arm}   sparse sampling: {stats['levels']}, "
        f"ky={stats['ky']}, kz={stats['kz']}",
        f"volume {sample}",
        "",
        "from structure_summary.json",
        f"  Dice        {stats['dice']:.3f}",
        f"  clDice      {stats['cl_dice']:.3f}",
        f"  skelF1      {stats['f1']:.3f}",
        f"  components  {stats['components_ref']:>7,} -> "
        f"{stats['components_pred']:>7,}",
        f"  junctions   {stats['junctions_ref']:>7,} -> "
        f"{stats['junctions_pred']:>7,}",
        f"  length mm   {_fmt_mm(stats['length_um_ref']):>7} -> "
        f"{_fmt_mm(stats['length_um_pred']):>7}",
        f"  planes scored   {scored_txt}",
        f"  ({stats['n_planes_gated']} planes gated: ref fills > 15 %)",
        "",
        "computed for this figure",
        f"  lost ref skeleton   {sc['break_voxels']:,} vox "
        f"({_fmt_pct(sc['break_fraction'])})",
        f"  invented pred skel  {sc['false_voxels']:,} vox "
        f"({_fmt_pct(sc['false_fraction'])})",
        f"  tolerance {sc['tolerance_vox']:.0f} voxel, all {n_planes} planes",
        f"  of the exported masks ({scored} non-empty)",
    ]
    ax_s.text(0.0, 1.0, "\n".join(lines), transform=ax_s.transAxes,
              ha="left", va="top", fontsize=7.4, family="monospace")
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    for sp in ax_s.spines.values():
        sp.set_visible(False)

    # (i) depth profile: an average is only honest next to its stratification.
    _depth_panel(ax_p, stats["depth_profile"], stats["cl_dice"])

    z_um, xy_um = float(voxel_um[0]), float(voxel_um[1])
    fig.suptitle(f"Fig. 5  {arm} / {sample}: where the reconstruction breaks "
                 f"(clDice {stats['cl_dice']:.3f}, skelF1 {stats['f1']:.3f}, "
                 f"planes scored {scored_txt})", fontsize=11.0, y=0.975)
    fig.text(0.5, 0.012,
             "gray = mask MIP | yellow = skeleton MIP | red = reference "
             "skeleton with no predicted skeleton within 1 voxel | cyan = "
             "predicted skeleton with no reference skeleton within 1 voxel | "
             f"voxels {xy_um:.2f} x {xy_um:.2f} x {z_um:.1f} um | "
             "global numbers are the summary's, break/false are computed here "
             "on the whole exported volume",
             ha="center", va="bottom", fontsize=6.1, color="0.30")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


def render_cross_volume(rows: list[dict], out_path, *, arm: str,
                        dpi: int = 140) -> Path:
    """The same four panels for every volume, plus the metrics as bars.

    The point of the summary is comparison: the worst volume has to look worse
    than its neighbours in the images *and* in the bars, not only in a table
    cell. Rows come pre-projected from :func:`prepare_panels`, so this function
    never holds more than one volume's 3-D arrays at a time.
    """
    import matplotlib.pyplot as plt

    n = len(rows)
    if n == 0:
        raise ValueError("no volumes to summarise")
    fig = plt.figure(figsize=(13.4, 2.35 * n + 3.0), dpi=dpi)
    gs = fig.add_gridspec(n + 1, 4, height_ratios=[1.0] * n + [2.1],
                          left=0.035, right=0.975, bottom=0.055, top=0.92,
                          wspace=0.10, hspace=0.30)
    for i, row in enumerate(rows):
        panels, stats = row["panels"], row["stats"]
        zr = panels["z_range"]
        zoom = f"z {zr[0]}-{zr[1] - 1}" if zr[1] > zr[0] else "empty"
        axes = [fig.add_subplot(gs[i, c]) for c in range(4)]
        _draw(axes[0], _compose(panels["ref_mip"],
                                [(panels["ref_skel_mip"], SKELETON_COLOR)]),
              f"{row['sample']}  reference ({zoom})" if i == 0 else
              f"{row['sample']}  reference")
        _draw(axes[1], _compose(panels["pred_mip"],
                                [(panels["pred_skel_mip"], SKELETON_COLOR)]),
              "reconstruction" if i == 0 else "")
        _draw(axes[2], _compose(panels["ref_mip"],
                                [(panels["break_mip"], BREAK_COLOR)],
                                gain=0.42),
              "break map (lost)" if i == 0 else "")
        _draw(axes[3], _compose(panels["pred_mip"],
                                [(panels["false_mip"], FALSE_COLOR)],
                                gain=0.42),
              "false connections (invented)" if i == 0 else "")
        sc = panels["stats"]
        axes[3].text(
            1.035, 0.5,
            f"clDice {stats['cl_dice']:.3f}\nskelF1 {stats['f1']:.3f}\n"
            f"lost {_fmt_pct(sc['break_fraction'])}\n"
            f"invented {_fmt_pct(sc['false_fraction'])}\n"
            f"planes {stats['planes_scored']}/{stats['n_planes']} "
            f"({100.0 * stats['planes_share']:.0f} %)",
            transform=axes[3].transAxes, fontsize=6.5, va="center", ha="left")

    ax_bar = fig.add_subplot(gs[n, :])
    width = 0.15
    metrics = [("Dice", "dice"), ("clDice", "cl_dice"), ("skelF1", "f1"),
               ("lost", None), ("invented", None)]
    colors = ("#94a3b8", "#2563eb", "#f59e0b", BREAK_COLOR, FALSE_COLOR)
    x = np.arange(n)
    for k, ((label, key), color) in enumerate(zip(metrics, colors)):
        vals = [(row["stats"][key] if key else
                 row["panels"]["stats"]["break_fraction" if label == "lost"
                                        else "false_fraction"])
                for row in rows]
        vals = [0.0 if not np.isfinite(v) else float(v) for v in vals]
        ax_bar.bar(x + (k - 2) * width, vals, width, label=label, color=color)
    for i, row in enumerate(rows):
        ax_bar.text(i, 1.035,
                    f"{row['stats']['planes_scored']}/{row['stats']['n_planes']}"
                    f" planes scored", ha="center", fontsize=6.3, color="0.30")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([row["sample"] for row in rows], fontsize=7.0)
    ax_bar.set_ylim(0.0, 1.12)
    ax_bar.set_ylabel("value (fraction for loss / invention)", fontsize=7.0)
    ax_bar.tick_params(labelsize=6.3)
    ax_bar.grid(alpha=0.22, axis="y")
    for sp in ax_bar.spines.values():
        sp.set_visible(False)
    ax_bar.legend(fontsize=6.5, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.suptitle(f"Fig. 5 summary  {arm}: network-level failure against the "
                 f"reference on every test volume (bars: summary metrics and "
                 f"maps computed here)", fontsize=10.8, y=0.965)
    fig.text(0.5, 0.006,
             "the same windows as the per-volume figures, one row per volume | "
             "lost / invented are fractions of the reference / predicted "
             "skeleton, computed on the whole exported (post-gating) volume | "
             "the plane share printed above each bar group is the coverage the "
             "summary's clDice was measured on",
             ha="center", va="bottom", fontsize=6.0, color="0.30")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure-dir", type=Path, action="append", default=None,
                    help="directory holding <arm>_<sample>_{ref,pred}_mask.tif, "
                         "the recon JSONs and structure_summary.json; repeatable "
                         "(default: the S1 and A1 y_k2 directories)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--samples", nargs="+", default=None,
                    help="default: every volume in structure_summary.json")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="default: <out>/cache")
    ap.add_argument("--crop-size", type=int, default=96)
    ap.add_argument("--slab", type=int, default=12,
                    help="planes in the bifurcation crop's z slab")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="recompute the cached skeletons")
    ap.add_argument("--no-summary-figure", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out / "cache"
    dirs = args.structure_dir or list(DEFAULT_STRUCTURE_DIRS)

    written: list[Path] = []
    for structure_dir in dirs:
        structure_dir = Path(structure_dir)
        summary_path = structure_dir / "structure_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"no structure_summary.json in {structure_dir}")
        summary = _load_json(summary_path)
        arm = str(summary["arm"])
        samples = args.samples or [v["sample_id"] for v in summary["volumes"]]
        rows = []
        for sample in samples:
            ref, pred = load_masks(structure_dir, arm, sample)
            meta = _load_json(structure_dir / f"{arm}_{sample}_recon.json")
            if list(meta.get("shape", [])) != list(ref.shape):
                raise RuntimeError(
                    f"{sample}: recon shape {meta.get('shape')} does not match "
                    f"the exported masks {list(ref.shape)}")
            ref_skel, pred_skel = load_or_compute_skeletons(
                ref, pred, cache_dir / f"{arm}_{sample}_skeleton.npz",
                rebuild=args.rebuild_cache)
            stats = stats_from_summary(summary, sample, ref.shape[0])
            verify_skeleton_counts(stats, ref_skel, pred_skel)
            crop = load_or_make_crop(
                out / f"crop_{arm}_{sample}.json", ref, ref_skel,
                size=args.crop_size, slab=args.slab, sample_id=sample)
            panels = prepare_panels(ref, pred, ref_skel, pred_skel, crop)
            geom = meta.get("geometry", {})
            voxel_um = (float(geom.get("z_step_um", 5.0)),
                        float(geom.get("voxel_xy_um", 0.98)),
                        float(geom.get("voxel_xy_um", 0.98)))
            png = render_volume(panels, stats, crop,
                                out / f"structure_{arm}_{sample}.png",
                                sample=sample, arm=arm, voxel_um=voxel_um,
                                dpi=args.dpi)
            written.append(png)
            print(f"{arm} {sample}: clDice {stats['cl_dice']:.3f} "
                  f"skelF1 {stats['f1']:.3f} "
                  f"lost {_fmt_pct(panels['stats']['break_fraction'])} "
                  f"invented {_fmt_pct(panels['stats']['false_fraction'])} "
                  f"-> {png}")
            rows.append({"sample": sample, "panels": panels, "stats": stats})
        if not args.no_summary_figure:
            png = render_cross_volume(
                rows, out / f"structure_cross_volume_{arm}.png", arm=arm,
                dpi=max(110, int(args.dpi * 0.8)))
            written.append(png)
    for p in written:
        print("wrote", p)
    print("ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

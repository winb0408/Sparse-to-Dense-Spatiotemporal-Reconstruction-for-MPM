"""Vessel-structure metrics: segmentation overlap and skeleton topology.

PSNR alone cannot tell a continuous capillary from a plausible-looking smear,
so every structural result also reports geometry (plan §9.1).

The reference segmentation is produced by a **fixed, training-free** rule
(hysteresis threshold on the reference image). It is therefore independent of
any prediction, which avoids the circularity of scoring a model with a
segmenter that was trained on that model's outputs -- but it is still a
threshold, not a hand-curated anatomical annotation. Report it as
"threshold-defined vessel reference", never as ground-truth anatomy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["otsu_threshold", "vessel_mask", "vesselness", "segment_pair",
           "dice", "cl_dice", "vessel_metrics", "mask_vessel_metrics"]

try:  # scikit-image is a declared dependency, but keep import errors legible
    from skimage.filters import threshold_otsu, sato
    from skimage.morphology import skeletonize
    from skimage.measure import label
    _SKIMAGE = True
except Exception:                                   # pragma: no cover
    _SKIMAGE = False

DEFAULT_SIGMAS = (1.0, 2.0)


def otsu_threshold(image: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold; falls back to a percentile split without scikit-image."""
    flat = np.asarray(image, dtype=np.float64).ravel()
    if _SKIMAGE:
        return float(threshold_otsu(flat, nbins=nbins))
    hist, edges = np.histogram(flat, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist).astype(float)
    m = np.cumsum(hist * centres)
    total_w, total_m = w[-1], m[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (total_m * w - m * total_w) ** 2 / (w * (total_w - w))
    return float(centres[int(np.nanargmax(between))])


def vessel_mask(image: np.ndarray, threshold: float | None = None,
                low_ratio: float = 1.0) -> np.ndarray:
    """Vessel mask at ``threshold``, optionally grown by hysteresis.

    ``low_ratio = 1.0`` (the default) is a plain threshold: ``image >= threshold``.
    Below 1.0 the mask is grown from every voxel at or above ``threshold`` into
    the connected regions that reach ``low_ratio * threshold``.

    The default used to be 0.6, and on a real Dryad stack that quietly flooded
    the background: measured on M2 planes, the fraction of the plane marked as
    vessel went **0.31 -> 0.87** at the surface and **0.06 -> 0.15** deep, because
    a 16-bit stack carries faint halo/in-scatter just above 0.6 * threshold
    wherever the vessels are dense. A reference mask covering 87 % of the image
    makes every overlap metric look excellent and destroys the skeleton counts
    (73 % of the volume above threshold turned the thinned result into a solid
    blob), so growth is now opt-in and every caller that wants it has to say so
    and report it. The 2 x 2 comparison this protects is the one that matters:
    the same threshold, from the reference only, applied unchanged to every arm.
    """
    img = np.asarray(image, dtype=np.float32)
    thr = otsu_threshold(img) if threshold is None else float(threshold)
    strong = img >= thr
    weak = img >= thr * low_ratio
    if not _SKIMAGE:
        return weak
    from scipy import ndimage as ndi
    keep = ndi.binary_propagation(strong, mask=weak)
    return keep | strong


def vesselness(image: np.ndarray, sigmas=DEFAULT_SIGMAS) -> np.ndarray:
    """Hessian-based tubular-structure response (Sato), bright ridges.

    Why not threshold the image directly: on the Dryad stacks a per-plane Otsu
    on raw intensity marks **37-55 % of a plane** as vessel, because the
    neuropil sits close to the vessel brightness and brightness varies steeply
    with depth. A mask covering a third of the volume is not a vessel tree, and
    skeletonising it yields a surface, not centrelines -- measured on 40 real
    planes, 95 % of the "skeleton" voxels came out as junction voxels (>= 3
    neighbours), which is the signature of a skeletonised solid.

    The Sato response is high only where the local Hessian looks like a bright
    tube, so its Otsu split lands at **4 % of the volume** -- the right order for
    a vascular volume fraction -- and the resulting skeleton is one voxel wide
    with 5 % junction voxels instead of the 95 % a solid-blob mask produced.

    ``sigmas`` are in **voxel** units and the Hessian is taken on the acquisition
    grid, which is anisotropic (5 um in z, 0.98 um in xy). So the response sees
    xy tubules of ~1-2 um radius, while a structure has to span more than one z
    step (~5 um) before the z curvature term contributes. That asymmetry is a
    property of the reference pipeline, identical for every prediction, and it is
    the reason a depth-dependent conclusion is read off the *vessel* metrics of
    slabs rather than off PSNR. ``(1, 2)`` rather than ``(1, 2, 3)``: the third scale
    smooths 15 um in z, larger than the capillary spacing, and on 20 real M2
    planes it costs a third more runtime to move the mask from 3.0 % to 4.2 % of
    the volume and the skeleton from 9183 to 8590 voxels. Both are reportable; the
    cheaper pair is the default so a full volume costs ~5 min rather than ~7.
    """
    img = np.asarray(image, dtype=np.float32)
    if not _SKIMAGE:
        return img
    out = sato(img, sigmas=list(sigmas), black_ridges=False, mode="reflect")
    return np.asarray(out, dtype=np.float32)


def segment_pair(pred: np.ndarray, ref: np.ndarray, *,
                 method: str = "vesselness",
                 sigmas=DEFAULT_SIGMAS,
                 threshold: float | None = None,
                 threshold_mode: str = "per_plane",
                 low_ratio: float = 1.0) -> dict:
    """Segment a prediction and its reference with **one** rule, fitted on the ref.

    This is the single place where the "threshold from the reference only" rule
    lives, so a caller cannot accidentally fit on the prediction. ``method``:

    ``vesselness``  Sato response on both images, threshold fitted on the
        reference response (per plane by default) and applied to the prediction
        response. This is the default because raw-intensity Otsu over-segments
        real stacks (see :func:`vesselness`).
    ``intensity``   threshold on the images themselves (the original recipe).

    Returns ``{"pred_mask", "ref_mask", "threshold", "threshold_mode",
    "segment_method", "threshold_p10_p90"}``; ``threshold`` is the median of the
    per-plane thresholds, ``threshold_p10_p90`` is their spread and is ``None``
    for a single plane or a global threshold.
    """
    p = np.asarray(pred, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    if p.shape != r.shape:
        raise ValueError(f"pred {p.shape} and ref {r.shape} differ")
    if method not in ("vesselness", "intensity"):
        raise ValueError(f"method must be 'vesselness' or 'intensity', got {method!r}")
    if threshold_mode not in ("per_plane", "global"):
        raise ValueError(f"threshold_mode must be 'per_plane' or 'global', "
                         f"got {threshold_mode!r}")

    if method == "vesselness":
        pv, rv = vesselness(p, sigmas), vesselness(r, sigmas)
    else:
        pv, rv = p, r

    spread = None
    if threshold is not None:
        thr = float(threshold)
    elif threshold_mode == "per_plane" and rv.ndim == 3:
        thresholds = np.array([otsu_threshold(rv[z]) for z in range(rv.shape[0])],
                              dtype=np.float64)
        thr = float(np.median(thresholds))
        spread = [float(np.percentile(thresholds, 10)),
                  float(np.percentile(thresholds, 90))]
    else:
        thr = otsu_threshold(rv)

    if threshold_mode == "per_plane" and threshold is None and rv.ndim == 3:
        pm = np.stack([_threshold_plane(pv[z], t, low_ratio) for z, t in
                       enumerate(thresholds)])
        rm = np.stack([_threshold_plane(rv[z], t, low_ratio) for z, t in
                       enumerate(thresholds)])
    else:
        pm = _threshold_plane(pv, thr, low_ratio)
        rm = _threshold_plane(rv, thr, low_ratio)
    return {"pred_mask": pm, "ref_mask": rm, "threshold": thr,
            "threshold_mode": threshold_mode, "segment_method": method,
            "threshold_p10_p90": spread}


def _threshold_plane(values: np.ndarray, threshold: float,
                     low_ratio: float) -> np.ndarray:
    """One-plane threshold; an all-zero plane yields an empty mask, not a full one."""
    if float(np.max(values, initial=0.0)) <= 0.0:
        return np.zeros(values.shape, dtype=bool)
    return vessel_mask(values, threshold=threshold, low_ratio=low_ratio)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """``2|A n B| / (|A| + |B|)``; identical inputs score exactly 1.0.

    Two empty masks score 1.0 (they agree that nothing is there) rather than
    0/0; the convention is recorded here because it silently changes the mean
    of a per-slice summary.
    """
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def _skeleton(mask: np.ndarray) -> np.ndarray:
    if not _SKIMAGE:
        return mask
    return skeletonize(mask.astype(bool))


def cl_dice(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    """Centreline Dice (Shit et al. 2021): overlap between skeleton and mask.

    Topology-sensitive: a prediction that fills a gap in a vessel scores high on
    Dice but low here, because its centreline runs where the reference has none.
    """
    p, r = pred_mask.astype(bool), ref_mask.astype(bool)
    if p.sum() == 0 or r.sum() == 0:
        return 0.0
    sp, sr = _skeleton(p), _skeleton(r)
    t_prec = (sp & r).sum() / max(1, sp.sum())
    t_sens = (sr & p).sum() / max(1, sr.sum())
    if t_prec + t_sens == 0:
        return 0.0
    return float(2 * t_prec * t_sens / (t_prec + t_sens))


def mask_vessel_metrics(pred_mask: np.ndarray, ref_mask: np.ndarray,
                        tolerance_vox: float = 0.0) -> dict:
    """Overlap metrics for two **already segmented** masks.

    The sibling of :func:`vessel_metrics` for callers that segmented with
    :func:`segment_pair` (e.g. on the vesselness response) and must not threshold
    a second time -- a second threshold would be a second operating point.
    """
    p, r = np.asarray(pred_mask, dtype=bool), np.asarray(ref_mask, dtype=bool)
    out = {
        "dice": dice(p, r),
        "cl_dice": cl_dice(p, r),
        "pred_fraction": float(p.mean()),
        "ref_fraction": float(r.mean()),
        "tolerance_vox": tolerance_vox,
    }
    if _SKIMAGE:
        out["n_components_pred"] = int(label(p).max())
        out["n_components_ref"] = int(label(r).max())
    else:                                               # pragma: no cover
        out["n_components_pred"] = -1
        out["n_components_ref"] = -1
    out["n_empty"] = int((not p.any()) or (not r.any()))
    return out


def vessel_metrics(pred: np.ndarray, ref: np.ndarray,
                   threshold: float | None = None,
                   tolerance_vox: float = 0.0) -> dict:
    """Structural metrics for one predicted plane or volume vs the reference.

    Returns
    -------
    dice, cl_dice, pred_fraction, ref_fraction, n_components_pred,
    n_components_ref, n_empty.

    ``threshold`` is fitted on the **reference** only, then applied to both
    images, so the operating point never depends on the prediction.
    """
    p = np.asarray(pred, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    thr = otsu_threshold(r) if threshold is None else float(threshold)

    pm = vessel_mask(p, threshold=thr)
    rm = vessel_mask(r, threshold=thr)

    out = {
        "dice": dice(pm, rm),
        "cl_dice": cl_dice(pm, rm),
        "pred_fraction": float(pm.mean()),
        "ref_fraction": float(rm.mean()),
        "threshold": thr,
        "tolerance_vox": tolerance_vox,
    }
    if _SKIMAGE:
        out["n_components_pred"] = int(label(pm).max())
        out["n_components_ref"] = int(label(rm).max())
    else:
        out["n_components_pred"] = -1
        out["n_components_ref"] = -1
    out["n_empty"] = int(pm.sum() == 0 or rm.sum() == 0)
    return out

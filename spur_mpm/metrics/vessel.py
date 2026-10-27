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

__all__ = ["otsu_threshold", "vessel_mask", "dice", "cl_dice", "vessel_metrics"]

try:  # scikit-image is a declared dependency, but keep import errors legible
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize
    from skimage.measure import label
    _SKIMAGE = True
except Exception:                                   # pragma: no cover
    _SKIMAGE = False


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
                low_ratio: float = 0.6) -> np.ndarray:
    """Hysteresis-style vessel mask: high threshold plus connected 60% regions."""
    img = np.asarray(image, dtype=np.float32)
    thr = otsu_threshold(img) if threshold is None else float(threshold)
    strong = img >= thr
    weak = img >= thr * low_ratio
    if not _SKIMAGE:
        return weak
    from scipy import ndimage as ndi
    keep = ndi.binary_propagation(strong, mask=weak)
    return keep | strong


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

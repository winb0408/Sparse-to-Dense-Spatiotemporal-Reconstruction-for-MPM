"""Qualitative figures: what does each arm actually produce?

Numbers alone cannot settle whether self-supervision works. A masked-measurement
arm can post a respectable held-out loss while producing a recon that ignores its
own input entirely (SPUR-E003b: SS0 errs by 0.068 at voxels it was *given*,
where interpolation errs by exactly 0). That failure mode is invisible in a
table and obvious in an image, so this script exists.

What it produces, per (fold, sample)::

    <out>/qualitative_<sample>.png  -- sparse input, dense reference, then one
                                       row per arm; eight views, one window per
                                       column, unmeasured voxels painted cyan
    <out>/error_<sample>.png        -- |pred - reference|, one shared error scale,
                                       with the measurement's own error on top
    <out>/support_<sample>.png      -- where the measurements were (what is given)
    <out>/profile_<sample>.png      -- one intensity cut across y, so skipped
                                       rows show up as gaps
    <out>/qualitative_<sample>.npz  -- raw arrays (one shared reference + one
                                       prediction per arm), for re-rendering

The figure is paper-grade: `magma` pseudocolour, a thin gutter between panels,
a short keyword per row and per column, and the quantitative metric for *that
panel* printed inside its lower-right corner. Unmeasured voxels are cyan, which
no point of `magma` reaches, so a gap can never be mistaken for a dark vessel.

One trap that shaped the column set: ``x`` and ``z`` are never subsampled, so an
XZ plane is either measured end-to-end or missing end-to-end depending *only* on
which y row it cuts. A single XZ column therefore shows either content the input
already had -- which is why an early version looked as if the input and the dense
reference were the same picture -- or an empty plane. Both cuts are shown side by
side (``XZ given`` / ``XZ gap``); the second is the panel that says what an arm
invents where nothing was measured.

Design rules, all of them there to keep the figure honest:

* **Same input as evaluation.** The forward pass reuses ``evaluate``'s feature
  selection, centre-plane selection and consistency projection, so the picture
  is of the model that produced the number, not a re-implementation of it.
* **Same window for every arm in a column.** Intensities are windowed on the
  *reference* percentiles of that view; otherwise a darker prediction looks
  like a different structure instead of a wrong one.
* **Same error scale for every arm.** One colour bar per error column, taken
  from the worst arm, so "less red" means "less error" and not "rescaled".
* **Reference is the dense ground truth** -- this is the one place where a
  self-supervised arm's dense reference is displayed. The checkpoint was chosen
  on held-out measurements before this script ran; nothing here feeds back.
* **Everything is ZYX with Z the physical axis**, and the z index runs in file
  order. Panels are labelled with indices, not with "top"/"bottom", because the
  depth-profile audit (SPUR-E000b) is what established which end is which.

    python -m spur_mpm.scripts.qualitative_figures --fold 0 \
        --arm A0 --arm S0=runs/E002_S0_fold0/best.pt \
        --arm SS1=runs/E003_SS1_fold0/best.pt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader

from spur_mpm.common.config import load_config
from spur_mpm.common.tensors import as_reconstruction, select_centre
from spur_mpm.data.datasets import ItemConfig, VolumeCache, VolumeWindowDataset
from spur_mpm.data.split import read_split
from spur_mpm.evaluate import _feature_slice, withholds_at_eval
from spur_mpm.models.registry import eval_features
from spur_mpm.metrics.image_metrics import mae, psnr, split_errors, ssim
from spur_mpm.metrics.vessel import mask_vessel_metrics, segment_pair
from spur_mpm.models.registry import build_model, get_arm
from spur_mpm.physics.acquisition import NoiseConfig
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.scripts.viewspec import (
    annotate, build_view_spec, check_group, check_protocols, fingerprint,
    load_view_spec, mask_geometry, protocol_columns, ref_windows,
    sampling_semantics, window_stability,
    save_view_spec, take, validate_view_spec,
)
from spur_mpm.ssl.holdout import HoldoutSpec

__all__ = ["predict_volume", "render", "main"]

DEFAULTS = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def _arm_spec(text: str) -> tuple[str, str, Path | None]:
    """``LABEL/ARM=CKPT`` / ``ARM=CKPT`` / ``ARM`` -> (label, arm, checkpoint).

    The label is what the figure row says. It differs from the arm name when a
    run is a variant of a registered arm -- E005's ``dc1`` is the SS1 recipe
    trained with ``loss.dc_weight=1.0``, and there is no registry entry to call
    by that name.
    """
    # A '/' before the '=' separates the label from the arm spec; a '/' after
    # it is just part of a checkpoint path.
    eq, slash = text.find("="), text.find("/")
    if slash != -1 and (eq == -1 or slash < eq):
        label, rest = text[:slash], text[slash + 1:]
    else:
        label, rest = "", text
    name, _, ckpt = rest.partition("=")
    name = name.strip()
    return (label.strip() or name), name, (Path(ckpt) if ckpt else None)


@torch.no_grad()
def predict_volume(cfg: dict, sample_id: str, arm_name: str,
                   ckpt: Path | None, device: torch.device) -> dict:
    """Reconstruct every plane of one volume with one arm."""
    arm = get_arm(arm_name)
    cache = VolumeCache.load(cfg["data"]["index"],
                             fold=int(cfg["data"].get("fold", 0)))
    sampling = SamplingSpec.from_dict(cfg["sampling"])
    psf = PSFConfig(**cfg["physics"]["psf"])
    noise = NoiseConfig(**cfg.get("noise", {"poisson_photons": 0.0,
                                            "gaussian_sigma": 0.0, "seed": 0}))
    features = eval_features(arm)

    input_support = str(cfg.get("eval", {}).get("input_support", "full"))
    withhold = withholds_at_eval(arm.supervision, input_support,
                                 bool(cfg.get("eval", {}).get(
                                     "holdout_baselines", False)))
    # A dual arm needs its complementary split even with withholding off, for the
    # same reason as in evaluate.py: each branch only ever sees half the rows.
    holdout = HoldoutSpec(ratio=float(cfg["selfsupervision"]["ratio"]),
                          geometry=str(cfg["selfsupervision"]["geometry"]),
                          seed=0) if (withhold or arm.dual) else None

    geom0 = cache.geometry(sample_id)
    context = int(cfg["data"].get("context_slices", 1))
    if cfg["data"].get("context_z_um"):
        context = geom0.context_depth(context_z_um=float(cfg["data"]["context_z_um"]))

    item_cfg = ItemConfig(
        sampling=sampling, psf=psf, noise=noise,
        warm_start_kind=arm.warm_start, lam=float(cfg["physics"]["lam"]),
        features=features, holdout=holdout, dual=arm.dual,
        supervision="self" if withhold else "dense",
        u_kind=cfg["physics"].get("u_kind", "geo"),
        geo_mode=cfg["physics"].get("geo_mode", "per_axis"),
    )
    ds = VolumeWindowDataset(cache, [sample_id], item_cfg,
                             patch_hw=tuple(cfg["data"]["patch_hw"]),
                             context_depth=context, z_stride=1)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=int(cfg["train"].get("num_workers", 2)))

    model = None
    if arm.needs_training:
        if ckpt is None:
            raise ValueError(f"arm {arm_name} needs a checkpoint")
        model = build_model(arm, context, base=int(cfg["model"]["base"]),
                            window=int(cfg["model"]["window"]),
                            depths=tuple(cfg["model"]["depths"]),
                            heads=tuple(cfg["model"]["heads"])).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=False)["model"])
        model.eval()

    projection = str(cfg.get("eval", {}).get("consistency_projection", "none"))
    preds, targets, m_acq, m_in, zs, y_given = [], [], [], [], [], []
    for batch in loader:
        if model is not None:
            pred = select_centre(as_reconstruction(
                model(batch["inputs"].to(device))).float(),
                                 batch["center_index"]).cpu()
        else:
            pred = batch["warmstart_center"]
        if projection == "hard" and model is not None:
            given = batch["in_mask"].bool()
            pred = torch.where(given, batch["y_in"], pred)
        preds.append(pred)
        # What the *pair* was handed. A dual arm splits the acquisition in two
        # and gives one half to each branch, so neither branch's own support is
        # the input of the model as a whole; the honest "Input" row for a dual
        # arm is the union of the two halves, which is exactly the acquisition.
        # Branch A's support is batch["in_mask"] and branch B's is
        # batch["hold_mask"]; the union's values are branch A's y_in where A
        # looked, and the real measurement (hold_target) elsewhere. For a
        # single-branch arm the union is in_mask itself and this is a no-op.
        m_a = batch["in_mask"].bool()
        if arm.dual:
            m_b = batch["hold_mask"].bool()
            y_given.append(torch.where(m_a, batch["y_in"],
                                       torch.where(m_b, batch["hold_target"],
                                                   torch.zeros_like(batch["y_in"]))))
            m_in.append(m_a | m_b)
        else:
            y_given.append(batch["y_in"])
            m_in.append(m_a)
        targets.append(batch["target"])
        m_acq.append(batch["acq_mask"].bool())
        zs.append(batch["query_z"])

    order = torch.argsort(torch.cat(zs))
    cat = lambda xs: torch.cat(xs)[order].numpy()
    return {
        "arm": arm_name,
        "sample_id": sample_id,
        "checkpoint": str(ckpt) if ckpt else None,
        "supervision": arm.supervision,
        "input_support": input_support,
        "sampling": sampling.tag,
        "sampling_fraction": float(sampling.ideal_fraction()),
        "consistency_projection": projection,
        # The number of planes the arm's window actually spans. Recorded
        # because a kz=4 figure at context 1 and one at context 7 look the same
        # in every other field of this sidecar, and the plan (§3.3) forbids
        # letting a kz setting quietly see more planes.
        "context_depth": int(context),
        "dual": arm.dual,
        "pred": cat(preds).astype(np.float32),
        # what the arm was handed: zero-filled measurements. Rendered with the
        # unmeasured voxels masked out, so the figure shows the acquisition.
        "y_in": cat(y_given).astype(np.float32),
        "target": cat(targets).astype(np.float32),
        "acq_mask": cat(m_acq),
        "in_mask": cat(m_in),
        "z": torch.cat(zs)[order].numpy(),
        # Physical pitch, so a saved view spec states the geometry it is
        # drawn in instead of an implicit isotropic assumption.
        "voxel_um": (float(geom0.z_step_um), float(geom0.voxel_xy_um),
                     float(geom0.voxel_xy_um)),
    }


def _metrics(vol: dict) -> dict:
    """Metrics on the same crop the figure shows, per plane then averaged.

    Plane-wise averaging matches how ``evaluate`` reports vessel metrics, so the
    numbers in the figure agree with ``samples.csv`` up to the crop.
    """
    p, t = torch.from_numpy(vol["pred"]), torch.from_numpy(vol["target"])
    acq = torch.from_numpy(vol["acq_mask"])
    out = {"psnr": float(psnr(p, t)), "mae": float(mae(p, t)),
           "ssim": float(np.mean([float(ssim(p[i][None, None], t[i][None, None]))
                                  for i in range(p.shape[0])]))}
    e = {}
    for i in range(p.shape[0]):
        em, ex = split_errors(p[i], t[i], acq[i])
        e.setdefault("e_measured", []).append(em)
        e.setdefault("e_missing", []).append(ex)
        # Same segmentation rule as ``evaluate``: the plane's Sato vesselness,
        # threshold fitted on the reference plane only.
        seg = segment_pair(p[i].numpy(), t[i].numpy(), method="vesselness",
                           threshold_mode="global", match_area=True)
        for k, v in mask_vessel_metrics(seg["pred_mask"], seg["ref_mask"]).items():
            e.setdefault(k, []).append(v)
        # The equal-area operating point, so the panel can show both numbers:
        # at a sparse budget the reference-fitted Dice mostly reports how far
        # the prediction's contrast moved, which is not a structural claim.
        area = mask_vessel_metrics(seg["pred_mask_area"], seg["ref_mask"])
        e.setdefault("dice_area", []).append(area["dice"])
    for k, v in e.items():
        out[k] = float(np.nanmean(v)) if v else float("nan")
    return out


def _tail(img: np.ndarray, q: float = 99.5) -> float:
    """High percentile of the finite voxels only.

    An XZ cut on a never-measured row is entirely NaN, and ``np.nanpercentile``
    still warns (and returns NaN) on such a slice. Returning 0 here keeps that
    column from poisoning the shared error scale.
    """
    v = img[np.isfinite(img)]
    return 0.0 if v.size == 0 else float(np.percentile(v, q))


def _mid(img: np.ndarray) -> float | None:
    """Mean of the finite voxels, or None when the panel has no data at all."""
    v = img[np.isfinite(img)]
    return None if v.size == 0 else float(v.mean())


# --------------------------------------------------------------------------
# paper-grade rendering: magma + one window per column + cyan for "no data"
# --------------------------------------------------------------------------

CMAP = "magma"
MISSING = "#22d3ee"          # bright cyan: no point of magma reaches it
_NOTE = dict(fontsize=5.6, color="white", ha="right", va="bottom",
             bbox=dict(facecolor="black", alpha=0.55, lw=0.0, pad=1.3))


def _show(ax, img, rng, cmap: str = CMAP, bad: str | None = MISSING):
    im = ax.imshow(img, cmap=cmap, vmin=rng[0], vmax=rng[1],
                   interpolation="nearest", aspect="auto", resample=False)
    if bad is not None:
        im.cmap.set_bad(bad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def _note(ax, text: str) -> None:
    """The panel's own number, inside its lower-right corner."""
    ax.text(0.972, 0.028, text, transform=ax.transAxes, zorder=6, **_NOTE)


def _panel_metrics(pred: np.ndarray, ref: np.ndarray) -> dict | None:
    """Score exactly the 2-D picture that is on screen.

    Computed on the shown panel rather than averaged over the volume, so the
    reader never has to reconcile a caption with a picture: the PSNR printed in
    a corner is the PSNR of that corner's pixels against the same crop of the
    dense reference.
    """
    m = np.isfinite(pred) & np.isfinite(ref)
    if not m.any():
        return None
    d = pred[m] - ref[m]
    mse = float(np.mean(d * d))
    p = 99.0 if mse <= 1e-12 else min(99.0, float(-10.0 * np.log10(mse)))
    out = {"psnr": p, "dice": float("nan"), "dice_area": float("nan"),
           "cover": float(m.mean())}
    if m.all():
        try:
            seg = segment_pair(pred, ref, method="vesselness",
                               threshold_mode="global", match_area=True)
            out["dice"] = float(mask_vessel_metrics(seg["pred_mask"],
                                                    seg["ref_mask"])["dice"])
            out["dice_area"] = float(mask_vessel_metrics(seg["pred_mask_area"],
                                                         seg["ref_mask"])["dice"])
        except Exception:                        # tiny crops can break the mask
            pass
    return out

def render(vols: list[dict], out_dir: Path, dpi: int = 200,
           tag: str = "", spec: dict | None = None,
           zoom: int = 96) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = vols[0]["sample_id"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = vols[0]["target"]
    if spec is None:
        spec = build_view_spec(ref, sample_id=sample,
                               voxel_um=vols[0].get("voxel_um", (1.0, 1.0, 1.0)),
                               zoom=zoom)
    else:
        validate_view_spec(spec, ref.shape)
    cols = [c["key"] for c in protocol_columns(spec, vols[0]["in_mask"])]
    col_meta = {c["key"]: c for c in protocol_columns(spec, vols[0]["in_mask"])}
    views = {c: take(ref, col_meta[c], spec) for c in cols}
    # One window per column, taken from the dense reference and reused by every
    # row. A dim prediction must read as a dim prediction; re-normalising each
    # panel would turn a wrong intensity back into a plausible picture.
    rng = ref_windows(ref, [col_meta[c] for c in cols], spec)
    rep = check_group(vols, spec)
    if rep["target_diff"] != 0.0:
        raise RuntimeError(f"arms disagree on the dense reference: "
                           f"max|d|={rep['target_diff']}; the figure would lie")
    written: list[Path] = []

    # ---- figure 1: sparse input -> reconstructions -> dense reference -------
    # Row order is the argument: what was measured, the dense truth every arm is
    # scored against, then one row per arm. Row 0 is a *masked* array so the
    # renderer cannot interpolate the gaps into something that looks inferable,
    # and the gaps are cyan on purpose -- grey on grey is what made an earlier
    # version read as if input and reference were the same picture.
    given = np.where(vols[0]["in_mask"], vols[0]["y_in"], np.nan)
    gviews = {c: take(given, col_meta[c], spec) for c in cols}
    labels = ["Input", "GT"] + [v["arm"] for v in vols]
    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, len(cols),
                             figsize=(2.30 * len(cols), 2.20 * n_rows),
                             gridspec_kw=dict(wspace=0.035, hspace=0.035),
                             squeeze=False)

    for c, col in enumerate(cols):
        g = gviews[col]
        _show(axes[0][c], g, rng[col])
        m = _panel_metrics(g, views[col])
        # An XZ cut on a row that was never measured has nothing to score; say
        # so instead of printing a placeholder number.
        _note(axes[0][c], "meas 0%\n(no data)" if m is None else
              f"meas {100 * m['cover']:.0f}%\nPSNR {m['psnr']:.2f}")
        cb = col_meta[col]
        axes[0][c].set_title(
            col if cb["role"] == "fixed" else f"{col} (diagnostic)\n{cb['note']}",
            fontsize=7.4 if cb["role"] != "fixed" else 8.5, pad=3.0)
        if cb["role"] == "fixed":
            axes[0][c].set_xlabel(cb["note"], fontsize=5.8, labelpad=1.0)

    for c, col in enumerate(cols):
        _show(axes[1][c], views[col], rng[col])
        _note(axes[1][c], "reference")

    for i, v in enumerate(vols):
        vv = {c: take(v["pred"], col_meta[c], spec) for c in cols}
        for c, col in enumerate(cols):
            _show(axes[i + 2][c], vv[col], rng[col])
            m = _panel_metrics(vv[col], views[col])
            if m is not None:
                # Both operating points, so a reader can see whether a low Dice
                # is missing structure or a prediction that came back dimmer.
                _note(axes[i + 2][c], f"PSNR {m['psnr']:.2f}\n"
                                      f"Dice {m['dice']:.3f}\n"
                                      f"Dice★ {m['dice_area']:.3f}")

    for r, lab in enumerate(labels):
        axes[r][0].text(-0.05, 0.5, lab, transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)

    fig.suptitle(sample, fontsize=10, y=0.995)
    geom = mask_geometry(vols[0]["in_mask"])
    dual = [v["arm"] for v in vols if v.get("dual")]
    fig.text(0.5, 0.001,
             f"magma, one window per column (dense reference, p0.5-p99.5) | "
             f"cyan = never measured | {vols[0].get('sampling', '?')} | "
             f"input support {vols[0]['input_support']}, "
             f"projection {vols[0]['consistency_projection']} | "
             + (f"{'/'.join(dual)}: Input = union of the two branches "
                f"(each saw half) | " if dual else "")
             + f"y period {geom['y_period']}, z planes "
               f"{geom['z_planes_measured']}/{geom['z_planes_total']}"
             + f" | zoom {spec['zoom']}x{spec['zoom']} at "
               f"(z={spec['z_view']}, y={spec['roi'][0]}, x={spec['roi'][1]})",
             ha="center", va="bottom", fontsize=6.2, color="0.25")
    p1 = out_dir / f"qualitative_{sample}{tag}.png"
    fig.savefig(p1, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    written.append(p1)

    # ---- figure 2: error maps, one shared scale for every row ----------------
    # Fixed panels only: an error map of a *diagnostic* panel would change which
    # pixels are on screen between protocols, which is the thing this rewrite
    # exists to stop.
    ecols = [c for c in cols if col_meta[c]["role"] == "fixed"][:4]
    eviews = {c: views[c] for c in ecols}
    errs = [{c: np.abs(take(v["pred"], col_meta[c], spec) - eviews[c])
             for c in ecols} for v in vols]
    # Row 0 is what the reconstruction cannot be blamed for: the measurements
    # the arm was given, minus the dense reference, on the voxels it did take.
    a_err = np.abs(given - ref)
    av = {c: take(a_err, col_meta[c], spec) for c in ecols}
    vmax = max(_tail(a[c]) for a in [*errs, av] for c in ecols)
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(len(vols) + 1, len(ecols),
                             figsize=(2.5 * len(ecols), 2.2 * (len(vols) + 1)),
                             gridspec_kw=dict(wspace=0.035, hspace=0.035),
                             squeeze=False)
    for c, col in enumerate(ecols):
        _show(axes[0][c], av[col], (0.0, vmax), cmap="inferno")
        mae_v = _mid(av[col])
        _note(axes[0][c], "no data" if mae_v is None else
              f"MAE {mae_v:.4f}\n(measured only)")
        axes[0][c].set_title(f"{col} ({col_meta[col]['note']})", fontsize=8.5,
                             pad=3.0)
    for i, (v, e) in enumerate(zip(vols, errs)):
        for c, col in enumerate(ecols):
            _show(axes[i + 1][c], e[col], (0.0, vmax), cmap="inferno")
            mae_e = _mid(e[col])
            _note(axes[i + 1][c], "no data" if mae_e is None
                  else f"MAE {mae_e:.4f}")
    for r, lab in enumerate(["Measured"] + [v["arm"] for v in vols]):
        axes[r][0].text(-0.05, 0.5, lab, transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"{sample} -- |pred - reference|, one scale for every row "
                 f"(0 to {vmax:.4f})", fontsize=10, y=0.995)
    fig.text(0.5, 0.001, f"cyan = never measured | inferno | "
                         f"MAE printed per panel, on the shown crop",
             ha="center", va="bottom", fontsize=6.2, color="0.25")
    p2 = out_dir / f"error_{sample}{tag}.png"
    fig.savefig(p2, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    written.append(p2)

    # ---- figure 3: what the arm was given ------------------------------------
    # Every sentence here is measured off the mask. The old caption asserted
    # "x and z are never subsampled", which is false for z_k4 and yz_* -- the
    # very protocols this panel exists to explain.
    acq = vols[0]["acq_mask"]
    geom = mask_geometry(acq)
    xy_key = "XY primary"
    xz_key = next((c for c in cols if col_meta[c]["axis"] == "xz"), "XZ")
    fig = plt.figure(figsize=(13.0, 3.6))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(take(acq.astype(np.float32), col_meta[xy_key], spec), cmap="gray",
               aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax1.set_title(f"measured voxels, XY z={spec['z_view']}", fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(take(acq.astype(np.float32), col_meta[xz_key], spec), cmap="gray",
               aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_title(f"measured voxels, XZ {col_meta[xz_key]['note']}", fontsize=9)
    ax2.set_xlabel("x"); ax2.set_ylabel("z")
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(acq.reshape(acq.shape[0], -1).mean(axis=1), np.arange(acq.shape[0]),
             lw=1.0, color="#c2410c")
    ax3.set_xlabel("measured fraction of the plane"); ax3.set_ylabel("z index")
    ax3.set_title(f"per-plane density "
                  f"({geom['z_planes_measured']}/{geom['z_planes_total']} planes)",
                  fontsize=9)
    ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(acq.mean(axis=(0, 2)), np.arange(acq.shape[1]), lw=1.0,
             color="#1d4ed8")
    ax4.set_xlabel("measured fraction of the row"); ax4.set_ylabel("y index")
    ax4.set_title(f"per-row density (period {geom['y_period']})", fontsize=9)
    ax4.grid(alpha=0.3)
    axes_text = (f"white = measured | y: "
                 + ("every row" if geom["y_dense"] else
                    f"period {geom['y_period']} ({geom['kept_y_rows']} rows kept)")
                 + " | x: " + ("every column" if geom["x_dense"] else "subsampled")
                 + " | z: " + ("every plane" if geom["z_dense"] else
                               f"{geom['z_planes_measured']}/"
                               f"{geom['z_planes_total']} planes kept")
                 + f" | measured fraction {100 * geom['measured_fraction']:.1f}%")
    fig.text(0.5, 0.005, axes_text, ha="center", fontsize=6.4, color="0.25")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    p3 = out_dir / f"support_{sample}{tag}.png"
    fig.savefig(p3, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    written.append(p3)

    # ---- figure 4: one row-cut, so the missing rows are visible as gaps ------
    # The 2D views above are the right overview but they cannot show whether a
    # method *fills* a skipped row or smears its neighbours. A single intensity
    # cut across y does, and it is the panel to look at first.
    z_mid = int(spec["z_view"])
    y0, x0, n = (int(v) for v in spec["roi"])
    x_cut = x0 + int(np.argmax(ref[z_mid, y0:y0 + n, x0:x0 + n].var(axis=0)))
    m_in0 = vols[0]["in_mask"][z_mid]
    y_given_row = vols[0]["y_in"][z_mid][:, x_cut]
    styles = ["#111111", "#c2410c", "#1d4ed8", "#15803d", "#7e22ce", "#a16207",
              "#be185d", "#0e7490"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    for ax, ylim in zip(axes, [None, (0, 34)]):
        ax.plot(ref[z_mid][:, x_cut], np.arange(ref.shape[1]), color="#111111",
                lw=1.6, label="GT", zorder=5)
        ax.plot(np.where(m_in0[:, x_cut], y_given_row, np.nan),
                np.arange(ref.shape[1]), "o", ms=3.0, mfc=MISSING, mec="none",
                label="measured", zorder=6)
        for k, v in enumerate(vols):
            ax.plot(v["pred"][z_mid][:, x_cut], np.arange(ref.shape[1]),
                    lw=1.0, alpha=0.95, color=styles[(k + 1) % len(styles)],
                    label=v["arm"])
        if ylim is not None:
            start = max(0, int(np.argmax(m_in0[:, x_cut])) - 6)
            ax.set_ylim(min(start + 34, ref.shape[1]), start)
        ax.set_xlabel("intensity (normalised)")
        ax.set_ylabel("row index y")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"whole column (z={z_mid}, x={x_cut})", fontsize=9)
    axes[1].set_title("zoom: measured rows (dots) vs skipped rows", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(f"{sample} -- intensity along y; the gaps are rows that were "
                 f"never measured", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p4 = out_dir / f"profile_{sample}{tag}.png"
    fig.savefig(p4, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    written.append(p4)
    return written

def render_comparison(groups: list[dict], out_dir: Path, spec: dict,
                      dpi: int = 200, sample: str = "",
                      name: str | None = None) -> list[Path]:
    """One figure that puts two or more sparse protocols side by side.

    Only the *fixed* panels are used, because those are the ones whose
    coordinates do not depend on the protocol: this is the figure the plan asks
    for (same sample, same ROI, same dense reference, different sampling). The
    measured-support mask goes in as its own column, since that is what actually
    differs between the rows.

    Every invariant is checked before a pixel is drawn, and a violation raises:
    a comparison figure whose rows look at different pixels is worse than no
    figure at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = check_protocols(groups, spec)
    if report["target_diff"] != 0.0 or report["fixed_view_diff"] != 0.0:
        raise RuntimeError(
            "the dense reference differs between protocols "
            f"(target max|d|={report['target_diff']}, fixed view max|d|="
            f"{report['fixed_view_diff']}): that is a data-path bug, not a "
            "figure bug -- fix the cache/normalisation before publishing")

    ref = np.asarray(groups[0]["vols"][0]["target"])
    cols = [c for c in protocol_columns(spec, groups[0]["vols"][0]["in_mask"])
            if c["role"] == "fixed"]
    meta = {c["key"]: c for c in cols}
    keys = [c["key"] for c in cols]
    views = {k: take(ref, meta[k], spec) for k in keys}
    rng = ref_windows(ref, cols, spec)

    rows = []
    for g in groups:
        rows.append(("input", g))
        for v in g["vols"]:
            rows.append((v["arm"], g))

    fig, axes = plt.subplots(len(rows) + 1, len(keys) + 1,
                             figsize=(2.30 * (len(keys) + 1), 2.05 * (len(rows) + 1)),
                             gridspec_kw=dict(wspace=0.035, hspace=0.035),
                             squeeze=False)
    for c, k in enumerate(keys):
        _show(axes[0][c], views[k], rng[k])
        axes[0][c].set_title(k, fontsize=8.5, pad=3.0)
        axes[0][c].set_xlabel(meta[k]["note"], fontsize=5.8, labelpad=1.0)
    axes[0][len(keys)].text(0.5, 0.5, "dense reference\n(same for every row)",
                            transform=axes[0][len(keys)].transAxes, ha="center",
                            va="center", fontsize=7)
    axes[0][len(keys)].set_xticks([]); axes[0][len(keys)].set_yticks([])

    for r, (label, g) in enumerate(rows, start=1):
        v = g["vols"][0]
        if label == "input":
            masked = np.where(v["in_mask"], v["y_in"], np.nan)
            for c, k in enumerate(keys):
                _show(axes[r][c], take(masked, meta[k], spec), rng[k])
                axes[r][c].set_title("")
            frac = float(np.mean(v["in_mask"]))
            axes[r][len(keys)].text(0.5, 0.5,
                                    f"measured\n{100 * frac:.1f}%",
                                    transform=axes[r][len(keys)].transAxes,
                                    ha="center", va="center", fontsize=7.5)
        else:
            vol = next(x for x in g["vols"] if x["arm"] == label)
            for c, k in enumerate(keys):
                img = take(vol["pred"], meta[k], spec)
                _show(axes[r][c], img, rng[k])
                m = _panel_metrics(img, views[k])
                if m is not None:
                    _note(axes[r][c], f"PSNR {m['psnr']:.2f}\n"
                                      f"Dice {m['dice']:.3f}\n"
                                      f"Dice★ {m['dice_area']:.3f}")
            axes[r][len(keys)].imshow(
                take(v["in_mask"].astype(np.float32), meta["XY primary"], spec),
                cmap="gray", aspect="auto", interpolation="nearest",
                vmin=0, vmax=1)
            axes[r][len(keys)].set_title("mask, XY primary", fontsize=7.4)
            axes[r][len(keys)].set_xticks([]); axes[r][len(keys)].set_yticks([])
        axes[r][0].text(-0.05, 0.5, f"{g['level']}\n{label}" if label != "input"
                        else f"{g['level']}\ninput",
                        transform=axes[r][0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)

    fig.suptitle(f"{sample} -- the same dense reference under different sparse "
                 f"protocols (fixed ROI z={spec['z_view']}, y={spec['y_view']}, "
                 f"x={spec['x_view']})", fontsize=10, y=0.995)
    fig.text(0.5, 0.001,
             "one window per column taken from the dense reference and reused by "
             "every row | cyan = never measured | the mask column is the only "
             "thing that is supposed to differ", ha="center", fontsize=6.2,
             color="0.25")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = "_".join(slug(g["level"]) for g in groups)
    png = out_dir / (name or f"compare_{sample}_{levels}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [png]


def _freeze(spec: dict, ref: np.ndarray, mask: np.ndarray, cfg: dict) -> dict:
    """Attach the reference-derived windows and the provenance to a view spec.

    Only the *fixed* panels' windows go in: a protocol diagnostic panel exists
    in one protocol only, so freezing its window would make the spec look
    protocol-dependent when the geometry is not.
    """
    cols = protocol_columns(spec, mask)
    fixed = [c for c in cols if c["role"] == "fixed"]
    return annotate(
        spec,
        windows=ref_windows(ref, fixed, spec),
        provenance=fingerprint({"index": cfg["data"]["index"],
                                "split": cfg["data"]["split"]}, cfg),
        columns=cols)


def slug(text: str) -> str:
    """A group label reduced to what a filename can carry."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _compare_groups(cfg: dict, manifest: dict, device, sample: str) -> list[dict]:
    """Run every (protocol, arm) of a comparison manifest."""
    from spur_mpm.common.config import apply_overrides

    groups = []
    for g in manifest["groups"]:
        cfg_g = apply_overrides(cfg, list(g.get("overrides", [])))
        vols = []
        for label, ckpt in g["arms"].items():
            v = predict_volume(cfg_g, sample, label, Path(ckpt) if ckpt else None,
                               device)
            v["arm"] = label
            vols.append(v)
        # The protocol as a spec, not just as the ``y_k4`` tag the volume carries:
        # the consistency report has to compare the mask on screen against the
        # ky/kz and expected fraction this panel was *meant* to be drawn under.
        groups.append({"level": g["level"], "vols": vols,
                       "sampling": SamplingSpec.from_dict(
                           cfg_g["sampling"]).to_dict()})
    return groups


def consistency_fields(groups: list[dict], spec: dict, ref: np.ndarray,
                       sample: str) -> dict:
    """The three report blocks that used to be written by hand.

    ``sampling_semantics`` needs the protocol *spec* (ky/kz and the fraction it
    implies). A volume sidecar only carries the ``y_k4`` tag, so the spec has to
    travel on the group; a group without one is a bug in the caller, and letting
    it fall through to ``{}`` would make both checks vacuous and pass.
    """
    windows = window_stability(spec, ref, groups[0]["vols"][0]["in_mask"])
    semantics, shapes = {}, {}
    for g in groups:
        if not isinstance(g.get("sampling"), dict):
            raise ValueError(f"group {g['level']!r} carries no sampling spec: "
                             f"the mask cannot be checked against a protocol")
        vol0 = g["vols"][0]
        mask = np.asarray(vol0["in_mask"])
        semantics[g["level"]] = sampling_semantics(mask, g["sampling"])
        shapes[g["level"]] = {
            "shape": list(mask.shape),
            "matches_spec": list(mask.shape) == list(spec["shape"]),
            "voxel_um": vol0.get("voxel_um"),
            "voxel_ok": (vol0.get("voxel_um") is None
                         or [float(v) for v in spec.get("voxel_um", [])]
                         == [float(v) for v in vol0["voxel_um"]]),
        }
    return {"sample_id": sample, "view_spec": spec,
            "protocols": [g["level"] for g in groups],
            "measured_fraction": {
                g["level"]: float(np.mean(g["vols"][0]["in_mask"]))
                for g in groups},
            "context_depth": {g["level"]: g["vols"][0].get("context_depth")
                              for g in groups},
            "display_windows": windows,
            "mask_semantics": semantics,
            "geometry": shapes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path("spur_mpm/configs/experiments/E002_supervised_y_k2.yaml"))
    ap.add_argument("--base", type=Path, default=DEFAULTS)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sample", default=None,
                    help="sample_id; default: first test volume of the fold")
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME or NAME=CKPT; repeatable. Default: A0 plus whatever "
                         "--experiments resolve for this fold")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="run-name prefixes, e.g. E002_S0 E003_SS1")
    ap.add_argument("--index", type=Path, default=Path("data_local/dryad_cache/index.json"),
                    help="volume index; base.yaml's placeholder path is not the "
                         "real Dryad cache")
    ap.add_argument("--out", type=Path, default=Path("spur_mpm/figures/qualitative"))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="a.b.c=value overrides, as in evaluate")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--tag", default="",
                    help="appended to every output name, so a second arm set for "
                         "the same sample does not overwrite the first figure")
    ap.add_argument("--view-spec", type=Path, default=None,
                    help="reuse a frozen view geometry instead of deriving one "
                         "from the reference (required for cross-protocol "
                         "comparison: the panels must not be re-chosen)")
    ap.add_argument("--save-view-spec", type=Path, default=None,
                    help="write the view geometry this figure used, so another "
                         "protocol can reuse exactly it")
    ap.add_argument("--zoom", type=int, default=96,
                    help="side of the square zoom panel, in voxels")
    ap.add_argument("--z-view", type=int, default=None,
                    help="explicit Z coordinate of the fixed XY panel and zoom "
                         "box; default: the middle plane. Cross-protocol "
                         "figures should name a plane every compared protocol "
                         "measured, so the shared column is informative in "
                         "every row")
    ap.add_argument("--y-view", type=int, default=None,
                    help="explicit Y coordinate of the fixed XZ panel")
    ap.add_argument("--x-view", type=int, default=None,
                    help="explicit X coordinate of the fixed YZ panel")
    ap.add_argument("--compare", type=Path, default=None,
                    help="JSON manifest of protocol groups; writes "
                         "compare_<sample>_<levels>.png and a consistency "
                         "report, and exits non-zero if the invariants fail")
    ap.add_argument("--figure-name", default=None,
                    help="file name for the --compare figure; default "
                         "compare_<sample>_<levels>.png. §4.4.1 asks for "
                         "compare_y_z_yz_25pct.png in a per-sample directory")
    ap.add_argument("--dump", default="f16", choices=["f16", "f32", "off"],
                    help="raw array dump: float16 (default, ~1/2 the size and "
                         "lossless enough for re-rendering), float32, or off")
    args = ap.parse_args(argv)

    # Same merge order as ``evaluate.main`` so the figure is rendered from the
    # config the numbers were produced with.
    from spur_mpm.common.config import apply_overrides, deep_merge
    cfg = deep_merge(load_config(args.base), load_config(args.config))
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    cfg.setdefault("data", {})["fold"] = args.fold
    cfg["data"]["index"] = str(args.index)

    folds = {int(f["fold"]): f for f in read_split(cfg["data"]["split"])["folds"]}
    test_ids = folds[args.fold]["test"]
    sample = args.sample or test_ids[0]

    # A bare index ("2") means "cuda:2", as in evaluate / benchmark; a device
    # string ("cuda:2", "cpu") is taken as given. Resolved here because the
    # compare branch needs it before the arm list is parsed.
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{dev}" if str(dev).isdigit() else dev)

    if args.compare:
        manifest = json.loads(Path(args.compare).read_text())
        sample = manifest.get("sample", sample)
        groups = _compare_groups(cfg, manifest, device, sample)
        ref = np.asarray(groups[0]["vols"][0]["target"])
        spec = (validate_view_spec(load_view_spec(args.view_spec), ref.shape)
                if args.view_spec else
                build_view_spec(ref, sample_id=sample,
                                voxel_um=groups[0]["vols"][0].get("voxel_um",
                                                                 (1.0, 1.0, 1.0)),
                                zoom=args.zoom, z_view=args.z_view,
                                y_view=args.y_view, x_view=args.x_view))
        out_dir = Path(args.out)
        spec = _freeze(spec, ref, groups[0]["vols"][0]["in_mask"], cfg)
        if args.save_view_spec:
            save_view_spec(spec, args.save_view_spec)
        written = render_comparison(groups, out_dir, spec, dpi=args.dpi,
                                    sample=sample, name=args.figure_name)
        report = check_protocols(groups, spec)
        # Three things the report is asked for that were being *stated* rather
        # than checked: that the window of every panel still comes from this
        # reference, that each mask really is the protocol it is filed under,
        # and that the shape/physical spacing agree with the frozen spec.
        fields = consistency_fields(groups, spec, ref, sample)
        windows = fields["display_windows"]
        semantics, shapes = fields["mask_semantics"], fields["geometry"]
        report.update(fields)
        (out_dir / "consistency_report.json").write_text(
            json.dumps(report, indent=2))
        for pth in written:
            print("wrote", pth)
        ok = (report["target_diff"] == 0.0 and report["fixed_view_diff"] == 0.0
              and windows["windows_ok"]
              and all(gr["target_diff"] == 0.0 and gr["input_diff"] == 0.0
                      and gr["view_diff"] == 0.0 for gr in report["groups"])
              and all(sem["fraction_ok"] and sem["axes_ok"]
                      for sem in semantics.values())
              and all(sh["matches_spec"] and sh["voxel_ok"]
                      for sh in shapes.values()))
        if not ok:
            print("CONSISTENCY FAILED", json.dumps(report, indent=2))
            return 2
        print("consistency ok: dense reference and fixed views identical across "
              f"{len(groups)} protocols")
        print("ALL_DONE")
        return 0

    arms: list[tuple[str, str, Path | None]] = [_arm_spec(a) for a in args.arm]
    for exp in (args.experiments or []):
        ckpt = Path("runs") / f"{exp}_fold{args.fold}" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{exp}: no checkpoint at {ckpt}")
        arms.append((exp.split("_", 1)[1], exp.split("_", 1)[1], ckpt))
    if not arms:
        raise SystemExit("give at least one --arm or --experiments")

    vols = []
    for label, arm_name, ckpt in arms:
        v = predict_volume(cfg, sample, arm_name, ckpt, device)
        v["arm"] = label
        if vols and (not np.array_equal(v["target"], vols[0]["target"]) or
                     not np.array_equal(v["acq_mask"], vols[0]["acq_mask"]) or
                     not np.array_equal(v["in_mask"], vols[0]["in_mask"]) or
                     not np.allclose(v["y_in"], vols[0]["y_in"])):
            raise RuntimeError("arms disagree on the reference volume or the "
                               "support: the items are no longer same-input and "
                               "the figure would lie")
        v["metrics"] = _metrics(v)
        vols.append(v)
        print(f"{label:8s} psnr={v['metrics']['psnr']:.4f} "
              f"dice={v['metrics'].get('dice', float('nan')):.4f} "
              f"e_measured={v['metrics']['e_measured']:.4f} "
              f"e_missing={v['metrics']['e_missing']:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = np.asarray(vols[0]["target"])
    spec = (validate_view_spec(load_view_spec(args.view_spec), ref.shape)
            if args.view_spec else
            build_view_spec(ref, sample_id=sample,
                            voxel_um=vols[0].get("voxel_um", (1.0, 1.0, 1.0)),
                            zoom=args.zoom, z_view=args.z_view,
                            y_view=args.y_view, x_view=args.x_view))
    spec = {k: v for k, v in spec.items()
            if k not in ("windows", "fingerprint", "projection")}
    spec = _freeze(spec, ref, vols[0]["in_mask"], cfg)
    if args.save_view_spec:
        save_view_spec(spec, args.save_view_spec)
    tag = args.tag
    if args.dump != "off":
        # Float16 halves the dump and is well inside the precision of the
        # figures; masks stay boolean so a re-analysis cannot change splits.
        # Every arm shares one reference volume and one support map (the loop
        # above asserts it on target, and the item builder derives the masks
        # from the config), so those are stored once instead of per arm.
        def _cast(a):
            return a.astype(np.float16) if (args.dump == "f16" and
                                            a.dtype == np.float32) else a
        np.savez_compressed(
            out_dir / f"qualitative_{sample}{tag}.npz",
            target=_cast(vols[0]["target"]),
            # y_in is what the arm was actually handed (the blurred, noisy
            # measurement, zero-filled). Stored once because it is shared, and
            # stored at all so the figure can be re-rendered without reloading
            # four checkpoints.
            y_in=_cast(vols[0]["y_in"]),
            acq_mask=vols[0]["acq_mask"],
            in_mask=vols[0]["in_mask"],
            z_index=vols[0]["z"],
            arms=np.array([v["arm"] for v in vols]),
            **{f"{v['arm']}_pred": _cast(v["pred"]) for v in vols})
    (out_dir / f"qualitative_{sample}{tag}.json").write_text(json.dumps(
        {"sample_id": sample, "fold": args.fold,
         # Which sparse pattern this figure is about -- the same for every arm,
         # which is why it sits at the top level and not in each arm's record.
         "sampling": vols[0].get("sampling"),
         "sampling_fraction": vols[0].get("sampling_fraction"),
         # Measured off the mask that is actually on screen, so a dump made
         # before the config string was recorded still states its geometry.
         "mask_geometry": mask_geometry(vols[0]["in_mask"]),
         "input_support": vols[0]["input_support"],
         "consistency_projection": vols[0]["consistency_projection"],
         "context_depth": vols[0]["context_depth"],
         # The frozen geometry, so a later figure can reuse these exact cuts.
         "view_spec": spec,
         # The panels actually drawn, with the coordinate each one shows and
         # whether it is a fixed panel or a protocol diagnostic. Recorded so a
         # reader (or a test) can check the labels without opening the PNG.
         "columns": protocol_columns(spec, vols[0]["in_mask"]),
         # The display window used for each panel, taken from the reference.
         "windows": {k: list(v) for k, v in ref_windows(
             np.asarray(vols[0]["target"]),
             protocol_columns(spec, vols[0]["in_mask"]), spec).items()},
         "arms": [{"arm": v["arm"], "checkpoint": v["checkpoint"],
                   "supervision": v["supervision"],
                   # A dual arm's Input row is the union of its two branches;
                   # each branch alone saw half of it.
                   "input_is_branch_union": v["dual"],
                   "metrics_on_shown_crop": v["metrics"]} for v in vols]},
        indent=2))
    for p in render(vols, out_dir, dpi=args.dpi, tag=args.tag, spec=spec):
        print("wrote", p)
    print("ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

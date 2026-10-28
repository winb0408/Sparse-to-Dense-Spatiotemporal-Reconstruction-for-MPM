"""Qualitative figures: what does each arm actually produce?

Numbers alone cannot settle whether self-supervision works. A masked-measurement
arm can post a respectable held-out loss while producing a recon that ignores its
own input entirely (SPUR-E003b: SS0 errs by 0.068 at voxels it was *given*,
where interpolation errs by exactly 0). That failure mode is invisible in a
table and obvious in an image, so this script exists.

What it produces, per (fold, sample)::

    <out>/qualitative_<sample>.png  -- one row per arm, six views, same window
    <out>/error_<sample>.png        -- |pred - reference|, one shared error scale
    <out>/support_<sample>.png      -- where the measurements were (what is given)
    <out>/qualitative_<sample>.npz  -- raw arrays (one shared reference + one
                                       prediction per arm), for re-rendering

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
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spur_mpm.common.config import load_config
from spur_mpm.common.tensors import select_centre
from spur_mpm.data.datasets import ItemConfig, VolumeCache, VolumeWindowDataset
from spur_mpm.data.split import read_split
from spur_mpm.evaluate import _eval_features, _feature_slice, withholds_at_eval
from spur_mpm.metrics.image_metrics import mae, psnr, split_errors, ssim
from spur_mpm.metrics.vessel import vessel_metrics
from spur_mpm.models.registry import build_model, get_arm
from spur_mpm.physics.acquisition import NoiseConfig
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.ssl.holdout import HoldoutSpec

__all__ = ["predict_volume", "render", "main"]

DEFAULTS = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def _arm_spec(text: str) -> tuple[str, Path | None]:
    name, _, ckpt = text.partition("=")
    return name.strip(), (Path(ckpt) if ckpt else None)


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
    features = _eval_features(arm)

    input_support = str(cfg.get("eval", {}).get("input_support", "full"))
    withhold = withholds_at_eval(arm.supervision, input_support)
    holdout = HoldoutSpec(ratio=float(cfg["selfsupervision"]["ratio"]),
                          geometry=str(cfg["selfsupervision"]["geometry"]),
                          seed=0) if withhold else None

    geom0 = cache.geometry(sample_id)
    context = int(cfg["data"].get("context_slices", 1))
    if cfg["data"].get("context_z_um"):
        context = geom0.context_depth(context_z_um=float(cfg["data"]["context_z_um"]))

    item_cfg = ItemConfig(
        sampling=sampling, psf=psf, noise=noise,
        warm_start_kind=arm.warm_start, lam=float(cfg["physics"]["lam"]),
        features=features, holdout=holdout,
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
    preds, targets, m_acq, m_in, zs = [], [], [], [], []
    for batch in loader:
        if model is not None:
            pred = select_centre(model(batch["inputs"].to(device)).float(),
                                 batch["center_index"]).cpu()
        else:
            pred = batch["inputs"][:, _feature_slice(features, "x0", context)]
            pred = pred[:, context // 2]
        if projection == "hard" and model is not None:
            given = batch["in_mask"].bool()
            pred = torch.where(given, batch["y_in"], pred)
        preds.append(pred)
        targets.append(batch["target"])
        m_acq.append(batch["acq_mask"].bool())
        m_in.append(batch["in_mask"].bool())
        zs.append(batch["query_z"])

    order = torch.argsort(torch.cat(zs))
    cat = lambda xs: torch.cat(xs)[order].numpy()
    return {
        "arm": arm_name,
        "sample_id": sample_id,
        "checkpoint": str(ckpt) if ckpt else None,
        "supervision": arm.supervision,
        "input_support": input_support,
        "consistency_projection": projection,
        "pred": cat(preds).astype(np.float32),
        "target": cat(targets).astype(np.float32),
        "acq_mask": cat(m_acq),
        "in_mask": cat(m_in),
        "z": torch.cat(zs)[order].numpy(),
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
        for k, v in vessel_metrics(p[i].numpy(), t[i].numpy()).items():
            e.setdefault(k, []).append(v)
    for k, v in e.items():
        out[k] = float(np.nanmean(v)) if v else float("nan")
    return out


def _views(vol: np.ndarray) -> dict:
    """The six panels, as 2D arrays, from a (Z, Y, X) array."""
    z, h, w = vol.shape
    return {
        "XY z=25%": vol[int(0.25 * z)],
        "XY z=50%": vol[int(0.50 * z)],
        "XY z=75%": vol[int(0.75 * z)],
        "XZ (y=mid)": vol[:, h // 2, :],
        "YZ (x=mid)": vol[:, :, w // 2],
        "MIP-z": vol.max(axis=0),
    }


def _window(img: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(img, (1.0, 99.5))
    return float(lo), float(max(hi, lo + 1e-6))


def render(vols: list[dict], out_dir: Path, dpi: int = 150) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = vols[0]["sample_id"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = vols[0]["target"]
    names = [v["arm"] for v in vols]
    views = _views(ref)
    cols = list(views)
    written: list[Path] = []

    # ---- figure 1: reconstruction -------------------------------------------
    wnd = {c: _window(views[c]) for c in cols}
    fig, axes = plt.subplots(len(vols) + 1, len(cols),
                             figsize=(3.0 * len(cols), 2.6 * (len(vols) + 1)),
                             squeeze=False)
    for r, col in enumerate(cols):
        ax = axes[0][r]
        ax.imshow(views[col], cmap="gray", vmin=wnd[col][0], vmax=wnd[col][1],
                  interpolation="nearest", aspect="auto")
        ax.set_title(col, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].set_ylabel("reference\n(dense GT)", fontsize=9)

    for i, v in enumerate(vols):
        m = _metrics(v)
        label = (f"{v['arm']}  ({ 'SS' if v['supervision'] == 'self' else 'sup' })\n"
                 f"PSNR {m['psnr']:.2f} dB | E_miss {m['e_missing']:.4f}\n"
                 f"E_meas {m['e_measured']:.4f}")
        vv = _views(v["pred"])
        for r, col in enumerate(cols):
            ax = axes[i + 1][r]
            ax.imshow(vv[col], cmap="gray", vmin=wnd[col][0], vmax=wnd[col][1],
                      interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and col.startswith("XY"):
                ax.set_ylabel(label, fontsize=8)
            if r == 0 and col.startswith("XZ"):
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(f"{sample} -- same data, same window, one panel per arm\n"
                 f"input support: {vols[0]['input_support']}, "
                 f"projection: {vols[0]['consistency_projection']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = out_dir / f"qualitative_{sample}.png"
    fig.savefig(p1, dpi=dpi); plt.close(fig)
    written.append(p1)

    # ---- figure 2: error maps, one shared scale per column -------------------
    ecols = ["XY z=50%", "XZ (y=mid)", "MIP-z"]
    errs = [{c: np.abs(_views(v["pred"])[c] - views[c]) for c in ecols}
            for v in vols]
    vmax = max(float(np.percentile(e[c], 99.5)) for e in errs for c in ecols)
    fig, axes = plt.subplots(len(vols) + 1, len(ecols),
                             figsize=(3.2 * len(ecols), 2.6 * (len(vols) + 1)),
                             squeeze=False)
    for r, col in enumerate(ecols):
        ax = axes[0][r]
        im = ax.imshow(np.zeros_like(views[col]), cmap="magma", vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_title(f"|ref| window ({col})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].set_ylabel("reference", fontsize=9)
    for i, (v, e) in enumerate(zip(vols, errs)):
        for r, col in enumerate(ecols):
            ax = axes[i + 1][r]
            im = ax.imshow(e[col], cmap="magma", vmin=0, vmax=vmax,
                           interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_ylabel(v["arm"], fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.5, label="|pred - reference| (shared scale)")
    fig.suptitle(f"{sample} -- absolute error, one colour scale for every arm",
                 fontsize=10)
    p2 = out_dir / f"error_{sample}.png"
    fig.savefig(p2, dpi=dpi); plt.close(fig)
    written.append(p2)

    # ---- figure 3: what the arm was given ------------------------------------
    acq = vols[0]["acq_mask"]
    fig = plt.figure(figsize=(11, 3.4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(_views(acq.astype(np.float32))["XY z=50%"], cmap="gray",
               aspect="auto", interpolation="nearest")
    ax1.set_title("acquired voxels, XY z=50%\n(white = measured)", fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(_views(acq.astype(np.float32))["XZ (y=mid)"], cmap="gray",
               aspect="auto", interpolation="nearest")
    ax2.set_title("acquired voxels, XZ\n(stripes = planes skipped by sampling)",
                  fontsize=9)
    ax2.set_xlabel("x"); ax2.set_ylabel("z")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(acq.reshape(acq.shape[0], -1).mean(axis=1), np.arange(acq.shape[0]),
             lw=1)
    ax3.set_xlabel("measured fraction of the plane"); ax3.set_ylabel("z index")
    ax3.set_title("per-plane measurement density", fontsize=9)
    ax3.grid(alpha=0.3)
    fig.suptitle(f"{sample} -- input support (the only voxels the arm observed)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p3 = out_dir / f"support_{sample}.png"
    fig.savefig(p3, dpi=dpi); plt.close(fig)
    written.append(p3)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path("spur_mpm/configs/experiments/E003_selfsup_y_k2.yaml"))
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
    ap.add_argument("--dpi", type=int, default=150)
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

    arms: list[tuple[str, Path | None]] = [_arm_spec(a) for a in args.arm]
    for exp in (args.experiments or []):
        ckpt = Path("runs") / f"{exp}_fold{args.fold}" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{exp}: no checkpoint at {ckpt}")
        arms.append((exp.split("_", 1)[1], ckpt))
    if not arms:
        raise SystemExit("give at least one --arm or --experiments")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    vols = []
    for name, ckpt in arms:
        v = predict_volume(cfg, sample, name, ckpt, device)
        if vols and (not np.array_equal(v["target"], vols[0]["target"]) or
                     not np.array_equal(v["acq_mask"], vols[0]["acq_mask"]) or
                     not np.array_equal(v["in_mask"], vols[0]["in_mask"])):
            raise RuntimeError("arms disagree on the reference volume or the "
                               "support: the items are no longer same-input and "
                               "the figure would lie")
        v["metrics"] = _metrics(v)
        vols.append(v)
        print(f"{name:8s} psnr={v['metrics']['psnr']:.4f} "
              f"dice={v['metrics'].get('dice', float('nan')):.4f} "
              f"e_measured={v['metrics']['e_measured']:.4f} "
              f"e_missing={v['metrics']['e_missing']:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
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
            out_dir / f"qualitative_{sample}.npz",
            target=_cast(vols[0]["target"]),
            acq_mask=vols[0]["acq_mask"],
            in_mask=vols[0]["in_mask"],
            z_index=vols[0]["z"],
            arms=np.array([v["arm"] for v in vols]),
            **{f"{v['arm']}_pred": _cast(v["pred"]) for v in vols})
    (out_dir / f"qualitative_{sample}.json").write_text(json.dumps(
        {"sample_id": sample, "fold": args.fold,
         "input_support": vols[0]["input_support"],
         "consistency_projection": vols[0]["consistency_projection"],
         "arms": [{"arm": v["arm"], "checkpoint": v["checkpoint"],
                   "supervision": v["supervision"],
                   "metrics_on_shown_crop": v["metrics"]} for v in vols]},
        indent=2))
    for p in render(vols, out_dir, dpi=args.dpi):
        print("wrote", p)
    print("ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

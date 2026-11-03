"""Evaluation against the dense reference, on frozen models only.

    python -m spur_mpm.evaluate --config <experiment>.yaml --arm SS1 \
        --ckpt runs/<run>/best.pt --split test

Two rules this file exists to enforce (plan §2.3, §5.3):

1. **The dense reference is read here and nowhere else** for self-supervised
   arms. It is used to score a frozen checkpoint that was selected on held-out
   measurements, never to choose it.
2. **Errors are split by measurement support.** ``E_measured``,
   ``E_missing`` (and ``E_hold`` when a holdout exists) are reported separately,
   because an aggregate PSNR can improve while the unmeasured region -- the only
   region the method actually claims to fix -- gets worse.

Baseline arms (A0/A1) run through the same window grid and the same seeds as the
trained arms, so the comparison is same-input, not same-distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spur_mpm.common.config import apply_overrides, load_config
from spur_mpm.common.tensors import select_centre
from spur_mpm.data.datasets import ItemConfig, VolumeCache, VolumeWindowDataset
from spur_mpm.data.split import read_split
from spur_mpm.metrics.image_metrics import (mae, psnr, split_errors,
                                           split_psnr, ssim)
from spur_mpm.metrics.statistics import summarise_by_group
from spur_mpm.metrics.vessel import mask_vessel_metrics, segment_pair
from spur_mpm.models.registry import build_model, eval_features, get_arm
from spur_mpm.physics.acquisition import NoiseConfig
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.ssl.holdout import HoldoutSpec

__all__ = ["evaluate", "main"]

DEFAULTS = Path(__file__).parent / "configs" / "base.yaml"

def _feature_slice(features: tuple[str, ...], want: str, context: int) -> slice:
    i = features.index(want)
    return slice(i * context, (i + 1) * context)


def withholds_at_eval(supervision: str, input_support: str,
                      holdout_baselines: bool = False) -> bool:
    """Should the evaluation item keep part of M_acq back?

    For a self-supervised arm under the training-time regime, yes -- that is the
    input it was trained on. Every other combination gets everything it
    measured, because withholding from a supervised arm would silently change
    *its* input budget too.

    ``holdout_baselines`` extends the withholding to the parameter-free arms
    (A0/A1). Those two see nothing but the given voxels, so under the default
    rule an A0 scored "under holdout" is really an A0 scored with twice the
    input, and quoting it next to a self-supervised number compares two
    different budgets. Set it to compare an interpolation floor against a
    network *on the same support*.
    """
    if input_support not in ("full", "holdout"):
        raise ValueError(f"eval.input_support must be 'full' or 'holdout', got "
                         f"{input_support!r}")
    if input_support != "holdout":
        return False
    return supervision == "self" or holdout_baselines


@torch.no_grad()
def _skeleton_row(pred_mask: np.ndarray, ref_mask: np.ndarray, *,
                  stats: bool = True, suffix: str = "") -> dict:
    """Skeleton-based structural scores for one centre plane.

    Takes the masks segmented by :func:`spur_mpm.evaluate.evaluate` -- on the
    plane's Sato vesselness, threshold from the reference only -- so the
    skeleton of a plane is measured on the same object as its Dice. On a raw MPM
    plane an intensity Otsu marks 37-55 % of the pixels as vessel, and a skeleton
    of that is a surface rather than centrelines. ``skel_overlap`` is
    skeleton-pixel agreement within one voxel; the length/branch/junction counts
    say whether the thinning survives -- a prediction that thickens every
    capillary can keep Dice up while merging vessels, which shows here and
    nowhere else.

    ``suffix="_area"`` on the equal-area mask carries the same caveat as the
    starred Dice columns: the reference-fitted skeleton F1 inherits the
    prediction's contrast drift, and at a 25 % budget that is worth 0.09 of F1.
    The counts are not repeated for the second mask -- they describe the
    prediction, and the reference side is identical -- so ``stats=False``.
    """
    from spur_mpm.metrics.skeleton import skeleton_overlap, skeleton_stats

    out = skeleton_overlap(pred_mask, ref_mask, tolerance_vox=1)
    if not stats:
        return {f"{k}{suffix}": v for k, v in out.items()}
    pred_stats = skeleton_stats(pred_mask)
    ref_stats = skeleton_stats(ref_mask)
    for key in ("n_branches", "n_junctions", "n_endpoints",
                "total_length_vox", "mean_radius_vox"):
        out[f"skel_pred_{key}"] = pred_stats.get(key)
        out[f"skel_ref_{key}"] = ref_stats.get(key)
    return out


def evaluate(cfg: dict, arm_name: str, ckpt: str | Path | None,
             split_name: str = "test", out_dir: Path | None = None,
             seed: int = 0, skeleton: bool = False) -> dict:
    device = torch.device(cfg["run"].get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    arm = get_arm(arm_name)
    cache = VolumeCache.load(cfg["data"]["index"],
                              fold=int(cfg["data"].get("fold", 0)))
    folds = {int(f["fold"]): f for f in read_split(cfg["data"]["split"])["folds"]}
    fold = folds[int(cfg["data"]["fold"])]
    sample_ids = fold[split_name]

    sampling = SamplingSpec.from_dict(cfg["sampling"])
    psf = PSFConfig(**cfg["physics"]["psf"])
    noise = NoiseConfig(**cfg.get("noise", {"poisson_photons": 0.0,
                                            "gaussian_sigma": 0.0, "seed": 0}))
    features = eval_features(arm)
    # How much of the acquired data the arm is given at scoring time.
    #
    # "full"     : every acquired measurement (M_acq). A self-supervised arm is
    #              trained with part of M_acq withheld as targets, but withholding
    #              it again at test time would hand it HALF the input the
    #              supervised arms get and turn S1 vs SS1 into a comparison of
    #              measurement budgets rather than of methods. This is the
    #              headline protocol.
    # "holdout"  : the training-time regime, kept as a secondary number because
    #              it is the only one where E_hold exists.
    input_support = str(cfg.get("eval", {}).get("input_support", "full"))
    withhold = withholds_at_eval(arm.supervision, input_support,
                                 bool(cfg.get("eval", {}).get(
                                     "holdout_baselines", False)))
    # A dual arm (E012) is *defined* by the complementary split, so it needs the
    # spec even when withholding is off: each of its branches must still be given
    # only its own half, otherwise it would be scored on an input it never saw.
    # The pair's coverage is then the whole acquisition, like every other arm's.
    holdout = HoldoutSpec(ratio=float(cfg["selfsupervision"]["ratio"]),
                          geometry=str(cfg["selfsupervision"]["geometry"]),
                          seed=seed) if (withhold or arm.dual) else None

    geom0 = cache.geometry(sample_ids[0])
    context = int(cfg["data"].get("context_slices", 1))
    if cfg["data"].get("context_z_um"):
        context = geom0.context_depth(context_z_um=float(cfg["data"]["context_z_um"]))

    item_cfg = ItemConfig(
        sampling=sampling, psf=psf, noise=noise,
        warm_start_kind=arm.warm_start, lam=float(cfg["physics"]["lam"]),
        # "self" is what makes _make_item withhold; a self-supervised arm scored
        # with the full acquired set is built as a dense item and scored the same
        # way, so nothing about the numbers depends on the label.
        features=features, holdout=holdout, dual=arm.dual,
        supervision="self" if withhold else "dense",
        u_kind=cfg["physics"].get("u_kind", "geo"),
        geo_mode=cfg["physics"].get("geo_mode", "per_axis"),
    )
    ds = VolumeWindowDataset(cache, sample_ids, item_cfg,
                             patch_hw=tuple(cfg["data"]["patch_hw"]),
                             context_depth=context,
                             z_stride=int(cfg["eval"]["z_stride"]))
    loader = DataLoader(ds, batch_size=max(1, int(cfg["train"]["batch"]) // 2),
                        num_workers=int(cfg["train"]["num_workers"]))

    model = None
    if arm.needs_training:
        if ckpt is None:
            raise ValueError(f"arm {arm.name} needs a checkpoint")
        model = build_model(arm, context, base=int(cfg["model"]["base"]),
                            window=int(cfg["model"]["window"]),
                            depths=tuple(cfg["model"]["depths"]),
                            heads=tuple(cfg["model"]["heads"])).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()

    projection = str(cfg.get("eval", {}).get("consistency_projection", "none"))
    if projection not in ("none", "hard"):
        raise ValueError("eval.consistency_projection must be 'none' or 'hard', "
                         f"got {projection!r}")

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            if model is not None:
                x = batch["inputs"].to(device)
                pred = select_centre(model(x).float(), batch["center_index"])
                pred = pred.cpu()
            else:
                # Use the same query-centred prior as inference. This also
                # works at Z borders and is independent of input feature order.
                pred = batch["warmstart_center"]

            target = batch["target"]
            acq = batch["acq_mask"].bool()
            hold = batch["hold_mask"].bool()

            # Optional hard consistency step, applied identically to every
            # trained arm: whatever the model was given back, it must agree with
            # it at those positions. Parameter-free, so it cannot flatter one
            # arm over another -- and without it a self-supervised model can
            # ignore its own input entirely (see SPUR-E003b).
            if projection == "hard" and model is not None:
                given = batch["in_mask"].bool()
                pred = torch.where(given, batch["y_in"], pred)

            for i in range(pred.shape[0]):
                p, t = pred[i], target[i]
                m_acq, m_hold = acq[i], hold[i]
                row = {
                    "sample_id": batch["sample_id"][i],
                    "query_z": int(batch["query_z"][i]),
                    "psnr": float(psnr(p, t)),
                    "ssim": float(ssim(p[None, None], t[None, None])),
                    "mae": float(mae(p, t)),
                    # A window with no missing voxel is a window the arm was
                    # handed in full: its error is ~0, so on a plane-skipping
                    # protocol a quarter of the test windows are copies scoring
                    # 120 dB and the headline PSNR is an average of copies and
                    # reconstructions. Counted, reported, and marked in the
                    # table -- the number to read there is PSNRmiss.
                    "fully_measured": bool(bool(m_acq.all())),
                }
                e_meas, e_miss = split_errors(p, t, m_acq)
                # A window with nothing on one side of the split leaves that key
                # absent rather than NaN, so the summary averages the windows
                # that actually have that population and reports how many did.
                if np.isfinite(e_meas):
                    row["e_measured"] = e_meas
                if np.isfinite(e_miss):
                    row["e_missing"] = e_miss
                # Same split, in dB. The headline PSNR averages a population the
                # arm was given with one it had to infer; a zero-parameter arm
                # that copies its input then looks competitive. These two make
                # that mixture visible without changing the protocol.
                ps_m, ps_x = split_psnr(p, t, m_acq)
                if np.isfinite(ps_m):
                    row["psnr_measured"] = ps_m
                if np.isfinite(ps_x):
                    row["psnr_missing"] = ps_x
                if bool(m_hold.any()):
                    row["e_hold"] = float(((p - t).abs() * m_hold).sum()
                                          / m_hold.sum())
                # One segmentation per plane, shared by the overlap numbers and
                # the skeleton: the response is the plane's Sato vesselness and
                # the threshold comes from the reference only, so no arm can
                # tune the operating point and the two tables agree by
                # construction.
                seg = segment_pair(p.numpy(), t.numpy(), method="vesselness",
                                   threshold_mode="global", match_area=True)
                row.update(mask_vessel_metrics(seg["pred_mask"], seg["ref_mask"]))
                row["vessel_threshold"] = seg["threshold"]
                row["vessel_method"] = seg["segment_method"]
                # The same prediction at an equal-area operating point. Kept in
                # its own columns because it answers a different question: the
                # reference-fitted numbers above include "did the prediction's
                # vessel response keep the reference's scale", these do not.
                area = mask_vessel_metrics(seg["pred_mask_area"], seg["ref_mask"])
                row["dice_area"] = area["dice"]
                row["cl_dice_area"] = area["cl_dice"]
                # Structural verdict (opt-in): Dice answers "how much vessel",
                # the skeleton answers "is it still a network".
                if skeleton:
                    row.update(_skeleton_row(seg["pred_mask"], seg["ref_mask"]))
                    row.update(_skeleton_row(seg["pred_mask_area"], seg["ref_mask"],
                                             stats=False, suffix="_area"))
                row["key"] = f"{row['sample_id']}_z{row['query_z']}"
                rows.append(row)

    for r in rows:
        r["group"] = str(r["sample_id"]).split("_FOV")[0]

    summary = {
        "arm": arm.name,
        "split": split_name,
        "fold": int(cfg["data"]["fold"]),
        "checkpoint": str(ckpt) if ckpt else None,
        "sampling": sampling.to_dict(),
        "context_depth": context,
        "physics": {"psf": psf.to_dict(), "lam": cfg["physics"]["lam"],
                    "u_kind": cfg["physics"].get("u_kind", "geo")},
        "supervision": arm.supervision,
        "input_support": input_support,
        "consistency_projection": projection,
        "n_windows": len(rows),
        "n_windows_fully_measured": int(sum(bool(r.get("fully_measured"))
                                            for r in rows)),
        "metrics": {},
    }
    for metric in ("psnr", "psnr_measured", "psnr_missing", "ssim", "mae",
                   "e_measured", "e_missing", "e_hold", "dice", "cl_dice",
                   "dice_area", "cl_dice_area",
                   "pred_fraction", "ref_fraction", "vessel_threshold",
                   "precision", "recall", "f1",
                   "precision_area", "recall_area", "f1_area",
                   "skel_pred_n_branches", "skel_ref_n_branches",
                   "skel_pred_n_junctions", "skel_ref_n_junctions",
                   "skel_pred_n_endpoints", "skel_ref_n_endpoints",
                   "skel_pred_total_length_vox", "skel_ref_total_length_vox",
                   "skel_pred_mean_radius_vox", "skel_ref_mean_radius_vox"):
        if any(metric in r for r in rows):
            summary["metrics"][metric] = summarise_by_group(rows, metric)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, out / "samples.csv")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"summary": summary, "rows": rows}


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write the per-window rows, taking the columns from the whole set.

    The column set cannot come from the first row: a metric that is undefined
    for one window is *absent*, not NaN (see the split-error note above), and
    under plane skipping the first window is usually a fully measured one with
    no ``psnr_missing``. Taking fieldnames from row 0 then dropped the column,
    and DictWriter rejected the later rows that did have it -- which is how the
    z-family re-score failed halfway through.
    """
    import csv
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--base", type=Path, default=DEFAULTS)
    p.add_argument("--arm", required=True)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skeleton", action="store_true",
                   help="add skeleton-based structural scores per window "
                        "(slower: one thinning per window)")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    from spur_mpm.common.config import deep_merge
    cfg = load_config(args.base)
    if args.config:
        cfg = deep_merge(cfg, load_config(args.config))
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    cfg["run"]["seed"] = args.seed

    out = args.out or Path(cfg["run"].get("out_root", "runs")) / \
        f"eval__{args.arm}__{args.split}"
    result = evaluate(cfg, args.arm, args.ckpt, split_name=args.split,
                      out_dir=out, seed=args.seed, skeleton=args.skeleton)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk != "group_means"})
                      for k, v in result["summary"]["metrics"].items()},
                     indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

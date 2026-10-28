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

import torch
from torch.utils.data import DataLoader

from spur_mpm.common.config import apply_overrides, load_config
from spur_mpm.common.tensors import select_centre
from spur_mpm.data.datasets import ItemConfig, VolumeCache, VolumeWindowDataset
from spur_mpm.data.split import read_split
from spur_mpm.metrics.image_metrics import mae, psnr, split_errors, ssim
from spur_mpm.metrics.statistics import summarise_by_group
from spur_mpm.metrics.vessel import vessel_metrics
from spur_mpm.models.registry import build_model, get_arm
from spur_mpm.physics.acquisition import NoiseConfig
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.ssl.holdout import HoldoutSpec

__all__ = ["evaluate", "main"]

DEFAULTS = Path(__file__).parent / "configs" / "base.yaml"

#: Features needed to reconstruct a baseline prediction from the same items.
_BASELINE_FEATURES = ("y", "mask", "x0")


def _eval_features(arm) -> tuple[str, ...]:
    if arm.needs_training:
        return tuple(arm.features)
    return _BASELINE_FEATURES


def _feature_slice(features: tuple[str, ...], want: str, context: int) -> slice:
    i = features.index(want)
    return slice(i * context, (i + 1) * context)


def withholds_at_eval(supervision: str, input_support: str) -> bool:
    """Should the evaluation item keep part of M_acq back?

    Only for a self-supervised arm scored under the training-time regime. Any
    other combination gets everything it measured: withholding from a
    supervised arm would silently change *its* input budget too.
    """
    if input_support not in ("full", "holdout"):
        raise ValueError(f"eval.input_support must be 'full' or 'holdout', got "
                         f"{input_support!r}")
    return supervision == "self" and input_support == "holdout"


@torch.no_grad()
def evaluate(cfg: dict, arm_name: str, ckpt: str | Path | None,
             split_name: str = "test", out_dir: Path | None = None,
             seed: int = 0) -> dict:
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
    features = _eval_features(arm)
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
    withhold = withholds_at_eval(arm.supervision, input_support)
    holdout = HoldoutSpec(ratio=float(cfg["selfsupervision"]["ratio"]),
                          geometry=str(cfg["selfsupervision"]["geometry"]),
                          seed=seed) if withhold else None

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
        features=features, holdout=holdout,
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
                x0 = batch["inputs"][:, _feature_slice(features, "x0", context)]
                pred = x0[:, context // 2]

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
                }
                e_meas, e_miss = split_errors(p, t, m_acq)
                row["e_measured"] = e_meas
                row["e_missing"] = e_miss
                if bool(m_hold.any()):
                    row["e_hold"] = float(((p - t).abs() * m_hold).sum()
                                          / m_hold.sum())
                row.update(vessel_metrics(p.numpy(), t.numpy()))
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
        "metrics": {},
    }
    for metric in ("psnr", "ssim", "mae", "e_measured", "e_missing", "e_hold",
                   "dice", "cl_dice"):
        if any(metric in r for r in rows):
            summary["metrics"][metric] = summarise_by_group(rows, metric)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, out / "samples.csv")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"summary": summary, "rows": rows}


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)


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
                      out_dir=out, seed=args.seed)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk != "group_means"})
                      for k, v in result["summary"]["metrics"].items()},
                     indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

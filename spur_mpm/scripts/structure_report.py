"""Structural verdict per test volume, with masks exported for ImageJ.

    python -m spur_mpm.scripts.structure_report \\
        --arm S1 --ckpt runs/E021_ladder/E021_S1_y_k2/best.pt \\
        --out runs/E021_ladder/E021_S1_y_k2/structure \\
        --set data.index=data_local/dryad_cache/index_cohort.json \\
        --set data.split=spur_mpm/experiments/E020_cohort/split.json \\
        --set sampling.mode=y --set sampling.ky=2

Why this exists next to ``evaluate.py``: ``evaluate`` scores each *plane* and is
fast enough to be the headline table, but the claim being made about
microvasculature is about a *network of vessels through the stack*, and three
things only appear at full-volume scale:

* whether thin structures stay connected across neighbouring planes,
* whether a model's errors are uniform with depth or concentrated in a band,
* the same numbers an anatomist reads off after ``Skeletonize`` in ImageJ.

So this script reconstructs the whole test volume through ``inference``, then
reports :func:`spur_mpm.metrics.skeleton.analyse_volume` -- Dice, clDice,
skeleton precision/recall/F1 within a voxel, branch/junction/endpoint counts,
total length, mean radius, and the same bundle per depth slab.

It also writes both masks as 8-bit multipage TIFF (``*_pred_mask.tif`` and
``*_ref_mask.tif``) so the comparison can be repeated by hand in ImageJ. Both
masks come from the *same* binarisation rule (Otsu fitted on the reference
only), which is the only way the two images are the same object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spur_mpm.common.config import apply_overrides, deep_merge, load_config
from spur_mpm.data.datasets import VolumeCache
from spur_mpm.data.split import read_split
from spur_mpm.inference import reconstruct_volume
from spur_mpm.metrics.skeleton import analyse_volume, export_imagej_mask
from spur_mpm.metrics.vessel import vessel_mask

__all__ = ["structure_for_sample", "main"]

DEFAULTS = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def structure_for_sample(cfg: dict, arm: str, ckpt: Path | None, sample_id: str,
                         out_dir: Path, *, n_slabs: int = 6,
                         device: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    recon = out_dir / f"{arm}_{sample_id}_recon.npy"
    meta = reconstruct_volume(cfg, arm, ckpt, sample_id, recon, device=device)
    pred = np.load(recon)

    cache = VolumeCache.load(cfg["data"]["index"],
                             fold=int(cfg["data"].get("fold", 0)))
    calib = cache.calibration_for(int(cfg["data"].get("fold", 0)))
    ref = calib.apply(cache.array(sample_id)).astype(np.float32)
    if ref.shape != pred.shape:
        raise RuntimeError(f"{sample_id}: reconstruction {pred.shape} does not "
                           f"match the reference {ref.shape}")
    geom = cache.geometry(sample_id)

    result = analyse_volume(pred, ref, voxel_um=(geom.z_step_um, geom.voxel_xy_um,
                                                 geom.voxel_xy_um),
                            n_slabs=n_slabs)
    # Export the very masks the numbers came from -- not a re-thresholded pair.
    threshold = result["threshold"]
    for name, img in (("pred", pred), ("ref", ref)):
        mask = vessel_mask(img, threshold=threshold)
        path = out_dir / f"{arm}_{sample_id}_{name}_mask.tif"
        export_imagej_mask(mask, path)
        result[f"{name}_mask_path"] = str(path)
    result.update({"sample_id": sample_id, "arm": arm, "ckpt": str(ckpt),
                   "sampling": cfg["sampling"], "recon": str(recon),
                   "recon_meta": meta})
    return result


def _row(label: str, r: dict) -> str:
    return (f"| {label} | {r['dice']:.4f} | {r['cl_dice']:.4f} | "
            f"{r['f1']:.4f} | {r['ref_n_components']} → {r['pred_n_components']} | "
            f"{r['ref_n_junctions']} → {r['pred_n_junctions']} | "
            f"{r['ref_total_length_um']:.0f} → {r['pred_total_length_um']:.0f} | "
            f"{r['ref_mean_radius_um']:.3f} → {r['pred_mean_radius_um']:.3f} |")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--base", type=Path, default=DEFAULTS)
    p.add_argument("--arm", required=True)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--samples", nargs="+", default=None,
                   help="default: the test split of the configured fold")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-slabs", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    cfg = load_config(args.base)
    if args.config:
        cfg = deep_merge(cfg, load_config(args.config))
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    samples = args.samples
    if not samples:
        folds = {int(f["fold"]): f
                 for f in read_split(cfg["data"]["split"])["folds"]}
        samples = folds[int(cfg["data"].get("fold", 0))]["test"]

    results = []
    for sid in samples:
        r = structure_for_sample(cfg, args.arm, args.ckpt, sid, args.out,
                                 n_slabs=args.n_slabs, device=args.device)
        results.append(r)
        print(f"{sid}: dice={r['dice']:.4f} clDice={r['cl_dice']:.4f} "
              f"skelF1={r['f1']:.4f} "
              f"components {r['ref_n_components']}->{r['pred_n_components']}",
              flush=True)

    summary = {
        "arm": args.arm,
        "checkpoint": str(args.ckpt),
        "levels": str(cfg["sampling"].get("mode")),
        "ky": cfg["sampling"].get("ky"),
        "kz": cfg["sampling"].get("kz"),
        "n_volumes": len(results),
        "volumes": results,
        "means": {k: float(np.mean([r[k] for r in results]))
                  for k in ("dice", "cl_dice", "f1", "precision", "recall",
                            "pred_fraction", "ref_fraction",
                            "pred_n_components", "ref_n_components",
                            "pred_n_junctions", "ref_n_junctions",
                            "pred_total_length_um", "ref_total_length_um",
                            "pred_mean_radius_um", "ref_mean_radius_um")},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "structure_summary.json").write_text(json.dumps(summary, indent=2))

    md = ["| volume | Dice | clDice | skelF1 | components ref→pred | "
          "junctions ref→pred | length µm ref→pred | radius µm ref→pred |",
          "|---|---|---|---|---|---|---|---|"]
    md += [_row(r["sample_id"], r) for r in results]
    m = summary["means"]
    md.append(_row("**mean**", {**{k: v for k, v in m.items()},
                                "ref_n_components": m["ref_n_components"],
                                "pred_n_components": m["pred_n_components"]}))
    md.append("")
    md.append(f"Binarisation: Otsu on the reference only, hysteresis 0.6 "
              f"(threshold {results[0]['threshold']:.6g}). "
              f"Skeleton overlap tolerance 1 voxel.")
    text = "\n".join(md)
    (args.out / "structure_summary.md").write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

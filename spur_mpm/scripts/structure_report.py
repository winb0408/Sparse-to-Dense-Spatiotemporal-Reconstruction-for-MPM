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

__all__ = ["structure_for_sample", "main"]

DEFAULTS = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def structure_for_sample(cfg: dict, arm: str, ckpt: Path | None, sample_id: str,
                         out_dir: Path, *, n_slabs: int = 6,
                         device: str | None = None,
                         keep_recon: bool = False,
                         segment: str = "vesselness",
                         max_planes: int | None = None) -> dict:
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
    if max_planes:
        # Vesselness costs ~2 s per plane at 5 sigma-scale, so a full 203-plane
        # volume is minutes per image. ``--max-planes`` keeps the report cheap
        # while a setup is being debugged; the headline table uses full volumes.
        pred, ref = pred[:max_planes], ref[:max_planes]

    result = analyse_volume(pred, ref, voxel_um=(geom.z_step_um, geom.voxel_xy_um,
                                                 geom.voxel_xy_um),
                            n_slabs=n_slabs, segment=segment,
                            return_masks=True)
    # Export the very masks the numbers came from, not a re-thresholded pair.
    for name in ("pred", "ref"):
        mask = result.pop(f"{name}_mask")
        path = out_dir / f"{arm}_{sample_id}_{name}_mask.tif"
        export_imagej_mask(mask, path)
        result[f"{name}_mask_path"] = str(path)
    result.update({"sample_id": sample_id, "arm": arm, "ckpt": str(ckpt),
                   "sampling": cfg["sampling"], "recon": str(recon),
                   "n_planes": int(pred.shape[0]), "recon_meta": meta})
    if not keep_recon:
        # A 200-plane float32 stack plus its warm start is ~450 MB per volume per
        # arm; keeping those for two dozen cells would fill the disk to hold data
        # that the TIFF masks and the JSON already summarise. Rerun with
        # --keep-recon when a figure needs the raw volume.
        for path in (recon, Path(str(recon).replace(".npy", ".mask.npy")),
                     Path(str(recon).replace(".npy", ".warmstart.npy"))):
            Path(path).unlink(missing_ok=True)
        result["recon"] = None
    return result


def _row(label: str, r: dict) -> str:
    return (f"| {label} | {r['dice']:.4f} | {r['cl_dice']:.4f} | "
            f"{r['f1']:.4f} | {r['ref_n_components']} → {r['pred_n_components']} | "
            f"{r['ref_n_junctions']} → {r['pred_n_junctions']} | "
            f"{r['ref_total_length_um']:.0f} → {r['pred_total_length_um']:.0f} | "
            f"{r['ref_mean_radius_um']:.3f} → {r['pred_mean_radius_um']:.3f} |")


def _n_planes(r: dict) -> int | None:
    """Planes in the volume: from the record, else from the exported mask.

    Needed because the gated share is a *fraction*; "57 gated planes" says
    nothing until it is set against 203, and a reader who assumes those were
    plane indices would be badly wrong.
    """
    if r.get("n_planes"):
        return int(r["n_planes"])
    path = r.get("pred_mask_path")
    if not path:
        return None
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            return int(len(tf.pages))
    except Exception:                                   # pragma: no cover
        return None


def render_summary(summary: dict) -> str:
    """The markdown table, including the share of planes that survived gating.

    Depth-stratified numbers only mean something if the reader knows how much of
    the stack was scored: on this cohort 28-50 % of the planes are gated out, so
    the Dice below is a Dice on the scored part, not on the volume.
    """
    results = summary["volumes"]
    md = ["| volume | Dice | clDice | skelF1 | components ref→pred | "
          "junctions ref→pred | length µm ref→pred | radius µm ref→pred | "
          "planes scored |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        n, gated = _n_planes(r), int(r.get("n_planes_gated") or 0)
        share = (f"{n - gated}/{n} ({100 * (n - gated) / n:.0f} %)"
                 if n else "–")
        md.append(_row(r["sample_id"], r).rstrip().rstrip("|").rstrip() +
                  f" | {share} |")
    m = summary["means"]
    md.append(_row("**mean**", {**{k: v for k, v in m.items()},
                                "ref_n_components": m["ref_n_components"],
                                "pred_n_components": m["pred_n_components"]}
                  ).rstrip().rstrip("|").rstrip() + " | – |")
    md.append("")
    # The count of gated planes is a *count*: quoting it alone invites the reader
    # to take it for a plane index, and quoting it without the denominator hides
    # that on this cohort a third of the stack is not scored at all.
    gated = [f"{r['sample_id']} {int(r.get('n_planes_gated') or 0)}/{_n_planes(r) or '?'}"
             for r in results]
    md.append(f"**Planes gated out** (reference mask fills > "
              f"{results[0].get('max_plane_fraction')} of the plane, so both "
              f"masks are emptied): {', '.join(gated)}. Every number above is a "
              "score on the planes that survived, not on the whole volume; "
              "report the share beside the table and spot-check the gated planes "
              "in ImageJ.")
    md.append(f"Binarisation: {results[0].get('segment_method', 'vesselness')} "
              f"response, threshold {results[0]['threshold']:.6g} fitted on the "
              f"reference only ({results[0].get('threshold_mode', 'per_plane')}), "
              f"foreground ref {results[0]['ref_fraction']:.1%} / "
              f"pred {results[0]['pred_fraction']:.1%}. "
              f"Skeleton overlap tolerance 1 voxel.")
    return "\n".join(md)


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
    p.add_argument("--keep-recon", action="store_true",
                   help="keep the ~450 MB per-volume .npy dumps (default: delete "
                        "them, the TIFF masks and JSON already carry the result)")
    p.add_argument("--segment", choices=("vesselness", "intensity"),
                   default="vesselness",
                   help="what gets thresholded (default: Sato vesselness, which "
                        "marks ~4%% of a real volume vs ~37%% for raw intensity)")
    p.add_argument("--max-planes", type=int, default=None,
                   help="analyse only the first N planes (debug runs)")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--from-summary", type=Path, default=None,
                   help="re-render an existing structure_summary.json (no GPU, "
                        "no inference) instead of analysing volumes again")
    args = p.parse_args(argv)

    if args.from_summary:
        summary = json.loads(args.from_summary.read_text())
        text = render_summary(summary)
        (args.from_summary.parent / "structure_summary.md").write_text(text + "\n")
        print(text)
        return 0

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
                                 n_slabs=args.n_slabs, device=args.device,
                                 keep_recon=args.keep_recon,
                                 segment=args.segment,
                                 max_planes=args.max_planes)
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

    text = render_summary(summary)
    (args.out / "structure_summary.md").write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

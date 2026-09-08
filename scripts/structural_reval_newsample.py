#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.infer_new_sparse_sample_mdc25 import parse_case, map_mode_to_model, normalize_volume
from scripts.eval_pollen_structural import (
    structural_metrics,
    make_foreground_mask,
    load_model_adaptive,
    mean_std,
)
from scripts.plot_pollen_comparison import predict_frame, parse_run_config
from scripts.summarize_newsample_predictions import enhance_detail_for_pseudocolor, pick_points


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def discover_cases(sample_root: Path) -> List[Dict]:
    dense_path = sample_root / "200um-f10-z5-dense" / "200um-f10-z5-dense.tif"
    dense = normalize_volume(tifffile.imread(str(dense_path)).astype(np.float32))
    cases = []
    for d in sorted(sample_root.iterdir()):
        if not d.is_dir() or "dense" in d.name:
            continue
        if not d.name.endswith(("X", "Y", "XY", "Z", "XYZ")):
            continue
        sparse_path = d / f"{d.name}.tif"
        mask_path = d / f"{d.name}_mask.tif"
        if not sparse_path.exists() or not mask_path.exists():
            continue
        k, raw_mode = parse_case(d.name)
        cases.append(
            {
                "case": d.name,
                "case_dir": d,
                "k": k,
                "raw_mode": raw_mode,
                "mode": map_mode_to_model(raw_mode),
                "sparse_path": sparse_path,
                "mask_path": mask_path,
                "dense": dense,
            }
        )
    return cases


def build_candidates(cases: List[Dict], runs_dir: Path, run_prefix: str) -> List[Dict]:
    run_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(run_prefix) and (d / "best.pth").exists()
    )
    candidates = []
    for c in cases:
        for rd in run_dirs:
            cfg = parse_run_config(rd.name)
            try:
                train_k = int(cfg.get("K", -1))
            except ValueError:
                continue
            if train_k != c["k"]:
                continue
            candidates.append(
                {
                    "case": c["case"],
                    "case_mode": c["mode"],
                    "case_k": c["k"],
                    "run_name": rd.name,
                    "algorithm": cfg.get("algorithm", "?"),
                    "train_sparse_mode": cfg.get("sparse_mode", "?"),
                    "extras": cfg.get("extras", ""),
                }
            )
    return candidates


def make_best_fixedz_figure(case: str, sparse: np.ndarray, gt: np.ndarray, pred: np.ndarray, z: int, out_path: Path, best_run: str) -> None:
    names = ["Sparse", "GT", f"Best({best_run})"]
    imgs = [sparse[z], gt[z], pred[z]]
    gray = [enhance_detail_for_pseudocolor(x) for x in imgs]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for i, (name, gimg) in enumerate(zip(names, gray)):
        axes[0, i].imshow(gimg, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(name)
        axes[0, i].axis("off")
        axes[1, i].imshow(gimg, cmap="turbo", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Detail gray")
    axes[1, 0].set_ylabel("Pseudo color")
    fig.suptitle(f"{case} | Fixed Z={z} | Structural Best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_best_pointcloud_figure(case: str, sparse: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: Path, best_run: str, threshold: float, max_points: int) -> None:
    vols = {"Sparse": sparse, "GT": gt, f"Best({best_run})": pred}
    fig = plt.figure(figsize=(12, 4.6), constrained_layout=True)
    for idx, (name, vol) in enumerate(vols.items(), start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        enh = enhance_detail_for_pseudocolor(vol)
        coords, colors = pick_points(enh, threshold=threshold, max_points=max_points, seed=42 + idx)
        if coords.size > 0:
            ax.scatter(
                coords[:, 2], coords[:, 1], coords[:, 0],
                c=colors, cmap="turbo", s=1.0, alpha=0.35, linewidths=0
            )
        ax.set_title(f"{name}\nN={coords.shape[0]}")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((vol.shape[2], vol.shape[1], vol.shape[0]))
        ax.view_init(elev=26, azim=42)
    fig.suptitle(f"{case} | 3D Point Cloud | Structural Best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Structure-first reevaluation on new 200um sample.")
    p.add_argument("--sample_root", type=Path, default=Path("/data2/wjb/threePM/ThreePM_openDatasets/200um-f10-z5_sparse"))
    p.add_argument("--runs_dir", type=Path, default=Path("/data2/wjb/threePM/runs"))
    p.add_argument("--output_dir", type=Path, default=Path("/data2/wjb/threePM/eval_newsample_mdc25_preds/structural_reval"))
    p.add_argument("--run_prefix", type=str, default="sparse3d_")
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--patch_hw", type=int, default=256)
    p.add_argument("--fg_percentile", type=float, default=70.0)
    p.add_argument("--fg_dilation", type=int, default=3)
    p.add_argument("--ssim_threshold", type=float, default=0.85)
    p.add_argument("--pc_threshold", type=float, default=0.70)
    p.add_argument("--pc_max_points", type=int, default=90000)
    p.add_argument("--reuse_metrics_csv", type=Path, default=None, help="Reuse existing structural_metrics_all.csv")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_cache_root = args.output_dir / "pred_cache"
    pred_cache_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = discover_cases(args.sample_root)
    candidates = build_candidates(cases, args.runs_dir, args.run_prefix)
    write_csv(
        args.output_dir / "candidates_by_case.csv",
        candidates,
        ["case", "case_mode", "case_k", "run_name", "algorithm", "train_sparse_mode", "extras"],
    )

    failures: List[Dict] = []
    frame_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    by_case: Dict[str, List[Dict]] = {}
    for c in candidates:
        by_case.setdefault(c["case"], []).append(c)

    if args.reuse_metrics_csv is None:
        for c in cases:
            case = c["case"]
            low_vol = normalize_volume(tifffile.imread(str(c["sparse_path"])).astype(np.float32))
            mask_vol = (tifffile.imread(str(c["mask_path"])) > 0).astype(np.float32)
            gt_vol = c["dense"]
            n = min(low_vol.shape[0], mask_vol.shape[0], gt_vol.shape[0], args.max_frames)
            low_vol = low_vol[:n]
            mask_vol = mask_vol[:n]
            gt_vol = gt_vol[:n]
            print(f"\n=== Case {case} | K={c['k']} | mode={c['mode']} | frames={n} ===")

            for cand in by_case.get(case, []):
                run_name = cand["run_name"]
                run_dir = args.runs_dir / run_name
                try:
                    model, opts = load_model_adaptive(run_dir, device)
                except Exception as exc:
                    failures.append({"case": case, "run_name": run_name, "stage": "load_model", "error": str(exc)})
                    print(f"  [Skip] {run_name}: load error")
                    continue

                per_frame: List[Dict] = []
                pred_stack: List[np.ndarray] = []
                for i in range(n):
                    try:
                        pred = predict_frame(model, opts, cand["algorithm"], low_vol, gt_vol, mask_vol, i, args.patch_hw, device)
                    except Exception as exc:
                        failures.append({"case": case, "run_name": run_name, "stage": f"infer_frame_{i}", "error": str(exc)})
                        per_frame = []
                        pred_stack = []
                        print(f"  [Skip] {run_name}: inference failed")
                        break

                    gt_n = ((gt_vol[i] - gt_vol[i].min()) / (gt_vol[i].max() - gt_vol[i].min() + 1e-8)).astype(np.float32)
                    fg = make_foreground_mask(gt_n, percentile=args.fg_percentile, dilation_iter=args.fg_dilation)
                    m = structural_metrics(pred, gt_n, fg)
                    row = {
                        "case": case,
                        "run_name": run_name,
                        "algorithm": cand["algorithm"],
                        "train_sparse_mode": cand["train_sparse_mode"],
                        "case_k": c["k"],
                        "train_mode": cand["train_sparse_mode"],
                        "frame": i,
                        **m,
                    }
                    frame_rows.append(row)
                    per_frame.append(row)
                    pred_stack.append(pred.astype(np.float32))

                if per_frame:
                    cache_dir = pred_cache_root / case
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(
                        str(cache_dir / f"{run_name}_pred.tif"),
                        (np.stack(pred_stack, axis=0).clip(0, 1) * 65535).astype(np.uint16),
                        imagej=True,
                    )
                    print(f"  {run_name}: cache ready ({len(per_frame)} frames)")

                del model
                torch.cuda.empty_cache()
    else:
        with open(args.reuse_metrics_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                casted = dict(row)
                for k in [
                    "case_k", "frame", "psnr_full", "ssim_full", "psnr_fg", "mae_fg", "ssim_fg", "ms_ssim_fg",
                    "ncc_fg", "psnr_fg_cal", "mae_fg_cal", "ssim_fg_cal", "ms_ssim_fg_cal", "composite_score", "fg_fraction",
                ]:
                    if k in casted and casted[k] != "":
                        casted[k] = float(casted[k]) if k not in ("case_k", "frame") else int(float(casted[k]))
                frame_rows.append(casted)

    # Save step outputs
    write_csv(
        args.output_dir / "inference_failures.csv",
        failures,
        ["case", "run_name", "stage", "error"],
    )
    if args.reuse_metrics_csv is None:
        if frame_rows:
            frame_cols = list(frame_rows[0].keys())
        else:
            frame_cols = ["case", "run_name", "algorithm", "train_sparse_mode", "case_k", "train_mode", "frame"]
        write_csv(args.output_dir / "structural_metrics_all.csv", frame_rows, frame_cols)

    # Build summary rows from frame-level metrics
    case_meta = {c["case"]: c for c in cases}
    cand_meta = {(x["case"], x["run_name"]): x for x in candidates}
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for r in frame_rows:
        grouped.setdefault((r["case"], r["run_name"]), []).append(r)
    for (case, run_name), per_frame in grouped.items():
        cmeta = case_meta.get(case, {})
        mmeta = cand_meta.get((case, run_name), {})
        summary = {
            "case": case,
            "case_mode": cmeta.get("mode", ""),
            "K": cmeta.get("k", mmeta.get("case_k", "")),
            "run_name": run_name,
            "algorithm": mmeta.get("algorithm", per_frame[0].get("algorithm", "")),
            "train_sparse_mode": mmeta.get("train_sparse_mode", per_frame[0].get("train_sparse_mode", "")),
            "extras": mmeta.get("extras", ""),
            "n_frames": len(per_frame),
        }
        metric_keys = [
            "fg_fraction", "psnr_full", "ssim_full", "psnr_fg", "ssim_fg",
            "ms_ssim_fg", "ncc_fg", "psnr_fg_cal", "ssim_fg_cal",
            "ms_ssim_fg_cal", "mae_fg_cal", "composite_score",
        ]
        for key in metric_keys:
            mean, std = mean_std(per_frame, key)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        summary["passes_ssim85"] = bool(summary["ssim_fg_cal_mean"] >= args.ssim_threshold)
        summary_rows.append(summary)

    # Select best
    best_rows: List[Dict] = []
    by_case_summary: Dict[str, List[Dict]] = {}
    for r in summary_rows:
        by_case_summary.setdefault(r["case"], []).append(r)
    for case, rows in by_case_summary.items():
        qualified = [r for r in rows if r["passes_ssim85"]]
        source = qualified if qualified else rows
        source_sorted = sorted(
            source,
            key=lambda x: (-float(x["composite_score_mean"]), -float(x["psnr_fg_cal_mean"]), -float(x["ssim_fg_cal_mean"]))
        )
        top = dict(source_sorted[0])
        top["selection_status"] = "qualified" if qualified else "no_model_meets_ssim85"
        best_rows.append(top)

    best_cols = [
        "case", "case_mode", "K", "run_name", "algorithm", "train_sparse_mode", "extras", "n_frames",
        "ssim_fg_cal_mean", "ms_ssim_fg_cal_mean", "ncc_fg_mean", "psnr_fg_cal_mean", "mae_fg_cal_mean",
        "composite_score_mean", "passes_ssim85", "selection_status",
    ]
    write_csv(args.output_dir / "best_per_case_structural.csv", best_rows, best_cols)

    md_path = args.output_dir / "best_per_case_structural.md"
    with open(md_path, "w") as f:
        f.write("# Best Per Case (Structure-First)\n\n")
        f.write(f"- Threshold: `ssim_fg_cal_mean >= {args.ssim_threshold}`\n\n")
        f.write("| Case | Status | Best Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal |\n")
        f.write("|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in sorted(best_rows, key=lambda x: x["case"]):
            f.write(
                f"| {r['case']} | {r['selection_status']} | {r['run_name']} | {r['algorithm']} | {r['train_sparse_mode']} | "
                f"{float(r['composite_score_mean']):.4f} | {float(r['ssim_fg_cal_mean']):.4f} | "
                f"{float(r['ms_ssim_fg_cal_mean']):.4f} | {float(r['ncc_fg_mean']):.4f} | {float(r['psnr_fg_cal_mean']):.2f} |\n"
            )

    # Render best figures
    fig_dir = args.output_dir / "figures_best_structural"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old_png in fig_dir.glob("*.png"):
        old_png.unlink()
    for c in cases:
        case = c["case"]
        best = next((r for r in best_rows if r["case"] == case), None)
        if best is None:
            continue
        run_name = best["run_name"]
        pred_path = pred_cache_root / case / f"{run_name}_pred.tif"
        if not pred_path.exists():
            continue
        pred = normalize_volume(tifffile.imread(str(pred_path)).astype(np.float32))
        low = normalize_volume(tifffile.imread(str(c["sparse_path"])).astype(np.float32))
        gt = c["dense"][: pred.shape[0]]
        low = low[: pred.shape[0]]
        z = pred.shape[0] // 2
        make_best_fixedz_figure(case, low, gt, pred, z, fig_dir / f"{case}_best_fixedZ_{run_name}.png", run_name)
        make_best_pointcloud_figure(case, low, gt, pred, fig_dir / f"{case}_best_pointcloud_{run_name}.png", run_name, args.pc_threshold, args.pc_max_points)

    print(f"\nDone. Outputs at: {args.output_dir}")


if __name__ == "__main__":
    main()


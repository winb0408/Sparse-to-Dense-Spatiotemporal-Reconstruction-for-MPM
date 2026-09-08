#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.eval_sparse3d_unified import load_model, evaluate_2d, evaluate_2p5d
from scripts.eval_pollen_sparse_models import evaluate_3d


def main():
    p = argparse.ArgumentParser(description="Evaluate MDC25 K2 optimized runs on Validation.")
    p.add_argument("--sparse_root", type=str, required=True)
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--patch_hw", type=int, default=256)
    p.add_argument("--run_filter", type=str, default="_mdc25")
    p.add_argument(
        "--dataset_protocol",
        type=str,
        default="mdc25_3d",
        choices=["mdc25_2d", "mdc25_3d"],
        help="MDC25 subset protocol. mdc25_2d only permits 2D XY runs; mdc25_3d permits XY/Z/XYZ stack evaluation.",
    )
    p.add_argument(
        "--allow_protocol_mismatch",
        action="store_true",
        default=False,
        help="Allow runs outside the selected MDC25 protocol for transfer/sanity checks.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_root = Path(args.sparse_root)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and args.run_filter in d.name and (d / "best.pth").exists()])
    rows = []
    for rd in run_dirs:
        name = rd.name
        mode = "xy" if "_xy_" in name else ("xyz" if "_xyz_" in name else "z")
        algo = "3d" if "_3d_" in name else ("2p5d" if "_2p5d_" in name else "2d")
        if args.dataset_protocol == "mdc25_2d" and not args.allow_protocol_mismatch:
            if algo != "2d" or mode != "xy":
                continue
            val_dir = sparse_root / "2d/K2_xy/Validation"
        else:
            val_dir = sparse_root / f"3d/K2_{mode}/Validation"
        model, opts = load_model(rd, device)
        if algo == "3d":
            stack_gt = sorted((val_dir / "stacks").glob("*_gt.npy"))[0]
            stack_sp = Path(str(stack_gt).replace("_gt.npy", "_sparse.npy"))
            gt = np.load(stack_gt).astype(np.float32)
            low = np.load(stack_sp).astype(np.float32)
            mask = (low > 1e-6).astype(np.float32)
            per = evaluate_3d(model, opts, low, gt, mask, args.max_frames, args.patch_hw, device)
        elif algo == "2p5d":
            per = evaluate_2p5d(model, opts, val_dir, device, args.max_frames, args.patch_hw)
        else:
            per = evaluate_2d(model, opts, val_dir, device, args.max_frames, args.patch_hw)
        ps = [r["psnr"] for r in per]
        ss = [r["ssim"] for r in per]
        rows.append({
            "run_name": name,
            "n_frames": len(per),
            "psnr_mean": float(np.mean(ps)) if ps else 0.0,
            "ssim_mean": float(np.mean(ss)) if ss else 0.0,
        })
        del model
        torch.cuda.empty_cache()

    csv_path = out_dir / "summary_all.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_name", "n_frames", "psnr_mean", "ssim_mean"])
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "summary_all.md"
    with open(md_path, "w") as f:
        f.write("# MDC25 K2 Optimized Evaluation\n\n")
        f.write("| Run | Frames | PSNR | SSIM |\n|---|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda x: -x["psnr_mean"]):
            f.write(f"| {r['run_name']} | {r['n_frames']} | {r['psnr_mean']:.2f} | {r['ssim_mean']:.4f} |\n")

    print(f"Done: {csv_path}")


if __name__ == "__main__":
    main()

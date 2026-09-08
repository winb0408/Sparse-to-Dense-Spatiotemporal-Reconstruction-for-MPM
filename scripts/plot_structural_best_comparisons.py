#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.summarize_newsample_predictions import enhance_detail_for_pseudocolor, pick_points  # noqa: E402


MODE_ORDER = ["X", "Y", "XY", "Z", "XYZ"]


def read_tif_norm(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    else:
        x = arr.astype(np.float32)
        x = (x - float(x.min())) / (float(x.max()) - float(x.min()) + 1e-8)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def parse_case_suffix(case_name: str) -> str:
    tail = case_name.split("-")[-1]
    return tail.split("x", 1)[1]


def load_best_rows(best_csv: Path) -> List[Dict]:
    rows = []
    with open(best_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def sort_case_rows(rows: List[Dict]) -> List[Dict]:
    return sorted(
        rows,
        key=lambda r: MODE_ORDER.index(parse_case_suffix(r["case"])) if parse_case_suffix(r["case"]) in MODE_ORDER else 99,
    )


def make_slice_compare(rows: List[Dict], sample_root: Path, pred_cache_root: Path, out_path: Path, z_index: int | None) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10.5, 2.6 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    for i, r in enumerate(rows):
        case = r["case"]
        run = r["run_name"]
        case_dir = sample_root / case
        sparse = read_tif_norm(case_dir / f"{case}.tif")
        gt = read_tif_norm(sample_root / "200um-f10-z5-dense" / "200um-f10-z5-dense.tif")
        pred = read_tif_norm(pred_cache_root / case / f"{run}_pred.tif")

        n = min(sparse.shape[0], gt.shape[0], pred.shape[0])
        z = n // 2 if z_index is None else int(np.clip(z_index, 0, n - 1))
        s = enhance_detail_for_pseudocolor(sparse[z])
        p = enhance_detail_for_pseudocolor(pred[z])
        g = enhance_detail_for_pseudocolor(gt[z])

        imgs = [s, p, g]
        titles = ["Sparse input", f"Best recon\n{run}", "GT dense"]
        for j in range(3):
            ax = axes[i, j]
            ax.imshow(imgs[j], cmap="magma", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(titles[j], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        mode = parse_case_suffix(case)
        axes[i, 0].set_ylabel(f"{case}\nmode={mode}", fontsize=9)

    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def make_pointcloud_compare(
    rows: List[Dict],
    sample_root: Path,
    pred_cache_root: Path,
    out_path: Path,
    threshold: float,
    max_points: int,
) -> None:
    n_rows = len(rows)
    fig = plt.figure(figsize=(12.5, 3.1 * n_rows), constrained_layout=True)
    for i, r in enumerate(rows):
        case = r["case"]
        run = r["run_name"]
        case_dir = sample_root / case
        sparse = read_tif_norm(case_dir / f"{case}.tif")
        gt = read_tif_norm(sample_root / "200um-f10-z5-dense" / "200um-f10-z5-dense.tif")
        pred = read_tif_norm(pred_cache_root / case / f"{run}_pred.tif")

        n = min(sparse.shape[0], gt.shape[0], pred.shape[0])
        vols = [sparse[:n], pred[:n], gt[:n]]
        names = ["Sparse", "Best", "GT"]

        for j, (name, vol) in enumerate(zip(names, vols), start=1):
            ax = fig.add_subplot(n_rows, 3, i * 3 + j, projection="3d")
            enh = enhance_detail_for_pseudocolor(vol)
            coords, colors = pick_points(enh, threshold=threshold, max_points=max_points, seed=100 + i * 7 + j)
            if coords.size > 0:
                ax.scatter(
                    coords[:, 2], coords[:, 1], coords[:, 0],
                    c=colors, cmap="magma", s=0.8, alpha=0.28, linewidths=0
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect((vol.shape[2], vol.shape[1], vol.shape[0]))
            ax.view_init(elev=26, azim=42)
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 1:
                mode = parse_case_suffix(case)
                ax.text2D(0.02, 0.95, f"{case}\nmode={mode}", transform=ax.transAxes, fontsize=9)

    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate grouped best structural comparison figures (2x vs 4x).")
    p.add_argument("--sample_root", type=Path, default=Path("/data2/wjb/threePM/ThreePM_openDatasets/200um-f10-z5_sparse"))
    p.add_argument("--struct_root", type=Path, default=Path("/data2/wjb/threePM/eval_newsample_mdc25_preds/structural_reval"))
    p.add_argument("--output_dir", type=Path, default=Path("/data2/wjb/threePM/eval_newsample_mdc25_preds/structural_reval/overview_grouped"))
    p.add_argument("--z_index", type=int, default=None)
    p.add_argument("--pc_threshold", type=float, default=0.70)
    p.add_argument("--pc_max_points", type=int, default=90000)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_rows = load_best_rows(args.struct_root / "best_per_case_structural.csv")
    pred_cache_root = args.struct_root / "pred_cache"

    for k in [2, 4]:
        rows_k = [r for r in best_rows if int(r["K"]) == k]
        rows_k = sort_case_rows(rows_k)
        if not rows_k:
            continue
        make_slice_compare(
            rows_k,
            sample_root=args.sample_root,
            pred_cache_root=pred_cache_root,
            out_path=args.output_dir / f"best_slice_compare_K{k}.png",
            z_index=args.z_index,
        )
        make_pointcloud_compare(
            rows_k,
            sample_root=args.sample_root,
            pred_cache_root=pred_cache_root,
            out_path=args.output_dir / f"best_pointcloud_compare_K{k}.png",
            threshold=args.pc_threshold,
            max_points=args.pc_max_points,
        )
        print(f"[Done] K={k}")

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


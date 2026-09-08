#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, uniform_filter


def read_tif_norm(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    arr = arr.astype(np.float32)
    if arr.dtype == np.uint16:
        arr = arr / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr / 255.0
    else:
        vmin = float(arr.min())
        vmax = float(arr.max())
        arr = (arr - vmin) / (vmax - vmin + 1e-8)
    return arr.clip(0.0, 1.0).astype(np.float32)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim_2d(pred: np.ndarray, gt: np.ndarray, win: int = 7) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_p = uniform_filter(pred, size=win)
    mu_g = uniform_filter(gt, size=win)
    sig_pp = uniform_filter(pred * pred, size=win) - mu_p ** 2
    sig_gg = uniform_filter(gt * gt, size=win) - mu_g ** 2
    sig_pg = uniform_filter(pred * gt, size=win) - mu_p * mu_g
    num = (2 * mu_p * mu_g + c1) * (2 * sig_pg + c2)
    den = (mu_p ** 2 + mu_g ** 2 + c1) * (sig_pp + sig_gg + c2)
    return float(np.mean(num / (den + 1e-12)))


def ms_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float32)
    p = pred.copy()
    g = gt.copy()
    val = 1.0
    for i, w in enumerate(weights):
        val *= max(ssim_2d(p, g), 0.0) ** float(w)
        if i < len(weights) - 1:
            p = uniform_filter(p, size=2)[::2, ::2]
            g = uniform_filter(g, size=2)[::2, ::2]
    return float(val)


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def ncc(pred: np.ndarray, gt: np.ndarray) -> float:
    pf = pred.flatten() - float(pred.mean())
    gf = gt.flatten() - float(gt.mean())
    return float(np.dot(pf, gf) / (np.linalg.norm(pf) * np.linalg.norm(gf) + 1e-10))


def enhance_detail_for_pseudocolor(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [0.5, 99.5])
    x = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    # Mild unsharp mask + gamma to highlight cell edges and dim structures.
    blur = gaussian_filter(x, sigma=1.0)
    sharp = np.clip(x + 1.2 * (x - blur), 0.0, 1.0)
    return np.power(sharp, 0.85).astype(np.float32)


def pick_points(vol: np.ndarray, threshold: float, max_points: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    mask = vol >= threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return coords, np.array([], dtype=np.float32)
    if coords.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[idx]
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    c = vol[z, y, x]
    return coords, c


def eval_one_case(case_dir: Path) -> List[Dict]:
    gt = read_tif_norm(case_dir / "gt_dense.tif")
    sparse = read_tif_norm(case_dir / "input_sparse.tif")
    models = {
        "sparse_input": sparse,
        "2d": read_tif_norm(case_dir / "pred_2d_wiener_maskaware.tif"),
        "2p5d": read_tif_norm(case_dir / "pred_2p5d_wiener_maskaware.tif"),
        "3d": read_tif_norm(case_dir / "pred_3d_wiener_maskaware.tif"),
    }
    n = min(*(v.shape[0] for v in [gt, *models.values()]))
    gt = gt[:n]
    rows: List[Dict] = []
    for model_name, pred in models.items():
        pred = pred[:n]
        ps = []
        ss = []
        ms = []
        maes = []
        nccs = []
        for i in range(n):
            ps.append(psnr(pred[i], gt[i]))
            ss.append(ssim_2d(pred[i], gt[i]))
            ms.append(ms_ssim(pred[i], gt[i]))
            maes.append(mae(pred[i], gt[i]))
            nccs.append(ncc(pred[i], gt[i]))
        rows.append({
            "case": case_dir.name,
            "model": model_name,
            "n_frames": n,
            "psnr_mean": float(np.mean(ps)),
            "psnr_std": float(np.std(ps)),
            "ssim_mean": float(np.mean(ss)),
            "ssim_std": float(np.std(ss)),
            "ms_ssim_mean": float(np.mean(ms)),
            "ms_ssim_std": float(np.std(ms)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "ncc_mean": float(np.mean(nccs)),
            "ncc_std": float(np.std(nccs)),
        })
    return rows


def make_fixed_z_figure(case_dir: Path, out_path: Path, z_index: int | None) -> None:
    gt = read_tif_norm(case_dir / "gt_dense.tif")
    sparse = read_tif_norm(case_dir / "input_sparse.tif")
    p2d = read_tif_norm(case_dir / "pred_2d_wiener_maskaware.tif")
    p25 = read_tif_norm(case_dir / "pred_2p5d_wiener_maskaware.tif")
    p3d = read_tif_norm(case_dir / "pred_3d_wiener_maskaware.tif")
    n = min(gt.shape[0], sparse.shape[0], p2d.shape[0], p25.shape[0], p3d.shape[0])
    z = n // 2 if z_index is None else int(np.clip(z_index, 0, n - 1))

    names = ["Sparse", "GT", "2D", "2.5D", "3D"]
    imgs = [sparse[z], gt[z], p2d[z], p25[z], p3d[z]]
    gray_imgs = [enhance_detail_for_pseudocolor(x) for x in imgs]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for i, (name, gimg) in enumerate(zip(names, gray_imgs)):
        axes[0, i].imshow(gimg, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(name)
        axes[0, i].axis("off")

    for i, (name, gimg) in enumerate(zip(names, gray_imgs)):
        axes[1, i].imshow(gimg, cmap="turbo", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Detail-enhanced gray", fontsize=11)
    axes[1, 0].set_ylabel("Pseudo color (turbo)", fontsize=11)
    fig.suptitle(f"{case_dir.name} | Fixed Z={z} Comparison", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_pointcloud_figure(case_dir: Path, out_path: Path, threshold: float, max_points: int) -> None:
    vols = {
        "Sparse": read_tif_norm(case_dir / "input_sparse.tif"),
        "GT": read_tif_norm(case_dir / "gt_dense.tif"),
        "2D": read_tif_norm(case_dir / "pred_2d_wiener_maskaware.tif"),
        "2.5D": read_tif_norm(case_dir / "pred_2p5d_wiener_maskaware.tif"),
        "3D": read_tif_norm(case_dir / "pred_3d_wiener_maskaware.tif"),
    }
    n = min(v.shape[0] for v in vols.values())
    for k in vols:
        vols[k] = vols[k][:n]

    fig = plt.figure(figsize=(20, 4.6), constrained_layout=True)
    for idx, (name, vol) in enumerate(vols.items(), start=1):
        ax = fig.add_subplot(1, 5, idx, projection="3d")
        enh = enhance_detail_for_pseudocolor(vol)
        coords, colors = pick_points(enh, threshold=threshold, max_points=max_points, seed=42 + idx)
        if coords.size > 0:
            ax.scatter(
                coords[:, 2], coords[:, 1], coords[:, 0],
                c=colors, cmap="turbo", s=1.0, alpha=0.35, linewidths=0
            )
        ax.set_title(f"{name}\nN={coords.shape[0]}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((vol.shape[2], vol.shape[1], vol.shape[0]))
        ax.view_init(elev=26, azim=42)
    fig.suptitle(f"{case_dir.name} | 3D Point Cloud Pseudocolor", fontsize=14)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize new-sample predictions and create pseudo-color overview figures.")
    p.add_argument("--pred_root", type=Path, default=Path("/data2/wjb/threePM/eval_newsample_mdc25_preds"))
    p.add_argument("--output_dir", type=Path, default=Path("/data2/wjb/threePM/eval_newsample_mdc25_preds/summary_and_figures"))
    p.add_argument("--z_index", type=int, default=None, help="Fixed z index for slice comparison (default=center).")
    p.add_argument("--pc_threshold", type=float, default=0.70)
    p.add_argument("--pc_max_points", type=int, default=90000)
    p.add_argument("--rank_metric", type=str, default="psnr_mean", choices=["psnr_mean", "ssim_mean", "ms_ssim_mean"])
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted([d for d in args.pred_root.iterdir() if d.is_dir() and (d / "gt_dense.tif").exists()])
    if not case_dirs:
        raise ValueError(f"No case directories found in: {args.pred_root}")

    all_rows: List[Dict] = []
    for case_dir in case_dirs:
        all_rows.extend(eval_one_case(case_dir))
        make_fixed_z_figure(case_dir, args.output_dir / f"{case_dir.name}_fixedZ_pseudocolor.png", args.z_index)
        make_pointcloud_figure(case_dir, args.output_dir / f"{case_dir.name}_pointcloud_pseudocolor.png", args.pc_threshold, args.pc_max_points)
        print(f"[Done] {case_dir.name}")

    summary_csv = args.output_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cols = [
            "case", "model", "n_frames",
            "psnr_mean", "psnr_std",
            "ssim_mean", "ssim_std",
            "ms_ssim_mean", "ms_ssim_std",
            "mae_mean", "mae_std",
            "ncc_mean", "ncc_std",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved summary: {summary_csv}")

    # Build a compact report-friendly markdown:
    # best row per case under selected metric, excluding sparse_input baseline.
    best_rows: Dict[str, Dict] = {}
    for row in all_rows:
        if row["model"] == "sparse_input":
            continue
        case = row["case"]
        score = float(row[args.rank_metric])
        if case not in best_rows or score > float(best_rows[case][args.rank_metric]):
            best_rows[case] = row

    md_path = args.output_dir / "summary_top.md"
    with open(md_path, "w") as f:
        f.write("# New Sample Best Model Summary\n\n")
        f.write(f"- Ranking metric: `{args.rank_metric}`\n")
        f.write(f"- Source summary: `{summary_csv}`\n\n")
        f.write("| Case | Best Model | PSNR | SSIM | MS-SSIM | Fixed-Z Figure | Point Cloud Figure |\n")
        f.write("|---|---|---:|---:|---:|---|---|\n")
        for case in sorted(best_rows.keys()):
            row = best_rows[case]
            fz = f"{case}_fixedZ_pseudocolor.png"
            pc = f"{case}_pointcloud_pseudocolor.png"
            f.write(
                f"| {case} | {row['model']} "
                f"| {float(row['psnr_mean']):.3f} "
                f"| {float(row['ssim_mean']):.4f} "
                f"| {float(row['ms_ssim_mean']):.4f} "
                f"| `{fz}` | `{pc}` |\n"
            )
    print(f"Saved top summary: {md_path}")


if __name__ == "__main__":
    main()


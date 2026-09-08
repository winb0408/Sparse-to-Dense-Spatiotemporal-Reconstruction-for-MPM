#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_pollen_comparison import load_volume, minmax_pair  # noqa: E402


def robust_range(img: np.ndarray) -> tuple[float, float]:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, [0.5, 99.5])
    if vmax <= vmin:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def choose_best(summary_csv: Path) -> Dict[str, Dict[str, Dict]]:
    best: Dict[str, Dict[str, Dict]] = {}
    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["case"]
            algo = row["algorithm"]
            score = float(row["composite_score_mean"])
            best.setdefault(case, {})
            if algo not in best[case] or score > float(best[case][algo]["composite_score_mean"]):
                best[case][algo] = row
    return best


def panel_title(prefix: str, row: Dict | None = None) -> str:
    if row is None:
        return prefix
    return (
        f"{prefix}\n{row['run_name']}\n"
        f"Score={float(row['composite_score_mean']):.3f} "
        f"SSIMfg={float(row['ssim_fg_cal_mean']):.3f}"
    )


def save_panel(
    out_path: Path,
    case: str,
    images: list[np.ndarray],
    titles: list[str],
    cmap: str,
    add_colorbar: bool,
) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 4.5), dpi=220)
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        vmin, vmax = robust_range(img)
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
    fig.suptitle(case, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot structure-ranked pollen qualitative panels.")
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets"),
    )
    p.add_argument(
        "--mask_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets/mask"),
    )
    p.add_argument("--eval_dir", type=Path, default=Path("/data2/wjb/threePM/eval_pollen_sparse_structural"))
    p.add_argument("--frame_idx", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = build_args()
    summary_csv = args.eval_dir / "summary_structural.csv"
    best = choose_best(summary_csv)
    pred_root = args.eval_dir / "predictions"
    dense_path = next(args.data_root.glob("*-dense-*.tif"))
    gt_vol = load_volume(dense_path)

    gray_out = args.eval_dir / "qualitative_structural"
    pseudo_out = args.eval_dir / "qualitative_structural_pseudocolor"
    sparse_paths = sorted([p for p in args.data_root.glob("*.tif") if "dense" not in p.name])

    for sp in sparse_paths:
        case = sp.stem
        if case not in best or not all(k in best[case] for k in ("2d", "2p5d", "3d")):
            print(f"[Skip] missing best algo for {case}")
            continue
        mask_path = args.mask_root / sp.name.replace(".tif", "_mask.tif")
        if not mask_path.exists():
            print(f"[Skip] missing mask for {case}")
            continue
        low_vol = load_volume(sp)
        mask_vol = (load_volume(mask_path) > 0).astype(np.float32)
        n = min(low_vol.shape[0], gt_vol.shape[0], mask_vol.shape[0])
        idx = min(max(args.frame_idx, 0), n - 1)

        low_n, gt_n = minmax_pair(low_vol[idx], gt_vol[idx])
        sparse_n = (low_n * mask_vol[idx]).clip(0, 1)
        images = [gt_n, sparse_n]
        titles = ["GT (dense)", "Sparse input"]

        for algo, label in (("2d", "Best 2D"), ("2p5d", "Best 2.5D"), ("3d", "Best 3D")):
            row = best[case][algo]
            pred_path = pred_root / case / f"{row['run_name']}_pred.npy"
            pred = np.load(pred_path)[idx]
            images.append(pred)
            titles.append(panel_title(label, row))

        save_panel(gray_out / f"{case}_structural_best5.png", case, images, titles, cmap="gray", add_colorbar=True)
        save_panel(
            pseudo_out / f"{case}_structural_best5_pseudocolor.png",
            case,
            images,
            titles,
            cmap="magma",
            add_colorbar=True,
        )
        print(f"[OK] {case}")


if __name__ == "__main__":
    main()

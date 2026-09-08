#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def load01(path: Path) -> np.ndarray:
    x = tifffile.imread(str(path)).astype(np.float32)
    vmin, vmax = float(x.min()), float(x.max())
    return ((x - vmin) / (vmax - vmin + 1e-8)).clip(0, 1)


def best_by_algo(rows, mode: str, algo_key: str):
    cand = [r for r in rows if f"_{mode}_" in r["run_name"] and algo_key in r["run_name"]]
    cand = sorted(cand, key=lambda x: -float(x["psnr_mean"]))
    return cand[0]["run_name"] if cand else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=str, required=True)
    p.add_argument("--sparse_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.summary_csv)))
    sparse_root = Path(args.sparse_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("xy", "z", "xyz"):
        vdir = sparse_root / f"3d/K2_{mode}/Validation"
        hp = sorted((vdir / "high").glob("*.tif"))[0]
        lp = sorted((vdir / "low").glob("*.tif"))[0]
        gt = load01(hp)
        low = load01(lp)
        # Placeholder: visualization focuses on data baseline for fast reporting.
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("GT")
        axes[0].axis("off")
        axes[1].imshow(low, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Sparse Input")
        axes[1].axis("off")
        fig.suptitle(f"MDC25 mode={mode} | best2d={best_by_algo(rows, mode, '_2d_')} | best2p5d={best_by_algo(rows, mode, '_2p5d_')} | best3d={best_by_algo(rows, mode, '_3d_')}")
        fig.tight_layout()
        fig.savefig(out_dir / f"qual_{mode}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare AI4Life-CIDC25 stacks into paired high/low TIFF frames for this project.

It expands [F, H, W] stacks into per-frame TIFF pairs and writes a dataset with:

output_root/
  Training/high/*.tif
  Training/low/*.tif
  Test/high/*.tif
  Test/low/*.tif

Example:
  python scripts/prepare_cidc25_pairs.py \
    --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
    --output_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0 \
    --train_gt A0 --train_noisy A1 B1 C2 D2 \
    --val_gt F0 --val_noisy F1 F2 F3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tifffile


def _read_stack(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim < 3:
        raise ValueError(f"Expected a 3D stack [F,H,W], got shape={arr.shape} for {path}")
    # Collapse possible channel dims by taking first channel.
    while arr.ndim > 3:
        arr = arr[:, 0]
    return arr


def _resolve_noisy(split_dir: Path, gt_stem: str, noisy_stems: List[str] | None) -> List[str]:
    if noisy_stems:
        return noisy_stems
    stems = sorted(p.stem for p in split_dir.glob("*.tif"))
    return [s for s in stems if s != gt_stem]


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_pairs(
    split_dir: Path,
    out_split: Path,
    gt_stem: str,
    noisy_stems: Iterable[str],
) -> int:
    gt_path = split_dir / f"{gt_stem}.tif"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT stack not found: {gt_path}")
    gt_stack = _read_stack(gt_path)
    gt_dtype = gt_stack.dtype

    high_dir = out_split / "high"
    low_dir = out_split / "low"
    _ensure_clean_dir(high_dir)
    _ensure_clean_dir(low_dir)

    saved = 0
    for noisy_stem in noisy_stems:
        noisy_path = split_dir / f"{noisy_stem}.tif"
        if not noisy_path.exists():
            raise FileNotFoundError(f"Noisy stack not found: {noisy_path}")
        noisy_stack = _read_stack(noisy_path)
        n_frames = min(gt_stack.shape[0], noisy_stack.shape[0])

        for i in range(n_frames):
            frame_id = f"{noisy_stem}_f{i:04d}"
            high_frame = np.asarray(gt_stack[i], dtype=gt_dtype)
            low_frame = np.asarray(noisy_stack[i], dtype=noisy_stack.dtype)
            tifffile.imwrite(str(high_dir / f"{frame_id}.tif"), high_frame)
            tifffile.imwrite(str(low_dir / f"{frame_id}.tif"), low_frame)
            saved += 1
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare CIDC25 into paired Training/Test high-low folders.")
    p.add_argument("--cidc_root", type=str, required=True, help="Path to AI4Life-CIDC25 root.")
    p.add_argument("--output_root", type=str, required=True, help="Output paired dataset root.")
    p.add_argument("--train_gt", type=str, default="A0", help="GT stack stem for Training split.")
    p.add_argument("--train_noisy", nargs="+", default=None, help="Noisy stems in Training split.")
    p.add_argument("--val_gt", type=str, default="F0", help="GT stack stem for Validation split.")
    p.add_argument("--val_noisy", nargs="+", default=None, help="Noisy stems in Validation split.")
    args = p.parse_args()

    cidc_root = Path(args.cidc_root)
    train_dir = cidc_root / "Training"
    val_dir = cidc_root / "Validation"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("CIDC root must contain Training/ and Validation/ directories.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_noisy = _resolve_noisy(train_dir, args.train_gt, args.train_noisy)
    val_noisy = _resolve_noisy(val_dir, args.val_gt, args.val_noisy)

    n_train = _save_pairs(train_dir, output_root / "Training", args.train_gt, train_noisy)
    n_test = _save_pairs(val_dir, output_root / "Test", args.val_gt, val_noisy)

    print("CIDC25 paired dataset prepared:")
    print(f"  output_root : {output_root}")
    print(f"  train_gt    : {args.train_gt}, noisy={train_noisy}, pairs={n_train}")
    print(f"  val_gt      : {args.val_gt}, noisy={val_noisy}, pairs={n_test}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_test_structure(
    sparse_dataset_root: Path, output_root: Path, overwrite: bool = False
) -> None:
    low_dir = output_root / "Test" / "low"
    high_dir = output_root / "Test" / "high"
    mask_dir = output_root / "Test" / "mask"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    gt_path = sparse_dataset_root / "pollen-fov1-dense-32.5mW.tif"
    gt_mask_path = sparse_dataset_root / "mask" / "pollen-fov1-dense-32.5mW_mask.tif"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT image not found: {gt_path}")
    if not gt_mask_path.exists():
        raise FileNotFoundError(f"GT mask not found: {gt_mask_path}")

    sparse_files = sorted(sparse_dataset_root.glob("*.tif"))
    sparse_files = [p for p in sparse_files if "dense" not in p.name.lower()]
    if not sparse_files:
        raise FileNotFoundError(f"No sparse 2D files found under: {sparse_dataset_root}")

    for low_path in sparse_files:
        stem = low_path.stem
        low_dst = low_dir / low_path.name
        high_dst = high_dir / low_path.name
        low_mask_src = sparse_dataset_root / "mask" / f"{stem}_mask.tif"
        low_mask_dst = mask_dir / f"{stem}_mask.tif"
        high_mask_dst = mask_dir / f"{stem}_gt_mask.tif"

        if not low_mask_src.exists():
            raise FileNotFoundError(f"Low mask not found: {low_mask_src}")

        if overwrite or not low_dst.exists():
            shutil.copy2(low_path, low_dst)
        if overwrite or not high_dst.exists():
            shutil.copy2(gt_path, high_dst)
        if overwrite or not low_mask_dst.exists():
            shutil.copy2(low_mask_src, low_mask_dst)
        if overwrite or not high_mask_dst.exists():
            shutil.copy2(gt_mask_path, high_mask_dst)

        print(f"[OK] Pair built: low={low_dst.name} <-> high={high_dst.name}")

    print(f"[Done] Test structure created at: {output_root / 'Test'}")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert pollen sparse 2D summary dataset into standard Test/low-high "
            "structure for evaluation."
        )
    )
    parser.add_argument(
        "--sparse_dataset_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets"),
        help="Input 2D sparse dataset root.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(
            "/data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets_LowHigh"
        ),
        help="Output dataset root containing Test/low and Test/high.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    build_test_structure(args.sparse_dataset_root, args.output_root, args.overwrite)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile


def parse_sampling_mode(folder_name: str) -> Tuple[int, str]:
    """Parse sampling factor and axis mode from folder name."""
    m = re.search(r"(\d+)x(XY|X|Y)", folder_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse sampling mode from folder name: {folder_name}")
    factor = int(m.group(1))
    mode = m.group(2).upper()
    return factor, mode


def infer_dense_shape(data_root: Path) -> Tuple[int, int]:
    """
    Infer GT dense shape from a dense acquisition folder.
    Priority:
      1) folder name containing 'dense' under Sparse/
      2) folder name containing 'dense' under dense/
    """
    candidates = list(data_root.glob("Sparse/*dense*/CellVideo1/CellVideo/*.tif"))
    if not candidates:
        candidates = list(data_root.glob("dense/*dense*/CellVideo1/CellVideo/*.tif"))
    if not candidates:
        raise FileNotFoundError(
            "No dense TIFF found. Expected something like "
            "'Sparse/*dense*/CellVideo1/CellVideo/*.tif' or 'dense/*dense*/CellVideo1/CellVideo/*.tif'."
        )
    dense = tifffile.imread(candidates[0])
    if dense.ndim == 2:
        h, w = dense.shape
    elif dense.ndim == 3:
        h, w = dense.shape[-2], dense.shape[-1]
    else:
        raise ValueError(f"Unsupported dense TIFF ndim={dense.ndim} from {candidates[0]}")
    return h, w


def restore_sparse_stack(
    sparse_stack: np.ndarray, dense_hw: Tuple[int, int], factor: int, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Restore sparse stack to dense size and create binary mask stack."""
    if sparse_stack.ndim == 2:
        sparse_stack = sparse_stack[None, ...]
        squeeze_back = True
    elif sparse_stack.ndim == 3:
        squeeze_back = False
    else:
        raise ValueError(f"Unsupported sparse stack ndim={sparse_stack.ndim}")

    n, hs, ws = sparse_stack.shape
    hd, wd = dense_hw

    out = np.zeros((n, hd, wd), dtype=sparse_stack.dtype)
    mask = np.zeros((n, hd, wd), dtype=np.uint8)

    row_step = factor if "Y" in mode else 1
    col_step = factor if "X" in mode else 1

    expected_h = hd // row_step
    expected_w = wd // col_step
    if hs != expected_h or ws != expected_w:
        raise ValueError(
            f"Sparse shape mismatch for mode {factor}x{mode}: "
            f"got {(hs, ws)}, expected {(expected_h, expected_w)} from dense {(hd, wd)}"
        )

    out[:, 0:hd:row_step, 0:wd:col_step] = sparse_stack
    mask[:, 0:hd:row_step, 0:wd:col_step] = 1

    if squeeze_back:
        out = out[0]
        mask = mask[0]
    return out, mask


def process_all(data_root: Path, output_root: Path, overwrite: bool = False) -> None:
    dense_hw = infer_dense_shape(data_root)
    sparse_tifs = sorted(data_root.glob("Sparse/*/CellVideo1/CellVideo/*.tif"))
    if not sparse_tifs:
        raise FileNotFoundError(f"No sparse TIFF files found under: {data_root / 'Sparse'}")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Dense shape inferred as HxW={dense_hw}")
    print(f"[Info] Found {len(sparse_tifs)} acquisitions in Sparse/")

    for tif_path in sparse_tifs:
        acq_name = tif_path.parents[2].name
        # dense acquisition is GT, skip preprocessing.
        if "dense" in acq_name.lower():
            print(f"[Skip] Dense acquisition: {acq_name}")
            continue

        factor, mode = parse_sampling_mode(acq_name)
        sparse_stack = tifffile.imread(tif_path)
        restored, mask = restore_sparse_stack(sparse_stack, dense_hw, factor, mode)

        out_dir = output_root / acq_name / "CellVideo1" / "CellVideo"
        out_dir.mkdir(parents=True, exist_ok=True)
        restored_path = out_dir / "CellVideo_restored.tif"
        mask_path = out_dir / "CellVideo_mask.tif"

        if (restored_path.exists() or mask_path.exists()) and not overwrite:
            print(f"[Skip] Exists (use --overwrite): {acq_name}")
            continue

        tifffile.imwrite(restored_path, restored, imagej=True)
        tifffile.imwrite(mask_path, mask, imagej=True)
        print(
            f"[OK] {acq_name}: sparse={tuple(sparse_stack.shape)} -> "
            f"restored={tuple(restored.shape)}, mask={tuple(mask.shape)}"
        )


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore pollen sparse acquisitions to GT size. "
            "Unscanned pixels are filled with 0 and a binary mask is generated."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/pollen_test"),
        help="Path to pollen_test root.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/pollen_test/preprocessed"),
        help="Output root for restored stacks and masks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    process_all(args.data_root, args.output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

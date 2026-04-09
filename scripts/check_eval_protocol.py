#!/usr/bin/env python3
"""
Protocol guardrail for evaluation splits.

Checks:
1) split directory contains high/ and low/
2) paired files match by stem
3) TIFF dimensionality statistics (2D vs stack)
4) fail-fast if stack found and stack_policy=error
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import tifffile


def _index_tiff_files(d: Path) -> Dict[str, Path]:
    exts = {".tif", ".tiff"}
    return {p.stem: p for p in sorted(d.iterdir()) if p.suffix.lower() in exts}


def _dim_hist(paths: List[Path]) -> Dict[int, int]:
    hist: Dict[int, int] = {}
    for p in paths:
        arr = tifffile.imread(str(p))
        hist[arr.ndim] = hist.get(arr.ndim, 0) + 1
    return hist


def _has_stack(paths: List[Path]) -> bool:
    for p in paths:
        if tifffile.imread(str(p)).ndim > 2:
            return True
    return False


def _fmt_hist(hist: Dict[int, int]) -> str:
    if not hist:
        return "{}"
    return ", ".join([f"{k}D:{v}" for k, v in sorted(hist.items())])


def main() -> int:
    ap = argparse.ArgumentParser(description="Check evaluation split protocol safety.")
    ap.add_argument("--split_dir", type=str, required=True, help="Directory with high/ and low/.")
    ap.add_argument(
        "--stack_policy",
        type=str,
        default="error",
        choices=["error", "first", "mean_projection", "max_projection"],
        help="Must match runtime eval stack policy.",
    )
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    high_dir = split_dir / "high"
    low_dir = split_dir / "low"

    if not high_dir.is_dir() or not low_dir.is_dir():
        print(f"[FAIL] split_dir must contain high/ and low/: {split_dir}")
        return 2

    hi = _index_tiff_files(high_dir)
    lo = _index_tiff_files(low_dir)
    common = sorted(set(hi) & set(lo))
    only_hi = sorted(set(hi) - set(lo))
    only_lo = sorted(set(lo) - set(hi))

    print(f"[INFO] split_dir: {split_dir}")
    print(f"[INFO] high count: {len(hi)}, low count: {len(lo)}, paired: {len(common)}")
    if only_hi:
        print(f"[WARN] only in high ({len(only_hi)}): {only_hi[:5]}")
    if only_lo:
        print(f"[WARN] only in low ({len(only_lo)}): {only_lo[:5]}")
    if not common:
        print("[FAIL] no matched pairs.")
        return 3

    high_paths = [hi[s] for s in common]
    low_paths = [lo[s] for s in common]
    hi_hist = _dim_hist(high_paths)
    lo_hist = _dim_hist(low_paths)
    print(f"[INFO] high ndim hist: {_fmt_hist(hi_hist)}")
    print(f"[INFO] low ndim hist : {_fmt_hist(lo_hist)}")

    stack_found = _has_stack(high_paths) or _has_stack(low_paths)
    if stack_found and args.stack_policy == "error":
        print("[FAIL] stack TIFF detected but stack_policy=error.")
        print("       Use a 2.5D/3D pipeline, or set explicit projection policy for 2D baseline.")
        return 4

    if stack_found:
        print(f"[WARN] stack TIFF detected and will be reduced by policy={args.stack_policy}.")
    else:
        print("[OK] all paired TIFF are 2D.")

    print("[OK] protocol checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


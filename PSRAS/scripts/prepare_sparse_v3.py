"""Protocol v3 data preparation for CIDC25 (PLAN.md Phase 1).

What this script does
---------------------
1. Fits a single intensity calibration (offset, scale) on the TRAINING
   stacks only (DEC-004) and saves it to calibration.json.
2. Converts every stack to a normalized float16 .npy for fast training IO.
3. Writes split.json with the stack-level split (PLAN.md M7): training /
   validation / test never share a stack.

What it deliberately does NOT do
--------------------------------
- No sparse simulation here.  The acquisition Y = M.(H*X + n) is generated
  at dataset runtime by physics.acquisition (single source of truth,
  deterministic under the config seed; DEC-007).  Pre-generating warm-start
  or interpolated volumes (the legacy failure mode) is impossible by
  construction.

Usage
-----
    python scripts/prepare_sparse_v3.py \
        --data_root ../ThreePM_openDatasets/AI4Life-CIDC25 \
        --out_dir ../runs_v3/data/cidc25_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.normalization import IntensityCalibration  # noqa: E402

# Stack-level split (PLAN.md M7).  Training stacks are simulated samples;
# F0 is the only clean stack and is reserved for val/test.  A temporal
# split within F0 keeps val and test disjoint while sharing content --
# documented as provisional until more samples are available.
SPLIT = {
    "train": {"stacks": ["A1", "B1", "C2", "D2"], "frame_range": None},
    "val": {"stacks": ["F0"], "frame_range": [0, 750]},
    "test": {"stacks": ["F0"], "frame_range": [750, 1500]},
}


def convert_stack(src: Path, calib: IntensityCalibration, out_path: Path) -> dict:
    vol = tifffile.imread(src)                       # int16 [T, H, W]
    norm = calib.apply(vol).astype(np.float16)       # [0, 1]
    np.save(out_path, norm)
    return {"source": str(src), "shape": list(norm.shape),
            "dtype": "float16", "normalized_range": [0.0, 1.0]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--low_pct", type=float, default=0.5)
    p.add_argument("--high_pct", type=float, default=99.9)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stacks_dir = args.out_dir / "stacks"
    stacks_dir.mkdir(exist_ok=True)

    train_stacks = [args.data_root / "Training" / f"{s}.tif" for s in SPLIT["train"]["stacks"]]
    print("[1/3] Fitting calibration on training stacks only ...")
    vols = [tifffile.imread(s) for s in train_stacks]
    calib = IntensityCalibration.fit(vols, args.low_pct, args.high_pct)
    del vols
    calib.save(args.out_dir / "calibration.json")
    print(f"      offset={calib.offset:.2f} scale={calib.scale:.2f}")

    print("[2/3] Converting stacks to normalized float16 .npy ...")
    manifest = {}
    for name, path in [(s, args.data_root / "Training" / f"{s}.tif") for s in SPLIT["train"]["stacks"]] + \
                      [(s, args.data_root / "Validation" / f"{s}.tif") for s in SPLIT["val"]["stacks"]]:
        info = convert_stack(path, calib, stacks_dir / f"{name}.npy")
        manifest[name] = info
        print(f"      {name}: {info['shape']}")

    print("[3/3] Writing split.json ...")
    (args.out_dir / "split.json").write_text(json.dumps(
        {"split": SPLIT, "manifest": manifest,
         "calibration": calib.to_dict(),
         "note": "Sparse observation is generated at dataset runtime via "
                 "physics.acquisition (DEC-007); nothing sparse is stored here."},
        indent=2))
    print(f"Done -> {args.out_dir}")


if __name__ == "__main__":
    main()

"""Prepare split v2 with honest semantics for the CIDC25 validation family.

DEC-025 (protocol v4, P0 fix).  Two facts drive this script:

1. The pre-v4 split used F0[0:750] as val and F0[750:1500] as test -- same
   acquisition stack, temporally separated.  F0 is content-independent from
   the training stacks (a different simulated sample, per the CIDC25
   readme), so the honest description is: "content-independent test with
   val/test temporal split inside F0" -- model selection (early stop, λ,
   architecture) touched F0, which must be stated, not hidden.

2. F1/F2/F3 are NOT independent stacks: measured on frames 100-103,
   corr(F1,F0)=0.71 / corr(F2,F0)=0.36 / corr(F3,F0)=0.18 with residual std
   0.072/0.180/0.309 -- they are the SAME F0 content at increasing noise
   levels (CIDC25 Task 2 noise-level variants).  Using them as a
   stack-independent test would be wrong (clean-model output vs noisy GT
   collapses to ~16 dB by construction, verified 2026-09-22).

This script therefore:
  1. converts F1/F2/F3 to normalized float16 stacks with the SAME
     calibration fitted on training stacks (calibration.json, unchanged);
  2. rewrites split.json:
       train        A1 B1 C2 D2                 (unchanged)
       val          F0 [0:750]                  (unchanged: model selection
                                                 continuity with legacy runs)
       test         F0 [750:1500]               (content-independent from
                                                 train; temporal split from
                                                 val -- selection-touched)
       noise_stress F1 F2 F3 full frames        (F0 content at noise levels
                                                 ~1x/2.5x/4x; GT must be the
                                                 paired clean F0 frame, NOT
                                                 the noisy stack itself --
                                                 requires paired-input mode,
                                                 pre-registered separately)

Local data supports exactly ONE clean unseen stack (F0).  A fully
selection-independent multi-stack test needs external data (MDC25 /
real acquisitions) -- see PLAN.md Phase 5/6.

Usage:
    python scripts/prepare_split_v2.py \
        --data_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
        --out_dir runs_v3/data/cidc25_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.normalization import IntensityCalibration  # noqa: E402

NEW_TEST_STACKS = ["F1", "F2", "F3"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    out_dir = args.out_dir
    split_path = out_dir / "split.json"
    split_doc = json.loads(split_path.read_text())
    calib = IntensityCalibration(**json.loads((out_dir / "calibration.json").read_text()))

    stacks_dir = out_dir / "stacks"
    for name in NEW_TEST_STACKS:
        out_path = stacks_dir / f"{name}.npy"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
        else:
            src = args.data_root / "Validation" / f"{name}.tif"
            vol = tifffile.imread(src)                    # int16 [T, H, W]
            norm = calib.apply(vol).astype(np.float16)    # [0, 1]
            np.save(out_path, norm)
            print(f"[converted] {name}: {src} -> {out_path} shape={norm.shape}")
        split_doc["manifest"].setdefault(name, {
            "source": str(args.data_root / "Validation" / f"{name}.tif"),
            "shape": list(np.load(out_path, mmap_mode="r").shape),
            "dtype": "float16", "normalized_range": [0.0, 1.0]})

    # Rewrite the split block with honest semantics (see module docstring).
    split_doc["split"] = {
        "train": {"stacks": ["A1", "B1", "C2", "D2"], "frame_range": None},
        "val": {"stacks": ["F0"], "frame_range": [0, 750]},
        "test": {"stacks": ["F0"], "frame_range": [750, 1500]},
        "noise_stress": {"stacks": NEW_TEST_STACKS, "frame_range": None},
    }
    split_doc["split_v2_note"] = (
        "DEC-025 (protocol v4): test = F0[750:1500] -- content-independent "
        "from train (F0 is a different simulated sample), temporal split from "
        "val (model selection touched F0; stated, not hidden). noise_stress = "
        "F1/F2/F3 = same F0 content at increasing noise (corr .71/.36/.18): "
        "NOT independent stacks; usable only with paired clean F0 frames as "
        "GT, pre-registered separately. Calibration fitted on training stacks "
        "only and never recomputed on any test stack.")
    split_path.write_text(json.dumps(split_doc, indent=2))
    print(f"[done] split.json rewritten: test=F0[750:1500] (honest semantics), "
          f"noise_stress={NEW_TEST_STACKS}, val=F0[0:750] (unchanged)")


if __name__ == "__main__":
    main()

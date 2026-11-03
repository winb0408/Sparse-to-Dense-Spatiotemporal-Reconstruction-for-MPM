#!/bin/bash
set -u
cd /ssd1/wjb/SPUR-MPM
echo "=== A. 3D build $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.build_sparse \
  --index data_local/dryad_cache/index_cohort.json \
  --split spur_mpm/experiments/E020_cohort/split.json \
  --fold 0 --out data_local/sparse/cohort3d --dump-y \
  > runs/E023_phase1/build3d.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"
echo "=== B. 2D build $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.build_sparse \
  --index data_local/dryad_cache/index_cohort.json \
  --split spur_mpm/experiments/E020_cohort/split.json \
  --fold 0 --as-2d --out data_local/sparse/cohort2d --dump-y \
  > runs/E023_phase1/build2d.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"
echo "=== C. mask cross-check $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.verify_dataset_masks \
  --index data_local/dryad_cache/index_cohort.json \
  --split spur_mpm/experiments/E020_cohort/split.json \
  --manifest data_local/sparse/cohort3d/manifest.json \
  --levels y_k2 y_k4 z_k3 yz_k2x2 \
  --depth 7 --patch 64 \
  --out data_local/sparse/mask_crosscheck.json \
  > runs/E023_phase1/crosscheck.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"
echo "=== D. sparse figures $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.sparse_figures \
  --index data_local/dryad_cache/index_cohort.json \
  --manifest3d data_local/sparse/cohort3d/manifest.json \
  --manifest2d data_local/sparse/cohort2d/manifest.json \
  --sample M2_FOVI_S1_vessels --out spur_mpm/figures/sparse \
  > runs/E023_phase1/sparse_figures.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"
echo ALL_DONE

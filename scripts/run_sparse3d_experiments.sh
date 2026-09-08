#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# CIDC25 3D Sparse Reconstruction Experiment Matrix
# ═══════════════════════════════════════════════════════════════════
#
# Phase 1: Baseline 2D / 2.5D / 3D under three sparse modes (9 runs)
# Phase 2: PSF-enhanced runs on best Phase 1 config
#
# Prerequisites:
#   1. CIDC25 data downloaded to CIDC_ROOT with Training/ containing .tif stacks
#   2. Paired frames prepared (for 2D/2.5D):
#      python scripts/prepare_cidc25_sparse3d.py \
#        --cidc_root $CIDC_ROOT --output_root $SPARSE_ROOT \
#        --gt_stem A1 --split Training --modes xy z xyz --sparsity_factors 4
#
# Usage:
#   bash scripts/run_sparse3d_experiments.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
CIDC_ROOT="/data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25"
SPARSE_ROOT="/data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d"
GT_STEM="A1"
K=4              # sparsity factor
EPOCHS=50        # quick experiments; increase for final
BATCH=4
PATCH=256
VOL_T=16         # 3D temporal patch depth
LR=1e-4

# ── Phase 0: Prepare sparse paired data ──────────────────────────
echo "=== Phase 0: Prepare sparse data ==="
python scripts/prepare_cidc25_sparse3d.py \
  --cidc_root "$CIDC_ROOT" \
  --output_root "$SPARSE_ROOT" \
  --gt_stem "$GT_STEM" \
  --split Training \
  --sparsity_factors $K \
  --modes xy z xyz \
  --max_frames 200

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Baseline experiments (9 runs = 3 sparse × 3 algorithm)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 1: Baseline experiments ==="

for SPARSE_MODE in xy z xyz; do
  DATA="${SPARSE_ROOT}/K${K}_${SPARSE_MODE}"

  # ── 2D: per-frame independent reconstruction ──────────────────
  # NOTE(2026-04-10): sparse3d_2d_xy_K4 已完成，续跑时跳过。
  if [ "$SPARSE_MODE" = "xy" ]; then
    echo ""
    echo "--- 2D / sparse_mode=${SPARSE_MODE} / K=${K} ---"
    echo "[SKIP] sparse3d_2d_xy_K4 已跑完，跳过。"
    # 原命令（保留用于记录）:
    # python train.py \
    #   --task spatial_2d --model swin_unet_2d \
    #   --data_root "${DATA}" \
    #   --patch_size $PATCH --batch_size $BATCH \
    #   --sparsity_factor $K --scan_pattern grid \
    #   --epochs $EPOCHS --lr $LR \
    #   --split_val_from_train 0.1 \
    #   --no_dino \
    #   --run_name "sparse3d_2d_${SPARSE_MODE}_K${K}"
  else
    echo ""
    echo "--- 2D / sparse_mode=${SPARSE_MODE} / K=${K} ---"
    python train.py \
      --task spatial_2d --model swin_unet_2d \
      --data_root "${DATA}" \
      --patch_size $PATCH --batch_size $BATCH \
      --sparsity_factor $K --scan_pattern grid \
      --epochs $EPOCHS --lr $LR \
      --split_val_from_train 0.1 \
      --no_dino \
      --run_name "sparse3d_2d_${SPARSE_MODE}_K${K}"
  fi

  # ── 2.5D: multi-frame temporal window ─────────────────────────
  # NOTE(2026-04-10): sparse3d_2p5d_xy_K4 已完成，续跑时跳过。
  if [ "$SPARSE_MODE" = "xy" ]; then
    echo ""
    echo "--- 2.5D / sparse_mode=${SPARSE_MODE} / K=${K} ---"
    echo "[SKIP] sparse3d_2p5d_xy_K4 已跑完，跳过。"
    # 原命令（保留用于记录）:
    # python train.py \
    #   --task spatial_2p5d --model swin_unet_2d \
    #   --data_root "${DATA}" \
    #   --temporal_window 3 \
    #   --patch_size $PATCH --batch_size $BATCH \
    #   --epochs $EPOCHS --lr $LR \
    #   --split_val_from_train 0.1 \
    #   --no_dino \
    #   --run_name "sparse3d_2p5d_${SPARSE_MODE}_K${K}"
  else
    echo ""
    echo "--- 2.5D / sparse_mode=${SPARSE_MODE} / K=${K} ---"
    python train.py \
      --task spatial_2p5d --model swin_unet_2d \
      --data_root "${DATA}" \
      --temporal_window 3 \
      --patch_size $PATCH --batch_size $BATCH \
      --epochs $EPOCHS --lr $LR \
      --split_val_from_train 0.1 \
      --no_dino \
      --run_name "sparse3d_2p5d_${SPARSE_MODE}_K${K}"
  fi

  # ── 3D: VideoSwinUNet volume reconstruction ───────────────────
  echo ""
  echo "--- 3D / sparse_mode=${SPARSE_MODE} / K=${K} ---"
  python train.py \
    --task spatiotemporal_3d --model video_swin_unet \
    --cidc25_root "$CIDC_ROOT" \
    --cidc25_gt_stem "$GT_STEM" \
    --data_root "${DATA}" \
    --sparse_mode $SPARSE_MODE \
    --sparsity_factor $K --sparsity_factor_z $K \
    --volume_patch_t $VOL_T \
    --patch_size $PATCH --batch_size 2 \
    --n_frames $VOL_T \
    --epochs $EPOCHS --lr $LR \
    --no_dino \
    --run_name "sparse3d_3d_${SPARSE_MODE}_K${K}"
done

# ═══════════════════════════════════════════════════════════════════
# Phase 2: PSF-enhanced experiments (on best sparse mode)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 2: PSF-enhanced runs ==="

BEST_MODE="xy"  # adjust after Phase 1 results

# 3D + PSF input channel
python train.py \
  --task spatiotemporal_3d --model video_swin_unet \
  --cidc25_root "$CIDC_ROOT" \
  --cidc25_gt_stem "$GT_STEM" \
  --data_root "${SPARSE_ROOT}/K${K}_${BEST_MODE}" \
  --sparse_mode $BEST_MODE \
  --sparsity_factor $K --sparsity_factor_z $K \
  --volume_patch_t $VOL_T \
  --patch_size $PATCH --batch_size 2 \
  --n_frames $VOL_T \
  --psf_input_channel \
  --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
  --epochs $EPOCHS --lr $LR \
  --no_dino \
  --run_name "sparse3d_3d_${BEST_MODE}_K${K}_psf_ch"

# 3D + PSF consistency loss
python train.py \
  --task spatiotemporal_3d --model video_swin_unet \
  --cidc25_root "$CIDC_ROOT" \
  --cidc25_gt_stem "$GT_STEM" \
  --data_root "${SPARSE_ROOT}/K${K}_${BEST_MODE}" \
  --sparse_mode $BEST_MODE \
  --sparsity_factor $K --sparsity_factor_z $K \
  --volume_patch_t $VOL_T \
  --patch_size $PATCH --batch_size 2 \
  --n_frames $VOL_T \
  --lambda_psf 0.1 \
  --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
  --epochs $EPOCHS --lr $LR \
  --no_dino \
  --run_name "sparse3d_3d_${BEST_MODE}_K${K}_psf_loss"

# 3D + PSF channel + PSF loss (full)
python train.py \
  --task spatiotemporal_3d --model video_swin_unet \
  --cidc25_root "$CIDC_ROOT" \
  --cidc25_gt_stem "$GT_STEM" \
  --data_root "${SPARSE_ROOT}/K${K}_${BEST_MODE}" \
  --sparse_mode $BEST_MODE \
  --sparsity_factor $K --sparsity_factor_z $K \
  --volume_patch_t $VOL_T \
  --patch_size $PATCH --batch_size 2 \
  --n_frames $VOL_T \
  --psf_input_channel --lambda_psf 0.1 \
  --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
  --epochs $EPOCHS --lr $LR \
  --no_dino \
  --run_name "sparse3d_3d_${BEST_MODE}_K${K}_psf_full"

# 3D + Wiener init (ablation)
python train.py \
  --task spatiotemporal_3d --model video_swin_unet \
  --cidc25_root "$CIDC_ROOT" \
  --cidc25_gt_stem "$GT_STEM" \
  --data_root "${SPARSE_ROOT}/K${K}_${BEST_MODE}" \
  --sparse_mode $BEST_MODE \
  --sparsity_factor $K --sparsity_factor_z $K \
  --volume_patch_t $VOL_T \
  --patch_size $PATCH --batch_size 2 \
  --n_frames $VOL_T \
  --wiener_init \
  --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
  --epochs $EPOCHS --lr $LR \
  --no_dino \
  --run_name "sparse3d_3d_${BEST_MODE}_K${K}_wiener"

echo ""
echo "=== All experiments complete ==="
echo "View results: tensorboard --logdir runs/"

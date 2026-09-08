#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# K=2 sparse sweep only (K=3 removed per request).
# Two workers: GPU 2 and GPU 3 run disjoint pending jobs in parallel.
# Skips: prepared datasets under SPARSE_ROOT, and runs with runs/<name>/best.pth.

CIDC_ROOT="/data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25"
SPARSE_ROOT="/data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d"
GT_STEM="A1"
K=2
MODES=(xy z xyz)

EPOCHS=50
BATCH_2D=4
BATCH_3D=2
PATCH=256
VOL_T=16
LR=1e-4

ensure_data() {
  local all_ready=1
  for mode in "${MODES[@]}"; do
    local base="${SPARSE_ROOT}/K${K}_${mode}/Training"
    local high_dir="${base}/high"
    local low_dir="${base}/low"
    local gt_npy="${base}/stacks/${GT_STEM}_gt.npy"
    local sparse_npy="${base}/stacks/${GT_STEM}_sparse.npy"
    if [ ! -d "${high_dir}" ] || [ ! -d "${low_dir}" ] || \
       [ ! -f "${gt_npy}" ] || [ ! -f "${sparse_npy}" ]; then
      all_ready=0
      break
    fi
  done

  if [ "${all_ready}" -eq 1 ]; then
    echo "[SKIP] K=${K} sparse datasets already prepared."
    return
  fi

  echo "=== Prepare sparse data for K=${K} ==="
  python scripts/prepare_cidc25_sparse3d.py \
    --cidc_root "${CIDC_ROOT}" \
    --output_root "${SPARSE_ROOT}" \
    --gt_stem "${GT_STEM}" \
    --split Training \
    --sparsity_factors "${K}" \
    --modes xy z xyz \
    --max_frames 200
}

# Append one pending job per line: GPU_ID|run_name|train.py args (no --run_name)
# Round-robin among *pending* jobs only: 2,3,2,3,...
job_idx=0
append_if_pending() {
  local run_name="$1"
  shift
  if [ -f "runs/${run_name}/best.pth" ]; then
    echo "[SKIP] ${run_name} (best.pth exists)"
    return
  fi
  local gpu=$(( 2 + (job_idx % 2) ))
  job_idx=$((job_idx + 1))
  echo "${gpu}|${run_name}|$*" >> "${QUEUE_FILE}"
}

worker() {
  local gpu="$1"
  local tag="$2"
  local myfile="${QUEUE_FILE}.${tag}"
  grep "^${gpu}|" "${QUEUE_FILE}" > "${myfile}" || true
  if [ ! -s "${myfile}" ]; then
    echo "[worker GPU${gpu}] no pending jobs."
    return
  fi
  echo "[worker GPU${gpu}] $(wc -l < "${myfile}") job(s)"
  while IFS= read -r line; do
    # line format: GPU|run_name| args...
    local run_name
    run_name=$(echo "${line}" | cut -d'|' -f2)
    local args
    args=$(echo "${line}" | cut -d'|' -f3-)
    echo ""
    echo "──────────────────────────────────────────────"
    echo "[RUN GPU${gpu}] ${run_name}"
    echo "──────────────────────────────────────────────"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py ${args} --run_name "${run_name}"
  done < "${myfile}"
}

QUEUE_FILE="$(mktemp -t k2_sweep_jobs.XXXXXX)"
trap 'rm -f "${QUEUE_FILE}" "${QUEUE_FILE}.a" "${QUEUE_FILE}.b" 2>/dev/null || true' EXIT

ensure_data

: > "${QUEUE_FILE}"

# --- Build job queue (K=2 only; same order as original sweep) ---
for mode in "${MODES[@]}"; do
  data="${SPARSE_ROOT}/K${K}_${mode}"
  append_if_pending "sparse3d_2d_${mode}_K${K}" \
    --task spatial_2d --model swin_unet_2d \
    --data_root "${data}" \
    --patch_size "${PATCH}" --batch_size "${BATCH_2D}" \
    --sparsity_factor "${K}" --scan_pattern grid \
    --epochs "${EPOCHS}" --lr "${LR}" \
    --split_val_from_train 0.1 --no_dino

  append_if_pending "sparse3d_2p5d_${mode}_K${K}" \
    --task spatial_2p5d --model swin_unet_2d \
    --data_root "${data}" \
    --temporal_window 3 \
    --patch_size "${PATCH}" --batch_size "${BATCH_2D}" \
    --epochs "${EPOCHS}" --lr "${LR}" \
    --split_val_from_train 0.1 --no_dino

  append_if_pending "sparse3d_3d_${mode}_K${K}" \
    --task spatiotemporal_3d --model video_swin_unet \
    --cidc25_root "${CIDC_ROOT}" --cidc25_gt_stem "${GT_STEM}" \
    --data_root "${data}" --sparse_mode "${mode}" \
    --sparsity_factor "${K}" --sparsity_factor_z "${K}" \
    --volume_patch_t "${VOL_T}" --n_frames "${VOL_T}" \
    --patch_size "${PATCH}" --batch_size "${BATCH_3D}" \
    --epochs "${EPOCHS}" --lr "${LR}" --no_dino

  append_if_pending "sparse3d_3d_${mode}_K${K}_wiener" \
    --task spatiotemporal_3d --model video_swin_unet \
    --cidc25_root "${CIDC_ROOT}" --cidc25_gt_stem "${GT_STEM}" \
    --data_root "${data}" --sparse_mode "${mode}" \
    --sparsity_factor "${K}" --sparsity_factor_z "${K}" \
    --volume_patch_t "${VOL_T}" --n_frames "${VOL_T}" \
    --patch_size "${PATCH}" --batch_size "${BATCH_3D}" \
    --wiener_init --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
    --epochs "${EPOCHS}" --lr "${LR}" --no_dino
done

data_xy="${SPARSE_ROOT}/K${K}_xy"

append_if_pending "sparse3d_2d_xy_K${K}_wiener" \
  --task spatial_2d --model swin_unet_2d \
  --data_root "${data_xy}" \
  --patch_size "${PATCH}" --batch_size "${BATCH_2D}" \
  --sparsity_factor "${K}" --scan_pattern grid \
  --wiener_init --psf_sigma_lateral 1.0 \
  --epochs "${EPOCHS}" --lr "${LR}" \
  --split_val_from_train 0.1 --no_dino

append_if_pending "sparse3d_2p5d_xy_K${K}_mask" \
  --task spatial_2p5d --model swin_unet_2d \
  --cidc25_root "${CIDC_ROOT}" --cidc25_gt_stem "${GT_STEM}" \
  --cidc25_split Training --sparse_mode xy \
  --sparsity_factor "${K}" --sparsity_factor_z "${K}" \
  --temporal_window 3 \
  --volume_patches_per_stack 200 \
  --patch_size "${PATCH}" --batch_size "${BATCH_2D}" \
  --epochs "${EPOCHS}" --lr "${LR}" \
  --split_val_from_train 0.1 --no_dino

append_if_pending "sparse3d_2p5d_xy_K${K}_mask_wiener" \
  --task spatial_2p5d --model swin_unet_2d \
  --cidc25_root "${CIDC_ROOT}" --cidc25_gt_stem "${GT_STEM}" \
  --cidc25_split Training --sparse_mode xy \
  --sparsity_factor "${K}" --sparsity_factor_z "${K}" \
  --temporal_window 3 \
  --volume_patches_per_stack 200 \
  --patch_size "${PATCH}" --batch_size "${BATCH_2D}" \
  --wiener_init --psf_sigma_lateral 1.0 \
  --epochs "${EPOCHS}" --lr "${LR}" \
  --split_val_from_train 0.1 --no_dino

if [ ! -s "${QUEUE_FILE}" ]; then
  echo "No pending jobs (everything already has best.pth)."
  exit 0
fi

echo ""
echo "=== Pending job queue (${QUEUE_FILE}) ==="
cat "${QUEUE_FILE}" | while IFS= read -r l; do echo "  ${l}"; done

worker 2 a &
pid2=$!
worker 3 b &
pid3=$!
ec2=0
ec3=0
wait "${pid2}" || ec2=$?
wait "${pid3}" || ec3=$?
if [ "${ec2}" -ne 0 ] || [ "${ec3}" -ne 0 ]; then
  echo "One or more workers failed (GPU2 exit=${ec2}, GPU3 exit=${ec3})."
  exit 1
fi

echo ""
echo "K=2 sweep workers finished."
echo "TensorBoard: tensorboard --logdir runs/"

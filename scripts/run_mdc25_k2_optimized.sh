#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MDC25_SPARSE_ROOT="/data2/wjb/threePM/ThreePM_openDatasets/MDC25_sparse"
K=2
MODES=(xy z xyz)

EPOCHS=50
BATCH_2D=4
BATCH_3D=2
PATCH=256
VOL_T=16
LR=1e-4
MAX_STACKS=200

QUEUE_FILE="$(mktemp -t mdc25_k2_jobs.XXXXXX)"
trap 'rm -f "${QUEUE_FILE}" "${QUEUE_FILE}.2" "${QUEUE_FILE}.3" 2>/dev/null || true' EXIT

need_path() {
  local p="$1"
  if [ ! -e "$p" ]; then
    echo "Missing required path: $p"
    exit 1
  fi
}

check_data_ready() {
  for mode in "${MODES[@]}"; do
    need_path "${MDC25_SPARSE_ROOT}/3d/K${K}_${mode}/Training/high"
    need_path "${MDC25_SPARSE_ROOT}/3d/K${K}_${mode}/Training/low"
    need_path "${MDC25_SPARSE_ROOT}/3d/K${K}_${mode}/Training/stacks"
    need_path "${MDC25_SPARSE_ROOT}/3d/K${K}_${mode}/Validation/high"
    need_path "${MDC25_SPARSE_ROOT}/3d/K${K}_${mode}/Validation/low"
    need_path "${MDC25_SPARSE_ROOT}/3d/K${K}_${mode}/Validation/stacks"
  done
}

append_if_pending() {
  local run_name="$1"
  shift
  if [ -f "runs/${run_name}/best.pth" ]; then
    echo "[SKIP] ${run_name} (best.pth exists)"
    return
  fi
  echo "${run_name}|$*" >> "${QUEUE_FILE}"
}

get_free_mem_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d '[:space:]'
}

pick_execution_mode() {
  local free2 free3
  free2="$(get_free_mem_mib 2 || echo 0)"
  free3="$(get_free_mem_mib 3 || echo 0)"
  echo "[GPU] free mem: GPU2=${free2}MiB, GPU3=${free3}MiB"
  # If both GPUs have >= 12GB free, run parallel workers; else queue on freer one.
  if [ "${free2}" -ge 12000 ] && [ "${free3}" -ge 12000 ]; then
    echo "parallel"
  else
    echo "single"
  fi
}

split_queue_round_robin() {
  : > "${QUEUE_FILE}.2"
  : > "${QUEUE_FILE}.3"
  local idx=0
  while IFS= read -r line; do
    if [ $((idx % 2)) -eq 0 ]; then
      echo "${line}" >> "${QUEUE_FILE}.2"
    else
      echo "${line}" >> "${QUEUE_FILE}.3"
    fi
    idx=$((idx + 1))
  done < "${QUEUE_FILE}"
}

worker() {
  local gpu="$1"
  local qfile="$2"
  if [ ! -s "${qfile}" ]; then
    echo "[worker GPU${gpu}] no pending jobs."
    return
  fi
  echo "[worker GPU${gpu}] $(wc -l < "${qfile}") job(s)"
  while IFS= read -r line; do
    local run_name args
    run_name="$(echo "${line}" | cut -d'|' -f1)"
    args="$(echo "${line}" | cut -d'|' -f2-)"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "[RUN GPU${gpu}] ${run_name}"
    echo "──────────────────────────────────────────────"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py ${args} --run_name "${run_name}"
  done < "${qfile}"
}

build_queue() {
  : > "${QUEUE_FILE}"
  for mode in "${MODES[@]}"; do
    data3d="${MDC25_SPARSE_ROOT}/3d/K${K}_${mode}"
    data2d="${MDC25_SPARSE_ROOT}/2d/K${K}_xy"
    # 2D optimized
    append_if_pending "sparse3d_2d_${mode}_K${K}_wiener_mdc25" \
      --task spatial_2d --model swin_unet_2d \
      --data_root "${data2d}" \
      --patch_size "${PATCH}" --batch_size "${BATCH_2D}" \
      --sparsity_factor "${K}" --scan_pattern grid \
      --patches_per_image 1 \
      --wiener_init --psf_sigma_lateral 1.0 \
      --epochs "${EPOCHS}" --lr "${LR}" \
      --split_val_from_train 0.1 --no_dino

    # 2.5D optimized (mask-aware path + wiener)
    append_if_pending "sparse3d_2p5d_${mode}_K${K}_mask_wiener_mdc25" \
      --task spatial_2p5d --model swin_unet_2d \
      --data_root "${data3d}" \
      --sparsity_factor "${K}" --sparsity_factor_z "${K}" \
      --temporal_window 3 --volume_patches_per_stack 1 \
      --volume_max_stacks "${MAX_STACKS}" \
      --patch_size "${PATCH}" --batch_size "${BATCH_2D}" \
      --wiener_init --psf_sigma_lateral 1.0 \
      --epochs "${EPOCHS}" --lr "${LR}" \
      --split_val_from_train 0.1 --no_dino

    # 3D optimized (wiener + mask default on)
    append_if_pending "sparse3d_3d_${mode}_K${K}_wiener_mdc25" \
      --task spatiotemporal_3d --model video_swin_unet \
      --data_root "${data3d}" \
      --sparse_mode "${mode}" \
      --sparsity_factor "${K}" --sparsity_factor_z "${K}" \
      --volume_patch_t "${VOL_T}" --n_frames "${VOL_T}" \
      --volume_patches_per_stack 1 --volume_max_stacks "${MAX_STACKS}" \
      --patch_size "${PATCH}" --batch_size "${BATCH_3D}" \
      --wiener_init --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
      --epochs "${EPOCHS}" --lr "${LR}" --no_dino
  done
}

check_data_ready
build_queue

if [ ! -s "${QUEUE_FILE}" ]; then
  echo "No pending jobs (all target runs already have best.pth)."
  exit 0
fi

echo ""
echo "=== Pending jobs ==="
cat "${QUEUE_FILE}" | while IFS= read -r l; do echo "  ${l}"; done

mode="$(pick_execution_mode)"
if [ "${mode}" = "parallel" ]; then
  split_queue_round_robin
  worker 2 "${QUEUE_FILE}.2" &
  pid2=$!
  worker 3 "${QUEUE_FILE}.3" &
  pid3=$!
  ec2=0; ec3=0
  wait "${pid2}" || ec2=$?
  wait "${pid3}" || ec3=$?
  if [ "${ec2}" -ne 0 ] || [ "${ec3}" -ne 0 ]; then
    echo "One or more workers failed (GPU2=${ec2}, GPU3=${ec3})."
    exit 1
  fi
else
  free2="$(get_free_mem_mib 2 || echo 0)"
  free3="$(get_free_mem_mib 3 || echo 0)"
  target_gpu=2
  if [ "${free3}" -gt "${free2}" ]; then
    target_gpu=3
  fi
  echo "[QUEUE] running sequentially on GPU${target_gpu}"
  worker "${target_gpu}" "${QUEUE_FILE}"
fi

echo ""
echo "MDC25 K=2 optimized sweep finished."

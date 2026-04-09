#!/bin/bash
# 按数据集简写归档训练/评估/可视化结果，便于区分不同数据源。
#
# 命名约定（简写）:
#   DD    = Denoising_Dataset (Zenodo Noisy Nuclei)
#   MDC25 = AI4Life-MDC25 2D denoising (Nuclei)（large+small 混合训练）
#
# 用法:
#   ./scripts/archive_results_by_dataset.sh DD     # 将当前 runs/eval_results/figures 归档为 *_DD
#   ./scripts/archive_results_by_dataset.sh MDC25  # 归档为 *_MDC25（例如混合训练后的结果）
#
# 若目标目录已存在则跳过；不会覆盖已有内容。

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUFFIX="${1:-DD}"
if [[ -z "$SUFFIX" ]]; then
  echo "Usage: $0 <dataset_short_name>   (e.g. DD or MDC25)"
  exit 1
fi

# runs: 将 run 重命名为 run_DD / run_MDC25 等
if [[ -d "runs/run" ]] && [[ ! -d "runs/run_${SUFFIX}" ]]; then
  mv "runs/run" "runs/run_${SUFFIX}"
  echo "[archive] runs/run -> runs/run_${SUFFIX}"
elif [[ -d "runs/run" ]]; then
  echo "[archive] skip runs/run (runs/run_${SUFFIX} already exists)"
fi

# eval_results -> eval_results_DD
if [[ -d "eval_results" ]] && [[ ! -d "eval_results_${SUFFIX}" ]]; then
  mv "eval_results" "eval_results_${SUFFIX}"
  echo "[archive] eval_results -> eval_results_${SUFFIX}"
elif [[ -d "eval_results" ]]; then
  echo "[archive] skip eval_results (eval_results_${SUFFIX} already exists)"
fi

# figures -> figures_DD
if [[ -d "figures" ]] && [[ ! -d "figures_${SUFFIX}" ]]; then
  mv "figures" "figures_${SUFFIX}"
  echo "[archive] figures -> figures_${SUFFIX}"
elif [[ -d "figures" ]]; then
  echo "[archive] skip figures (figures_${SUFFIX} already exists)"
fi

echo "[archive] done (suffix=${SUFFIX})"

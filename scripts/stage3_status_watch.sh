#!/usr/bin/env bash
set -euo pipefail
ROOT="/data2/wjb/threePM"
TOPIC_DIR="$ROOT/projects/three-photon-spatiotemporal-sparse-sampling-fast-high-quality-reconstruction"
LOG_FILE="$TOPIC_DIR/stage3_logs/stage3_pipeline.log"
OUT_FILE="$TOPIC_DIR/STAGE3_STATUS.md"

while true; do
  NOW=$(date '+%F %T')
  PIPE_PID=$(pgrep -f "stage3_auto_pipeline.sh" | head -n 1 || true)
  TRAIN_PID=$(pgrep -f "train.py --task spatial_2p5d --model swin_unet_2d --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0" | head -n 1 || true)

  if [[ -n "${PIPE_PID}" ]]; then
    RUN_STATE="RUNNING (pipeline pid=${PIPE_PID})"
  else
    RUN_STATE="NOT_RUNNING"
  fi

  TRAIN_ELAPSED="N/A"
  if [[ -n "${TRAIN_PID}" ]]; then
    TRAIN_ELAPSED=$(ps -p "$TRAIN_PID" -o etime= | tr -d ' ')
  fi

  GPU_SNAPSHOT=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true)
  LOG_MTIME="N/A"
  LOG_SIZE="0"
  if [[ -f "$LOG_FILE" ]]; then
    LOG_MTIME=$(date -d "@$(stat -c %Y "$LOG_FILE")" '+%F %T')
    LOG_SIZE=$(stat -c %s "$LOG_FILE")
  fi

  {
    echo "# STAGE3 Live Status"
    echo
    echo "- Updated: ${NOW}"
    echo "- Pipeline state: ${RUN_STATE}"
    echo "- Train pid: ${TRAIN_PID:-N/A}"
    echo "- Train elapsed: ${TRAIN_ELAPSED}"
    echo "- Log file: ${LOG_FILE}"
    echo "- Log mtime: ${LOG_MTIME}"
    echo "- Log size(bytes): ${LOG_SIZE}"
    echo
    echo "## GPU Snapshot"
    echo '```'
    echo "$GPU_SNAPSHOT"
    echo '```'
    echo
    echo "## Last 25 log lines"
    echo '```'
    if [[ -f "$LOG_FILE" ]]; then
      tail -n 25 "$LOG_FILE"
    else
      echo "(log file not created yet)"
    fi
    echo '```'
  } > "$OUT_FILE"

  sleep 300
done

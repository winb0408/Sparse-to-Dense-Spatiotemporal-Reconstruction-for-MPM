#!/usr/bin/env bash
# Score frozen checkpoints on the held-out animal.
#
# Deliberately separate from training: this is the only place a trained arm's
# dense reference is read, and it runs after the weights are frozen. Each fold
# goes to its own GPU; the three evaluations are independent.
#
#   bash spur_mpm/experiments/eval_trained.sh E002_S0 E002_S1
#
set -eu
cd "$(dirname "$0")/../.."

INDEX=data_local/dryad_cache/index.json
SPLIT=spur_mpm/experiments/E000_data_audit/split.json

for run_prefix in "$@"; do
  for fold in 0 1 2; do
    run="runs/${run_prefix}_fold${fold}"
    if [[ ! -f "$run/best.pt" ]]; then
      echo "skip ${run}: no best.pt"
      continue
    fi
    arm="${run_prefix#E???_}"          # E002_S0 -> S0
    arm="${arm%%_*}"
    CUDA_VISIBLE_DEVICES=$fold python -m spur_mpm.evaluate \
      --arm "$arm" --ckpt "$run/best.pt" --fold "$fold" --split test \
      --set data.index="$INDEX" --set data.split="$SPLIT" \
      --set data.patch_hw='[512,512]' --set train.num_workers=0 \
      --out "runs/${run_prefix}_fold${fold}_test" \
      > "/tmp/eval_${run_prefix}_fold${fold}.log" 2>&1 &
  done
  wait
  echo "evaluated ${run_prefix}"
done
echo ALL_EVALS_DONE

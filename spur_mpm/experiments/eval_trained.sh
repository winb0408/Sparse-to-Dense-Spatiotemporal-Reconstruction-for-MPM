#!/usr/bin/env bash
# Score frozen checkpoints on the held-out animal.
#
# Deliberately separate from training: this is the only place a trained arm's
# dense reference is read, and it runs after the weights are frozen. Each fold
# goes to its own GPU; the three evaluations are independent.
#
#   bash spur_mpm/experiments/eval_trained.sh E002_S0 E002_S1
#
# INPUT_SUPPORT=holdout keeps the training-time withholding and writes to
# runs_holdout/ instead, so the secondary protocol never mixes with the
# headline table. For self-supervised arms that regime is the only one with an
# E_hold number; for supervised arms it is the same run as "full".
set -eu
cd "$(dirname "$0")/../.."

INDEX=data_local/dryad_cache/index.json
SPLIT=spur_mpm/experiments/E000_data_audit/split.json
SUPPORT="${INPUT_SUPPORT:-full}"
# CONFIG names the experiment YAML whose sampling geometry the run was trained
# under. Without it the evaluation falls back to base.yaml -- fine for the k_y=2
# batches, wrong for anything that changes the lattice (E009 k_y=4).
CFG_ARGS=()
if [[ -n "${CONFIG:-}" ]]; then CFG_ARGS=(--config "$CONFIG"); fi
# CKPT selects which weights to score; default is the selected best.pt.
CKPT_NAME="${CKPT:-best.pt}"
if [[ "$SUPPORT" == "full" ]]; then OUT_ROOT=runs; else OUT_ROOT=runs_holdout; fi

for run_prefix in "$@"; do
  for fold in 0 1 2; do
    run="runs/${run_prefix}_fold${fold}"
    if [[ ! -f "$run/$CKPT_NAME" ]]; then
      echo "skip ${run}: no $CKPT_NAME"
      continue
    fi
    arm="${run_prefix#E???_}"          # E002_S0 -> S0
    arm="${arm%%_*}"
    # Variants of one architecture (E005_ratio010) carry no arm definition of
    # their own; ARM=SS1 says which feature/loss recipe to score them with.
    arm="${ARM:-$arm}"
    CUDA_VISIBLE_DEVICES=$fold python -m spur_mpm.evaluate \
      "${CFG_ARGS[@]}" --arm "$arm" --ckpt "$run/$CKPT_NAME" --fold "$fold" --split test \
      --set data.index="$INDEX" --set data.split="$SPLIT" \
      --set data.patch_hw='[512,512]' --set train.num_workers=0 \
      --set eval.input_support="$SUPPORT" \
      --out "${OUT_ROOT}/${run_prefix}_fold${fold}_test${OUT_SUFFIX:-}" \
      > "/tmp/eval_${run_prefix}_fold${fold}_${SUPPORT}.log" 2>&1 &
  done
  wait
  echo "evaluated ${run_prefix} (${SUPPORT})"
done
echo ALL_EVALS_DONE

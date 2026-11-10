#!/usr/bin/env bash
# Re-score the checkpoints that predate the Z-boundary fix, so the review can
# see whether that fix moved the numbers it was blamed for (PR #1, item B4).
#
# Why this is not the same thing as the E021 ladder: E021 retrained every arm
# under the fixed code on the current fixed cohort, so it answers "what is true
# now". This script answers a different question -- "how much of the withdrawn
# table was the bug?" -- by running the *old* weights through the *new* window
# construction on the *old* split they were trained on. Nothing here is a
# headline number; the old split has the selection history documented in the
# runbook §6, and E002/E009 are not the paper's cohort.
#
#   bash spur_mpm/experiments/reval_pre_fix_checkpoints.sh
#
# DEVICE defaults to 2 and only one evaluation runs at a time. That is
# deliberate: the card is shared, our footprint is ~360 MB, and a serialised
# eval neither evicts the resident training nor produces a latency claim that
# would be wrong on a contended card.
set -eu
cd "$(dirname "$0")/../.."

DCACHE_INDEX=data_local/dryad_cache/index.json
SPLIT=spur_mpm/experiments/E000_data_audit/split.json
OUT="${OUT:-runs/E023_phase1/reval}"
DEVICE="${DEVICE:-2}"
E009_CFG=spur_mpm/experiments/E009_sampling_y_k4.yaml
mkdir -p "$OUT"

# one(arm, fold, optional config, optional checkpoint)
one() {
  local arm="$1" fold="$2" cfg="${3:-}" ckpt="${4:-}"
  local name="${5:-${arm}}"
  local args=(--arm "$arm" --fold "$fold" --split test
              --set data.index="$DCACHE_INDEX" --set data.split="$SPLIT"
              --set data.patch_hw='[512,512]' --set train.num_workers=0
              --set eval.input_support=full
              --out "$OUT/${name}_fold${fold}_test")
  [[ -n "$cfg" ]]  && args+=(--config "$cfg")
  [[ -n "$ckpt" ]] && args+=(--ckpt "$ckpt")
  echo "-> $name fold$fold"
  CUDA_VISIBLE_DEVICES="$DEVICE" python -m spur_mpm.evaluate "${args[@]}" \
    > "$OUT/${name}_fold${fold}.log" 2>&1
  tail -2 "$OUT/${name}_fold${fold}.log"
}

for arm in S0 S1; do                  # E002: the k_y=2 cohort
  for fold in 0 1 2; do
    one "$arm" "$fold" "" "runs/E002_${arm}_fold${fold}/best.pt" "E002_${arm}"
  done
done

for fold in 0 1 2; do                 # A0 under k_y=4 needs no weights
  one A0 "$fold" "$E009_CFG" "" "E009_A0"
done
for fold in 0 1 2; do
  one S1 "$fold" "$E009_CFG" "runs/E009_S1_fold${fold}/best.pt" "E009_S1"
done

echo "wrote $OUT"

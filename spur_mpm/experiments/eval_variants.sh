#!/usr/bin/env bash
# Secondary evaluation protocols, kept out of the headline table.
#
#   bash spur_mpm/experiments/eval_variants.sh E003_SS0 E003_SS1
#
# Writes to runs_dc/ (hard consistency projection) and runs_holdout/ (the
# training-time withholding), never into runs/, so a secondary number cannot
# end up in the primary comparison by accident.
#
# ``eval.holdout_baselines=true`` is passed unconditionally: under
# ``input_support=holdout`` every arm must see the same half of the rows,
# otherwise a parameter-free arm silently gets twice the input (E010 result 0,
# 29.1583 dB for an "A0 under holdout" that was really fed the full mask). The
# flag is inert under ``input_support=full``, so it cannot leak into the
# headline protocol.
set -eu
cd "$(dirname "$0")/../.."

INDEX=data_local/dryad_cache/index.json
SPLIT=spur_mpm/experiments/E000_data_audit/split.json
SUPPORT="${INPUT_SUPPORT:-full}"
PROJECTION="${PROJECTION:-hard}"

if [[ "$PROJECTION" == "none" && "$SUPPORT" == "full" ]]; then
  echo "refusing to duplicate the headline table into runs/" >&2
  exit 1
fi
if [[ "$SUPPORT" == "holdout" ]]; then OUT_ROOT=runs_holdout; else OUT_ROOT=runs_dc; fi

for run_prefix in "$@"; do
  for fold in 0 1 2; do
    run="runs/${run_prefix}_fold${fold}"
    # A run label is not necessarily an arm name (E005_dc1 runs the SS1 arm with
    # dc_weight=1.0), so let the caller state the arm when the label is opaque.
    arm="${run_prefix#E???_}"
    arm="${arm%%_*}"
    arm="${ARM:-$arm}"
    ckpt="$run/best.pt"
    if [[ ! -f "$ckpt" ]]; then
      # Parameter-free arms have no checkpoint, but they are exactly the ones
      # the same-input comparison needs (A0 is the baseline MMS is judged
      # against). Only allow the ckpt-less path when the arm is one of them.
      case "$arm" in
        A0|A1) ckpt="" ;;
        *) echo "skip ${run} (no best.pt)"; continue ;;
      esac
    fi
    CUDA_VISIBLE_DEVICES=$fold python -m spur_mpm.evaluate \
      --arm "$arm" ${ckpt:+--ckpt "$ckpt"} --fold "$fold" --split test \
      --set data.index="$INDEX" --set data.split="$SPLIT" \
      --set data.patch_hw='[512,512]' --set train.num_workers=0 \
      --set eval.input_support="$SUPPORT" \
      --set eval.consistency_projection="$PROJECTION" \
      --set eval.holdout_baselines=true \
      --out "${OUT_ROOT}/${run_prefix}_fold${fold}_test" \
      > "/tmp/ev_${run_prefix}_f${fold}_${SUPPORT}_${PROJECTION}.log" 2>&1 &
  done
  wait
  echo "evaluated ${run_prefix} (${SUPPORT}/${PROJECTION})"
done
echo ALL_VARIANTS_DONE

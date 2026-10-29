#!/usr/bin/env bash
# Score E011 (S1h) and E012 / E012b (SS1c) on the held-out animal.
#
# One protocol: eval.input_support=full, i.e. every acquired row at test time.
# That is the point of both experiments -- they change what an arm is given while
# *training*, and this protocol is the one where all four rows of the table are
# scored on the same measured information:
#
#   S1   trains on 50% of the volume, sees 50% at test
#   S1h  trains on 25% (the rows SS1 gets) with a dense target, sees 50% at test
#   SS1  trains on 25% + 25% withheld measurements,                sees 50% at test
#   SS1c trains as a pair: branch A 25% / branch B 25%, disjoint,  sees 50% at test
#
# A "holdout" protocol is deliberately absent for S1h: withholding only ever
# affects arms whose autonomy is "self", so scoring a supervised arm there would
# silently hand it the full mask and produce a row that looks like a same-input
# control without being one. The dual arm is in the same position for a different
# reason -- it always splits, so both protocols coincide for it.
#
# S1 and SS1 are re-scored here too, so every row of the table comes from one
# script and one protocol.
#
# Reading the output: S1h is scored with --arm S1 (it *is* the S1 recipe, only
# the input support differs), so its summary.json says arm=S1. Build the table
# from the run *directories* -- E011_S1h_*, E012_SS1c_*, E002_S1_*, E003_SS1_* --
# not from the arm field, or the two S1 rows collide (collect_results keys by
# arm and would silently drop one).
set -u
cd "$(dirname "$0")/../.."

CFG_E011=spur_mpm/configs/experiments/E011_same_input_supervised_y_k2.yaml
CFG_E012=spur_mpm/configs/experiments/E012_complementary_y_k2.yaml
CFG_E012B=spur_mpm/configs/experiments/E012b_complementary_agree_y_k2.yaml
INDEX=data_local/dryad_cache/index.json
SPLIT=spur_mpm/experiments/E000_data_audit/split.json
OUT=runs_e011

score () {   # score <run-prefix> <arm> [config]
  local prefix=$1 arm=$2 cfg=${3:-}
  local cfg_args=()
  [[ -n "$cfg" ]] && cfg_args=(--config "$cfg")
  for fold in 0 1 2; do
    local run="runs/${prefix}_fold${fold}"
    if [[ ! -f "$run/best.pt" ]]; then
      echo "skip ${run}: no best.pt"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$fold python -m spur_mpm.evaluate \
      "${cfg_args[@]}" --arm "$arm" --ckpt "$run/best.pt" --fold "$fold" --split test \
      --set data.index="$INDEX" --set data.split="$SPLIT" \
      --set data.patch_hw='[512,512]' --set train.num_workers=0 \
      --set eval.input_support=full \
      --out "${OUT}/${prefix}_fold${fold}_test_full" \
      > "/tmp/eval_${prefix}_fold${fold}_full.log" 2>&1 &
  done
  wait
  echo "evaluated ${prefix} (full)"
}

score E011_S1h   S1   "$CFG_E011"
score E012_SS1c  SS1c "$CFG_E012"
score E012b_SS1c SS1c "$CFG_E012B"
score E002_S1    S1
score E003_SS1   SS1
echo ALL_EVALS_DONE

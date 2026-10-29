#!/usr/bin/env bash
# E011 + E012 -- the two answers to "the self-supervised arm sees half the data".
#
# E011  S1h : supervised arm on exactly the rows SS1 gets (dense target). Sizes
#             the part of the supervised win that is just extra input rows.
# E012  SS1c: dual-branch complementary MMS. Two branches on disjoint halves,
#             each reconstructing the other's rows, so the pair uses 100% of the
#             acquisition and still needs no dense reference.
# E012b SS1c with loss.dual_agree=0.3 -- the same design plus branch agreement
#             where neither branch was given anything.
#
# Everything is scored later by eval_e011_e012.sh at eval.input_support=full, so
# all four arms (S1, S1h, SS1, SS1c) see the whole acquisition at test time.
#
# Three waves, one job per GPU, GPUs 0/1/2: GPU 3 belongs to another user.
set -u
cd /ssd1/wjb/SPUR-MPM
INDEX=data_local/dryad_cache/index.json
SPLIT=spur_mpm/experiments/E000_data_audit/split.json
FOLD_GPU=(0 1 2)
fail=0

launch () {   # launch <run-name> <config> <arm> <fold> <extra...>
  local name=$1 cfg=$2 arm=$3 fold=$4 gpu=${FOLD_GPU[$4]}
  shift 4
  CUDA_VISIBLE_DEVICES=$gpu python -m spur_mpm.train \
    --config "$cfg" --arm "$arm" --fold "$fold" --name "$name" \
    --set data.index="$INDEX" --set data.split="$SPLIT" \
    --set data.patch_hw='[256,256]' --set train.num_workers=4 \
    "$@" > "/tmp/${name}.log" 2>&1 &
  echo "launched ${name} on GPU${gpu} (pid $!)"
}

wave () {     # wave <config> <arm> <name-prefix> [extra...]
  local cfg=$1 arm=$2 prefix=$3
  shift 3
  local pids=()
  for f in 0 1 2; do
    launch "${prefix}_fold${f}" "$cfg" "$arm" "$f" "$@"
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  echo "[$(date +%H:%M:%S)] wave ${prefix} done (${fail} failure(s) so far)"
}

wave spur_mpm/configs/experiments/E011_same_input_supervised_y_k2.yaml S1  E011_S1h
wave spur_mpm/configs/experiments/E012_complementary_y_k2.yaml      SS1c E012_SS1c
wave spur_mpm/configs/experiments/E012b_complementary_agree_y_k2.yaml SS1c E012b_SS1c

echo "E011/E012 training finished, ${fail} run(s) failed"
echo ALL_DONE

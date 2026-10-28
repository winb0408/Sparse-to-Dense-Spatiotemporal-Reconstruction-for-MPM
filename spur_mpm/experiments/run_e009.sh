#!/usr/bin/env bash
# E009 -- sampling geometry sweep at k_y = 4 (three missing rows between
# measured ones), same backbone, same budget, same folds.
#
# Question: E005 showed the self-supervised arm with hard measurement
# consistency respects its inputs but still loses 2.0 dB to interpolation, and
# the whole deficit sits in the missing rows. At k_y=2 those rows are one away
# from a measured row, which is interpolation's best case; at k_y=4 the gap is
# three rows, so the same comparison becomes informative.
#
# Two waves, never more than two jobs per GPU:
#   wave 1 : S1 (supervised ceiling) + SS1 (E003 recipe, unchanged), folds 0-2
#   wave 2 : SS1dc1 (E005's fix, dc_weight=1.0), folds 0-2
#
# GPUs: an external job appeared on GPU 3 while this batch was prepared, so the
# wave uses 0/1/2 and shares GPU 2 with that user's much smaller process. Check
# nvidia-smi before re-running.
set -u
cd /ssd1/wjb/SPUR-MPM
CFG=spur_mpm/configs/experiments/E009_sampling_y_k4.yaml
FOLD_GPU=(0 1 2)

launch () {   # launch <name> <arm> <fold> <gpu> <extra...>
  local name=$1 arm=$2 fold=$3 gpu=$4
  shift 4
  CUDA_VISIBLE_DEVICES=$gpu python -m spur_mpm.train \
    --config $CFG --arm "$arm" --fold "$fold" --name "E009_${name}_fold${fold}" \
    --set data.index=data_local/dryad_cache/index.json \
    --set data.split=spur_mpm/experiments/E000_data_audit/split.json \
    --set data.patch_hw='[256,256]' --set train.num_workers=4 \
    "$@" > "/tmp/E009_${name}_fold${fold}.log" 2>&1 &
  echo "launched E009_${name}_fold${fold} on GPU${gpu} (pid $!)"
}

fail=0
wait_all () { for p in "$@"; do wait "$p" || fail=$((fail+1)); done; }

pids=()
for name_arm in S1:S1 SS1:SS1; do
  name=${name_arm%%:*}; arm=${name_arm#*:}
  for f in 0 1 2; do
    launch "$name" "$arm" "$f" "${FOLD_GPU[$f]}"
    pids+=($!)
  done
done
wait_all "${pids[@]}"

pids=()
for f in 0 1 2; do
  launch SS1dc1 SS1 "$f" "${FOLD_GPU[$f]}" --set loss.dc_weight=1.0
  pids+=($!)
done
wait_all "${pids[@]}"

echo "E009 training finished, $fail run(s) failed"
echo ALL_DONE

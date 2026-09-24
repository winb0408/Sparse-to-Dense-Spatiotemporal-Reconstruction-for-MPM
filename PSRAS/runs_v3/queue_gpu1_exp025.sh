#!/bin/bash
# EXP-025 GPU1 chain (rebalanced 2026-09-24: GPU2 is shared with a foreign
# 24GB job, so most YZ runs moved here). SY_s0 may already be running
# (orphaned from the first launch) -- wait for it, then continue.
cd /data2/wjb/threePM/PSRAS
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu1_exp025.log
MIX_Y="y:2,4,8"
MIX_YZ="yz2:2x2,2x4,2x8,4x2,4x4,8x2"
wait_for () {
  while [ ! -f "runs_v3/$1/final_val.json" ]; do sleep 300; done
}
run () {
  name=$1; seed=$2; mix=$3; extra=$4
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] SKIP $name (done)" >> $LOG; return
  fi
  echo "[$(date +%m%d-%H:%M)] START $name" >> $LOG
  CUDA_VISIBLE_DEVICES=1 $PY scripts/train.py --preset b4f --model univ4 \
    --steps 7950 --seed $seed --regime_mix "$mix" $extra --name $name \
    > runs_v3/$name.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END $name (exit $?)" >> $LOG
}
echo "[$(date +%m%d-%H:%M)] (rebalanced chain v2: waiting for SY_s0 if running)" >> $LOG
wait_for UNIV4_SY_s0
run UNIV4_SY_s1 1 "$MIX_Y"
run UNIV4_SY_s2 2 "$MIX_Y"
run UNIV4_SYZ_s1 1 "$MIX_YZ"
run UNIV4_SYZ_s2 2 "$MIX_YZ"
run UNIV4_SYZ_noAZ_s0 0 "$MIX_YZ" "--no_az"
run UNIV4_SYZ_noAZ_s2 2 "$MIX_YZ" "--no_az"
echo "[$(date +%m%d-%H:%M)] GPU1 EXP-025 chain DONE" >> $LOG

#!/bin/bash
# EXP-025 GPU2 chain (rebalanced: GPU2 shared with a foreign job -> only
# SYZ_s0 (already running) + one noAZ run stay here).
cd /data2/wjb/threePM/PSRAS
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu2_exp025.log
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
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train.py --preset b4f --model univ4 \
    --steps 7950 --seed $seed --regime_mix "$mix" $extra --name $name \
    > runs_v3/$name.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END $name (exit $?)" >> $LOG
}
echo "[$(date +%m%d-%H:%M)] (rebalanced chain v2: waiting for SYZ_s0)" >> $LOG
wait_for UNIV4_SYZ_s0
run UNIV4_SYZ_noAZ_s1 1 "$MIX_YZ" "--no_az"
echo "[$(date +%m%d-%H:%M)] GPU2 EXP-025 chain DONE" >> $LOG

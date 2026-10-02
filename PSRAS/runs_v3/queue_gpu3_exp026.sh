#!/bin/bash
# EXP-026 GPU3 chain (2026-09-26): U15 unified 15-point x3 seeds.
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu3_exp026.log
MIX_ALL="y:2,4,8;yz2:2x2,2x4,2x8,4x2,4x4,8x2;xyz2:2x2,2x4,2x8,4x2,4x4,8x2"
run () {
  name=$1; seed=$2
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] SKIP $name (done)" >> $LOG; return
  fi
  echo "[$(date +%m%d-%H:%M)] START $name" >> $LOG
  CUDA_VISIBLE_DEVICES=3 $PY scripts/train.py --preset b4f --model univ4 --no_az \
    --steps 7950 --seed $seed --regime_mix "$MIX_ALL" --name $name \
    > runs_v3/$name.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END $name (exit $?)" >> $LOG
}
run UNIV4_U15_s0 0
run UNIV4_U15_s1 1
run UNIV4_U15_s2 2
echo "[$(date +%m%d-%H:%M)] GPU3 EXP-026 chain DONE" >> $LOG

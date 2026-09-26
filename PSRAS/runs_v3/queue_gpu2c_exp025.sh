#!/bin/bash
# EXP-025 GPU2c chain (2026-09-25 22:40): rebuilt after ENOSPC crash of
# SYZ_noAZ_s2 at step 7500 (root /tmp full). TMPDIR moved to /data2.
# Covers SYZ_s1 -> SYZ_s2 on GPU2.
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu2c_exp025.log
MIX_YZ="yz2:2x2,2x4,2x8,4x2,4x4,8x2"
run () {
  name=$1; seed=$2; extra=$3
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] SKIP $name (done)" >> $LOG; return
  fi
  echo "[$(date +%m%d-%H:%M)] START $name" >> $LOG
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train.py --preset b4f --model univ4 \
    --steps 7950 --seed $seed --regime_mix "$MIX_YZ" $extra --name $name \
    > runs_v3/$name.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END $name (exit $?)" >> $LOG
}
run UNIV4_SYZ_s1 1 ""
run UNIV4_SYZ_s2 2 ""
echo "[$(date +%m%d-%H:%M)] GPU2c EXP-025 chain DONE" >> $LOG

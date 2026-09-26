#!/bin/bash
# EXP-025 GPU3b chain (2026-09-25 22:40): rerun SYZ_noAZ_s2 (ENOSPC crash at
# step 7500 on root /tmp). TMPDIR moved to /data2. GPU3 is otherwise idle.
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu3b_exp025.log
if [ -f "runs_v3/UNIV4_SYZ_noAZ_s2/final_val.json" ]; then
  echo "[$(date +%m%d-%H:%M)] SKIP UNIV4_SYZ_noAZ_s2 (done)" >> $LOG
else
  echo "[$(date +%m%d-%H:%M)] START UNIV4_SYZ_noAZ_s2" >> $LOG
  CUDA_VISIBLE_DEVICES=3 $PY scripts/train.py --preset b4f --model univ4 \
    --steps 7950 --seed 2 --regime_mix "yz2:2x2,2x4,2x8,4x2,4x4,8x2" --no_az \
    --name UNIV4_SYZ_noAZ_s2 > runs_v3/UNIV4_SYZ_noAZ_s2.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END UNIV4_SYZ_noAZ_s2 (exit $?)" >> $LOG
fi
echo "[$(date +%m%d-%H:%M)] GPU3b EXP-025 chain DONE" >> $LOG

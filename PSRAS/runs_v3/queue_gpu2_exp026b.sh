#!/bin/bash
# EXP-026 GPU2b chain (2026-10-03 06:55): SXYZ_s1 OOM'd at step 450 (foreign
# 26GB process, now gone). Rerun after SXYZ_s2 finishes on GPU2.
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu2_exp026.log
while [ ! -f runs_v3/UNIV4_SXYZ_s2/final_val.json ]; do sleep 300; done
if [ ! -f runs_v3/UNIV4_SXYZ_s1/final_val.json ]; then
  echo "[$(date +%m%d-%H:%M)] START UNIV4_SXYZ_s1 (GPU2 rerun)" >> $LOG
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train.py --preset b4f --model univ4 --no_az \
    --steps 7950 --seed 1 --regime_mix "xyz2:2x2,2x4,2x8,4x2,4x4,8x2" \
    --name UNIV4_SXYZ_s1 > runs_v3/UNIV4_SXYZ_s1.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END UNIV4_SXYZ_s1 (exit $?)" >> $LOG
fi
echo "[$(date +%m%d-%H:%M)] GPU2b EXP-026 chain DONE" >> $LOG

#!/bin/bash
# EXP-026 U15_s1 rerun (2026-10-03 03:55): OOM'd at step 450 when a foreign
# 26GB process landed on GPU3. Waits for U15_s2 to finish, then waits for a
# GPU (2 or 3) with enough free memory before relaunching.
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu3_exp026.log
while [ ! -f runs_v3/UNIV4_U15_s2/final_val.json ]; do sleep 300; done
GPU=""
while [ -z "$GPU" ]; do
  for g in 2 3; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$used" -lt 8000 ]; then GPU=$g; break; fi
  done
  [ -z "$GPU" ] && sleep 600
done
echo "[$(date +%m%d-%H:%M)] RERUN UNIV4_U15_s1 on GPU$GPU" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --preset b4f --model univ4 --no_az \
  --steps 7950 --seed 1 --regime_mix "y:2,4,8;yz2:2x2,2x4,2x8,4x2,4x4,8x2;xyz2:2x2,2x4,2x8,4x2,4x4,8x2" \
  --name UNIV4_U15_s1 > runs_v3/UNIV4_U15_s1.out 2>&1
echo "[$(date +%m%d-%H:%M)] END UNIV4_U15_s1 rerun (exit $?)" >> $LOG
echo "[$(date +%m%d-%H:%M)] U15_s1 rerun chain DONE" >> $LOG

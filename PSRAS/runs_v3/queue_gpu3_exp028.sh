#!/bin/bash
# EXP-028 regime-weighted (y:yz2:xyz2 = 1:1:2) U15W retrain, 3 seeds serial on GPU3.
# C payoff: crashed runs resume from resume.pt instead of full restart.
export CUDA_VISIBLE_DEVICES=3
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
cd /data2/wjb/threePM/PSRAS
LOG=runs_v3/queue_gpu3_exp028.log
MIX="y:2,4,8;yz2:2x2,2x4,2x8,4x2,4x4,8x2;xyz2:2x2,2x4,2x8,4x2,4x4,8x2"
run() {
  local name=$1 seed=$2
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] SKIP $name (done)" >> $LOG; return
  fi
  local extra=""
  if [ -f "runs_v3/$name/resume.pt" ]; then
    extra="--resume"
    echo "[$(date +%m%d-%H:%M)] RESUME $name from resume.pt" >> $LOG
  else
    echo "[$(date +%m%d-%H:%M)] START $name" >> $LOG
  fi
  CUDA_VISIBLE_DEVICES=3 $PY scripts/train.py --preset b4f --model univ4 --no_az \
    --steps 7950 --seed $seed --regime_mix "$MIX" \
    --regime_weights "y:1,yz2:1,xyz2:2" $extra --name $name \
    >> runs_v3/$name.out 2>&1
  # completion judged by final_val.json existence (DEC-027 lesson 3)
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] END $name (final_val present)" >> $LOG
  else
    echo "[$(date +%m%d-%H:%M)] CRASH $name (rerun chain to resume)" >> $LOG
  fi
}
run UNIV4_U15W_s0 0
run UNIV4_U15W_s1 1
run UNIV4_U15W_s2 2
echo "[$(date +%m%d-%H:%M)] GPU3 EXP-028 chain DONE" >> $LOG

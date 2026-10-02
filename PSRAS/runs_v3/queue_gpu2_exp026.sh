#!/bin/bash
# EXP-026 GPU2 chain (2026-09-26): SXYZ specialist x3 seeds.
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu2_exp026.log
MIX_XYZ="xyz2:2x2,2x4,2x8,4x2,4x4,8x2"
run () {
  name=$1; seed=$2
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] SKIP $name (done)" >> $LOG; return
  fi
  echo "[$(date +%m%d-%H:%M)] START $name" >> $LOG
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train.py --preset b4f --model univ4 --no_az \
    --steps 7950 --seed $seed --regime_mix "$MIX_XYZ" --name $name \
    > runs_v3/$name.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END $name (exit $?)" >> $LOG
}
run UNIV4_SXYZ_s0 0
run UNIV4_SXYZ_s1 1
run UNIV4_SXYZ_s2 2
echo "[$(date +%m%d-%H:%M)] GPU2 EXP-026 chain DONE" >> $LOG

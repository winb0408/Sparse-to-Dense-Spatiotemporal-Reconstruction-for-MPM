#!/bin/bash
# EXP-025 Phase 4 Unified Framework - GPU3 chain (unified, 3 seeds + noAZ)
cd /data2/wjb/threePM/PSRAS
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu3_exp025.log
MIX_ALL="y:2,4,8;yz2:2x2,2x4,2x8,4x2,4x4,8x2"
run () {
  name=$1; seed=$2; mix=$3; extra=$4
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] SKIP $name (done)" >> $LOG; return
  fi
  echo "[$(date +%m%d-%H:%M)] START $name" >> $LOG
  CUDA_VISIBLE_DEVICES=3 $PY scripts/train.py --preset b4f --model univ4 \
    --steps 7950 --seed $seed --regime_mix "$mix" $extra --name $name \
    > runs_v3/$name.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END $name (exit $?)" >> $LOG
}
run UNIV4_U_s0 0 "$MIX_ALL"
run UNIV4_U_noAZ_s0 0 "$MIX_ALL" "--no_az"
run UNIV4_U_s1 1 "$MIX_ALL"
run UNIV4_U_noAZ_s1 1 "$MIX_ALL" "--no_az"
run UNIV4_U_s2 2 "$MIX_ALL"
run UNIV4_U_noAZ_s2 2 "$MIX_ALL" "--no_az"
echo "[$(date +%m%d-%H:%M)] GPU3 EXP-025 chain DONE" >> $LOG

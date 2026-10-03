#!/bin/bash
# EXP-026 GPU3c chain (2026-10-03 06:55): U15_s1 and U15_s2 both OOM'd at
# step ~450 when a foreign 26GB process shared GPU3 (now gone). Rerun both
# sequentially on GPU3 with an idle-GPU guard (skip if already running).
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
LOG=runs_v3/queue_gpu3_exp026.log
wait_idle () {  # GPU3 must have <8GB used and no foreign process
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    [ "$used" -lt 8000 ] && return
    sleep 300
  done
}
run () {
  name=$1; seed=$2; mix=$3
  if [ -f "runs_v3/$name/final_val.json" ]; then
    echo "[$(date +%m%d-%H:%M)] SKIP $name (done)" >> $LOG; return
  fi
  wait_idle
  echo "[$(date +%m%d-%H:%M)] START $name (GPU3)" >> $LOG
  CUDA_VISIBLE_DEVICES=3 $PY scripts/train.py --preset b4f --model univ4 --no_az \
    --steps 7950 --seed $seed --regime_mix "$mix" --name $name \
    > runs_v3/$name.out 2>&1
  echo "[$(date +%m%d-%H:%M)] END $name (exit $?)" >> $LOG
}
MIX_ALL="y:2,4,8;yz2:2x2,2x4,2x8,4x2,4x4,8x2;xyz2:2x2,2x4,2x8,4x2,4x4,8x2"
run UNIV4_U15_s1 1 "$MIX_ALL"
run UNIV4_U15_s2 2 "$MIX_ALL"
echo "[$(date +%m%d-%H:%M)] GPU3c EXP-026 chain DONE" >> $LOG

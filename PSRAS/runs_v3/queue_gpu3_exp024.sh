#!/bin/bash
# EXP-024 Phase 2 YZ mainline (DEC-026 protocol v4, pre-registered)
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp_train
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=3
log() { echo "[$(date +%m%d-%H:%M)] $*" >> runs_v3/queue_gpu3_exp024.log; }
run() {
  local name=$1; shift 1
  if [ -f "runs_v3/${name}/log.csv" ] && [ $(tail -1 "runs_v3/${name}/log.csv" | cut -d, -f1) -ge 7950 ]; then
    log "SKIP $name (already complete)"
    return 0
  fi
  log "START $name"
  /data2/wjb/anaconda3/envs/threePM/bin/python scripts/train.py "$@" --name "$name" > "runs_v3/${name}.out" 2>&1
  log "END $name (exit $?)"
}
run YZ_2x8_b4i --preset b4i --regime 2.5d --mode yz2 --K 2 --kz 8 --steps 8000
run YZ_2x8_b2_s1 --preset b2 --regime 2.5d --mode yz2 --K 2 --kz 8 --steps 8000 --seed 1
run YZ_2x8_b1_s2 --preset b1 --regime 2.5d --mode yz2 --K 2 --kz 8 --steps 8000 --seed 2
run YZ_2x8_b4f_s2 --preset b4f --regime 2.5d --mode yz2 --K 2 --kz 8 --steps 8000 --seed 2
run YZ_2x4_b4i --preset b4i --regime 2.5d --mode yz2 --K 2 --kz 4 --steps 8000
run YZ_4x4_b2 --preset b2 --regime 2.5d --mode yz2 --K 4 --kz 4 --steps 8000
run YZ_4x4_b1_s1 --preset b1 --regime 2.5d --mode yz2 --K 4 --kz 4 --steps 8000 --seed 1
run YZ_4x4_b4f_s1 --preset b4f --regime 2.5d --mode yz2 --K 4 --kz 4 --steps 8000 --seed 1
run YZ_4x4_b4i_s2 --preset b4i --regime 2.5d --mode yz2 --K 4 --kz 4 --steps 8000 --seed 2
run YZ_2x2_b2 --preset b2 --regime 2.5d --mode yz2 --K 2 --kz 2 --steps 8000
run YZ_2x2_b4f --preset b4f --regime 2.5d --mode yz2 --K 2 --kz 2 --steps 8000
run YZ_4x2_b2 --preset b2 --regime 2.5d --mode yz2 --K 4 --kz 2 --steps 8000
run YZ_8x2_b1 --preset b1 --regime 2.5d --mode yz2 --K 8 --kz 2 --steps 8000
run YZ_8x2_b4f --preset b4f --regime 2.5d --mode yz2 --K 8 --kz 2 --steps 8000
log "GPU3 EXP-024 chain DONE"

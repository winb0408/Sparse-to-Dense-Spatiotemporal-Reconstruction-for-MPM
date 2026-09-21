#!/bin/bash
# EXP-022 Phase 1 Y-sparse corrected baseline (DEC-025 protocol v4, pre-registered)
# 12 runs per GPU: b{1,2,3} x K{2,4,8,16} x seeds -- split by K so each chain
# covers all arms/seed combos; ~1.5h per 2d run.
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp_train
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=1
log() { echo "[$(date +%m%d-%H:%M)] $*" >> runs_v3/queue_gpu1_exp022.log; }
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
run Y_K2_B1 --preset b1 --regime 2d --mode y --K 2 --steps 8000
run Y_K2_B2 --preset b2 --regime 2d --mode y --K 2 --steps 8000 --lam_wiener 3e-6
run Y_K2_B3 --preset b3 --regime 2d --mode y --K 2 --steps 8000 --lam_wiener 3e-6
run Y_K16_B1_s0 --preset b1 --regime 2d --mode y --K 16 --steps 8000 --seed 0
run Y_K16_B2_s0 --preset b2 --regime 2d --mode y --K 16 --steps 8000 --lam_wiener 3e-6 --seed 0
run Y_K16_B3_s0 --preset b3 --regime 2d --mode y --K 16 --steps 8000 --lam_wiener 3e-6 --seed 0
run Y_K16_B1_s0 --preset b1 --regime 2d --mode y --K 16 --steps 8000 --seed 0
run Y_K16_B2_s0 --preset b2 --regime 2d --mode y --K 16 --steps 8000 --lam_wiener 3e-6 --seed 0
run Y_K16_B3_s0 --preset b3 --regime 2d --mode y --K 16 --steps 8000 --lam_wiener 3e-6 --seed 0
run Y_K16_B1_s0 --preset b1 --regime 2d --mode y --K 16 --steps 8000 --seed 0
run Y_K16_B2_s0 --preset b2 --regime 2d --mode y --K 16 --steps 8000 --lam_wiener 3e-6 --seed 0
run Y_K16_B3_s0 --preset b3 --regime 2d --mode y --K 16 --steps 8000 --lam_wiener 3e-6 --seed 0
log "GPU1 EXP-022 chain DONE"

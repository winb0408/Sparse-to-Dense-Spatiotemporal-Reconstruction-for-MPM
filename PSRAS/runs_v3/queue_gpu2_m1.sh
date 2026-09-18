#!/bin/bash
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp_train
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=2
log() { echo "[$(date +%m%d-%H:%M)] $*" >> runs_v3/queue_gpu2_m1.log; }
run() {
  local name=$1; shift 1
  log "START $name"
  /data2/wjb/anaconda3/envs/threePM/bin/python scripts/train.py "$@" --name "$name" > "runs_v3/${name}.out" 2>&1
  log "END $name (exit $?)"
}
run b2_2p5d_z_K8_p8k_nopsf --preset b2 --regime 2.5d --mode z --K 8 --steps 8000 --psf_xy 0 --psf_z 0
run b3_2p5d_z_K8_p8k_nopsf --preset b3 --regime 2.5d --mode z --K 8 --steps 8000 --psf_xy 0 --psf_z 0 --lam_wiener 3e-6
log "GPU2 M1 chain DONE"

#!/bin/bash
# EXP-020 GPU2 repair chain: the 3 runs crashed by the root-disk-full event
# (12:50). TMPDIR redirected to /data2 so DataLoader IPC is immune to /tmp
# pressure. Order: cheap vswin 2d first, then rvrt xyz, then sunetr xyz.
cd /data2/wjb/threePM/PSRAS
mkdir -p /data2/wjb/tmp_train
export TMPDIR=/data2/wjb/tmp_train
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
export CUDA_VISIBLE_DEVICES=2
run() {
  echo "[$(date +%m%d-%H:%M)] START $1" >> runs_v3/queue_gpu2_repair.log
  shift
  $PY scripts/train.py "$@" >> runs_v3/queue_gpu2_repair.log 2>&1
  echo "[$(date +%m%d-%H:%M)] END   (exit $?)" >> runs_v3/queue_gpu2_repair.log
}
run b1_2d_xy_K8_p8k_vswin --preset b1 --regime 2d --mode xy --K 8 --steps 8000 --batch 8 --model vswin --name b1_2d_xy_K8_p8k_vswin
run b3_2p5d_xyz_K8_p8k_lam3e6_rvrt --preset b3 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model rvrt --name b3_2p5d_xyz_K8_p8k_lam3e6_rvrt
run b3_2p5d_xyz_K8_p8k_lam3e6_sunetr --preset b3 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model sunetr --name b3_2p5d_xyz_K8_p8k_lam3e6_sunetr
echo "[$(date +%m%d-%H:%M)] GPU2 repair chain DONE" >> runs_v3/queue_gpu2_repair.log

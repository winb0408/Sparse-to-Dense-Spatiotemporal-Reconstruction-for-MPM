#!/bin/bash
# EXP-020 GPU2 chain: Video Swin (batch 8 x accum 1)
cd /data2/wjb/threePM/PSRAS
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
export CUDA_VISIBLE_DEVICES=2
run() {
  echo "[$(date +%m%d-%H:%M)] START $1" >> runs_v3/queue_gpu2.log
  shift
  $PY scripts/train.py "$@" >> runs_v3/queue_gpu2.log 2>&1
  echo "[$(date +%m%d-%H:%M)] END   (exit $?)" >> runs_v3/queue_gpu2.log
}
run b3_2p5d_z_K8_p8k_lam3e6_vswin --preset b3 --regime 2.5d --mode z --K 8 --steps 8000 --batch 8 --lam_wiener 3e-6 --model vswin --name b3_2p5d_z_K8_p8k_lam3e6_vswin
run b1_2p5d_z_K8_p8k_vswin --preset b1 --regime 2.5d --mode z --K 8 --steps 8000 --batch 8 --model vswin --name b1_2p5d_z_K8_p8k_vswin
run b3_2p5d_xyz_K8_p8k_lam3e6_vswin --preset b3 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 8 --lam_wiener 3e-6 --model vswin --name b3_2p5d_xyz_K8_p8k_lam3e6_vswin
run b1_2p5d_xyz_K8_p8k_vswin --preset b1 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 8 --model vswin --name b1_2p5d_xyz_K8_p8k_vswin
run b3_2d_xy_K8_p8k_vswin --preset b3 --regime 2d --mode xy --K 8 --steps 8000 --batch 8 --lam_wiener 3e-6 --model vswin --name b3_2d_xy_K8_p8k_vswin
run b1_2d_xy_K8_p8k_vswin --preset b1 --regime 2d --mode xy --K 8 --steps 8000 --batch 8 --model vswin --name b1_2d_xy_K8_p8k_vswin
echo "[$(date +%m%d-%H:%M)] GPU2 chain DONE" >> runs_v3/queue_gpu2.log

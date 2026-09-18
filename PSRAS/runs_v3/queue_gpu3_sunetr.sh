#!/bin/bash
# EXP-020 GPU3 chain: swin2d xyz (ours) -> SwinUNETR (batch 2 x accum 4)
cd /data2/wjb/threePM/PSRAS
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
export CUDA_VISIBLE_DEVICES=3
run() {
  echo "[$(date +%m%d-%H:%M)] START $1" >> runs_v3/queue_gpu3.log
  shift
  $PY scripts/train.py "$@" >> runs_v3/queue_gpu3.log 2>&1
  echo "[$(date +%m%d-%H:%M)] END   (exit $?)" >> runs_v3/queue_gpu3.log
}
run b3_2p5d_xyz_K8_p8k_lam3e6 --preset b3 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 8 --lam_wiener 3e-6 --name b3_2p5d_xyz_K8_p8k_lam3e6
run b1_2p5d_xyz_K8_p8k --preset b1 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 8 --name b1_2p5d_xyz_K8_p8k
run b4_2p5d_xyz_K8_p8k_lam3e6 --preset b4 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 8 --lam_wiener 3e-6 --name b4_2p5d_xyz_K8_p8k_lam3e6
run b3_2p5d_z_K8_p8k_lam3e6_sunetr --preset b3 --regime 2.5d --mode z --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model sunetr --name b3_2p5d_z_K8_p8k_lam3e6_sunetr
run b1_2p5d_z_K8_p8k_sunetr --preset b1 --regime 2.5d --mode z --K 8 --steps 8000 --batch 2 --accum 4 --model sunetr --name b1_2p5d_z_K8_p8k_sunetr
run b3_2p5d_xyz_K8_p8k_lam3e6_sunetr --preset b3 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model sunetr --name b3_2p5d_xyz_K8_p8k_lam3e6_sunetr
run b1_2p5d_xyz_K8_p8k_sunetr --preset b1 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 2 --accum 4 --model sunetr --name b1_2p5d_xyz_K8_p8k_sunetr
run b3_2d_xy_K8_p8k_sunetr --preset b3 --regime 2d --mode xy --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model sunetr --name b3_2d_xy_K8_p8k_sunetr
run b1_2d_xy_K8_p8k_sunetr --preset b1 --regime 2d --mode xy --K 8 --steps 8000 --batch 2 --accum 4 --model sunetr --name b1_2d_xy_K8_p8k_sunetr
echo "[$(date +%m%d-%H:%M)] GPU3 chain DONE" >> runs_v3/queue_gpu3.log

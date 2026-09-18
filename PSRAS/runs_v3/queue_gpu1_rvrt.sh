#!/bin/bash
# EXP-020 GPU1 chain: RVRT (fp32, batch 2 x accum 4)
cd /data2/wjb/threePM/PSRAS
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
export CUDA_VISIBLE_DEVICES=1
run() {
  echo "[$(date +%m%d-%H:%M)] START $1" >> runs_v3/queue_gpu1.log
  shift
  $PY scripts/train.py "$@" >> runs_v3/queue_gpu1.log 2>&1
  echo "[$(date +%m%d-%H:%M)] END   (exit $?)" >> runs_v3/queue_gpu1.log
}
run b3_2p5d_z_K8_p8k_lam3e6_rvrt --preset b3 --regime 2.5d --mode z --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model rvrt --name b3_2p5d_z_K8_p8k_lam3e6_rvrt
run b1_2p5d_z_K8_p8k_rvrt --preset b1 --regime 2.5d --mode z --K 8 --steps 8000 --batch 2 --accum 4 --model rvrt --name b1_2p5d_z_K8_p8k_rvrt
run b3_2p5d_xyz_K8_p8k_lam3e6_rvrt --preset b3 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model rvrt --name b3_2p5d_xyz_K8_p8k_lam3e6_rvrt
run b1_2p5d_xyz_K8_p8k_rvrt --preset b1 --regime 2.5d --mode xyz --K 8 --steps 8000 --batch 2 --accum 4 --model rvrt --name b1_2p5d_xyz_K8_p8k_rvrt
run b3_2d_xy_K8_p8k_rvrt --preset b3 --regime 2d --mode xy --K 8 --steps 8000 --batch 2 --accum 4 --lam_wiener 3e-6 --model rvrt --name b3_2d_xy_K8_p8k_rvrt
run b1_2d_xy_K8_p8k_rvrt --preset b1 --regime 2d --mode xy --K 8 --steps 8000 --batch 2 --accum 4 --model rvrt --name b1_2d_xy_K8_p8k_rvrt
echo "[$(date +%m%d-%H:%M)] GPU1 chain DONE" >> runs_v3/queue_gpu1.log

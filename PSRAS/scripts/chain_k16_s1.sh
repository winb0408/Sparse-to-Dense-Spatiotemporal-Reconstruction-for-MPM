#!/bin/bash
# K16 second-seed chain (EXP-015(6)): b4 -> b0 -> b2 -> b3, seed 1, GPU2
# Launch: nohup bash scripts/chain_k16_s1.sh > /tmp/k16_s1_chain.log 2>&1 &
P=/data2/wjb/anaconda3/envs/threePM/bin/python
export CUDA_VISIBLE_DEVICES=2
cd /data2/wjb/threePM/PSRAS
$P scripts/train.py --preset b4 --regime 2.5d --K 16 --steps 8000 --batch 8 --lam_wiener 3e-6 --seed 1 --name b4_2p5d_z_K16_p8k_lam3e6_s1 > runs_v3/b4_2p5d_z_K16_p8k_lam3e6_s1.out 2>&1
$P scripts/train.py --preset b0 --regime 2.5d --K 16 --steps 8000 --batch 8 --lam_wiener 3e-6 --seed 1 --name b0_2p5d_z_K16_p8k_lam3e6_s1 > runs_v3/b0_2p5d_z_K16_p8k_lam3e6_s1.out 2>&1
$P scripts/train.py --preset b2 --regime 2.5d --K 16 --steps 8000 --batch 8 --lam_wiener 3e-6 --seed 1 --name b2_2p5d_z_K16_p8k_lam3e6_s1 > runs_v3/b2_2p5d_z_K16_p8k_lam3e6_s1.out 2>&1
$P scripts/train.py --preset b3 --regime 2.5d --K 16 --steps 8000 --batch 8 --lam_wiener 3e-6 --seed 1 --name b3_2p5d_z_K16_p8k_lam3e6_s1 > runs_v3/b3_2p5d_z_K16_p8k_lam3e6_s1.out 2>&1

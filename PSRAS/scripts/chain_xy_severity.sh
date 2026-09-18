#!/bin/bash
# EXP-016: xy severity line K in {2,8}, B in {b0,b2,b3,b4}, seed 0, GPU3
# Order: K2 (b2,b3,b0,b4) then K8 (b2,b3,b0,b4) -- core B2/B3 comparison first.
P=/data2/wjb/anaconda3/envs/threePM/bin/python
export CUDA_VISIBLE_DEVICES=3
cd /data2/wjb/threePM/PSRAS

for K in 2 8; do
  for PRESET in b2 b3 b0 b4; do
    NAME=${PRESET}_2d_xy_K${K}_p8k_lam3e6
    echo "[chain] launching $NAME"
    $P scripts/train.py --preset $PRESET --regime 2d --K $K --steps 8000 \
      --batch 8 --lam_wiener 3e-6 --name $NAME \
      > runs_v3/${NAME}.out 2>&1
    echo "[chain] finished $NAME"
  done
done
echo "[chain] all done"

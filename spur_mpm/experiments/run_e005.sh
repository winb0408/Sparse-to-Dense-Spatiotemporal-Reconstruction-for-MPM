#!/usr/bin/env bash
# E005 -- can a self-supervised arm be made to use its input?
#
# SPUR-E003b: SS0/SS1 score 22.5/24.3 dB against interpolation's 29.1 and err by
# 0.049-0.068 at the voxels they were *given*, where interpolation errs by
# exactly 0. MMS constrains Omega_hold and nothing ties the output to y on
# Omega_in. Two knobs claim to fix that, and this experiment tests both
# separately so the effect of each is attributable:
#
#   ratio010 : selfsupervision.ratio=0.1 -- the training-time holdout shares
#              match the evaluation-time sampling density (k_y=2 skips half the
#              rows and ratio=0.5 halves that again, so training sees 25% of
#              rows while evaluation gives 50%; 0.1 brings them together).
#   dc1      : loss.dc_weight=1.0 -- the data-consistency term is now plumbed
#              (y_in/in_mask reach the criterion); it penalises disagreement
#              with the given measurements directly.
#
# Six runs. GPU 2 is left alone: it is occupied by another user's job.
set -u
cd /ssd1/wjb/SPUR-MPM
GPUS=(0 1 3 0 1 3)
CFG=spur_mpm/configs/experiments/E003_selfsup_y_k2.yaml

pids=()
i=0
for variant in ratio010:--set\ selfsupervision.ratio=0.1 dc1:--set\ loss.dc_weight=1.0; do
  name=${variant%%:*}
  extra=${variant#*:}
  for f in 0 1 2; do
    gpu=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$gpu python -m spur_mpm.train \
      --config $CFG --arm SS1 --fold $f --name E005_${name}_fold${f} \
      --set data.index=data_local/dryad_cache/index.json \
      --set data.split=spur_mpm/experiments/E000_data_audit/split.json \
      --set data.patch_hw='[256,256]' --set train.num_workers=4 \
      $extra > /tmp/E005_${name}_fold${f}.log 2>&1 &
    pids+=($!)
    echo "launched E005_${name}_fold${f} on GPU$gpu (pid $!)"
    i=$((i+1))
  done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "E005 training finished, $fail run(s) failed"
echo ALL_DONE

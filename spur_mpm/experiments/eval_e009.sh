#!/usr/bin/env bash
# Score the k_y=4 batch (E009): the interpolation baseline first, then every
# trained arm through eval_trained.sh under the E009 geometry.
#
#   bash spur_mpm/experiments/eval_e009.sh
#
# A0 needs no checkpoint: it is the bilinear interpolation of the same sparse
# measurement the trained arms get, so it must be scored under k_y=4 too --
# quoting the k_y=2 baseline (29.06 dB) here would make the comparison
# meaningless, since interpolation is expected to fall a long way at k_y=4.
set -eu
cd "$(dirname "$0")/../.."

CFG=spur_mpm/configs/experiments/E009_sampling_y_k4.yaml
INDEX=data_local/dryad_cache/index.json
SPLIT=spur_mpm/experiments/E000_data_audit/split.json

for fold in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$fold python -m spur_mpm.evaluate \
    --config "$CFG" --arm A0 --fold "$fold" --split test \
    --set data.index="$INDEX" --set data.split="$SPLIT" \
    --set data.patch_hw='[512,512]' --set train.num_workers=0 \
    --out "runs/E009_A0_fold${fold}_test" \
    > "/tmp/eval_E009_A0_fold${fold}.log" 2>&1 &
done
wait
echo "evaluated E009_A0 (bilinear at k_y=4)"

# The run prefixes carry the recipe only implicitly (E009_SS1dc1 has no arm
# definition of its own), so each is scored with an explicit ARM.
ARM=S1  CONFIG="$CFG" bash spur_mpm/experiments/eval_trained.sh E009_S1
ARM=SS1 CONFIG="$CFG" bash spur_mpm/experiments/eval_trained.sh E009_SS1 E009_SS1dc1

echo ALL_DONE

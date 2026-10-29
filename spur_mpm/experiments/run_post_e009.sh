#!/usr/bin/env bash
# Unattended follow-up queue for the k_y=4 sweep.
#
# Nothing here decides anything: it waits for E009's nine runs, scores them,
# then collects the two follow-ups that the E005 result made necessary and that
# need no human decision between steps:
#
#   1. score E009 (A0 at k_y=4 + the three arms)              -> runs/E009_*_test
#   2. re-score E005's dc1 arm from last.pt, not best.pt      -> runs/..._test_last
#      E005 showed held-out loss is a bad selector, so the question "how much
#      did selection cost?" has to be answered, not assumed.
#   3. the secondary holdout protocol for the self-supervised arms
#                                                             -> runs_holdout/
#   4. qualitative figures including the dc1 arm              -> figures/qualitative
#
# Everything is appended to the log with timestamps so the run can be audited
# in the morning without reading nine separate /tmp files.
set -u
cd /ssd1/wjb/SPUR-MPM
TEE=/tmp/post_e009.log
exec > >(tee -a "$TEE") 2>&1

say () { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for the nine E009 runs to finish"
deadline=$(( $(date +%s) + 6*3600 ))
while :; do
  done_n=0
  for name in S1 SS1 SS1dc1; do
    for f in 0 1 2; do
      [[ -f "runs/E009_${name}_fold${f}/final_val.json" ]] && done_n=$((done_n+1))
    done
  done
  say "  ${done_n}/9 finished"
  [[ $done_n -eq 9 ]] && break
  if (( $(date +%s) > deadline )); then say "TIMEOUT waiting for training"; exit 1; fi
  sleep 120
done

say "step 1: scoring E009 (A0 must be re-scored at k_y=4)"
bash spur_mpm/experiments/eval_e009.sh || say "WARN eval_e009 returned $?"

say "step 2: E005 dc1 scored from last.pt, to size the selection effect"
ARM=SS1 CKPT=last.pt OUT_SUFFIX=_last \
  bash spur_mpm/experiments/eval_trained.sh E005_dc1 \
  || say "WARN last.pt eval returned $?"

say "step 3: same-input holdout protocol (no projection) for every arm"
# A0 is run first and without a checkpoint: the whole point is that the
# parameter-free baseline also loses half of its measured rows. Without
# eval.holdout_baselines it would silently get the full mask.
INPUT_SUPPORT=holdout PROJECTION=none ARM=A0 \
  bash spur_mpm/experiments/eval_variants.sh E000_A0heldout \
  || say "WARN A0 holdout returned $?"
INPUT_SUPPORT=holdout PROJECTION=none ARM=SS1 \
  bash spur_mpm/experiments/eval_variants.sh E003_SS1 E005_dc1 \
  || say "WARN holdout variants returned $?"

say "step 4: qualitative figures including dc1"
# The arm syntax is LABEL/ARM=CKPT: a run label is not an arm name, so
# "--arm dc1=..." dies with "unknown arm 'dc1'".
for f in 0 1 2; do
  python -m spur_mpm.scripts.qualitative_figures --fold "$f" \
    --arm A0 --arm S1/S1=runs/E002_S1_fold$f/best.pt \
    --arm SS1/SS1=runs/E003_SS1_fold$f/best.pt \
    --arm dc1/SS1=runs/E005_dc1_fold$f/best.pt \
    --out spur_mpm/figures/qualitative --device cuda:0 \
    || say "WARN figures fold $f returned $?"
done

say "done"
echo ALL_DONE

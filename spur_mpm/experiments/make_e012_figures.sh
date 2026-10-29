#!/usr/bin/env bash
# Qualitative figure for E012: what does the complementary pair actually build?
#
#   bash spur_mpm/experiments/make_e012_figures.sh
#
# Rows: Input / GT / A0 / S1 / SS1 / SS1c, one column per view, tag _e012 so it
# does not overwrite the headline figure. The panel to look at first is
# "XZ gap" -- a y row that was never measured, where the pair has no measurement
# in either branch. If the complementary split helps at all, that is where it
# should show, and the error figure gives the same comparison with one scale.
#
# Timing: launch it after E012 training has written best.pt for all three folds;
# it loads four checkpoints and reconstructs a whole volume per arm.
set -u
cd "$(dirname "$0")/../.."
CFG=spur_mpm/configs/experiments/E003_selfsup_y_k2.yaml
OUT=spur_mpm/figures/qualitative

for f in 0 1 2; do
  python -m spur_mpm.scripts.qualitative_figures --fold "$f" --config "$CFG" \
    --arm A0 \
    --arm S1/S1=runs/E002_S1_fold$f/best.pt \
    --arm SS1/SS1=runs/E003_SS1_fold$f/best.pt \
    --arm SS1c/SS1c=runs/E012_SS1c_fold$f/best.pt \
    --out "$OUT" --device cuda:$f --tag _e012 \
    > "/tmp/figs_e012_fold$f.log" 2>&1 &
done
wait
echo ALL_DONE

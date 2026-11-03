#!/bin/bash
# The full re-score was launched with --context-slices 7, which renames every
# cell whose own minimum context is not 7 (y_k2 -> y_k2_c7, y_k3 -> y_k3_c7, ...).
# Those *_c7 siblings were never trained, so run_ladder reported
# `no_checkpoint` and left the real cells scored with the pre-fix Z-border code.
# Re-score exactly those nine, at each level's own minimum context.
set -u
cd /ssd1/wjb/SPUR-MPM
R=runs/E023_phase1
L=runs/E021_ladder

python -u -m spur_mpm.scripts.run_ladder \
  --stage ablation --stage sweep --eval-only --force --skeleton \
  --only E021_S0_y_k2 --only E021_S1_y_k2 --only E021_S2_y_k2 \
  --only E021_S1g_y_k2 --only E021_S2g_y_k2 \
  --only E021_S1_y_k3 --only E021_S1_y_k8 \
  --only E021_S1_z_k2 --only E021_S1_z_k3 \
  --devices 2 --jobs 2 --out-root $L > $R/rescore_own_context.out 2>&1
echo "rescore rc=$?"

python -m spur_mpm.scripts.ladder_report --root $L \
  --out $L/ladder.md --csv $L/ladder.csv > $R/report.out 2>&1
cp $L/ladder.md $L/ladder_report.md
cp $L/ladder.csv $L/ladder_report.csv
echo "report rc=$?"
echo OWN_CONTEXT_RESCORE_DONE

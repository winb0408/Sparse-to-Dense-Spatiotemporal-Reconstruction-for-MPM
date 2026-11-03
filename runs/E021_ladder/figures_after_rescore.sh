#!/bin/bash
# Wait for the 27-cell re-score, then draw the two qualitative figures that the
# claims are actually about: the 25 % row-skipping level and the 25 % plane-
# skipping level. Both use the run's own config.yaml, so the sampling, context
# and calibration are the ones the checkpoints were trained under.
set -u
cd /ssd1/wjb/SPUR-MPM
while pgrep -f "rescore_all.sh" > /dev/null; do sleep 60; done
date '+%F %H:%M  rescore gone; drawing figures'
for lvl in y_k4 z_k4; do
  ck=runs/E021_ladder/E021_S1_${lvl}/best.pt
  c2=runs/E021_ladder/E021_S2_${lvl}/best.pt
  args=(--config runs/E021_ladder/E021_S1_${lvl}/config.yaml
        --index data_local/dryad_cache/index_cohort.json
        --out runs/E021_ladder/figures --tag _${lvl} --device 2
        --arm A0 --arm A1 --arm S1=${ck} --arm S2=${c2})
  python -u -m spur_mpm.scripts.qualitative_figures "${args[@]}" \
    > runs/E021_ladder/figures_${lvl}.out 2>&1
  date "+%F %H:%M  ${lvl} done (rc=$?)"
done

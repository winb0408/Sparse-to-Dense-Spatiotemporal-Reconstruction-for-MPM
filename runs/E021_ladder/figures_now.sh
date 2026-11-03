#!/bin/bash
set -u
cd /ssd1/wjb/SPUR-MPM
for lvl in y_k4 z_k4; do
  python -u -m spur_mpm.scripts.qualitative_figures \
    --config runs/E021_ladder/E021_S1_${lvl}/config.yaml \
    --index data_local/dryad_cache/index_cohort.json \
    --out runs/E021_ladder/figures --tag _${lvl} --device 2 \
    --arm A0 --arm A1 \
    --arm S1=runs/E021_ladder/E021_S1_${lvl}/best.pt \
    --arm S2=runs/E021_ladder/E021_S2_${lvl}/best.pt \
    > runs/E021_ladder/figures_${lvl}.out 2>&1
  echo "$(date '+%H:%M') ${lvl} rc=$?"
done

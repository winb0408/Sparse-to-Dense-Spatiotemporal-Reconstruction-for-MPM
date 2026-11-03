#!/bin/bash
# Wait for the sweep/matched retry to finish, then re-score every finished cell
# with the equal-area columns and the skeleton, and rebuild the table.
# Nothing here retrains: checkpoints do not depend on the metric.
set -u
cd /ssd1/wjb/SPUR-MPM
while pgrep -f "spur_mpm.scripts.run_ladder --stage" > /dev/null; do
  sleep 60
done
date '+%F %H:%M  retry finished'
python -u -m spur_mpm.scripts.run_ladder \
  --stage ablation --stage sweep --stage matched \
  --eval-only --force --skeleton --devices 2 --jobs 2 \
  > runs/E021_ladder/evalonly_skeleton_area.out 2>&1
date '+%F %H:%M  rescore finished'
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder \
  --out runs/E021_ladder/ladder.md \
  --csv runs/E021_ladder/ladder.csv > runs/E021_ladder/ladder_report.out 2>&1
date '+%F %H:%M  table rebuilt'
tail -4 runs/E021_ladder/ladder_report.out

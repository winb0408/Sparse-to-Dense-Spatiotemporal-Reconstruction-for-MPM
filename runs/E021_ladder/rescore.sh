#!/bin/bash
set -u
cd /ssd1/wjb/SPUR-MPM
python -u -m spur_mpm.scripts.run_ladder \
  --stage ablation --stage sweep --stage matched \
  --eval-only --force --skeleton --devices 2 --jobs 2 \
  > runs/E021_ladder/evalonly_skeleton_area.out 2>&1
date '+%F %H:%M  rescore finished'
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder \
  --out runs/E021_ladder/ladder.md \
  --csv runs/E021_ladder/ladder.csv > runs/E021_ladder/ladder_report.out 2>&1
date '+%F %H:%M  table rebuilt'

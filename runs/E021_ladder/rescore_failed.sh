#!/bin/bash
set -u
cd /ssd1/wjb/SPUR-MPM
python -u -m spur_mpm.scripts.run_ladder \
  --stage sweep --stage matched --eval-only --skeleton --devices 2 --jobs 2 \
  > runs/E021_ladder/evalonly_retry_z.out 2>&1
date '+%F %H:%M  rescore retry finished'
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder \
  --out runs/E021_ladder/ladder.md \
  --csv runs/E021_ladder/ladder.csv > runs/E021_ladder/ladder_report.out 2>&1
date '+%F %H:%M  table rebuilt'

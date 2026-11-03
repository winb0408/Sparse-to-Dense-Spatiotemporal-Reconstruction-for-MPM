#!/bin/bash
set -u
cd /ssd1/wjb/SPUR-MPM
python -u -m spur_mpm.scripts.run_ladder --stage sweep --only z_k3 \
  --eval-only --force --skeleton --devices 2 --jobs 1 \
  > runs/E021_ladder/rescore_z_k3.out 2>&1
echo "z_k3 rc=$? $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.run_ladder --stage control --context-slices 7 \
  --only y_k4_c7 --only yz_k2x2_c7 \
  --eval-only --force --skeleton --devices 2 --jobs 2 \
  > runs/E021_ladder/rescore_c7.out 2>&1
echo "c7 rc=$? $(date '+%F %H:%M')"
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder \
  --out runs/E021_ladder/ladder.md --csv runs/E021_ladder/ladder.csv \
  > runs/E021_ladder/ladder_report.out 2>&1
echo "report rc=$? $(date '+%F %H:%M')"; echo ALL_DONE

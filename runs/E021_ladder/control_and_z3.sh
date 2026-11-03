#!/bin/bash
set -u
cd /ssd1/wjb/SPUR-MPM
echo "=== z_k3 (sweep, own context) $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.run_ladder \
  --stage sweep --devices 2 --jobs 2 \
  > runs/E021_ladder/sweep_z3.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"
echo "=== context control at 7 (S1 y_k4 yz_k2x2; z_k4 reuses its own cell) $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.run_ladder \
  --stage control --context-slices 7 --devices 2 --jobs 2 \
  > runs/E021_ladder/control7_train.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder \
  --out runs/E021_ladder/ladder.md --csv runs/E021_ladder/ladder.csv \
  > runs/E021_ladder/ladder_report.out 2>&1
echo "table rebuilt rc=$? $(date '+%F %H:%M')"
echo ALL_DONE

#!/bin/bash
set -u
cd /ssd1/wjb/SPUR-MPM
R=runs/E023_phase1

echo "=== ablation + sweep + floors (new: S2g y_k2) $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.run_ladder --stage ablation --stage sweep --stage floors \
  --devices 2 --jobs 2 --out-root runs/E021_ladder > $R/ladder_ablation.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"

echo "=== matched at one 7-plane context, all seven arms $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.run_ladder --stage matched --context-slices 7 \
  --devices 2 --jobs 2 --out-root runs/E021_ladder > $R/ladder_matched_c7.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"

echo "=== full re-score of every cell (Z-border fix changed the predictions) $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.run_ladder --stage ablation --stage sweep --stage matched \
  --stage floors --stage control --context-slices 7 --eval-only --force --skeleton \
  --devices 2 --jobs 2 --out-root runs/E021_ladder > $R/rescore_all.out 2>&1
echo "rc=$? $(date '+%F %H:%M')"
echo FULL_RESCORE_DONE

python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder \
  --out runs/E021_ladder/ladder.md --csv runs/E021_ladder/ladder.csv > $R/report.out 2>&1
echo "report rc=$? $(date '+%F %H:%M')"
echo ALL_DONE

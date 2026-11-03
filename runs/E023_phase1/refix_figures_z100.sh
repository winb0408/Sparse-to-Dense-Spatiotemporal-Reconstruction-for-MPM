#!/bin/bash
# The first pass fixed the XY panel at z = Z//2 = 101, which z_k4 and yz_k2x2
# *skip* -- so the shared column was blank in the input row of exactly the
# protocols the figure exists to compare.  Re-draw with z = 100, a plane all
# three 25 % protocols measured (y_k4 25 %; z_k4 100 %; yz_k2x2 50 %), named
# explicitly (not chosen from any mask).  The z=101 set is kept, not overwritten.
set -u
cd /ssd1/wjb/SPUR-MPM
R=runs/E023_phase1
INDEX=data_local/dryad_cache/index_cohort.json
SPLIT=spur_mpm/experiments/E020_cohort/split.json
L=runs/E021_ladder
FIG=$L/figures_fixed/M2_FOVII_S1_vessels
OLD=$L/figures_fixed/M2_FOVII_S1_vessels_z101
SAMPLE=M2_FOVII_S1_vessels

[ -d "$OLD" ] || mv "$FIG" "$OLD"
mkdir -p "$FIG"

for spec in \
  "y_k4|y|4|1|E021_S1_y_k4_c7" \
  "z_k4|z|1|4|E021_S1_z_k4" \
  "yz_k2x2|yz|2|2|E021_S1_yz_k2x2_c7" ; do
  IFS="|" read -r level mode ky kz cell <<< "$spec"
  echo "=== $level (context 7, z_view 100) $(date '+%F %H:%M')"
  vs=""
  [ -f "$FIG/view_spec.json" ] && vs="--view-spec $FIG/view_spec.json"
  python -u -m spur_mpm.scripts.qualitative_figures \
    --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
    --index $INDEX --fold 0 --sample $SAMPLE \
    --arm A0 --arm S1=$L/$cell/best.pt \
    --out "$FIG" --tag "_$level" --dump f32 --device 2 \
    --z-view 100 --save-view-spec $FIG/view_spec.json $vs \
    --set data.split=$SPLIT \
    --set sampling.mode=$mode --set sampling.ky=$ky --set sampling.kz=$kz \
    --set data.context_slices=7 \
    > $R/qual_fixed_${level}_z100.out 2>&1
  echo "rc=$?"
done

echo "=== compare (z_view 100) $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.qualitative_figures \
  --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --index $INDEX --fold 0 --device 2 \
  --compare $R/compare_y_z_yz_25pct.json \
  --view-spec $FIG/view_spec.json \
  --figure-name compare_y_z_yz_25pct.png --out "$FIG" \
  --set data.split=$SPLIT \
  > $R/compare_y_z_yz_25pct.out 2>&1
echo "rc=$?"
tail -2 $R/compare_y_z_yz_25pct.out
echo REFIX_DONE

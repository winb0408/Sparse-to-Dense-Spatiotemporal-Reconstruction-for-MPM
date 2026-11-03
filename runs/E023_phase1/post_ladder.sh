#!/bin/bash
# Runbook §4.3 / §4.4 / §4.4.1 D / §4.5, to run *after* the ladder pipeline
# (ladder training, equal-context control, re-score) has finished, so nothing
# else is competing for the card: the §4.5 numbers are timings.
set -u
cd /ssd1/wjb/SPUR-MPM
R=runs/E023_phase1
INDEX=data_local/dryad_cache/index_cohort.json
SPLIT=spur_mpm/experiments/E020_cohort/split.json
L=runs/E021_ladder
FIG=$L/figures_fixed/M2_FOVII_S1_vessels
mkdir -p "$FIG" runs/E021_ladder/figures/qual_y_k2

echo "=== §4.3 structure report: whole-volume 3D skeleton, S1 @ y_k2 $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.structure_report \
  --arm S1 --ckpt $L/E021_S1_y_k2/best.pt \
  --out $L/E021_S1_y_k2/structure --device 2 \
  --set data.index=$INDEX --set data.split=$SPLIT \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1 \
  > $R/structure_report.out 2>&1
echo "rc=$?"

echo "=== §4.4 qualitative ladder figure, y_k2, five arms $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.qualitative_figures \
  --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --index $INDEX --fold 0 --sample M2_FOVI_S1_vessels \
  --arm A0 \
  --arm S0=$L/E021_S0_y_k2/best.pt \
  --arm S1=$L/E021_S1_y_k2/best.pt \
  --arm S2=$L/E021_S2_y_k2/best.pt \
  --arm S2g=$L/E021_S2g_y_k2/best.pt \
  --out $L/figures/qual_y_k2 --dump f32 --device 2 \
  --save-view-spec $L/figures/qual_y_k2/view_spec.json \
  --set data.split=$SPLIT \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1 \
  > $R/qual_y_k2.out 2>&1
echo "rc=$?"

# --- §4.4.1 D: per-level fixed-view figures in one per-sample directory ------
# All three levels run at the *same* 7-plane context and read the same frozen
# ViewSpec: the y_k4 and yz_k2x2 cells are the equal-context controls.
for spec in \
  "y_k4|y|4|1|E021_S1_y_k4_c7" \
  "z_k4|z|1|4|E021_S1_z_k4" \
  "yz_k2x2|yz|2|2|E021_S1_yz_k2x2_c7" ; do
  IFS="|" read -r level mode ky kz cell <<< "$spec"
  echo "=== §4.4.1 D qualitative_*/support_*: $level (context 7) $(date '+%F %H:%M')"
  vs=""
  [ -f "$FIG/view_spec.json" ] && vs="--view-spec $FIG/view_spec.json"
  python -u -m spur_mpm.scripts.qualitative_figures \
    --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
    --index $INDEX --fold 0 --sample M2_FOVII_S1_vessels \
    --arm A0 --arm S1=$L/$cell/best.pt \
    --out "$FIG" --tag "_$level" --dump f32 --device 2 \
    --save-view-spec $FIG/view_spec.json $vs \
    --set data.split=$SPLIT \
    --set sampling.mode=$mode --set sampling.ky=$ky --set sampling.kz=$kz \
    --set data.context_slices=7 \
    > $R/qual_fixed_$level.out 2>&1
  echo "rc=$?"
done

echo "=== §4.4.1 D compare_y_z_yz_25pct.png + consistency_report.json $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.qualitative_figures \
  --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --index $INDEX --fold 0 --device 2 \
  --compare $R/compare_y_z_yz_25pct.json \
  --view-spec $FIG/view_spec.json \
  --figure-name compare_y_z_yz_25pct.png \
  --out "$FIG" \
  --set data.split=$SPLIT \
  > $R/compare_y_z_yz_25pct.out 2>&1
echo "rc=$?"
tail -2 $R/compare_y_z_yz_25pct.out

echo "=== §4.5 benchmark (nothing else on the card) $(date '+%F %H:%M')"
python -u -m spur_mpm.scripts.benchmark --markdown \
  --arms S0 S1 S2 S1g S2g --device 2 \
  --out $L/benchmark_supervised.json > $R/benchmark.out 2>&1
echo "rc=$?"
echo POST_LADDER_DONE

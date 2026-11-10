#!/usr/bin/env bash
# Regenerate the §4.4.1 acceptance set so its metadata is produced by code.
#
# The 03:49 figure set described itself as "none (single XY/XZ/YZ slices; mip_z
# kept for a future projection)" -- a constant written before the layout gained
# a MIP-z column, and never re-read against the picture. The note is now derived
# from the column list (`viewspec.projection_note`), so the stored view_spec.json
# and the per-figure sidecars have to be rewritten by the renderer, not patched.
#
# Pixels are unchanged on purpose: the spec is loaded with --view-spec, so the
# cuts and the display windows are the frozen ones. Only the metadata moves.
set -u
cd "$(dirname "$0")/../.."
R=runs/E023_phase1
L=runs/E021_ladder
FIG=$L/figures_fixed/M2_FOVII_S1_vessels
INDEX=data_local/dryad_cache/index_cohort.json
SPLIT=spur_mpm/experiments/E020_cohort/split.json

cp "$FIG/view_spec.json" "$FIG/view_spec_old_note.json"

echo "=== compare_y_z_yz_25pct.png $(date '+%H:%M')"
python -u -m spur_mpm.scripts.qualitative_figures \
  --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --index $INDEX --fold 0 --device 2 \
  --compare $R/compare_y_z_yz_25pct.json \
  --view-spec "$FIG/view_spec_old_note.json" \
  --figure-name compare_y_z_yz_25pct.png \
  --save-view-spec "$FIG/view_spec.json" \
  --out "$FIG" --set data.split=$SPLIT \
  > $R/refreeze_compare.out 2>&1
echo "compare rc=$?"
tail -2 $R/refreeze_compare.out

for spec in \
  "y_k4|y|4|1|E021_S1_y_k4_c7" \
  "z_k4|z|1|4|E021_S1_z_k4" \
  "yz_k2x2|yz|2|2|E021_S1_yz_k2x2_c7" ; do
  IFS="|" read -r level mode ky kz cell <<< "$spec"
  echo "=== $level $(date '+%H:%M')"
  python -u -m spur_mpm.scripts.qualitative_figures \
    --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
    --index $INDEX --fold 0 --sample M2_FOVII_S1_vessels \
    --arm A0 --arm S1=$L/$cell/best.pt \
    --out "$FIG" --tag "_$level" --dump f32 --device 2 \
    --z-view 100 --view-spec "$FIG/view_spec.json" \
    --set data.split=$SPLIT \
    --set sampling.mode=$mode --set sampling.ky=$ky --set sampling.kz=$kz \
    --set data.context_slices=7 \
    > $R/refreeze_qual_${level}.out 2>&1
  echo "$level rc=$?"
done
echo REFREEZE_DONE

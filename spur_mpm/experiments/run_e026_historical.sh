#!/usr/bin/env bash
# E026 historical-condition audit: YZ k_y=2, k_z=2 (25% retained).
# Real data AND matched-known-PSF synthetic control; never mix claims.
# This is a targeted mechanism study, NOT an exact CIDC25 PSRAS reproduction.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${FOLD:=0}" "${SEED:=0}" "${KY:=2}" "${KZ:=2}"
: "${SOURCE_KIND:=matched_synthetic}" "${STAGE:=all}" "${DRY:=1}"
# BATCH x ACCUM keeps the effective batch at 8 while halving peak activation
# memory (co-tenancy on a shared card). Applied to EVERY arm in the group, so
# within-group comparisons are unaffected; the values land in each run config.
: "${OUT:=runs/E026_psras_condition}" "${GPU:=}" "${STEPS:=3000}"
: "${BATCH:=8}" "${ACCUM:=1}"
: "${CFG:=spur_mpm/configs/experiments/E026_psras_condition.yaml}"
: "${INDEX:=data_local/dryad_cache/index_cohort.json}"
: "${SPLIT:=spur_mpm/experiments/E020_cohort/split.json}"
# Historical PSRAS sigma = (1.0 XY px, 1.5 Z px).
# Current cohort voxel size = (0.98 um XY, 5 um Z); known H = (0.98, 7.5) um.
: "${KNOWN_XY_UM:=0.98}" "${KNOWN_Z_UM:=7.5}" "${LAM:=0.01}"
if [[ "$SOURCE_KIND" != real && "$SOURCE_KIND" != matched_synthetic ]]; then
  echo "SOURCE_KIND must be real or matched_synthetic" >&2; exit 2
fi
if (( KY < 2 || KZ < 2 )); then
  echo "E026 requires two-axis sparsity KY>=2 KZ>=2" >&2; exit 2
fi
CONTEXT=$((2*KZ-1))
TAG="E026_${SOURCE_KIND}_yz_k${KY}_kz${KZ}_fold${FOLD}_s${SEED}"
SEL="$OUT/fold${FOLD}_yz_k${KY}_kz${KZ}_selection.json"
mkdir -p "$OUT"
if [[ "$SOURCE_KIND" == real ]]; then
  if [[ "$STAGE" == select || "$STAGE" == all ]]; then
    if [[ "$DRY" == 1 ]]; then
      echo "DRY: choose effective kernel on TRAIN animals only -> $SEL"
    elif [[ -s "$SEL" ]]; then
      echo "reuse frozen selection $SEL"
    else
      python -m spur_mpm.scripts.select_adaptive_prior \
        --index "$INDEX" --split "$SPLIT" --fold "$FOLD" \
        --mode yz --ky "$KY" --kz "$KZ" \
        --sigmas-xy 0 0.5 0.98 \
        --sigmas-z 0 2.5 5 7.5 \
        --lambdas 0.005 0.01 0.05 --out "$SEL"
    fi
  fi
  [[ "$STAGE" == select ]] && exit 0
  if [[ ! -s "$SEL" ]]; then
    if [[ "$DRY" == 1 ]]; then
      echo "DRY: need train-only selection before real-data training"; exit 0
    fi
    echo "Missing train-only selection: $SEL" >&2; exit 2
  fi
  mapfile -t PARAMS < <(python - "$SEL" "$FOLD" "$KY" "$KZ" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
p, s = j["protocol"], j["selected"]
assert j["status"] == "selected_on_training_volumes_only"
assert (p["fold"], p["mode"], p["ky"], p["kz"]) == (
    int(sys.argv[2]), "yz", int(sys.argv[3]), int(sys.argv[4]))
for k in ("sigma_xy_um", "sigma_z_um", "lam", "psf_source"):
    print(s[k])
PY
)
  SX="${PARAMS[0]}"; SZ="${PARAMS[1]}"; LAM="${PARAMS[2]}"
  PRIOR_SOURCE="${PARAMS[3]}"
  FORWARD_XY=0; FORWARD_Z=0; FORWARD_SOURCE=null
else
  [[ "$STAGE" == select ]] && {
    echo "Matched synthetic H is KNOWN, no blind fit performed."; exit 0;
  }
  SX="$KNOWN_XY_UM"; SZ="$KNOWN_Z_UM"
  PRIOR_SOURCE=synthetic_extra_degradation
  FORWARD_XY="$KNOWN_XY_UM"; FORWARD_Z="$KNOWN_Z_UM"
  FORWARD_SOURCE=synthetic_extra_degradation
fi
common=(--config "$CFG" --fold "$FOLD" --seed "$SEED"
        --set "data.index=$INDEX" --set "data.split=$SPLIT"
        --set sampling.mode=yz --set "sampling.ky=$KY"
        --set "sampling.kz=$KZ" --set "data.context_slices=$CONTEXT"
        --set "physics.psf.sigma_xy_um=$SX"
        --set "physics.psf.sigma_z_um=$SZ"
        --set "physics.psf_source=$PRIOR_SOURCE" --set "physics.lam=$LAM"
        --set "physics.forward_extra_psf.sigma_xy_um=$FORWARD_XY"
        --set "physics.forward_extra_psf.sigma_z_um=$FORWARD_Z"
        --set "physics.forward_extra_psf_source=$FORWARD_SOURCE"
        --set "data.patch_hw=[256,256]" --set "train.steps=$STEPS"
        --set "train.batch=$BATCH" --set "train.accum=$ACCUM")
if [[ "$DRY" != 1 && -z "$GPU" ]]; then
  echo "Set GPU=<owned GPU index> for evaluation and training" >&2; exit 2
fi
run_eval() {
  local name="$1" arm="$2" ckpt="${3:-}" ctrl="${4:-none}"
  local dir="$OUT/${name}_test"
  local args=(--arm "$arm" --split test --out "$dir"
              "${common[@]}" --set "model.phys_control=$ctrl")
  [[ -n "$ckpt" ]] && args+=(--ckpt "$ckpt")
  if [[ "$DRY" == 1 ]]; then
    printf 'DRY eval:'; printf ' %q' python -m spur_mpm.evaluate "${args[@]}"
    echo
  else
    [[ ! -e "$dir" ]] || {
      echo "Refuse overwrite: $dir" >&2; exit 3;
    }
    CUDA_VISIBLE_DEVICES="$GPU" python -m spur_mpm.evaluate "${args[@]}"
  fi
}
run_train() {
  local name="$1" arm="$2" ctrl="$3" aux="$4"
  local dir="$OUT/$name"
  local args=(--arm "$arm" --name "$name" --out-root "$OUT"
              "${common[@]}" --set "model.phys_control=$ctrl"
              --set "loss.branch_obs_weight=$aux"
              --set "loss.branch_phys_weight=$aux")
  if [[ "$DRY" == 1 ]]; then
    printf 'DRY train:'; printf ' %q' python -m spur_mpm.train "${args[@]}"
    echo
  else
    [[ ! -e "$dir" ]] || {
      echo "Refuse overwrite: $dir" >&2; exit 3;
    }
    CUDA_VISIBLE_DEVICES="$GPU" python -m spur_mpm.train "${args[@]}"
    [[ -s "$dir/final_val.json" && -s "$dir/best.pt" ]] || {
      echo "Training incomplete for $dir; archive as invalid/interrupted" >&2; exit 4;
    }
  fi
}
if [[ "$STAGE" == floors || "$STAGE" == all ]]; then
  run_eval "${TAG}_I0" P2_I0
  run_eval "${TAG}_W0_full3d" P2_W0
  run_eval "${TAG}_W0_legacy" P3_Wlegacy
fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then
  run_train "${TAG}_ResidualI" P2_ResidualI none 0
  run_eval "${TAG}_ResidualI" P2_ResidualI "$OUT/${TAG}_ResidualI/best.pt"
  run_train "${TAG}_P3_interp" P3_DualAdaptive interp_only 0.05
  run_eval "${TAG}_P3_interp" P3_DualAdaptive \
    "$OUT/${TAG}_P3_interp/best.pt" interp_only
  run_train "${TAG}_P3_wiener" P3_DualAdaptive none 0.05
  run_eval "${TAG}_P3_wiener" P3_DualAdaptive \
    "$OUT/${TAG}_P3_wiener/best.pt"
fi
echo "E026 $TAG finished. No runs were removed. Keep correct negative results."

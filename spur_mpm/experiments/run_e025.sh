#!/usr/bin/env bash
# E025: 75% missing (25% retained) PR3 adaptive-prior ladder.
# This script never deletes, overwrites or modifies E024.
# DRY=1 by default; GPU runs require explicit GPU=<owned physical index>.
set -euo pipefail
cd "$(dirname "$0")/../.."
CFG="${CFG:-spur_mpm/configs/experiments/E002_supervised_y_k2.yaml}"
INDEX="${INDEX:-data_local/dryad_cache/index_cohort.json}"
SPLIT="${SPLIT:-spur_mpm/experiments/E020_cohort/split.json}"
OUT="${OUT:-runs/E025_adaptive_dual_prior}"
FOLD="${FOLD:-0}"
SEED="${SEED:-0}"
KY="${KY:-4}"     # y_k4: 25% retained = 75% missing (NOT 75% retained)
GPU="${GPU:-}"
DRY="${DRY:-1}"
STAGE="${STAGE:-all}" # select | floors | train | all
SEL="$OUT/fold${FOLD}_y_k${KY}_prior_selection.json"
mkdir -p "$OUT"

if [[ "$STAGE" == "select" || "$STAGE" == "all" ]]; then
  if [[ "$DRY" == "1" ]]; then
    echo "DRY selection: fold=$FOLD, y_k$KY, output=$SEL"
  elif [[ -s "$SEL" ]]; then
    echo "Reuse frozen train-only selection: $SEL"
  else
    python -m spur_mpm.scripts.select_adaptive_prior \
      --index "$INDEX" --split "$SPLIT" --fold "$FOLD" --mode y --ky "$KY" \
      --out "$SEL"
  fi
fi
if [[ "$STAGE" == "select" ]]; then exit 0; fi
if [[ "$DRY" != "1" && -z "$GPU" ]]; then
  echo "Set GPU=<owned device index> for compute runs" >&2; exit 2
fi
if [[ ! -s "$SEL" ]]; then
  if [[ "$DRY" == "1" ]]; then
    echo "DRY: selection missing at $SEL; run STAGE=select DRY=0 first"
    exit 0
  fi
  echo "Missing frozen train-only PSF selection $SEL" >&2; exit 2
fi

# Read only scalar values from frozen JSON; never evaluate shell source.
mapfile -t PARAMS < <(python - "$SEL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["selected"]
assert d["status"] == "selected_on_training_volumes_only"
assert d["protocol"]["mode"] == "y"
assert s["psf_source"] in ("identity", "estimated_from_data")
for k in ("sigma_xy_um", "sigma_z_um", "lam", "psf_source"):
    print(s[k])
PY
)
SX="${PARAMS[0]}"; SZ="${PARAMS[1]}"; LAM="${PARAMS[2]}"; SOURCE="${PARAMS[3]}"
common=(--config "$CFG" --fold "$FOLD" --seed "$SEED"
        --set "data.index=$INDEX" --set "data.split=$SPLIT"
        --set "sampling.mode=y" --set "sampling.ky=$KY"
        --set "sampling.kz=1"
        --set "physics.psf.sigma_xy_um=$SX"
        --set "physics.psf.sigma_z_um=$SZ"
        --set "physics.psf_source=$SOURCE"
        --set "physics.lam=$LAM"
        --set "physics.forward_extra_psf.sigma_xy_um=0"
        --set "physics.forward_extra_psf.sigma_z_um=0"
        --set "physics.forward_extra_psf_source=null")
tag="E025_y_k${KY}_fold${FOLD}_s${SEED}"
run_eval () {
  local name="$1" arm="$2" ckpt="${3:-}" ctrl="${4:-none}"
  local args=(--arm "$arm" --split test --out "$OUT/${name}_test"
              "${common[@]}" --set "data.patch_hw=[512,512]"
              --set "model.phys_control=$ctrl")
  [[ -n "$ckpt" ]] && args+=(--ckpt "$ckpt")
  if [[ "$DRY" == "1" ]]; then
    printf 'DRY eval: %q ' python -m spur_mpm.evaluate "${args[@]}"
    echo; return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python -m spur_mpm.evaluate "${args[@]}"
}
run_train () {
  local name="$1" arm="$2" ctrl="$3" aw="$4"
  local args=(--arm "$arm" --name "$name" --out-root "$OUT"
              "${common[@]}" --set "data.patch_hw=[256,256]"
              --set "model.phys_control=$ctrl"
              --set "loss.branch_obs_weight=$aw"
              --set "loss.branch_phys_weight=$aw")
  if [[ "$DRY" == "1" ]]; then
    printf 'DRY train: %q ' python -m spur_mpm.train "${args[@]}"
    echo; return 0
  fi
  if [[ -e "$OUT/$name" ]]; then
    echo "Refusing existing run: $OUT/$name (archive before a different run)" >&2
    exit 3
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python -m spur_mpm.train "${args[@]}"
}
if [[ "$STAGE" == "floors" || "$STAGE" == "all" ]]; then
  run_eval "${tag}_P2_I0" P2_I0
  run_eval "${tag}_P2_W0" P2_W0
fi
if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  for variant in residual dual_interp dual_wiener; do
    case "$variant" in
      residual) arm=P2_ResidualI; ctrl=none; aux=0;;
      dual_interp) arm=P3_DualAdaptive; ctrl=interp_only; aux=0.05;;
      dual_wiener) arm=P3_DualAdaptive; ctrl=none; aux=0.05;;
    esac
    name="${tag}_$variant"
    run_train "$name" "$arm" "$ctrl" "$aux"
    run_eval "$name" "$arm" "$OUT/$name/best.pt" "$ctrl"
  done
fi
echo "E025 stage=$STAGE finished. Preserve every correct positive/negative run."

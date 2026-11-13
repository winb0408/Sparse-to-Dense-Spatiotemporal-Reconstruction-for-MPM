#!/usr/bin/env bash
# E025: 75% missing (25% retained) PR3 adaptive-prior ladder.
# This script never deletes, overwrites or modifies E024.
# DRY=1 by default; GPU runs require explicit GPU=<owned physical index>.
set -euo pipefail
cd "$(dirname "$0")/../.."
# E025 base config: y_k4 + PR3 branch auxiliary weights. E002 is kept
# only as the y_k2 headline protocol and would mislabel this run.
CFG="${CFG:-spur_mpm/configs/experiments/E025_adaptive_y_k4.yaml}"
INDEX="${INDEX:-data_local/dryad_cache/index_cohort.json}"
SPLIT="${SPLIT:-spur_mpm/experiments/E020_cohort/split.json}"
OUT="${OUT:-runs/E025_adaptive_dual_prior}"
FOLD="${FOLD:-0}"
SEED="${SEED:-0}"
KY="${KY:-4}"     # y_k4: 25% retained = 75% missing (NOT 75% retained)
# Co-tenancy knob, same meaning as in run_e026_historical.sh: BATCH x ACCUM keeps
# the effective batch at 8 while lowering peak activation memory. y_k4 with
# context 1 is far smaller than the yz group, so the default (8 x 1) is kept.
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-1}"
GPU="${GPU:-}"
DRY="${DRY:-1}"
STAGE="${STAGE:-all}" # select | floors | train | analysis | all
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
# Launch gate: the device must be visible, and GPU 2 must be under half full.
# A dead handle hangs the job instead of failing it (GPU 2, 2026-11-13).
if [[ "$DRY" != "1" ]]; then
  source spur_mpm/experiments/_gpu_gate.sh
  gpu_guard "$GPU" || exit 2
fi
# Every GPU stage is bounded: a job that stops making progress must fail loudly
# rather than spin. 9000 s is far above a 3000-step arm and a full test eval.
JOB_TIMEOUT="${JOB_TIMEOUT:-9000}"
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
        --set "physics.forward_extra_psf_source=null"
        --set "train.batch=$BATCH" --set "train.accum=$ACCUM")
tag="E025_y_k${KY}_fold${FOLD}_s${SEED}"
run_eval () {
  local name="$1" arm="$2" ckpt="${3:-}" ctrl="${4:-none}"
  local args=(--arm "$arm" --split test --out "$OUT/${name}_test"
              "${common[@]}" --set "data.patch_hw=[256,256]"
              --set "model.phys_control=$ctrl")
  [[ -n "$ckpt" ]] && args+=(--ckpt "$ckpt")
  if [[ "$DRY" == "1" ]]; then
    printf 'DRY eval: %q ' python -m spur_mpm.evaluate "${args[@]}"
    echo; return 0
  fi
  # Resume, never overwrite: skip an evaluation that already wrote its summary
  # so an interrupted group can be restarted. A directory without one is a
  # half-run and is left for a human to inspect.
  if [[ -s "$OUT/${name}_test/summary.json" ]]; then
    echo "skip eval $name: $OUT/${name}_test already has summary.json"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" timeout "$JOB_TIMEOUT" python -m spur_mpm.evaluate "${args[@]}"
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
  # A run counts as done only when the checkpoint AND its final validation
  # exist; a half-run is neither skipped nor overwritten.
  if [[ -s "$OUT/$name/final_val.json" && -s "$OUT/$name/best.pt" ]]; then
    echo "skip train $name: $OUT/$name is complete"
    return 0
  fi
  if [[ -e "$OUT/$name" ]]; then
    echo "Refusing existing run: $OUT/$name (archive before a different run)" >&2
    exit 3
  fi
  CUDA_VISIBLE_DEVICES="$GPU" timeout "$JOB_TIMEOUT" python -m spur_mpm.train "${args[@]}"
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
# ---- analysis: structure reports, comparison, qualitative figures ----------
# Same shape as run_e026_historical.sh STAGE=analysis, for the y-only protocol.
# The two groups must be analysed by the same tools with the same flags, or a
# difference between them could be a difference in how they were scored.
if [[ "$STAGE" == "analysis" || "$STAGE" == "all" ]]; then
  [[ "$DRY" == 1 || -n "$GPU" ]] || {
    echo "Set GPU=<owned GPU index> for analysis (it reconstructs volumes)" >&2
    exit 2
  }
  set_tail=("${common[@]:6}")   # common is (--config X --fold F --seed S --set ...)
  fig_dir="$OUT/figures/${tag}"
  arms=(
    "I0|P2_I0||none"
    "W0|P2_W0||none"
    "ResidualI|P2_ResidualI|$OUT/${tag}_residual/best.pt|none"
    "P3_interp|P3_DualAdaptive|$OUT/${tag}_dual_interp/best.pt|interp_only"
    "P3_wiener|P3_DualAdaptive|$OUT/${tag}_dual_wiener/best.pt|none"
  )
  mkdir -p "$OUT/logs"
  run_structure () {
    local label="$1" arm="$2" ckpt="$3" ctrl="$4"
    local dir="$OUT/structure_$label"
    local args=(--arm "$arm" --out "$dir" --device cuda:0
                --config "$CFG" "${set_tail[@]}"
                --set "model.phys_control=$ctrl")
    [[ -n "$ckpt" ]] && args+=(--ckpt "$ckpt")
    [[ -n "${SAMPLES:-}" ]] && args+=(--samples $SAMPLES)
    if [[ "$DRY" == 1 ]]; then
      printf 'DRY structure:'; printf ' %q' python -m spur_mpm.scripts.structure_report "${args[@]}"
      echo; return 0
    fi
    [[ ! -e "$dir" ]] || { echo "Refuse overwrite: $dir" >&2; return 3; }
    CUDA_VISIBLE_DEVICES="$GPU" timeout "$JOB_TIMEOUT" python -m spur_mpm.scripts.structure_report \
      "${args[@]}" > "$OUT/logs/structure_$label.log" 2>&1 || {
        echo "STRUCTURE FAILED $label (see $OUT/logs/structure_$label.log)" >&2
        return 1; }
    echo "structure $label done"
  }
  for spec in "${arms[@]}"; do
    IFS='|' read -r label arm ckpt ctrl <<< "$spec"
    if [[ -n "$ckpt" && "$DRY" != 1 && ! -s "$ckpt" ]]; then
      echo "skip structure $label: no checkpoint (arm is NOT_RUN)" >&2
      continue
    fi
    run_structure "$label" "$arm" "$ckpt" "$ctrl" || true
  done

  if [[ "$DRY" == 1 ]]; then
    for pair in "P3_wiener P3_interp" "P3_wiener I0"; do
      echo "DRY compare_structure: $pair"
    done
  else
    for pair in "P3_wiener P3_interp" "P3_wiener I0"; do
      set -- $pair
      [[ -d "$OUT/structure_$1" && -d "$OUT/structure_$2" ]] || continue
      python -m spur_mpm.scripts.compare_structure \
        --runs "$OUT/structure_$1" "$OUT/structure_$2" \
        > "$OUT/structure_compare_$1_vs_$2.md" 2>&1 \
        || echo "compare_structure $pair failed" >&2
    done
    python -m spur_mpm.scripts.collect_results \
      --pattern "$OUT/${tag}_*_test" --index "$INDEX" \
      --out "$OUT/table_${tag}.md" > "$OUT/logs/collect.log" 2>&1 \
      || echo "collect_results failed" >&2
  fi

  fig_common=(--config "$CFG" --fold "$FOLD" --index "$INDEX" "${set_tail[@]}"
              --out "$fig_dir" --device cuda:0 --error-on-missing)
  fig_a=(--arm "I0/P2_I0" --arm "W0/P2_W0")
  [[ -s "$OUT/${tag}_residual/best.pt" ]] && \
    fig_a+=(--arm "ResidI/P2_ResidualI=$OUT/${tag}_residual/best.pt")
  [[ -s "$OUT/${tag}_dual_wiener/best.pt" ]] && \
    fig_a+=(--arm "P3_W/P3_DualAdaptive=$OUT/${tag}_dual_wiener/best.pt")
  fig_b=(--arm "I0/P2_I0")
  [[ -s "$OUT/${tag}_residual/best.pt" ]] && \
    fig_b+=(--arm "ResidI/P2_ResidualI=$OUT/${tag}_residual/best.pt")
  [[ -s "$OUT/${tag}_dual_interp/best.pt" ]] && \
    fig_b+=(--arm "P3_int/P3_DualAdaptive=$OUT/${tag}_dual_interp/best.pt")
  if [[ "$DRY" == 1 ]]; then
    printf 'DRY qualfig (phys_control=none):'; printf ' %q' python -m spur_mpm.scripts.qualitative_figures "${fig_common[@]}" "${fig_a[@]}" --set "model.phys_control=none" --save-view-spec "$fig_dir/viewspec_${tag}.json"
    echo
    printf 'DRY qualfig (phys_control=interp_only):'; printf ' %q' python -m spur_mpm.scripts.qualitative_figures "${fig_common[@]}" "${fig_b[@]}" --set "model.phys_control=interp_only" --view-spec "$fig_dir/viewspec_${tag}.json" --tag _interp
    echo
  else
    mkdir -p "$fig_dir"
    CUDA_VISIBLE_DEVICES="$GPU" timeout "$JOB_TIMEOUT" python -m spur_mpm.scripts.qualitative_figures \
      "${fig_common[@]}" "${fig_a[@]}" --set "model.phys_control=none" \
      --save-view-spec "$fig_dir/viewspec_${tag}.json" \
      > "$OUT/logs/qualfig_none.log" 2>&1 \
      || echo "qualitative figures (none) failed" >&2
    CUDA_VISIBLE_DEVICES="$GPU" timeout "$JOB_TIMEOUT" python -m spur_mpm.scripts.qualitative_figures \
      "${fig_common[@]}" "${fig_b[@]}" --set "model.phys_control=interp_only" \
      --view-spec "$fig_dir/viewspec_${tag}.json" --tag _interp \
      > "$OUT/logs/qualfig_interp.log" 2>&1 \
      || echo "qualitative figures (interp_only) failed" >&2
  fi
fi

echo "E025 stage=$STAGE finished. Preserve every correct positive/negative run."

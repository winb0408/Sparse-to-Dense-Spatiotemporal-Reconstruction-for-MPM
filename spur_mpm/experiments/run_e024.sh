#!/usr/bin/env bash
# E024 -- the PR2 dual-prior ladder on the fixed cohort.
#
# This is the *first* execution of the new method generation. Nothing here has
# been run yet: the code and its unit tests are green, the GPU experiments are
# NOT_RUN (docs/PHASE1_PR2_ACCEPTANCE.md). The script exists so that "run it"
# is one reviewable command rather than a paragraph of ad-hoc invocations.
#
# Order is not negotiable (plan §6, §7): a residual arm may not be read before
# the floor under it exists, and a dual arm may not be read before the
# capacity-matched single-branch arm does.
#
#   floors    P2_I0, P2_W0                     no training, no GPU needed
#   residual  P2_Direct, P2_ResidualI          does the residual anchor buy?
#   dual      P2_DualI, P2_DualW               does the second encoder? does W0?
#   controls  P2_DualW with phys_control=zeros / shuffle
#   all       the four above, in that order
#
# Usage:
#   DRY=1 bash spur_mpm/experiments/run_e024.sh                    # what would run
#   GPUS=0,1 bash spur_mpm/experiments/run_e024.sh                 # actually run
#   STAGE=dual GPUS=0,3 SEEDS="0 1 2" bash spur_mpm/experiments/run_e024.sh
#   SKIP_DONE=1 STAGE=all GPUS=0 DRY=0 bash spur_mpm/experiments/run_e024.sh
#                                                       # fill in what is missing
#
# Three rules this script refuses to break:
#   * no GPU list => no run. A card carrying someone else's job is not a
#     benchmarking device, and quietly falling back to GPU 0 is how a timing
#     number becomes meaningless.
#   * GPU 2 is on probation. Filling it has hung and restarted this server, so
#     it is used only while under half full -- checked at launch, not assumed.
#   * DRY is the default. Writing this file must not launch anything.
set -u
cd "$(dirname "$0")/../.."

CFG="${CFG:-spur_mpm/configs/experiments/E002_supervised_y_k2.yaml}"
INDEX="${INDEX:-data_local/dryad_cache/index_cohort.json}"
SPLIT="${SPLIT:-spur_mpm/experiments/E020_cohort/split.json}"
OUT="${OUT:-runs/E024_dual_prior}"
STAGE="${STAGE:-floors}"
GPUS="${GPUS:-}"
SEEDS="${SEEDS:-0}"
FOLDS="${FOLDS:-0 1 2}"
DRY="${DRY:-1}"

# The sampling protocol is the one thing whose *name* has to follow its
# *values*: a run directory called `y_k2` produced under `yz_k2x2` is exactly
# the kind of mislabelling the results table cannot recover from. So the tag is
# derived here rather than typed into each call.
MODE="${MODE:-y}"
KY="${KY:-2}"
KZ="${KZ:-1}"
CONTEXT="${CONTEXT:-}"            # empty = whatever the config says
if [[ "$MODE" == "y" ]]; then
  TAG="y_k${KY}"
elif [[ "$MODE" == "z" ]]; then
  TAG="z_k${KZ}"
else
  TAG="${MODE}_k${KY}x${KZ}"
fi

# Extra --set overrides for anything not covered above (physics, loss, budget).
EXTRA=("$@")

read -r -a GPU_LIST <<< "${GPUS//,/ }"
if [[ "$DRY" != "1" && ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "refusing to run: set GPUS=0,1 (physical indices you own)" >&2
  exit 2
fi

# GPU 2 is on probation, not off limits. Filling it has hung this server and
# triggered an unattended restart, taking other users' jobs down with ours. So
# it may be used only while it is **less than half full**, and that is measured
# at launch time rather than assumed: "it was half free an hour ago" is exactly
# the shape of the last incident. The threshold is deliberately conservative --
# a card this one's occupancy can push to 100 % is not worth a second arm today.
#
#   GPU2_MAX_FRACTION=0.5   (default) ceiling on used/total before the launch
#   ALLOW_GPU2=1            skip the check; a visible act, not a forgotten one
GPU2_MAX_FRACTION="${GPU2_MAX_FRACTION:-0.5}"
gpu_mem () {   # gpu_mem <index> -> "<used_mib>,<total_mib>", empty if unreadable
  "${NVIDIA_SMI:-nvidia-smi}" --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}
if [[ " ${GPU_LIST[*]} " == *" 2 "* && "${ALLOW_GPU2:-0}" != "1" ]]; then
  mem="$(gpu_mem 2)"; used="${mem%%,*}"; total="${mem##*,}"
  if ! [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || (( total == 0 )); then
    echo "refusing to use GPU 2: could not read its occupancy (got '$mem')." >&2
    echo "An unreadable card is not evidence that it is free." >&2
    exit 2
  fi
  if awk -v u="$used" -v t="$total" -v f="$GPU2_MAX_FRACTION" \
       'BEGIN{exit !(u/t >= f)}'; then
    echo "refusing to use GPU 2: ${used}/${total} MiB is $(awk -v u="$used" \
         -v t="$total" 'BEGIN{printf "%.1f", 100*u/t}')%, at or above the" \
         "$(awk -v f="$GPU2_MAX_FRACTION" 'BEGIN{printf "%.0f", 100*f}')% ceiling." >&2
    echo "GPU 2 may be used only while under half full; ALLOW_GPU2=1 overrides." >&2
    exit 2
  fi
  echo "GPU 2 at ${used}/${total} MiB -- under the ceiling, allowed this launch" >&2
fi

# A filled-in ladder is the common case for a second launch: some stages have
# results, others were interrupted. SKIP_DONE=1 makes `STAGE=all` mean "finish
# what is missing" instead of "redo six hours of finished work". A run counts
# as done only when both the checkpoint and its test summary exist -- a
# checkpoint without a summary is a half-run, and re-doing it is cheaper than
# reporting a number nobody can trace.
SKIP_DONE="${SKIP_DONE:-0}"
is_done () {   # is_done <run_name>
  [[ -s "$OUT/$1/best.pt" && -s "$OUT/$1_test/summary.json" ]]
}

common=(--config "$CFG"
        --set data.index="$INDEX" --set data.split="$SPLIT"
        --set sampling.mode="$MODE" --set sampling.ky="$KY" --set sampling.kz="$KZ"
        --set data.patch_hw='[256,256]' --set train.num_workers=4)
[[ -n "$CONTEXT" ]] && common+=(--set "data.context_slices=$CONTEXT")

run_one () {   # run_one <run_name> <arm> <fold> <seed> <gpu> [extra args...]
  local name=$1 arm=$2 fold=$3 seed=$4 gpu=$5
  shift 5
  if [[ "$SKIP_DONE" == "1" && "$DRY" != "1" ]] && is_done "$name"; then
    echo "skip $name (checkpoint and test summary both present)"
    return 0
  fi
  local cmd=(python -m spur_mpm.train --arm "$arm" --fold "$fold" --seed "$seed"
             --name "$name" --out-root "$OUT" "${common[@]}" "${EXTRA[@]}"
             "$@")
  if [[ "$DRY" == "1" ]]; then
    echo "CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    echo "    then: evaluate $arm on $OUT/$name/best.pt (--split test)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "/tmp/${name}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $name (rc=$rc, see /tmp/${name}.log)" >&2
    return $rc
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -m spur_mpm.evaluate \
    --config "$CFG" --arm "$arm" --ckpt "$OUT/$name/best.pt" --fold "$fold" \
    --seed "$seed" --split test --out "$OUT/${name}_test" \
    --set data.index="$INDEX" --set data.split="$SPLIT" \
    --set sampling.mode="$MODE" --set sampling.ky="$KY" --set sampling.kz="$KZ" \
    --set data.patch_hw='[512,512]' --set train.num_workers=0 \
    ${CONTEXT:+--set data.context_slices=$CONTEXT} "${EXTRA[@]}" \
    "$@" > "/tmp/${name}_test.log" 2>&1 || {
      echo "EVAL FAILED $name (see /tmp/${name}_test.log)" >&2
      return 1
    }
  echo "done $name"
}

eval_only () {  # eval_only <run_name> <arm> <fold> <seed> <gpu> [extra...]
  # Parameter-free arms: no checkpoint, no training, just the operator.
  local name=$1 arm=$2 fold=$3 seed=$4 gpu=$5
  shift 5
  if [[ "$SKIP_DONE" == "1" && "$DRY" != "1" && -s "$OUT/${name}_test/summary.json" ]]; then
    echo "skip $name (floor summary present)"
    return 0
  fi
  local cmd=(python -m spur_mpm.evaluate --config "$CFG" --arm "$arm"
             --fold "$fold" --seed "$seed" --split test
             --out "$OUT/${name}_test" --set data.index="$INDEX"
             --set data.split="$SPLIT" --set sampling.mode="$MODE" \
             --set sampling.ky="$KY" --set sampling.kz="$KZ" \
             --set data.patch_hw='[512,512]' --set train.num_workers=0
             ${CONTEXT:+--set data.context_slices=$CONTEXT} "${EXTRA[@]}" "$@")
  if [[ "$DRY" == "1" ]]; then
    echo "CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "/tmp/${name}_test.log" 2>&1 \
    || { echo "EVAL FAILED $name (see /tmp/${name}_test.log)" >&2; return 1; }
  echo "done $name"
}

# Round-robin over whatever was handed in; never more than one job per card.
next_gpu () {
  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then echo "?"; return 0; fi   # dry run
  echo "${GPU_LIST[$(( $1 % ${#GPU_LIST[@]} ))]}"
}

fail=0
pids=()
slot=0

# --- floors: the two operators, no network -------------------------------
if [[ "$STAGE" == "floors" || "$STAGE" == "all" ]]; then
  for arm in P2_I0 P2_W0; do
    for f in $FOLDS; do
      eval_only "E024_${arm}_${TAG}_fold${f}" "$arm" "$f" 0 "$(next_gpu $slot)" \
        >/dev/null 2>&1 &
      pids+=($!); slot=$((slot+1))
    done
  done
  if [[ "$DRY" == "1" ]]; then
    for arm in P2_I0 P2_W0; do
      for f in $FOLDS; do
        eval_only "E024_${arm}_${TAG}_fold${f}" "$arm" "$f" 0 "$(next_gpu $slot)"
      done
    done
  else
    for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  fi
  echo "--- floors: P2_W0 == P2_I0 is the EXPECTED result under identity PSF ---"
fi

# --- the trained ladder --------------------------------------------------
# Two levels of ordering, and they are not the same:
#   * across stages, strictly sequential. A residual arm must not be read before
#     the floor under it, and a dual arm must not be read before the
#     capacity-matched single-branch arm -- so `dual` does not start until every
#     `residual` run has finished, pass or fail.
#   * within a stage, the arms are independent of each other (same data, same
#     budget, different model), so they are launched together and spread over
#     whatever cards were handed in. Launching them one at a time leaves the
#     second card idle for hours, which is how "no GPU available" becomes
#     self-fulfilling.
launch () {   # launch <stage-name> <arm> [extra args...]
  local stage=$1 arm=$2
  shift 2
  [[ "$STAGE" == "$stage" || "$STAGE" == "all" ]] || return 0
  # A control is the same arm with a different model, so it needs a different
  # run name -- otherwise phys_control=zeros would overwrite the headline run
  # it is supposed to be compared against.
  local tag=""
  case "$*" in
    *phys_control=zeros*)   tag="_ctrlzeros" ;;
    *phys_control=shuffle*) tag="_ctrlshuffle" ;;
  esac
  for f in $FOLDS; do
    for s in $SEEDS; do
      local name="E024_${arm}_${TAG}_fold${f}_s${s}${tag}"
      if [[ "$DRY" == "1" ]]; then
        run_one "$name" "$arm" "$f" "$s" "$(next_gpu $slot)" "$@"
      else
        run_one "$name" "$arm" "$f" "$s" "$(next_gpu $slot)" "$@" &
        pids+=($!)
      fi
      slot=$((slot+1))
    done
  done
}

wait_stage () {   # drain this stage before the next one may start
  [[ "$DRY" == "1" ]] && { pids=(); return 0; }
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  if [[ $fail -gt 0 ]]; then
    echo "stage failed ($fail run(s)); stopping -- a later stage read on top of" \
         "a missing floor is exactly the comparison this order exists to prevent" >&2
    exit 1
  fi
  pids=()
}

pids=(); slot=0
launch residual P2_Direct
launch residual P2_ResidualI
wait_stage

launch dual     P2_DualI
launch dual     P2_DualW
wait_stage

# The controls are P2_DualW with the candidate's information destroyed, same
# parameters, same budget. Not separate arms: an arm record would be a second
# thing to keep in sync with the headline one.
launch controls P2_DualW --set model.phys_control=zeros
launch controls P2_DualW --set model.phys_control=shuffle
wait_stage

# P2_DualWU is NOT launched here: under an identity PSF u_physics is identically
# zero, so the gate is a constant and the arm degenerates to P2_DualW. It gets
# launched only in a labelled non-identity group (plan §6), which does not exist
# yet.

if [[ "$DRY" == "1" ]]; then
  echo "DRY run: nothing was launched. Set DRY=0 and GPUS=<indices> to execute."
elif [[ $fail -gt 0 ]]; then
  echo "$fail run(s) failed" >&2
  exit 1
else
  echo "E024 $STAGE finished; report with:"
  echo "  python -m spur_mpm.scripts.collect_results --root $OUT"
fi

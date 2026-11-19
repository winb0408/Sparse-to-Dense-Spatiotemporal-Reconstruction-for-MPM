#!/usr/bin/env bash
# Vessel-protocol A/B driver (E026 §4.3, asked for on 2026-11-18).
#
# Re-scores the three finished groups under the vessel rule the whole-volume
# protocol uses -- per-plane threshold and the 15 % flood gate -- into
# `*_perplane_test` directories beside the published `*_test` ones. Evaluation
# only: no training, nothing already published is overwritten.
#
# Why a driver of its own: `STAGE=vessel_ab` exists in both experiment scripts,
# but the A/B spans three groups, each of which must wait for a card under the
# occupancy ceiling. This script is that queue, and it lives in the repository
# so an interrupted run can be restarted without reconstructing it from a shell
# history (`/tmp` is not durable: the first version of this file was lost with
# the session that wrote it).
#
# Usage:
#   DRY=1 bash spur_mpm/experiments/run_vessel_ab.sh          # show the steps
#   bash spur_mpm/experiments/run_vessel_ab.sh                # queue and run
#   GPU_MAX_FRACTION=0.5 bash spur_mpm/experiments/run_vessel_ab.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source spur_mpm/experiments/_gpu_gate.sh

DRY="${DRY:-0}"
TS=$(date +%Y%m%d_%H%M)
LOG="runs/VESSEL_AB_queue_${TS}.log"
mkdir -p runs
say () { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

wait_gpu () {   # first index in 0 1 3 that passes the gate; 10-minute poll
  local idx
  while true; do
    if idx=$(gpu_free_index 0 1 3); then printf '%s\n' "$idx"; return 0; fi
    sleep 600
  done
}

run_step () {   # run_step <label> <logfile> <env assignment>... -- <cmd...>
  local label="$1" log="$2"; shift 2
  local envs=() attempt rc=0
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  if [[ "$DRY" == "1" ]]; then
    printf 'DRY %s:' "$label"; printf ' %q' env "${envs[@]}" GPU='<idx>' "$@"; echo
    return 0
  fi
  for attempt in 1 2 3; do
    G=$(wait_gpu); say "$label -> GPU $G (attempt $attempt)"
    env "${envs[@]}" GPU="$G" "$@" >> "$log" 2>&1
    rc=$?
    say "$label exited rc=$rc"
    # Every stage resumes (finished arms are skipped), so a retry after a driver
    # hiccup costs only the work that did not happen.
    [[ $rc -eq 0 ]] && return 0
    say "$label failed; waiting 10 min before the next attempt"
    sleep 600
  done
  return $rc
}

say "vessel A/B start: per-plane threshold + 15 % gate, into *_perplane_test"

run_step "E026 real vessel_ab" \
  "runs/E026_psras_condition/logs/vessel_ab_real_${TS}.log" \
  DRY=0 STAGE=vessel_ab SOURCE_KIND=real -- \
  bash spur_mpm/experiments/run_e026_historical.sh

run_step "E026 matched_synthetic vessel_ab" \
  "runs/E026_psras_condition/logs/vessel_ab_msyn_${TS}.log" \
  DRY=0 STAGE=vessel_ab SOURCE_KIND=matched_synthetic -- \
  bash spur_mpm/experiments/run_e026_historical.sh

run_step "E025 vessel_ab" \
  "runs/E025_adaptive_dual_prior/logs/vessel_ab_${TS}.log" \
  DRY=0 STAGE=vessel_ab -- \
  bash spur_mpm/experiments/run_e025.sh

say "vessel A/B queue done"

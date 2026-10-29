#!/usr/bin/env bash
# Re-render every qualitative figure family from the checkpoints on disk.
#
#   bash spur_mpm/experiments/make_all_figures.sh [fold ...]     # default: 0 1 2
#
# Three arm sets per fold, each with its own tag so they do not overwrite each
# other:
#
#   (no tag)      k_y=2, A0 + the two k_y=2 trained arms
#   _with_dc1     k_y=2 plus the measurement-consistency arm of E005
#   _ky4          k_y=4, where interpolation has to invent three rows at a time
#
# The tag is also the only thing that keeps a fold's dump from another fold's:
# the volume is chosen by fold, and M1/M2/M3 belong to folds 0/1/2.
#
# One process per fold, one GPU each. Each process renders its three arm sets
# back to back; four checkpoints fit on one card comfortably and running them
# concurrently is what makes the whole sweep a coffee break instead of an hour.
set -u
cd "$(dirname "$0")/../.."

FOLDS=${*:-0 1 2}
CFG_K2=spur_mpm/configs/experiments/E003_selfsup_y_k2.yaml
CFG_K4=spur_mpm/configs/experiments/E009_sampling_y_k4.yaml
OUT=spur_mpm/figures/qualitative
mkdir -p "$OUT"

say () { echo "[$(date +%H:%M:%S)] $*"; }

run_fold () {
  local f=$1
  local dev=cuda:$1
  # S0/SS0 are the capacity controls (Y + M only, no physics features): the pair
  # that says whether a gain comes from the warm start or from the backbone, so
  # they belong in the main figure and not only in the tables.
  say "fold $f: k_y=2, base arms"
  python -m spur_mpm.scripts.qualitative_figures --fold "$f" \
    --config "$CFG_K2" \
    --arm A0 \
    --arm S0/S0=runs/E002_S0_fold$f/best.pt \
    --arm S1/S1=runs/E002_S1_fold$f/best.pt \
    --arm SS0/SS0=runs/E003_SS0_fold$f/best.pt \
    --arm SS1/SS1=runs/E003_SS1_fold$f/best.pt \
    --out "$OUT" --device "$dev" \
    || say "WARN fold $f base returned $?"

  say "fold $f: k_y=2 + dc1"
  python -m spur_mpm.scripts.qualitative_figures --fold "$f" \
    --config "$CFG_K2" \
    --arm A0 \
    --arm S1/S1=runs/E002_S1_fold$f/best.pt \
    --arm SS1/SS1=runs/E003_SS1_fold$f/best.pt \
    --arm dc1/SS1=runs/E005_dc1_fold$f/best.pt \
    --out "$OUT" --device "$dev" --tag _with_dc1 \
    || say "WARN fold $f dc1 returned $?"

  say "fold $f: k_y=4"
  python -m spur_mpm.scripts.qualitative_figures --fold "$f" \
    --config "$CFG_K4" \
    --arm A0 \
    --arm S1/S1=runs/E009_S1_fold$f/best.pt \
    --arm SS1/SS1=runs/E009_SS1_fold$f/best.pt \
    --arm dc1/SS1=runs/E009_SS1dc1_fold$f/best.pt \
    --out "$OUT" --device "$dev" --tag _ky4 \
    || say "WARN fold $f ky4 returned $?"
  say "fold $f done"
}

for f in $FOLDS; do
  run_fold "$f" > "/tmp/figs_fold${f}.log" 2>&1 &
done
wait
say "all folds returned"
grep -h "psnr=\|WARN\|Error" /tmp/figs_fold*.log | tail -40
echo ALL_DONE

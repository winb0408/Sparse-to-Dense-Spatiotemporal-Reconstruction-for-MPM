#!/usr/bin/env bash
# One volume of the five-arm y_k2 comparison (see qual_all_test_volumes.sh).
# Split out so the four held-out volumes can run as independent jobs: each writes
# filenames that carry its own sample id, so they never touch the same file.
set -euo pipefail
cd "$(dirname "$0")/../.."

SAMPLE="${1:?usage: qual_one_volume.sh <sample_id> [device]}"
DEV="${2:-cuda:2}"
OUT=runs/E021_ladder/figures/qual_y_k2
mkdir -p "$OUT"

python -u -m spur_mpm.scripts.qualitative_figures \
  --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --index data_local/dryad_cache/index_cohort.json --fold 0 \
  --sample "$SAMPLE" \
  --arm A0 \
  --arm S0=runs/E021_ladder/E021_S0_y_k2/best.pt \
  --arm S1=runs/E021_ladder/E021_S1_y_k2/best.pt \
  --arm S2=runs/E021_ladder/E021_S2_y_k2/best.pt \
  --arm S2g=runs/E021_ladder/E021_S2g_y_k2/best.pt \
  --out "$OUT" --device "$DEV" \
  --set data.split=spur_mpm/experiments/E020_cohort/split.json \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1
echo "QUAL_DONE $SAMPLE"

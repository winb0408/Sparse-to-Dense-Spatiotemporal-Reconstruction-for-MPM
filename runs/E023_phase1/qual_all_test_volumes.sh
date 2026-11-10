#!/usr/bin/env bash
# §4.4: "正式结果同时提供所有可用测试卷与失败案例".
#
# The five-arm y_k2 comparison was produced for one volume only. This runs the
# same frozen-protocol figure for every volume of the held-out animal (the whole
# of M2, four volumetric fields), so the paper figure set covers the test animal
# rather than a single chosen field -- and so the worst volume is visible next to
# the best one instead of being described in prose.
#
# Same five arms, same config, same sampling spec; only --sample changes.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=runs/E021_ladder/figures/qual_y_k2
for SAMPLE in M2_FOVII_S2_vessels M2_FOVI_S1_vessels M2_FOVI_S2_vessels; do
  echo "=== $SAMPLE ==="
  python -m spur_mpm.scripts.qualitative_figures \
    --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
    --index data_local/dryad_cache/index_cohort.json --fold 0 \
    --sample "$SAMPLE" \
    --arm A0 \
    --arm S0=runs/E021_ladder/E021_S0_y_k2/best.pt \
    --arm S1=runs/E021_ladder/E021_S1_y_k2/best.pt \
    --arm S2=runs/E021_ladder/E021_S2_y_k2/best.pt \
    --arm S2g=runs/E021_ladder/E021_S2g_y_k2/best.pt \
    --out "$OUT" --device cuda:2 \
    --set data.split=spur_mpm/experiments/E020_cohort/split.json \
    --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1
done
echo QUAL_ALL_VOLUMES_DONE

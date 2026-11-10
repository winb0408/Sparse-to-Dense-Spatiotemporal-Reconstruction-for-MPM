#!/usr/bin/env bash
# §4.5 second half: the real end-to-end cost of one whole mouse volume.
#
# A per-patch latency is not a deployment number. This runs the actual
# full-volume path (index/cache load -> window construction -> per-plane forward
# -> stacking -> np.save) for the parameter-free floor (A0) and for a trained arm
# (S1), on the same test volume, with the exact config that produced E021_S1_y_k2.
#
# A0 is included on purpose: it has no parameters, which the previous table
# rendered as "latency n/a" -- i.e. the interpolation floor appeared to cost
# nothing, which is precisely the comparison the table exists to make.
set -euo pipefail
cd "$(dirname "$0")/../.."

CFG=runs/E021_ladder/E021_S1_y_k2/config.yaml
CKPT=runs/E021_ladder/E021_S1_y_k2/best.pt
SAMPLE=M2_FOVII_S1_vessels
OUT=runs/E023_phase1/e2e
mkdir -p "$OUT"

run_one () {  # arm  ckpt("" for a baseline)
  local arm="$1" ck="$2"
  local args=(python -m spur_mpm.inference --config "$CFG" --arm "$arm"
              --sample "$SAMPLE" --out "$OUT/${arm}_${SAMPLE}.npy"
              --set run.device=cuda:2)
  [[ -n "$ck" ]] && args+=(--ckpt "$ck")
  echo "=== $arm ==="
  /usr/bin/time -f "process_wall_s %e" "${args[@]}" > "$OUT/${arm}.json.log" 2> "$OUT/${arm}.time"
  cat "$OUT/${arm}.time"
  python - "$OUT/${arm}_${SAMPLE}.json" "$arm" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
t = m["timing"]
print(f'{sys.argv[2]}: {t["wall_seconds"]} s for {t["n_planes"]} planes '
      f'({t["ms_per_plane"]} ms/plane, {t["device"]})')
PY
}

run_one A0 ""
run_one S1 "$CKPT"

python - <<'PY'
import json
from pathlib import Path
out = Path("runs/E023_phase1/e2e")
rows = []
for arm in ("A0", "S1"):
    meta = json.load(open(out / f"{arm}_M2_FOVII_S1_vessels.json"))
    proc = {}
    for line in (out / f"{arm}.time").read_text().splitlines():
        if line.startswith("process_wall_s"):
            proc["process_wall_s"] = float(line.split()[1])
    rows.append({"arm": arm, **meta["timing"], **proc,
                 "sample_id": meta["sample_id"],
                 "shape": meta["shape"],
                 "context_depth": meta["context_depth"]})
payload = {"what": "whole-volume end-to-end wall time, one process per arm",
           "device": rows[0]["device"],
           "note": "wall_seconds is timed inside reconstruct_volume (cache load "
                   "through np.save); process_wall_s adds interpreter start and "
                   "config load. Baselines have no parameters but are not free.",
           "arms": rows}
p = out / "benchmark_end_to_end.json"
p.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
print(json.dumps(payload, indent=1))
PY
echo E2E_DONE

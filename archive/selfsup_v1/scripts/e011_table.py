"""E011/E012/E012b result table, keyed by run directory rather than by arm name.

S1h is scored with ``--arm S1`` (it *is* the S1 recipe, only the input support
differs), so its summary.json says ``arm=S1``.  ``collect_results`` keys by the
arm field, which would make the two S1 rows collide and silently drop one, so
this table reads the run directories produced by ``eval_e011_e012.sh`` instead.

Usage:
    python -m spur_mpm.scripts.e011_table [--root runs_e011]
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROWS = [
    ("S1   (50% input, dense target)", "E002_S1"),
    ("S1h  (25% input, dense target)", "E011_S1h"),
    ("SS1  (25% + 25% withheld meas.)", "E003_SS1"),
    ("SS1c (dual complementary)", "E012_SS1c"),
    ("SS1c+agree (dual + L_agree)", "E012b_SS1c"),
]
METRICS = ["psnr", "psnr_measured", "psnr_missing", "dice",
           "e_measured", "e_missing"]
FOLDS = [0, 1, 2]


def load(root: pathlib.Path, prefix: str) -> dict:
    out = {}
    for f in FOLDS:
        p = root / f"{prefix}_fold{f}_test_full" / "summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            out[f] = {m: d["metrics"].get(m, {}).get("mean_of_group_means")
                      for m in METRICS}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("runs_e011"))
    args = ap.parse_args(argv)

    data = [(label, load(args.root, p)) for label, p in ROWS]
    for m in METRICS:
        print(f"\n### {m}")
        print("| arm | " + " | ".join(f"fold{f}" for f in FOLDS) + " | mean |")
        print("|---" * (len(FOLDS) + 2) + "|")
        for label, d in data:
            vals = [d[f][m] for f in FOLDS if f in d and d[f][m] is not None]
            cells = ["—" if f not in d or d[f][m] is None else f"{d[f][m]:.4f}"
                     for f in FOLDS]
            mean = f"{sum(vals) / len(vals):.4f}" if vals else "—"
            print(f"| {label} | " + " | ".join(cells) + f" | {mean} |")

    # paired deltas against SS1, the arm the complementary design has to beat
    ss1 = dict(data)["SS1  (25% + 25% withheld meas.)"]
    print("\n### paired deltas vs SS1 (per fold)")
    for label, d in data:
        if label.startswith("SS1 "):
            continue
        for m in ["psnr", "psnr_measured", "psnr_missing", "dice"]:
            diffs = [d[f][m] - ss1[f][m] for f in FOLDS
                     if f in d and f in ss1 and None not in (d[f][m], ss1[f][m])]
            if not diffs:
                continue
            sign = "".join("+" if x > 0 else "-" for x in diffs)
            print(f"{label:32s} {m:14s} Δ=" + ", ".join(f"{x:+.4f}" for x in diffs)
                  + f"  mean={sum(diffs)/len(diffs):+.4f}  signs={sign}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

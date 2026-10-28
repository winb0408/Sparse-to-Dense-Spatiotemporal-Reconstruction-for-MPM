"""Rebuild experiment tables from saved run directories.

    python -m spur_mpm.scripts.collect_results --pattern 'runs/E001_*' \
        --index data_local/dryad_cache/index.json --out runs/E001_table.md

Numbers in the paper must come from this script running over ``summary.json``
and ``samples.csv``, not from anything typed into a document. Two views:

* **main table** -- one column per run, aggregated over the fold's test animal;
* **depth bands** -- the same metrics split by physical depth (µm), because a
  mean over a whole stack hides the fact that the deepest planes are ~10x
  weaker in signal than the brightest ones, and interpolation degrades there
  first. Plane indices are converted to micrometres using the cache geometry,
  never by assuming a step.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

__all__ = ["load_runs", "main_table", "compare_arms", "cross_animal_table",
           "depth_table"]

METRICS = ("psnr", "ssim", "mae", "e_measured", "e_missing", "dice", "cl_dice")


def load_runs(pattern: str) -> list[dict]:
    """Every directory under ``pattern`` that has a ``summary.json``."""
    runs = []
    for path in sorted(Path().glob(pattern)):
        summary = path / "summary.json"
        if not summary.exists():
            continue
        payload = json.loads(summary.read_text())
        payload["_path"] = str(path)
        payload["_samples"] = str(path / "samples.csv")
        runs.append(payload)
    if not runs:
        raise SystemExit(f"no run directories matched {pattern!r}")
    return runs


def _fold_of(run: dict) -> int:
    return int(run.get("fold", 0))


def main_table(runs: list[dict]) -> str:
    arms = sorted({r["arm"] for r in runs})
    folds = sorted({_fold_of(r) for r in runs})

    lines = ["| metric | " + " | ".join(f"{a} fold{f}" for a in arms for f in folds)
             + " | " + " | ".join(f"{a} mean" for a in arms) + " |",
             "|---" * (1 + len(arms) * len(folds) + len(arms)) + "|"]

    for metric in METRICS:
        cells, means = [], []
        for arm in arms:
            per_fold = []
            for fold in folds:
                run = next((r for r in runs
                            if r["arm"] == arm and _fold_of(r) == fold), None)
                value = (run or {}).get("metrics", {}).get(metric, {}).get(
                    "mean_of_group_means")
                cells.append("—" if value is None else f"{value:.4f}")
                if value is not None:
                    per_fold.append(value)
            means.append(f"{sum(per_fold) / len(per_fold):.4f}" if per_fold else "—")
        lines.append(f"| {metric} | " + " | ".join(cells + means) + " |")
    return "\n".join(lines)


def compare_arms(runs: list[dict], arm_a: str, arm_b: str) -> str:
    """Paired per-fold delta, ``arm_b - arm_a``.

    Paired is the only comparison this dataset supports: between-animal spread
    is ~2.8 dB, far larger than any method effect, so an unpaired mean over
    three animals cannot resolve a difference. Within a fold both arms see the
    same test animal and the same sampling instance, and the animal-specific
    component cancels.

    With three PAIRS a paired t-test would be theatre. What is reported is the
    per-fold delta, the mean delta, and the sign count -- ``3/3 folds agree``
    is a fact, ``p < 0.05`` on n=3 would not be.
    """
    folds = sorted({_fold_of(r) for r in runs})
    lines = [f"| metric | " + " | ".join(f"fold{f}" for f in folds)
             + " | mean Δ | folds favouring " + arm_b + " |",
             "|---" * (2 + len(folds) + 2) + "|"]
    for metric in METRICS:
        cells, deltas, signs = [], [], 0
        for fold in folds:
            a = next((r for r in runs if r["arm"] == arm_a
                      and _fold_of(r) == fold), None)
            b = next((r for r in runs if r["arm"] == arm_b
                      and _fold_of(r) == fold), None)
            va = (a or {}).get("metrics", {}).get(metric, {}).get(
                "mean_of_group_means")
            vb = (b or {}).get("metrics", {}).get(metric, {}).get(
                "mean_of_group_means")
            if va is None or vb is None:
                cells.append("—")
                continue
            d = vb - va
            cells.append(f"{d:+.4f}")
            deltas.append(d)
            # lower is better for mae / e_*, higher for the rest
            better = d < 0 if metric in ("mae", "e_measured", "e_missing") \
                else d > 0
            signs += int(better)
        mean = f"{sum(deltas) / len(deltas):+.4f}" if deltas else "—"
        lines.append(f"| {metric} | " + " | ".join(cells + [mean,
                                                           f"{signs}/{len(deltas)}"]) + " |")
    return "\n".join(lines)


def cross_animal_table(runs: list[dict]) -> str:
    """Fold means summarised over animals, reported as mean (min-max).

    Each fold holds out one whole animal, so the folds are the animals. With
    three animals a bootstrap interval is not informative, and printing one
    would dress up a three-point sample as precision; a range is what the data
    supports.
    """
    arms = sorted({r["arm"] for r in runs})
    lines = ["| metric | arm | n animals | mean | min | max |", "|---|---|---|---|---|---|"]
    for metric in METRICS:
        for arm in arms:
            vals = [r["metrics"][metric]["mean_of_group_means"] for r in runs
                    if r["arm"] == arm and metric in r.get("metrics", {})
                    and r["metrics"][metric].get("mean_of_group_means") is not None]
            if not vals:
                lines.append(f"| {metric} | {arm} | 0 | — | — | — |")
                continue
            lines.append(f"| {metric} | {arm} | {len(vals)} | "
                         f"{sum(vals) / len(vals):.4f} | {min(vals):.4f} | "
                         f"{max(vals):.4f} |")
    return "\n".join(lines)


def _step_of(index_path: Path, sample_id: str) -> tuple[float, float]:
    payload = json.loads(index_path.read_text())
    for vol in payload["volumes"]:
        if vol["sample_id"] == sample_id:
            geom = vol.get("geometry", vol)
            origin = geom.get("depth_origin_um")
            return (float(geom.get("z_step_um") or 1.0),
                    float(origin) if origin is not None else 0.0)
    return 1.0, 0.0


DEPTH_METRICS = ("psnr", "dice", "cl_dice")


def depth_table(runs: list[dict], index_path: Path, band_um: float = 200.0) -> str:
    """Per-arm depth profile. Arms must stay apart: pooling them would report
    the average of the methods as if it were a property of the data."""
    arms = sorted({run.get("arm", "?") for run in runs})
    steps: dict[str, float] = {}
    chunks: list[str] = []

    for arm in arms:
        buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for run in runs:
            if run.get("arm", "?") != arm:
                continue
            path = Path(run["_samples"])
            if not path.exists():
                continue
            for row in csv.DictReader(open(path)):
                sid = row["sample_id"]
                if sid not in steps:
                    steps[sid] = _step_of(index_path, sid)[0]
                z_um = float(row["query_z"]) * steps[sid]
                lo = int(z_um // band_um) * int(band_um)
                band = f"{lo}-{lo + int(band_um)}"
                for metric in DEPTH_METRICS:
                    buckets[band][metric].append(float(row[metric]))

        if not buckets:
            continue
        lines = [f"### {arm}", "",
                 "| depth (µm) | n | " + " | ".join(DEPTH_METRICS) + " |",
                 "|---|---|" + "---|" * len(DEPTH_METRICS)]
        for band in sorted(buckets, key=lambda b: int(b.split("-")[0])):
            n = len(buckets[band][DEPTH_METRICS[0]])
            vals = [f"{sum(buckets[band][m]) / n:.4f}" for m in DEPTH_METRICS]
            lines.append(f"| {band} | {n} | " + " | ".join(vals) + " |")
        chunks.append("\n".join(lines))

    return "\n\n".join(chunks) if chunks else "_no per-sample rows found_"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", required=True, help="glob of run directories")
    p.add_argument("--index", type=Path, default=None,
                   help="cache index, needed for the depth table")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--band-um", type=float, default=200.0)
    p.add_argument("--compare", nargs=2, metavar=("ARM_A", "ARM_B"),
                   default=None, help="add a paired per-fold delta table")
    args = p.parse_args(argv)

    runs = load_runs(args.pattern)
    parts = [f"# {args.title}" if args.title else "# Results", "",
             f"runs: {len(runs)}", "", "## 主表（按 fold，测试动物）", "",
             main_table(runs), "",
             "## 跨动物汇总（每折留出一只鼠；n=3，报范围不报置信区间）", "",
             cross_animal_table(runs)]
    if args.compare:
        parts += ["", f"## 配对比较（{args.compare[1]} − {args.compare[0]}，逐折）", "",
                  compare_arms(runs, *args.compare)]
    if args.index:
        parts += ["", f"## 按深度分层（band {args.band_um:g} µm）", "",
                  depth_table(runs, args.index, args.band_um)]
    parts += ["", "## 运行清单", "", "| arm | fold | run | checkpoint |", "|---|---|---|---|"]
    for run in runs:
        parts.append(f"| {run['arm']} | {_fold_of(run)} | `{run['_path']}` | "
                     f"{run.get('checkpoint') or '—'} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    print("\n".join(parts[:40]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

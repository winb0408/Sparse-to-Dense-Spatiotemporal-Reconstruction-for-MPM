"""Turn the ladder's run directories into one table, with the floors under it.

    python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder

Reads only ``eval_test/summary.json`` -- never a number retyped by hand -- so a
table cannot drift from the run that produced it. Two things this table does
that a bare metric dump does not:

* **It puts the floors next to the arms.** ``A0`` (bilinear) and ``A1``
  (fill-first Wiener) need no training, so every trained arm has a
  parameter-free number directly below it. A delta with no floor under it is not
  a result, and interpolation is a strong floor on this data, so an arm that
  cannot clear A0 has not demonstrated anything.
* **It refuses small numbers.** Between-animal variance is ~2.8 dB and ~0.08
  Dice; on this fixed single-animal cohort the honest resolution is the paired
  per-sample spread, but the *claim* threshold used everywhere in this project
  is 1 dB / 0.03 Dice. Deltas under that are printed as ``~`` rather than as a
  win, so the table cannot be read as a ranking that the data does not support.

Depth-stratified conclusions are deliberately absent here: PSNR on deep planes is
dominated by the weaker signal, so structure is judged by ``dice`` / ``cl_dice``
(and by ``metrics/skeleton.py`` for the skeleton-level verdict).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

__all__ = ["collect", "render", "innovations", "budget_curve", "axis_comparison",
           "main"]

#: The resolution below which a difference is not claimed (see Project_log).
PSNR_FLOOR_DB = 1.0
DICE_FLOOR = 0.03

#: The three levels that spend the same 25 % of the voxels on a different axis,
#: so "does the axis of the sparsity matter" is asked at a constant budget.
MATCHED_LEVELS = ("y_k4", "z_k4", "yz_k2x2")

ARMS = ("A0", "A1", "S0", "S1", "S2", "S1g", "S2g")
#: metric key -> column label; ``None`` means "not in summary.json" when the
#: evaluation ran without --skeleton.
COLUMNS = (("psnr", "PSNR↑"), ("psnr_measured", "PSNRmeas↑"),
           ("psnr_missing", "PSNRmiss↑"), ("dice", "Dice↑"),
           ("cl_dice", "clDice↑"), ("f1", "skelF1↑"),
           # same prediction, equal-area operating point: Dice without the
           # calibration term. Read the pair, never this row alone.
           ("dice_area", "Dice★↑"), ("cl_dice_area", "clDice★↑"))


def collect(root: Path) -> dict[tuple[str, str], dict]:
    """``{(arm, level): summary}`` for every finished cell under ``root``."""
    cells: dict[tuple[str, str], dict] = {}
    for d in sorted(root.glob("E021_*")):
        summary = d / "eval_test" / "summary.json"
        if not summary.exists():
            continue
        arm, level = d.name[len("E021_"):].split("_", 1)
        try:
            blob = json.loads(summary.read_text())
        except json.JSONDecodeError:
            continue
        means = {k: v.get("mean_of_group_means", v.get("mean"))
                 for k, v in blob["metrics"].items() if isinstance(v, dict)}
        means["_n"] = blob.get("n_windows")
        means["_path"] = str(summary)
        cells[(arm, level)] = means
    return cells


def _fmt(value, width: int = 8) -> str:
    return f"{value:{width}.4f}" if isinstance(value, (int, float)) else " " * (width - 2) + "– "


def render(cells: dict[tuple[str, str], dict], levels: list[str] | None = None) -> str:
    levels = levels or sorted({lvl for _, lvl in cells},
                              key=lambda s: (s.split("_")[0], s))
    arms = [a for a in ARMS if any((a, lvl) in cells for lvl in levels)]
    out: list[str] = []
    for metric, label in COLUMNS:
        if not any(metric in cells.get((a, lvl), {}) for a in arms for lvl in levels):
            continue
        out.append(f"\n### {label}\n")
        out.append("| level | " + " | ".join(arms) + " |")
        out.append("|---" * (len(arms) + 1) + "|")
        for lvl in levels:
            row = [lvl] + [_fmt(cells.get((a, lvl), {}).get(metric)) for a in arms]
            out.append("| " + " | ".join(row) + " |")
        if metric == "psnr":
            out.append("\n(dB; ``–`` = not run)\n")
    return "\n".join(out)


def innovations(cells: dict[tuple[str, str], dict], level: str) -> str:
    """Paired deltas for the innovation ablation at one level."""
    base = {a: cells.get((a, level)) for a in ARMS}
    pairs = [("physics warm start (X0)", "S0", "S1"),
             ("uncertainty head (U)", "S1", "S2"),
             ("gated prior fusion", "S1", "S1g"),
             ("all three (S1g over S0)", "S0", "S1g")]
    out = [f"\n### Innovation attribution at ``{level}``\n",
           "| step | PSNR Δ (dB) | PSNRmiss Δ (dB) | Dice Δ | clDice Δ | claim |",
           "|---|---|---|---|---|---|"]
    for name, a, b in pairs:
        lo, hi = base.get(a), base.get(b)
        if not lo or not hi:
            continue
        d_psnr = hi.get("psnr", float("nan")) - lo.get("psnr", float("nan"))
        d_miss = hi.get("psnr_missing", float("nan")) - lo.get("psnr_missing", float("nan"))
        d_dice = hi.get("dice", float("nan")) - lo.get("dice", float("nan"))
        d_cl = hi.get("cl_dice", float("nan")) - lo.get("cl_dice", float("nan"))
        claimed = (d_psnr >= PSNR_FLOOR_DB) or (d_dice >= DICE_FLOOR)
        verdict = "yes" if claimed else "no (< 1 dB and < .03 Dice)"
        out.append(f"| {name} | {d_psnr:+.3f} | {d_miss:+.3f} | {d_dice:+.4f} | "
                   f"{d_cl:+.4f} | {verdict} |")
    return "\n".join(out)


def _num(value, spec: str = ".4f") -> str:
    """Format a metric, or ``–`` when the run did not produce it."""
    return format(value, spec) if isinstance(value, (int, float)) and value == value else "–"


def _fraction(level: str) -> float:
    """Measured-voxel fraction the protocol promises, straight from the spec.

    A forced-context cell is named ``<level>_c<N>``; its budget is the level's
    budget, so the suffix is stripped before the lookup.
    """
    from spur_mpm.data.sparse import LEVELS_3D
    base = level.rsplit("_c", 1)[0]
    spec = LEVELS_3D.get(base)
    return spec.ideal_fraction() if spec is not None else float("nan")


def budget_curve(cells: dict[tuple[str, str], dict], arm: str = "S1") -> str:
    """One arm across the ladder, ordered by the budget each level spends.

    A row per level, densest first, with the deltas against the densest level
    that ran. This is the table that says where the method saturates: a metric
    that stops moving between two rows is a budget the method no longer uses.
    """
    levels = [lvl for a, lvl in cells if a == arm]
    if not levels:
        return ""
    levels.sort(key=lambda s: (-_fraction(s), s))
    ref = levels[0]
    base = cells[(arm, ref)]
    out = [f"\n### Budget curve for ``{arm}``\n",
           "| level | measured % | PSNR | Dice | clDice | skelF1 | ΔDice vs "
           f"{ref} | claim |",
           "|---|---|---|---|---|---|---|---|"]
    for lvl in levels:
        m = cells[(arm, lvl)]
        d_dice = m.get("dice", float("nan")) - base.get("dice", float("nan"))
        verdict = "—" if lvl == ref else (
            "yes" if abs(d_dice) >= DICE_FLOOR else "no (< .03 Dice)")
        out.append(f"| {lvl} | {_fraction(lvl) * 100:.1f} | "
                   f"{_num(m.get('psnr'), '.2f')} | "
                   f"{_num(m.get('dice'))} | {_num(m.get('cl_dice'))} | "
                   f"{_num(m.get('f1'))} | {d_dice:+.4f} | {verdict} |")
    return "\n".join(out)


def axis_comparison(cells: dict[tuple[str, str], dict],
                    levels: tuple[str, ...] = MATCHED_LEVELS) -> str:
    """At one budget, is the axis the sparsity sits on worth a claim?

    ``y_k4`` / ``z_k4`` / ``yz_k2x2`` all measure 25 % of the voxels. If the
    spread across those three is under the floor for every arm, then the axis
    does not matter at this budget and no "Z-skipping is better" sentence may be
    written -- which is the sentence this table exists to police.
    """
    arms = [a for a in ARMS if sum((a, lvl) in cells for lvl in levels) >= 2]
    if not arms:
        return ""
    out = ["\n### Axis of the sparsity at a matched 25 % budget\n",
           "| arm | " + " | ".join(f"{k} {lvl}" for lvl in levels
                                   for k in ("Dice",)) + " | ΔDice | ΔclDice | claim |",
           "|---" * (len(levels) + 3) + "|"]
    for arm in arms:
        dice = [cells.get((arm, lvl), {}).get("dice", float("nan")) for lvl in levels]
        cl = [cells.get((arm, lvl), {}).get("cl_dice", float("nan")) for lvl in levels]
        span_dice = _span(dice)
        span_cl = _span(cl)
        claimed = span_dice >= DICE_FLOOR or span_cl >= DICE_FLOOR
        verdict = ("axis matters" if claimed else
                   "no claim (spread < .03)")
        out.append(f"| {arm} | " + " | ".join(f"{d:.4f}" for d in dice) +
                   f" | {span_dice:.4f} | {span_cl:.4f} | {verdict} |")
    return "\n".join(out)


def _span(values: list[float]) -> float:
    """Max minus min over the values that exist; ``nan`` when fewer than two."""
    real = [v for v in values if v == v]
    return max(real) - min(real) if len(real) >= 2 else float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("runs/E021_ladder"))
    p.add_argument("--levels", nargs="+", default=None)
    p.add_argument("--ablation-level", default="y_k2")
    p.add_argument("--curve-arm", default="S1",
                   help="arm whose budget curve is printed ('' to skip)")
    p.add_argument("--out", type=Path, default=None, help="write markdown here")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args(argv)

    cells = collect(args.root)
    if not cells:
        raise SystemExit(f"no finished cells under {args.root}")
    blocks = [render(cells, args.levels)]
    if args.ablation_level:
        blocks.append(innovations(cells, args.ablation_level))
    if args.curve_arm:
        blocks.append(budget_curve(cells, args.curve_arm))
    blocks.append(axis_comparison(cells))
    blocks.append(f"Floors: a difference is only claimed above {PSNR_FLOOR_DB:g} dB "
                  f"or {DICE_FLOOR:g} Dice. Source: {len(cells)} cells under "
                  f"`{args.root}`. ★ = same prediction segmented at an "
                  "equal-area operating point (`match_foreground`); it removes "
                  "the prediction's contrast drift and must be read beside the "
                  "column without a star, not instead of it.")
    text = "\n\n".join(b.strip("\n") for b in blocks if b)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nwrote {args.out}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "level"] + [c for c, _ in COLUMNS] + ["n_windows"])
            for (arm, level), m in sorted(cells.items()):
                w.writerow([arm, level] + [m.get(c) for c, _ in COLUMNS] + [m.get("_n")])
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

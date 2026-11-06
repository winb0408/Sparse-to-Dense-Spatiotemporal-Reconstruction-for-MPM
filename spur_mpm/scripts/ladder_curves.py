"""The performance-versus-sparsity curves the tables cannot show.

    python -m spur_mpm.scripts.ladder_curves --root runs/E021_ladder \
        --out runs/E021_ladder/figures/curves

`ladder.md` answers "which arm wins at each level". It cannot answer the two
questions a curve answers, and those are the ones the plan (§7 Fig. 4) asks for:

``curve_<family>.png``
    Structure against the fraction of voxels actually measured, one line per arm,
    with the parameter-free floors drawn as horizontal references. A line that
    sits on its floor is an arm the network did not improve on; a line that
    crosses another between two densities is a claim that depends on the budget.

``curve_matched_25pct.png``
    The same 25 % budget spent along three different axes (Y, Z, YZ), at equal
    7-plane Z context, per arm. The vertical axis is Dice★ -- the equal-area
    operating point -- because a threshold-calibration difference between arms
    would otherwise be read as a structural one.

Numbers come from ``ladder.csv``, which `ladder_report` writes from each cell's
``eval_test/summary.json``; nothing here is retyped. The claim floor (0.03 Dice)
is stated in the caption rather than drawn, because it constrains *differences
between arms*, not the absolute height of any one bar.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["read_curve", "fraction_of", "render_family", "render_matched", "main"]

#: Kept-voxel fraction of a level tag: ``y_k4`` keeps 1/4 of the rows, ``z_k4``
#: 1/4 of the planes, ``yz_k2x2`` both.
FAMILY_COLORS = {"A0": "#888888", "A1": "#d62728", "S0": "#1f77b4",
                 "S1": "#2ca02c", "S2": "#9467bd", "S1g": "#ff7f0e",
                 "S2g": "#8c564b"}
MATCHED = ("y_k4", "z_k4", "yz_k2x2")          # same 25 % budget, three axes
CONTROL = ("y_k4_c7", "z_k4", "yz_k2x2_c7")    # ... at equal 7-plane context
DICE_FLOOR = 0.03                              # Project_log claim threshold


def fraction_of(level: str) -> float | None:
    """Measured fraction implied by a level tag; ``None`` for ``full``/unknown."""
    tag = level.replace("_c7", "")
    if tag == "full":
        return 1.0
    kind = tag.split("_")[0]
    try:
        # ``yz_k2x2`` packs both factors into one token; ``y_k4`` carries one each.
        ks = [int(n) for tok in tag.split("_")[1:]
              for n in tok.lstrip("k").split("x")]
    except ValueError:
        # A malformed tag cannot be given an x position; dropping the point is
        # better than crashing the whole figure over one stray level name.
        return None
    if kind == "y" and len(ks) == 1:
        return 1.0 / ks[0]
    if kind == "z" and len(ks) == 1:
        return 1.0 / ks[0]
    if kind == "yz" and len(ks) == 2:
        return 1.0 / (ks[0] * ks[1])
    return None


def read_curve(path: Path) -> dict[tuple[str, str], dict]:
    """``{(arm, level): {metric: float}}`` from the ladder CSV."""
    rows = {}
    for r in csv.DictReader(path.open()):
        vals = {}
        for k, v in r.items():
            if k in ("arm", "level") or v in ("", "nan", "NaN", None):
                continue
            try:
                vals[k] = float(v)
            except (TypeError, ValueError):
                continue
        rows[(r["arm"], r["level"])] = vals
    return rows


def _family(level: str, kind: str) -> bool:
    """Is this level in the Y / Z / YZ sweep family (the ``_c7`` copy excluded)?

    The forced-context copies answer a different question (equal usable depth)
    and would sit on top of their own level at the same fraction, so plotting
    both would double-draw the same budget with two different inputs. The family
    curves therefore draw the plain level only; the ``_c7`` copies appear in the
    matched figure, where the context *is* the controlled variable.
    """
    if level.endswith("_c7"):
        return False
    return level.split("_")[0] == kind


def render_family(rows: dict, kind: str, out: Path, metric: str = "dice_area",
                  dpi: int = 200) -> Path | None:
    arms = sorted({a for a, lvl in rows if _family(lvl, kind)})
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    drew = False
    for arm in arms:
        pts = sorted((fraction_of(lvl), rows[(arm, lvl)].get(metric))
                     for a, lvl in rows
                     if a == arm and _family(lvl, kind)
                     and rows[(arm, lvl)].get(metric) is not None
                     and fraction_of(lvl) is not None)
        if not pts:
            continue
        drew = True
        xs, ys = zip(*pts)
        trained = arm.startswith("S")
        ax.plot(xs, ys, marker="o" if trained else "s",
                ls="-" if trained else "--",
                color=FAMILY_COLORS.get(arm), label=arm,
                lw=1.8 if trained else 1.2, ms=5,
                alpha=1.0 if trained else 0.75)
    if not drew:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xticks(sorted({fraction_of(lvl) for _, lvl in rows
                          if _family(lvl, kind) and fraction_of(lvl)}))
    ax.set_xticklabels([f"{f:.0%}" for f in ax.get_xticks()])
    ax.set_xlabel("voxels actually measured")
    ax.set_ylabel({"dice_area": "Dice★ (equal-area)", "dice": "Dice",
                   "cl_dice_area": "clDice★", "psnr": "PSNR (dB)"}[metric])
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="lower right")
    ax.set_title(f"{kind.upper()} sparsity sweep — dashed = parameter-free floor, "
                 f"solid = trained arm", fontsize=9.5)
    fig.text(0.01, 0.005,
             "Dice★ is the equal-area operating point, so the gap between arms is "
             "structural rather than threshold calibration; differences below "
             f"{DICE_FLOOR} Dice are inside this project's claim floor.",
             fontsize=6.2, color="0.3")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"curve_{kind}.png"
    fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    return p


def render_matched(rows: dict, out: Path, metric: str = "dice_area",
                   control: bool = True, dpi: int = 200) -> Path | None:
    """The same budget on three axes: does *where* the sparsity is matter?"""
    levels = [lvl for lvl in (CONTROL if control else MATCHED)
              if any(lvl == l for _, l in rows)]
    if len(levels) < 2:
        return None
    arms = sorted({a for a, lvl in rows if lvl in levels})
    arms = [a for a in ("A0", "A1", "S0", "S1", "S2", "S1g", "S2g") if a in arms]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    width = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        xs, ys = [], []
        for j, lvl in enumerate(levels):
            val = rows.get((arm, lvl), {}).get(metric)
            if val is None:
                continue
            xs.append(j + (i - (len(arms) - 1) / 2) * width)
            ys.append(val)
        ax.bar(xs, ys, width=width, color=FAMILY_COLORS.get(arm),
               label=arm, alpha=0.9)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{lvl}\n({fraction_of(lvl):.0%})" for lvl in levels])
    lo = min(v[metric] for k, v in rows.items()
             if k[1] in levels and v.get(metric) is not None)
    hi = max(v[metric] for k, v in rows.items()
             if k[1] in levels and v.get(metric) is not None)
    ax.set_ylim(min(lo, hi) - 0.25, hi + 0.06)
    ax.set_ylabel({"dice_area": "Dice★ (equal-area)"}.get(metric, metric))
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, ncol=4, frameon=False)
    ax.set_title(f"Same 25 % budget, three axes"
                 f"{' (7-plane context forced)' if control else ''}", fontsize=9.5)
    # The claim threshold applies to *differences between bars*, not to the bar
    # heights, so it is stated instead of being drawn as a band at the bottom of
    # the axis -- a band there reads as "bars this low are invalid", which is not
    # what the project's floor means.
    fig.text(0.01, 0.005,
             f"Bars are absolute scores. A difference between two bars smaller "
             f"than {DICE_FLOOR} Dice is inside this project's claim floor and is "
             f"not reported as a win.", fontsize=6.2, color="0.3")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"curve_matched_25pct{'_c7' if control else ''}.png"
    fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("runs/E021_ladder/ladder.csv"))
    ap.add_argument("--out", type=Path, default=Path("runs/E021_ladder/figures/curves"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--metric", default="dice_area")
    args = ap.parse_args(argv)
    rows = read_curve(args.csv)
    written = []
    for kind in ("y", "z", "yz"):
        p = render_family(rows, kind, args.out, metric=args.metric, dpi=args.dpi)
        if p:
            written.append(p)
    for control in (True, False):
        p = render_matched(rows, args.out, metric=args.metric,
                           control=control, dpi=args.dpi)
        if p:
            written.append(p)
    for p in written:
        print("wrote", p)
    if not written:
        print(f"no levels in {args.csv} matched a family; nothing drawn")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

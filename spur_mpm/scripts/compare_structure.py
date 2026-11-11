"""Put two arms' whole-volume skeleton reports side by side, volume by volume.

`structure_report.py` writes one `structure_summary.{json,md}` per arm and its
table is already long; concatenating two of them makes the reader do the
subtraction in their head, which is exactly where the deep-layer sign error in
`docs/PHASE1_PR2_E024_RESULTS.md` §4.1 came from. This script prints one block
per volume with both arms and the delta, and refuses the cases where a delta
would be misleading:

* **different volume sets** -- comparing means over different volumes is not a
  comparison. Missing volumes in one arm are named, not skipped.
* **different binarisation thresholds** -- each arm's threshold is fitted on its
  own reference; if the arms disagree, the Dice difference is partly a
  threshold difference and the table says so. (It should not happen: the
  reference is the same array.)
* **very different scored-plane shares** -- planes are gated out per volume;
  if the two arms scored different numbers of planes the Dice is over a
  different region. The share is printed in every row for this reason.

Nothing here decides whether a difference is real. On this cohort the
cross-animal spread is ~0.08 Dice, and differences below 0.03 Dice must not be
claimed; the script prints the number and leaves the verdict to the reader.

    python -m spur_mpm.scripts.compare_structure \
        --runs runs/E024_dual_prior/structure_P2_I0 \
               runs/E024_dual_prior/structure_P2_DualW
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["load", "table"]

# Dice is the headline column because a skeleton report is about structure; the
# plane share rides in the same row so the reader sees the region it was scored
# on. PSNR is deliberately absent: §4.1 of the results doc is the record of what
# happens when a structural claim is read off a photometric number.
COLUMNS = (("dice", "Dice"), ("cl_dice", "clDice"), ("f1", "skelF1"))


def load(paths: list[Path]) -> list[dict]:
    """One summary per arm, in the order given, each with its label."""
    out = []
    for p in paths:
        f = p / "structure_summary.json" if p.is_dir() else p
        if not f.exists():
            raise SystemExit(f"{f} is missing; has that arm finished?")
        s = json.loads(f.read_text())
        # The summary records the arm it was written for; using the directory
        # name instead would let a renamed directory silently mislabel a column
        # (the same failure mode as the `P2_DualW` control columns).
        out.append({"label": s.get("arm") or p.parent.name, "json": s, "dir": p})
    return out


def _pair(arms: list[dict]) -> tuple[str, list[str], list[str]]:
    """The volume order of the first arm, plus any set disagreement."""
    first = arms[0]["json"]
    order = [v["sample_id"] for v in first["volumes"]]
    problems = []
    for a in arms[1:]:
        got = [v["sample_id"] for v in a["json"]["volumes"]]
        missing = [s for s in order if s not in got]
        extra = [s for s in got if s not in order]
        if missing or extra:
            problems.append(f"{a['label']}: missing {missing or '-'}, "
                            f"extra {extra or '-'}")
    return order, problems


def _by_volume(arm: dict) -> dict[str, dict]:
    return {v["sample_id"]: v for v in arm["json"]["volumes"]}


def table(arms: list[dict]) -> str:
    order, problems = _pair(arms)
    maps = [_by_volume(a) for a in arms]
    labels = [a["label"] for a in arms]
    head = ["volume", "planes scored"]
    for lab in labels:
        head += [f"{lab} " + name for _, name in COLUMNS]
    for lab in labels[1:]:
        head += [f"Δ({lab}−{labels[0]}) " + name for _, name in COLUMNS]
    md = ["| " + " | ".join(head) + " |",
          "|" + "---|" * len(head)]

    def cell(v: dict, key: str) -> str:
        return "–" if v is None else f"{v[key]:.4f}"

    for sid in order:
        # The plane share is reported once: every arm scores the same reference
        # planes, so a second copy would only invite the reader to subtract them.
        v0 = maps[0].get(sid)
        n, gated = (v0 or {}).get("n_planes"), (v0 or {}).get("n_planes_gated")
        share = f"{n - gated}/{n}" if n else "–"
        row = [sid, share]
        for m in maps:
            v = m.get(sid)
            row += [cell(v, k) for k, _ in COLUMNS]
        for m in maps[1:]:
            v = m.get(sid)
            row += ["–" if (v is None or v0 is None)
                    else f"{v[k] - v0[k]:+.4f}" for k, _ in COLUMNS]
        md.append("| " + " | ".join(row) + " |")

    mean_row = ["**mean**", "–"]
    for a in arms:
        m = a["json"]["means"]
        mean_row += [f"{m[k]:.4f}" for k, _ in COLUMNS]
    base_means = arms[0]["json"]["means"]
    for a in arms[1:]:
        mean_row += [f"{a['json']['means'][k] - base_means[k]:+.4f}"
                     for k, _ in COLUMNS]
    md.append("| " + " | ".join(mean_row) + " |")

    notes = []
    # Compared per volume, not per arm: one summary holds a threshold per
    # volume, so an arm-level `min` would hide a single volume binarising at a
    # different level -- which is exactly where a Dice delta would be a
    # threshold artefact.
    mismatched = []
    for sid in order:
        base = (maps[0].get(sid) or {}).get("threshold")
        for a, m in zip(arms[1:], maps[1:]):
            t = (m.get(sid) or {}).get("threshold")
            if base is not None and t is not None and abs(t - base) > 1e-9:
                mismatched.append(f"{sid}: {arms[0]['label']} {base:.6g} vs "
                                  f"{a['label']} {t:.6g}")
    if mismatched:
        notes.append("**The arms binarised at different thresholds** — "
                     + "; ".join(mismatched)
                     + ". Part of every Δ above is a threshold difference, "
                     "not a structure difference.")
    if problems:
        notes.append("**Volume sets differ** — " + "; ".join(problems)
                     + ". A mean over different volumes is not a comparison.")
    shares = []
    for sid in order:
        v = maps[0].get(sid) or {}
        if v.get("n_planes"):
            shares.append(f"{sid} {v['n_planes'] - int(v['n_planes_gated'])}/"
                          f"{v['n_planes']}")
    if shares:
        notes.append("Planes scored (reference mask fills > 15 % of a plane, so "
                     "both masks are emptied there): " + ", ".join(shares)
                     + ". Every Dice above is on the planes that survived; a "
                     "volume with a third of its planes scored is reporting a "
                     "Dice on a third of its depth.")
    notes.append("These are **whole-volume** numbers at a threshold fitted on "
                 "the reference (per plane), not the window-level masked Dice "
                 "in `docs/PHASE1_PR2_E024_RESULTS.md` §2; the two are not "
                 "comparable and the threshold is not the same number. "
                 "Cross-animal spread on this cohort is ~0.08 Dice — treat "
                 "differences below 0.03 as noise.")
    # Blank line before the notes keeps the markdown from swallowing the table.
    return "\n".join(md) + "\n\n" + "\n\n".join(notes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True,
                   help="structure output directories (or their summary.json)")
    p.add_argument("--out", type=Path, default=None,
                   help="write here instead of stdout")
    args = p.parse_args(argv)
    text = table(load(args.runs))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

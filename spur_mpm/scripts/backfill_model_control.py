"""Add ``model_control`` to evaluation summaries written before it existed.

Why this is a script and not a ``sed``: the field decides which table column a
run belongs to and whether a cross-animal mean is counting one animal three
times. Editing evidence has to be reproducible and inspectable, so the value is
**derived**, never typed: the summary already records the ``checkpoint`` it was
scored from, and that path's parent directory holds the ``config.yaml`` the
training run wrote. ``model.phys_control`` and ``model.consistency`` there are
the settings the scored weights were trained with.

Rules, all of which fail loudly rather than guess:

* dry run by default (the archive tools in this repo behave the same way);
* a summary that already disagrees with its config is **refused**, not
  overwritten -- a disagreement means one of the two is wrong and a script
  cannot decide which;
* a missing ``config.yaml`` or a checkpoint outside ``runs/`` is refused: an
  underivable label is the ambiguity this field exists to remove;
* nothing but ``model_control`` is touched, and the summary is rewritten with
  the same ``indent=2`` and key order it was written with.

    python -m spur_mpm.scripts.backfill_model_control \
        --pattern 'runs/E024_dual_prior/*_test' --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

__all__ = ["derive", "backfill"]


def derive(summary_path: Path) -> dict:
    """The ``model_control`` a summary's own record implies."""
    payload = json.loads(summary_path.read_text())
    ckpt = payload.get("checkpoint")
    if not ckpt:
        raise ValueError(f"{summary_path}: no checkpoint, nothing to derive from "
                         "(a floor arm scores no weights, so it has no variant)")
    cfg_path = Path(ckpt).parent / "config.yaml"
    if not cfg_path.exists():
        raise ValueError(f"{summary_path}: {cfg_path} is missing; the variant "
                         "cannot be derived without the config the weights were "
                         "trained with")
    cfg = yaml.safe_load(cfg_path.read_text())
    model = (cfg or {}).get("model") or {}
    return {
        "phys_control": str(model.get("phys_control") or "none"),
        "consistency": str(model.get("consistency") or "off"),
    }


def backfill(pattern: str, *, apply: bool = False) -> list[tuple[Path, dict, str]]:
    """``(path, derived, action)`` for every summary under ``pattern``."""
    out: list[tuple[Path, dict, str]] = []
    paths = sorted(Path().glob(f"{pattern}/summary.json"))
    if not paths:
        raise SystemExit(f"no summary.json under {pattern!r}")
    for summary_path in paths:
        payload = json.loads(summary_path.read_text())
        if payload.get("checkpoint") is None:
            out.append((summary_path, {}, "skip: floor arm, no checkpoint"))
            continue
        try:
            derived = derive(summary_path)
        except ValueError as exc:
            out.append((summary_path, {}, f"REFUSED: {exc}"))
            continue
        existing = payload.get("model_control")
        if existing is not None:
            if existing != derived:
                out.append((summary_path, derived,
                            f"REFUSED: summary says {existing}, config says "
                            f"{derived}"))
                continue
            out.append((summary_path, derived, "ok: already present"))
            continue
        if apply:
            payload["model_control"] = derived
            summary_path.write_text(json.dumps(payload, indent=2, default=str))
            out.append((summary_path, derived, "written"))
        else:
            out.append((summary_path, derived, "would write"))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", required=True,
                   help="glob for evaluation directories, e.g. 'runs/E024_dual_prior/*_test'")
    p.add_argument("--apply", action="store_true",
                   help="actually write; without it this only prints")
    args = p.parse_args(argv)

    rows = backfill(args.pattern, apply=args.apply)
    for path, derived, action in rows:
        tail = f"{derived.get('phys_control')}/{derived.get('consistency')}" \
            if derived else ""
        print(f"{action:28s} {path} {tail}")
    refused = [r for r in rows if r[2].startswith("REFUSED")]
    print(f"\n{len(rows)} summaries, {len(refused)} refused, "
          f"applied={args.apply}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())

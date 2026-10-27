"""Build the grouped split (leave-one-mouse-out by default).

    python -m spur_mpm.scripts.make_split \
        --manifest spur_mpm/experiments/E000_data_audit/manifest.json \
        --out spur_mpm/experiments/E000_data_audit/split.json \
        --group-field mouse_id --mode lomo

Refuses to write a split that leaks between train/val/test. If there are fewer
than three independent animals it fails with an explanation rather than falling
back to random patch splitting (plan §2.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spur_mpm.data.manifest import read_manifest
from spur_mpm.data.split import SplitConfig, SplitError, make_folds, write_split

__all__ = ["build_split"]


def _usable(rows: list[dict]) -> list[dict]:
    """Rows with resolvable identity; a row with issues has no place in a split."""
    out = []
    for r in rows:
        if r.get("issues"):
            continue
        if not str(r.get("sample_id", "")).strip():
            continue
        out.append(r)
    return out


def build_split(manifest_path: str | Path, out_path: str | Path,
                cfg: SplitConfig) -> dict:
    rows = read_manifest(manifest_path)
    usable = _usable(rows)
    skipped = len(rows) - len(usable)
    if skipped:
        print(f"skipping {skipped} manifest row(s) with unresolved issues")
    if not usable:
        raise SplitError("no usable rows in the manifest")
    folds = make_folds(usable, cfg)
    return write_split(folds, cfg, out_path, rows=usable)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--group-field", default="mouse_id")
    p.add_argument("--mode", default="lomo", choices=["lomo", "holdout"])
    p.add_argument("--val-groups", type=int, default=1)
    p.add_argument("--test-groups", default="", help="comma list, holdout mode only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = SplitConfig(group_field=args.group_field, mode=args.mode,
                      val_groups=args.val_groups, seed=args.seed,
                      test_groups=[g.strip() for g in args.test_groups.split(",")
                                   if g.strip()])
    try:
        payload = build_split(args.manifest, args.out, cfg)
    except SplitError as exc:
        print(f"SPLIT FAILED: {exc}")
        return 1

    for fold in payload["folds"]:
        print(f"fold {fold['fold']}: test={fold['groups']['test']} "
              f"val={fold['groups']['val']} train={fold['groups']['train']}")
    print(f"fingerprint: {payload['fingerprint']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _usable(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Rows that may enter a split, plus the reasons the others may not.

    ``use`` is the spec's explicit exclusion (duplicate, out of scope, wrong
    axes); ``issues`` is what the audit could not resolve. Both keep a row out,
    and both are reported rather than silently dropped.
    """
    out, skipped = [], []
    for r in rows:
        sid = str(r.get("sample_id", "")).strip()
        use = str(r.get("use", "trainable") or "trainable")
        if use != "trainable":
            skipped.append(f"{sid}: use={use} ({r.get('exclude_reason') or 'no reason'})")
            continue
        if r.get("issues"):
            skipped.append(f"{sid}: unresolved issue: {r['issues']}")
            continue
        if not sid:
            skipped.append("<no sample_id>")
            continue
        out.append(r)
    return out, skipped


def build_split(manifest_path: str | Path, out_path: str | Path,
                cfg: SplitConfig) -> dict:
    rows = read_manifest(manifest_path)
    usable, skipped = _usable(rows)
    if skipped:
        print(f"skipping {len(skipped)} of {len(rows)} manifest row(s):")
        for line in skipped:
            print(f"  - {line}")
    if not usable:
        raise SplitError("no usable rows in the manifest")
    folds = make_folds(usable, cfg)
    return write_split(folds, cfg, out_path, rows=usable)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--group-field", default="mouse_id")
    p.add_argument("--inner-field", default="fov_id",
                   help="validation unit within the remaining animals ('' = whole animal)")
    p.add_argument("--mode", default="lomo", choices=["lomo", "holdout"])
    p.add_argument("--val-units", type=int, default=1)
    p.add_argument("--test-groups", default="", help="comma list, holdout mode only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = SplitConfig(group_field=args.group_field, inner_field=args.inner_field,
                      mode=args.mode, val_units=args.val_units, seed=args.seed,
                      test_groups=[g.strip() for g in args.test_groups.split(",")
                                   if g.strip()])
    try:
        payload = build_split(args.manifest, args.out, cfg)
    except SplitError as exc:
        print(f"SPLIT FAILED: {exc}")
        return 1

    for fold in payload["folds"]:
        print(f"fold {fold['fold']}:")
        print(f"  test  ({len(fold['test']):2d}) {fold['groups']['test']}  -> {fold['test']}")
        print(f"  val   ({len(fold['val']):2d}) {fold['inner_groups']['val']}")
        print(f"  train ({len(fold['train']):2d}) {fold['inner_groups']['train']}")
    print(f"fingerprint: {payload['fingerprint']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

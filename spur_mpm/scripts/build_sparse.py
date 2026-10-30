"""Build the sparse-acquisition ladder and prove the construction is correct.

    # 3-D (Dryad): Y / Z / Y+Z ladders, every volume of the fixed cohort
    python -m spur_mpm.scripts.build_sparse \
        --index data_local/dryad_cache/index_cohort.json \
        --split spur_mpm/experiments/E020_cohort/split.json \
        --out   data_local/sparse/cohort3d

    # 2-D (Y-direction only), using each volume's middle plane
    python -m spur_mpm.scripts.build_sparse \
        --index data_local/dryad_cache/index_cohort.json --as-2d \
        --out   data_local/sparse/cohort2d

What it writes
--------------
``<out>/manifest.json``            one entry per (sample, level): shape, measured
                                   fraction vs the ideal fraction, kept scan
                                   lines, the compact row/plane lattice
                                   (``start/step/count``), a SHA-1 of the mask
``<out>/masks/<level>/<id>.npz``   the support, bit-packed (``unpackbits`` it and
                                   you have ``M`` back)
``<out>/verification.json``        every check for every construction

The command exits non-zero if any check fails. The checks are the point: they
assert ``Y = M . D`` voxel for voxel, that unmeasured voxels are exactly zero,
that a kept row is kept across all of X, and that the mask is exactly the lattice
the spec declares. A reader who does not trust the reconstruction code can verify
the *input* without running the model at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from spur_mpm.data.datasets import VolumeCache
from spur_mpm.data.sparse import (
    LADDER_NOTE,
    build_sparse,
    levels_for,
    verify_construction,
    write_entry,
)
from spur_mpm.data.split import read_split
from spur_mpm.physics.acquisition import NoiseConfig
from spur_mpm.physics.psf import PSFConfig

__all__ = ["build"]


def _cohort_ids(split: dict) -> dict[int, dict[str, list[str]]]:
    out = {}
    for fold in split["folds"]:
        fi = int(fold["fold"])
        by_part = {part: sorted(fold[part]) for part in ("train", "val", "test")}
        where = {}
        for part, ids in by_part.items():
            for sid in ids:
                where[sid] = part
        out[fi] = {"where": where, **by_part}
    return out


def _print_table(rows: list[dict]) -> None:
    head = (f"{'level':10s} {'n':>3s} {'shape':>14s} {'measured':>9s} "
            f"{'ideal':>7s} {'lines':>8s} {'time':>7s}  checks")
    print(head)
    print("-" * len(head))
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["level"], []).append(row)
    for level, items in groups.items():
        shapes = sorted({tuple(i["shape"]) for i in items})
        shape_txt = str(shapes[0]) if len(shapes) == 1 else f"{len(shapes)} shapes"
        frac = np.array([i["measured_fraction"] for i in items])
        lines = np.array([i["n_scan_lines"] for i in items])
        total = np.array([i["n_scan_lines_total"] for i in items])
        ok = all(i["ok"] for i in items)
        print(f"{level:10s} {len(items):3d} {shape_txt:>14s} "
              f"{frac.mean():9.4f} {items[0]['ideal_fraction']:7.4f} "
              f"{lines.mean():8.0f} {lines.sum() / total.sum():7.3f}  "
              f"{'ok' if ok else 'FAILED'}"
              + ("" if len(items) == 1 else f"  (measured {frac.min():.3f}-{frac.max():.3f})"))


def build(index_path: Path, split_path: Path | None, out_dir: Path, *,
          as_2d: bool, fold: int, levels: list[str] | None,
          samples: list[str] | None, noise: NoiseConfig,
          psf: PSFConfig, dump_y: bool, plane: int | None,
          verbose: bool) -> int:
    cache = VolumeCache.load(index_path, fold=fold)
    if split_path is not None:
        split = read_split(split_path)
        cohort = _cohort_ids(split)[fold]
        ids = samples or cohort["train"] + cohort["val"] + cohort["test"]
    else:
        cohort = {"where": {}}
        ids = samples or sorted(cache.volumes)

    table = levels_for(2 if as_2d else 3)
    if levels:
        unknown = [name for name in levels if name not in table]
        if unknown:
            raise SystemExit(f"unknown level(s) {unknown}; available: {sorted(table)}")
        table = {k: table[k] for k in levels}

    out_dir.mkdir(parents=True, exist_ok=True)
    entries, checks = [], []
    rows: list[dict] = []

    for sid in ids:
        arr = cache.array(sid)
        dense = cache.calibration.apply(np.asarray(arr)).astype(np.float32)
        if as_2d:
            zq = int(plane) if plane is not None else dense.shape[0] // 2
            dense = np.ascontiguousarray(dense[zq])
            label = f"{sid}#z{zq}"
        else:
            label = sid

        for name, spec in table.items():
            built = build_sparse(dense, spec, psf=psf, noise=noise)
            verdict = verify_construction(dense, built["y"], built["mask"], spec,
                                          noisy=(noise.poisson_photons > 0
                                                 or noise.gaussian_sigma > 0))
            entry = write_entry(out_dir, label, name, built, write_y=dump_y)
            entry.update({"part": cohort["where"].get(sid, "unassigned"),
                          "fold": fold, "ok": verdict["ok"]})
            entries.append(entry)
            checks.append({"sample_id": label, "level": name, **verdict})
            rows.append({**entry, "ok": verdict["ok"]})
            if verbose:
                print(f"  {label:28s} {name:8s} measured={built['measured_fraction']:.4f} "
                      f"ideal={built['ideal_fraction']:.4f} ok={verdict['ok']}", flush=True)

    payload = {
        "note": LADDER_NOTE,
        "index": str(index_path), "split": str(split_path) if split_path else None,
        "fold": fold, "ndim": 2 if as_2d else 3,
        "as_2d": as_2d, "plane": plane,
        "calibration": cache.calibration.to_dict(),
        "noise": noise.to_dict(), "psf": psf.to_dict(),
        "levels": {name: spec.to_dict() for name, spec in table.items()},
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "verification.json").write_text(json.dumps(checks, indent=2))

    n_bad = sum(1 for c in checks if not c["ok"])
    print()
    print(f"{'2-D Y-only ladder' if as_2d else '3-D ladder'} "
          f"({len(ids)} image(s), fold {fold}, "
          f"noise={noise.to_dict()})")
    _print_table(rows)
    print()
    print(f"wrote {out_dir / 'manifest.json'} "
          f"({len(entries)} entries) and {(out_dir / 'verification.json')}")
    if n_bad:
        print(f"\n*** {n_bad} construction(s) FAILED verification ***", file=sys.stderr)
        for c in checks:
            if not c["ok"]:
                bad = {k: v for k, v in c.items() if v is False}
                print(f"  {c['sample_id']} {c['level']}: {bad}", file=sys.stderr)
        return 1
    print("all constructions verified: Y = M . D exactly on the support, "
          "0.0 off it, kept rows full width, lattice as declared")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--levels", nargs="*", default=None,
                   help="subset of level names (default: the whole ladder)")
    p.add_argument("--samples", nargs="*", default=None)
    p.add_argument("--as-2d", action="store_true",
                   help="treat each volume as one 2-D plane (Y-direction only)")
    p.add_argument("--plane", type=int, default=None,
                   help="which Z plane for --as-2d (default: the middle one)")
    p.add_argument("--poisson-photons", type=float, default=0.0)
    p.add_argument("--gaussian-sigma", type=float, default=0.0)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--sigma-xy-um", type=float, default=0.0,
                   help="non-zero = explicit degradation stress test, not the main line")
    p.add_argument("--dump-y", action="store_true",
                   help="also store the zero-filled measurement Y (float16)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    psf = PSFConfig(sigma_xy_um=args.sigma_xy_um,
                    voxel_xy_um=0.98, voxel_z_um=5.0)
    if args.sigma_xy_um and args.as_2d:
        psf = psf.lateral_only()
    noise = NoiseConfig(poisson_photons=args.poisson_photons,
                        gaussian_sigma=args.gaussian_sigma, seed=args.noise_seed)
    return build(args.index, args.split, args.out, as_2d=args.as_2d,
                 fold=args.fold, levels=args.levels, samples=args.samples,
                 noise=noise, psf=psf, dump_y=args.dump_y, plane=args.plane,
                 verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())

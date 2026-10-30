"""Bridge between the verified sparse construction and what the network sees.

``data/sparse.py`` builds and *verifies* the sparse acquisition. ``data/datasets.py``
builds its own mask from the same ``SamplingSpec`` at training time. Two code
paths that agree by construction is not the same thing as two code paths that
agree, and the difference matters here: the claim being made to a reader is
"the network was trained on exactly the acquisition that was verified", so it has
to be checked, not assumed.

    python -m spur_mpm.scripts.verify_dataset_masks --levels y_k2 z_k4

For every requested level and every sample in the split, at the extreme and
central windows, this rebuilds the item through the *training* code path
(``_make_item``) and compares it to the manifest's bit-packed mask cropped at the
same origin:

* the window mask must equal the crop of the full-volume mask, which is the
  property that lets patches share one lattice;
* the measured input ``Y`` must equal ``mask * calibrated dense`` on the window,
  and must be exactly ``0.0`` everywhere the mask is off.

A disagreement is a real bug (a different phase convention, an off-by-one crop,
a calibration applied on one side only) and is reported with the offending
coordinates rather than as a bare assertion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spur_mpm.data.datasets import ItemConfig, VolumeCache, _make_item
from spur_mpm.data.split import read_split
from spur_mpm.data.sparse import LEVELS_3D, load_mask
from spur_mpm.physics.psf import PSFConfig

__all__ = ["check_level", "main"]

DEFAULT_INDEX = "data_local/dryad_cache/index_cohort.json"
DEFAULT_SPLIT = "spur_mpm/experiments/E020_cohort/split.json"
DEFAULT_MANIFEST = "data_local/sparse/cohort3d/manifest.json"


def _windows(shape: tuple[int, int, int], depth: int, hw: tuple[int, int]):
    """Extreme and central windows: the borders are where crops go wrong."""
    d, h, w = depth, hw[0], hw[1]
    zs = sorted({0, (shape[0] - d) // 2, shape[0] - d})
    ys = sorted({0, (shape[1] - h) // 2, shape[1] - h})
    xs = sorted({0, (shape[2] - w) // 2, shape[2] - w})
    return [(z, y, x) for z in zs for y in ys for x in xs]


def check_level(manifest: dict, level: str, sample_ids: list[str], cache: VolumeCache,
                *, depth: int = 3, hw: tuple[int, int] = (64, 64),
                lam: float = 0.01) -> dict:
    from spur_mpm.physics.sampling import SamplingSpec

    by_key = {(e["sample_id"], e["level"]): e for e in manifest["entries"]}
    spec = LEVELS_3D[level]
    cfg = ItemConfig(sampling=spec, psf=PSFConfig(), warm_start_kind="wiener",
                     lam=lam, features=("y", "mask", "x0"), supervision="dense")
    problems: list[str] = []
    n_checked = 0
    sealed = 0
    for sid in sample_ids:
        entry = by_key.get((sid, level))
        if entry is None:
            problems.append(f"{sid}/{level}: missing from the manifest")
            continue
        full = load_mask(entry)
        arr = cache.array(sid)
        if tuple(full.shape) != tuple(arr.shape):
            problems.append(f"{sid}/{level}: mask shape {full.shape} != volume "
                            f"{arr.shape}")
            continue
        calib = cache.calibration_for(manifest.get("fold", 0))
        for z0, y0, x0 in _windows(arr.shape, depth, hw):
            raw = arr[z0:z0 + depth, y0:y0 + hw[0], x0:x0 + hw[1]]
            dense = calib.apply(raw).astype(np.float32)
            item = _make_item(dense, cache.geometry(sid), cfg, z0, y0, x0,
                              holdout_seed=0)
            want_mask = full[z0:z0 + depth, y0:y0 + hw[0], x0:x0 + hw[1]]
            n_checked += 1
            where = f"{sid}/{level} window z{z0} y{y0} x{x0}"
            # 'inputs' is features x depth: y block first, mask block second.
            y_block = item["inputs"][0:depth].numpy().astype(np.float32)
            m_block = item["inputs"][depth:2 * depth].numpy().astype(bool)
            if not np.array_equal(m_block, want_mask):
                bad = np.argwhere(m_block != want_mask)
                problems.append(
                    f"{where}: input mask differs at {len(bad)} of {m_block.size} "
                    f"voxels, first {bad[0].tolist()} "
                    f"(item {bool(m_block[tuple(bad[0])])}, "
                    f"manifest {bool(want_mask[tuple(bad[0])])})")
                continue
            if not np.array_equal(item["acq_mask"].numpy().astype(bool),
                                  want_mask[depth // 2]):
                problems.append(f"{where}: centre acq_mask disagrees with the "
                                "manifest crop")
            off = ~want_mask
            if off.any():
                bad_off = y_block[off]
                if np.any(bad_off != 0.0):
                    problems.append(
                        f"{where}: {int(np.count_nonzero(bad_off))} unmeasured "
                        f"voxels are not 0.0 (max |y| = "
                        f"{float(np.abs(bad_off).max()):.3e})")
                else:
                    sealed += int(bad_off.size)
                err = float(np.abs(y_block[want_mask] - dense[want_mask]).max())
                if err != 0.0:
                    problems.append(
                        f"{where}: measured voxels differ from the dense reference "
                        f"by up to {err:.3e}")
    return {"level": level, "windows_checked": n_checked,
            "unmeasured_voxels_verified": sealed,
            "problems": problems, "ok": not problems}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=Path(DEFAULT_INDEX))
    p.add_argument("--split", type=Path, default=Path(DEFAULT_SPLIT))
    p.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    p.add_argument("--levels", nargs="+", default=["y_k2", "y_k4", "z_k3",
                                                  "yz_k2x2"])
    p.add_argument("--samples", nargs="+", default=None,
                   help="default: every sample in the split")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    for level in args.levels:
        if level not in manifest["levels"]:
            raise SystemExit(f"{level} not in {args.manifest} "
                             f"(has {sorted(manifest['levels'])})")
        if level not in LEVELS_3D:
            raise SystemExit(f"{level} is not a 3-D level")
    holdout = read_split(args.split)
    fold = holdout["folds"][args.fold] if "folds" in holdout else holdout
    sample_ids = args.samples or sorted(fold["train"] + fold["val"] + fold["test"])

    cache = VolumeCache.load(args.index, args.fold)
    rows = [check_level(manifest, level, sample_ids, cache, depth=args.depth,
                        hw=(args.patch, args.patch)) for level in args.levels]

    for row in rows:
        status = "ok" if row["ok"] else f"{len(row['problems'])} PROBLEM(S)"
        print(f"{row['level']:10s} windows={row['windows_checked']:5d}  {status}")
        for msg in row["problems"][:8]:
            print(f"    {msg}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")
    if not all(r["ok"] for r in rows):
        return 1
    print("the training input equals the verified construction at every window "
          "checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Convert audited TIFF stacks into memory-mappable ``.npy`` caches.

    python -m spur_mpm.scripts.prepare_data \
        --manifest spur_mpm/experiments/E000_data_audit/manifest.json \
        --split    spur_mpm/experiments/E000_data_audit/split.json \
        --cache    data_local/cache \
        --index    data_local/cache/index.json

Three things happen, in this order, and the order matters:

1. every volume is re-read and its actual shape is checked against the manifest
   (an axis mix-up caught here is cheap; caught after training it is not);
2. raw pixel values are cached **without** normalization, so the calibration can
   be re-derived and inverted exactly;
3. the intensity calibration is fitted on the **training groups of one fold
   only**. Fitting it on all data would leak test-set statistics into every
   self-supervised run (plan §2.2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spur_mpm.data.datasets import CachedVolume, write_cache_index
from spur_mpm.data.geometry import VolumeGeometry
from spur_mpm.data.manifest import read_manifest, read_manifest_meta, resolve_row_path
from spur_mpm.data.normalize import IntensityCalibration
from spur_mpm.data.split import read_split

__all__ = ["prepare"]


def _load_tiff(path: Path) -> np.ndarray:
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim == 4 and arr.shape[0] == 1:      # single-channel CZYX
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D ZYX stack, got shape {arr.shape}")
    return arr


def prepare(manifest_path: str | Path, split_path: str | Path, cache_dir: str | Path,
            index_path: str | Path, calib_fold: int = 0,
            calib_out: str | Path | None = None) -> dict:
    manifest_path = Path(manifest_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    data_root = read_manifest_meta(manifest_path).get("dataset_root")
    split = read_split(split_path)
    folds = {int(f["fold"]): f for f in split["folds"]}
    if calib_fold not in folds:
        raise SystemExit(f"calib_fold {calib_fold} not in the split ({sorted(folds)})")
    train_ids = set(folds[calib_fold]["train"])

    volumes: list[CachedVolume] = []
    arrays_by_id: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    for row in rows:
        sid = str(row["sample_id"])
        if row.get("issues"):
            skipped.append(f"{sid}: {row['issues']}")
            continue
        if str(row.get("use", "trainable") or "trainable") != "trainable":
            skipped.append(f"{sid}: use={row.get('use')} "
                           f"({row.get('exclude_reason') or 'no reason'})")
            continue
        src = resolve_row_path(row, manifest_path, root=data_root)
        if not src.exists():
            skipped.append(f"{sid}: file not found at {src}")
            continue

        arr = _load_tiff(src)
        declared = row.get("shape")
        if declared:
            declared = tuple(int(v) for v in str(declared).split("x")) \
                if isinstance(declared, str) else tuple(declared)
            if declared != arr.shape:
                skipped.append(f"{sid}: shape {arr.shape} != manifest {declared}")
                continue

        geom = VolumeGeometry.from_dict({**row, "shape": arr.shape})
        geom.shape = tuple(int(v) for v in arr.shape)
        geom.validate(actual_shape=arr.shape)

        out = cache_dir / f"{sid}.npy"
        if not out.exists() or out.stat().st_size != arr.nbytes:
            np.save(out, arr)
        volumes.append(CachedVolume(sid, str(out), geom))
        arrays_by_id[sid] = arr

    if not volumes:
        detail = "\n  ".join(skipped[:20]) or "(no rows in manifest)"
        raise SystemExit(f"no volumes prepared; every row was skipped:\n  {detail}")
    if not set(train_ids) & set(arrays_by_id):
        raise SystemExit(
            f"fold {calib_fold} has no training volumes among the prepared files; "
            "the calibration must be fitted on training groups only")

    # One calibration per fold, fitted on that fold's *training* animals. A
    # single global calibration would normalise each test animal with constants
    # derived, in part, from itself.
    calib_paths: dict[int, Path] = {}
    calib_dicts: dict[int, dict] = {}
    for fi, fold in sorted(folds.items()):
        ids = [sid for sid in arrays_by_id if sid in set(fold["train"])]
        if not ids:
            raise SystemExit(f"fold {fi} has no prepared training volumes")
        calib = IntensityCalibration.fit([arrays_by_id[s] for s in ids])
        path = (Path(calib_out) if (calib_out and fi == calib_fold)
                else cache_dir / f"calibration_fold{fi}.json")
        calib.save(path)
        calib_paths[fi] = path
        calib_dicts[fi] = {**calib.to_dict(), "n_volumes": len(ids),
                           "train_ids": sorted(ids)}

    calib = IntensityCalibration.load(calib_paths[calib_fold])
    write_cache_index(volumes, calib_paths[calib_fold], index_path,
                      calibrations=calib_paths)

    summary = {
        "n_volumes": len(volumes),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "calibration": calib.to_dict(),
        "calibration_fold": calib_fold,
        "calibrations": {str(k): v for k, v in calib_dicts.items()},
        "index": str(index_path),
    }
    (Path(index_path).parent / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--calib-fold", type=int, default=0)
    p.add_argument("--calib-out", type=Path, default=None)
    args = p.parse_args(argv)

    summary = prepare(args.manifest, args.split, args.cache, args.index,
                      calib_fold=args.calib_fold, calib_out=args.calib_out)
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_volumes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

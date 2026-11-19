"""Does the historical Wiener ordering matter at E026's context length?

The E026 group carries two floors that are supposed to be different operators:
``P2_W0`` (``wiener_fill_all``: fill every hole, one global 3-D FFT) and
``P3_Wlegacy`` (``wiener_fill_first``: lateral Wiener on the measured planes,
then axial interpolation). On the E026 test volumes their per-window
predictions agree to ~1e-7, i.e. they cannot be two independent floors.

This script measures *why*, on a TRAIN volume, without training and without
scoring any test animal: for a sweep of axial context lengths T it builds the
same lattice window the dataset builds (phase from the global stack) and
compares the two warm starts on the centre plane, the only plane an evaluation
scores. It is a diagnostic, not an experiment: no arm, no checkpoint, no
method result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spur_mpm.data.datasets import ItemConfig, VolumeCache, VolumeWindowDataset
from spur_mpm.data.split import folds_by_name, read_split
from spur_mpm.models.registry import eval_features, get_arm
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import warm_start

DEFAULT_INDEX = "data_local/dryad_cache/index_cohort.json"
DEFAULT_SPLIT = "spur_mpm/experiments/E020_cohort/split.json"


def _dataset(cache, sid, geom, h, ky, kz, lam, arm_name, depth):
    arm = get_arm(arm_name)
    cfg = ItemConfig(sampling=SamplingSpec(mode="yz", ky=ky, kz=kz), psf=h,
                     psf_source="synthetic_extra_degradation", forward_psf=h,
                     forward_psf_source="synthetic_extra_degradation",
                     warm_start_kind=arm.warm_start, lam=lam,
                     features=eval_features(arm), check_leakage=False)
    return VolumeWindowDataset(cache, [sid], cfg, patch_hw=(256, 256),
                               context_depth=depth, z_stride=1)


def run(*, index: str, split: str, fold: int, lam: float, depths: list[int],
        max_windows: int, out: Path) -> dict:
    folds = folds_by_name(read_split(split))
    cache = VolumeCache.load(index, fold=fold)
    sid = [s for s in folds[fold]["train"] if s in cache][0]
    geom = cache.geometry(sid)
    known = PSFConfig(0.98, 7.5, geom.voxel_xy_um, geom.z_step_um)
    rows = []
    for depth in depths:
        ds_all = _dataset(cache, sid, geom, known, 2, 2, lam, "P2_W0", depth)
        ds_leg = _dataset(cache, sid, geom, known, 2, 2, lam, "P3_Wlegacy", depth)
        # Both datasets hold the same window list; take a spread of phases.
        idxs = np.linspace(0, len(ds_all) - 1, min(max_windows, len(ds_all)))
        diffs, ce, we, in_ws = [], [], [], []
        for i in idxs.astype(int):
            a, b = ds_all[i], ds_leg[i]
            n = depth
            m = a["inputs"][n:2 * n].bool()
            centre = int(a["center_index"])
            wa = warm_start(a["inputs"][:n], m, known, lam,
                            mode=get_arm("P2_W0").warm_start)
            wb = warm_start(a["inputs"][:n], m, known, lam,
                            mode=get_arm("P3_Wlegacy").warm_start)
            diffs.append(float((wa[centre] - wb[centre]).abs().mean()))
            ce.append(float((~m).float().mean()))
            in_ws.append(int(m[centre].any()))
        rows.append({"context_depth": depth,
                     "n_windows": int(len(idxs)),
                     "centre_mean_abs_diff_mean": float(np.mean(diffs)),
                     "centre_mean_abs_diff_max": float(np.max(diffs)),
                     "centre_measured_frac": float(np.mean(in_ws)),
                     "missing_frac": float(np.mean(ce))})
    summary = {"kind": "diagnostic", "volume": sid, "fold": fold, "lam": lam,
               "protocol": {"mode": "yz", "ky": 2, "kz": 2},
               "rows": rows,
               "note": ("Diagnostic on a TRAIN volume, no training, no test "
                        "animal scored. It shows how far the two Wiener "
                        "orderings can disagree as a function of the axial "
                        "context T the evaluation window provides.")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "operator_context.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7, 9, 15])
    p.add_argument("--max-windows", type=int, default=24)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    res = run(index=a.index, split=a.split, fold=a.fold, lam=a.lam,
              depths=a.depths, max_windows=a.max_windows, out=a.out)
    for r in res["rows"]:
        print(f"T={r['context_depth']:2d}  n={r['n_windows']:3d} "
              f"centre|dA-dB| mean={r['centre_mean_abs_diff_mean']:.3e} "
              f"max={r['centre_mean_abs_diff_max']:.3e} "
              f"centre_measured={r['centre_measured_frac']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

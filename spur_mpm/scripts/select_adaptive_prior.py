"""Select a *task-effective* Wiener prior on training volumes only.

This is not optical PSF calibration. Given a dense recorded reference D, we
hide lines according to the requested sampling protocol and select the
candidate whose reconstruction best predicts the withheld measured image.
Candidate identity is legal and must win whenever all alternatives are worse.

Do not use test samples or promote the selected effective kernel to a
calibrated optical PSF. The forward acquisition blur remains identity.

Example:
 python -m spur_mpm.scripts.select_adaptive_prior --fold 0 --ky 4 \\
   --out runs/E025_adaptive/fold0/prior_selection.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from spur_mpm.data.datasets import VolumeCache
from spur_mpm.data.sparse import measured_mask
from spur_mpm.data.split import folds_by_name, read_split
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import interp_linear, wiener_deconv_3d

DEFAULT_INDEX = "data_local/dryad_cache/index_cohort.json"
DEFAULT_SPLIT = "spur_mpm/experiments/E020_cohort/split.json"


def protocol_mask(patch: tuple[int, int, int], starts: tuple[int, int, int],
                  spec: SamplingSpec) -> torch.Tensor:
    """The measurement mask of a crop, on the *global* sampling lattice.

    A crop that starts at an odd row must sample the odd rows of the stack, not
    the odd rows of the crop: otherwise every training patch invents its own
    phase and the candidate is fitted to a protocol nobody will deploy. The
    phase therefore comes from ``starts``, not from the crop index.
    """
    ky, kz = int(spec.ky), int(spec.kz)
    z, h, w = patch
    if spec.mode == "y":
        sampled_y = ((torch.arange(h) + starts[1]) % ky) == 0
        return sampled_y[None, :, None].expand(z, h, w).clone()
    if spec.mode == "z":
        sampled_z = ((torch.arange(z) + starts[0]) % kz) == 0
        return sampled_z[:, None, None].expand(z, h, w).clone()
    if spec.mode == "yz":
        sampled_y = ((torch.arange(h) + starts[1]) % ky) == 0
        sampled_z = ((torch.arange(z) + starts[0]) % kz) == 0
        return (sampled_z[:, None, None] &
                sampled_y[None, :, None]).expand(z, h, w).clone()
    return torch.from_numpy(measured_mask(tuple(patch), spec))


def choose_prior(*, index: str, split: str, fold: int, mode: str,
                 ky: int, kz: int, patch: tuple[int, int, int],
                 n_patches: int, sigmas_xy: tuple[float, ...],
                 sigmas_z: tuple[float, ...], lambdas: tuple[float, ...],
                 seed: int = 0) -> dict:
    """Training-only held-out-image selection; returned parameters are not calibrated."""
    if n_patches <= 0 or min(patch) <= 0:
        raise ValueError("patch sizes and n_patches must be positive")
    if any(not math.isfinite(x) or x < 0 for x in (*sigmas_xy, *sigmas_z)):
        raise ValueError("invalid PSF widths")
    if any(not math.isfinite(v) or v <= 0 for v in lambdas):
        raise ValueError("lambda must be positive finite")
    folds = folds_by_name(read_split(split))
    if fold not in folds:
        raise ValueError(f"unknown fold {fold}")
    train_ids = list(folds[fold]["train"])
    test_ids = set(folds[fold]["test"])
    if set(train_ids) & test_ids:
        raise RuntimeError("train/test overlap: refusing PSF parameter selection")
    cache = VolumeCache.load(index, fold=fold)
    ids = [sid for sid in train_ids if sid in cache]
    if not ids:
        raise ValueError("no train samples in cache")
    spec = SamplingSpec(mode=mode, ky=ky, kz=kz)
    candidates = [(sx, sz, lam)
                  for sx in sigmas_xy for sz in sigmas_z for lam in lambdas]
    if not any(sx == 0 and sz == 0 for sx, sz, _ in candidates):
        raise ValueError("identity candidate mandatory")
    sums = np.zeros(len(candidates), dtype=np.float64)
    counts = np.zeros(len(candidates), dtype=np.int64)
    details: list[dict] = []
    for vol_idx, sid in enumerate(ids):
        arr = cache.calibration.apply(cache.array(sid))
        geom = cache.geometry(sid)
        if any(d > a for d, a in zip(patch, arr.shape)):
            raise ValueError(f"patch {patch} exceeds {sid} {arr.shape}")
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, fold, vol_idx]).generate_state(1)[0])
        scores = np.zeros(len(candidates), dtype=np.float64)
        for j in range(n_patches):
            starts = [int(rng.integers(0, a - d + 1))
                      for a, d in zip(arr.shape, patch)]
            sl = tuple(slice(st, st + d) for st, d in zip(starts, patch))
            dense = torch.as_tensor(np.ascontiguousarray(arr[sl]),
                                    dtype=torch.float32)
            mask = protocol_mask(patch, starts, spec)
            if not mask.any() or bool(mask.all()):
                continue
            y = torch.where(mask, dense, torch.zeros_like(dense))
            i0 = interp_linear(y, mask)
            missing = ~mask
            for k, (sx, sz, lam) in enumerate(candidates):
                psf = PSFConfig(sigma_xy_um=sx, sigma_z_um=sz,
                                voxel_xy_um=geom.voxel_xy_um,
                                voxel_z_um=geom.z_step_um)
                pred = (i0 if psf.is_identity else
                        wiener_deconv_3d(i0, psf, lam))
                err = ((pred[missing] - dense[missing]) ** 2).mean()
                if not bool(torch.isfinite(err)):
                    raise RuntimeError("non-finite candidate loss")
                scores[k] += float(err)
                counts[k] += 1
        details.append({"train_volume": sid, "n_requested_patches": n_patches})
        sums += scores
    valid = counts > 0
    if not valid.all():
        raise RuntimeError("no missing-voxel scores for one or more candidates")
    means = sums / counts
    # Strict evidence requirement: keep identity for ties and near-ties.
    order = sorted(range(len(candidates)),
                   key=lambda k: (float(means[k]), candidates[k][0] != 0 or
                                  candidates[k][1] != 0, candidates[k]))
    best_i = order[0]
    best = candidates[best_i]
    rows = [{"sigma_xy_um": sx, "sigma_z_um": sz, "lam": lam,
             "mse_missing": float(means[k]), "n_patches": int(counts[k])}
            for k, (sx, sz, lam) in enumerate(candidates)]
    identity_mse = min(row["mse_missing"] for row in rows
                       if row["sigma_xy_um"] == 0 and row["sigma_z_um"] == 0)
    out = {
        "status": "selected_on_training_volumes_only",
        "interpretation": "effective held-out prediction kernel; NOT optical PSF",
        "protocol": {"mode": mode, "ky": ky, "kz": kz,
                     "fold": fold, "seed": seed, "patch": list(patch)},
        "train_volumes": ids, "test_volumes_excluded": sorted(test_ids),
        "n_requested_patches_per_volume": n_patches,
        "candidates": rows,
        "baseline_identity_mse": identity_mse,
        "selected": {"sigma_xy_um": best[0], "sigma_z_um": best[1],
                     "lam": best[2], "mse_missing": float(means[best_i]),
                     "psf_source": ("identity" if best[:2] == (0, 0)
                                    else "estimated_from_data")},
        "warning": ("This only selects a task-effective candidate against "
                    "held-out dense-recorded pixels. It does not identify "
                    "the microscope PSF or guarantee structural improvement.")
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--mode", choices=("y", "z", "yz"), default="y")
    p.add_argument("--ky", type=int, default=4)
    p.add_argument("--kz", type=int, default=1)
    p.add_argument("--patch", type=int, nargs=3, default=(7, 128, 128))
    p.add_argument("--n-patches", type=int, default=4)
    p.add_argument("--sigmas-xy", type=float, nargs="+",
                   default=(0.0, 0.2, 0.4, 0.8, 1.2))
    p.add_argument("--sigmas-z", type=float, nargs="+", default=(0.0,))
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=(0.001, 0.01, 0.1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    res = choose_prior(index=args.index, split=args.split, fold=args.fold,
                       mode=args.mode, ky=args.ky, kz=args.kz,
                       patch=tuple(args.patch), n_patches=args.n_patches,
                       sigmas_xy=tuple(args.sigmas_xy),
                       sigmas_z=tuple(args.sigmas_z),
                       lambdas=tuple(args.lambdas), seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"out": str(args.out), "selected": res["selected"],
                      "identity_mse": res["baseline_identity_mse"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

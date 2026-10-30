"""Prove, on real data, that the self-supervised path cannot see its targets.

The claim "zero dense labels" is only worth the test behind it. This script
loads the *real* prepared cache and, for every fold's held-out animal, checks
three things a broken implementation would fail:

1. **Blindness.** Locate the holdout voxels, overwrite those voxels of the dense
   image with garbage, rebuild the item with the same everything, and require
   the network inputs to be bit-identical. Every feature the arm can read is in
   that tensor (``y / mask / x0``; ``U`` is derived from ``y`` and ``mask``), so
   an unchanged input means the holdout cannot reach the model through any
   channel.
2. **The targets are measurements.** The centre-plane holdout target must equal
   the acquired value at those positions -- not a dense reference (that would
   make the arm supervised) and not zero (that would train the network to
   predict black).
3. **The partition is honest.** ``M_in`` and ``M_hold`` partition ``M_acq``
   exactly: disjoint, and together equal to the sampling lattice the protocol
   specifies, at the measured fraction.

Real data, not the synthetic smoke set -- said out loud because a leakage test
that only runs on generated volumes proves nothing about this deposit. Writes a
JSON artefact and exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spur_mpm.data.datasets import ItemConfig, VolumeCache, _make_item
from spur_mpm.data.split import read_split
from spur_mpm.physics.acquisition import AcquisitionConfig, NoiseConfig, observe
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.ssl.holdout import HoldoutSpec, split_holdout


def _config(sampling: SamplingSpec, ratio: float, geometry: str) -> ItemConfig:
    return ItemConfig(
        sampling=sampling,
        psf=PSFConfig(),
        noise=NoiseConfig(),
        warm_start_kind="wiener",
        features=("y", "mask", "x0"),
        supervision="self",
        holdout=HoldoutSpec(ratio=ratio, geometry=geometry, seed=0),
    )


def _window(dense: np.ndarray, z: int, depth: int) -> tuple[np.ndarray, int]:
    z0 = max(0, min(z - depth // 2, dense.shape[0] - depth))
    return dense[z0:z0 + depth], z0


def check_volume(cache: VolumeCache, sid: str, cfg: ItemConfig, depth: int,
                 seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    # Same normalisation the training dataset applies, with the fold's own
    # calibration: the check must run on exactly what the model would see.
    dense = cache.calibration.apply(cache.array(sid)).astype(np.float32)
    geom = cache.geometry(sid)
    spec = cfg.sampling
    problems: list[str] = []
    fractions: list[float] = []
    planes = sorted({dense.shape[0] // 4, dense.shape[0] // 2,
                     3 * dense.shape[0] // 4})

    for z in planes:
        patch, z0 = _window(dense, z, depth)
        # Full-width window, so the lateral phase is the global one (a 256 patch
        # aligned at y0=0 shares the same lattice). The Z phase must follow the
        # stack, otherwise the lattice jumps between planes.
        y0 = x0 = 0
        phase = (z0 % max(spec.kz, 1), 0, 0)

        for seed in seeds:
            base = _make_item(patch, geom, cfg, z0, y0, x0, holdout_seed=seed)

            # --- where the holdout is, recovered from the public pieces ------
            acq = observe(torch.from_numpy(patch),
                          AcquisitionConfig(sampling=spec, psf=cfg.psf,
                                            noise=cfg.noise, phase=phase))
            if not torch.equal(acq.mask[base["center_index"]].float(),
                               base["acq_mask"]):
                problems.append(f"{sid} z={z} seed={seed}: M_acq disagrees with "
                                "the sampling lattice the item reports")
            spec_h = HoldoutSpec(ratio=cfg.holdout.ratio,
                                 geometry=cfg.holdout.geometry, seed=seed)
            m_in, m_hold = split_holdout(acq.mask, spec_h)
            if not torch.equal(m_hold[base["center_index"]].float(),
                               base["hold_mask"]):
                problems.append(f"{sid} z={z} seed={seed}: centre-plane holdout "
                                "differs from the full-window split")
            if not bool(m_hold.any()):
                problems.append(f"{sid} z={z} seed={seed}: empty holdout")
                continue
            fractions.append(float(m_hold.sum()) / max(1, int(acq.mask.sum())))

            # --- 1) blindness: garbage where the targets are, inputs unchanged
            hm3 = m_hold.numpy()
            poisoned = patch.copy()
            # Always differs from the original, so the test cannot pass vacuously.
            poisoned[hm3] = (poisoned[hm3] + 0.37) % 1.0
            polluted = _make_item(poisoned, geom, cfg, z0, y0, x0,
                                  holdout_seed=seed)
            if not torch.equal(base["inputs"], polluted["inputs"]):
                delta = (base["inputs"] - polluted["inputs"]).abs()
                which = int((delta.sum(dim=(-2, -1)) > 0).nonzero()[0]) \
                    if delta.any() else -1
                problems.append(
                    f"{sid} z={z} seed={seed}: network inputs changed when only "
                    f"holdout voxels changed (first affected channel {which})")
            # --- 1b) positive control: poisoned *measured* voxels must be seen
            # A leakage test that cannot fail proves nothing. Perturb voxels that
            # are in M_in but not in the holdout; the inputs have to move.
            visible = (m_in & ~m_hold)
            for zz in range(visible.shape[0]):
                if not bool(visible[zz].any()):
                    continue
                idx = visible[zz].nonzero()
                pick = idx[: min(64, idx.shape[0])]
                salted = patch.copy()
                salted[zz, pick[:, 0], pick[:, 1]] = (
                    salted[zz, pick[:, 0], pick[:, 1]] + 0.37) % 1.0
                salted_item = _make_item(salted, geom, cfg, z0, y0, x0,
                                         holdout_seed=seed)
                if torch.equal(base["inputs"], salted_item["inputs"]):
                    problems.append(
                        f"{sid} z={z} seed={seed}: sensitivity control failed -- "
                        "perturbing measured voxels left the inputs unchanged, "
                        "so the blindness result below means nothing")

            # --- 2) the target is the acquired value at those positions -----
            ht, hm = base["hold_target"], base["hold_mask"].bool()
            if not torch.allclose(ht[hm], torch.from_numpy(
                    patch[base["center_index"].item()])[hm], rtol=0, atol=0):
                problems.append(f"{sid} z={z} seed={seed}: holdout target is not "
                                "the acquired measurement")
            if not bool((ht[hm] != 0).any()):
                problems.append(f"{sid} z={z} seed={seed}: holdout targets are "
                                "all zero -- MMS would train the network to "
                                "predict black")

            # --- 3) partition honesty ----------------------------------------
            if bool((m_in & m_hold).any()):
                problems.append(f"{sid} z={z} seed={seed}: M_in and M_hold "
                                "overlap")
            if not torch.equal(m_in | m_hold, acq.mask):
                problems.append(f"{sid} z={z} seed={seed}: M_in|M_hold != M_acq")

    return {"sample_id": sid, "n_planes": len(planes), "n_seeds": len(seeds),
            "holdout_fraction": (sum(fractions) / len(fractions)) if fractions
            else 0.0, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--geometry", default="row")
    p.add_argument("--ky", type=int, default=2)
    p.add_argument("--context", type=int, default=1)
    p.add_argument("--roles", nargs="*", default=["train", "val", "test"],
                   help="which split roles to probe")
    args = p.parse_args(argv)

    split = read_split(args.split)
    folds = ([int(f["fold"]) for f in split["folds"]] if args.folds is None
             else args.folds)

    results, all_problems = [], []
    for fold in folds:
        entry = next(f for f in split["folds"] if int(f["fold"]) == fold)
        cache = VolumeCache.load(args.index, fold=fold)   # fold's own calibration
        cfg = _config(SamplingSpec(mode="y", ky=args.ky), args.ratio,
                      args.geometry)
        # Every volume a self-supervised arm could read, not just the test
        # animal: a leak into the training volumes is just as fatal and would
        # never appear in a test-set number.
        for role in args.roles:
            for sid in entry.get(role, []):
                res = check_volume(cache, sid, cfg, args.context)
                res.update({"fold": fold, "role": role})
                results.append(res)
                all_problems += res["problems"]

    payload = {
        "data": "real",
        "artifacts": {"index": str(args.index), "split": str(args.split)},
        "sampling": {"mode": "y", "ky": args.ky},
        "holdout": {"ratio": args.ratio, "geometry": args.geometry},
        "context_depth": args.context,
        "volumes": results,
        "problems": all_problems,
        "ok": not all_problems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    fracs = [r["holdout_fraction"] for r in results if r["n_planes"]]
    print(f"volumes checked: {len(results)}  planes/volume: "
          f"{results[0]['n_planes'] if results else 0}")
    if fracs:
        print(f"holdout share of M_acq: {min(fracs):.3f}-{max(fracs):.3f} "
              f"(requested {args.ratio})")
    if all_problems:
        for prob in all_problems[:20]:
            print("  VIOLATION:", prob)
        print(f"FAILED: {len(all_problems)} violation(s); wrote {args.out}")
        return 1
    print(f"OK: holdout is blind on real data; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

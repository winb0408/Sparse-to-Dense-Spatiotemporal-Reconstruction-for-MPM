"""Estimate an effective PSF for a cohort that ships no optical calibration.

    python -m spur_mpm.scripts.estimate_psf \
        --index data_local/dryad_cache/index_cohort.json \
        --split spur_mpm/experiments/E020_cohort/split.json \
        --fold 0 --out runs/E024_dual_prior/psf_estimate

Two estimators, reported side by side because they answer different questions
(see :mod:`spur_mpm.physics.psf_estimate`):

1. **arithmetic** -- the diffraction limit implied by the excitation wavelength
   and an *assumed* NA. The deposit does not contain the NA, so this branch runs
   over a small grid of plausible objectives and prints them all; picking one
   would be an unmarked assumption dressed as a measurement.
2. **held-out prediction** -- for a grid of widths, deconvolve the interpolated
   observation and measure how well it predicts the rows the sampling protocol
   skipped. Those skipped rows are real measurements, so this needs no dense
   reference beyond what the scan already has, and it involves **training
   animals only**: the script refuses a fold whose volumes it cannot verify are
   outside the test group.

The output directory receives ``psf_estimate.json`` (everything, machine
readable) and ``psf_estimate.md`` (the table a reader actually looks at). No
config file is edited by this script: promoting an estimate into a run is an
explicit ``--set physics.psf...`` plus ``--set physics.psf_source=estimated_from_data``,
so an estimated optics never silently becomes the default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spur_mpm.data.datasets import VolumeCache
from spur_mpm.data.sparse import measured_mask
from spur_mpm.data.split import folds_by_name, read_split
from spur_mpm.physics.psf_estimate import (
    best_sigma,
    diffraction_limited_sigma,
    holdout_width_scan,
)
from spur_mpm.physics.sampling import SamplingSpec

__all__ = ["estimate_psf"]

#: Objectives worth printing for a 1300 nm-class three-photon setup. The
#: deposit does not say which one was used, so none of them is "the" answer.
NA_CANDIDATES = (0.8, 1.0, 1.05)
N_MEDIUM = 1.33          # water / brain, both ~1.33
N_PHOTON = 3


def _patches(volume: np.ndarray, n: int, shape: tuple[int, int, int],
             seed: int) -> list[np.ndarray]:
    """Deterministic random crops, skipping crops that are almost all black."""
    z, h, w = shape
    if volume.shape[0] < z or volume.shape[1] < h or volume.shape[2] < w:
        raise ValueError(f"volume {volume.shape} is smaller than patch {shape}")
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        z0 = int(rng.integers(0, volume.shape[0] - z + 1))
        h0 = int(rng.integers(0, volume.shape[1] - h + 1))
        w0 = int(rng.integers(0, volume.shape[2] - w + 1))
        crop = np.asarray(volume[z0:z0 + z, h0:h0 + h, w0:w0 + w], dtype=np.float32)
        # A crop that is >99 % background says nothing about the optics and would
        # dilute the aggregate by voting for "no blur needed".
        if float((crop > 0.05).mean()) < 0.01:
            continue
        out.append(crop)
    if not out:
        raise ValueError("every candidate crop was empty; check the calibration")
    return out


def _aggregate(scans: list[list[dict]]) -> list[dict]:
    """Mean/median of the held-out scores across patches, one row per sigma."""
    keys = [(r["axis"], r["sigma_um"]) for r in scans[0]]
    rows = []
    for i, (axis, sigma) in enumerate(keys):
        psnr = np.array([s[i]["psnr"] for s in scans], dtype=float)
        delta = np.array([s[i]["delta_psnr_vs_identity"] for s in scans], dtype=float)
        changed = np.array([s[i]["changed_fraction"] for s in scans], dtype=float)
        rows.append({
            "axis": axis,
            "sigma_um": sigma,
            "sigma_px": scans[0][i]["sigma_px"],
            "is_identity": scans[0][i]["is_identity"],
            "psnr_mean": float(psnr.mean()),
            "psnr_median": float(np.median(psnr)),
            "delta_psnr_mean": float(delta.mean()),
            "delta_psnr_median": float(np.median(delta)),
            "changed_fraction_mean": float(changed.mean()),
            "n_patches": len(scans),
        })
    return rows


def estimate_psf(index: str | Path, split: str | Path, *, fold: int = 0,
                 mode: str = "y", ky: int = 2, kz: int = 1,
                 patch: tuple[int, int, int] = (32, 256, 256), n_patches: int = 6,
                 lam: float = 0.01, sigmas_xy=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
                                               1.5, 2.0, 3.0),
                 sigmas_z=(0.0, 1.0, 2.0, 3.0, 5.0, 8.0),
                 device: str = "cpu", seed: int = 0) -> dict:
    split_payload = read_split(split)
    folds = folds_by_name(split_payload)
    if fold not in folds:
        raise SystemExit(f"fold {fold} is not in {split} (have {sorted(folds)})")
    train_ids = list(folds[fold]["train"])
    test_ids = set(folds[fold]["test"])

    cache = VolumeCache.load(index, fold=fold)
    usable = [s for s in train_ids if s in cache]
    if not usable:
        raise SystemExit(f"none of the fold-{fold} training volumes are in {index}")
    leaked = [s for s in usable if s in test_ids]
    if leaked:
        raise SystemExit(f"refusing to estimate from test volumes: {leaked}")

    spec = SamplingSpec(mode=mode, ky=ky, kz=kz)
    scans_xy: list[list[dict]] = []
    scans_z: list[list[dict]] = []
    per_volume: dict[str, dict] = {}

    for sample_id in usable:
        geometry = cache.geometry(sample_id)
        dense = cache.calibration.apply(cache.array(sample_id))
        crops = _patches(dense, n_patches, patch, seed=seed)
        vol_xy, vol_z = [], []
        for crop in crops:
            t = torch.from_numpy(np.ascontiguousarray(crop)).to(device)
            mask = torch.from_numpy(
                measured_mask(tuple(crop.shape), spec)).to(device)
            vol_xy.append(holdout_width_scan(
                t, mask, axis="xy", sigmas_um=sigmas_xy, lam=lam,
                voxel_xy_um=geometry.voxel_xy_um, voxel_z_um=geometry.z_step_um))
            vol_z.append(holdout_width_scan(
                t, mask, axis="z", sigmas_um=sigmas_z, lam=lam,
                voxel_xy_um=geometry.voxel_xy_um, voxel_z_um=geometry.z_step_um))
        scans_xy.extend(vol_xy)
        scans_z.extend(vol_z)
        per_volume[sample_id] = {
            "n_patches": len(crops),
            "best_xy": best_sigma(_aggregate(vol_xy)),
            "best_z": best_sigma(_aggregate(vol_z)),
        }

    agg_xy = _aggregate(scans_xy)
    agg_z = _aggregate(scans_z)
    arithmetic = [diffraction_limited_sigma(
        excitation_nm=1340, na=na, n_medium=N_MEDIUM, n_photon=N_PHOTON,
        voxel_xy_um=0.98, voxel_z_um=5.0) for na in NA_CANDIDATES]

    return {
        "protocol": {"fold": fold, "split": str(split), "index": str(index),
                     "sampling": {"mode": mode, "ky": ky, "kz": kz},
                     "patch": list(patch), "n_patches": n_patches,
                     "lam": lam, "seed": seed,
                     "volumes": usable, "test_ids": sorted(test_ids)},
        "arithmetic": arithmetic,
        "heldout_xy": agg_xy,
        "heldout_z": agg_z,
        "best_xy": best_sigma(agg_xy),
        "best_z": best_sigma(agg_z),
        "per_volume": per_volume,
    }


def _md(result: dict) -> str:
    p = result["protocol"]
    lines = [
        "# Effective PSF estimate (no calibration exists for this deposit)",
        "",
        f"- fold {p['fold']}, sampling `{p['sampling']['mode']}` k={p['sampling']['ky']}x"
        f"{p['sampling']['kz']}, {p['n_patches']} patches of "
        f"{'x'.join(str(v) for v in p['patch'])} per volume, lambda {p['lam']}",
        f"- training volumes only: {', '.join(p['volumes'])}",
        "",
        "## 1. Arithmetic from assumed optics",
        "",
        "The deposit ships no NA (README: \"imaging parameters are described in "
        "the manuscript Methods\"). Every row below is therefore an **assumption**, "
        "printed so that none of them can be quietly picked.",
        "",
        "| NA | sigma_xy (um) | sigma_z (um) | sigma_xy (voxels) | sigma_z (voxels) |",
        "|---|---|---|---|---|",
    ]
    for a in result["arithmetic"]:
        lines.append(
            f"| {a['na']:.2f} | {a['sigma_xy_um']:.3f} | {a['sigma_z_um']:.3f} | "
            f"{a['sigma_xy_px']:.3f} | {a['sigma_z_px']:.3f} |")
    lines += [
        "",
        "A PSF narrower than one voxel cannot be recovered from sampled data, and "
        "on this grid both columns land below 1 -- the sampling, not the optics, "
        "is what limits these stacks.",
        "",
        "## 2. Width that best predicts the withheld rows",
        "",
        "`delta` is held-out PSNR minus the interpolation baseline (sigma = 0).",
        "`changed` is the fraction of voxels the deconvolution actually moved: a",
        "row with `changed ~ 0` was skipped, not tested.",
        "",
        "### lateral (sigma_z = 0)",
        "",
        "| sigma_xy (um) | px | PSNR mean | delta PSNR | changed |",
        "|---|---|---|---|---|",
    ]
    for r in result["heldout_xy"]:
        lines.append(f"| {r['sigma_um']:.2f} | {r['sigma_px']:.3f} | "
                     f"{r['psnr_mean']:.4f} | {r['delta_psnr_mean']:+.4f} | "
                     f"{r['changed_fraction_mean']:.4f} |")
    lines += ["", "### axial (sigma_xy = 0)", "",
              "| sigma_z (um) | px | PSNR mean | delta PSNR | changed |",
              "|---|---|---|---|---|"]
    for r in result["heldout_z"]:
        lines.append(f"| {r['sigma_um']:.2f} | {r['sigma_px']:.3f} | "
                     f"{r['psnr_mean']:.4f} | {r['delta_psnr_mean']:+.4f} | "
                     f"{r['changed_fraction_mean']:.4f} |")
    bx, bz = result["best_xy"], result["best_z"]
    lines += [
        "",
        "## Reading",
        "",
        f"- best lateral width: sigma_xy = {bx['sigma_um']:.2f} um "
        f"({bx['delta_psnr_mean']:+.4f} dB vs interpolation)",
        f"- best axial width: sigma_z = {bz['sigma_um']:.2f} um "
        f"({bz['delta_psnr_mean']:+.4f} dB vs interpolation)",
        "- A best width at 0.00 means the withheld rows are predicted best by *not* "
        "deconvolving. That is an estimate of identity, not a failure of the search.",
        "- Nothing here is a calibration. Any run using these numbers must set "
        "`physics.psf_source=estimated_from_data`; refusing to do so is the point.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="data_local/dryad_cache/index_cohort.json")
    ap.add_argument("--split", default="spur_mpm/experiments/E020_cohort/split.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--mode", default="y")
    ap.add_argument("--ky", type=int, default=2)
    ap.add_argument("--kz", type=int, default=1)
    ap.add_argument("--patch", default="32,256,256")
    ap.add_argument("--n-patches", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/E024_dual_prior/psf_estimate")
    args = ap.parse_args(argv)

    patch = tuple(int(v) for v in args.patch.split(","))
    result = estimate_psf(
        args.index, args.split, fold=args.fold, mode=args.mode, ky=args.ky,
        kz=args.kz, patch=patch, n_patches=args.n_patches, lam=args.lam,
        device=args.device, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "psf_estimate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    (out / "psf_estimate.md").write_text(_md(result))
    print(_md(result))
    print(f"wrote {out/'psf_estimate.json'} and {out/'psf_estimate.md'}")


if __name__ == "__main__":
    main()

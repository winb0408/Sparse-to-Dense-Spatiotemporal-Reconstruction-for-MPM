"""Precompute the ROIs the E027 figures are drawn from.

One pass per (protocol, arm), cached as ``.npz`` so a figure can be re-rendered
in seconds without touching a GPU -- Issue #4 requires a fresh checkout plus
archived intermediates to reproduce every figure without retraining.

What is computed, per protocol:

* ``dense``  the recorded reference (the target D),
* ``sparse`` the observation Y zero-filled, ``mask`` its support,
* ``i0``     interpolation of Y,
* ``w0_paper`` the literature-FWHM Wiener initialisation (M1's negative result --
  kept because Fig. 2 has to show it, not hide it),
* ``<arm>``  a trained arm's whole-volume reconstruction, only when its
  checkpoint exists for that protocol (otherwise the ADC reports it as NOT_RUN).

The floors are computed on the **whole volume** and then cropped: a Wiener solve
on a slab would see a boundary that does not exist in the experiment.

Usage:
    python -m spur_mpm.scripts.paper_figures.cache_rois --out reports/.../rois \
        --sample M2_FOVII_S1_vessels --z0 89 --z1 113 [--arms gpu]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spur_mpm.common.config import (apply_overrides, deep_merge, load_config,
                                    save_config)
from spur_mpm.data.datasets import VolumeCache
from spur_mpm.data.split import folds_by_name, read_split
from spur_mpm.inference import reconstruct_volume
from spur_mpm.physics.acquisition import AcquisitionConfig, NoiseConfig, observe
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import warm_start

CFG = "spur_mpm/configs/experiments/E027_literature_psf_figures.yaml"

#: The protocols Fig. 2 shows. Retained fraction is 1/(ky*kz).
PROTOCOLS = {
    "Y50": {"mode": "y", "ky": 2, "kz": 1},
    "Y75": {"mode": "y", "ky": 4, "kz": 1},
    "YZ75": {"mode": "yz", "ky": 2, "kz": 2},
}

#: Trained arms that actually exist for each protocol. A protocol without a
#: checkpoint gets ``NOT_RUN`` rather than a column filled from another protocol.
ARMS = {
    "Y75": {
        "ResidualI": ("P2_ResidualI",
                      "runs/E025_adaptive_dual_prior/E025_y_k4_fold0_s0_residual/best.pt"),
        "Proposed": ("P3_DualAdaptive",
                     "runs/E025_adaptive_dual_prior/E025_y_k4_fold0_s0_dual_wiener/best.pt"),
    },
    "YZ75": {
        "ResidualI": ("P2_ResidualI",
                      "runs/E026_psras_condition/E026_real_yz_k2_kz2_fold0_s0_ResidualI/best.pt"),
        "Proposed": ("P3_DualAdaptive",
                     "runs/E026_psras_condition/E026_real_yz_k2_kz2_fold0_s0_P3_wiener/best.pt"),
    },
    "Y50": {},
}


def floors(cache, sid, fold, proto: dict, paper: PSFConfig, lam: float) -> dict:
    """dense / sparse / mask / i0 / w0 on the whole volume, then cropped later."""
    geom = cache.geometry(sid)
    dense = cache.calibration_for(fold).apply(cache.array(sid)).astype(np.float32)
    spec = SamplingSpec(mode=proto["mode"], ky=proto["ky"], kz=proto["kz"])
    acq = observe(torch.from_numpy(dense),
                  AcquisitionConfig(sampling=spec, psf=PSFConfig(),
                                    noise=NoiseConfig(poisson_photons=0.0,
                                                      gaussian_sigma=0.0, seed=0),
                                    phase=(0, 0, 0)))
    y = acq.y.numpy().astype(np.float32)
    m = acq.mask.numpy().astype(bool)
    y0 = torch.from_numpy(y)
    m0 = torch.from_numpy(m)
    i0 = warm_start(y0, m0, PSFConfig(0.0, 0.0, geom.voxel_xy_um, geom.z_step_um),
                    lam, mode="bilinear").numpy()
    w0 = warm_start(y0, m0, PSFConfig(paper.sigma_xy_um, paper.sigma_z_um,
                                      geom.voxel_xy_um, geom.z_step_um),
                    lam, mode="wiener").numpy()
    return {"dense": dense, "sparse": y, "mask": m, "i0": i0, "w0_paper": w0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="M2_FOVII_S1_vessels")
    ap.add_argument("--fold", type=int, default=0)
    # The slab is centred on a **measured** plane for the kz = 2 protocol: with a
    # lattice phase of 0 the odd planes are entirely unmeasured, and a figure
    # whose XY panel is 100 % holes cannot show an in-plane comparison. z = 100
    # is measured for kz = 2 and is fully measured for kz = 1.
    ap.add_argument("--z0", type=int, default=88)
    ap.add_argument("--z1", type=int, default=112)
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    ap.add_argument("--arms", default="none", choices=["none", "gpu"],
                    help="'gpu' runs the trained arms too (needs a free card)")
    ap.add_argument("--reuse", action="store_true",
                    help="re-crop existing recon_*.npy instead of re-running inference")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    # `load_config` reads one file; the experiment configs are partial and the
    # scripts merge them over `base.yaml` (train/evaluate do this via --base).
    # A figure script that skipped the merge would fail on `model.base`, which
    # only base.yaml carries.
    cfg = deep_merge(load_config("spur_mpm/configs/base.yaml"), load_config(CFG))
    # `data.fold` is not a key in the base config, so it is set here rather than
    # through apply_overrides (which refuses to create keys, on purpose).
    cfg["data"]["fold"] = int(a.fold)
    cache = VolumeCache.load(cfg["data"]["index"], fold=a.fold)
    geom = cache.geometry(a.sample)
    paper = PSFConfig(**cfg["physics"]["psf"])
    lam = float(cfg["physics"]["lam"])
    a.out.mkdir(parents=True, exist_ok=True)

    manifest = {"sample": a.sample, "fold": a.fold, "z0": a.z0, "z1": a.z1,
                "voxel_um": {"xy": geom.voxel_xy_um, "z": geom.z_step_um},
                "lam": lam, "psf": paper.to_dict(),
                "psf_source": cfg["physics"]["psf_source"], "protocols": {}}

    for key in a.protocols:
        proto = PROTOCOLS[key]
        arrays = floors(cache, a.sample, a.fold, proto, paper, lam)
        entry = {"sampling": proto, "retained_fraction": float(arrays["mask"].mean()),
                 "arms": {}}

        for name, (arm, ckpt) in ARMS.get(key, {}).items():
            stem = a.out / f"recon_{key}_{name}"
            cached = Path(f"{stem}.npy")
            if a.arms != "gpu" or not Path(ckpt).exists():
                entry["arms"][name] = {"arm": arm, "ckpt": ckpt,
                                       "state": "NOT_RUN" if not Path(ckpt).exists()
                                                else "cached_elsewhere"}
                continue
            if a.reuse and cached.exists():
                # Re-crop from a previous reconstruction: the slab boundaries are
                # a display choice, not a computation, so moving them must not
                # cost another 203-plane inference pass.
                arrays[name] = np.load(cached)
                entry["arms"][name] = {"arm": arm, "ckpt": ckpt, "state": "RUN",
                                       "reused": True}
                continue
            ov = [f"sampling.mode={proto['mode']}", f"sampling.ky={proto['ky']}",
                  f"sampling.kz={proto['kz']}",
                  f"data.context_slices={2 * proto['kz'] - 1}",
                  # Trained with the identity prior; evaluating with the paper
                  # kernel would be an out-of-distribution comparison
                  # (Issue #4 forbids exactly that).
                  "physics.psf.sigma_xy_um=0.0", "physics.psf.sigma_z_um=0.0",
                  "physics.psf_source=identity"]
            arm_cfg = apply_overrides(cfg, ov)
            stem = a.out / f"recon_{key}_{name}"
            # `reconstruct_volume` np.save()s the volume at `stem` (+ ".npy"),
            # keeps the mask and warm start beside it, and *returns metadata* --
            # a caller that treats the return value as an array gets a dict.
            reconstruct_volume(arm_cfg, arm, ckpt, a.sample, stem, device="cuda:0")
            arrays[name] = np.load(f"{stem}.npy")
            save_config(arm_cfg, a.out / f"config_{key}_{name}.yaml")
            entry["arms"][name] = {"arm": arm, "ckpt": ckpt, "state": "RUN"}

        crop = {k: v[a.z0:a.z1] for k, v in arrays.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(a.out / f"roi_{key}.npz", **crop)
        manifest["protocols"][key] = entry
        print(f"{key}: {list(crop)} -> roi_{key}.npz", flush=True)

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {a.out/'manifest.json'}")


if __name__ == "__main__":
    main()

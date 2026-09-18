"""Protocol v3 datasets: sparse observation generated at RUNTIME (DEC-007).

The dataset draws a GT patch from a (normalized, float16) stack, pushes it
through the single acquisition operator, builds model inputs via the single
observation builder, and returns everything the model/loss needs:

    inputs  : selected features stacked as channels [C_total, H, W]
              (features: Y, M, X0, C_s, U_p; slices concatenated for T>1)
    target  : GT patch [T, H, W]
    query   : Q mask over slices -- loss is computed only where Q=1
              (2D: T=1/Q=[1]; 2.5D: T=3/Q=[0,1,0] center slice)

Patch RNG is derived from (seed, index), so every run is reproducible.
Validation uses a deterministic patch grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.observation import build_observation
from physics.acquisition import AcquisitionConfig, NoiseConfig, simulate_acquisition
from physics.psf import PSFConfig

__all__ = ["CIDC25SparseDataset", "make_val_dataset", "window_T"]

FEATURES = ("y", "mask", "x0", "confidence", "uncertainty")


def _derive_seed(base: int, *idx: int) -> int:
    ss = np.random.SeedSequence([base, *idx])
    return int(ss.generate_state(1)[0])


def window_T(regime: str, mode: str, K: int) -> int:
    """Input window depth.

    2d: a single slice. 2.5d lateral modes: 3-slice context (legacy).
    2.5d axial modes (z/yz/xyz): 2K-1 slices, so the window centered on the
    query slice always contains both bracketing measured planes (every K
    consecutive slices hold exactly one knot; the nearest knots sit within
    t±(K-1)). Without this, a T=3 window can fall entirely inside an axial
    gap and the task becomes ill-posed from y alone (DEC-012).
    """
    if regime == "2d":
        return 1
    if mode in ("z", "yz", "xyz"):
        return 2 * K - 1
    return 3


class CIDC25SparseDataset(Dataset):
    """Training dataset with runtime acquisition (Protocol B)."""

    def __init__(self, data_dir: str | Path, split: str = "train",
                 regime: str = "2d",              # 2d | 2.5d
                 mode: str = "xy", K: int = 4, pattern: str = "grid",
                 psf: PSFConfig | None = None,
                 noise: NoiseConfig | None = None,
                 patch_hw: tuple[int, int] = (256, 256),
                 patches_per_epoch: int = 1000, seed: int = 0,
                 warm_start_kind: str = "wiener", lam: float = 0.01,
                 features: tuple[str, ...] = FEATURES,
                 t_override: int | None = None):
        self.dir = Path(data_dir)
        split_info = json.loads((self.dir / "split.json").read_text())
        self.split = split_info["split"][split]
        self.regime = regime
        self.T = window_T(regime, mode, K)
        # EXP-017 context-scaling: explicit window-depth override (must be
        # odd; query stays the center slice). T < 2K-1 may leave knotless
        # windows (DEC-012 ill-posedness) -- x0 then degrades to 0 by
        # _axial_interp; pre-registered as part of the curve's meaning.
        if t_override is not None:
            assert t_override % 2 == 1, f"T override must be odd, got {t_override}"
            self.T = t_override
        half = self.T // 2
        self.query = ([1] * self.T if self.T == 1
                      else [0] * half + [1] + [0] * (self.T - half - 1))
        self.acq = AcquisitionConfig(
            mode=mode, K=K, pattern=pattern,
            psf=psf if psf is not None else PSFConfig(),
            noise=noise if noise is not None else NoiseConfig(),
        )
        self.patch_hw = tuple(patch_hw)
        self.patches_per_epoch = patches_per_epoch
        self.seed = seed
        self.warm_start_kind = warm_start_kind
        self.lam = lam
        self.features = features
        self.stacks = [np.load(self.dir / "stacks" / f"{s}.npy", mmap_mode="r")
                       for s in self.split["stacks"]]
        self.frame_ranges = [self.split["frame_range"] or [0, s.shape[0]]
                             for s in self.stacks]
        self.fixed_items: list[dict] | None = None

    def __len__(self) -> int:
        return self.patches_per_epoch

    def _draw_patch(self, index: int):
        rng = np.random.default_rng(_derive_seed(self.seed, index))
        si = int(rng.integers(len(self.stacks)))
        stack, (f0, f1) = self.stacks[si], self.frame_ranges[si]
        th, ph = self.T, self.patch_hw[0]
        pw = self.patch_hw[1]
        t0 = int(rng.integers(f0, max(f0 + 1, f1 - th + 1)))
        y0 = int(rng.integers(0, stack.shape[1] - ph + 1))
        x0 = int(rng.integers(0, stack.shape[2] - pw + 1))
        gt = np.asarray(stack[t0:t0 + th, y0:y0 + ph, x0:x0 + pw], dtype=np.float32)
        seeds = (_derive_seed(self.seed, index, 1000), _derive_seed(self.seed, index, 2000))
        # Stack-frame lattice phase (DEC-012): keeps the sampled pattern
        # aligned with the stack, not re-anchored to each window.
        phases = (t0 % self.acq.K, y0 % self.acq.K, x0 % self.acq.K)
        return gt, seeds, phases

    def __getitem__(self, index: int):
        if self.fixed_items is not None:
            return self.fixed_items[index]
        gt, (mask_seed, noise_seed), phases = self._draw_patch(index)
        gt_t = torch.from_numpy(gt)
        # Per-sample deterministic seeds for the mask and noise.
        acq = AcquisitionConfig(
            mode=self.acq.mode, K=self.acq.K, pattern=self.acq.pattern,
            psf=self.acq.psf,
            noise=NoiseConfig(self.acq.noise.poisson_photons,
                              self.acq.noise.gaussian_sigma, noise_seed),
            mask_seed=mask_seed, phase_t=phases[0], phase_y=phases[1],
            phase_x=phases[2],
        )
        obs = simulate_acquisition(gt_t, acq)
        mi = build_observation(obs, psf=self.acq.psf, lam=self.lam,
                               warm_start_kind=self.warm_start_kind)
        feats = {"y": mi.y, "mask": mi.mask, "x0": mi.x0,
                 "confidence": mi.confidence, "uncertainty": mi.uncertainty}
        # [C_total, H, W]: slices concatenated as channels.
        chans = torch.cat([feats[f].reshape(-1, *gt_t.shape[1:]) for f in self.features], 0)
        return {"inputs": chans, "target": gt_t,
                "query": torch.tensor(self.query, dtype=torch.float32)}


def make_val_dataset(data_dir: str | Path, split: str, regime: str, n_per_stack: int = 8,
                     patch_hw: tuple[int, int] = (256, 256), **kwargs) -> CIDC25SparseDataset:
    """Deterministic validation/test dataset: a fixed grid of patch centres,
    fixed acquisition seeds."""
    ds = CIDC25SparseDataset(data_dir, split=split, regime=regime,
                             patch_hw=patch_hw, patches_per_epoch=0, **kwargs)
    items = []
    for si, (stack, (f0, f1)) in enumerate(zip(ds.stacks, ds.frame_ranges)):
        th = ds.T
        t0s = np.linspace(f0, max(f0, f1 - th), n_per_stack, dtype=int)
        for t0 in t0s:
            y0 = (stack.shape[1] - patch_hw[0]) // 2
            x0 = (stack.shape[2] - patch_hw[1]) // 2
            gt = np.asarray(stack[t0:t0 + th, y0:y0 + patch_hw[0], x0:x0 + patch_hw[1]],
                            dtype=np.float32)
            acq = AcquisitionConfig(
                mode=ds.acq.mode, K=ds.acq.K, pattern=ds.acq.pattern, psf=ds.acq.psf,
                noise=NoiseConfig(ds.acq.noise.poisson_photons,
                                  ds.acq.noise.gaussian_sigma,
                                  _derive_seed(ds.seed, 7777, si, int(t0))),
                mask_seed=_derive_seed(ds.seed, 8888, si, int(t0)),
                phase_t=t0 % ds.acq.K, phase_y=y0 % ds.acq.K, phase_x=x0 % ds.acq.K,
            )
            obs = simulate_acquisition(torch.from_numpy(gt), acq)
            mi = build_observation(obs, psf=ds.acq.psf, lam=ds.lam,
                                   warm_start_kind=ds.warm_start_kind)
            feats = {"y": mi.y, "mask": mi.mask, "x0": mi.x0,
                     "confidence": mi.confidence, "uncertainty": mi.uncertainty}
            chans = torch.cat([feats[f].reshape(-1, *gt.shape[1:]) for f in ds.features], 0)
            items.append({"inputs": chans, "target": torch.from_numpy(gt),
                          "query": torch.tensor(ds.query, dtype=torch.float32)})
    ds.fixed_items = items
    ds.patches_per_epoch = len(items)
    return ds

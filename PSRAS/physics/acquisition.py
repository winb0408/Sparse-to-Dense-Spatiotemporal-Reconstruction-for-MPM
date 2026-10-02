"""Unified acquisition simulation (Protocol B, PLAN.md §5).

The single forward model of the whole project:

    Y = M . (H*X + n)

with strictly separated stages, in this order:

    1. optical degradation   H  (anisotropic PSF blur, optional)
    2. sampling              M  (measurement support)
    3. noise                 n  (Poisson shot + Gaussian readout, on measured
                                 positions only -- noise arises during
                                 measurement)
    4. geometric resampling  R  (Protocol A only: real low-resolution
                                 acquisition mapped back to the network grid;
                                 deterministic preprocessing, NOT
                                 reconstruction)

Warm-start X0 is built separately at dataset runtime (data/observation.py);
it is never stored as part of the observation (PLAN.md P0-1/P0-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .psf import PSFConfig
from .sampling import make_mask

__all__ = ["NoiseConfig", "AcquisitionConfig", "Observation", "simulate_acquisition",
           "blur_3d_fft"]


def blur_3d_fft(x: torch.Tensor, psf: PSFConfig) -> torch.Tensor:
    """Apply the anisotropic PSF via FFT with the analytic OTF (H=identity if sigma=0)."""
    if psf.is_identity:
        return x
    squeeze = x.ndim == 3
    if squeeze:
        x = x[None]
    T, H, W = x.shape[-3:]
    Xf = torch.fft.fftn(x.float(), dim=(-3, -2, -1))
    from .psf import gaussian_otf_3d
    otf = gaussian_otf_3d((T, H, W), psf, device=x.device, dtype=torch.float32)
    out = torch.fft.ifftn(otf[None] * Xf, dim=(-3, -2, -1)).real
    return out[0] if squeeze else out


@dataclass(frozen=True)
class NoiseConfig:
    poisson_photons: float = 0.0   # peak photon budget; 0 disables shot noise
    gaussian_sigma: float = 0.0    # readout noise std in normalized [0,1] units
    seed: int = 0

    def to_dict(self) -> dict:
        return {"poisson_photons": self.poisson_photons,
                "gaussian_sigma": self.gaussian_sigma, "seed": self.seed}


@dataclass(frozen=True)
class AcquisitionConfig:
    mode: str = "xy"               # xy | y | z | yz | xyz | yz2 | xyz2
    K: int = 2
    K2: int = 0                    # yz2: independent sparsity along the 2nd
                                   # axis (ky=K rows, kz=K2 planes); 0 = unused
    pattern: str = "grid"
    psf: PSFConfig = field(default_factory=PSFConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    mask_seed: int = 0
    # Stack-frame lattice phase: global slice/row/col g is measured iff
    # (g - phase) % K == 0. Keeps the sampled lattice aligned across the
    # windows cropped from one stack (DEC-012).
    phase_t: int = 0
    phase_y: int = 0
    phase_x: int = 0

    def to_dict(self) -> dict:
        return {"mode": self.mode, "K": self.K, "K2": self.K2,
                "pattern": self.pattern,
                "psf": self.psf.to_dict(), "noise": self.noise.to_dict(),
                "mask_seed": self.mask_seed,
                "phase": [self.phase_t, self.phase_y, self.phase_x]}


@dataclass
class Observation:
    """Raw sparse observation on the network grid (Protocol B)."""
    y: torch.Tensor            # [T, H, W] zero-filled sparse measurement
    mask: torch.Tensor         # [T, H, W] bool measurement support
    metadata: dict = field(default_factory=dict)

    @property
    def sampling_ratio(self) -> float:
        return float(self.mask.float().mean())


def _add_noise(x: torch.Tensor, mask: torch.Tensor, noise: NoiseConfig) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(noise.seed)
    y = x.clone()
    if noise.poisson_photons > 0:
        lam = (y * noise.poisson_photons).clamp(min=0)
        # torch Poisson on CPU generator for reproducibility; sample on the
        # measured positions only.  lam[idx] is a flat 1-D selection, so the
        # draws are scattered back into the masked positions (EXP-019: the
        # torch.where form here never shape-matched and the path had never
        # been exercised while poisson_photons stayed 0).
        shot = torch.poisson(lam[mask].cpu(), generator=g).to(y.device)
        y = y.clone()
        y[mask] = shot.to(y.dtype) / noise.poisson_photons
    if noise.gaussian_sigma > 0:
        r = torch.randn(y.shape, generator=g).to(y.device) * noise.gaussian_sigma
        y = torch.where(mask, y + r, y)
    return y


def simulate_acquisition(x: torch.Tensor, cfg: AcquisitionConfig) -> Observation:
    """Simulate the sparse acquisition of a normalized ground-truth volume.

    x : [T, H, W] float tensor in [0, 1] (see data/normalization.py).
    """
    if x.ndim != 3:
        raise ValueError(f"expected [T, H, W], got {tuple(x.shape)}")
    T, H, W = x.shape

    # 1. optical degradation
    hx = blur_3d_fft(x, cfg.psf)

    # 2. sampling support
    if cfg.mode == "yz2":
        from .sampling import make_mask_yz2
        mask = make_mask_yz2(cfg.K, cfg.K2 or cfg.K, T, H, W,
                             seed=cfg.mask_seed, phase_t=cfg.phase_t,
                             phase_y=cfg.phase_y)
    elif cfg.mode == "xyz2":
        # EXP-026 (Phase 5): independent lateral cross spacing (K) and
        # plane spacing (K2) -- extreme spatial undersampling stress-test.
        from .sampling import make_mask_xyz2
        mask = make_mask_xyz2(cfg.K, cfg.K2 or cfg.K, T, H, W,
                              seed=cfg.mask_seed, phase_t=cfg.phase_t,
                              phase_y=cfg.phase_y, phase_x=cfg.phase_x)
    else:
        mask = make_mask(cfg.mode, cfg.K, T, H, W, cfg.pattern, cfg.mask_seed,
                         phase_t=cfg.phase_t, phase_y=cfg.phase_y,
                         phase_x=cfg.phase_x)

    # 3. noise on measured positions
    y = _add_noise(hx, mask, cfg.noise)
    y = torch.where(mask, y, torch.zeros_like(y))  # zero-fill unmeasured

    return Observation(
        y=y,
        mask=mask,
        metadata={"acquisition": cfg.to_dict(),
                  "shape": [T, H, W],
                  "sampling_ratio": float(mask.float().mean())},
    )

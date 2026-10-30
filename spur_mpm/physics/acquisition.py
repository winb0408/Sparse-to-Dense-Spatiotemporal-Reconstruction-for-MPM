"""The single forward observation model.

Image domain **A** (the project's main line, plan §3.1)::

    Y = M_acq . (H * R) + noise        with H = identity by default

where ``R`` is the dense fluorescence image *as the microscope records it*.
``R`` already contains the true PSF and shot noise of the real acquisition, so
``H`` defaults to identity: blurring it again would be a fabrication. A
non-identity ``H`` is only valid for a labelled degradation stress test or for
domain **B** (latent emitter inversion), which needs a calibrated PSF.

Stage order is fixed and observable, because mixing them is exactly how the
old project created ambiguous comparisons:

    1. optical degradation   H   (identity = no-op)
    2. sampling support      M   (virtual_sparse)
    3. measurement noise     n   (only where M = 1: noise happens at acquisition)

``virtual_sparse`` never implies measured speed-up, dose reduction or a real
hardware scan; only ``hardware_sparse`` data may support those claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from spur_mpm.physics.psf import PSFConfig, gaussian_otf_3d
from spur_mpm.physics.sampling import SamplingSpec

__all__ = ["NoiseConfig", "AcquisitionConfig", "Observation", "observe", "apply_otf"]


def apply_otf(x: torch.Tensor, psf: PSFConfig) -> torch.Tensor:
    """Forward optical operator ``H * x`` via FFT (identity when sigma = 0).

    Accepts ``[H, W]``, ``[Z, H, W]`` or batched ``[B, Z, H, W]``. On a 2-D image
    only the lateral optics can act, so a PSF carrying an axial width is refused
    instead of quietly dropping it: the caller has to say
    ``psf.lateral_only()`` out loud.
    """
    if psf.is_identity:
        return x
    if x.ndim == 2:
        if psf.sigma_z_px != 0.0:
            raise ValueError(
                f"2-D input but the PSF has sigma_z = {psf.sigma_z_px} px; use "
                "psf.lateral_only() to state that only the lateral optics apply")
        xf = torch.fft.fftn(x.float(), dim=(-2, -1))
        otf = gaussian_otf_2d(tuple(x.shape), psf.sigma_xy_px, device=x.device)
        return torch.fft.ifftn(otf * xf, dim=(-2, -1)).real
    squeeze = x.ndim == 3
    if squeeze:
        x = x[None]
    d, h, w = x.shape[-3:]
    xf = torch.fft.fftn(x.float(), dim=(-3, -2, -1))
    otf = gaussian_otf_3d((d, h, w), psf, device=x.device)
    out = torch.fft.ifftn(otf[None] * xf, dim=(-3, -2, -1)).real
    return out[0] if squeeze else out


@dataclass(frozen=True)
class NoiseConfig:
    """Measurement noise applied at measured positions only.

    ``poisson_photons`` is the peak photon budget; 0 disables shot noise.
    ``gaussian_sigma`` is readout noise std in normalized [0, 1] units.
    """

    poisson_photons: float = 0.0
    gaussian_sigma: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict:
        return {"poisson_photons": self.poisson_photons,
                "gaussian_sigma": self.gaussian_sigma, "seed": self.seed}


@dataclass(frozen=True)
class AcquisitionConfig:
    """Everything needed to realise one virtual acquisition."""

    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    psf: PSFConfig = field(default_factory=PSFConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    phase: tuple[int, int, int] = (0, 0, 0)

    def to_dict(self) -> dict:
        return {"sampling": self.sampling.to_dict(), "psf": self.psf.to_dict(),
                "noise": self.noise.to_dict(), "phase": list(self.phase)}


@dataclass
class Observation:
    """Sparse observation on the reconstruction grid.

    ``y``    zero-filled measurement, ``[Z, H, W]`` or ``[H, W]`` float
    ``mask`` bool measurement support, same shape
    """

    y: torch.Tensor
    mask: torch.Tensor
    metadata: dict = field(default_factory=dict)

    @property
    def sampling_fraction(self) -> float:
        return float(self.mask.float().mean())


def _add_noise(x: torch.Tensor, mask: torch.Tensor, noise: NoiseConfig) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(noise.seed)
    y = x.clone()
    if noise.poisson_photons > 0:
        lam = (y * noise.poisson_photons).clamp(min=0)
        shot = torch.poisson(lam[mask].cpu(), generator=g).to(y.device)
        y[mask] = shot.to(y.dtype) / noise.poisson_photons
    if noise.gaussian_sigma > 0:
        r = torch.randn(y.shape, generator=g).to(y.device) * noise.gaussian_sigma
        y = torch.where(mask, y + r, y)
    return y


def observe(dense: torch.Tensor, cfg: AcquisitionConfig) -> Observation:
    """Realise ``Y = M . (H * dense) + noise`` on a ``[Z, H, W]`` or ``[H, W]`` image.

    The stage order is the same in both cases; only the depth stage disappears in
    2-D. Positions with ``M = 0`` are exactly zero: they were never measured, and
    filling them here would hand the reconstruction the answer it must infer.
    """
    if dense.ndim not in (2, 3):
        raise ValueError(f"expected [Z, H, W] or [H, W], got {tuple(dense.shape)}")
    shape = tuple(dense.shape)

    blurred = apply_otf(dense, cfg.psf)
    mask = cfg.sampling.mask(shape, phase=cfg.phase)
    y = _add_noise(blurred, mask, cfg.noise)
    y = torch.where(mask, y, torch.zeros_like(y))

    return Observation(
        y=y, mask=mask,
        metadata={"acquisition": cfg.to_dict(),
                  "shape": list(shape),
                  "ndim": len(shape),
                  "sampling_fraction": float(mask.float().mean())},
    )

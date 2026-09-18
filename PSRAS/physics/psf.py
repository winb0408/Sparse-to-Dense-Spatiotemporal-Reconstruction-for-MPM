"""Anisotropic 3D Gaussian PSF / OTF.

The PSF is defined in physical units (um) and converted to pixels via the
voxel size, so the same optical model transfers across datasets with
different sampling grids.

Conventions
-----------
- Volumes are torch tensors of shape [T, H, W] (or batched [B, T, H, W]).
- The OTF H(f) is built analytically on the torch.fft frequency grid:

      H(fx, fy, fz) = exp(-2 * pi^2 * (sx^2 fx^2 + sy^2 fy^2 + sz^2 fz^2))

  which is the exact Fourier transform of the separable Gaussian PSF.
  sigma = 0 gives H == 1 (identity / no blur), used for the no-PSF
  ablation line (PLAN.md DEC-002).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["PSFConfig", "gaussian_otf_3d", "gaussian_otf_2d"]


@dataclass(frozen=True)
class PSFConfig:
    """Anisotropic Gaussian PSF in physical units.

    A sigma of 0 in a given axis means "no degradation along that axis"
    (e.g. xy-sparse modes use sigma_z = 0 so each slice is deblurred
    laterally only).
    """

    sigma_xy_um: float = 0.0
    sigma_z_um: float = 0.0
    voxel_xy_um: float = 1.0
    voxel_z_um: float = 1.0

    @property
    def sigma_xy_px(self) -> float:
        return self.sigma_xy_um / self.voxel_xy_um

    @property
    def sigma_z_px(self) -> float:
        return self.sigma_z_um / self.voxel_z_um

    @property
    def is_identity(self) -> bool:
        return self.sigma_xy_px == 0.0 and self.sigma_z_px == 0.0

    def to_dict(self) -> dict:
        return {
            "sigma_xy_um": self.sigma_xy_um,
            "sigma_z_um": self.sigma_z_um,
            "voxel_xy_um": self.voxel_xy_um,
            "voxel_z_um": self.voxel_z_um,
        }


def _freq_grid(shape: tuple[int, ...], device, dtype) -> list[torch.Tensor]:
    """Per-axis frequency grids, broadcast-ready: [T,1,1], [1,H,1], [1,1,W]."""
    grids = [torch.fft.fftfreq(n, device=device, dtype=dtype) for n in shape]
    out = []
    for d, g in enumerate(grids):
        view = [1] * len(shape)
        view[d] = -1
        out.append(g.view(view))
    return out


def gaussian_otf_3d(shape: tuple[int, int, int], psf: PSFConfig,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """Analytic anisotropic Gaussian OTF on the 3D FFT grid.

    Parameters
    ----------
    shape : (T, H, W)
    psf   : PSFConfig with sigma in pixels (physical units already converted).

    Returns
    -------
    Real tensor [T, H, W] with values in (0, 1]; H(f) = exp(-2 pi^2 sum sigma_i^2 f_i^2).
    """
    sz, sx = psf.sigma_z_px, psf.sigma_xy_px
    fz, fy, fx = _freq_grid(shape, device, dtype)
    expo = -2.0 * (torch.pi ** 2) * (
        (sz ** 2) * fz**2 + (sx ** 2) * fy**2 + (sx ** 2) * fx**2
    )
    return torch.exp(expo)


def gaussian_otf_2d(shape: tuple[int, int], sigma_xy_px: float,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """Lateral-only Gaussian OTF on the 2D FFT grid, shape (H, W)."""
    fy, fx = _freq_grid(shape, device, dtype)
    expo = -2.0 * (torch.pi ** 2) * (sigma_xy_px ** 2) * (fy**2 + fx**2)
    return torch.exp(expo)

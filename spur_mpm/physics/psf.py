"""Anisotropic 3-D Gaussian PSF / OTF in physical units.

Ported from PSRAS/physics/psf.py with two changes required by the new
protocol:

1. The stack axis is named ``Z`` (physical depth) rather than ``T``. The old
   code used ``T`` because its first axis was a calcium-imaging *time* axis;
   writing that as ``Z`` is one of the errors this project exists to avoid.
2. The default PSF is the **identity** operator. Real Dryad 3P stacks already
   carry the true optical blur, so re-applying a synthetic Gaussian would
   double-blur them (plan §3.1). A non-identity PSF is only for explicitly
   labelled degradation stress tests.

Conventions
-----------
- Volumes are torch tensors of shape ``[Z, H, W]`` (or batched ``[B, Z, H, W]``).
- ``H(f) = exp(-2 pi^2 (sx^2 fx^2 + sy^2 fy^2 + sz^2 fz^2))``, the exact
  Fourier transform of a separable Gaussian with sigma in pixels.
- ``sigma = 0`` on an axis means "no degradation along that axis"; all-zero
  sigma gives ``H == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["PSFConfig", "gaussian_otf_3d", "gaussian_otf_2d"]


@dataclass(frozen=True)
class PSFConfig:
    """Anisotropic Gaussian PSF with physical voxel sizes.

    ``sigma_xy_um``/``sigma_z_um`` are optical widths; dividing by the voxel
    size gives the pixel-domain sigma used by the OTF. Keeping the physical
    units means one PSF object transfers between stacks with different
    sampling grids.
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
        """True when the operator neither blurs nor cross-talks between axes."""
        return self.sigma_xy_px == 0.0 and self.sigma_z_px == 0.0

    def lateral_only(self) -> "PSFConfig":
        """Same lateral optics with the axial blur switched off (2-D deconvolution)."""
        return PSFConfig(self.sigma_xy_um, 0.0, self.voxel_xy_um, self.voxel_z_um)

    @classmethod
    def from_pixels(cls, sigma_xy_px: float, sigma_z_px: float = 0.0,
                    voxel_xy_um: float = 1.0, voxel_z_um: float = 1.0) -> "PSFConfig":
        """Build from pixel sigmas (explicitly a degradation stress test)."""
        return cls(sigma_xy_px * voxel_xy_um, sigma_z_px * voxel_z_um,
                   voxel_xy_um, voxel_z_um)

    def to_dict(self) -> dict:
        return {
            "sigma_xy_um": self.sigma_xy_um,
            "sigma_z_um": self.sigma_z_um,
            "voxel_xy_um": self.voxel_xy_um,
            "voxel_z_um": self.voxel_z_um,
            "sigma_xy_px": self.sigma_xy_px,
            "sigma_z_px": self.sigma_z_px,
            "is_identity": self.is_identity,
        }


def _freq_grid(shape: tuple[int, ...], device, dtype) -> list[torch.Tensor]:
    """Per-axis frequency grids, broadcast-ready: [D,1,1], [1,H,1], [1,1,W]."""
    out = []
    for d, n in enumerate(shape):
        g = torch.fft.fftfreq(n, device=device, dtype=dtype)
        view = [1] * len(shape)
        view[d] = -1
        out.append(g.view(view))
    return out


def gaussian_otf_3d(shape: tuple[int, int, int], psf: PSFConfig,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """Analytic anisotropic Gaussian OTF on the [Z, H, W] FFT grid.

    Returns a real tensor of ``shape`` with values in (0, 1].
    """
    sz, sxy = psf.sigma_z_px, psf.sigma_xy_px
    fz, fy, fx = _freq_grid(shape, device, dtype)
    expo = -2.0 * (torch.pi ** 2) * (
        (sz ** 2) * fz ** 2 + (sxy ** 2) * fy ** 2 + (sxy ** 2) * fx ** 2
    )
    return torch.exp(expo)


def gaussian_otf_2d(shape: tuple[int, int], sigma_xy_px: float,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """Lateral-only Gaussian OTF on the (H, W) FFT grid."""
    fy, fx = _freq_grid(shape, device, dtype)
    expo = -2.0 * (torch.pi ** 2) * (sigma_xy_px ** 2) * (fy ** 2 + fx ** 2)
    return torch.exp(expo)

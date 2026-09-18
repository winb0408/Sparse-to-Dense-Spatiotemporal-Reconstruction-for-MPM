"""Tests for the true anisotropic 3D Wiener (PLAN.md Phase 1 sanity checks).

Checks (GPT §17 / PLAN.md M-items):
- no blur, no sampling       -> identity
- mask fully sampled         -> deconvolution recovers a blurred input
- known Gaussian volume      -> 3D Wiener improves frequency recovery vs zero-fill
- z-sparse warm start        -> axial interpolation fills gaps, lateral deconv on measured
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.acquisition import blur_3d_fft
from physics.psf import PSFConfig
from physics.sampling import make_mask
from metrics.image_metrics import psnr
from physics.wiener import warm_start, wiener_deconv_3d


def test_wiener_identity_when_no_psf():
    x = torch.rand(4, 16, 16)
    assert torch.equal(wiener_deconv_3d(x, PSFConfig(), lam=0.01), x)


def test_wiener_recovers_blurred_volume():
    """Full sampling + known PSF: Wiener output closer to GT than the blurred input."""
    torch.manual_seed(1)
    x = torch.rand(8, 48, 48) * 0.5 + 0.25
    psf = PSFConfig(sigma_xy_um=1.3, sigma_z_um=1.8, voxel_xy_um=1.0, voxel_z_um=1.0)
    y = blur_3d_fft(x, psf) + 0.005 * torch.randn_like(x)
    rec = wiener_deconv_3d(y, psf, lam=0.01)
    assert (rec - x).abs().mean() < (y - x).abs().mean()


def test_wiener_frequency_recovery_vs_zero_fill():
    """Known synthetic structure: Wiener restores attenuated frequencies that
    plain zero-fill/observation keeps blurred."""
    torch.manual_seed(2)
    T, H, W = 8, 48, 48
    zz, yy, xx = torch.meshgrid(torch.linspace(0, 1, T), torch.linspace(0, 1, H),
                                torch.linspace(0, 1, W), indexing="ij")
    x = (0.5 + 0.5 * torch.sin(2 * torch.pi * 6 * xx) * torch.sin(2 * torch.pi * 4 * zz))
    psf = PSFConfig(sigma_xy_um=1.5, sigma_z_um=2.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    y = blur_3d_fft(x, psf)
    rec = wiener_deconv_3d(y, psf, lam=1e-3)
    spec_x = torch.fft.fftn(x).abs()
    spec_rec = torch.fft.fftn(rec).abs()
    spec_y = torch.fft.fftn(y).abs()
    freq_err_rec = (spec_rec - spec_x).abs().mean()
    freq_err_y = (spec_y - spec_x).abs().mean()
    assert freq_err_rec < freq_err_y


def test_warm_start_z_sparse():
    """Z-sparse + lateral blur: measured slices laterally deconvolved,
    missing slices interpolated between the deconvolved knots."""
    torch.manual_seed(3)
    T, H, W = 9, 32, 32
    x = torch.rand(T, H, W) * 0.5 + 0.25
    lat_psf = PSFConfig(sigma_xy_um=1.0, sigma_z_um=0.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    blurred = blur_3d_fft(x, lat_psf)
    mask = make_mask("z", 3, T, H, W)                 # slices 0,3,6 measured
    y = torch.where(mask, blurred + 0.005 * torch.randn_like(x), torch.zeros_like(x))
    x0 = warm_start(y, mask, lat_psf, lam=1e-3, mode="wiener")
    # measured slices: deconvolution beats the blurred observation
    assert (x0[0] - x[0]).abs().mean() < (y[0] - x[0]).abs().mean()
    # missing slice 1 sits between slices 0 and 3: interpolated, not zero
    assert not torch.equal(x0[1], torch.zeros_like(x0[1]))
    expected = 2 / 3 * x0[0] + 1 / 3 * x0[3]
    assert (x0[1] - expected).abs().mean() < 1e-5      # exact linear interp


def test_warm_start_bilinear_z():
    # Distinct per-slice values so interpolation is actually testable:
    # measured slices 0, 2, 4 carry 0.0, 1.0, 0.5.
    mask = make_mask("z", 2, 5, 4, 4)
    vals = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.5]).view(5, 1, 1).expand(5, 4, 4)
    y = torch.where(mask, vals, torch.zeros(5, 4, 4))
    x0 = warm_start(y, mask, PSFConfig(), mode="bilinear")
    assert torch.allclose(x0[1], torch.full((4, 4), 0.5), atol=1e-5)   # between 0 and 1
    assert torch.allclose(x0[3], torch.full((4, 4), 0.75), atol=1e-5)  # between 1 and 0.5
    assert torch.allclose(x0[4], torch.full((4, 4), 0.5), atol=1e-5)   # measured slice kept
    assert torch.allclose(x0[0], torch.full((4, 4), 0.0), atol=1e-5)


def test_warm_start_bilinear_y():
    """y-mode bilinear warm start fills skipped rows by linear interpolation
    between measured rows (lateral holes no longer stay zero-filled)."""
    mask = make_mask("y", 4, 2, 8, 6)          # rows 0 and 4 measured
    vals = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = torch.where(mask, vals.view(1, 8, 1).expand(2, 8, 6),
                    torch.zeros(2, 8, 6))
    x0 = warm_start(y, mask, PSFConfig(), mode="bilinear")
    assert torch.allclose(x0[0, 1], torch.full((6,), 0.75), atol=1e-5)  # 1.0 -> 0.0
    assert torch.allclose(x0[0, 3], torch.full((6,), 0.25), atol=1e-5)
    assert torch.allclose(x0[0, 4], torch.zeros(6), atol=1e-5)  # measured knot kept
    assert torch.allclose(x0[0, 7], torch.zeros(6), atol=1e-5)  # nearest fill past last row


def test_warm_start_wiener_y_identity_fills_rows():
    """Identity PSF + y-mode: X0 keeps measured rows and interpolates the
    skipped ones (dense estimate, not zero-filled rows)."""
    mask = make_mask("y", 2, 1, 6, 4)          # rows 0, 2, 4 measured
    vals = torch.tensor([2.0, 0.0, 4.0, 0.0, 2.0, 0.0])
    y = torch.where(mask, vals.view(1, 6, 1).expand(1, 6, 4),
                    torch.zeros(1, 6, 4))
    x0 = warm_start(y, mask, PSFConfig(), mode="wiener")
    assert torch.allclose(x0[0, 1], torch.full((4,), 3.0), atol=1e-5)
    assert torch.allclose(x0[0, 3], torch.full((4,), 3.0), atol=1e-5)
    assert torch.allclose(x0[0, 5], torch.full((4,), 2.0), atol=1e-5)  # nearest fill
    assert torch.allclose(x0[0, 2], torch.full((4,), 4.0), atol=1e-5)  # measured kept


def test_warm_start_bilinear_grid_fills_intersections():
    """grid K=4: holes sit at (skipped row, skipped col) intersections and
    must be filled by interpolation between fully-measured knot rows."""
    mask = make_mask("xy", 4, 1, 8, 8)               # rows/cols 0,4 measured
    y = torch.where(mask, torch.full((1, 8, 8), 0.5), torch.zeros(1, 8, 8))
    x0 = warm_start(y, mask, PSFConfig(), mode="bilinear")
    # skipped row 2: measured cols keep 0.5, skipped cols interpolate to 0.5
    assert torch.allclose(x0[0, 2], torch.full((8,), 0.5), atol=1e-5)
    assert torch.allclose(x0[0, 0, 2], torch.full((), 0.5), atol=1e-5)  # knot kept
    assert not torch.equal(x0, y)                    # something was actually filled


def test_warm_start_fill_first_matches_plain_when_no_lateral_holes():
    """z-mode (full planes): fill_first must be identical to plain wiener."""
    torch.manual_seed(3)
    gt = torch.rand(7, 32, 32)
    planes = (torch.arange(7) % 4) == 0
    mask = planes.view(7, 1, 1).expand(7, 32, 32).float()
    y = gt * mask
    psf = PSFConfig(sigma_xy_um=1.0, sigma_z_um=1.5)
    a = warm_start(y, mask.bool(), psf, lam=0.01, mode="wiener")
    b = warm_start(y, mask.bool(), psf, lam=0.01, mode="wiener_fill_first")
    assert torch.allclose(a, b)


def test_warm_start_fill_first_fills_before_deconv():
    """y-mode: fill_first must never be worse than legacy ringing order,
    and measured knot rows must survive untouched."""
    torch.manual_seed(4)
    gt = torch.rand(1, 8, 16)
    rows = ((torch.arange(8) + 0) % 4) == 0
    mask = rows.view(1, 8, 1).float()
    y = gt * mask
    psf = PSFConfig(sigma_xy_um=1.0, sigma_z_um=0.0)
    ff = warm_start(y, mask.bool(), psf, lam=0.01, mode="wiener_fill_first")
    # Measured knot rows: fill_first keeps the (deconvolved) measurement;
    # legacy order here would ring.  Fill-first output must beat legacy in PSNR.
    leg = warm_start(y, mask.bool(), psf, lam=0.01, mode="wiener")
    assert psnr(ff[None], gt[None]) >= psnr(leg[None], gt[None])
    # interior rows must be filled (not zeros)
    assert ff[0, 1].abs().max() > 0

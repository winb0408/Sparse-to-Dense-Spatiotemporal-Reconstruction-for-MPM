"""Tests for reconstruction + data-consistency losses."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.data_consistency import DataConsistencyLoss
from losses.reconstruction import ReconstructionLoss, ms_ssim
from physics.psf import PSFConfig


def test_ms_ssim_range_and_identity():
    x = torch.rand(2, 1, 64, 64)
    assert ms_ssim(x, x).min() > 0.999
    s = ms_ssim(x, x * 0.5 + 0.25)
    assert (s >= 0).all() and (s <= 1).all()


def test_recon_loss_zero_for_identical():
    crit = ReconstructionLoss()
    x = torch.rand(1, 1, 64, 64)
    out = crit(x, x)
    assert out["loss"].item() < 1e-4


def test_recon_loss_backward():
    crit = ReconstructionLoss()
    pred = torch.rand(1, 1, 64, 64, requires_grad=True)
    out = crit(pred, torch.rand(1, 1, 64, 64))
    out["loss"].backward()
    assert torch.isfinite(pred.grad).all()


def test_dc_loss_rejects_identity_psf():
    with pytest.raises(ValueError):
        DataConsistencyLoss(PSFConfig())


def test_dc_loss_matches_measurement_domain():
    """pred == H^-1(Y) scenario: perfect DC when pred reproduces the blurred obs."""
    torch.manual_seed(0)
    T, H, W = 4, 32, 32
    x = torch.rand(T, H, W) * 0.5 + 0.25
    psf = PSFConfig(sigma_xy_um=1.0, sigma_z_um=1.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    crit = DataConsistencyLoss(psf)
    # the sharp GT re-explains the blurred observation exactly: H(x) == y
    from physics.acquisition import blur_3d_fft
    y = blur_3d_fft(x, psf)
    mask = torch.ones_like(y)
    assert crit(x, y, mask).item() < 1e-6
    # a wrong prediction incurs positive loss
    assert crit(torch.rand_like(y), y, mask).item() > 0.01
    # unmeasured regions are excluded -- with a LATERAL-ONLY operator the
    # slices are independent, so prediction errors confined to unmeasured
    # slices cannot affect the measured domain at all.
    lat = PSFConfig(sigma_xy_um=1.0, sigma_z_um=0.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    crit_lat = DataConsistencyLoss(lat)
    y_lat = blur_3d_fft(x, lat)
    mask2 = torch.ones_like(y_lat)
    mask2[:2] = False
    pred = x.clone()          # correct everywhere it is measured
    pred[:2] = torch.rand_like(x[:2])   # garbage only on unmeasured slices
    assert crit_lat(pred, y_lat, mask2).item() < 1e-6
    # but the same garbage IS penalised once those slices are measured
    assert crit_lat(pred, y_lat, torch.ones_like(y_lat)).item() > 0.01

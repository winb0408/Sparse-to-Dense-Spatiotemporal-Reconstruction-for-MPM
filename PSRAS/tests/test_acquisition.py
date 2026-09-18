"""Sanity tests for the unified acquisition operator (PLAN.md Phase 1 §tests).

Run:  cd PSRAS && python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.acquisition import (AcquisitionConfig, NoiseConfig, Observation,
                                 blur_3d_fft, simulate_acquisition)
from physics.psf import PSFConfig
from physics.sampling import make_mask


def test_acquisition_identity():
    """H=delta, no noise, K=1  =>  Y == X (forward-model consistency)."""
    x = torch.rand(8, 32, 32)
    cfg = AcquisitionConfig(mode="xy", K=1, psf=PSFConfig(), noise=NoiseConfig())
    obs = simulate_acquisition(x, cfg)
    assert torch.allclose(obs.y, x, atol=1e-6)
    assert bool(obs.mask.all())
    assert obs.sampling_ratio == 1.0


def test_mask_geometry():
    """K=2 grid keeps every other row/column; K=2 z keeps every other slice."""
    m_xy = make_mask("xy", 2, 4, 8, 8)
    assert bool(m_xy[0, ::2, :].all()) and bool(m_xy[0, :, ::2].all())
    assert not bool(m_xy[0, 1, 1])                       # odd row/col hole
    assert m_xy.float().mean().item() == pytest.approx(0.75, abs=1e-6)  # rows+cols union
    m_z = make_mask("z", 2, 6, 4, 4)
    assert bool(m_z[::2].all()) and not bool(m_z[1::2].any())
    m_xyz = make_mask("xyz", 2, 6, 8, 8)
    assert not bool(m_xyz[1::2].any())                   # axial gaps present


def test_mask_determinism():
    a = make_mask("xy", 4, 3, 16, 16, pattern="random", seed=7)
    b = make_mask("xy", 4, 3, 16, 16, pattern="random", seed=7)
    c = make_mask("xy", 4, 3, 16, 16, pattern="random", seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_mask_y_line_skipping():
    """y mode keeps every K-th full row (ratio 1/K); yz adds axial plane gaps."""
    m_y = make_mask("y", 4, 3, 16, 16)
    assert bool(m_y[0, ::4, :].all())          # kept rows fully measured in x
    assert not bool(m_y[0, 1::4, :].any())     # skipped rows fully missing
    assert m_y.float().mean().item() == pytest.approx(0.25, abs=1e-6)
    m_yz = make_mask("yz", 4, 16, 16, 16)
    assert not bool(m_yz[1::4].any())          # axial gaps present
    assert m_yz.float().mean().item() == pytest.approx(1.0 / 16, abs=1e-6)


def test_blur_reduces_energy_and_is_zero_for_identity():
    x = torch.zeros(4, 64, 64)
    x[2, 32, 32] = 1.0
    assert torch.equal(blur_3d_fft(x, PSFConfig()), x)
    psf = PSFConfig(sigma_xy_um=1.5, sigma_z_um=2.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    y = blur_3d_fft(x, psf)
    assert y.sum().item() == pytest.approx(1.0, abs=1e-4)  # OTF(0)=1 preserves flux
    assert y.max() < 1.0                                    # energy spread out


def test_noise_only_on_measured_positions():
    x = torch.ones(4, 16, 16)
    cfg = AcquisitionConfig(mode="z", K=2, psf=PSFConfig(),
                            noise=NoiseConfig(gaussian_sigma=0.1, seed=3))
    obs = simulate_acquisition(x, cfg)
    assert torch.equal(obs.y[~obs.mask], torch.zeros_like(obs.y[~obs.mask]))  # unmeasured == 0
    assert obs.y[obs.mask].std() > 0                                          # measured got noise


def test_psf_blur_then_wiener_roundtrip():
    """Wiener deconvolution of a PSF-blurred volume should beat the blurred input."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from physics.wiener import wiener_deconv_3d
    torch.manual_seed(0)
    x = torch.rand(4, 32, 32) * 0.5 + 0.25
    psf = PSFConfig(sigma_xy_um=1.2, sigma_z_um=0.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    y = blur_3d_fft(x, psf)
    rec = wiener_deconv_3d(y, psf, lam=1e-4)
    assert (rec - x).abs().mean() < (y - x).abs().mean()


def test_mask_phase_anchors_to_stack_frame():
    """phase shifts the lattice: global slice/row (g) measured iff
    (g - phase) % K == 0, so windows cropped at different origins stay
    aligned with one stack-level pattern (DEC-012)."""
    # z: window at global slices 2..4 (phase_t=2) measures its LAST slice (4).
    m = make_mask("z", 4, 3, 4, 4, phase_t=2)
    assert m[:, 0, 0].tolist() == [False, False, True]
    # grid: window origin at global (y=2, x=3): global rows 4,8 -> local 2,6;
    # global cols 4,8 -> local 1,5.
    m = make_mask("xy", 4, 1, 8, 8, phase_y=2, phase_x=3)
    assert bool(m[0, 2, :].all()) and bool(m[0, :, 1].all())
    assert not bool(m[0, 0, 0])
    # phases default to legacy behaviour
    legacy = make_mask("z", 4, 3, 4, 4)
    assert legacy[:, 0, 0].tolist() == [True, False, False]

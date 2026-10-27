"""Forward model and warm-start behaviour."""

import pytest
import torch

from spur_mpm.physics.acquisition import AcquisitionConfig, NoiseConfig, observe
from spur_mpm.physics.psf import PSFConfig, gaussian_otf_3d
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import warm_start

SHAPE = (6, 32, 32)


def dense() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(SHAPE, generator=g)


def test_identity_psf_otf_is_one():
    otf = gaussian_otf_3d(SHAPE, PSFConfig())
    assert torch.allclose(otf, torch.ones_like(otf))


def test_observation_only_fills_measured_positions():
    x = dense()
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode="y", ky=2)))
    assert torch.equal(obs.y[obs.mask], x[obs.mask])
    assert float(obs.y[~obs.mask].abs().sum()) == 0.0
    assert obs.sampling_fraction == pytest.approx(0.5, abs=0.05)


def test_noise_only_touches_measured_positions():
    x = dense()
    cfg = AcquisitionConfig(sampling=SamplingSpec(mode="z", kz=2),
                            noise=NoiseConfig(poisson_photons=100.0, seed=0))
    obs = observe(x, cfg)
    assert float(obs.y[~obs.mask].abs().sum()) == 0.0


def test_identity_psf_collapses_wiener_to_interpolation():
    """Documented degeneracy: without a PSF the Wiener family is gap filling.

    If this ever stops holding, either the PSF handling or the warm-start
    changed, and the "physics prior adds nothing beyond interpolation" claim
    in the project log no longer applies.
    """
    x = dense()
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode="y", ky=2)))
    bilinear = warm_start(obs.y, obs.mask, PSFConfig(), mode="bilinear")
    wiener = warm_start(obs.y, obs.mask, PSFConfig(), mode="wiener")
    fill_first = warm_start(obs.y, obs.mask, PSFConfig(), mode="wiener_fill_first")
    assert torch.allclose(bilinear, wiener, atol=1e-6)
    assert torch.allclose(bilinear, fill_first, atol=1e-6)


def test_warm_start_fills_and_preserves_measurements():
    x = dense()
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode="y", ky=2)))
    x0 = warm_start(obs.y, obs.mask, PSFConfig(), mode="wiener")
    assert torch.equal(x0[obs.mask], obs.y[obs.mask])       # never overwrites data
    assert float(x0[~obs.mask].abs().sum()) > 0.0           # and fills the gaps


def test_deconvolution_actually_changes_output():
    psf = PSFConfig(sigma_xy_um=1.5, sigma_z_um=0.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    x = dense()
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode="full"), psf=psf))
    x0 = warm_start(obs.y, obs.mask, psf, lam=0.01, mode="wiener")
    assert not torch.allclose(x0, obs.y, atol=1e-4)


def test_iterdc_runs_and_stays_finite():
    x = dense()
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode="yz", ky=2, kz=2)))
    out = warm_start(obs.y, obs.mask, PSFConfig(), mode="iterdc:1e-3:3")
    assert torch.isfinite(out).all()
    assert out.shape == SHAPE

"""Normalization-leakage and observation-builder tests (PLAN.md P0-3 sanity).

The core guarantee: once fitted on training stacks, the calibration applies
stored constants only.  Nothing in the eval path derives statistics from the
sample being processed -- enforced here by construction checks.
"""

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.normalization import IntensityCalibration
from data.observation import build_observation
from physics.acquisition import AcquisitionConfig, NoiseConfig, simulate_acquisition
from physics.psf import PSFConfig


def test_calibration_uses_stored_constants_only():
    """apply() must be a pure affine map of stored constants."""
    calib = IntensityCalibration(offset=100.0, scale=10000.0)
    x = torch.tensor([[-247.0, 0.0], [100.0, 13306.0]])
    out = calib.apply(x)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0], [0.0, 1.0]]), atol=1e-4)
    # A different sample with wildly different statistics must NOT change constants.
    out2 = calib.apply(x * 3 + 500)
    assert torch.allclose(out2, ((x * 3 + 500 - 100.0) / 10000.0).clamp(0, 1), atol=1e-6)


def test_fit_is_training_only_by_signature():
    """fit() takes explicit training volumes; there is no path that reads a
    per-sample statistic at apply time."""
    src = inspect.getsource(IntensityCalibration.apply)
    for banned in (".min()", ".max()", "percentile"):
        assert banned not in src, f"apply() must not compute {banned}"


def test_observation_builder_contract():
    """build_observation returns (Y, M, X0, C_s, U_p) with correct semantics."""
    torch.manual_seed(0)
    x = torch.rand(9, 32, 32) * 0.5 + 0.25
    psf = PSFConfig(sigma_xy_um=1.2, sigma_z_um=0.0, voxel_xy_um=1.0, voxel_z_um=1.0)
    cfg = AcquisitionConfig(mode="z", K=3, psf=psf, noise=NoiseConfig(gaussian_sigma=0.01))
    obs = simulate_acquisition(x, cfg)
    mi = build_observation(obs, psf=psf, lam=1e-3, warm_start_kind="wiener")

    assert mi.y.shape == x.shape and mi.mask.shape == x.shape
    assert torch.equal(mi.mask, obs.mask.float())
    assert mi.confidence.shape == x.shape and mi.uncertainty.shape == x.shape
    # z-K3: measured slices 0,3,6.  A measured pixel has full lateral support
    # on its own slice but no axial support from its unmeasured neighbours,
    # so its confidence is strictly higher than the axial-gap slices' but is
    # NOT 1 (the "fully sampled -> 1" property only holds for full masks,
    # cf. test_confidence_fully_sampled_is_one).
    c, u = mi.confidence, mi.uncertainty
    # with sigma_z=0 optics, unmeasured slices get no axial coupling:
    # reconstruction there is fully unsupported => uncertainty 1
    assert u[1, 8, 8] > 0.99 and u[3, 8, 8] < 0.01
    assert c.min() >= 0.0 and c.max() <= 1.0
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_confidence_decays_with_distance_to_support():
    """C_s must decrease monotonically with distance from the only
    measured slice (and U_p, derived from the same geometry, must increase)."""
    mask = torch.zeros(9, 8, 8, dtype=torch.bool)
    mask[0] = True
    obs = type("O", (), {"y": torch.rand(9, 8, 8), "mask": mask,
                         "metadata": {"acquisition": {"K": 9}}})()
    mi = build_observation(obs, psf=PSFConfig(sigma_xy_um=1.0, sigma_z_um=1.0),
                           lam=1e-3, warm_start_kind="bilinear")
    col_c = mi.confidence[:, 4, 4]
    col_u = mi.uncertainty[:, 4, 4]
    assert col_c[1] > col_c[3] > col_c[5] > col_c[7]
    # with sigma_z>0 the operator couples slices, so uncertainty decays with
    # distance from the measured support (saturating at 1 far away)
    assert col_u[1] <= col_u[3] <= col_u[5] <= col_u[7]
    assert col_u[7] > col_u[1]


def test_confidence_fully_sampled_is_one():
    from physics.sampling import make_mask
    mask = make_mask("xy", 1, 3, 16, 16)
    obs = type("O", (), {"y": torch.rand(3, 16, 16), "mask": mask,
                         "metadata": {"acquisition": {"K": 1}}})()
    mi = build_observation(obs, psf=None, warm_start_kind="bilinear")
    assert torch.allclose(mi.confidence, torch.ones(3, 16, 16), atol=1e-4)
    assert torch.allclose(mi.uncertainty, torch.zeros(3, 16, 16), atol=1e-4)

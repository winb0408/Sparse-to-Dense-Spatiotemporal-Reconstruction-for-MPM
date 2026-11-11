"""Contract of the no-calibration PSF estimators.

Two properties matter more than any number here:

1. the arithmetic refuses the inputs that make it meaningless (NA >= n), and
   reports its width **in voxels** -- a PSF narrower than a voxel cannot be
   recovered from a sampled stack, and that is the fact this cohort turns on;
2. the held-out scan recovers a width it can see, returns identity for data
   with no blur, and never forgets to report that a row where ``W0 == I0`` was
   skipped rather than tested.
"""

from __future__ import annotations

import torch

from spur_mpm.data.sparse import measured_mask
from spur_mpm.physics.psf import PSFConfig, gaussian_otf_3d
from spur_mpm.physics.psf_estimate import (
    best_sigma,
    diffraction_limited_sigma,
    holdout_width_scan,
)
from spur_mpm.physics.prior import PSF_SOURCES, psf_source_label
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import interp_linear, wiener_fill_all


# --- 1. the arithmetic ------------------------------------------------------

def test_the_diffraction_limit_is_reported_in_voxels_not_only_microns():
    d = diffraction_limited_sigma(excitation_nm=1340, na=1.05, n_medium=1.33,
                                  n_photon=3, voxel_xy_um=0.98, voxel_z_um=5.0)
    assert d["sigma_xy_um"] > 0 and d["sigma_z_um"] > 0
    # thinner than a voxel on both axes, which is the finding, not an accident
    assert d["sigma_xy_px"] < 1.0
    assert d["sigma_z_px"] < 1.0
    assert d["sigma_xy_px"] == d["sigma_xy_um"] / 0.98
    assert d["psf"]["is_identity"] is False


def test_the_arithmetic_scales_the_way_the_formula_says():
    base = diffraction_limited_sigma(excitation_nm=1340, na=1.05)
    wider_xy = diffraction_limited_sigma(excitation_nm=1340, na=1.0)
    longer = diffraction_limited_sigma(excitation_nm=1700, na=1.05)
    one_photon = diffraction_limited_sigma(excitation_nm=1340, na=1.05, n_photon=1)
    assert wider_xy["sigma_xy_um"] > base["sigma_xy_um"]      # lower NA, wider
    assert longer["sigma_xy_um"] > base["sigma_xy_um"]        # longer lambda, wider
    assert one_photon["sigma_xy_um"] > base["sigma_xy_um"]    # no 3-photon narrowing


def test_an_na_at_or_above_the_index_is_refused():
    for na in (1.33, 1.4):
        try:
            diffraction_limited_sigma(excitation_nm=1340, na=na, n_medium=1.33)
        except ValueError as exc:
            assert "below the medium index" in str(exc)
        else:
            raise AssertionError(f"NA={na} should have been refused")


def test_non_physical_optical_parameters_are_refused():
    for kwargs in ({"excitation_nm": 0}, {"excitation_nm": -1}, {"na": 0},
                   {"n_medium": 0}, {"voxel_xy_um": 0}, {"n_photon": 0}):
        args = {"excitation_nm": 1340, "na": 1.0, **kwargs}
        try:
            diffraction_limited_sigma(**args)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should have been refused")


# --- 2. the held-out scan ---------------------------------------------------

def _phantom(shape=(12, 48, 48), seed=0) -> torch.Tensor:
    """Sparse blob phantom: structures thin enough for blur to be visible."""
    g = torch.Generator().manual_seed(seed)
    z, h, w = shape
    vol = torch.zeros(shape)
    for _ in range(28):
        cz = int(torch.randint(1, z - 1, (1,), generator=g))
        cy = int(torch.randint(3, h - 3, (1,), generator=g))
        cx = int(torch.randint(3, w - 3, (1,), generator=g))
        vol[cz, cy, cx] = 1.0
    # one-voxel blobs blurred only by the PSF: the object itself is a delta grid
    return vol.clamp(0, 1)


def _blur(vol: torch.Tensor, psf: PSFConfig) -> torch.Tensor:
    f = torch.fft.fftn(vol, dim=(-3, -2, -1))
    otf = gaussian_otf_3d(tuple(vol.shape), psf)
    return torch.fft.ifftn(f * otf, dim=(-3, -2, -1)).real


def test_the_scan_recovers_a_lateral_width_it_can_see():
    voxel = 0.98
    true_um = 2.0
    sharp = _phantom()
    blurred = _blur(sharp, PSFConfig(true_um, 0.0, voxel, 5.0))
    mask = torch.from_numpy(measured_mask(tuple(sharp.shape),
                                          SamplingSpec(mode="y", ky=2)))
    rows = holdout_width_scan(blurred, mask, ref=sharp, axis="xy",
                              sigmas_um=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                              lam=1e-3, voxel_xy_um=voxel)
    chosen = best_sigma(rows)
    assert abs(chosen["sigma_um"] - true_um) <= 0.5, chosen
    # and the scan must actually have done something at that width
    assert chosen["changed_fraction"] > 0.5


def test_data_with_no_blur_estimates_identity():
    sharp = _phantom()
    mask = torch.from_numpy(measured_mask(tuple(sharp.shape),
                                          SamplingSpec(mode="y", ky=2)))
    rows = holdout_width_scan(sharp, mask, ref=sharp, axis="xy",
                              sigmas_um=(0.0, 0.5, 1.0, 2.0), lam=1e-3)
    assert best_sigma(rows)["sigma_um"] == 0.0
    # every wider candidate has to be worse, not merely non-better
    assert all(r["delta_psnr_vs_identity"] < 0 for r in rows[1:])


def test_scoring_against_the_measurements_is_not_the_same_question_as_scoring_against_the_object():
    """The two `ref` choices give different answers, and the signature says so.

    With the blurred volume as its own reference, deconvolving cannot help: the
    held-out voxels *are* blurred, so sharpening moves away from them. That is
    the real-data case, and it is why the module refuses to call the number a
    PSF.
    """
    voxel = 0.98
    sharp = _phantom()
    blurred = _blur(sharp, PSFConfig(2.0, 0.0, voxel, 5.0))
    mask = torch.from_numpy(measured_mask(tuple(sharp.shape),
                                          SamplingSpec(mode="y", ky=2)))
    rows = holdout_width_scan(blurred, mask, axis="xy",
                              sigmas_um=(0.0, 1.0, 2.0), lam=1e-3,
                              voxel_xy_um=voxel)
    assert best_sigma(rows)["sigma_um"] == 0.0


def test_the_axial_scan_only_moves_the_axial_width():
    sharp = _phantom()
    mask = torch.from_numpy(measured_mask(tuple(sharp.shape),
                                          SamplingSpec(mode="z", kz=2)))
    rows = holdout_width_scan(sharp, mask, ref=sharp, axis="z",
                              sigmas_um=(0.0, 5.0), lam=1e-3,
                              voxel_xy_um=0.98, voxel_z_um=5.0)
    assert rows[1]["sigma_px"] == 1.0     # 5 um at a 5 um step
    assert rows[0]["is_identity"] is True


def test_the_scan_refuses_masks_that_leave_nothing_to_score():
    vol = _phantom()
    for mask, needle in (
            (torch.ones_like(vol, dtype=torch.bool), "nothing withheld"),
            (torch.zeros_like(vol, dtype=torch.bool), "nothing to reconstruct")):
        try:
            holdout_width_scan(vol, mask, sigmas_um=(0.0, 1.0))
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError("an unusable mask should have been refused")


def test_the_scan_requires_an_identity_baseline_and_a_valid_axis():
    vol = _phantom()
    mask = torch.from_numpy(measured_mask(tuple(vol.shape),
                                          SamplingSpec(mode="y", ky=2)))
    for kwargs, needle in (({"sigmas_um": (0.5, 1.0)}, "must start at 0.0"),
                           ({"sigmas_um": (0.0, -1.0)}, "must be >= 0"),
                           ({"sigmas_um": (0.0,), "axis": "t"}, "axis must be")):
        try:
            holdout_width_scan(vol, mask, **kwargs)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"{kwargs} should have been refused")


# --- 3. the label this estimate is allowed to wear --------------------------

def test_an_estimated_psf_has_its_own_label_and_is_not_the_calibrated_one():
    assert "estimated_from_data" in PSF_SOURCES
    assert "estimated_from_data" != "calibrated_optical"
    psf = PSFConfig(sigma_xy_um=0.6, sigma_z_um=0.0, voxel_xy_um=0.98,
                    voxel_z_um=5.0)
    assert psf_source_label(psf, "estimated_from_data") == "estimated_from_data"


def test_an_estimate_without_its_label_is_still_refused():
    psf = PSFConfig(sigma_xy_um=0.6, sigma_z_um=0.0)
    try:
        psf_source_label(psf, None)
    except ValueError as exc:
        assert "psf_source" in str(exc)
    else:
        raise AssertionError("an unlabelled estimated PSF should be refused")


# --- 4. Gate 1: lambda stability on a phantom with a known width -------------
#
# The pair of results below is the whole point of scanning lambda at all. When a
# blur IS present, deconvolving beats interpolating -- so the machinery works.
# When lambda is too small, noise gets amplified faster than blur gets removed.
# Both have to be pinned, because "lower lambda is sharper" is the intuition
# that makes an over-deconvolved figure look better than it is.

def _phantom_pair(shape=(16, 64, 64), psf_um=1.0, noise=0.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    sharp = torch.zeros(shape)
    for _ in range(120):
        z = int(torch.randint(0, shape[0], (1,), generator=g))
        y = int(torch.randint(0, shape[1], (1,), generator=g))
        x = int(torch.randint(0, shape[2], (1,), generator=g))
        sharp[z, y, x] = 1.0
    psf = PSFConfig(psf_um, 0.0, 0.98, 5.0)
    blurred = torch.fft.ifftn(torch.fft.fftn(sharp) * gaussian_otf_3d(shape, psf),
                              dim=(-3, -2, -1)).real
    if noise:
        blurred = blurred + noise * torch.randn(shape, generator=g)
    mask = torch.from_numpy(measured_mask(shape, SamplingSpec(mode="y", ky=2)))
    y = torch.where(mask, blurred, torch.zeros_like(blurred))
    return sharp, y, mask, psf


def _held_rmse(pred, sharp, mask):
    held = ~mask
    return float(((pred[held] - sharp[held]) ** 2).mean().sqrt())


def test_wiener_stays_finite_across_five_orders_of_magnitude_of_lambda():
    sharp, y, mask, psf = _phantom_pair()
    for lam in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        w0 = wiener_fill_all(y, mask, psf, lam)
        assert torch.isfinite(w0).all(), lam
        # bounded, not merely finite: a Wiener with a vanishing lambda divides
        # by an almost-zero OTF and can return 1e6 without ever producing a NaN
        assert float(w0.abs().max()) < 10.0, (lam, float(w0.abs().max()))


def test_when_a_blur_is_present_deconvolution_beats_interpolation():
    """The `there is blur` case -- it must work, or the negative real-data
    result would be uninformative (an operator that never helps explains
    nothing)."""
    sharp, y, mask, psf = _phantom_pair()
    baseline = _held_rmse(interp_linear(y, mask), sharp, mask)
    best = min(_held_rmse(wiener_fill_all(y, mask, psf, lam), sharp, mask)
               for lam in (1e-3, 1e-2, 1e-1))
    assert best < baseline, (best, baseline)


def test_too_small_a_lambda_is_worse_than_a_regularised_one_even_without_noise():
    sharp, y, mask, psf = _phantom_pair()
    err = {lam: _held_rmse(wiener_fill_all(y, mask, psf, lam), sharp, mask)
           for lam in (1e-5, 1e-2)}
    assert err[1e-5] > err[1e-2], err


def test_under_noise_over_deconvolution_loses_to_not_deconvolving_at_all():
    sharp, y, mask, psf = _phantom_pair(noise=0.01)
    baseline = _held_rmse(interp_linear(y, mask), sharp, mask)
    aggressive = _held_rmse(wiener_fill_all(y, mask, psf, 1e-4), sharp, mask)
    regularised = _held_rmse(wiener_fill_all(y, mask, psf, 1e-1), sharp, mask)
    assert aggressive > 2.0 * baseline, (aggressive, baseline)
    assert regularised < aggressive, (regularised, aggressive)

"""The physics-prior contract: I0, W0, phys_delta and what may be claimed.

The plan's Gate 1 (PR2 §7) is a set of equalities and refusals, not a metric:

* ``I0`` is interpolation of the *measurement*, never of a dense reference;
* ``H = I`` gives ``W0 == I0`` exactly, so a run under an identity PSF cannot
  report a deconvolution gain;
* a non-identity ``H`` must fill every hole **before** the FFT, and must carry
  a provenance label;
* the new mask-aware inverse holds unmeasured voxels at ``I0`` while the old
  ``iterative_dc_inverse`` is left shrinking toward zero -- deliberately, and
  separately.

If one of these stops holding, the corresponding sentence in
``docs/PHASE1_PR2_METHOD_SPEC.md`` is wrong, which is why they are tests.
"""

import pytest
import torch

from spur_mpm.data.observation import FEATURES, build_model_inputs
from spur_mpm.physics.acquisition import AcquisitionConfig, observe
from spur_mpm.physics.prior import (
    mask_aware_inverse,
    physics_prior,
    prior_summary,
)
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import (
    WARM_START_MODES,
    interp_linear,
    iterative_dc_inverse,
    warm_start,
    wiener_deconv_3d,
    wiener_fill_all,
)

SHAPE = (7, 32, 32)
BLUR = PSFConfig(sigma_xy_um=1.5, sigma_z_um=4.0, voxel_xy_um=0.98,
                 voxel_z_um=5.0)


def observation(mode="y", ky=2, kz=1, shape=SHAPE, seed=0):
    x = torch.rand(shape, generator=torch.Generator().manual_seed(seed))
    obs = observe(x, AcquisitionConfig(
        sampling=SamplingSpec(mode=mode, ky=ky, kz=kz)))
    return x, obs


# ---------------------------------------------------------------------------
# 1. I0, W0, and the identity case
# ---------------------------------------------------------------------------

def test_i0_is_interpolation_of_the_measurement_only():
    _, obs = observation("y", ky=2)
    i0 = interp_linear(obs.y, obs.mask)
    assert torch.equal(i0[obs.mask], obs.y[obs.mask])
    assert torch.allclose(i0, warm_start(obs.y, obs.mask, PSFConfig(),
                                         mode="bilinear"))


def test_identity_psf_gives_w0_equals_i0_exactly():
    """Not "close": the Wiener step is not run at all when there is no optics."""
    _, obs = observation("y", ky=2)
    i0 = interp_linear(obs.y, obs.mask)
    w0 = wiener_fill_all(obs.y, obs.mask, PSFConfig())
    assert torch.equal(w0, i0)
    prior = physics_prior(obs.y, obs.mask, PSFConfig())
    assert torch.equal(prior["wiener"], prior["interp"])
    assert float(prior["phys_delta"].abs().sum()) == 0.0
    assert prior["metadata"]["phys_delta_zero"] is True


def test_a_non_identity_psf_actually_changes_the_candidate():
    _, obs = observation("y", ky=2)
    prior = physics_prior(obs.y, obs.mask, BLUR, psf_source="synthetic_extra_degradation")
    assert not torch.equal(prior["wiener"], prior["interp"])
    assert float(prior["phys_delta"].abs().max()) > 0
    for key in ("interp", "wiener", "phys_delta"):
        assert bool(torch.isfinite(prior[key]).all()), key


def test_fill_first_and_fill_last_are_not_the_same_operator():
    """Why the order matters, measured: zero-filled holes change the result.

    ``W0`` is computed on a *filled* volume. Convolving the zero-filled
    observation instead is what the old ``wiener`` mode does, and it produces a
    different image -- the difference is the ringing/darkening the fill is
    there to prevent. This test does not claim which is better; it claims they
    are distinguishable, which is what makes the fix meaningful.
    """
    x, obs = observation("y", ky=2)
    filled = wiener_fill_all(obs.y, obs.mask, BLUR)
    holes_and_all = wiener_deconv_3d(obs.y, BLUR)
    assert not torch.allclose(filled, holes_and_all, atol=1e-6)
    # Where it goes wrong, measured rather than asserted: convolving the
    # zero-filled observation drags the hole regions toward zero (the blur
    # kernel reaches into the zeros), so the deconvolved estimate is roughly
    # half as bright there as the true image -- a bias in exactly the region
    # the reconstruction is supposed to be inventing. Filling first removes it.
    holes = ~obs.mask
    truth = float(x[holes].mean())
    fill_first = float(filled[holes].mean())
    zero_filled = float(holes_and_all[holes].mean())
    assert truth > 0.4
    assert abs(fill_first - truth) / truth < 0.1
    assert zero_filled < 0.6 * truth


def test_w0_never_reads_the_dense_reference():
    """The mask probe: change GT at unmeasured voxels, nothing may move."""
    x, obs = observation("y", ky=2)
    moved = x.clone()
    moved[~obs.mask] = 17.0
    obs_b = observe(moved, AcquisitionConfig(
        sampling=SamplingSpec(mode="y", ky=2)))
    assert torch.equal(obs.y, obs_b.y) and torch.equal(obs.mask, obs_b.mask)
    a = physics_prior(obs.y, obs.mask, BLUR, psf_source="synthetic_extra_degradation")
    b = physics_prior(obs_b.y, obs_b.mask, BLUR,
                      psf_source="synthetic_extra_degradation")
    for key in ("interp", "wiener", "phys_delta", "u_physics"):
        assert torch.equal(a[key], b[key]), key
    ia = build_model_inputs(obs.y, obs.mask, BLUR,
                            psf_source="synthetic_extra_degradation")
    ib = build_model_inputs(obs_b.y, obs_b.mask, BLUR,
                            psf_source="synthetic_extra_degradation")
    for feat in FEATURES:
        assert torch.equal(ia.feature(feat), ib.feature(feat)), feat


# ---------------------------------------------------------------------------
# 2. every axis is filled before the FFT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kw", [("z", {"kz": 3}), ("yz", {"ky": 2, "kz": 2})])
def test_axial_holes_are_filled_before_deconvolution(mode, kw):
    """The plan's T1 complaint: the old path FFT'd hole-filled volumes along Z."""
    _, obs = observation(mode, **kw)
    i0 = interp_linear(obs.y, obs.mask)
    # No zero-valued hole survives the fill: every unmeasured voxel carries a
    # value interpolated from measured knots.
    mask_z = obs.mask.amax(dim=(-2, -1))
    assert (~mask_z).any(), "this protocol must actually skip planes"
    assert float(i0[~mask_z].abs().amax()) > 0.0
    assert torch.allclose(i0, wiener_fill_all(obs.y, obs.mask, PSFConfig()))
    out = wiener_fill_all(obs.y, obs.mask, BLUR)
    assert bool(torch.isfinite(out).all())


def test_the_old_fill_first_mode_still_only_fills_lateral_holes():
    """Old semantics must survive: checkpoints were trained with them.

    ``wiener_fill_first`` fills laterally and lets the Z pipeline place the
    planes; ``wiener_fill_all`` interpolates along Z first. For a Y-only mask
    they coincide (there are no Z gaps), for a Z mask they need not.
    """
    assert "wiener_fill_first" in WARM_START_MODES
    assert "wiener_fill_all" in WARM_START_MODES
    _, obs_y = observation("y", ky=2)
    assert torch.allclose(warm_start(obs_y.y, obs_y.mask, PSFConfig(),
                                     mode="wiener_fill_first"),
                          warm_start(obs_y.y, obs_y.mask, PSFConfig(),
                                     mode="wiener_fill_all"))
    _, obs_z = observation("z", kz=3)
    old = warm_start(obs_z.y, obs_z.mask, BLUR, mode="wiener_fill_first")
    new = warm_start(obs_z.y, obs_z.mask, BLUR, mode="wiener_fill_all")
    assert not torch.equal(old, new)


def test_a_window_with_no_measurement_at_all_is_safe():
    """Padding, or a mask that missed everything, must not crash or invent data."""
    y = torch.zeros(5, 8, 8)
    m = torch.zeros(5, 8, 8, dtype=torch.bool)
    assert float(interp_linear(y, m).abs().sum()) == 0.0
    assert float(wiener_fill_all(y, m, BLUR).abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# 3. refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, -0.1, float("inf"), float("nan")])
def test_an_invalid_lambda_is_refused(lam):
    _, obs = observation("y", ky=2)
    with pytest.raises(ValueError, match="lambda"):
        wiener_fill_all(obs.y, obs.mask, PSFConfig(), lam=lam)
    with pytest.raises(ValueError, match="lambda"):
        physics_prior(obs.y, obs.mask, PSFConfig(), lam=lam)


def test_a_negative_psf_sigma_is_refused():
    _, obs = observation("y", ky=2)
    bad = PSFConfig(sigma_xy_um=-1.0)
    with pytest.raises(ValueError, match="sigma"):
        wiener_fill_all(obs.y, obs.mask, bad)


def test_non_finite_input_is_refused():
    _, obs = observation("y", ky=2)
    y = obs.y.clone()
    y[obs.mask] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        physics_prior(y, obs.mask, PSFConfig())


def test_an_unlabelled_non_identity_psf_is_refused():
    """`H != I` must say where it came from before it may produce a number."""
    _, obs = observation("y", ky=2)
    with pytest.raises(ValueError, match="psf_source"):
        physics_prior(obs.y, obs.mask, BLUR)
    with pytest.raises(ValueError, match="unknown psf_source"):
        physics_prior(obs.y, obs.mask, BLUR, psf_source="looked_about_right")
    ok = physics_prior(obs.y, obs.mask, BLUR, psf_source="calibrated_optical")
    assert ok["metadata"]["psf_source"] == "calibrated_optical"
    identity = physics_prior(obs.y, obs.mask, PSFConfig())
    assert identity["metadata"]["psf_source"] == "identity"


def test_a_shape_mismatch_is_refused():
    _, obs = observation("y", ky=2)
    with pytest.raises(ValueError, match="disagree"):
        physics_prior(obs.y, obs.mask[:, :16], PSFConfig())


# ---------------------------------------------------------------------------
# 4. the mask-aware inverse holds the prior; the old one does not
# ---------------------------------------------------------------------------

def test_mask_aware_inverse_holds_unmeasured_voxels_at_i0():
    y = torch.zeros(3, 8, 8)
    m = torch.zeros(3, 8, 8, dtype=torch.bool)
    m[1, 2, :] = True
    y[1, 2, :] = 4.0
    i0 = interp_linear(y, m)
    assert float(i0[1, 2, :].mean()) == pytest.approx(4.0)
    x = mask_aware_inverse(y, m, i0, PSFConfig(), lam=0.01, steps=40)
    unmeasured = ~m
    assert torch.allclose(x[unmeasured], i0[unmeasured], atol=1e-6)
    # And measured voxels stay near the measurement instead of being shrunk.
    assert float((x[m] - y[m]).abs().max()) < 0.1


def test_the_old_iterative_inverse_is_unchanged_and_still_shrinks_toward_zero():
    """A regression lock, not an endorsement: the old operator is not fixed here.

    Re-centring it would change what an existing warm-start mode means, and
    old checkpoints were trained against the old meaning. The new operator is
    :func:`mask_aware_inverse`; this test fails if someone "tidies up" the old
    one by accident.
    """
    y = torch.zeros(3, 8, 8)
    m = torch.zeros(3, 8, 8, dtype=torch.bool)
    m[1, 2, :] = True
    y[1, 2, :] = 4.0
    old = iterative_dc_inverse(y, m, PSFConfig(), lam=0.5, steps=40)
    new = mask_aware_inverse(y, m, interp_linear(y, m), PSFConfig(), lam=0.5, steps=40)
    assert float(old[~m].abs().max()) < float(new[~m].abs().max())
    assert float(new[~m].abs().max()) > 0.0


def test_mask_aware_inverse_is_finite_and_improves_the_fit():
    _, obs = observation("y", ky=2)
    i0 = interp_linear(obs.y, obs.mask)
    x = mask_aware_inverse(obs.y, obs.mask, i0, BLUR, lam=0.05, steps=20)
    assert bool(torch.isfinite(x).all())
    m = obs.mask
    assert float((x[m] - obs.y[m]).abs().mean()) < 0.5


@pytest.mark.parametrize("steps", [0, -3])
def test_a_non_positive_step_count_is_refused(steps):
    _, obs = observation("y", ky=2)
    with pytest.raises(ValueError, match="steps"):
        mask_aware_inverse(obs.y, obs.mask, interp_linear(obs.y, obs.mask),
                           PSFConfig(), steps=steps)


# ---------------------------------------------------------------------------
# 5. the builder exposes the new features without changing the old ones
# ---------------------------------------------------------------------------

def test_the_builder_exposes_the_new_features():
    _, obs = observation("y", ky=2)
    mi = build_model_inputs(obs.y, obs.mask, PSFConfig(), warm_start_kind="bilinear")
    for feat in ("y", "mask", "x0", "confidence", "uncertainty",
                 "interp", "wiener", "phys_delta", "u_sampling", "u_physics"):
        assert feat in FEATURES
        assert mi.feature(feat).shape == obs.y.shape, feat
    assert mi.valid_z.shape == (obs.y.shape[0],)
    assert bool(mi.valid_z.all())
    assert mi.metadata["psf_source"] == "identity"


def test_the_old_x0_semantics_are_untouched():
    """``x0`` still means "this arm's warm start", not "I0" and not "W0"."""
    _, obs = observation("y", ky=2)
    for mode in ("bilinear", "wiener", "wiener_fill_first", "wiener_fill_all"):
        mi = build_model_inputs(obs.y, obs.mask, PSFConfig(), warm_start_kind=mode)
        assert torch.equal(mi.x0, warm_start(obs.y, obs.mask, PSFConfig(), mode=mode)), mode
    bil = build_model_inputs(obs.y, obs.mask, PSFConfig(), warm_start_kind="bilinear")
    assert torch.equal(bil.x0, bil.interp)
    assert torch.equal(bil.wiener, bil.interp)


def test_u_physics_is_zero_when_the_candidates_agree():
    """Under H = I there is nothing for a gate to weigh -- and the arm that
    would use it must not be launched on this cohort."""
    _, obs = observation("y", ky=2)
    mi = build_model_inputs(obs.y, obs.mask, PSFConfig())
    assert float(mi.u_physics.abs().max()) == 0.0
    mi_b = build_model_inputs(obs.y, obs.mask, BLUR,
                              psf_source="synthetic_extra_degradation")
    assert float(mi_b.u_physics.max()) > 0.0
    assert float(mi_b.u_physics.min()) >= 0.0
    assert float(mi_b.u_physics.max()) <= 1.0


def test_valid_z_is_carried_through():
    _, obs = observation("y", ky=2)
    vz = torch.tensor([False, True, True, True, True, True, False])
    mi = build_model_inputs(obs.y, obs.mask, PSFConfig(), valid_z=vz)
    assert torch.equal(mi.valid_z, vz)
    with pytest.raises(ValueError, match="valid_z"):
        build_model_inputs(obs.y, obs.mask, PSFConfig(), valid_z=vz[:3])


def test_valid_z_is_not_a_stackable_channel():
    _, obs = observation("y", ky=2)
    mi = build_model_inputs(obs.y, obs.mask, PSFConfig())
    with pytest.raises(KeyError):
        mi.stack(("y", "valid_z"))


def test_the_prior_summary_is_json_able_and_says_what_it_is():
    _, obs = observation("y", ky=2)
    s = prior_summary(physics_prior(obs.y, obs.mask, PSFConfig()))
    assert s["psf_source"] == "identity" and s["phys_delta_zero"] is True
    assert set(s) >= {"interp_range", "wiener_range", "phys_delta_abs_mean",
                      "phys_delta_abs_max", "u_physics_mean", "operator"}
    assert "NOT a calibrated" in s["u_physics_kind"]

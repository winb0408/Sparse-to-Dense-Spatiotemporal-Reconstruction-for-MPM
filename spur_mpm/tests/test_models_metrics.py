"""Model wiring, uncertainty fields and metrics."""

import numpy as np
import pytest
import torch

from spur_mpm.metrics.image_metrics import mae, psnr, split_errors, ssim
from spur_mpm.metrics.statistics import bootstrap_ci, group_mean, paired_delta
from spur_mpm.metrics.vessel import (cl_dice, dice, mask_vessel_metrics,
                                    segment_pair, vessel_metrics)
from spur_mpm.models.registry import ARMS, build_model, get_arm
from spur_mpm.models.uncertainty import (
    geometric_uncertainty,
    physics_uncertainty,
    sampling_confidence,
)

SHAPE = (4, 32, 32)


@pytest.mark.parametrize("arm_name", [a for a in ARMS if ARMS[a].needs_training])
def test_model_forward_shape(arm_name):
    arm = get_arm(arm_name)
    model = build_model(arm, context_depth=1, base=8, window=4,
                        depths=(1, 1, 1), heads=(2, 4, 8))
    # A dual arm is handed one feature stack per branch, so its input tensor is
    # twice as wide while its output stays a single plane per branch.
    n = 2 if arm.dual else 1
    x = torch.rand(2, n * len(arm.features), 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_context_depth_becomes_channels():
    arm = get_arm("SS1")
    model = build_model(arm, context_depth=3, base=8, window=4,
                        depths=(1, 1, 1), heads=(2, 4, 8))
    out = model(torch.rand(1, len(arm.features) * 3, 32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_fusion_keeps_backbone_width_comparable():
    plain = get_arm("SS1")
    fused = get_arm("SS2g")
    assert fused.features == tuple(ARMS["SS2"].features)
    assert plain.fusion is False and fused.fusion is True


def test_untrained_arm_cannot_build_a_network():
    with pytest.raises(ValueError):
        build_model(get_arm("A0"), context_depth=1)


def test_confidence_is_one_when_fully_sampled():
    mask = torch.ones(SHAPE, dtype=torch.bool)
    c = sampling_confidence(mask, sigma=1.0)
    assert float(c.min()) > 0.95


def test_geometric_uncertainty_behaves():
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[:, ::2, :] = True                       # y k=2 lattice, fully measured rows
    u = geometric_uncertainty(mask, ky=2, kz=1)
    assert u.shape == SHAPE
    assert float(u[mask].abs().max()) == pytest.approx(0.0, abs=1e-6)  # measured -> certain
    assert float(u[~mask].max()) > 0.5                        # gaps -> uncertain
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0


def test_geometric_uncertainty_respects_axial_pitch():
    mask = torch.zeros((8, 8, 8), dtype=torch.bool)
    mask[::4] = True                              # plane lattice kz=4
    per_axis = geometric_uncertainty(mask, ky=1, kz=4, mode="per_axis")
    physical = geometric_uncertainty(mask, ky=1, kz=4, mode="physical",
                                     voxel_z_um=5.0, voxel_xy_um=0.98)
    assert float(per_axis[mask].abs().max()) == pytest.approx(0.0, abs=1e-6)
    assert not torch.allclose(per_axis, physical)


def test_physics_uncertainty_identity_falls_back_to_not_measured():
    from spur_mpm.physics.psf import PSFConfig
    mask = torch.zeros((4, 4, 4), dtype=torch.bool)
    mask[0, 0, 0] = True
    u = physics_uncertainty(mask, PSFConfig())
    assert float(u[0, 0, 0]) == 0.0 and float(u[1, 1, 1]) == 1.0


def test_image_metrics():
    a = torch.rand(2, 16, 16)
    assert float(psnr(a, a)) > 100
    assert float(ssim(a[:, None], a[:, None])) == pytest.approx(1.0, abs=1e-4)
    assert float(mae(a, a)) == pytest.approx(0.0)
    mask = torch.zeros(2, 16, 16, dtype=torch.bool)
    mask[:, :8] = True
    e_meas, e_miss = split_errors(a, a + 0.1, mask)
    assert e_meas == pytest.approx(0.1, abs=1e-5) and e_miss == pytest.approx(0.1, abs=1e-5)


def test_vessel_metrics_identical_is_perfect():
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64), dtype=np.float32)
    img[30:34, :] = 1.0
    img[:, 30:34] += 1.0
    img += 0.02 * rng.random(img.shape)
    m = vessel_metrics(img, img)
    assert m["dice"] == pytest.approx(1.0)
    assert m["cl_dice"] == pytest.approx(1.0)
    assert dice(np.ones((4, 4)), np.ones((4, 4))) == 1.0
    assert cl_dice(np.zeros((4, 4), bool), np.ones((4, 4), bool)) == 0.0


def test_segment_pair_never_lets_the_prediction_move_the_threshold():
    """The operating point comes from the reference; a blank prediction must not
    be able to redefine "vessel" so that it agrees with itself."""
    from scipy.ndimage import uniform_filter
    rng = np.random.default_rng(1)
    # Correlated background rather than pixel white noise: the Sato response is
    # a second-derivative filter, so uncorrelated noise produces a ridge-like
    # response everywhere and would make the fraction assertion meaningless.
    ref = uniform_filter(rng.normal(0.4, 0.05, (6, 48, 48)), size=(1, 3, 3)
                         ).astype(np.float32)
    ref[:, 20:23, :] = 1.0
    good = ref + rng.normal(0, 0.01, ref.shape).astype(np.float32)
    blank = np.zeros_like(ref)

    a = segment_pair(good, ref, method="vesselness", threshold_mode="per_plane")
    b = segment_pair(blank, ref, method="vesselness", threshold_mode="per_plane")
    assert a["threshold"] == pytest.approx(b["threshold"])
    assert b["pred_mask"].sum() == 0                  # all-zero response, empty mask
    assert a["ref_mask"].sum() == b["ref_mask"].sum()  # reference mask is untouched
    # And the response is what gets thresholded, so a bright hollow background
    # does not become vessel: the tube is a few percent of the volume.
    assert 0.0 < float(a["ref_mask"].mean()) < 0.15
    m = mask_vessel_metrics(a["pred_mask"], a["ref_mask"])
    assert m["dice"] > 0.5
    with pytest.raises(ValueError):
        segment_pair(good, ref, method="otsu")


def test_flood_gate_empties_planes_the_threshold_cannot_describe():
    """A plane the reference segmentation fills is dropped, not scored.

    On a real M2 volume a third of some planes came out above threshold; those
    planes then contributed 3437 mm of "vessel" to a 170 um slab where the clean
    slabs of the same volume contribute 20-40 mm. The gate is decided on the
    reference, and both masks are emptied together.
    """
    ref = np.zeros((6, 40, 40), dtype=np.float32)
    ref[:, 18:21, 3:37] = 1.0                 # a tube: a few percent of each plane
    ref[4:] = 1.0                             # two planes are flooded solid
    pred = ref.copy()
    pred[:, 18:21, 3:37] = 0.0

    open_gate = segment_pair(pred, ref, method="intensity", max_fraction=None)
    closed = segment_pair(pred, ref, method="intensity", max_fraction=0.5)
    assert closed["n_planes_gated"] == 2
    # The gated planes exist in neither mask now, so they cannot be scored.
    assert closed["ref_mask"][4:].sum() == 0
    assert closed["pred_mask"][4:].sum() == 0
    # Everything below the gate is untouched, so the comparison is like-for-like.
    for z in range(4):
        assert np.array_equal(closed["ref_mask"][z], open_gate["ref_mask"][z])
    assert segment_pair(pred, ref, method="intensity",
                        max_fraction=0.2)["n_planes_gated"] == 2
    with pytest.raises(ValueError):
        segment_pair(pred, ref, method="intensity", max_fraction=0.0)


def test_group_statistics():
    values = [1.0, 2.0, 3.0, 10.0]
    groups = ["a", "a", "b", "c"]
    per = group_mean(values, groups)
    assert per == {"a": 1.5, "b": 3.0, "c": 10.0}
    mean, lo, hi = bootstrap_ci(list(per.values()))
    assert lo <= mean <= hi


def test_paired_delta_matches_on_key():
    a = [{"key": "x", "m": 2.0}, {"key": "y", "m": 3.0}]
    b = [{"key": "x", "m": 1.0}, {"key": "y", "m": 3.5}]
    d = paired_delta(a, b, "m")
    assert d["n"] == 2
    assert d["delta"] == pytest.approx(0.25, abs=1e-6)

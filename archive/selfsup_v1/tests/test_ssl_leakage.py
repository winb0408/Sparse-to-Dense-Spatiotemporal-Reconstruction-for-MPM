"""The anti-leakage contract of the self-supervised path.

These are the tests that decide whether "zero dense labels" is true. They fail
loudly if the holdout ever reaches the network through any channel.
"""

import pytest
import torch

from spur_mpm.data.observation import build_model_inputs
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.ssl.holdout import HoldoutSpec, split_holdout
from spur_mpm.ssl.leakage import (
    LeakageError,
    assert_holdout_is_measured,
    assert_inputs_blind,
    assert_split_disjoint,
)
from spur_mpm.ssl.mms import MMSLoss


SHAPE = (8, 32, 32)


def _acq_mask(ky: int = 2) -> torch.Tensor:
    return SamplingSpec(mode="y", ky=ky).mask(SHAPE)


@pytest.mark.parametrize("geometry", ["row", "plane", "scatter"])
def test_split_is_a_partition(geometry):
    mask = _acq_mask()
    m_in, m_hold = split_holdout(mask, HoldoutSpec(ratio=0.5, geometry=geometry, seed=0))
    assert torch.equal(m_in | m_hold, mask)          # covers exactly Omega_acq
    assert not bool((m_in & m_hold).any())           # disjoint
    assert m_in.any() and m_hold.any()


def test_row_geometry_withholds_whole_rows():
    mask = _acq_mask(ky=2)
    m_in, m_hold = split_holdout(mask, HoldoutSpec(ratio=0.5, geometry="row", seed=0))
    for held in torch.nonzero(m_hold.any(dim=(0, 2))).flatten():
        row = m_hold[:, held, :]
        assert bool(row.all()), "a held-out row must be withheld completely"


def test_ratio_extremes_are_rejected():
    mask = _acq_mask()
    with pytest.raises(ValueError):
        split_holdout(mask, HoldoutSpec(ratio=0.0))
    with pytest.raises(ValueError):
        split_holdout(mask, HoldoutSpec(ratio=1.0))


def test_too_few_structural_units_raises_rather_than_silently_degrading():
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[0, 0, :] = True                              # one measured row only
    with pytest.raises(ValueError, match="needs >=2 measured"):
        split_holdout(mask, HoldoutSpec(ratio=0.5, geometry="row"))


def test_guards_catch_the_classic_bugs():
    mask = _acq_mask()
    m_in, m_hold = split_holdout(mask, HoldoutSpec(ratio=0.5, geometry="row", seed=0))

    assert_split_disjoint(m_in, m_hold)
    with pytest.raises(LeakageError):
        assert_split_disjoint(mask, m_hold)           # overlapping on purpose

    assert_holdout_is_measured(m_hold, mask)
    unmeasured = ~mask
    with pytest.raises(LeakageError, match="never measured"):
        assert_holdout_is_measured(m_hold | unmeasured, mask)

    y_blind = torch.where(m_in, torch.rand(SHAPE), torch.zeros(SHAPE))
    assert_inputs_blind(y_blind, m_hold)
    with pytest.raises(LeakageError, match="held-out"):
        assert_inputs_blind(torch.rand(SHAPE), m_hold)


def test_network_inputs_are_invariant_to_heldout_values():
    """The decisive test: changing Y_hold must change nothing the model sees.

    The pipeline is: split first, rebuild Y_in / M_in from the reduced mask,
    then build X0 / C_s / U from those. If any of those stages peeked at the
    withheld measurement, the inputs would move when Y_hold moves.
    """
    psf = PSFConfig()
    mask = _acq_mask(ky=2)
    m_in, m_hold = split_holdout(mask, HoldoutSpec(ratio=0.5, geometry="row", seed=3))

    base = torch.rand(SHAPE)
    y_a = torch.where(m_in, base, torch.zeros(SHAPE))
    y_b = y_a.clone()
    y_b[m_hold] = (y_b[m_hold] + 5.0)                 # perturb only withheld values

    a = build_model_inputs(y_a, m_in, psf=psf)
    b = build_model_inputs(y_b, m_in, psf=psf)
    for field in ("y", "mask", "x0", "confidence", "uncertainty"):
        assert torch.allclose(getattr(a, field), getattr(b, field), atol=1e-7), \
            f"{field} depends on held-out values -> the holdout leaked"


def test_mms_targets_are_measurements_not_zeros():
    mask = _acq_mask(ky=2)
    m_in, m_hold = split_holdout(mask, HoldoutSpec(ratio=0.5, geometry="row", seed=1))
    y = torch.where(mask, torch.full(SHAPE, 0.7), torch.zeros(SHAPE))
    hold_target = torch.where(m_hold, y, torch.zeros(SHAPE))
    assert float(hold_target[m_hold].min()) == pytest.approx(0.7)


def test_mms_loss_ignores_unheld_positions():
    crit = MMSLoss(rho="l1")
    pred = torch.zeros(4, 4)
    target = torch.zeros(4, 4)
    hold = torch.zeros(4, 4, dtype=torch.bool)
    hold[0] = True
    target[0] = 1.0
    target[1:] = 99.0                                  # must be ignored
    res = crit(pred, target, hold)
    assert float(res["loss"]) == pytest.approx(1.0)


# --- window configuration -------------------------------------------------

def test_plane_skipping_requires_a_window_deep_enough_to_see_a_plane():
    from spur_mpm.data.datasets import required_context_depth, validate_context_depth

    spec = SamplingSpec(mode="z", kz=2, ky=2)
    assert required_context_depth(2) == 3
    with pytest.raises(ValueError, match="skips planes"):
        validate_context_depth(spec, context_depth=1)
    validate_context_depth(spec, context_depth=3)      # one plane each side

    kz4 = SamplingSpec(mode="yz", kz=4, ky=2)
    assert required_context_depth(4) == 7
    with pytest.raises(ValueError, match="at least 7"):
        validate_context_depth(kz4, context_depth=3)
    validate_context_depth(kz4, context_depth=7)


def test_in_plane_sampling_needs_no_axial_context():
    from spur_mpm.data.datasets import validate_context_depth

    validate_context_depth(SamplingSpec(mode="y", ky=2), context_depth=1)
    validate_context_depth(SamplingSpec(mode="xy", ky=2), context_depth=1)


def test_data_consistency_term_can_actually_be_reached():
    """dc_weight used to be a config knob that raised on the first step.

    The training criterion was called without y_in/m_in, so any nonzero
    dc_weight died with 'requires y_in and m_in' -- and nothing tested it.
    """
    pred = torch.zeros(2, 8, 8)
    target = torch.full((2, 8, 8), 0.5)
    hold = torch.zeros(2, 8, 8)
    hold[:, ::2] = 1.0
    y_in = torch.full((2, 8, 8), 0.5)
    m_in = 1.0 - hold

    without = MMSLoss(rho="charbonnier", dc_weight=0.0)(pred, target, hold)["loss"]
    crit = MMSLoss(rho="charbonnier", dc_weight=1.0)
    res = crit(pred, target, hold, y_in=y_in, m_in=m_in)
    # pred is wrong everywhere; DC must add a penalty for the measured half
    assert res["loss"] > without
    assert "dc" in res and float(res["dc"]) > 0


def test_item_exposes_the_input_support_for_consistency():
    """A DC term cannot be computed if the item hides M_in and y_in."""
    import numpy as np

    from spur_mpm.data.datasets import _make_item
    from spur_mpm.data.geometry import VolumeGeometry
    from spur_mpm.data.datasets import ItemConfig

    dense = np.random.default_rng(0).random((5, 16, 16)).astype(np.float32)
    geom = VolumeGeometry(sample_id="S", mouse_id="1", session_id="1",
                          fov_id="I", shape=dense.shape, axes="ZYX",
                          voxel_xy_um=0.98, z_step_um=5.0)
    cfg = ItemConfig(sampling=SamplingSpec(mode="y", ky=2),
                     features=("y", "mask"),
                     supervision="self",
                     holdout=HoldoutSpec(ratio=0.5, geometry="row", seed=0))
    item = _make_item(dense, geom, cfg, z0=0, y0=0, x0=0, holdout_seed=0)
    assert "y_in" in item and "in_mask" in item
    m_in, m_hold = item["in_mask"].bool(), item["hold_mask"].bool()
    assert not bool((m_in & m_hold).any())
    assert torch.equal(item["y_in"][~m_in], torch.zeros_like(item["y_in"][~m_in]))

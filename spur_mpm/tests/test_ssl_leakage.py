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

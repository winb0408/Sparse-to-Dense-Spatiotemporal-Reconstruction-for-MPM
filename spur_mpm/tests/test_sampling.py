"""Sampling masks: geometry, fractions, phase alignment."""

import pytest
import torch

from spur_mpm.physics.sampling import SamplingSpec


SHAPE = (16, 64, 64)


def test_full_mask_is_everything():
    m = SamplingSpec(mode="full").mask(SHAPE)
    assert m.all() and m.dtype == torch.bool


def test_y_mode_keeps_complete_rows():
    m = SamplingSpec(mode="y", ky=4).mask(SHAPE)
    assert m.shape == SHAPE
    # every kept row is measured across the full X extent
    rows = m[0].any(dim=1)
    assert torch.equal(m[0][rows], torch.ones_like(m[0][rows]))
    # and X is never partially sampled on a kept row
    assert float(m.float().mean()) == pytest.approx(0.25, abs=0.02)


@pytest.mark.parametrize("mode,kwargs,expected", [
    ("y", {"ky": 2}, 0.5),
    ("y", {"ky": 4}, 0.25),
    ("z", {"kz": 2}, 0.5),
    ("z", {"kz": 4}, 0.25),
    ("yz", {"ky": 2, "kz": 2}, 0.25),
    ("yz", {"ky": 2, "kz": 4}, 0.125),
])
def test_ideal_fraction_matches_realised(mode, kwargs, expected):
    spec = SamplingSpec(mode=mode, **kwargs)
    realised = float(spec.mask(SHAPE).float().mean())
    assert realised == pytest.approx(expected, abs=0.02)
    assert spec.ideal_fraction() == pytest.approx(expected, abs=1e-9)


def test_phase_keeps_lattice_stack_aligned():
    """Two adjacent patches must sit on one global lattice, not two lattices."""
    spec = SamplingSpec(mode="y", ky=4)
    whole = spec.mask((8, 32, 32))
    # a window starting at row 1 with the matching phase reproduces the slice
    for y0 in range(4):
        sub = spec.mask((8, 8, 32), phase=(0, y0 % 4, 0))
        assert torch.equal(sub, whole[:, y0:y0 + 8])


def test_ky_kz_validation():
    with pytest.raises(ValueError):
        SamplingSpec(mode="y", ky=0)
    with pytest.raises(ValueError):
        SamplingSpec(mode="nope")


def test_xy_grid_and_random():
    grid = SamplingSpec(mode="xy", ky=2, pattern="grid").mask(SHAPE)
    rnd = SamplingSpec(mode="xy", ky=2, pattern="random", seed=0).mask(SHAPE)
    assert not torch.equal(grid, rnd)
    assert SamplingSpec(mode="xy", ky=2, pattern="grid").ideal_fraction() == pytest.approx(0.75)

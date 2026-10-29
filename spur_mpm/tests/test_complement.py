"""The complementary split and the dual arm built on it.

The property that makes E012 worth running is exactly the one that is easy to get
wrong: the two halves must be disjoint (or a branch would be scored on a row it
was handed) and must together be the acquisition (or the pair would quietly use
less than 100% of the data, which is the deficit the arm exists to remove).
"""

from __future__ import annotations

import torch

from spur_mpm.data.datasets import ItemConfig
from spur_mpm.models.complement import ComplementaryDual
from spur_mpm.models.registry import build_model, count_parameters, get_arm
from spur_mpm.ssl.holdout import HoldoutSpec
from spur_mpm.ssl.complement import split_complement


def _row_mask(z: int = 4, h: int = 16, w: int = 8, ky: int = 2) -> torch.Tensor:
    m = torch.zeros(z, h, w, dtype=torch.bool)
    m[:, ::ky, :] = True
    return m


def test_complement_is_disjoint_and_complete():
    mask = _row_mask()
    spec = HoldoutSpec(ratio=0.5, geometry="row", seed=3)
    a, b = split_complement(mask, spec)
    assert not bool((a & b).any()), "halves overlap"
    assert torch.equal(a | b, mask), "halves do not cover the acquisition"
    # even split: 8 measured rows of 16 -> 4 + 4
    assert int(a[:, :, 0].any(0).sum()) == int(b[:, :, 0].any(0).sum()) == 4
    # row geometry means whole rows, never lone voxels
    assert torch.equal(a, a.any(dim=2, keepdim=True).expand_as(a))


def test_complement_never_returns_an_empty_half():
    # two measured rows only: ratio 0.5 would round to 1, and both halves must
    # stay non-empty rather than letting one branch train on nothing
    mask = torch.zeros(1, 4, 2, dtype=torch.bool)
    mask[:, ::3, :] = True                      # rows 0 and 3
    a, b = split_complement(mask, HoldoutSpec(ratio=0.99, geometry="row", seed=0))
    assert bool(a.any()) and bool(b.any())


def test_complement_rejects_an_empty_mask():
    for geom in ("row", "plane"):
        try:
            split_complement(torch.zeros(2, 4, 4, dtype=torch.bool),
                             HoldoutSpec(ratio=0.5, geometry=geom, seed=0))
        except ValueError:
            continue
        raise AssertionError(f"empty mask accepted for geometry {geom}")


def test_dual_arm_carries_two_backbones_and_splits_its_input():
    arm = get_arm("SS1c")
    assert arm.dual
    model = build_model(arm, context_depth=1, base=8, window=4,
                        depths=(1, 1, 1), heads=(2, 2, 2))
    assert isinstance(model, ComplementaryDual)
    single = build_model(get_arm("SS1"), context_depth=1, base=8, window=4,
                         depths=(1, 1, 1), heads=(2, 2, 2))
    assert count_parameters(model) == 2 * count_parameters(single)

    x = torch.zeros(2, 2 * len(arm.features), 8, 8)
    ya, yb = model.forward_both(x)
    assert ya.shape == yb.shape == (2, 1, 8, 8)
    assert torch.allclose(model(x), 0.5 * (ya + yb))


def test_dual_item_config_demands_a_split_spec():
    from spur_mpm.data.datasets import _make_item
    from spur_mpm.data.geometry import VolumeGeometry

    dense = torch.rand(3, 8, 8).numpy()
    geom = VolumeGeometry(sample_id="S", mouse_id="1", session_id="1",
                          fov_id="I", shape=dense.shape, axes="ZYX")
    cfg = ItemConfig(dual=True, holdout=None)
    try:
        _make_item(dense, geom, cfg, 0, 0, 0, 0)
    except ValueError:
        return
    raise AssertionError("dual item built without a HoldoutSpec")

"""Tests for the Fig. 5 renderer's non-plotting halves.

The figure itself is checked by eye; what has to be checked mechanically is the
part that can lie. A volume-mean clDice hides *where* a reconstruction fails,
and two opposite failures -- lost branches and invented connections -- push the
mean the same way. So the break and false maps must be non-empty exactly at the
defect and empty when there is none, an empty pair must not quietly score 1.0,
and the bifurcation crop must not move between renders of the same volume.

Everything here runs on tiny synthetic tubes: no TIFF, no summary, no matplotlib.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spur_mpm.scripts.structure_figure import (
    TOLERANCE_VOX,
    break_false_stats,
    break_map,
    choose_crop,
    false_map,
    load_or_make_crop,
    mip,
    prepare_panels,
    scored_z_range,
)


def tube(shape=(21, 64, 64), gap=None, bridge=False):
    """A straight thick tube along x, plus a short vertical stub that makes a
    junction, optionally cut by a gap in x or joined by a spurious bridge."""
    m = np.zeros(shape, bool)
    m[9:12, 20:23, 5:60] = True        # main vessel
    m[9:12, 40:43, 5:60] = True        # crossing vessel -> real junction
    if bridge:
        m[9:12, 22:41, 30:33] = True   # connection that does not exist in ref
    if gap:
        m[:, :, gap[0]:gap[1]] = False
    return m


@pytest.fixture(scope="module")
def skels():
    from spur_mpm.metrics.skeleton import skeleton

    ref = tube()
    return skeleton(ref), skeleton(tube(bridge=True))


# --------------------------------------------------------------------------
# break / false maps sit exactly on the defect
# --------------------------------------------------------------------------

def test_identical_masks_give_empty_break_and_false_maps(skels):
    ref_skel, _ = skels
    b = break_map(ref_skel, ref_skel)
    f = false_map(ref_skel, ref_skel)
    assert b.shape == ref_skel.shape
    assert b.dtype == bool and f.dtype == bool
    assert not b.any()
    assert not f.any()


def test_deliberate_gap_shows_up_as_breaks_only_there(skels):
    from spur_mpm.metrics.skeleton import skeleton

    ref_skel, _ = skels
    ref = tube()
    broken = tube(gap=(30, 35))
    pred_skel = skeleton(broken)

    b = break_map(ref_skel, pred_skel)
    f = false_map(pred_skel, ref_skel)

    assert b.any(), "a gap in the prediction must register as lost structure"
    # every break voxel lies in the cut slab (plus one voxel of edge effect)
    assert b[:, :, 29:36].sum() == b.sum()
    # the prediction invents nothing, so the false map must stay empty
    assert not f.any()


def test_spurious_bridge_shows_up_as_false_connection_only_there(skels):
    ref_skel, pred_skel = skels
    b = break_map(ref_skel, pred_skel)
    f = false_map(pred_skel, ref_skel)

    assert f.any(), "a bridge absent from the reference must register as invented"
    zs, ys, xs = np.where(f)
    # the bridge runs along y at x ~= 31, between the two crossing vessels
    assert xs.min() >= 30 and xs.max() <= 32
    assert ys.min() >= 21 and ys.max() <= 42
    # it is invented structure, not lost structure
    assert not b.any()


def test_empty_pair_does_not_score_one_silently():
    empty = np.zeros((8, 16, 16), bool)
    assert not break_map(empty, empty).any()
    assert not false_map(empty, empty).any()
    stats = break_false_stats(empty, empty)
    # NaN, not 1.0: nothing was measured, so nothing may look perfect
    assert np.isnan(stats["break_fraction"])
    assert np.isnan(stats["false_fraction"])
    assert stats["break_voxels"] == 0
    assert stats["false_voxels"] == 0


def test_empty_prediction_loses_whole_reference_skeleton(skels):
    ref_skel, _ = skels
    empty = np.zeros_like(ref_skel)
    b = break_map(ref_skel, empty)
    assert b.sum() == ref_skel.sum()
    stats = break_false_stats(ref_skel, empty)
    assert stats["break_fraction"] == pytest.approx(1.0)
    assert np.isnan(stats["false_fraction"])


def test_tolerance_is_honoured():
    a = np.zeros((5, 9, 9), bool)
    b = np.zeros((5, 9, 9), bool)
    a[2, 4, 2] = True
    b[2, 4, 6] = True
    assert break_map(a, b, tolerance_vox=3).any()
    assert not break_map(a, b, tolerance_vox=4).any()


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def test_scored_z_range_and_mip_use_nonempty_planes():
    ref = tube()
    pred = tube(gap=(40, 50))
    z0, z1 = scored_z_range(ref, pred)
    assert (z0, z1) == (9, 12)
    m = mip(pred, (z0, z1))
    assert m.shape == ref.shape[1:]
    assert m.any()
    assert not mip(pred, (0, 0)).any()


def test_scored_z_range_empty_volume_is_degenerate():
    empty = np.zeros((6, 8, 8), bool)
    assert scored_z_range(empty, empty) == (0, 0)


# --------------------------------------------------------------------------
# the frozen crop
# --------------------------------------------------------------------------

def test_choose_crop_is_deterministic_and_inside_bounds(skels):
    from spur_mpm.metrics.skeleton import skeleton

    ref = tube()
    ref_skel = skeleton(ref)
    a = choose_crop(ref, ref_skel, size=16, slab=5)
    b = choose_crop(ref, ref_skel, size=16, slab=5)
    assert (a["z0"], a["y0"], a["x0"]) == (b["z0"], b["y0"], b["x0"])
    z, h, w = ref.shape
    for start, span, extent in (("z0", "nz", z), ("y0", "ny", h),
                                ("x0", "nx", w)):
        assert a[start] >= 0
        assert a[start] + a[span] <= extent


def test_frozen_crop_is_reused_when_the_mask_changes(tmp_path, skels):
    from spur_mpm.metrics.skeleton import skeleton

    path = tmp_path / "crop.json"
    ref = tube()
    first = load_or_make_crop(path, ref, skeleton(ref), size=16, slab=5,
                              sample_id="volA")
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["sample_id"] == "volA"

    # a different mask, same volume shape: the crop must not move
    other = tube(bridge=True)
    other[2:5, 5:15, 5:15] = True      # new foreground far from the first crop
    second = load_or_make_crop(path, other, skeleton(other), size=16, slab=5,
                               sample_id="volA")
    assert (first["z0"], first["y0"], first["x0"]) == \
           (second["z0"], second["y0"], second["x0"])
    assert json.loads(path.read_text())["z0"] == first["z0"]


def test_crop_that_does_not_fit_its_volume_is_rejected(tmp_path, skels):
    from spur_mpm.metrics.skeleton import skeleton

    path = tmp_path / "crop.json"
    ref = tube()
    load_or_make_crop(path, ref, skeleton(ref), size=16, slab=5)
    too_small = np.zeros((4, 8, 8), bool)
    with pytest.raises(ValueError):
        load_or_make_crop(path, too_small, skeleton(too_small), size=16, slab=5)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def test_prepare_panels_assembles_every_panel(skels):
    from spur_mpm.metrics.skeleton import skeleton

    from spur_mpm.metrics.skeleton import skeleton

    ref = tube()
    # one defect of each kind, so both maps have something to show
    pred = tube(bridge=True, gap=(50, 55))
    ref_skel = skeleton(ref)
    pred_skel = skeleton(pred)
    crop = choose_crop(ref, ref_skel, size=16, slab=5)
    panels = prepare_panels(ref, pred, ref_skel, pred_skel, crop)

    assert panels["z_range"] == (9, 12)
    for key in ("ref_mip", "pred_mip", "break_mip", "false_mip",
                "ref_crop_mip", "pred_crop_mip"):
        assert panels[key].ndim == 2
    # the crop pair is the same window in both volumes
    assert panels["ref_crop_mip"].shape == panels["pred_crop_mip"].shape
    assert panels["break_mip"].any()
    assert panels["false_mip"].any()
    assert 0.0 < panels["stats"]["break_fraction"] < 1.0
    assert 0.0 < panels["stats"]["false_fraction"] < 1.0


def test_prepare_panels_rejects_mismatched_shapes(skels):
    ref_skel, pred_skel = skels
    ref = tube()
    pred = tube()
    crop = choose_crop(ref, ref_skel, size=16, slab=5)
    with pytest.raises(ValueError):
        prepare_panels(ref, pred[:-1], ref_skel, pred_skel, crop)


def test_default_tolerance_is_one_voxel():
    assert TOLERANCE_VOX == 1.0

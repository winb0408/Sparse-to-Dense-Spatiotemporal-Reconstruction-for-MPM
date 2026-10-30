"""Skeleton metrics must be exact where they claim exactness and honest where
the input carries no information.

Two kinds of mistake are expensive here. An *inexact* length (counting voxels
instead of chain length) quietly inflates every diagonal vessel, so a model that
draws smooth tubes looks better than one that draws axis-aligned ones. An
*dishonest* degenerate case (0.0 or 1.0 for "nothing was segmented") silently
pulls a group mean towards "no change" and hides a plane where the prediction
recovered no vessel at all. Both are pinned down below with synthetic masks
small enough to run in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spur_mpm.metrics.skeleton import (
    analyse_volume,
    depth_profile,
    export_imagej_mask,
    skeleton,
    skeleton_length_density,
    skeleton_length_vox,
    skeleton_overlap,
    skeleton_stats,
)
from spur_mpm.metrics.vessel import otsu_threshold, vessel_mask

VOXEL = (5.0, 0.98, 0.98)


def _line(n: int = 10, axis: int = 2) -> np.ndarray:
    """A one-voxel-wide run of ``n`` voxels along ``axis`` in a 3-D volume."""
    shape = [1, 1, 1]
    shape[axis] = n
    return np.ones(tuple(shape), dtype=bool)


def _y_shape() -> np.ndarray:
    """A thin 2-D Y: one stem plus two diagonal arms out of a single voxel."""
    mask = np.zeros((21, 21), dtype=bool)
    for step in range(7):
        mask[10 - step, 10] = True
        mask[10 + step, 10 - step] = True
        mask[10 + step, 10 + step] = True
    return mask


def _vessel_image(shape=(6, 20, 24), *, gap=None, extra_tube: bool = False) -> np.ndarray:
    """Synthetic stack: bright tubes on a noisy background (no real data)."""
    rng = np.random.default_rng(0)
    img = 0.02 * rng.random(shape).astype(np.float32)
    img[:, 5:8, 2:22] = 1.0              # tube along x
    img[:, 14:17, 2:22] = 1.0            # second tube, parallel
    img[:, 9:12, 10:13] = 1.0            # tube along z, joining both
    if extra_tube:
        img[:, 2:4, 2:22] = 1.0
    if gap is not None:
        img[:, :, gap] = 0.0
    return img


def _masks(gap=None, extra_tube: bool = False):
    ref_img = _vessel_image(gap=gap, extra_tube=extra_tube)
    return vessel_mask(ref_img, threshold=0.5), ref_img


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_straight_line_length_is_exactly_n_minus_one(axis):
    line = _line(10, axis)
    assert int(skeleton(line).sum()) == 10
    # Exact equality, not approx: this is the property that separates chain
    # length from voxel count and it must hold bit-for-bit.
    assert skeleton_length_vox(line) == 9.0


def test_diagonal_steps_count_their_true_length():
    diagonal = np.zeros((1, 12, 12), dtype=bool)
    diagonal[0, np.arange(10), np.arange(10)] = True
    assert skeleton_length_vox(diagonal) == pytest.approx(9 * np.sqrt(2))

    corner = np.zeros((12, 12, 12), dtype=bool)
    corner[np.arange(10), np.arange(10), np.arange(10)] = True
    assert skeleton_length_vox(corner) == pytest.approx(9 * np.sqrt(3))


def test_y_shape_has_one_junction_three_endpoints_three_branches():
    stats = skeleton_stats(_y_shape(), voxel_um=VOXEL)
    assert stats["n_components"] == 1
    assert stats["n_junctions"] == 1
    assert stats["n_endpoints"] == 3
    assert stats["n_branches"] == 3
    # Two diagonal arms of five steps each, one straight arm, plus the three
    # steps that meet at the junction (one straight, two diagonal).
    assert stats["total_length_vox"] == pytest.approx(6.0 + 12 * np.sqrt(2))
    assert stats["max_branch_length_vox"] == pytest.approx(5 * np.sqrt(2))
    assert stats["mean_radius_vox"] == pytest.approx(1.0)    # one voxel half-width
    assert stats["mean_radius_um"] == pytest.approx(0.98)    # in-plane spacing


def test_two_disjoint_vessels_give_two_components():
    two = np.zeros((1, 10, 20), dtype=bool)
    two[0, 2, 1:8] = True
    two[0, 7, 12:19] = True
    stats = skeleton_stats(two)
    assert stats["n_components"] == 2
    assert stats["n_junctions"] == 0
    assert stats["n_endpoints"] == 4
    assert stats["n_branches"] == 2
    assert stats["total_length_vox"] == pytest.approx(12.0)
    assert stats["mean_branch_length_vox"] == pytest.approx(6.0)
    assert stats["median_branch_length_vox"] == pytest.approx(6.0)
    assert stats["bounding_box_fraction"] == pytest.approx(14 / (6 * 18))


def test_empty_mask_returns_documented_degenerate_values():
    empty = np.zeros((3, 5, 6), dtype=bool)
    stats = skeleton_stats(empty, voxel_um=VOXEL)
    assert stats["n_voxels"] == 0
    assert stats["n_skeleton_voxels"] == 0
    assert stats["n_components"] == 0
    assert stats["n_branches"] == 0
    for key in ("total_length_vox", "mean_branch_length_vox",
                "median_branch_length_vox", "max_branch_length_vox",
                "mean_radius_vox", "bounding_box_fraction", "total_length_um"):
        assert stats[key] == 0.0, key
    assert int(skeleton(empty).sum()) == 0
    assert skeleton_length_vox(empty) == 0.0
    assert skeleton_length_density(empty, VOXEL) == 0.0
    assert skeleton_overlap(empty, empty) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    # Without a calibration the physical values are unknown, not 1 um.
    assert np.isnan(skeleton_stats(empty)["mean_radius_um"])
    assert np.isnan(skeleton_stats(empty)["total_length_um"])


def test_length_density_uses_the_bounding_box_of_the_mask():
    line = _line(10, axis=2)                                  # 1 x 1 x 10 voxels
    # 9 voxel steps of 0.98 um = 8.82 um = 0.00882 mm, in a box of
    # 5.0 um x 0.98 um x 9.8 um = 48.02 um^3 = 4.802e-8 mm^3.
    expected = (9 * 0.98 / 1000.0) / (5.0 * 0.98 * 9.8 / 1e9)
    assert skeleton_length_density(line, VOXEL) == pytest.approx(expected)


def test_skeleton_overlap_with_itself_is_exactly_one():
    mask = _y_shape()
    assert skeleton_overlap(mask, mask) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert skeleton_overlap(mask, mask, tolerance_vox=0)["f1"] == 1.0


def test_shifted_prediction_needs_a_tolerance():
    ref = np.zeros((1, 20, 30), dtype=bool)
    ref[0, 10, 5:25] = True
    pred = np.zeros_like(ref)
    pred[0, 11, 5:25] = True                     # same vessel, one voxel off
    strict = skeleton_overlap(pred, ref, tolerance_vox=0)
    assert strict["f1"] < 1.0
    assert strict["precision"] == 0.0 and strict["recall"] == 0.0
    loose = skeleton_overlap(pred, ref, tolerance_vox=1)
    assert loose["precision"] == 1.0
    assert loose["recall"] == 1.0
    assert loose["f1"] == 1.0


def test_export_imagej_mask_round_trips_through_tifffile(tmp_path):
    mask = np.zeros((4, 6, 7), dtype=bool)
    mask[1:3, 2:5, 1:6] = True
    mask[0, 0, 0] = True
    path = export_imagej_mask(mask, tmp_path / "mask.tif")
    assert path == str(tmp_path / "mask.tif")

    back = tifffile.imread(path)
    assert back.shape == mask.shape
    assert set(np.unique(back).tolist()) == {0, 255}
    assert np.array_equal(back > 0, mask)
    with tifffile.TiffFile(path) as tif:
        assert tif.is_imagej and len(tif.pages) == mask.shape[0]


def test_export_imagej_mask_rejects_zip_targets(tmp_path):
    with pytest.raises(ValueError):
        export_imagej_mask(np.ones((2, 2, 2), dtype=bool), tmp_path / "mask.zip")


def test_analyse_volume_merges_the_three_layers_on_one_threshold():
    ref_mask, ref_img = _masks()
    pred_mask, _ = _masks(gap=12)                # prediction breaks both tubes
    # These synthetics hand over already-binarised masks, so they exercise the
    # ``intensity`` path: the vesselness default is about how a *raw* stack is
    # segmented and is covered separately below.
    out = analyse_volume(pred_mask.astype(np.float32), ref_img, voxel_um=VOXEL,
                         n_slabs=3, tolerance_vox=1, segment="intensity")
    for key in ("dice", "cl_dice", "pred_fraction", "ref_fraction",
                "n_components_pred", "n_components_ref", "precision", "recall", "f1",
                "length_density_ratio", "depth_profile"):
        assert key in out, key
    for prefix in ("pred_", "ref_"):
        for key in skeleton_stats(ref_mask).keys():
            assert f"{prefix}{key}" in out, f"{prefix}{key}"
    assert out["dice"] < 1.0
    assert out["n_components_pred"] > out["n_components_ref"]
    # A cut splits a tube but does not shorten it by much, so total length is the
    # wrong place to look for the damage -- the component count is the signature.
    # Length is asserted as "not longer": inventing length is the failure that
    # matters, and it is the one a thick, over-segmenting prediction produces.
    assert out["pred_total_length_vox"] <= out["ref_total_length_vox"]
    assert len(out["depth_profile"]) == 3


def test_analyse_volume_fits_the_threshold_on_the_reference_only():
    ref_mask, ref_img = _masks()
    good = np.asarray(ref_img, dtype=np.float32)
    broken = good.copy()
    broken[:, :, 12:14] = 0.0
    a = analyse_volume(good, ref_img, voxel_um=VOXEL, n_slabs=2, segment="intensity")
    b = analyse_volume(broken, ref_img, voxel_um=VOXEL, n_slabs=2, segment="intensity")
    # Default is per plane, and the reported number is the median of those, which
    # on a brightness-flat synthetic is within a rounding step of the global Otsu.
    assert a["threshold_mode"] == "per_plane"
    assert a["threshold"] != pytest.approx(otsu_threshold(ref_img) + 1.0)
    assert a["threshold"] == pytest.approx(otsu_threshold(ref_img), abs=5e-3)
    assert b["threshold"] == a["threshold"]      # prediction cannot move the operating point
    assert a["dice"] == 1.0 and a["length_density_ratio"] == pytest.approx(1.0)
    assert b["dice"] < 1.0
    # Not 1.0-and-not-longer: a gap splits a vessel without removing much of its
    # length, so the ratio may legitimately be 1.0. What must never happen is a
    # prediction scoring *more* length than the reference.
    assert b["length_density_ratio"] <= 1.0 + 1e-6
    # The global mode must reproduce the classic recipe exactly.
    g = analyse_volume(good, ref_img, voxel_um=VOXEL, n_slabs=2,
                       threshold_mode="global", segment="intensity")
    assert g["threshold"] == pytest.approx(otsu_threshold(ref_img))
    # An explicit threshold must reproduce the mode it stands in for exactly --
    # here the global one, since a single number cannot stand for a per-plane set.
    c = analyse_volume(good, ref_img, threshold=g["threshold"], voxel_um=VOXEL,
                       n_slabs=2, segment="intensity")
    assert c["dice"] == g["dice"]
    assert c["ref_total_length_vox"] == g["ref_total_length_vox"]


def test_vesselness_segmentation_marks_the_tube_not_the_background():
    """The default segmenter must find tubules, not "the bright half".

    Measured on real Dryad planes, an intensity Otsu marks 37-55 % of the plane
    as vessel and the resulting 3-D skeleton is 95 % junction voxels -- a surface,
    not centrelines. The Sato response has to do better on a synthetic with the
    same failure mode: a bright tube on a background that is itself bright and
    textured.
    """
    rng = np.random.default_rng(0)
    ref = rng.normal(0.45, 0.05, (24, 64, 64)).astype(np.float32)
    ref[:, 30:33, :] = 1.0                       # a 3-voxel tube through the stack
    ref[:, :, 50:52] = 1.0                       # and one across it
    pred = ref.copy()
    pred[:, 30:33, :] = 0.4                      # the prediction loses one tube

    out = analyse_volume(pred, ref, voxel_um=VOXEL, n_slabs=2)
    assert out["segment_method"] == "vesselness"
    # A tube of 3 x 64 voxels plus 24 x 2 x 64 is a few percent of the volume,
    # nowhere near the third an intensity Otsu would take.
    assert 0.01 < out["ref_fraction"] < 0.10
    assert out["dice"] < 0.9                     # losing 3 of 5 columns must show
    assert out["threshold"] == pytest.approx(
        analyse_volume(ref, ref, voxel_um=VOXEL, n_slabs=2)["threshold"])


def test_analyse_volume_reports_nan_ratio_when_a_mask_is_empty():
    _, ref_img = _masks()
    blank = np.zeros_like(ref_img)
    out = analyse_volume(blank, ref_img, threshold=0.5, voxel_um=VOXEL, n_slabs=2,
                         segment="intensity")
    assert out["pred_total_length_vox"] == 0.0
    assert out["pred_fraction"] == 0.0
    assert np.isnan(out["length_density_ratio"])
    assert out["f1"] == 0.0


def test_depth_profile_slabs_tile_the_stack_and_carry_physical_bounds():
    ref_mask, ref_img = _masks()                 # 6 planes, 5 um apart
    profile = depth_profile(ref_mask, ref_mask, VOXEL, n_slabs=2)
    assert [slab["z_lo_um"] for slab in profile] == [0.0, 15.0]
    assert [slab["z_hi_um"] for slab in profile] == [15.0, 30.0]
    top_level = analyse_volume(ref_img, ref_img, voxel_um=VOXEL, n_slabs=2,
                               segment="intensity")
    # ``threshold_mode`` / ``threshold_p10_p90`` / ``segment_method`` describe
    # the whole-volume choice, and ``threshold`` is the one operating point
    # fitted on the reference, so they belong to the top level and not to a slab.
    expected = (set(top_level)
                - {"depth_profile", "threshold_mode", "threshold_p10_p90",
                   "segment_method", "threshold"}
                | {"z_lo_um", "z_hi_um"})
    assert set(profile[0]) == expected


def test_single_plane_is_one_slab_of_one_axial_sample():
    plane = np.zeros((40, 40), dtype=np.float32)
    plane[18:21, 3:37] = 1.0
    out = analyse_volume(plane, plane, threshold=0.5, voxel_um=VOXEL,
                         segment="intensity")
    assert len(out["depth_profile"]) == 1
    assert out["depth_profile"][0]["z_lo_um"] == 0.0
    assert out["depth_profile"][0]["z_hi_um"] == VOXEL[0]
    # The box is the 3 x 34 voxel mask bbox extended by one plane of axial
    # sampling (5 um), so the length per mm^3 is reported in mm^-2.
    length_mm = skeleton_length_vox(vessel_mask(plane, threshold=0.5)) * 0.98 / 1000.0
    box_mm3 = 5.0 * (3 * 0.98) * (34 * 0.98) / 1e9
    assert out["ref_length_density"] == pytest.approx(length_mm / box_mm3)


def test_depth_profile_never_returns_empty_slabs():
    mask = np.zeros((2, 6, 6), dtype=bool)
    mask[:, 2:4, 2:4] = True
    profile = depth_profile(mask, mask, VOXEL, n_slabs=6)
    assert len(profile) == 2                     # clipped to the number of planes
    assert [slab["z_hi_um"] - slab["z_lo_um"] for slab in profile] == [5.0, 5.0]
    with pytest.raises(ValueError):
        depth_profile(mask, mask, VOXEL, n_slabs=0)


def test_branch_lengths_do_not_re_thin_and_stay_exact():
    """A regression for the cost fix, pinned by the numbers it must not change.

    ``_branch_lengths`` used to rescan the whole array once per branch *and*
    re-thin each segment; a 200-plane stack took over half an hour. It now takes
    each segment's bounding box from one ``find_objects`` pass and measures the
    already-thin segment directly. The lengths must be identical, so this test
    fixes them with a shape whose answer is worked out by hand: a symmetric
    cross with 21-voxel arms.

    Why four arms of 9 voxels and not 10: ``_neighbour_counts`` uses the full
    8-neighbourhood, and next to the centre of a cross the diagonal neighbours
    are skeleton voxels too, so **five** voxels count as junctions, not one.
    Each arm is therefore the 9 voxels beyond that blob, and a 9-voxel straight
    run has chain length 8 (a run of N voxels has N-1 steps).
    """
    from spur_mpm.metrics.skeleton import _branch_lengths, _neighbour_counts, skeleton

    thin = np.zeros((21, 21), dtype=bool)
    thin[10, 0:21] = True
    thin[0:21, 10] = True
    junctions = thin & (_neighbour_counts(thin) >= 3)
    assert junctions.sum() == 5
    assert sorted(_branch_lengths(thin, junctions)) == pytest.approx([8.0] * 4)

    # A diagonal line has no junction, so it is one branch, and the diagonal
    # step costs sqrt(2) rather than 1: 39 steps, 40 voxels.
    line = skeleton(np.eye(40, dtype=bool))
    assert _branch_lengths(line, np.zeros_like(line)) == pytest.approx(
        [39 * np.sqrt(2)])

"""Skeleton tolerance: one voxel is 0.98 um across the plane and 5 um in depth.

The published structure tables score skeleton overlap at ``tolerance_vox=1``.
Read back as a physical distance that is a tenfold range, so the sweep has to
report micrometres computed with the real axial pitch, and it has to refuse to
report them when the pitch is unknown rather than quietly counting voxels.
"""
import json

import numpy as np
import pytest
import tifffile

from spur_mpm.scripts.tolerance_sweep import (
    DEFAULT_TOLERANCES_UM,
    _pairs,
    _sampling_for,
    main,
    matched_fraction,
    overlap_at,
    render,
)


def _dot(idx, shape=(9, 5, 5)):
    """One skeleton voxel, so a shift is not hidden by overlap with the rest."""
    m = np.zeros(shape, dtype=bool)
    m[idx] = True
    return m


def test_an_axial_shift_is_five_micrometres_not_one():
    """The same one-voxel error passes a 5 um tolerance and fails a 1 um one."""
    a, b = _dot((4, 2, 2)), _dot((5, 2, 2))        # one voxel apart along Z
    spacing = (5.0, 0.98, 0.98)
    assert matched_fraction(b, a, 1.0, spacing) == pytest.approx(0.0)
    assert matched_fraction(b, a, 5.0, spacing) == pytest.approx(1.0)


def test_a_lateral_shift_of_one_voxel_is_under_a_micrometre():
    a, b = _dot((4, 2, 2)), _dot((4, 3, 2))        # one voxel apart across Y
    spacing = (5.0, 0.98, 0.98)
    assert matched_fraction(b, a, 1.0, spacing) == pytest.approx(1.0)


def test_without_spacing_the_tolerance_is_in_voxels():
    """The old table's meaning has to stay computable, or nothing can be compared."""
    a, b = _dot((4, 2, 2)), _dot((5, 2, 2))
    assert matched_fraction(b, a, 1.0, None) == pytest.approx(1.0)
    assert matched_fraction(b, a, 0.0, None) == pytest.approx(0.0)


def test_two_empty_skeletons_agree_and_one_empty_does_not():
    empty = np.zeros((4, 4, 4), dtype=bool)
    full = np.zeros((4, 4, 4), dtype=bool)
    full[1, 1, 1] = True
    assert matched_fraction(empty, full, 1.0, None) == 0.0
    assert matched_fraction(full, empty, 1.0, None) == 0.0
    assert overlap_at(empty, empty, 1.0)["f1"] == 1.0


def test_overlap_reports_precision_recall_and_f1_consistently():
    a, b = _dot((4, 2, 2)), _dot((4, 2, 2))
    got = overlap_at(a, b, 1.0, (5.0, 0.98, 0.98))
    assert got == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_spacing_is_read_from_the_recon_json_not_assumed(tmp_path):
    recon = tmp_path / "x_recon.json"
    recon.write_text(json.dumps({"geometry": {"z_step_um": 5.0,
                                              "voxel_xy_um": 0.98}}))
    assert _sampling_for(tmp_path / "ref.tif", recon) == (5.0, 0.98, 0.98)


def test_missing_spacing_is_none_so_the_caller_can_refuse(tmp_path):
    recon = tmp_path / "x_recon.json"
    recon.write_text(json.dumps({"geometry": {"voxel_xy_um": 0.98}}))
    assert _sampling_for(tmp_path / "ref.tif", recon) is None
    assert _sampling_for(tmp_path / "ref.tif", None) is None


def test_pairs_finds_the_masks_that_share_a_stem(tmp_path):
    shape = (3, 4, 4)
    for stem in ("S1_M2_FOVI_S1_vessels", "S1_M2_FOVII_S2_vessels"):
        tifffile.imwrite(tmp_path / f"{stem}_pred_mask.tif",
                         np.zeros(shape, dtype=np.uint8))
        tifffile.imwrite(tmp_path / f"{stem}_ref_mask.tif",
                         np.zeros(shape, dtype=np.uint8))
    (tmp_path / "S1_M2_FOVI_S1_vessels_recon.json").write_text("{}")
    got = _pairs(tmp_path)
    # sorted() order: "FOVII" sorts before "FOVI_", so the tuple is what
    # the glob gives, not the order the files were written in.
    assert [v for v, *_ in got] == ["M2_FOVII_S2_vessels", "M2_FOVI_S1_vessels"]
    assert all(ref.exists() for _, _, ref, _ in got)


def test_a_directory_without_masks_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError, match="run structure_report first"):
        _pairs(tmp_path)


def test_the_table_lists_every_tolerance_it_was_asked_for():
    row = {"arm": "S1", "volume": "v", "ref_skeleton_voxels": 10,
           "pred_skeleton_voxels": 9, "tolerance_um": {}, "tolerance_vox": {}}
    for t in (0.0, 1.0, 5.0):
        row["tolerance_um"][f"{t:g}"] = {"f1": 0.5}
    row["tolerance_vox"]["1"] = {"f1": 0.6}
    md = render([row], (0.0, 1.0, 5.0), (1.0,), "z=5 µm, xy=0.98 µm")
    header = next(ln for ln in md.splitlines() if ln.startswith("| arm |"))
    assert "0 µm" in header and "1 µm" in header and "5 µm" in header
    assert "1 vox" in header
    assert "| S1 | v | 10 → 9 | 0.5000 | 0.5000 | 0.5000 | 0.6000 |" in md


def test_a_tolerance_without_a_number_prints_a_dash_not_a_zero():
    row = {"arm": "S1", "volume": "v", "ref_skeleton_voxels": 1,
           "pred_skeleton_voxels": 1, "tolerance_um": {}, "tolerance_vox": {}}
    md = render([row], (1.0,), (1.0,), "z=5 µm")
    assert "| — |" in md and "| 0.0000 |" not in md


def test_the_cli_refuses_a_volume_with_unknown_spacing(tmp_path):
    shape = (3, 4, 4)
    sdir = tmp_path / "E021_S1_y_k2" / "structure"
    sdir.mkdir(parents=True)
    tifffile.imwrite(sdir / "S1_v_pred_mask.tif", np.zeros(shape, dtype=np.uint8))
    tifffile.imwrite(sdir / "S1_v_ref_mask.tif", np.zeros(shape, dtype=np.uint8))
    with pytest.raises(SystemExit, match="no voxel spacing"):
        main(["--structure", str(sdir), "--out", str(tmp_path / "out")])


def test_the_default_sweep_spans_zero_to_beyond_one_axial_step():
    assert 0.0 in DEFAULT_TOLERANCES_UM
    assert max(DEFAULT_TOLERANCES_UM) >= 5.0

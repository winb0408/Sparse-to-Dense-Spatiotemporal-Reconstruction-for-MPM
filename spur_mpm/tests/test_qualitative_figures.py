"""Invariants of the qualitative-figure geometry.

The figures are the part of this project a human reads first, and the failure
mode that motivated these tests was invisible in any table: the panel coordinates
were re-chosen from the *mask*, so two figures that claim to compare `y_k4` with
`z_k4` were actually showing different rows, and a Z-only protocol was labelled
"XZ given / XZ gap" as if it had subsampled rows.

These tests pin the properties the runbook asks for:
a frozen, JSON-round-trippable view spec; coordinates that do not depend on the
protocol; labels derived from the mask that is on screen; one display window
per panel taken from the dense reference; and a hard failure when the reference
itself differs between protocols.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spur_mpm.scripts.viewspec import (
    build_view_spec,
    check_group,
    check_protocols,
    load_view_spec,
    mask_geometry,
    protocol_columns,
    ref_windows,
    save_view_spec,
    take,
    validate_view_spec,
)

SHAPE = (9, 32, 32)          # small but odd enough to expose off-by-one crops


def _ref(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = rng.random(SHAPE).astype(np.float32)
    # a bright tube so the zoom ROI has something to centre on
    vol[:, 14:18, :] += 2.0
    return vol


def _masks() -> dict[str, np.ndarray]:
    z, h, w = SHAPE
    out = {}
    m = np.zeros(SHAPE, bool); m[:, ::2, :] = True
    out["y_k2"] = m
    m = np.zeros(SHAPE, bool); m[::2] = True
    out["z_k2"] = m
    m = np.zeros(SHAPE, bool); m[::2, ::2, :] = True
    out["yz_k2x2"] = m
    m = np.ones(SHAPE, bool)
    out["full"] = m
    return out


def _vol(ref, mask, arm="S1", pred=None, y_in=None):
    return {"arm": arm, "target": ref,
            "in_mask": mask, "acq_mask": mask,
            "y_in": np.where(mask, ref, 0.0) if y_in is None else y_in,
            "pred": ref if pred is None else pred}


# --- (i) one dense reference, many protocols -------------------------------

def test_the_same_reference_holds_for_every_protocol():
    ref = _ref()
    groups = [{"level": k, "vols": [_vol(ref, m)]}
              for k, m in _masks().items()]
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    rep = check_protocols(groups, spec)
    assert rep["target_diff"] == 0.0
    assert rep["fixed_view_diff"] == 0.0


def test_a_reference_that_depends_on_the_protocol_is_a_hard_failure():
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    # a single changed voxel anywhere breaks the reference invariant ...
    other = ref.copy()
    other[0, 0, 0] += 1e-3
    groups = [{"level": "y_k2", "vols": [_vol(ref, _masks()["y_k2"])]},
              {"level": "z_k2", "vols": [_vol(other, _masks()["z_k2"])]}]
    assert check_protocols(groups, spec)["target_diff"] > 0.0
    # ... and one inside a fixed panel is visible on the figure itself
    other = ref.copy()
    other[spec["z_view"], spec["y_view"], spec["x_view"]] += 1e-3
    groups[1]["vols"] = [_vol(other, _masks()["z_k2"])]
    assert check_protocols(groups, spec)["fixed_view_diff"] > 0.0


# --- (ii) the frozen spec gives the same cuts to every protocol ------------

def test_fixed_panels_are_protocol_independent():
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    per_protocol = {}
    for level, mask in _masks().items():
        cols = {c["key"]: c for c in protocol_columns(spec, mask)}
        fixed = {k: (c["axis"], c["coord"]) for k, c in cols.items()
                 if c["role"] == "fixed"}
        per_protocol[level] = fixed
        for key, col in cols.items():
            if col["role"] == "fixed":
                np.testing.assert_array_equal(take(ref, col, spec),
                                              take(ref, cols[key], spec))
    first = next(iter(per_protocol.values()))
    for level, fixed in per_protocol.items():
        assert fixed == first, f"{level} re-chose its fixed panels"


def test_a_saved_spec_round_trips_and_is_validated():
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    path = save_view_spec(spec, _tmpdir() / "view_spec.json")
    again = load_view_spec(path)
    assert again == spec
    assert validate_view_spec(again, SHAPE) is again
    # a spec from another volume size must not be silently clipped
    with pytest.raises(ValueError):
        validate_view_spec(again, (SHAPE[0], SHAPE[1] + 4, SHAPE[2]))


def _tmpdir():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp())


# --- (iii) Z-only: planes, not rows ---------------------------------------

def test_z_only_protocol_reports_planes_and_never_rows():
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    mask = _masks()["z_k2"]
    cols = {c["key"]: c for c in protocol_columns(spec, mask)}
    assert "XY measured plane" in cols and "XY skipped plane" in cols
    # the label that was wrong before: a Z protocol does not subsample rows
    assert "XZ measured row" not in cols
    assert "XZ skipped row" not in cols
    geom = mask_geometry(mask)
    assert geom["y_dense"] is True and geom["z_dense"] is False
    # the reported coordinates agree with the mask itself
    assert mask[cols["XY measured plane"]["coord"]].all()
    assert not mask[cols["XY skipped plane"]["coord"]].any()
    assert geom["z_planes_measured"] == int(mask.any(axis=(1, 2)).sum())


# --- (iv) YZ: planes and rows, both labelled ------------------------------

def test_yz_protocol_labels_both_axes_with_the_mask():
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    mask = _masks()["yz_k2x2"]
    cols = {c["key"]: c for c in protocol_columns(spec, mask)}
    assert {"XY measured plane", "XY skipped plane",
            "XZ measured row", "XZ skipped row"} <= set(cols)
    zm = cols["XY measured plane"]["coord"]
    zg = cols["XY skipped plane"]["coord"]
    assert mask[zm].any() and not mask[zg].any()
    # a visited plane is only *partly* measured here, and the note says so
    assert "50% of voxels measured" in cols["XY measured plane"]["note"]
    ym = cols["XZ measured row"]["coord"]
    yg = cols["XZ skipped row"]["coord"]
    assert mask[:, ym, :].any() and not mask[:, yg, :].any()


def test_y_only_protocol_reports_rows_and_never_planes():
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    cols = {c["key"]: c for c in protocol_columns(spec, _masks()["y_k2"])}
    assert {"XZ measured row", "XZ skipped row"} <= set(cols)
    assert "XY skipped plane" not in cols          # every plane was measured
    geom = mask_geometry(_masks()["y_k2"])
    assert geom["z_dense"] is True and geom["y_dense"] is False
    assert geom["y_period"] == 2


# --- (v) one window per panel, from the reference ------------------------

def test_display_windows_come_from_the_reference_not_from_the_prediction():
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    cols = protocol_columns(spec, _masks()["y_k2"])
    windows = ref_windows(ref, cols, spec)
    dim = ref * 0.25
    # the same window is used whatever the arm produced: a dim prediction must
    # read as dim, not be re-normalised into a plausible picture
    assert ref_windows(dim, cols, spec) != windows
    assert set(windows) == {c["key"] for c in cols}


# --- (vi) same-input check inside a group --------------------------------

def test_arms_that_were_not_handed_the_same_input_are_rejected():
    ref = _ref()
    mask = _masks()["y_k2"]
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    good = _vol(ref, mask, "S1")
    same = _vol(ref, mask, "S2", pred=ref * 0.9)
    assert check_group([good, same], spec)["input_diff"] == 0.0
    broken = _vol(ref, mask, "S2", y_in=np.where(mask, ref * 1.01, 0.0))
    rep = check_group([good, broken], spec)
    assert rep["input_diff"] > 0.0


def test_a_group_with_a_different_reference_is_rejected():
    ref = _ref()
    mask = _masks()["y_k2"]
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    a = _vol(ref, mask, "S1")
    b = _vol(ref + 0.01, mask, "S2")
    rep = check_group([a, b], spec)
    assert rep["target_diff"] > 0.0


def test_group_labels_become_filename_safe():
    """A group may be labelled "y_k4, c7"; the figure name must still be a name."""
    from spur_mpm.scripts.qualitative_figures import slug
    assert slug("y_k4") == "y_k4"
    assert slug("y_k4, c7") == "y_k4_c7"
    assert slug("yz_k2x2 (25%)") == "yz_k2x2_25"


def test_a_frozen_spec_carries_its_windows_and_provenance(tmp_path):
    """§4.4.1 D: one file per sample states the cuts, the windows and the source."""
    import json
    from spur_mpm.scripts.qualitative_figures import _freeze
    from spur_mpm.scripts.viewspec import load_view_spec, save_view_spec
    ref = _ref()
    mask = _masks()["y_k2"]
    spec = build_view_spec(ref, sample_id="s", zoom=16)
    cfg = {"data": {"index": str(tmp_path / "index.json"),
                    "split": str(tmp_path / "split.json")},
           "sampling": {"mode": "y", "ky": 2, "kz": 1}}
    (tmp_path / "index.json").write_text("{}")
    (tmp_path / "split.json").write_text("{}")
    for f in ("index.json", "split.json"):
        p = tmp_path / f
        p.write_text(p.read_text() + " ")
    frozen = _freeze(spec, ref, mask, cfg)
    # the windows are the reference's, one per fixed panel -- not per diagnostic
    assert set(frozen["windows"]) <= {c["key"] for c in
                                      protocol_columns(spec, mask)}
    assert all(len(v) == 2 and v[0] < v[1] for v in frozen["windows"].values())
    assert frozen["fingerprint"]["config_sha256"]
    # a loaded frozen spec still validates against the same volume
    out = save_view_spec(frozen, tmp_path / "view_spec.json")
    again = validate_view_spec(load_view_spec(out), ref.shape)
    assert again["roi"] == frozen["roi"] and again["z_view"] == frozen["z_view"]
    # and the hash tracks the config, so a changed input cannot reuse a spec
    other = _freeze(spec, ref, mask, {**cfg, "sampling": {"mode": "z", "kz": 4}})
    assert (other["fingerprint"]["config_sha256"]
            != frozen["fingerprint"]["config_sha256"])


def test_the_fixed_xy_panel_can_be_named_explicitly():
    """§4.4.1 B: the shared XY column should be a plane every protocol measured."""
    ref = _ref()
    spec = build_view_spec(ref, sample_id="s", zoom=16, z_view=4)
    assert spec["z_view"] == 4
    # and a coordinate outside the volume is refused, not clipped
    import pytest
    with pytest.raises(ValueError, match="outside the volume"):
        build_view_spec(ref, sample_id="s", zoom=16, z_view=ref.shape[0])


def test_projection_note_describes_the_panels_actually_drawn():
    """The note is a claim about the picture; it must come from the columns.

    A constant said "none (single XY/XZ/YZ slices; mip_z kept for a future
    projection)" after a MIP-z column had been added, so every stored
    view_spec.json described a figure without a projection while the PNG had
    one.
    """
    from spur_mpm.scripts.viewspec import annotate, projection_note

    mask = np.ones(SHAPE, dtype=bool)
    spec = build_view_spec(_ref(), sample_id="S")
    cols = protocol_columns(spec, mask)
    assert any(c["axis"] == "mipz" for c in cols)
    note = projection_note(cols)
    assert "MIP-z" in note
    assert "none" not in note.lower()
    for panel in ("XY", "XZ", "YZ", "Zoom"):
        assert panel in note, panel
    # and it wins over a stale note loaded from an older frozen spec
    frozen = dict(spec, projection="none (single XY/XZ/YZ slices)")
    assert "MIP-z" in annotate(frozen, columns=cols)["projection"]


def test_projection_note_without_a_projection_panel():
    from spur_mpm.scripts.viewspec import projection_note

    cols = [{"key": "XY", "axis": "xy"}, {"key": "XZ", "axis": "xz"}]
    assert projection_note(cols).startswith("no projection")
    assert "not recorded" in projection_note(None)


def test_the_mask_must_be_the_protocol_it_is_filed_under():
    """A `z_k4` figure whose mask subsamples rows is the §4.4.1 failure mode."""
    from spur_mpm.scripts.viewspec import sampling_semantics

    z, h, w = 40, 64, 64
    rows = np.zeros((z, h, w), dtype=bool)
    rows[:, ::4, :] = True                     # y_k4, every plane visited
    planes = np.zeros((z, h, w), dtype=bool)
    planes[::4] = True                         # z_k4, whole planes skipped

    y4 = sampling_semantics(rows, {"ky": 4, "kz": 1, "ideal_fraction": 0.25})
    z4 = sampling_semantics(planes, {"ky": 1, "kz": 4, "ideal_fraction": 0.25})
    assert y4["axes_ok"] and y4["fraction_ok"]
    assert z4["axes_ok"] and z4["fraction_ok"]
    # the same mask under the other label is caught on the axis flags
    assert not sampling_semantics(rows, {"ky": 1, "kz": 4})["axes_ok"]
    assert not sampling_semantics(planes, {"ky": 4, "kz": 1})["axes_ok"]
    # and a mask that is far denser than the protocol claims is caught on size
    dense = np.ones((z, h, w), dtype=bool)
    bad = sampling_semantics(dense, {"ky": 8, "kz": 1, "ideal_fraction": 0.125})
    assert not bad["fraction_ok"]


def test_frozen_windows_are_the_ones_this_reference_derives():
    """If the spec was frozen from another volume, the two sets must disagree."""
    from spur_mpm.scripts.viewspec import annotate, ref_windows, window_stability

    spec = build_view_spec(_ref(), sample_id="S")
    cols = protocol_columns(spec, np.ones(SHAPE, dtype=bool))
    fixed = [c for c in cols if c["role"] == "fixed"]
    frozen = annotate(spec, windows=ref_windows(_ref(), fixed, spec), columns=cols)
    good = window_stability(frozen, _ref(), np.ones(SHAPE, dtype=bool))
    assert good["windows_ok"] and good["window_diff"] == 0.0

    other = annotate(spec, windows=ref_windows(_ref(seed=7), fixed, spec) , columns=cols)
    bad = window_stability(other, _ref(), np.ones(SHAPE, dtype=bool))
    assert not bad["windows_ok"] and bad["window_diff"] > 0

    empty = dict(frozen)
    empty.pop("windows")
    assert not window_stability(empty, _ref(), np.ones(SHAPE, dtype=bool))["windows_ok"]

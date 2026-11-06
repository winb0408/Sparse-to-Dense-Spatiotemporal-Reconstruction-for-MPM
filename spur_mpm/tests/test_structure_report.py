"""Tests for the structure report's markdown rendering.

The table quotes two different component counts from the same JSON: the number
of 26-connected **skeleton** fragments (``ref_n_components``) and the number of
connected **mask** blobs (``n_components_ref``). They differ by orders of
magnitude (1580 vs a handful on one M2 volume), so a header that just says
"components" is a claim the reader cannot check. These tests pin the wording to
the fact it quotes.

The second test covers the no-GPU path: ``--from-summary`` reads everything it
needs from the JSON, so it must not also demand ``--arm`` / ``--out``.
"""

from __future__ import annotations

import json

import pytest

from spur_mpm.scripts.structure_report import main, render_summary


def _record(**over) -> dict:
    r = {
        "sample_id": "S1",
        "dice": 0.9,
        "cl_dice": 0.8,
        "f1": 0.7,
        "ref_n_components": 11,
        "pred_n_components": 12,
        "ref_n_junctions": 3,
        "pred_n_junctions": 4,
        "ref_total_length_um": 100.0,
        "pred_total_length_um": 90.0,
        "ref_mean_radius_um": 1.0,
        "pred_mean_radius_um": 1.1,
        "n_planes": 10,
        "n_planes_gated": 2,
        "max_plane_fraction": 0.15,
        "segment_method": "vesselness",
        "threshold": 0.04,
        "threshold_mode": "per_plane",
        "ref_fraction": 0.02,
        "pred_fraction": 0.02,
        # The mask-blob counts live in the JSON too and are NOT what the table
        # quotes; keeping them different makes a mix-up visible in the test.
        # Deliberately odd values: a round number would collide with a Dice.
        "n_components_ref": 7777,
        "n_components_pred": 6666,
    }
    r.update(over)
    return r


def _summary(*records) -> dict:
    rows = list(records) or [_record()]
    keys = ("dice", "cl_dice", "f1", "ref_n_components", "pred_n_components",
            "ref_n_junctions", "pred_n_junctions", "ref_total_length_um",
            "pred_total_length_um", "ref_mean_radius_um", "pred_mean_radius_um")
    return {"volumes": rows,
            "means": {k: sum(r[k] for r in rows) / len(rows) for k in keys}}


def test_header_names_the_skeleton_component_count():
    md = render_summary(_summary())
    header = md.splitlines()[0]
    assert "skeleton components ref→pred" in header
    # and it must not be readable as the mask-blob count
    assert "| components ref→pred |" not in header
    assert "mask blobs are recorded separately" in md


def test_table_quotes_skeleton_components_not_mask_blobs():
    md = render_summary(_summary())
    row = [ln for ln in md.splitlines() if ln.startswith("| S1 |")][0]
    assert "11 → 12" in row
    assert "7777" not in row and "6666" not in row


def test_empty_summary_does_not_invent_rows():
    md = render_summary(_summary())
    assert len([ln for ln in md.splitlines() if ln.startswith("| ")]) == 3


def test_from_summary_rerenders_without_arm_or_out(tmp_path):
    path = tmp_path / "structure_summary.json"
    path.write_text(json.dumps(_summary()))
    assert main(["--from-summary", str(path)]) == 0
    text = (tmp_path / "structure_summary.md").read_text()
    assert "skeleton components ref→pred" in text


def test_analysing_still_requires_arm_and_out():
    with pytest.raises(SystemExit):
        main([])

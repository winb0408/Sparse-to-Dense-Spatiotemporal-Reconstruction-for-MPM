"""The depth profiler decides what counts as a defect and what is expected.

Getting that distinction wrong is expensive in both directions: calling depth
attenuation a defect would throw away every stack, and calling saturation a
note would let clipped pixels into the calibration and the metrics.
"""

from __future__ import annotations

import numpy as np

from spur_mpm.scripts.profile_depth import _notes, _problems, profile_volume, summarise


def _plane_row(mean: float, p99: float, sat: float = 0.0, zero: float = 0.0) -> dict:
    return {"sample_id": "S", "mouse_id": "1", "session_id": "1", "fov_id": "I",
            "plane": 0, "depth_um": 0.0, "mean": mean, "median": 0.0,
            "p99": p99, "std": 0.0, "zero_frac": zero, "saturated_frac": sat}


def test_saturation_is_a_defect_not_a_note():
    rows = [_plane_row(100, 200, sat=0.05)] + [_plane_row(100, 200)] * 9
    problems = _problems(rows)
    assert problems and "full scale" in problems[0]
    assert not any("SNR regime" in p for p in problems)


def test_depth_attenuation_is_a_note_not_a_defect():
    rows = [_plane_row(100, 200)] + [_plane_row(70, 140)] * 9
    assert _problems(rows) == []
    notes = _notes(rows)
    assert notes and "p99 falls" in notes[0]


def test_flat_stack_raises_neither():
    rows = [_plane_row(100, 200)] * 10
    assert _problems(rows) == []
    assert _notes(rows) == []


class _Geom:
    mouse_id = "1"
    session_id = "1"
    fov_id = "I"
    z_step_um = 5.0
    depth_origin_um = 0.0


class _Cache:
    def __init__(self, arr):
        self._arr = arr

    def array(self, sid):
        return self._arr

    def geometry(self, sid):
        return _Geom()


def test_profile_reads_every_plane_with_physical_depth():
    arr = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4) * 100
    rows = profile_volume(_Cache(arr), "S", chunk=4)
    assert [r["plane"] for r in rows] == list(range(6))
    assert rows[1]["depth_um"] == 5.0
    assert rows[-1]["depth_um"] == 25.0


def test_summary_reports_the_span_in_micrometres():
    arr = np.zeros((5, 4, 4), dtype=np.uint16)
    summaries = summarise(profile_volume(_Cache(arr), "S", chunk=5))
    assert summaries[0]["n_planes"] == 5
    assert summaries[0]["depth_span_um"] == 20.0

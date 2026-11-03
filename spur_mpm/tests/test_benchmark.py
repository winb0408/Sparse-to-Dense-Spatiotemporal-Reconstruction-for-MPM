"""The cost table must not price the parameter-free floor at zero.

A0/A1 have no network, so they have no parameters and no MACs -- but they do run
an operator over every voxel of the volume. The table used to print
``latency n/a`` for them, which reads as "the floor is free" and is exactly the
comparison a complexity table exists to make.
"""

from __future__ import annotations

import pytest

from spur_mpm.scripts.benchmark import benchmark_all, format_markdown

IDENTITY_PSF = {"sigma_xy_um": 0.0, "sigma_z_um": 0.0,
                "voxel_xy_um": 0.98, "voxel_z_um": 5.0}
# Small enough that the test is about wiring, not throughput.
SMALL = dict(patch=16, n_slices=4, warmup=0, iters=1)
MODEL_CFG = {"base": 4, "window": 8, "depths": [1, 1, 1], "heads": [1, 1, 1]}


def _payload(psf):
    return benchmark_all(["A0", "A1"], device="cpu", model_cfg=MODEL_CFG,
                         context_depth=1, amp=False, psf=psf, lam=0.01, **SMALL)


def test_baselines_are_timed_not_dropped():
    payload = _payload(IDENTITY_PSF)
    for row in payload["arms"]:
        assert row["params"] == 0
        assert row["macs_per_slice"] is None, "no network, so no MACs"
        assert row["sec_per_volume"] is not None, row["note"]
        assert row["sec_per_volume"] > 0.0
        assert row["latency_ms_per_slice"] is not None
        assert "not zero" in row["note"]


def test_without_a_psf_the_baseline_still_reports_rather_than_raising():
    """A caller that has no physics config gets the old honest note, not a crash."""
    payload = benchmark_all(["A0"], device="cpu", model_cfg=MODEL_CFG,
                            context_depth=1, amp=False, psf=None, **SMALL)
    row = payload["arms"][0]
    assert row["sec_per_volume"] is None
    assert "not applicable" in row["note"]


def test_markdown_says_the_floor_was_measured():
    text = format_markdown(_payload(IDENTITY_PSF))
    assert "parameter-free floors have no MACs" in text
    assert "A0, A1" in text
    assert "latency n/a" not in text.replace("| n/a |", "| MACs-n/a |")


def test_a_warm_start_that_cannot_run_is_a_note_not_a_traceback(monkeypatch):
    import spur_mpm.scripts.benchmark as benchmark

    def boom(*a, **k):
        raise RuntimeError("no fft here")

    monkeypatch.setattr(benchmark, "warm_start", boom)
    row = _payload(IDENTITY_PSF)["arms"][0]
    assert row["sec_per_volume"] is None
    assert "could not be timed" in row["note"]
    assert "no fft here" in row["note"]

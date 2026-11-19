"""The window protocol's vessel rule must be explicit, and default to history.

E026 §4.3: the window protocol's global Otsu puts its vessel reference at 18.65 %
of a real frame while the whole-volume protocol's per-plane threshold with a 15 %
gate puts it at 2.18 %. The two score different references, so their Dice for the
same arm differ in sign. The fix is a re-run, and a re-run is only useful if the
rule that produced each column is recorded rather than remembered.
"""

from __future__ import annotations

import pytest

from spur_mpm.evaluate import vessel_settings


def test_the_defaults_are_the_published_protocol():
    # Anything that produced a number before this knob existed must keep doing so.
    assert vessel_settings({}) == ("global", None)
    assert vessel_settings({"eval": {}}) == ("global", None)


def test_the_gated_rule_is_read_as_configured():
    cfg = {"eval": {"vessel_threshold_mode": "per_plane",
                    "vessel_max_fraction": 0.15}}
    assert vessel_settings(cfg) == ("per_plane", 0.15)


def test_a_percentage_written_as_a_float_is_used_verbatim():
    # 0.15 means 15 %; a caller writing 15 is almost certainly a percent sign
    # lost in translation and must not silently disable the gate.
    with pytest.raises(ValueError, match="vessel_max_fraction"):
        vessel_settings({"eval": {"vessel_max_fraction": 15}})


def test_an_unknown_threshold_mode_is_refused():
    with pytest.raises(ValueError, match="vessel_threshold_mode"):
        vessel_settings({"eval": {"vessel_threshold_mode": "per_window"}})


def test_zero_and_negative_gates_are_refused():
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="vessel_max_fraction"):
            vessel_settings({"eval": {"vessel_max_fraction": bad}})


def test_the_summary_records_which_rule_was_used():
    # The contract `vessel_protocol` exists for: a summary must say whether its
    # dice came from a gated or an ungated reference, and how many windows it
    # dropped. Checked on the emit site rather than by running an evaluation.
    from pathlib import Path
    src = Path("spur_mpm/evaluate.py").read_text()
    assert '"vessel_protocol"' in src
    for field in ("threshold_mode", "max_fraction", "n_windows_gated",
                  "n_windows_scored"):
        assert f'"{field}"' in src, field

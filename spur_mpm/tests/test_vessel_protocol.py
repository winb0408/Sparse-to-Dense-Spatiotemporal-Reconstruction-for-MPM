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


def test_a_single_window_is_a_plane_and_the_gate_applies_to_it():
    # The bug this pins: the gate returned early for anything that was not 3-D,
    # so `evaluate.py` (one plane per window) gated nothing while its summary
    # said max_fraction=0.15. Two floors re-scored under the "gated" rule came
    # back numerically identical to the ungated run, n_windows_gated = 0.
    import numpy as np
    from spur_mpm.metrics.vessel import _gate_flooded_planes

    ref = np.zeros((32, 32), bool)
    ref.ravel()[: int(0.20 * 32 * 32)] = True          # 20 % of the frame
    pred = ref.copy()

    pm, rm, n = _gate_flooded_planes(pred, ref, 0.15)

    assert n == 1, "a flooded single window must be reported as gated"
    assert not pm.any() and not rm.any(), "both masks are emptied together"


def test_a_single_window_under_the_gate_is_left_alone():
    import numpy as np
    from spur_mpm.metrics.vessel import _gate_flooded_planes

    ref = np.zeros((32, 32), bool)
    ref.ravel()[: int(0.10 * 32 * 32)] = True          # 10 % of the frame
    pred = ref.copy()

    pm, rm, n = _gate_flooded_planes(pred, ref, 0.15)

    assert n == 0
    assert pm.sum() == pred.sum() and rm.sum() == ref.sum()


def test_a_stack_still_gates_per_plane():
    import numpy as np
    from spur_mpm.metrics.vessel import _gate_flooded_planes

    ref = np.zeros((2, 16, 16), bool)
    ref[0, :, :] = True                                 # plane 0 is entirely flooded
    ref[1, :2, :2] = True                               # plane 1 is clean
    pm, rm, n = _gate_flooded_planes(ref.copy(), ref, 0.15)

    assert n == 1
    assert not rm[0].any() and rm[1].any()


def test_an_array_that_is_neither_a_plane_nor_a_stack_is_refused():
    import numpy as np
    import pytest as _pytest
    from spur_mpm.metrics.vessel import _gate_flooded_planes

    with _pytest.raises(ValueError, match="one plane"):
        _gate_flooded_planes(np.zeros(8, bool), np.zeros(8, bool), 0.15)


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


def test_a_literature_kernel_is_a_label_of_its_own():
    """Issue #4: the bead-FWHM Gaussian may not borrow another licence."""
    import pytest as _pytest
    from spur_mpm.physics.prior import PSF_SOURCES, _source_for
    from spur_mpm.physics.psf import PSFConfig

    paper = PSFConfig(0.36946, 1.9494, 0.98, 5.0)
    assert _source_for(paper, "literature_bead_fwhm_gaussian") == \
        "literature_bead_fwhm_gaussian"
    assert "literature_bead_fwhm_gaussian" in PSF_SOURCES
    # And an assumed-identity label cannot describe a kernel that blurs.
    with _pytest.raises(ValueError, match="unlabelled_identity_assumed"):
        _source_for(paper, "unlabelled_identity_assumed")

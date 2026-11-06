"""Fig. 2 must be regenerated from the registry, never retyped.

The figure's whole claim is "this table is the code".  That is only true if a
tick cannot disagree with ``ARMS`` and a missing measurement cannot be rounded
into a zero, so the tests below flip one registry field at a time and check the
cell follows, and feed a source with a null and check it stays ``n/a``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from spur_mpm.models.registry import ARMS
from spur_mpm.scripts.framework_figure import (
    COLUMN_BY_KEY,
    FROZEN,
    NA,
    Number,
    cell_for,
    format_number,
    group_of,
    load_numbers,
    main,
    matrix_blocks,
    matrix_rows,
    pipeline_facts,
)


def _numbers(names=None, **overrides):
    """Synthetic numbers: enough for every non-frozen arm to be drawable."""
    out = {name: {} for name in (ARMS if names is None else names)
           if group_of(ARMS[name]) != "frozen"}
    for arm, fields in overrides.items():
        out.setdefault(arm, {}).update(fields)
    return out


def _num(value, field="params_m"):
    return Number(value, "synthetic.json", f"arms[?].{field}")


# ---------------------------------------------------------------------------
# (i) every tick is derived from ARMS
# ---------------------------------------------------------------------------

def test_uncertainty_column_follows_the_features_tuple():
    arm = ARMS["S1"]
    off = cell_for(arm, COLUMN_BY_KEY["uncertainty"])
    on = cell_for(dataclasses.replace(
        arm, features=arm.features + ("uncertainty",)), COLUMN_BY_KEY["uncertainty"])
    assert off.tick is False and on.tick is True
    assert on.field == "ARMS['S1'].features"


def test_gate_column_follows_the_fusion_flag():
    arm = ARMS["S0"]
    assert cell_for(arm, COLUMN_BY_KEY["fusion"]).tick is False
    assert cell_for(dataclasses.replace(arm, fusion=True),
                    COLUMN_BY_KEY["fusion"]).tick is True


def test_trunk_column_follows_needs_training():
    arm = ARMS["A0"]                       # a floor: no network
    assert cell_for(arm, COLUMN_BY_KEY["trunk"]).tick is False
    assert cell_for(dataclasses.replace(arm, needs_training=True),
                    COLUMN_BY_KEY["trunk"]).tick is True


def test_x0_column_is_the_feature_not_the_warm_start_mode():
    # A1 warm-starts but does not feed X0: the two facts must not be conflated.
    assert cell_for(ARMS["A1"], COLUMN_BY_KEY["x0"]).tick is False
    assert cell_for(ARMS["A1"], COLUMN_BY_KEY["warm_start"]).text == "wiener_fill_first"
    assert cell_for(ARMS["S1"], COLUMN_BY_KEY["x0"]).tick is True


def test_warm_start_text_and_group_follow_the_registry():
    arm = ARMS["S0"]
    assert cell_for(arm, COLUMN_BY_KEY["warm_start"]).text == arm.warm_start
    assert group_of(arm) == "supervised"
    assert group_of(dataclasses.replace(arm, needs_training=False)) == "floors"
    assert group_of(dataclasses.replace(arm, supervision="self")) == "frozen"


def test_matrix_row_follows_an_edited_registry():
    arms = {"S2": dataclasses.replace(ARMS["S2"], features=("y", "mask"),
                                      fusion=True)}
    row = matrix_rows(arms=arms, numbers=_numbers())[0]
    assert row.cell("x0").tick is False
    assert row.cell("uncertainty").tick is False
    assert row.cell("fusion").tick is True


def test_pipeline_channel_counts_are_the_declared_feature_lengths():
    facts = pipeline_facts()
    assert facts["features_per_arm"]["S0"] == len(ARMS["S0"].features)
    assert "S1" not in facts["frozen_arms"]
    assert set(facts["fusion_arms"]) == {n for n, a in ARMS.items() if a.fusion}


# ---------------------------------------------------------------------------
# (ii) a null stays n/a, never 0
# ---------------------------------------------------------------------------

def test_null_number_formats_as_na_not_zero():
    assert format_number(None, "macs_per_slice") == NA
    assert format_number(Number(None, "synthetic.json", "f"), "macs_per_slice") == NA


def test_null_in_the_source_reaches_the_cell_as_na(tmp_path):
    src = tmp_path / "numbers.json"
    src.write_text(json.dumps({"arms": [
        {"arm": name, "params_m": 1.0, "macs_per_slice": 2.3e10,
         "latency_ms_per_slice": 2.0}
        for name, arm in ARMS.items()
        if group_of(arm) != "frozen" and name != "S0"]
        + [{"arm": "S0", "params_m": 1.0, "macs_per_slice": None,
            "latency_ms_per_slice": 2.0}]}),
        encoding="utf-8")
    rows = matrix_rows(numbers=load_numbers((src,)))
    cell = [r for r in rows if r.arm.name == "S0"][0].cell("macs")
    assert cell.text == NA
    assert cell.text != "0"
    assert "null" in cell.field


def test_measured_arm_without_macs_prints_na(tmp_path):
    # A0/A1 are real rows with a real params value but no MAC count.
    rows = matrix_rows()                       # the repo's own sources
    a0 = [r for r in rows if r.arm.name == "A0"][0]
    assert a0.cell("macs").text == NA
    assert a0.cell("params").text == "0.000"


# ---------------------------------------------------------------------------
# (iii) rendering is side-effect-clean and works without the real data
# ---------------------------------------------------------------------------

def test_synthetic_render_writes_pngs_and_provenance(tmp_path):
    src = tmp_path / "numbers.json"
    src.write_text(json.dumps({"arms": [
        {"arm": name, "params_m": 2.5, "macs_per_slice": 2.3e10,
         "latency_ms_per_slice": 28.0}
        for name, arm in ARMS.items() if group_of(arm) != "frozen"]}),
        encoding="utf-8")
    rc = main(["--benchmark", str(src), "--complexity", str(src),
               "--out", str(tmp_path / "out"), "--dpi", "60"])
    assert rc == 0

    for name in ("fig2_pipeline.png", "fig2_ablation_matrix.png"):
        png = (tmp_path / "out" / name).read_bytes()
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 1000

    prov = json.loads((tmp_path / "out" / "fig2_provenance.json")
                      .read_text(encoding="utf-8"))
    assert set(prov["arms"]) == set(ARMS)
    assert prov["arms"]["S0"]["cells"]["y"]["source"].endswith("registry.py")
    facts = pipeline_facts()
    assert set(facts["warm_start_modes"]) >= {"bilinear", "wiener"}


# ---------------------------------------------------------------------------
# (iv) nothing is silently skipped
# ---------------------------------------------------------------------------

def test_every_registry_arm_appears_exactly_once():
    rows = matrix_rows()
    names = [r.arm.name for r in rows]
    assert sorted(names) == sorted(ARMS)
    assert len(names) == len(set(names))


def test_blocks_cover_every_row():
    rows = matrix_rows()
    blocks = matrix_blocks(rows)
    assert sum(len(b.rows) for b in blocks) == len(rows)
    assert [b.key for b in blocks] == ["floors", "supervised", "frozen"]


def test_frozen_rows_say_frozen_rather_than_blank():
    rows = {r.arm.name: r for r in matrix_rows()}
    for name, arm in ARMS.items():
        if arm.supervision == "self":
            assert rows[name].frozen is True
            assert rows[name].cell("params").text == FROZEN
            assert rows[name].cell("macs").text == FROZEN
            # the design is still shown, in grey
            assert rows[name].cell("y").tick is True


def test_supervised_arm_without_a_measurement_is_an_error():
    with pytest.raises(KeyError, match="no measured numbers"):
        matrix_rows(arms=ARMS, numbers={})


def test_frozen_arm_carrying_numbers_is_an_error():
    numbers = _numbers()
    numbers["SS0"] = {"params_m": _num(1.0)}
    with pytest.raises(RuntimeError, match="frozen"):
        matrix_rows(numbers=numbers)


# ---------------------------------------------------------------------------
# (i') the lattice list on the panel is the code's list
# ---------------------------------------------------------------------------

def test_sampling_modes_are_quoted_from_the_module():
    from spur_mpm.physics.sampling import MODES

    assert tuple(pipeline_facts()["sampling_modes"]) == tuple(MODES)


def test_lattice_text_follows_the_fact_not_a_literal():
    from spur_mpm.scripts.framework_figure import _lattice_text

    facts = pipeline_facts()
    facts["sampling_modes"] = ["y", "q"]
    assert _lattice_text(facts) == "lattices: y | q"


def test_every_builtin_mode_is_accepted_by_spec():
    from spur_mpm.physics.sampling import MODES, SamplingSpec

    for mode in MODES:
        assert SamplingSpec(mode=mode).mode == mode

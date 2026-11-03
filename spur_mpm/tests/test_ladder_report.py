"""The two tables that police the axis claim, tested on numbers I chose.

``ladder_report`` has no computation of its own -- it formats ``summary.json``.
The part worth testing is therefore the *rhetoric*: a spread under the floor must
render as "no claim", a spread over it as "axis matters", and a metric the run did
not produce must read as missing rather than as ``nan`` (a bare ``nan`` in a table
looks like a number that was measured and failed, which is not what happened).

The budget curve is tested for ordering instead: levels must appear densest
first, and the reference row is the densest one that ran.
"""

from __future__ import annotations

from spur_mpm.scripts.ladder_report import (
    DICE_FLOOR,
    axis_comparison,
    budget_curve,
)


def _cell(dice: float, cl: float = 0.70, psnr: float = 29.0, f1: float | None = None):
    out = {"dice": dice, "cl_dice": cl, "psnr": psnr}
    if f1 is not None:
        out["f1"] = f1
    return out


def test_axis_comparison_refuses_a_spread_under_the_floor():
    cells = {("S1", "y_k4"): _cell(0.720), ("S1", "z_k4"): _cell(0.721),
             ("S1", "yz_k2x2"): _cell(0.722)}
    text = axis_comparison(cells)
    assert "no claim" in text
    assert "axis matters" not in text


def test_axis_comparison_claims_a_spread_over_the_floor():
    cells = {("S1", "y_k4"): _cell(0.700), ("S1", "z_k4"): _cell(0.700 + DICE_FLOOR + 0.01),
             ("S1", "yz_k2x2"): _cell(0.700)}
    text = axis_comparison(cells)
    assert "axis matters" in text
    assert "no claim" not in text


def test_axis_comparison_needs_two_levels_to_have_a_spread():
    # one level present -> a spread is undefined, so the arm is not reported
    text = axis_comparison({("A1", "y_k4"): _cell(0.84)})
    assert text == ""


def test_missing_skeleton_metric_prints_as_missing_not_nan():
    cells = {("S1", "y_k4"): _cell(0.72), ("S1", "y_k2"): _cell(0.75, f1=0.70)}
    text = budget_curve(cells, "S1")
    assert "nan" not in text
    assert "0.7000" in text          # y_k2 has a skeleton F1 and shows it
    assert "| – |" in text           # y_k4 does not, and says so


def test_budget_curve_is_ordered_densest_first():
    cells = {("S1", "y_k8"): _cell(0.70), ("S1", "y_k2"): _cell(0.75),
             ("S1", "y_k4"): _cell(0.72)}
    rows = [ln for ln in budget_curve(cells, "S1").splitlines() if ln.startswith("| y_k")]
    assert [r.split("|")[1].strip() for r in rows] == ["y_k2", "y_k4", "y_k8"]


def test_budget_curve_measures_each_level_against_the_densest_that_ran():
    cells = {("S1", "y_k4"): _cell(0.72), ("S1", "y_k8"): _cell(0.70)}
    text = budget_curve(cells, "S1")
    # y_k4 is the reference, so the claim column cannot call it a win or a loss
    assert "| — |" in text
    assert "-0.0200" in text         # y_k8 loses 0.02 Dice, under the floor
    assert "no (< .03 Dice)" in text


def test_forced_context_gets_its_own_cell_name():
    from spur_mpm.scripts.run_ladder import cell_name

    # the same level trained at its own minimum context and at a fixed context
    # are different experiments; one must not overwrite the other
    assert cell_name("S1", "y_k4") == "E021_S1_y_k4"
    assert cell_name("S1", "y_k4", 7) == "E021_S1_y_k4_c7"


def test_level_context_is_the_minimum_the_protocol_needs():
    from spur_mpm.scripts.run_ladder import level_context

    assert level_context("y_k4") == 1        # z is never skipped
    assert level_context("z_k2") == 3        # 2*(2-1)+1
    assert level_context("z_k4") == 7        # 2*(4-1)+1
    assert level_context("yz_k2x2") == 3
    assert level_context("yz_k2x4") == 7     # the plane spacing decides


def test_only_filter_matches_the_final_cell_name():
    from spur_mpm.scripts.run_ladder import select

    pairs = [("S1", "y_k4"), ("S1", "y_k8"), ("A0", "y_k4")]
    # an arm+level pair, not a level: "y_k4" must not drag y_k8 along
    assert select(pairs, ["y_k4"]) == [("S1", "y_k4"), ("A0", "y_k4")]
    # Under a forced context both y_k4 cells *own* the _c7 name, so "y_k4"
    # selects them too -- substring matching cannot tell the two experiments
    # apart, and it should not pretend to: the safe way to re-score the
    # own-context cell is to drop --context-slices from that invocation.
    assert select(pairs, ["y_k4_c7"], 7) == [("S1", "y_k4"), ("A0", "y_k4")]
    assert select(pairs, ["y_k4"], 7) == [("S1", "y_k4"), ("A0", "y_k4")]
    assert select(pairs, ["S1"]) == [("S1", "y_k4"), ("S1", "y_k8")]


def test_attribution_ladder_reaches_the_complete_arm():
    """S2g is S1g plus U, so the table must be able to ask that one question."""
    from spur_mpm.scripts.ladder_report import innovations

    cells = {("S0", "y_k2"): {"psnr": 30.0, "dice": 0.70},
             ("S1", "y_k2"): {"psnr": 30.0, "dice": 0.70},
             ("S1g", "y_k2"): {"psnr": 30.0, "dice": 0.70},
             ("S2g", "y_k2"): {"psnr": 31.5, "dice": 0.75}}
    text = innovations(cells, "y_k2")
    assert "U under gating (S2g over S1g)" in text
    assert "everything (S2g over S0)" in text
    # the one line that has a real delta must be the one that claims it
    rows = [ln for ln in text.splitlines() if "S2g over S1g" in ln]
    assert rows and "yes" in rows[0]


def test_the_equal_context_table_uses_the_c7_cells():
    """plan §3.3: the axis verdict must survive at one fixed Z context."""
    from spur_mpm.scripts.ladder_report import CONTROL_LEVELS, axis_comparison
    cells = {}
    for lvl, dice in zip(CONTROL_LEVELS, (0.55, 0.53, 0.58)):
        for arm in ("S0", "S1", "S2", "S1g", "S2g"):
            cells[(arm, lvl)] = {"dice": dice, "cl_dice": dice,
                                 "dice_area": dice, "cl_dice_area": dice}
    text = axis_comparison(cells, levels=CONTROL_LEVELS,
                           title="Axis at a matched 25 % budget, one "
                                 "seven-plane context (plan §3.3)")
    assert "one seven-plane context" in text
    for lvl in CONTROL_LEVELS:
        assert lvl in text
    # S2g is in this table; the non-forced one cannot show it (it has no
    # plain y_k4 cell), so the equal-context table is where its axis verdict lives
    assert "| S2g |" in text

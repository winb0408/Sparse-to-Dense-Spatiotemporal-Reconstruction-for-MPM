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

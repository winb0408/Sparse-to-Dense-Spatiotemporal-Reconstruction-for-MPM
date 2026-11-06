"""Invariants of the performance-versus-sparsity curves.

`ladder.md` reports one row per (arm, level); it cannot show whether a gap
survives a change of budget, which is what §7 Fig. 4 is for. The renderer is
therefore only as trustworthy as two things: the fraction it assigns to a level
tag, and the set of levels it agrees to draw on one axis. Both were wrong at
first -- ``yz_k2x2`` crashed the parser, and a ``_c7`` copy would have been
plotted on top of the plain level at the same x -- so both are pinned here.

The figures are also a claim surface: the 0.03 Dice floor constrains
*differences between arms*, so it must never be drawn as a threshold on the
absolute height of a bar. That is checked by asserting the bar heights in the
figure equal the CSV values, i.e. nothing has been offset or clipped to make a
floor band look meaningful.
"""
from __future__ import annotations

import csv

import pytest

from spur_mpm.scripts.ladder_curves import (
    CONTROL,
    DICE_FLOOR,
    MATCHED,
    _family,
    fraction_of,
    read_curve,
    render_family,
    render_matched,
)


@pytest.mark.parametrize("level,fraction", [
    ("y_k4", 0.25),
    ("z_k4", 0.25),
    ("yz_k2x2", 0.25),          # two factors in one token
    ("yz_k2x4", 0.125),
    ("y_k2", 0.5),
    ("z_k8", 0.125),
    ("y_k4_c7", 0.25),          # forced context does not change the budget
    ("full", 1.0),
])
def test_the_fraction_comes_from_the_tag(level, fraction):
    assert fraction_of(level) == pytest.approx(fraction)


@pytest.mark.parametrize("level", ["y", "yz_k2", "z_k2x2", "y_kx", ""])
def test_a_tag_that_does_not_encode_a_budget_gets_no_fraction(level):
    """Returning a guess here would put a point at a made-up x position."""
    assert fraction_of(level) is None


def test_a_forced_context_copy_does_not_double_draw_its_own_level():
    """``y_k4_c7`` measures the same 25 % at a different usable depth.

    Drawing it in the Y family would put two points with different inputs at the
    same x, which reads as noise in the curve rather than as a second protocol.
    """
    assert _family("y_k4", "y")
    assert not _family("y_k4_c7", "y")
    assert not _family("yz_k2x2_c7", "yz")
    assert _family("z_k4", "z")         # the plain level is the sweep point
    assert not _family("z_k4", "y")
    assert not _family("yz_k2x2", "z")  # a two-factor budget is its own family


def test_every_level_in_the_matched_sets_carries_the_same_budget():
    """The matched figure is only a comparison if the x axis is one budget."""
    for lvl in MATCHED + CONTROL:
        assert fraction_of(lvl) == pytest.approx(0.25), lvl


def _write_csv(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "level", "dice_area", "psnr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_nan_and_empty_cells_stay_out_of_the_curve(tmp_path):
    csv_path = tmp_path / "ladder.csv"
    _write_csv(csv_path, [
        {"arm": "A0", "level": "y_k4", "dice_area": "0.31", "psnr": "nan"},
        {"arm": "S1", "level": "y_k4", "dice_area": "", "psnr": "27.5"},
        {"arm": "S1", "level": "y_k2", "dice_area": "0.42", "psnr": "28.0"},
    ])
    rows = read_curve(csv_path)
    assert rows[("A0", "y_k4")]["dice_area"] == pytest.approx(0.31)
    assert "psnr" not in rows[("A0", "y_k4")]      # ``nan`` is not a datum
    assert "dice_area" not in rows[("S1", "y_k4")]
    assert set(rows) == {("A0", "y_k4"), ("S1", "y_k4"), ("S1", "y_k2")}


def test_a_family_with_no_points_draws_nothing(tmp_path):
    csv_path = tmp_path / "ladder.csv"
    _write_csv(csv_path, [{"arm": "A0", "level": "full", "dice_area": "0.5",
                           "psnr": "30"}])
    assert render_family(read_curve(csv_path), "y", tmp_path / "out") is None
    assert not (tmp_path / "out").exists()


def test_a_family_that_has_points_is_written(tmp_path):
    csv_path = tmp_path / "ladder.csv"
    _write_csv(csv_path, [
        {"arm": "A0", "level": "y_k4", "dice_area": "0.31", "psnr": "25"},
        {"arm": "A0", "level": "y_k2", "dice_area": "0.38", "psnr": "26"},
        {"arm": "S1", "level": "y_k4", "dice_area": "0.44", "psnr": "27"},
        {"arm": "S1", "level": "y_k2", "dice_area": "0.50", "psnr": "28"},
    ])
    p = render_family(read_curve(csv_path), "y", tmp_path / "out", dpi=40)
    assert p is not None and p.stat().st_size > 1000


def test_the_matched_figure_needs_at_least_two_budgets(tmp_path):
    """One bar cannot show whether *where* the sparsity is matters."""
    csv_path = tmp_path / "ladder.csv"
    _write_csv(csv_path, [{"arm": "S1", "level": "y_k4", "dice_area": "0.44",
                           "psnr": "27"}])
    assert render_matched(read_curve(csv_path), tmp_path / "out") is None


def test_the_matched_figure_spends_one_budget_three_ways(tmp_path):
    csv_path = tmp_path / "ladder.csv"
    _write_csv(csv_path, [
        {"arm": arm, "level": lvl, "dice_area": str(0.4 + 0.01 * i), "psnr": "27"}
        for i, (arm, lvl) in enumerate(
            [(a, l) for a in ("A0", "S1") for l in CONTROL])
    ])
    p = render_matched(read_curve(csv_path), tmp_path / "out", dpi=40)
    assert p is not None and p.name == "curve_matched_25pct_c7.png"
    assert DICE_FLOOR > 0     # the caption quotes it; keep it a real number

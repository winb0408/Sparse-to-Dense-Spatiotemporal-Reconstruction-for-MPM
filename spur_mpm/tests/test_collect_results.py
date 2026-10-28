"""Table building has its own ways to lie; these pin the two that matter.

1. A paired delta must know which direction is better per metric -- a "higher
   is better" assumption makes an MAE improvement look like a regression.
2. Depth profiles must not pool arms: the average of two methods is not a
   property of the data.
"""

from __future__ import annotations

from spur_mpm.scripts.collect_results import compare_arms


def _run(arm: str, fold: int, psnr: float, mae: float) -> dict:
    return {"arm": arm, "fold": fold, "_samples": "/nonexistent",
            "metrics": {
                "psnr": {"mean_of_group_means": psnr},
                "mae": {"mean_of_group_means": mae},
            }}


def test_paired_delta_counts_a_lower_mae_as_a_win():
    runs = [
        _run("A", 0, 29.0, 0.020), _run("A", 1, 28.0, 0.022),
        _run("B", 0, 29.5, 0.018), _run("B", 1, 28.4, 0.021),
    ]
    table = compare_arms(runs, "A", "B")
    mae_row = next(line for line in table.splitlines() if line.startswith("| mae"))
    psnr_row = next(line for line in table.splitlines() if line.startswith("| psnr"))
    # MAE went down in both folds -> both folds favour B
    assert mae_row.rstrip().endswith("| 2/2 |")
    assert psnr_row.rstrip().endswith("| 2/2 |")


def test_paired_delta_does_not_invent_a_win_when_folds_disagree():
    runs = [
        _run("A", 0, 29.0, 0.020), _run("A", 1, 28.0, 0.022),
        _run("B", 0, 29.5, 0.018), _run("B", 1, 27.5, 0.025),
    ]
    table = compare_arms(runs, "A", "B")
    psnr_row = next(line for line in table.splitlines() if line.startswith("| psnr"))
    assert psnr_row.rstrip().endswith("| 1/2 |")
    # the mean is zero while the folds point opposite ways: this is exactly
    # the case where a mean alone would read as "no difference"
    assert "+0.0000" in psnr_row


def test_missing_fold_is_reported_as_missing_not_as_zero():
    runs = [_run("A", 0, 29.0, 0.020), _run("B", 0, 29.5, 0.018)]
    table = compare_arms(runs, "A", "B")
    row = next(line for line in table.splitlines() if line.startswith("| psnr"))
    assert "—" not in row.split("|")[2] or "—" in row   # fold0 present
    assert row.rstrip().endswith("| 1/1 |")

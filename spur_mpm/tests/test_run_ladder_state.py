"""The ladder's progress file must never lose a finished cell.

A forced-context rerun (``--context-slices``) is a *different* cell with the
same ``(arm, level)``. Keying the progress file on ``(arm, level)`` made the
``_c7`` row replace the minimum-context row, after which ``done()`` could no
longer see that the earlier cell was finished and a later run retrained it.
"""
from __future__ import annotations

from spur_mpm.scripts.run_ladder import State


def _row(name: str, arm: str = "S1", level: str = "y_k4", status: str = "ok"):
    return {"arm": arm, "level": level, "name": name, "status": status}


def test_a_forced_context_cell_does_not_replace_its_sibling(tmp_path):
    path = tmp_path / "ladder.json"
    state = State(path)
    state.add(_row("E021_S1_y_k4"))
    state.add(_row("E021_S1_y_k4_c7"))

    assert state.done("S1", "y_k4", "E021_S1_y_k4") is not None
    assert state.done("S1", "y_k4", "E021_S1_y_k4_c7") is not None
    # both survive a reload: the file, not the object, is what the next run reads
    assert len(State(path).rows) == 2


def test_rerunning_the_same_cell_replaces_only_that_row(tmp_path):
    path = tmp_path / "ladder.json"
    state = State(path)
    state.add(_row("E021_S1_y_k4", status="train_failed"))
    state.add(_row("E021_S1_y_k4"))
    state.add(_row("E021_S1_z_k4", level="z_k4"))

    assert [r["status"] for r in State(path).rows
            if r["name"] == "E021_S1_y_k4"] == ["ok"]
    assert State(path).done("S1", "z_k4", "E021_S1_z_k4") is not None

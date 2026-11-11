"""A table column must name the run it came from, controls included.

E024 gave this repo a real wrong number: `P2_DualW`, `P2_DualW[ctrl=zeros]` and
`P2_DualW[ctrl=shuffle]` are three runs of the *same arm*, so `collect_results`
keyed them on ``arm``, put the first one it met in the headline column (the
shuffle control, 29.4031 instead of 29.4121) and reported a cross-animal mean
over one animal as ``n = 3``. A parameter-matched control is precisely the arm
whose number must never be confused with the headline, so these tests pin the
grouping key rather than any particular dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spur_mpm.scripts.collect_results import (
    _label, _pick, cross_animal_table, depth_table, load_runs, main_table,
)


def _index(root: Path) -> Path:
    """`depth_table` needs the voxel spacing; write the smallest index that has it."""
    path = root / "index.json"
    path.write_text(json.dumps({"volumes": [
        {"sample_id": "M2", "geometry": {"z_step_um": 1.0, "depth_origin_um": 0.0}}]}))
    return path


def _write(root: Path, name: str, *, arm: str = "P2_DualW", phys: str | None = "none",
           psnr: float = 29.4, dice: float = 0.734, fold: int = 0,
           n_windows: int = 2) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    summary = {
        "arm": arm, "split": "test", "fold": fold, "checkpoint": f"{name}/best.pt",
        "n_windows": n_windows,
        "metrics": {k: {"metric": k, "n_units": n_windows, "n_groups": 1,
                        "mean_of_group_means": v, "ci_estimable": False}
                    for k, v in {"psnr": psnr, "ssim": 0.76, "mae": 0.019,
                                 "e_measured": 0.0002, "e_missing": 0.0375,
                                 "dice": dice, "cl_dice": 0.729}.items()},
    }
    if phys is not None:
        summary["model_control"] = {"phys_control": phys, "consistency": "off"}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    (d / "samples.csv").write_text(
        "sample_id,query_z,query_y,query_x,psnr,dice,cl_dice\n"
        + "".join(f"M2,{z},0,0,30.0,{dice},0.72\n" for z in range(n_windows)))
    return d


def test_a_control_keeps_its_own_label():
    assert _label({"arm": "P2_DualW", "model_control": {"phys_control": "none"}}) \
        == "P2_DualW"
    assert _label({"arm": "P2_DualW",
                   "model_control": {"phys_control": "zeros"}}) \
        == "P2_DualW[ctrl=zeros]"


def test_a_summary_without_the_field_keeps_its_old_label():
    """Historical summaries predate `model_control`; their tables must not move."""
    assert _label({"arm": "S1"}) == "S1"


def test_the_headline_column_is_not_whichever_run_sorts_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "main", psnr=29.4121, dice=0.7343)
    _write(tmp_path, "shuffle", phys="shuffle", psnr=29.4031, dice=0.7370)
    _write(tmp_path, "zeros", phys="zeros", psnr=29.3209, dice=0.7356)
    runs = load_runs("*")
    table = main_table(runs)
    # three distinct columns, and the plain arm carries the plain run's value
    assert table.count("P2_DualW[ctrl=shuffle] fold0") == 1
    assert table.count("P2_DualW[ctrl=zeros] fold0") == 1
    row = next(line for line in table.splitlines() if line.startswith("| psnr "))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    header = [c.strip() for c in table.splitlines()[0].split("|")[1:-1]]
    assert cells[header.index("P2_DualW fold0")] == "29.4121"
    assert cells[header.index("P2_DualW[ctrl=shuffle] fold0")] == "29.4031"


def test_one_animal_is_not_counted_as_three(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "main", psnr=29.4121)
    _write(tmp_path, "shuffle", phys="shuffle", psnr=29.4031)
    _write(tmp_path, "zeros", phys="zeros", psnr=29.3209)
    table = cross_animal_table(load_runs("*"))
    rows = [line for line in table.splitlines() if "| psnr |" in line]
    assert len(rows) == 3
    assert all(line.split("|")[3].strip() == "1" for line in rows), rows


def test_depth_bands_are_not_pooled_across_controls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "main", dice=0.70)
    _write(tmp_path, "zeros", phys="zeros", dice=0.60)
    table = depth_table(load_runs("*"), _index(tmp_path), band_um=200)
    assert "### P2_DualW\n" in table
    assert "### P2_DualW[ctrl=zeros]\n" in table
    # each section keeps its own two windows rather than merging into four
    assert table.count("| 0-200 | 2 |") == 2


def test_an_ambiguous_label_is_refused_not_guessed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    """Two runs, same arm, neither recording its variant: the table cannot say
    which one a column is, so it must not pick one by directory order."""
    _write(tmp_path, "a", phys=None, psnr=29.4121)
    _write(tmp_path, "b", phys=None, psnr=29.4031)
    runs = load_runs("*")
    with pytest.raises(SystemExit) as exc:
        _pick(runs, "P2_DualW", 0)
    assert "model_control" in str(exc.value)

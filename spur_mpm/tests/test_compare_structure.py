"""A side-by-side skeleton table must not quietly compare two different things.

`structure_report.py` fits the vesselness threshold on the reference and gates
out planes where the reference mask fills more than 15 % of the plane, so two
arm summaries can differ in *what region* they scored. Concatenating two tables
hides that; these tests pin the three cases where a delta would be misleading,
plus the label provenance that made `P2_DualW`'s control columns wrong once
already.
"""

from __future__ import annotations

import json
from pathlib import Path

from spur_mpm.scripts.compare_structure import load, table


def _summary(root: Path, name: str, *, arm: str, volumes: dict[str, dict]) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    vols = []
    for sid, over in volumes.items():
        v = {"sample_id": sid, "n_planes": 100, "n_planes_gated": 20,
             "threshold": 0.04, "dice": 0.80, "cl_dice": 0.70, "f1": 0.65}
        v.update(over)
        vols.append(v)
    payload = {"arm": arm, "volumes": vols,
               "means": {"dice": sum(v["dice"] for v in vols) / len(vols),
                         "cl_dice": sum(v["cl_dice"] for v in vols) / len(vols),
                         "f1": sum(v["f1"] for v in vols) / len(vols)}}
    (d / "structure_summary.json").write_text(json.dumps(payload))
    return d


def test_the_delta_is_the_difference_not_the_value(tmp_path):
    a = _summary(tmp_path, "a", arm="P2_I0", volumes={"v1": {"dice": 0.80}})
    b = _summary(tmp_path, "b", arm="P2_DualW", volumes={"v1": {"dice": 0.83}})
    text = table(load([a, b]))
    row = next(line for line in text.splitlines() if line.startswith("| v1 "))
    assert row.split("|")[-4].strip() == "+0.0300"


def test_a_volume_missing_from_one_arm_is_named_not_skipped(tmp_path):
    """A mean over different volumes is not a comparison."""
    a = _summary(tmp_path, "a", arm="P2_I0",
                 volumes={"v1": {}, "v2": {"dice": 0.60}})
    b = _summary(tmp_path, "b", arm="P2_DualW", volumes={"v1": {}})
    text = table(load([a, b]))
    assert "Volume sets differ" in text
    assert "missing ['v2']" in text
    # and the row still shows the gap rather than silently dropping the volume
    row = next(line for line in text.splitlines() if line.startswith("| v2 "))
    assert "–" in row


def test_a_different_binarisation_threshold_is_called_out(tmp_path):
    """Each arm fits its own threshold; if they differ, part of Δ is the fit."""
    a = _summary(tmp_path, "a", arm="P2_I0", volumes={"v1": {"threshold": 0.04}})
    b = _summary(tmp_path, "b", arm="P2_DualW",
                 volumes={"v1": {"threshold": 0.04, "dice": 0.81}})
    assert "different thresholds" not in table(load([a, b]))
    # one volume disagreeing is enough -- an arm-level minimum would hide this
    b2 = _summary(tmp_path, "b2", arm="P2_DualW",
                  volumes={"v1": {"threshold": 0.041}})
    text = table(load([a, b2]))
    assert "different thresholds" in text
    assert "v1" in next(n for n in text.splitlines() if "different thresholds" in n)


def test_the_column_label_comes_from_the_summary_not_the_directory(tmp_path):
    """A renamed output directory must not relabel a column."""
    a = _summary(tmp_path, "P2_DualW_looks_like_this", arm="P2_DualW[ctrl=zeros]",
                 volumes={"v1": {}})
    assert "P2_DualW[ctrl=zeros] Dice" in table(load([a]))

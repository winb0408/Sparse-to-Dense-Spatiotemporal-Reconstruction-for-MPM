"""The cleanup tool must be unable to destroy evidence, whatever it is told.

Every test here is a refusal that has to happen *before* the disk is touched:
the cost of getting this wrong is a lost checkpoint or a deleted negative
result, which no later commit can restore. The tool's default is dry-run, and
the allowlist is the only thing that can turn a plan into a deletion.
"""
import json
import os

import pytest

from spur_mpm.scripts.audit_runs import (
    PROTECTED,
    SafetyRefusal,
    apply_plan,
    classify,
    inventory,
    plan,
    render_markdown,
)


def _fake_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "runs" / "E021_ladder" / "E021_S1_y_k2").mkdir(parents=True)
    (root / "runs" / "E021_ladder" / "E021_S1_y_k2" / "summary.json").write_text("{}")
    (root / "runs" / "pilot_S1_fold0").mkdir(parents=True)
    (root / "runs" / "pilot_S1_fold0" / "best.pt").write_bytes(b"x" * 16)
    (root / "runs" / "temp_run").mkdir(parents=True)
    (root / "runs" / "temp_run" / "log.csv").write_text("a\n")
    (root / "docs").mkdir()
    (root / "docs" / "RESULTS.md").write_text("see runs/E021_ladder/x\n")
    return root


def _payload(root, **kw):
    return plan(root, **kw)


def _allow(tmp_path, paths, name="allow.json"):
    p = tmp_path / name
    p.write_text(json.dumps(paths))
    return p


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def test_a_published_run_is_category_A_and_not_deletable():
    cat, action, _ = classify("runs/E021_ladder/E021_S1_y_k2", [])
    assert cat == "A" and action == "keep"


def test_the_pre_fix_cohort_is_C_and_kept_for_the_pending_rescore():
    cat, action, reason = classify("runs/E002_S1_fold0", [])
    assert cat == "C" and action == "keep-evidence"
    assert "B4" in reason


def test_a_referenced_smoke_run_is_downgraded_to_review():
    """README.md mentions runs/smoke: it cannot be a silent delete candidate."""
    cat, action, reason = classify("runs/smoke", ["README.md"])
    assert cat == "D" and action == "review" and "README.md" in reason


def test_an_unknown_directory_is_review_not_delete():
    cat, action, _ = classify("runs/something_new", [])
    assert cat == "review" and action == "review"


def test_inventory_records_what_a_manifest_needs(tmp_path):
    root = _fake_repo(tmp_path)
    entries = {e["path"]: e for e in inventory(root)}
    assert "runs/pilot_S1_fold0" in entries
    e = entries["runs/pilot_S1_fold0"]
    for field in ("path", "category", "reason", "run_id", "referenced_by",
                  "sha256", "bytes", "action", "archive_destination"):
        assert field in e, field
    assert e["bytes"] == 16 and e["file_count"] == 1
    assert entries["runs/E021_ladder"]["category"] == "A"


def test_the_same_tree_hashes_the_same(tmp_path):
    """A manifest is only useful if regenerating it reproduces the digest."""
    a = {e["path"]: e["sha256"] for e in inventory(_fake_repo(tmp_path / "a"))}
    b = {e["path"]: e["sha256"] for e in inventory(_fake_repo(tmp_path / "b"))}
    assert a == b and a["runs/E021_ladder"]


def test_the_manifest_lists_the_negative_results_as_kept(tmp_path):
    doc = render_markdown(_payload(_fake_repo(tmp_path)))
    assert "run `runs/E021_ladder`" not in doc        # it is a table row, not prose
    assert "| `runs/E021_ladder` | A | keep |" in doc
    assert "负结果" in doc


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_apply_is_a_dry_run_unless_asked(tmp_path):
    root = _fake_repo(tmp_path)
    payload = _payload(root)
    payload["entries"] = [dict(e, action="delete") for e in payload["entries"]
                          if e["path"] == "runs/temp_run"]
    allow = _allow(tmp_path, ["runs/temp_run"])
    removed = apply_plan(payload, allow, root, apply=False)
    assert removed == [str(root / "runs" / "temp_run")]
    assert (root / "runs" / "temp_run" / "log.csv").exists()


def test_apply_deletes_only_what_the_manifest_approved(tmp_path):
    root = _fake_repo(tmp_path)
    payload = _payload(root)
    payload["entries"] = [dict(e, action="delete", referenced_by=[])
                          for e in payload["entries"]
                          if e["path"] == "runs/temp_run"]
    allow = _allow(tmp_path, ["runs/temp_run"])
    removed = apply_plan(payload, allow, root, apply=True)
    assert removed and not (root / "runs" / "temp_run").exists()
    assert (root / "runs" / "pilot_S1_fold0" / "best.pt").exists()


@pytest.mark.parametrize("path", ["runs/../data_local/x", "./runs/temp_run",
                                  "/etc/passwd", "runs/temp_*", "runs/temp?run"])
def test_a_path_that_is_not_a_plain_relative_name_is_refused(tmp_path, path):
    root = _fake_repo(tmp_path)
    payload = _payload(root)
    allow = _allow(tmp_path, [path])
    with pytest.raises(SafetyRefusal):
        apply_plan(payload, allow, root, apply=False)


def test_a_protected_path_is_refused_even_if_the_allowlist_names_it(tmp_path):
    assert "runs/E021_ladder" in PROTECTED
    root = _fake_repo(tmp_path)
    payload = _payload(root)
    payload["entries"] = [dict(e, action="delete")
                          for e in payload["entries"]
                          if e["path"] == "runs/E021_ladder"]
    allow = _allow(tmp_path, ["runs/E021_ladder"])
    with pytest.raises(SafetyRefusal, match="protected"):
        apply_plan(payload, allow, root, apply=True)
    assert (root / "runs" / "E021_ladder" / "E021_S1_y_k2" / "summary.json").exists()


def test_a_symlink_pointing_out_of_the_tree_is_refused(tmp_path):
    root = _fake_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep me")
    link = root / "runs" / "linked_run"
    os.symlink(outside, link)
    payload = _payload(root)
    payload["entries"].append({"path": "runs/linked_run", "action": "delete",
                               "referenced_by": []})
    allow = _allow(tmp_path, ["runs/linked_run"])
    with pytest.raises(SafetyRefusal, match="symlink|escapes"):
        apply_plan(payload, allow, root, apply=True)
    assert (outside / "precious.txt").exists()


def test_a_path_absent_from_the_manifest_cannot_be_deleted(tmp_path):
    root = _fake_repo(tmp_path)
    payload = _payload(root)
    allow = _allow(tmp_path, ["runs/not_inventoried"])
    with pytest.raises(SafetyRefusal, match="not in the reviewed manifest"):
        apply_plan(payload, allow, root, apply=False)


def test_a_manifest_entry_marked_keep_cannot_be_deleted(tmp_path):
    root = _fake_repo(tmp_path)
    payload = _payload(root)              # pilot is the only delete candidate
    allow = _allow(tmp_path, ["runs/temp_run"])
    with pytest.raises(SafetyRefusal, match="not 'delete'"):
        apply_plan(payload, allow, root, apply=False)


def test_a_referenced_entry_is_refused_at_apply_time(tmp_path):
    """The plan and the manifest can disagree; the manifest is not re-trusted."""
    root = _fake_repo(tmp_path)
    payload = _payload(root)
    payload["entries"] = [dict(e, action="delete", referenced_by=["docs/RESULTS.md"])
                          for e in payload["entries"]
                          if e["path"] == "runs/temp_run"]
    allow = _allow(tmp_path, ["runs/temp_run"])
    with pytest.raises(SafetyRefusal, match="still referenced"):
        apply_plan(payload, allow, root, apply=False)


def test_apply_without_an_allowlist_file_refuses(tmp_path):
    root = _fake_repo(tmp_path)
    with pytest.raises(SafetyRefusal, match="allowlist not found"):
        apply_plan(_payload(root), tmp_path / "nope.json", root, apply=True)


def test_an_allowlist_entry_marked_keep_is_skipped_not_deleted(tmp_path):
    root = _fake_repo(tmp_path)
    payload = _payload(root)
    payload["entries"] = [dict(e, action="delete")
                          for e in payload["entries"]
                          if e["path"] == "runs/temp_run"]
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([{"path": "runs/temp_run", "action": "delete"},
                                 {"path": "runs/E021_ladder", "action": "keep"}]))
    removed = apply_plan(payload, allow, root, apply=True)
    assert removed == [str(root / "runs" / "temp_run")]
    assert (root / "runs" / "E021_ladder").exists()

"""Inventory ``runs/`` and propose a cleanup list -- never delete by default.

    python -m spur_mpm.scripts.audit_runs                    # dry run, writes nothing
    python -m spur_mpm.scripts.audit_runs --out-dir docs/archive
    python -m spur_mpm.scripts.audit_runs --apply \
        --allowlist docs/archive/cleanup_candidates.json

Why this exists
---------------
``runs/`` is git-ignored except for a few hundred force-added result files, so
"it is in .gitignore" says nothing about whether a directory is safe to remove
-- and some of what is in there is a **correct experiment with a negative
result** (E021's ablations) or the only copy of a checkpoint that a review item
still needs (E002/E009, for the re-score in PR #1 item B4). The plan (PR2 §9)
therefore asks for a classification pass first, and for every deletion to name
the rule that allowed it.

Categories (PR2 §9.1)::

    A  correct run, result supports the hypothesis      -> KEEP
    B  correct run, result does not support it          -> KEEP (evidence)
    C  protocol-invalid run                             -> KEEP evidence, may drop big files later
    D  smoke / temporary / regenerable                  -> candidate for deletion
    E  raw data, annotations, valid weights             -> NEVER touched

The reference scan is what makes the decision more than a name match: a run
that any tracked document, script or config mentions is downgraded to
``review`` no matter which category its name falls in -- including when the
mention is of a *parent* directory, because that is how it usually happens
(``README.md`` says ``runs/smoke``, not ``runs/smoke/smoke_SS1``).

Safety rules enforced by :func:`apply_plan`, all of which raise
:class:`SafetyRefusal` rather than skipping quietly:

* the path must resolve inside the repository root (a symlink out is refused);
* no ``..`` component, no glob metacharacter;
* the path must be inside ``runs/``;
* the protected prefixes are refused even if a hand-edited allowlist says otherwise;
* only ``action == "delete"`` entries may be removed;
* the allowlist must name the exact path (no prefix matching, no wildcards);
* apply requires the allowlist file that was reviewed, and re-checks references
  against the manifest before touching the disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "CATEGORIES",
    "POLICY",
    "PROTECTED",
    "SafetyRefusal",
    "apply_plan",
    "classify",
    "inventory",
    "main",
    "plan",
    "reference_index",
    "render_markdown",
]

#: Category -> (meaning, default disposition). PR2 §9.1, verbatim in intent.
CATEGORIES = {
    "A": ("correct run, result supports the claim", "keep"),
    "B": ("correct run, negative result -- evidence, keep it", "keep"),
    "C": ("protocol-invalid run (wrong checkpoint / split / axis)", "keep-evidence"),
    "D": ("smoke, temporary or regenerable artefact", "delete"),
    "E": ("raw data, annotations or valid weights", "protected"),
    #: Not a category of the plan; the honest label for "no rule covers this".
    "review": ("unclassified -- a human decides before anything is touched",
               "review"),
}

#: Explicit name rules, longest prefix wins. Anything unmatched is ``review``.
POLICY: tuple[tuple[str, str, str], ...] = (
    ("runs/E021_ladder", "A",
     "primary supervised ladder; its negative ablations are published evidence"),
    ("runs/E023_phase1", "A",
     "audit logs, re-scores and full-volume structure exports"),
    ("runs/E024_dual_prior", "A",
     "PR2 dual-prior ladder; its no-Wiener-gain result is the published "
     "evidence for plan SS8.2 (docs/PHASE1_PR2_E024_RESULTS.md)"),
    ("runs/E025_adaptive_dual_prior", "B",
     "PR3 y-only negative control; kept whatever sign it comes out with"),
    ("runs/E026_psras_condition", "B",
     "PR3 historical-condition re-check; its matched_synthetic null is evidence "
     "(docs/PHASE1_PR3_E026_RESULTS.md)"),
    ("runs/E001", "B",
     "parameter-free baselines on the old cohort; superseded, still correct"),
    ("runs/E002", "C",
     "pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 "
     "item B4 re-scores these checkpoints on the fixed code"),
    ("runs/E009", "C",
     "same cohort as E002 at k_y=4; kept for the same B4 comparison"),
    ("runs/pilot_S1_fold0", "D",
     "one-off pilot run before the E002 cohort existed"),
    ("runs/smoke", "D",
     "smoke-test run with random weights; runs/smoke/audit is the data audit"),
)

#: Never removable, whatever the allowlist says (PR2 §9.2 step 2).
PROTECTED: tuple[str, ...] = (
    "data_local",
    "runs/E021_ladder",
    "runs/E023_phase1",
    "docs/RESULTS.md",
    "docs/PHASE1_AUDIT_RESPONSE.md",
    "docs/archive",
    "spur_mpm",
)

#: Text files scanned for mentions of a run directory.
SCAN_SUFFIXES = (".md", ".py", ".sh", ".yaml", ".yml", ".json", ".csv", ".txt")
SCAN_SKIP_DIRS = {".git", "runs", "__pycache__", ".pytest_cache", "node_modules"}

#: Files bigger than this are hashed by name+size only; hashing tens of GB of
#: checkpoints would dominate the audit and buy nothing (the pointer is what
#: matters for a manifest).
HASH_LIMIT_BYTES = 64 * 1024 * 1024


class SafetyRefusal(RuntimeError):
    """A deletion was refused; the message names the rule that refused it."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> list[str]:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             text=True, check=False)
    except OSError:
        return []
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _sha256_file(path: Path) -> str | None:
    """Hash a file, or ``None`` when it is over :data:`HASH_LIMIT_BYTES`."""
    try:
        if path.stat().st_size > HASH_LIMIT_BYTES:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sha256_manifest(root: Path, files: list[Path]) -> str:
    """Digest of ``(relative path, size)`` for every file under a directory.

    A directory has no bytes of its own, so "the sha256 of this run directory"
    has to be defined; this is that definition, and it is reproducible without
    reading 4 GB. Per-file hashes are recorded separately for the small files.
    """
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: str(q)):
        rel = p.relative_to(root).as_posix()
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        h.update(f"{rel}\0{size}\n".encode())
    return h.hexdigest()


def _under(path: str, prefix: str) -> bool:
    """True when ``path`` is ``prefix`` or inside it (component-wise, not strncmp)."""
    p, q = path.split("/"), prefix.split("/")
    return p[:len(q)] == q


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


# ---------------------------------------------------------------------------
# reference scan
# ---------------------------------------------------------------------------

def reference_index(root: Path) -> dict[str, list[str]]:
    """``{run directory name: [tracked files that mention it]}``.

    Scans every tracked text file outside ``runs/``. A mention of a parent is
    credited to the child as well (see the module docstring), so a run is only
    "unreferenced" when nothing in the repository talks about it or its folder.
    """
    names = {p.name for p in (root / "runs").iterdir()} if (root / "runs").is_dir() else set()
    if not names:
        return {}
    hits: dict[str, list[str]] = {n: [] for n in names}
    for rel in _git(root, "ls-files") or []:
        if Path(rel).suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SCAN_SKIP_DIRS for part in Path(rel).parts):
            continue
        try:
            text = (root / rel).read_text(errors="ignore")
        except OSError:
            continue
        for name in names:
            marker = f"runs/{name}"
            # A file *inside* the run mentioning its own path is not evidence
            # that something else depends on it.
            if rel.startswith(f"runs/{name}/"):
                continue
            if marker in text:
                hits[name].append(rel)
    return hits


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def classify(relpath: str, referenced_by: list[str] | None = None) -> tuple[str, str, str]:
    """Return ``(category, action, reason)`` for one path under ``runs/``.

    ``action`` is ``keep`` | ``keep-evidence`` | ``review`` | ``delete``. Only
    ``delete`` may be applied, and only when nothing references the path.
    """
    referenced_by = list(referenced_by or [])
    category, reason = "review", "no rule covers this path; inspect before acting"
    best = ""
    for prefix, cat, why in POLICY:
        # String prefix, not path prefix: the run directories are named
        # ``runs/E002_S0_fold0``, not ``runs/E002/S0_fold0``.
        if relpath.startswith(prefix) and len(prefix) > len(best):
            best, category, reason = prefix, cat, why
    action = "review" if category == "review" else CATEGORIES[category][1]
    if referenced_by and action == "delete":
        action = "review"
        reason = (f"{reason}; but {len(referenced_by)} tracked file(s) refer to it, "
                  f"e.g. {referenced_by[0]}")
    return category, action, reason


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def _dir_entry(root: Path, path: Path, tracked: set[str]) -> dict:
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    sizes = {}
    for p in files:
        try:
            sizes[p] = p.stat().st_size
        except OSError:
            sizes[p] = 0
    small = [(p, h) for p in files[:400]
             if (h := _sha256_file(p)) is not None]
    rel = _rel(root, path)
    return {
        "path": rel,
        "kind": "dir",
        "run_id": path.name,
        "bytes": sum(sizes.values()),
        "file_count": len(files),
        "sha256": _sha256_manifest(root, files),
        "sha256_kind": f"manifest of (relpath,size); per-file hashes for the "
                       f"first {len(small)} files under {HASH_LIMIT_BYTES >> 20} MiB",
        "file_sha256": {_rel(root, p): h for p, h in small},
        "tracked_files": sum(1 for p in files if _rel(root, p) in tracked),
        "top_level": sorted({_rel(root, p).split("/")[2] for p in files
                             if len(_rel(root, p).split("/")) > 2})[:20],
    }


def inventory(root: Path, include_caches: bool = False) -> list[dict]:
    """One entry per run directory under ``runs/`` (plus caches if asked)."""
    runs = root / "runs"
    if not runs.is_dir():
        return []
    tracked = set(_git(root, "ls-files"))
    refs = reference_index(root)
    entries: list[dict] = []
    for path in sorted(p for p in runs.iterdir() if p.is_dir()):
        entry = _dir_entry(root, path, tracked)
        category, action, reason = classify(entry["path"], refs.get(path.name))
        entry.update(category=category,
                     category_meaning=CATEGORIES[category][0],
                     action=action, reason=reason,
                     referenced_by=refs.get(path.name, []),
                     archive_destination=None if action == "delete" else entry["path"])
        entries.append(entry)
    if include_caches:
        for pattern in ("__pycache__", ".pytest_cache"):
            for path in sorted(root.rglob(pattern)):
                if ".git" in path.parts or not path.is_dir():
                    continue
                if pattern == "__pycache__" and any(
                        p.name == "__pycache__" for p in path.relative_to(root).parents):
                    continue          # only the outermost one
                entries.append({
                    "path": _rel(root, path), "kind": "cache", "run_id": path.name,
                    "bytes": sum(f.stat().st_size for f in path.rglob("*")
                                 if f.is_file()),
                    "file_count": sum(1 for f in path.rglob("*") if f.is_file()),
                    "sha256": None, "sha256_kind": "regenerable cache; not hashed",
                    "file_sha256": {}, "tracked_files": 0, "top_level": [],
                    "category": "D", "category_meaning": CATEGORIES["D"][0],
                    "action": "review",
                    "reason": "regenerable Python cache; delete only via allowlist",
                    "referenced_by": [], "archive_destination": None,
                })
    return entries


def plan(root: Path, include_caches: bool = False) -> dict:
    entries = inventory(root, include_caches=include_caches)
    head = _git(root, "rev-parse", "HEAD")
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(root),
        "head_sha": head[0] if head else None,
        "branch": (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or [None])[0],
        "protected": list(PROTECTED),
        "policy": [{"prefix": p, "category": c, "reason": w} for p, c, w in POLICY],
        "categories": {k: {"meaning": v[0], "disposition": v[1]}
                       for k, v in CATEGORIES.items()},
        "totals": {
            "entries": len(entries),
            "bytes": sum(e["bytes"] for e in entries),
            "by_action": _counts(entries, "action"),
            "by_category": _counts(entries, "category"),
        },
        "entries": entries,
    }


def _counts(entries: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for e in entries:
        out[e[key]] = out.get(e[key], 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _gb(n: float) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    return f"{n / 1024 ** 2:.1f} MiB"


def render_markdown(payload: dict) -> str:
    lines = [
        "# 历史产物归档清单（ARCHIVE_MANIFEST_20261110）",
        "",
        "> 由 `python -m spur_mpm.scripts.audit_runs --out-dir docs/archive` 生成，**默认 dry-run**。",
        "> 本文件是只读盘点，不是删除记录；任何删除都要另一份 reviewed allowlist 和 "
        "`docs/archive/CLEANUP_REPORT_20261110.md`。",
        "",
        f"- 采集时间（UTC）：`{payload['generated_utc']}`",
        f"- 分支：`{payload.get('branch')}`",
        f"- HEAD：`{payload.get('head_sha')}`",
        f"- 条目：{payload['totals']['entries']}　合计 {_gb(payload['totals']['bytes'])}",
        f"- 处置分布：`{json.dumps(payload['totals']['by_action'], ensure_ascii=False)}`",
        f"- 类别分布：`{json.dumps(payload['totals']['by_category'], ensure_ascii=False)}`",
        "",
        "## 为什么这份清单里还有负结果",
        "",
        "类别 B/C 的目录**不因为结果不好或协议作废就删除**：它们是证伪记录，也是 PR #1 遗留",
        "复核项（B4 旧 checkpoint 重评）唯一能用的权重来源。清单只回答“这些文件是什么、",
        "谁在引用、能不能再生”，删不删是另一个决定。",
        "",
        "## 受保护路径（任何 allowlist 都改不动）",
        "",
    ]
    lines += [f"- `{p}`" for p in payload["protected"]]
    lines += ["", "## 分类规则", "",
              "| 前缀 | 类别 | 处置 | 依据 |", "|---|---|---|---|"]
    for row in payload["policy"]:
        lines.append(f"| `{row['prefix']}` | {row['category']} | "
                     f"{CATEGORIES[row['category']][1]} | {row['reason']} |")
    lines += ["", "## 逐条盘点", "",
              "| 路径 | 类别 | 处置 | 大小 | 文件数 | 已跟踪 | 被引用 | sha256（清单摘要） | 理由 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in payload["entries"]:
        sha = (e["sha256"] or "—")[:16]
        refs = len(e["referenced_by"])
        lines.append(
            f"| `{e['path']}` | {e['category']} | {e['action']} | {_gb(e['bytes'])} | "
            f"{e['file_count']} | {e['tracked_files']} | {refs} | `{sha}` | {e['reason']} |")
    lines += ["", "## 引用明细", ""]
    any_ref = False
    for e in payload["entries"]:
        if not e["referenced_by"]:
            continue
        any_ref = True
        lines.append(f"### `{e['path']}`")
        lines.append("")
        lines += [f"- `{r}`" for r in e["referenced_by"]]
        lines.append("")
    if not any_ref:
        lines += ["（无）", ""]
    lines += [
        "## 恢复方式",
        "",
        "本清单不含任何已被删除的**唯一**副本：可删除项（`action=delete`）都是可再生的",
        "smoke/pilot 产物，重建命令见各自 `config.yaml` 与实验脚本。E002/E009 的 checkpoint",
        "只在 PR #1 B4 重评完成后才可能降级，且届时会先写入归档索引再动盘。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# apply (the only function that deletes, and only with an allowlist)
# ---------------------------------------------------------------------------

def _check_path(relpath: str, root: Path, entries: dict[str, dict]) -> Path:
    """Return the absolute path, or raise :class:`SafetyRefusal`."""
    if not relpath or relpath.strip() != relpath:
        raise SafetyRefusal(f"empty or padded path: {relpath!r}")
    if any(ch in relpath for ch in "*?[]"):
        raise SafetyRefusal(f"wildcards are not allowed: {relpath!r}")
    parts = Path(relpath).parts
    if ".." in parts:
        raise SafetyRefusal(f"'..' component refused: {relpath!r}")
    if Path(relpath).is_absolute():
        raise SafetyRefusal(f"absolute path refused (must be repo-relative): {relpath!r}")
    if not _under(relpath, "runs") and "__pycache__" not in parts \
            and ".pytest_cache" not in parts:
        raise SafetyRefusal(f"outside runs/: {relpath!r}")
    for prefix in PROTECTED:
        if _under(relpath, prefix):
            raise SafetyRefusal(f"protected path: {relpath!r} is under {prefix!r}")
    target = (root / relpath)
    try:
        resolved = target.resolve()
        resolved.relative_to(root.resolve())
    except (ValueError, OSError) as exc:
        raise SafetyRefusal(f"path escapes the repository root: {relpath!r} ({exc})")
    if target.is_symlink():
        raise SafetyRefusal(f"symlink refused: {relpath!r}")
    entry = entries.get(relpath)
    if entry is None:
        raise SafetyRefusal(f"not in the reviewed manifest: {relpath!r}")
    if entry.get("action") != "delete":
        raise SafetyRefusal(f"manifest action is {entry.get('action')!r}, not 'delete': "
                            f"{relpath!r}")
    if entry.get("referenced_by"):
        raise SafetyRefusal(f"still referenced by {entry['referenced_by'][:3]}: "
                            f"{relpath!r}")
    return target


def _read_allowlist(allowlist_path: str | Path) -> list[str]:
    path = Path(allowlist_path)
    if not path.is_file():
        raise SafetyRefusal(f"allowlist not found: {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("allow", "allowed", "entries"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise SafetyRefusal(f"allowlist {path} has no 'allow' list")
    if not isinstance(payload, list):
        raise SafetyRefusal(f"allowlist {path} must be a list of repo-relative paths")
    out = []
    for item in payload:
        if isinstance(item, dict):
            if item.get("action", "delete") != "delete":
                continue
            item = item.get("path")
        if not isinstance(item, str):
            raise SafetyRefusal(f"allowlist entry is not a path: {item!r}")
        out.append(item)
    return out


def apply_plan(payload: dict, allowlist_path: str | Path, root: Path,
               apply: bool = False) -> list[str]:
    """Delete the allowlisted, manifest-approved entries. Dry run by default."""
    allowed = _read_allowlist(allowlist_path)
    entries = {e["path"]: e for e in payload.get("entries", [])}
    targets = [_check_path(p, root, entries) for p in allowed]
    if not apply:
        return [str(t) for t in targets]
    removed = []
    for target in targets:
        if target.is_dir():
            for f in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if f.is_dir() and not f.is_symlink():
                    f.rmdir()
                elif f.is_file() and not f.is_symlink():
                    f.unlink()
            target.rmdir()
        elif target.is_file():
            target.unlink()
        removed.append(str(target))
    return removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write ARCHIVE_MANIFEST_<date>.md and "
                        "cleanup_candidates.json here (default: print only)")
    p.add_argument("--date", default="20261110")
    p.add_argument("--include-caches", action="store_true",
                   help="also list __pycache__/.pytest_cache (never auto-deleted)")
    p.add_argument("--allowlist", type=Path, default=None)
    p.add_argument("--apply", action="store_true",
                   help="actually delete; requires --allowlist and a saved manifest")
    p.add_argument("--manifest", type=Path, default=None,
                   help="saved cleanup_candidates.json to apply against")
    args = p.parse_args(argv)

    root = args.root.resolve()
    payload = plan(root, include_caches=args.include_caches)

    if args.apply:
        if args.allowlist is None or args.manifest is None:
            raise SystemExit("--apply needs both --allowlist and --manifest; the "
                             "manifest is what proves each path was reviewed")
        saved = json.loads(args.manifest.read_text())
        doc = render_markdown(payload)
        if args.out_dir:
            (args.out_dir).mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"ARCHIVE_MANIFEST_{args.date}.md").write_text(doc)
            (args.out_dir / "cleanup_candidates.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False))
        removed = apply_plan(saved, args.allowlist, root)
        print(f"deleted {len(removed)} path(s):")
        for r in removed:
            print(f"  {r}")
        return 0

    if args.allowlist is not None:
        preview = apply_plan(payload, args.allowlist, root, apply=False)
        print(f"dry run: {len(preview)} path(s) would be deleted")
        for r in preview:
            print(f"  {r}")

    doc = render_markdown(payload)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"ARCHIVE_MANIFEST_{args.date}.md").write_text(doc)
        (args.out_dir / "cleanup_candidates.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote {args.out_dir / f'ARCHIVE_MANIFEST_{args.date}.md'}")
        print(f"wrote {args.out_dir / 'cleanup_candidates.json'}")
    else:
        print(doc)
    print(json.dumps(payload["totals"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

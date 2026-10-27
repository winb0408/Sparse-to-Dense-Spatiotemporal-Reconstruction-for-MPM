"""Grouped train/val/test splits with enforced leakage checks.

The statistical unit is the **animal** (or an independent FOV when no animal
identity exists) -- never a patch, a slice or a frame. Plan §2.3 requires
``leave-one-mouse-out`` here: with a handful of mice, random patch splitting
would report a random-split number as if it were generalisation.

Guarantees enforced on write and re-checked on load:

* no group appears in more than one of train/val/test;
* the same longitudinal session pair of one animal never straddles splits;
* every fold's test group is disjoint from its training groups.

A fold is a dict::

    {"fold": 0, "test": ["<sample_id>", ...], "val": [...], "train": [...],
     "groups": {"test": [...], "val": [...], "train": [...]}}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SplitConfig", "SplitError", "make_folds", "assert_no_leakage",
           "split_fingerprint", "write_split", "read_split", "group_key_of"]


class SplitError(ValueError):
    """Raised when a split would leak between train and test."""


@dataclass
class SplitConfig:
    """How to build folds.

    ``mode``
        ``lomo``  leave-one-group-out; one fold per group (the default and the
                  only mode that supports a generalisation claim with few animals).
        ``holdout`` a single split holding out ``test_groups`` groups.
    ``val_groups``
        How many of the remaining groups go to validation (they are never used
        for training). Must leave at least one training group.
    """

    group_field: str = "mouse_id"
    mode: str = "lomo"
    val_groups: int = 1
    seed: int = 0
    test_groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"group_field": self.group_field, "mode": self.mode,
                "val_groups": self.val_groups, "seed": self.seed,
                "test_groups": list(self.test_groups)}


def group_key_of(row: dict, cfg: SplitConfig) -> str:
    """Group identifier for one manifest row, with an explicit fallback order."""
    for key in (cfg.group_field, "mouse_id", "sample_id"):
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() not in ("nan", "none", "unknown", "?"):
            return f"{key}:{value}"
    raise SplitError(f"no usable grouping key in row {row.get('sample_id')!r}; "
                     "animal identity is required for a grouped split")


def make_folds(rows: list[dict], cfg: SplitConfig) -> list[dict]:
    """Build folds from manifest rows (one row per volume)."""
    if not rows:
        raise SplitError("no rows to split")

    by_group: dict[str, list[str]] = {}
    for row in rows:
        by_group.setdefault(group_key_of(row, cfg), []).append(str(row["sample_id"]))

    groups = sorted(by_group)
    if len(groups) < 3:
        raise SplitError(
            f"only {len(groups)} independent group(s) ({groups}). A grouped "
            "train/val/test split needs at least 3; patch-level random splits "
            "are not an acceptable substitute (plan §2.3). Supply more animals "
            "or an external test set before claiming generalisation.")

    if cfg.mode == "lomo":
        test_sets = [[g] for g in groups]
    elif cfg.mode == "holdout":
        if not cfg.test_groups:
            raise SplitError("holdout mode requires test_groups")
        missing = set(cfg.test_groups) - set(groups)
        if missing:
            raise SplitError(f"unknown test groups: {sorted(missing)}")
        test_sets = [list(cfg.test_groups)]
    else:
        raise SplitError(f"unknown split mode {cfg.mode!r}")

    folds: list[dict] = []
    for fi, test_groups in enumerate(test_sets):
        rest = [g for g in groups if g not in test_groups]
        if len(rest) < 2:
            raise SplitError(
                f"fold {fi}: {len(rest)} group(s) left after holding out "
                f"{test_groups}; need >=1 for train and >=1 for val")
        ordered = _shuffled(rest, cfg.seed + fi)
        n_val = max(1, min(cfg.val_groups, len(ordered) - 1))
        val_groups = ordered[:n_val]
        train_groups = ordered[n_val:]

        fold = {"fold": fi}
        for name, gs in (("test", test_groups), ("val", val_groups),
                         ("train", train_groups)):
            fold[name] = sorted(s for g in gs for s in by_group[g])
            fold.setdefault("groups", {})[name] = list(gs)
        assert_no_leakage(fold, rows, cfg)
        folds.append(fold)
    return folds


def _shuffled(items: list[str], seed: int) -> list[str]:
    import random
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def assert_no_leakage(fold: dict, rows: list[dict], cfg: SplitConfig) -> None:
    """Raise :class:`SplitError` if any independent unit crosses splits."""
    problems: list[str] = []

    sets = {name: set(fold.get(name, [])) for name in ("train", "val", "test")}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            problems.append(f"sample ids in both {a} and {b}: {sorted(overlap)[:5]}")

    row_by_id = {str(r["sample_id"]): r for r in rows}
    group_sets = {name: {group_key_of(row_by_id[s], cfg) for s in ids if s in row_by_id}
                  for name, ids in sets.items()}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        shared = group_sets[a] & group_sets[b]
        if shared:
            problems.append(f"groups in both {a} and {b}: {sorted(shared)}")

    # Longitudinal sessions of the same animal are inside a group, so a shared
    # fov_id across splits means the grouping field is wrong. Only comparable
    # when both identifiers are actually recorded: two blank mouse ids are not
    # evidence that the same physical FOV appears twice.
    def _fov_key(row: dict) -> tuple[str, str] | None:
        mouse = str(row.get("mouse_id", "") or "").strip()
        fov = str(row.get("fov_id", "") or "").strip()
        return (mouse, fov) if mouse and fov else None

    fovs: dict[str, set] = {name: set() for name in sets}
    for name, ids in sets.items():
        for sid in ids:
            row = row_by_id.get(sid)
            key = _fov_key(row) if row is not None else None
            if key is not None:
                fovs[name].add(key)
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        shared = fovs[a] & fovs[b]
        if shared:
            problems.append(f"(mouse, fov) pairs in both {a} and {b}: "
                            f"{sorted(shared)[:5]}")

    if not sets["train"]:
        problems.append("empty training split")
    if not sets["test"]:
        problems.append("empty test split")

    if problems:
        raise SplitError("split leakage detected: " + " | ".join(problems))


def split_fingerprint(folds: list[dict], cfg: SplitConfig) -> str:
    """Stable hash of the split definition (records into every run config)."""
    payload = json.dumps({"folds": folds, "config": cfg.to_dict()},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_split(folds: list[dict], cfg: SplitConfig, path: str | Path,
                rows: list[dict] | None = None) -> dict:
    """Write ``split.json``; returns the payload (includes the fingerprint)."""
    for fold in folds:
        if rows is not None:
            assert_no_leakage(fold, rows, cfg)
    payload = {"config": cfg.to_dict(), "folds": folds,
               "fingerprint": split_fingerprint(folds, cfg)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload


def read_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def folds_by_name(payload: dict) -> dict[int, dict]:
    return {int(f["fold"]): f for f in payload["folds"]}

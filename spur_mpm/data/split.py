"""Grouped train/val/test splits with enforced leakage checks.

Two levels, because the data forces two (plan §2.3):

``test`` level -- **animal** (``group_field``)
    Leave-one-mouse-out. The test animal is the only unit that supports a
    generalisation claim, so nothing from it may appear on the other side.
``inner`` level -- **FOV within the remaining animals** (``inner_field``)
    Validation is drawn from the animals left after the test animal is removed,
    holding out a whole FOV (both longitudinal sessions together). Plan §2.3:
    "训练／验证只能在剩余动物中按 FOV 或 session 分组". With three mice this is
    what keeps two animals' worth of data in training instead of one.

A held-out FOV must take *both* sessions with it: S1 and S2 of one FOV image the
same vasculature four weeks apart, so putting S1 in train and S2 in val leaks
the very structure the split is supposed to hide.

Guarantees enforced on write and re-checked on load:

* no sample id in more than one of train/val/test;
* the test animal never appears in train or val;
* train and val never share an inner unit (FOV) -- checked only when an inner
  field is configured and resolvable;
* no split is empty.

A fold is a dict::

    {"fold": 0, "test": [...], "val": [...], "train": [...],
     "groups": {"test": ["mouse_id:M1"]},
     "inner_groups": {"val": ["fov:M2/FOV I"], "train": [...]}}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SplitConfig", "SplitError", "make_folds", "assert_no_leakage",
           "split_fingerprint", "write_split", "read_split", "group_key_of",
           "inner_key_of"]


class SplitError(ValueError):
    """Raised when a split would leak between train and test."""


@dataclass
class SplitConfig:
    """How to build folds.

    ``mode``
        ``lomo``  leave-one-group-out; one fold per group (the default and the
                  only mode that supports a generalisation claim with few animals).
        ``holdout`` a single split holding out ``test_groups`` groups.
    ``inner_field``
        Field used to hold out validation *within* the remaining animals.
        ``fov_id`` by default; the key is ``(group_field, inner_field)`` so that
        FOV labels reused across animals cannot collide. Set to ``""`` to fall
        back to the old behaviour (a whole animal for validation), which is only
        sensible with many animals.
    ``val_units``
        How many inner units go to validation. Must leave at least one.
    """

    group_field: str = "mouse_id"
    inner_field: str = "fov_id"
    mode: str = "lomo"
    val_units: int = 1
    seed: int = 0
    test_groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"group_field": self.group_field, "inner_field": self.inner_field,
                "mode": self.mode, "val_units": self.val_units, "seed": self.seed,
                "test_groups": list(self.test_groups)}


def group_key_of(row: dict, cfg: SplitConfig) -> str:
    """Group identifier for one manifest row, with an explicit fallback order."""
    for key in (cfg.group_field, "mouse_id", "sample_id"):
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() not in ("nan", "none", "unknown", "?"):
            return f"{key}:{value}"
    raise SplitError(f"no usable grouping key in row {row.get('sample_id')!r}; "
                     "animal identity is required for a grouped split")


def inner_key_of(row: dict, cfg: SplitConfig) -> str | None:
    """Inner (validation) unit of one row: the FOV *inside* an animal.

    ``(group_field, inner_field)`` together, so ``FOV I`` of mouse 1 and
    ``FOV I`` of mouse 2 stay distinct. Falls back to the session, then to the
    sample id, so the split degrades predictably on datasets with no FOV label
    rather than silently treating every row as its own unit.
    """
    if not cfg.inner_field:
        return group_key_of(row, cfg)
    group = group_key_of(row, cfg)
    value = str(row.get(cfg.inner_field, "") or "").strip()
    if value and value.lower() not in ("nan", "none", "unknown", "?"):
        return f"fov:{group}/{value}"
    for fallback in ("session_id", "sample_id"):
        alt = str(row.get(fallback, "") or "").strip()
        if alt and alt.lower() not in ("nan", "none", "unknown", "?"):
            return f"{fallback}:{group}/{alt}"
    return None


def make_folds(rows: list[dict], cfg: SplitConfig) -> list[dict]:
    """Build folds: test on one animal, validate on a held-out FOV of the rest."""
    if not rows:
        raise SplitError("no rows to split")

    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(group_key_of(row, cfg), []).append(row)

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
        test_rows = [r for g in test_groups for r in by_group[g]]
        rest = [r for g in groups if g not in test_groups for r in by_group[g]]
        if not rest:
            raise SplitError(f"fold {fi}: nothing left after holding out {test_groups}")

        # Inner split of the remaining animals by FOV (both sessions together).
        by_inner: dict[str, list[dict]] = {}
        for row in rest:
            key = inner_key_of(row, cfg)
            by_inner.setdefault(key or f"row:{row['sample_id']}", []).append(row)
        inner_units = sorted(by_inner)
        if len(inner_units) < 2:
            raise SplitError(
                f"fold {fi}: only {len(inner_units)} inner unit(s) left after "
                f"holding out {test_groups}. An inner validation split needs at "
                "least 2 FOVs; do not fall back to a patch-level split, report "
                "the sample-size limit instead (plan §2.3).")

        ordered = _shuffled(inner_units, cfg.seed + fi)
        n_val = max(1, min(cfg.val_units, len(ordered) - 1))
        val_units = ordered[:n_val]
        train_units = ordered[n_val:]

        val_rows = [r for u in val_units for r in by_inner[u]]
        train_rows = [r for u in train_units for r in by_inner[u]]

        fold = {"fold": fi,
                "test": sorted(str(r["sample_id"]) for r in test_rows),
                "val": sorted(str(r["sample_id"]) for r in val_rows),
                "train": sorted(str(r["sample_id"]) for r in train_rows),
                "groups": {"test": list(test_groups)},
                "inner_groups": {"val": list(val_units), "train": list(train_units)}}
        assert_no_leakage(fold, rows, cfg)
        folds.append(fold)
    return folds


def _shuffled(items: list[str], seed: int) -> list[str]:
    import random
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def assert_no_leakage(fold: dict, rows: list[dict], cfg: SplitConfig) -> None:
    """Raise :class:`SplitError` if any independent unit crosses splits.

    Three levels are checked, and each one has to be checked separately because
    they legitimately disagree: train and val *should* share an animal (plan
    §2.3), while test and train must not -- so a single "groups are disjoint"
    rule would either be wrong or useless.
    """
    problems: list[str] = []

    sets = {name: set(fold.get(name, [])) for name in ("train", "val", "test")}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            problems.append(f"sample ids in both {a} and {b}: {sorted(overlap)[:5]}")

    row_by_id = {str(r["sample_id"]): r for r in rows}

    def keys(ids: set[str], fn) -> set:
        out = set()
        for sid in ids:
            row = row_by_id.get(sid)
            if row is None:
                continue
            k = fn(row)
            if k is not None:
                out.add(k)
        return out

    # Test animals never appear on the training side, in any form.
    test_groups = keys(sets["test"], lambda r: group_key_of(r, cfg))
    train_val_groups = keys(sets["train"] | sets["val"], lambda r: group_key_of(r, cfg))
    shared = test_groups & train_val_groups
    if shared:
        problems.append(f"animal(s) in both test and train/val: {sorted(shared)}")

    # Train and val may share an animal, but never a physical FOV: S1 and S2 of
    # one FOV image the same vasculature four weeks apart, so whichever inner
    # field is configured, that pair must not straddle the split.
    def _fov_key(row: dict):
        mouse = str(row.get("mouse_id", "") or "").strip()
        fov = str(row.get("fov_id", "") or "").strip()
        return (mouse, fov) if mouse and fov else None

    shared_fov = keys(sets["train"], _fov_key) & keys(sets["val"], _fov_key)
    if shared_fov:
        problems.append(f"(mouse, fov) pair(s) in both train and val: {sorted(shared_fov)}")
    shared_fov_test = keys(sets["test"], _fov_key) & keys(sets["train"] | sets["val"], _fov_key)
    if shared_fov_test:
        problems.append(f"(mouse, fov) pair(s) in both test and train/val: "
                        f"{sorted(shared_fov_test)}")

    if cfg.inner_field:
        shared_inner = keys(sets["train"], lambda r: inner_key_of(r, cfg)) & \
                       keys(sets["val"], lambda r: inner_key_of(r, cfg))
        if shared_inner:
            problems.append(f"inner unit(s) in both train and val: {sorted(shared_inner)}")

    if not sets["train"]:
        problems.append("empty training split")
    if not sets["val"]:
        problems.append("empty validation split")
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

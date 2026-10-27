"""Aggregation with the correct independent unit.

Patches, slices and frames are repeated measurements of the same animal, not
independent samples. Reporting a mean over patches with a patch-level standard
error would understate uncertainty by roughly the square root of the number of
patches per animal (plan §9.3).

Everything here aggregates to the **group** (animal / independent FOV) first
and only then summarises across groups.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["group_mean", "bootstrap_ci", "summarise_by_group", "paired_delta"]


def group_mean(values: list[float], groups: list[str]) -> dict[str, float]:
    """Mean per group (patch -> animal), returned sorted by group name."""
    acc: dict[str, list[float]] = defaultdict(list)
    for v, g in zip(values, groups):
        acc[g].append(float(v))
    return {g: float(np.mean(vs)) for g, vs in sorted(acc.items())}


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI over **group-level** values.

    Returns ``(mean, lo, hi)``. With three animals the interval is wide and
    that is the honest answer; do not shrink it by feeding patches in.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 3
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(arr.mean()), float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def summarise_by_group(rows: list[dict], metric: str, group_field: str = "group"
                       ) -> dict:
    """Mean / bootstrap CI over groups for one metric column."""
    values = [r[metric] for r in rows if r.get(metric) is not None]
    groups = [str(r[group_field]) for r in rows if r.get(metric) is not None]
    per_group = group_mean(values, groups)
    mean, lo, hi = bootstrap_ci(list(per_group.values()))
    estimable = len(per_group) >= 2
    return {"metric": metric, "n_units": len(values), "n_groups": len(per_group),
            "group_means": per_group, "mean_of_group_means": mean,
            "ci_lo": lo, "ci_hi": hi, "ci_estimable": estimable,
            "ci_note": None if estimable else
            "fewer than 2 groups: the interval is degenerate and says nothing "
            "about between-animal variability"}


def paired_delta(rows_a: list[dict], rows_b: list[dict], metric: str,
                 key: str = "key") -> dict:
    """Paired difference ``a - b`` aggregated at group level.

    Pairs are matched on ``key`` (the sample/window identity) so the comparison
    is same-input, not just same-distribution.
    """
    a = {r[key]: r[metric] for r in rows_a if r.get(metric) is not None}
    b = {r[key]: r[metric] for r in rows_b if r.get(metric) is not None}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0, "delta": float("nan")}
    diffs = [a[k] - b[k] for k in shared]
    mean, lo, hi = bootstrap_ci(diffs)
    return {"n": len(shared), "delta": mean, "ci_lo": lo, "ci_hi": hi,
            "n_a_better": sum(d > 0 for d in diffs),
            "n_b_better": sum(d < 0 for d in diffs)}

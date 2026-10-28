"""Grouped splits must never let one animal appear on both sides.

Two levels are tested separately because they legitimately differ (plan §2.3):
the test *animal* is isolated, while validation is a held-out *FOV* drawn from
the animals that remain.
"""

import pytest

from spur_mpm.data.split import (
    SplitConfig,
    SplitError,
    assert_no_leakage,
    inner_key_of,
    make_folds,
    split_fingerprint,
)


def rows(mice=3, fovs=2, sessions=1) -> list[dict]:
    out = []
    for m in range(1, mice + 1):
        for s in range(1, sessions + 1):
            for f in range(1, fovs + 1):
                out.append({"sample_id": f"M{m}_S{s}_FOV{f}", "mouse_id": f"M{m}",
                            "session_id": f"S{s}", "fov_id": f"FOV{f}"})
    return out


def test_leave_one_mouse_out_covers_every_animal_once():
    data = rows(mice=4)
    folds = make_folds(data, SplitConfig(mode="lomo"))
    assert len(folds) == 4
    tested = [f["groups"]["test"][0] for f in folds]
    assert sorted(tested) == ["mouse_id:M1", "mouse_id:M2", "mouse_id:M3", "mouse_id:M4"]
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert not set(f["val"]) & set(f["test"])


def test_validation_is_a_whole_fov_inside_the_remaining_animals():
    data = rows(mice=3, fovs=2, sessions=2)
    for fold in make_folds(data, SplitConfig(mode="lomo")):
        test_mice = {f"mouse_id:{r.split('_')[0]}" for r in fold["test"]}
        val_mice = {f"mouse_id:{r.split('_')[0]}" for r in fold["val"]}
        train_mice = {f"mouse_id:{r.split('_')[0]}" for r in fold["train"]}
        assert not test_mice & (val_mice | train_mice)     # animal level
        assert val_mice & train_mice                       # inner level shares an animal
        assert not set(fold["inner_groups"]["val"]) & set(fold["inner_groups"]["train"])
        # both sessions of a held-out FOV travel together
        val_fovs = {r.split("_")[0] + "_" + r.split("_")[2] for r in fold["val"]}
        assert len(val_fovs) == 1


def test_longitudinal_sessions_stay_inside_one_group():
    data = rows(mice=3, sessions=2)
    folds = make_folds(data, SplitConfig(mode="lomo"))
    for f in folds:
        train_mice = {r.split("_")[0] for r in f["train"]}
        test_mice = {r.split("_")[0] for r in f["test"]}
        assert not train_mice & test_mice


def test_session_cannot_be_split_from_its_twin_fov():
    """Holding out one session but not the other would leak the same vessels.

    Grouping by session is allowed by plan §2.3 for datasets where sessions are
    independent acquisitions; here they are the same FOV four weeks apart, so
    the (mouse, fov) guard has to catch it whatever the inner field says.
    """
    data = rows(mice=3, fovs=1, sessions=2)          # one FOV per mouse, two sessions
    with pytest.raises(SplitError, match="fov"):
        make_folds(data, SplitConfig(mode="lomo", inner_field="session_id"))


def test_two_animals_is_not_enough():
    with pytest.raises(SplitError, match="at least 3"):
        make_folds(rows(mice=2), SplitConfig(mode="lomo"))


def test_deliberate_leak_is_detected():
    data = rows(mice=3)
    cfg = SplitConfig(mode="holdout", test_groups=["mouse_id:M1"])
    fold = make_folds(data, cfg)[0]
    fold["train"] = fold["train"] + fold["test"][:1]        # inject the leak
    with pytest.raises(SplitError, match="leakage"):
        assert_no_leakage(fold, data, cfg)


def test_fov_moved_from_train_into_val_is_detected():
    data = rows(mice=3)
    cfg = SplitConfig(mode="holdout", test_groups=["mouse_id:M1"])
    fold = make_folds(data, cfg)[0]
    fold["val"] = fold["val"] + fold["train"][:1]
    with pytest.raises(SplitError, match="inner unit"):
        assert_no_leakage(fold, data, cfg)


def test_fingerprint_is_stable_and_sensitive():
    data = rows(mice=3)
    cfg = SplitConfig(mode="lomo", seed=0)
    a = split_fingerprint(make_folds(data, cfg), cfg)
    b = split_fingerprint(make_folds(data, cfg), cfg)
    assert a == b
    c = split_fingerprint(make_folds(data, SplitConfig(mode="lomo", seed=1)),
                          SplitConfig(mode="lomo", seed=1))
    assert a != c


def test_inner_key_keeps_same_fov_label_apart_across_mice():
    cfg = SplitConfig()
    a = inner_key_of({"sample_id": "a", "mouse_id": "M1", "fov_id": "I"}, cfg)
    b = inner_key_of({"sample_id": "b", "mouse_id": "M2", "fov_id": "I"}, cfg)
    assert a != b


def test_missing_group_field_falls_back_then_raises():
    data = [{"sample_id": "a", "mouse_id": ""}, {"sample_id": "b", "mouse_id": ""},
            {"sample_id": "c", "mouse_id": ""}]
    folds = make_folds(data, SplitConfig(mode="lomo"))
    assert len(folds) == 3          # falls back to sample_id, one row per group
    with pytest.raises(SplitError):
        make_folds([{"sample_id": ""}, {"sample_id": ""}, {"sample_id": ""}],
                   SplitConfig(mode="lomo"))

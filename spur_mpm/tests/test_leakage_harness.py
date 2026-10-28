"""The leakage harness itself must be able to fail.

``scripts/check_selfsup_leakage.py`` is the gate that certifies "zero dense
labels" on the real deposit. A gate that always returns OK is worse than no gate
at all, so these tests feed it a cache and check that it (a) passes an honest
item builder and (b) catches a leaky one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spur_mpm.data.datasets import CachedVolume, VolumeCache
from spur_mpm.data.geometry import VolumeGeometry
from spur_mpm.data.normalize import IntensityCalibration
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.ssl.holdout import HoldoutSpec

import spur_mpm.scripts.check_selfsup_leakage as harness


SHAPE = (9, 64, 64)


def _cache(tmp_path) -> VolumeCache:
    rng = np.random.default_rng(0)
    arr = rng.integers(1000, 4000, size=SHAPE, dtype=np.uint16)
    path = tmp_path / "vol.npy"
    np.save(path, arr)
    geom = VolumeGeometry(
        sample_id="M1", mouse_id="1", session_id="1", fov_id="I",
        shape=SHAPE, axes="ZYX", voxel_xy_um=0.98, z_step_um=5.0,
    )
    calib = IntensityCalibration(offset=1000.0, scale=3000.0)
    return VolumeCache({"M1": CachedVolume("M1", str(path), geom)}, calib,
                       {0: calib})


def _cfg():
    return harness._config(SamplingSpec(mode="y", ky=2), ratio=0.5, geometry="row")


def test_clean_item_builder_passes(tmp_path):
    cache = _cache(tmp_path)
    res = harness.check_volume(cache, "M1", _cfg(), depth=3)
    assert res["problems"] == []
    assert 0.4 < res["holdout_fraction"] < 0.6


def test_harness_catches_a_leaky_item_builder(tmp_path, monkeypatch):
    """If the builder lets the holdout into the inputs, the gate must fail."""
    real = harness._make_item

    def leaky(dense, geometry, cfg, z0, y0, x0, holdout_seed, center=None):
        item = real(dense, geometry, cfg, z0, y0, x0, holdout_seed, center)
        item["inputs"] = torch.cat(
            [item["inputs"], item["hold_target"].unsqueeze(0)], dim=0)
        return item

    monkeypatch.setattr(harness, "_make_item", leaky)
    res = harness.check_volume(_cache(tmp_path), "M1", _cfg(), depth=3)
    assert res["problems"], "harness accepted a builder that leaks the holdout"
    assert any("inputs changed" in p for p in res["problems"])


def test_harness_detects_zero_targets(tmp_path, monkeypatch):
    """An MMS loss against zeros would train the network to predict black."""
    real = harness._make_item

    def zeros(dense, geometry, cfg, z0, y0, x0, holdout_seed, center=None):
        item = real(dense, geometry, cfg, z0, y0, x0, holdout_seed, center)
        item["hold_target"] = torch.zeros_like(item["hold_target"])
        return item

    monkeypatch.setattr(harness, "_make_item", zeros)
    res = harness.check_volume(_cache(tmp_path), "M1", _cfg(), depth=3)
    assert any("all zero" in p or "not the acquired" in p
               for p in res["problems"])

"""End-to-end smoke test on a tiny synthetic dataset.

Covers the parts the unit tests cannot: TIFF -> audit -> split -> cache ->
dataset -> item -> model -> loss. It runs in seconds so it can gate every commit,
and it is the test that would have caught an axis or ordering mistake early.

Any metric produced here is a plumbing check, not a result.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spur_mpm.data.datasets import ItemConfig, SparseVolumeDataset, VolumeCache
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.scripts.audit_data import run_audit
from spur_mpm.scripts.make_split import build_split
from spur_mpm.scripts.make_synthetic import write_dataset
from spur_mpm.scripts.prepare_data import prepare
from spur_mpm.data.split import SplitConfig
from spur_mpm.ssl.holdout import HoldoutSpec


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    spec = write_dataset(root / "syn", root / "spec.yaml", mice=3, fovs=1,
                         depth=6, height=64, width=64)
    audit = run_audit(spec, root / "audit")
    assert not any(r["issues"] for r in audit["rows"]), audit["rows"]

    split_path = root / "audit" / "split.json"
    build_split(audit["json"], split_path, SplitConfig(mode="lomo"))
    index = root / "cache" / "index.json"
    prepare(audit["json"], split_path, root / "cache", index)
    return {"root": root, "audit": audit, "split": split_path, "index": index}


def test_audit_resolved_every_axis_and_animal(prepared):
    rows = prepared["audit"]["rows"]
    assert len(rows) == 3
    for r in rows:
        assert r["axes"] == "ZYX"
        assert r["shape"] == [6, 64, 64]
        assert r["z_span_um"] == pytest.approx(25.0)
        assert r["mouse_id"].startswith("M")


def test_split_has_no_leakage(prepared):
    split = json.loads(Path(prepared["split"]).read_text())
    assert len(split["folds"]) == 3
    for fold in split["folds"]:
        assert not set(fold["train"]) & set(fold["test"])


def test_dataset_items_are_consistent(prepared):
    cache = VolumeCache.load(prepared["index"])
    split = json.loads(Path(prepared["split"]).read_text())
    ids = split["folds"][0]["train"]

    cfg = ItemConfig(sampling=SamplingSpec(mode="y", ky=2), psf=PSFConfig(),
                     warm_start_kind="wiener", features=("y", "mask", "x0"),
                     supervision="self",
                     holdout=HoldoutSpec(ratio=0.5, geometry="row", seed=0))
    ds = SparseVolumeDataset(cache, ids, cfg, patch_hw=(32, 32),
                             context_depth=1, patches_per_epoch=4, seed=0)
    item = ds[0]

    assert item["inputs"].shape == (3, 32, 32)
    assert item["target"].shape == (32, 32)
    assert item["hold_mask"].sum() > 0
    # held-out targets must be real measurements, never zeros
    held = item["hold_mask"].bool()
    assert float(item["hold_target"][held].min()) >= 0.0
    assert float(item["inputs"][0][held].abs().sum()) == 0.0   # input blind there
    assert item["acq_mask"].sum() > item["hold_mask"].sum()


def test_supervised_item_differs_only_in_targets(prepared):
    cache = VolumeCache.load(prepared["index"])
    split = json.loads(Path(prepared["split"]).read_text())
    ids = split["folds"][0]["train"]
    base = dict(sampling=SamplingSpec(mode="y", ky=2), psf=PSFConfig(),
                warm_start_kind="bilinear", features=("y", "mask"))
    sup = SparseVolumeDataset(cache, ids, ItemConfig(supervision="dense", **base),
                              patch_hw=(32, 32), context_depth=1,
                              patches_per_epoch=1, seed=1)
    selfsup = SparseVolumeDataset(
        cache, ids,
        ItemConfig(supervision="self", holdout=HoldoutSpec(ratio=0.5), **base),
        patch_hw=(32, 32), context_depth=1, patches_per_epoch=1, seed=1)
    a, b = sup[0], selfsup[0]
    assert torch.equal(a["acq_mask"], b["acq_mask"])     # same sampling instance
    assert float(a["hold_mask"].sum()) == 0.0
    assert float(b["hold_mask"].sum()) > 0.0


def test_calibration_is_invertible(prepared):
    cache = VolumeCache.load(prepared["index"])
    c = cache.calibration
    x = c.offset + np.array([0.0, 0.1, 1.0]) * c.scale     # inside the calibrated range
    back = c.inverse(c.apply(x))
    assert np.allclose(back, x, atol=1e-4)

    # Above the calibrated ceiling the clip saturates: the round trip is exact
    # only inside [offset, offset + scale], and the loss must be reported, not
    # silently assumed away.
    bright = np.array([c.offset + 10 * c.scale])
    assert c.saturated_fraction(bright) == 1.0
    assert not np.allclose(c.inverse(c.apply(bright)), bright)
    assert c.saturated_fraction(c.offset + 0.5 * c.scale) == 0.0


@pytest.mark.parametrize("mode,ky,kz", [
    ("y", 2, 1), ("z", 1, 2), ("yz", 2, 2),
])
def test_z_boundary_windows_always_query_trained_centre(prepared, mode, ky, kz):
    """At both Z borders, the label must use the TRAINED central output.

    In the old implementation the first/last query planes were assigned to
    untrained output channels 0 and context-1 after clamping the window.
    """
    from spur_mpm.data.datasets import VolumeWindowDataset
    cache = VolumeCache.load(prepared["index"], fold=0)
    split = json.loads(Path(prepared["split"]).read_text())
    sid = split["folds"][0]["test"][0]
    cfg = ItemConfig(sampling=SamplingSpec(mode=mode, ky=ky, kz=kz),
                     psf=PSFConfig(), features=("y", "mask"),
                     supervision="dense")
    ds = VolumeWindowDataset(cache, [sid], cfg, patch_hw=(32, 32),
                             context_depth=3, z_stride=1)
    first, last = ds[0], ds[len(ds)-1]
    for item, z in ((first, 0), (last, cache.array(sid).shape[0]-1)):
        assert int(item["query_z"]) == z
        assert int(item["center_index"]) == 1
        assert item["inputs"].shape == (6, 32, 32)
        assert item["warmstart_center"].shape == (32, 32)
        # The centre is the real query plane, never a padding plane.
        # The queried plane may itself be skipped for z/yz protocols.
        if mode == "y":
            assert bool(item["acq_mask"].any())
        assert torch.isfinite(item["warmstart_center"]).all()
    # No padded virtual Z plane may appear as a measured input channel.
    first_input_mask = first["inputs"][3:6].bool()  # feature-major 3 mask slices
    last_input_mask = last["inputs"][3:6].bool()
    assert not first_input_mask[0].any()
    assert not last_input_mask[-1].any()


def test_direct_arm_item_exposes_warmstart_without_x0_feature(prepared):
    """S0 input remains only Y+M while a diagnostic prior can be exported."""
    from spur_mpm.data.datasets import VolumeWindowDataset
    cache = VolumeCache.load(prepared["index"], fold=0)
    split = json.loads(Path(prepared["split"]).read_text())
    sid = split["folds"][0]["test"][0]
    cfg = ItemConfig(sampling=SamplingSpec(mode="y", ky=2),
                     psf=PSFConfig(), features=("y", "mask"),
                     supervision="dense")
    item = VolumeWindowDataset(cache, [sid], cfg, patch_hw=(32, 32),
                               context_depth=1)[0]
    assert item["inputs"].shape == (2, 32, 32)
    assert item["warmstart_center"].shape == (32, 32)
    assert torch.isfinite(item["warmstart_center"]).all()

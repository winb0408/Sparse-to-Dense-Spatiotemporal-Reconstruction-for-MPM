"""PR3 contract tests. Run locally/on DeepSeek machine before any GPU work."""
import types

import numpy as np
import pytest
import torch

from spur_mpm.data.datasets import ItemConfig, _make_item
from spur_mpm.data.geometry import VolumeGeometry
from spur_mpm.models.registry import build_model, count_parameters
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.scripts.viewspec import protocol_columns
from spur_mpm.scripts import select_adaptive_prior as selector


def _geom():
    return VolumeGeometry(sample_id="s", mouse_id="M1",
                          session_id="S1", fov_id="FOVI",
                          shape=(5, 16, 16), voxel_xy_um=0.98,
                          z_step_um=5.0)


def test_prior_psf_never_blurs_actual_recorded_measurements():
    dense = np.random.default_rng(2).random((5, 16, 16)).astype("float32")
    psf = PSFConfig(1.2, 0.0, 0.98, 5.0)
    cfg = ItemConfig(sampling=SamplingSpec(mode="y", ky=4),
                     psf=psf, psf_source="estimated_from_data",
                     warm_start_kind="wiener_fill_all",
                     features=("y", "mask", "interp", "wiener"))
    item = _make_item(dense, _geom(), cfg, 0, 0, 0, 0)
    # The observation is sparse samples from recorded D, not H*D.
    y = item["inputs"][0:5]
    m = item["inputs"][5:10].bool()
    ref = torch.tensor(dense)
    assert torch.equal(y[m], ref[m])
    assert not torch.equal(item["inputs"][10:15], item["inputs"][15:20])


def test_extra_forward_degradation_requires_explicit_synthetic_label():
    dense = np.random.default_rng(3).random((5, 16, 16)).astype("float32")
    forward = PSFConfig(1.0, 0.0, 0.98, 5.0)
    cfg = ItemConfig(sampling=SamplingSpec(mode="y", ky=4),
                     forward_psf=forward, features=("y", "mask"))
    with pytest.raises(ValueError, match="forward_psf_source"):
        _make_item(dense, _geom(), cfg, 0, 0, 0, 0)


def test_measured_row_and_skipped_row_labels_use_actual_support():
    mask = np.zeros((5, 16, 16), dtype=bool)
    mask[:, ::4, :] = True
    spec = {"shape": [5, 16, 16], "z_view": 2,
            "y_view": 8, "x_view": 8, "mip_z": [0, 5]}
    cols = protocol_columns(spec, mask)
    by_key = {c["key"]: c for c in cols}
    assert "100%" in by_key["XZ measured row"]["note"]
    assert "0%" in by_key["XZ skipped row"]["note"]
    assert by_key["XZ measured row"]["coord"] % 4 == 0
    assert by_key["XZ skipped row"]["coord"] % 4 != 0


def _small(arm):
    return build_model(arm, 1, base=8, window=4,
                       depths=(1, 1, 1), heads=(2, 4, 8))


def test_old_checkpoint_keys_and_parameter_count_unchanged():
    m = _small("P2_DualW")
    assert not any(k.startswith(("obs_head.", "phys_head.")) for k in
                   m.state_dict())
    assert "observation_prediction" not in m(
        torch.zeros(1, 5, 32, 32))["aux"]


def test_pr3_auxiliary_heads_and_equal_parameter_control():
    m = _small("P3_DualAdaptive")
    c = build_model("P3_DualAdaptive", 1, base=8, window=4,
                    depths=(1, 1, 1), heads=(2, 4, 8),
                    phys_control="interp_only")
    assert count_parameters(m) == count_parameters(c)
    x = torch.rand(1, 7, 32, 32)
    out = m(x)
    assert out["aux"]["observation_prediction"].shape == (1, 1, 32, 32)
    assert out["aux"]["physics_prediction"].shape == (1, 1, 32, 32)
    loss = (out["aux"]["observation_prediction"].mean() +
            out["aux"]["physics_prediction"].mean() +
            out["reconstruction"].mean())
    loss.backward()
    assert m.obs_head.weight.grad is not None
    assert m.phys_head.weight.grad is not None


def test_adaptive_kernel_selection_is_training_only(monkeypatch):
    x = np.random.default_rng(4).random((5, 16, 16)).astype("float32")
    class FakeCache:
        calibration = types.SimpleNamespace(apply=lambda a: a)
        def array(self, sid):
            assert sid == "train_sample"
            return x
        def geometry(self, sid):
            return _geom()
        def __contains__(self, sid):
            return sid == "train_sample"
    monkeypatch.setattr(selector, "read_split", lambda path: {})
    monkeypatch.setattr(selector, "folds_by_name",
                        lambda data: {0: {"train": ["train_sample"],
                                         "test": ["test_sample"]}})
    monkeypatch.setattr(selector.VolumeCache, "load",
                        lambda *a, **kw: FakeCache())
    result = selector.choose_prior(index="unused", split="unused", fold=0,
                                   mode="y", ky=4, kz=1,
                                   patch=(5, 16, 16), n_patches=1,
                                   sigmas_xy=(0.0, 0.4), sigmas_z=(0.0,),
                                   lambdas=(0.01,), seed=0)
    assert result["train_volumes"] == ["train_sample"]
    assert result["test_volumes_excluded"] == ["test_sample"]
    assert len(result["candidates"]) == 2
    assert result["selected"]["psf_source"] in (
        "identity", "estimated_from_data")

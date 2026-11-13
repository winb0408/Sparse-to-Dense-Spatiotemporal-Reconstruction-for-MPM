"""PR3 contract tests. Run locally/on DeepSeek machine before any GPU work."""
import types

import numpy as np
import pytest
import torch

from spur_mpm.data.datasets import ItemConfig, _make_item
from spur_mpm.data.geometry import VolumeGeometry
from spur_mpm.models.registry import ARMS, build_model, count_parameters
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.scripts.viewspec import protocol_columns
from spur_mpm.scripts import select_adaptive_prior as selector


def _geom(shape=(5, 16, 16)):
    return VolumeGeometry(sample_id="s", mouse_id="M1",
                          session_id="S1", fov_id="FOVI",
                          shape=tuple(shape), voxel_xy_um=0.98,
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


def test_historical_lateral_then_axial_operator_differs_from_global_3d():
    """The old PSRAS YZ b4f was NOT one full 3-D Wiener on filled Z."""
    from spur_mpm.physics.acquisition import AcquisitionConfig, observe
    from spur_mpm.physics.wiener import wiener_fill_all, warm_start
    x = torch.rand((5, 24, 24), generator=torch.Generator().manual_seed(91))
    h = PSFConfig(sigma_xy_um=0.98, sigma_z_um=7.5,
                  voxel_xy_um=0.98, voxel_z_um=5.0)
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode="yz",
                                   ky=2, kz=2), psf=h))
    legacy = warm_start(obs.y, obs.mask, h, lam=0.01,
                        mode="wiener_fill_first")
    new3d = wiener_fill_all(obs.y, obs.mask, h, lam=0.01)
    assert torch.isfinite(legacy).all() and torch.isfinite(new3d).all()
    assert not torch.allclose(legacy, new3d, atol=1e-5, rtol=1e-4)


def test_matched_synthetic_yz_uses_known_h_once_and_target_remains_d():
    """Virtual known-H experiment is restorative Y=M*H(D), target D."""
    from spur_mpm.physics.acquisition import apply_otf
    dense = np.random.default_rng(51).random((5, 16, 16)).astype("float32")
    h = PSFConfig(sigma_xy_um=0.98, sigma_z_um=7.5,
                  voxel_xy_um=0.98, voxel_z_um=5.0)
    cfg = ItemConfig(sampling=SamplingSpec(mode="yz", ky=2, kz=2),
                     psf=h, psf_source="synthetic_extra_degradation",
                     forward_psf=h,
                     forward_psf_source="synthetic_extra_degradation",
                     warm_start_kind="wiener_fill_all",
                     features=("y", "mask", "interp", "wiener"))
    item = _make_item(dense, _geom(), cfg, 0, 0, 0, 1)
    y, mask = item["inputs"][0:5], item["inputs"][5:10].bool()
    assert torch.equal(item["target"], torch.from_numpy(dense[2]))
    ref = torch.from_numpy(dense)
    once = apply_otf(ref, h)
    assert torch.allclose(y[mask], once[mask], atol=1e-6)
    # Not D, and not H applied twice: a double blur is the failure mode this
    # test exists for, and it hides behind "y is blurred" unless checked.
    assert not torch.allclose(y[mask], ref[mask], atol=1e-4)
    assert not torch.allclose(y[mask], apply_otf(once, h)[mask], atol=1e-5)
    assert int(mask.sum()) > 0 and not bool(mask.all())


# ---------------------------------------------------------------------------
# PR3 §B.3: the prior kernel and the forward degradation are separate things
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ky", [4, 8])
def test_a_prior_kernel_never_changes_a_single_measured_value(ky):
    """Changing only the *prior* sigma must leave Y bit-identical.

    The failure this guards against is the one the historical PSRAS code could
    not have caught: one ``psf`` object used both to blur the target and to
    deconvolve the candidate. Here the two are separate config fields.
    """
    dense = np.random.default_rng(7).random((6, 32, 32)).astype("float32")
    base = dict(sampling=SamplingSpec(mode="y", ky=ky),
                warm_start_kind="wiener_fill_all",
                features=("y", "mask", "interp", "wiener"))
    a = _make_item(dense, _geom(dense.shape), ItemConfig(**base), 0, 0, 0, 0)
    b = _make_item(dense, _geom(dense.shape), ItemConfig(
        psf=PSFConfig(1.0, 0.0, 0.98, 5.0), psf_source="estimated_from_data",
        **base), 0, 0, 0, 0)
    assert torch.equal(a["inputs"][0:6], b["inputs"][0:6])       # Y unchanged
    assert torch.equal(a["inputs"][6:12], b["inputs"][6:12])     # M unchanged
    assert not torch.equal(a["inputs"][18:24], b["inputs"][18:24])  # W0 differs


def test_a_labelled_forward_degradation_does_change_the_measurement():
    dense = np.random.default_rng(8).random((6, 32, 32)).astype("float32")
    h = PSFConfig(1.0, 0.0, 0.98, 5.0)
    plain = _make_item(dense, _geom(dense.shape), ItemConfig(
        sampling=SamplingSpec(mode="y", ky=4), features=("y", "mask")),
        0, 0, 0, 0)
    blur = _make_item(dense, _geom(dense.shape), ItemConfig(
        sampling=SamplingSpec(mode="y", ky=4), forward_psf=h,
        forward_psf_source="synthetic_extra_degradation", features=("y", "mask")),
        0, 0, 0, 0)
    m = plain["inputs"][6:12].bool()
    assert torch.equal(plain["inputs"][0:6][m], torch.from_numpy(dense)[m])
    assert not torch.allclose(blur["inputs"][0:6][m], plain["inputs"][0:6][m],
                              atol=1e-4)
    # the sampling support itself is untouched: blur is not resampling
    assert torch.equal(blur["inputs"][6:12], plain["inputs"][6:12])


def test_every_entry_point_reads_the_same_forward_degradation_keys():
    """train / evaluate / inference / figures must agree on the acquisition.

    A fourth entry point that forgets these keys would reconstruct with a
    different forward model than it trained with, and nothing else would fail.
    """
    from pathlib import Path
    import spur_mpm
    root = Path(spur_mpm.__file__).parent
    for rel in ("train.py", "evaluate.py", "inference.py",
                "scripts/qualitative_figures.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "forward_extra_psf" in src, f"{rel} ignores the forward degradation"
        assert "forward_extra_psf_source" in src, f"{rel} drops the forward label"


# ---------------------------------------------------------------------------
# PR3 §B.2: archived PR2 checkpoints keep loading, PR3 ones carry their heads
# ---------------------------------------------------------------------------

def test_a_pr2_state_dict_round_trips_and_does_not_fit_the_pr3_model():
    p2 = _small("P2_DualW")
    sd = p2.state_dict()
    assert _small("P2_DualW").load_state_dict(sd, strict=True) is not None
    p3 = _small("P3_DualAdaptive")
    with pytest.raises(RuntimeError, match="obs_head"):
        p3.load_state_dict(sd, strict=True)
    # ...and the other direction too: a PR3 checkpoint is not a PR2 checkpoint.
    with pytest.raises(RuntimeError, match="obs_head"):
        p2.load_state_dict(p3.state_dict(), strict=True)


def test_the_pr3_heads_add_exactly_two_pointwise_projections():
    import dataclasses
    arm = ARMS["P3_DualAdaptive"]
    p3 = build_model(arm, 1, base=8, window=4, depths=(1, 1, 1), heads=(2, 4, 8))
    no_heads = build_model(dataclasses.replace(arm, branch_auxiliary=False),
                           1, base=8, window=4, depths=(1, 1, 1), heads=(2, 4, 8))
    d = count_parameters(p3) - count_parameters(no_heads)
    # one 1x1 projection per branch: base*(2**0) inputs -> context planes
    assert d == 2 * (8 * p3.t + p3.t)
    assert "obs_head.weight" in p3.state_dict()
    assert "phys_head.weight" in p3.state_dict()


# ---------------------------------------------------------------------------
# PR3 §B.6/B.8: the control differs only in the Wiener channels, and the
# YZ context=3 path trains
# ---------------------------------------------------------------------------

def _capture_phys_input(model, x):
    """Return what the physics branch was actually handed.

    The hook must return None: PyTorch replaces the module output with any
    non-None hook return, and a one-line ``lambda: seen.setdefault(...)``
    silently feeds a tensor where the model expects a list of features.
    """
    seen = {}

    def hook(module, inputs, output):
        seen["x"] = inputs[0].detach().clone()

    handle = model.phys.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()
    return seen["x"]


def test_interp_only_differs_from_none_only_in_the_wiener_and_delta_channels():
    feat = ARMS["P3_DualAdaptive"].features
    hw = 16
    torch.manual_seed(0)
    treat = build_model("P3_DualAdaptive", 1, base=4, window=2,
                        depths=(1, 1, 1), heads=(2, 4, 8))
    t = treat.t
    v = {f: torch.rand(1, t, hw, hw) for f in feat}
    v["phys_delta"] = v["wiener"] - v["interp"]
    x = torch.cat([v[f] for f in feat], dim=1)
    ctrl = build_model("P3_DualAdaptive", 1, base=4, window=2,
                       depths=(1, 1, 1), heads=(2, 4, 8), phys_control="interp_only")
    a = _capture_phys_input(treat, x)
    b = _capture_phys_input(ctrl, x)
    phys = ARMS["P3_DualAdaptive"].phys_features
    at = {f: a[:, i * t:(i + 1) * t] for i, f in enumerate(phys)}
    bt = {f: b[:, i * t:(i + 1) * t] for i, f in enumerate(phys)}
    assert torch.equal(at["interp"], bt["interp"])
    assert torch.equal(at["mask"], bt["mask"])
    # u_physics is a function of (M, H, lam), NOT of W0: the control keeps it,
    # otherwise the control would know strictly less than the treatment for a
    # reason unrelated to the deconvolved candidate.
    assert torch.equal(at["u_physics"], bt["u_physics"])
    assert not torch.equal(at["wiener"], bt["wiener"])
    assert torch.equal(bt["wiener"], bt["interp"])
    assert torch.equal(bt["phys_delta"], torch.zeros_like(bt["phys_delta"]))


@pytest.mark.parametrize("center", [0, 1, 2])
def test_a_yz_context_3_centre_phase_trains_end_to_end(center):
    """YZ k_z=2 needs T=3, and every centre phase must be trainable."""
    dense = np.random.default_rng(11).random((3, 32, 32)).astype("float32")
    geom = VolumeGeometry(sample_id="s", mouse_id="M1", session_id="S1",
                          fov_id="FOVI", shape=(3, 32, 32),
                          voxel_xy_um=0.98, z_step_um=5.0)
    cfg = ItemConfig(sampling=SamplingSpec(mode="yz", ky=2, kz=2),
                     features=ARMS["P3_DualAdaptive"].features,
                     warm_start_kind="wiener_fill_all", check_leakage=False)
    item = _make_item(dense, geom, cfg, 0, 0, 0, 0, center=center,
                      valid_z=np.ones(3, dtype=bool))
    x = item["inputs"].unsqueeze(0)
    assert x.shape[1] == len(cfg.features) * 3
    model = build_model("P3_DualAdaptive", 3, base=8, window=4,
                        depths=(1, 1, 1), heads=(2, 4, 8))
    out = model(x)
    aux = out["aux"]
    assert aux["observation_prediction"].shape == (1, 3, 32, 32)
    assert aux["physics_prediction"].shape == (1, 3, 32, 32)
    c = aux["observation_prediction"].mean() + aux["physics_prediction"].mean()
    (c + out["reconstruction"].mean()).backward()
    assert model.obs_head.weight.grad is not None
    grad = model.obs_head.weight.grad.abs().sum().item()
    assert grad > 0, "zero-initialised head received no gradient on the first step"


# ---------------------------------------------------------------------------
# PR3 §C: the historical operator is kept as its own arm, not renamed to W0
# ---------------------------------------------------------------------------

def test_the_legacy_operator_is_a_different_operator_under_axial_h():
    from spur_mpm.physics.acquisition import AcquisitionConfig, observe
    from spur_mpm.physics.wiener import warm_start, wiener_fill_all
    torch.manual_seed(3)
    x = torch.rand((5, 24, 24))
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode="yz", ky=2, kz=2),
                                       psf=PSFConfig()))
    ident = PSFConfig()
    legacy_id = warm_start(obs.y, obs.mask, ident, 0.01, mode="wiener_fill_first")
    full_id = wiener_fill_all(obs.y, obs.mask, ident, 0.01)
    assert torch.allclose(legacy_id, full_id, atol=1e-6)   # both are interp
    h = PSFConfig(0.98, 7.5, 0.98, 5.0)
    legacy = warm_start(obs.y, obs.mask, h, 0.01, mode="wiener_fill_first")
    full = wiener_fill_all(obs.y, obs.mask, h, 0.01)
    assert torch.isfinite(legacy).all() and torch.isfinite(full).all()
    assert not torch.allclose(legacy, full, atol=1e-3)
    assert (legacy - full).abs().mean() > 0.05 * full.abs().mean()


def test_the_legacy_arm_is_registered_as_its_own_operator():
    assert ARMS["P3_Wlegacy"].warm_start == "wiener_fill_first"
    assert ARMS["P2_W0"].warm_start == "wiener_fill_all"
    assert not ARMS["P3_Wlegacy"].needs_training
    assert ARMS["P3_Wlegacy"].family == "pr3"


def test_branch_auxiliary_weights_are_refused_on_an_arm_without_heads():
    from spur_mpm.models.registry import get_arm
    from spur_mpm.train import branch_aux_weights
    cfg = {"loss": {"branch_obs_weight": 0.05, "branch_phys_weight": 0.05}}
    with pytest.raises(ValueError, match="PR3 auxiliary arm"):
        branch_aux_weights(cfg, get_arm("P2_ResidualI"))
    assert branch_aux_weights(cfg, get_arm("P3_DualAdaptive")) == (0.05, 0.05)
    # zero weights stay legal everywhere: an E024-style arm reproduces its old
    # objective bit for bit instead of being forced into a new code path.
    zero = {"loss": {"branch_obs_weight": 0.0, "branch_phys_weight": 0.0}}
    assert branch_aux_weights(zero, get_arm("P2_ResidualI")) == (0.0, 0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        branch_aux_weights({"loss": {"branch_obs_weight": -1.0}},
                           get_arm("P3_DualAdaptive"))


def test_the_selection_mask_keeps_the_global_lattice_phase():
    """A crop starting at an odd row samples odd rows, not odd crop rows."""
    spec = SamplingSpec(mode="y", ky=4, kz=1)
    a = selector.protocol_mask((3, 8, 4), (0, 0, 0), spec)
    b = selector.protocol_mask((3, 8, 4), (0, 1, 0), spec)
    assert a[:, 0, 0].all() and not a[:, 1, 0].any()
    assert not b[:, 0, 0].any(), "phase must shift with the crop start"
    assert b[:, 3, 0].all()        # 3 + 1 = 4, on the global lattice
    zspec = SamplingSpec(mode="yz", ky=2, kz=2)
    m = selector.protocol_mask((5, 8, 4), (1, 0, 0), zspec)
    assert not m[0].any() and m[1].any()      # odd start => odd planes


def test_the_selection_reports_every_candidate_and_keeps_identity_legal(
        monkeypatch):
    x = np.random.default_rng(5).random((5, 16, 16)).astype("float32")

    class FakeCache:
        calibration = types.SimpleNamespace(apply=lambda a: a)
        def array(self, sid):
            return x
        def geometry(self, sid):
            return _geom()
        def __contains__(self, sid):
            return True

    monkeypatch.setattr(selector, "read_split", lambda path: {})
    monkeypatch.setattr(selector, "folds_by_name",
                        lambda data: {0: {"train": ["t"], "test": ["e"]}})
    monkeypatch.setattr(selector.VolumeCache, "load", lambda *a, **kw: FakeCache())
    res = selector.choose_prior(index="i", split="s", fold=0, mode="yz",
                                ky=2, kz=2, patch=(5, 16, 16), n_patches=2,
                                sigmas_xy=(0.0, 0.5), sigmas_z=(0.0, 2.5),
                                lambdas=(0.01, 0.05), seed=0)
    assert len(res["candidates"]) == 8          # full cross product, nothing hidden
    assert res["status"] == "selected_on_training_volumes_only"
    assert res["test_volumes_excluded"] == ["e"]
    assert all(c["n_patches"] == 2 for c in res["candidates"])
    ident = [c for c in res["candidates"]
             if c["sigma_xy_um"] == 0 and c["sigma_z_um"] == 0]
    assert ident and res["baseline_identity_mse"] == min(
        c["mse_missing"] for c in ident)
    # identity must be reachable, and a tie must not be broken toward a kernel
    with pytest.raises(ValueError, match="identity candidate mandatory"):
        selector.choose_prior(index="i", split="s", fold=0, mode="yz", ky=2, kz=2,
                              patch=(5, 16, 16), n_patches=1,
                              sigmas_xy=(0.5,), sigmas_z=(0.0,),
                              lambdas=(0.01,), seed=0)

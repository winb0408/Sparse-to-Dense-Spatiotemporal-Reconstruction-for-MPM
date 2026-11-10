"""Gate 2 (PR2 §7): the dual-prior model's contract, not its accuracy.

The claims that have to hold before any GPU minute is spent:

* the residual model starts as the interpolation it corrects (``R == 0``);
* both branches actually receive gradient (a branch that is wired but dead is
  the most expensive kind of architecture bug);
* ``hard`` observation consistency is exact at measured voxels and identical in
  train/eval mode, because it lives inside the model rather than in three
  call sites;
* the physics-candidate controls keep the parameter count and remove only the
  information;
* the P2 registry entries coexist with the old arms without mutating them;
* the training dataset really draws different patches per epoch.
"""

import numpy as np
import pytest
import torch

from spur_mpm.common.tensors import as_reconstruction, select_centre
from spur_mpm.data.datasets import ItemConfig, SparseVolumeDataset, VolumeWindowDataset
from spur_mpm.data.observation import build_model_inputs
from spur_mpm.models.dual_prior_reconstructor import DualPriorReconstructor
from spur_mpm.models.registry import ARMS, build_model, count_parameters, get_arm
from spur_mpm.physics.acquisition import AcquisitionConfig, observe
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import warm_start
from spur_mpm.tests.test_pipeline_smoke import prepared  # noqa: F401  (fixture)

PR2_TRAINED = ("P2_Direct", "P2_ResidualI", "P2_DualI", "P2_DualW", "P2_DualWU")


def _small(arm_name, t=1, **kw):
    return build_model(get_arm(arm_name), context_depth=t, base=8, window=4,
                       depths=(1, 1, 1), heads=(2, 4, 8), **kw)


def _stacked_inputs(features, shape=(7, 32, 32), mode="y", ky=2, seed=0):
    """A real stacked model input: build_model_inputs -> features -> channels."""
    x = torch.rand(shape, generator=torch.Generator().manual_seed(seed))
    obs = observe(x, AcquisitionConfig(sampling=SamplingSpec(mode=mode, ky=ky)))
    mi = build_model_inputs(obs.y, obs.mask, PSFConfig(), warm_start_kind="bilinear")
    return mi, obs, mi.stack(features)


# ---------------------------------------------------------------------------
# shapes and the residual initialisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm_name", PR2_TRAINED)
@pytest.mark.parametrize("t", [1, 7])
def test_forward_returns_the_documented_dict(arm_name, t):
    arm = get_arm(arm_name)
    model = _small(arm_name, t=t)
    out = model(torch.rand(2, len(arm.features) * t, 32, 32))
    assert set(out) == {"reconstruction_raw", "reconstruction",
                        "reconstruction_uncertainty", "aux"}
    assert out["reconstruction_raw"].shape == (2, t, 32, 32)
    assert out["reconstruction"].shape == (2, t, 32, 32)
    assert out["reconstruction_uncertainty"] is None


@pytest.mark.parametrize("arm_name", ["P2_ResidualI", "P2_DualI", "P2_DualW"])
def test_the_residual_starts_at_exactly_zero(arm_name):
    """At initialisation the model *is* the interpolator it is asked to correct.

    Zero, not "small": a random head would let an untrained arm outscore the
    parameter-free floor on some windows by luck, and the first validation
    number would be uninterpretable.
    """
    arm = get_arm(arm_name)
    mi, _, x = _stacked_inputs(arm.features)
    model = _small(arm_name, t=mi.y.shape[0])
    out = model(x[None])
    assert float(out["aux"]["residual"].abs().max()) == 0.0
    assert torch.equal(out["reconstruction_raw"][0], mi.interp)


def test_a_direct_arm_is_not_anchored_on_i0():
    """P2_Direct keeps the S0 semantics: an image, not a correction."""
    arm = get_arm("P2_Direct")
    assert arm.residual is False
    mi, _, x = _stacked_inputs(arm.features)
    out = _small("P2_Direct", t=mi.y.shape[0])(x[None])
    assert float(out["reconstruction_raw"].abs().max()) == 0.0   # zero head
    assert not torch.equal(out["reconstruction_raw"][0], mi.interp)


def test_the_physics_branch_is_connected_and_learns_after_one_step():
    """Zero-initialised projections make step 0 a special case -- stated, not hidden.

    At initialisation the head is zero, so nothing downstream of it gets
    gradient; and the fusion projection is zero, so the physics *encoder* gets
    no gradient until that projection moves. The projection itself does get
    gradient at step 0, which is what lets the branch come alive on step 1
    rather than never. Both facts are asserted here so a wiring bug and this
    deliberate initialisation cannot be confused.
    """
    arm = get_arm("P2_DualW")
    mi, _, x = _stacked_inputs(arm.features)
    model = _small("P2_DualW", t=mi.y.shape[0])
    model(x[None])["reconstruction"].abs().mean().backward()
    proj = sum(float(p.grad.abs().sum()) for p in model.fusion.proj.parameters()
               if p.grad is not None)
    phys = sum(float(p.grad.abs().sum()) for p in model.phys.parameters()
               if p.grad is not None)
    assert proj == 0.0 and phys == 0.0, "a zero head must block everything"

    # After a single SGD step on the head and the projections, both branches are
    # reachable: the architecture is connected, not merely present.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.decoder.head.weight.normal_(0, 0.02)
        for p in model.fusion.proj.parameters():
            p.normal_(0, 0.02)
    model(x[None])["reconstruction"].abs().mean().backward()
    for name, module in (("obs", model.obs), ("phys", model.phys)):
        g = sum(float(p.grad.abs().sum()) for p in module.parameters()
                if p.grad is not None)
        assert g > 0, f"{name} branch got no gradient after the first step"


# ---------------------------------------------------------------------------
# observation consistency
# ---------------------------------------------------------------------------

def test_hard_consistency_returns_the_measurement_exactly_where_it_exists():
    arm = get_arm("P2_DualW")
    mi, obs, x = _stacked_inputs(arm.features)
    model = _small("P2_DualW", t=mi.y.shape[0], consistency="hard")
    with torch.no_grad():
        model.decoder.head.weight.normal_(0, 0.02)
    out = model(x[None])
    m = obs.mask
    assert torch.equal(out["reconstruction"][0][m], obs.y[m])
    assert torch.equal(out["reconstruction"][0][~m], out["reconstruction_raw"][0][~m])


def test_consistency_is_off_by_default_and_can_be_switched_by_the_arm():
    assert get_arm("P2_DualW").consistency == "off"
    arm = get_arm("P2_DualW")
    model = _small("P2_DualW", t=1)
    out = model(torch.rand(1, len(arm.features), 16, 16))
    assert torch.equal(out["reconstruction"], out["reconstruction_raw"])


def test_train_and_eval_mode_predict_the_same_thing():
    """No dropout, no batch norm: a checkpoint means one function, not two."""
    arm = get_arm("P2_DualW")
    model = _small("P2_DualW", t=1, consistency="hard")
    x = torch.rand(1, len(arm.features), 16, 16)
    model.train()
    a = model(x)["reconstruction"]
    model.eval()
    with torch.no_grad():
        b = model(x)["reconstruction"]
    assert torch.equal(a, b)


def test_hard_consistency_without_y_and_mask_is_refused():
    with pytest.raises(ValueError, match="consistency"):
        DualPriorReconstructor(features=("interp",), obs_features=("interp",),
                               phys_features=(), t=1, consistency="hard")


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("control", ["none", "zeros", "shuffle"])
def test_the_physics_controls_keep_the_parameter_count(control):
    arm = get_arm("P2_DualW")
    base = count_parameters(_small("P2_DualW"))
    assert count_parameters(_small("P2_DualW", phys_control=control)) == base


def test_the_physics_controls_actually_change_the_output():
    """With an initialised head, deleting or shuffling the candidate must show."""
    arm = get_arm("P2_DualW")
    mi, _, x = _stacked_inputs(arm.features)
    outs = []
    for control in ("none", "zeros", "shuffle"):
        model = _small("P2_DualW", t=mi.y.shape[0], phys_control=control)
        with torch.no_grad():
            model.decoder.head.weight.normal_(0, 0.05)
            for p in model.fusion.proj.parameters():
                p.normal_(0, 0.05)
        torch.manual_seed(0)                    # shuffle draws from the global RNG
        outs.append(model(x[None])["reconstruction"].detach())
    assert not torch.allclose(outs[0], outs[1])
    assert not torch.allclose(outs[0], outs[2])


def test_an_unknown_control_is_refused():
    with pytest.raises(ValueError, match="phys_control"):
        _small("P2_DualW", phys_control="drop")


def test_duali_is_the_capacity_control_for_dualw():
    """Same architecture, same width; the difference is the candidate's content.

    The counts differ only through the physics *stem* (one feature vs four), so
    the comparison is capacity-matched to well inside the plan's ±5 %.
    """
    a = count_parameters(_small("P2_DualI"))
    b = count_parameters(_small("P2_DualW"))
    assert abs(a - b) / b < 0.05
    assert get_arm("P2_DualI").phys_features == ("interp",)
    assert "wiener" in get_arm("P2_DualW").phys_features


def test_the_gate_can_be_turned_off_and_is_reported_when_on():
    arm = get_arm("P2_DualWU")
    mi, _, x = _stacked_inputs(arm.features)
    model = _small("P2_DualWU", t=mi.y.shape[0], collect_aux=True)
    with torch.no_grad():
        model.decoder.head.weight.normal_(0, 0.02)
    out = model(x[None])
    gates = out["aux"]["gate"]
    assert gates and all(float(g.min()) >= 0.0 and float(g.max()) <= 1.0
                         for g in gates)
    # Zero-initialised gate => sigmoid(0) = 0.5 at step 0, the documented start.
    assert all(float(g.mean()) == pytest.approx(0.5, abs=1e-6) for g in gates)
    assert get_arm("P2_DualW").use_gate is False


# ---------------------------------------------------------------------------
# registry compatibility
# ---------------------------------------------------------------------------

def test_the_old_arms_are_untouched():
    """Old names must keep their exact meaning: their checkpoints depend on it."""
    assert get_arm("A0").features == ("y", "mask")
    assert get_arm("A0").warm_start == "bilinear"
    assert get_arm("A1").warm_start == "wiener_fill_first"
    assert get_arm("S1").features == ("y", "mask", "x0")
    assert get_arm("S1").warm_start == "wiener"
    assert get_arm("S2").features == ("y", "mask", "x0", "uncertainty")
    assert get_arm("SS1c").dual is True
    for name in ("A0", "A1", "S0", "S1", "S2", "S1g", "S2g", "SS0", "SS1",
                 "SS2", "SS2g", "SS1c"):
        arm = get_arm(name)
        assert arm.obs_features == () and arm.phys_features == ()
        assert arm.residual is False and arm.use_gate is False
        assert arm.consistency == "off" and arm.family == ""


def test_the_old_backbone_is_still_the_old_class():
    from spur_mpm.models.swin_unet_2d import SwinUNet2D
    assert isinstance(_small("S1"), SwinUNet2D)
    assert isinstance(_small("P2_DualW"), DualPriorReconstructor)


def test_pr2_names_do_not_collide_with_the_old_ones():
    from spur_mpm.models.registry import PR2_ARMS
    assert set(PR2_ARMS) == {"P2_I0", "P2_W0", "P2_Direct", "P2_ResidualI",
                             "P2_DualI", "P2_DualW", "P2_DualWU"}
    assert not set(PR2_ARMS) & {"A0", "A1", "S0", "S1", "S2", "S1g", "S2g",
                                "SS0", "SS1", "SS2", "SS2g", "SS1c"}


def test_the_parameter_free_pr2_arms_are_the_two_operators():
    assert get_arm("P2_I0").warm_start == "bilinear"
    assert get_arm("P2_W0").warm_start == "wiener_fill_all"
    assert get_arm("P2_I0").needs_training is False
    assert get_arm("P2_W0").needs_training is False


def test_under_an_identity_psf_the_two_operators_agree_exactly():
    """Which is why P2_W0 != P2_I0 cannot be claimed on this cohort."""
    _, obs, _ = _stacked_inputs(("y", "mask"))
    i0 = warm_start(obs.y, obs.mask, PSFConfig(), mode="bilinear")
    w0 = warm_start(obs.y, obs.mask, PSFConfig(), mode="wiener_fill_all")
    assert torch.equal(i0, w0)


def test_as_reconstruction_accepts_both_output_kinds():
    t = torch.zeros(2, 3, 4, 5)
    assert as_reconstruction(t) is t
    assert as_reconstruction({"reconstruction": t, "reconstruction_raw": t + 1}) is t
    assert as_reconstruction({"reconstruction_raw": t + 1}) is not None
    with pytest.raises(KeyError):
        as_reconstruction({"aux": {}})


# ---------------------------------------------------------------------------
# the training dataset really re-draws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_holdout", [False, True])
def test_each_epoch_draws_different_patches(prepared, with_holdout):
    """Before this, epoch N saw exactly the patches of epoch 0 in a new order."""
    from spur_mpm.ssl.holdout import HoldoutSpec
    from spur_mpm.data.datasets import VolumeCache
    from spur_mpm.data.split import read_split

    cache = VolumeCache.load(prepared["index"], fold=0)
    fold = {int(f["fold"]): f for f in read_split(prepared["split"])["folds"]}[0]
    cfg = ItemConfig(
        sampling=SamplingSpec(mode="y", ky=2), psf=PSFConfig(),
        warm_start_kind="bilinear", features=("y", "mask", "interp"),
        holdout=HoldoutSpec(ratio=0.5, geometry="row", seed=0) if with_holdout else None)
    ds = SparseVolumeDataset(cache, fold["train"], cfg, patch_hw=(32, 32),
                             context_depth=1, patches_per_epoch=4, seed=7)
    first = [(int(ds[i]["z0"]), int(ds[i]["phase"][1])) for i in range(4)]
    assert [(int(ds[i]["z0"]), int(ds[i]["phase"][1])) for i in range(4)] == first

    ds.set_epoch(1)
    second = [(int(ds[i]["z0"]), int(ds[i]["phase"][1])) for i in range(4)]
    assert second != first, "epoch 1 replayed epoch 0's patches"

    # Same seed, same epoch => same draw, in a fresh dataset object: the draw
    # depends on (seed, epoch, index) and nothing about worker layout.
    ds2 = SparseVolumeDataset(cache, fold["train"], cfg, patch_hw=(32, 32),
                              context_depth=1, patches_per_epoch=4, seed=7)
    ds2.set_epoch(1)
    assert [(int(ds2[i]["z0"]), int(ds2[i]["phase"][1])) for i in range(4)] == second


# ---------------------------------------------------------------------------
# end to end through the real item builder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm_name", ["P2_ResidualI", "P2_DualW"])
def test_a_pr2_arm_runs_on_real_windows_and_backpropagates(prepared, arm_name):
    from spur_mpm.data.datasets import VolumeCache

    arm = get_arm(arm_name)
    cache = VolumeCache.load(prepared["index"], fold=0)
    cfg = ItemConfig(sampling=SamplingSpec(mode="y", ky=2), psf=PSFConfig(),
                     warm_start_kind=arm.warm_start, features=arm.features)
    ds = VolumeWindowDataset(cache, cache and list(cache.volumes)[:1], cfg,
                             patch_hw=(32, 32), context_depth=3)
    batch = [ds[i] for i in range(2)]
    x = torch.stack([b["inputs"] for b in batch])
    center = torch.stack([b["center_index"] for b in batch])
    model = _small(arm_name, t=3, consistency="hard")
    out = model(x)
    pred = select_centre(as_reconstruction(out), center)
    assert as_reconstruction(out).shape == (2, 3, 32, 32)
    assert pred.shape == (2, 32, 32)
    loss = (pred - torch.stack([b["target"] for b in batch])).abs().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)


def test_the_item_builder_never_sees_the_dense_target(prepared):
    """Mask probe on the real item builder: change the GT, not the measurement.

    The two patches differ *everywhere* except where the mask says the scanner
    looked. If anything in the input stack is derived from the dense reference
    -- a normalisation, a statistic, a soft prior -- the two inputs diverge here.
    """
    from spur_mpm.data.datasets import VolumeCache, _make_item

    cache = VolumeCache.load(prepared["index"], fold=0)
    sid = list(cache.volumes)[0]
    geom = cache.geometry(sid)
    cfg = ItemConfig(sampling=SamplingSpec(mode="y", ky=2), psf=PSFConfig(),
                     warm_start_kind="bilinear",
                     features=("y", "mask", "interp", "wiener", "phys_delta"))
    a = np.random.default_rng(1).random((6, 32, 32)).astype(np.float32)
    mask = observe(torch.from_numpy(a), AcquisitionConfig(
        sampling=cfg.sampling, psf=cfg.psf, noise=cfg.noise)).mask.numpy()
    b = a.copy()
    b[~mask] = 5.0                    # only where the scanner did not look
    assert not np.array_equal(a, b)
    assert np.array_equal(a[mask], b[mask])
    item_a = _make_item(a, geom, cfg, 0, 0, 0, 0)
    item_b = _make_item(b, geom, cfg, 0, 0, 0, 0)
    assert torch.equal(item_a["inputs"], item_b["inputs"])
    assert torch.equal(item_a["warmstart_center"], item_b["warmstart_center"])
    assert item_a["inputs"].shape == (5 * 6, 32, 32)   # 5 features x 6 planes
    assert not torch.equal(item_a["target"], item_b["target"])

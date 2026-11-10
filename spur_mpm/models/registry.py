"""The ablation arm registry and model factory.

Every arm in plan §4.3 is one declarative record, so switching arm changes only
a config field -- never a separate copy of the data pipeline. That is what makes
the comparisons defensible: same data, same sampling instance, same backbone,
same budget; only the inputs / objective differ.

    A0   interpolation                      no training   sampling floor
    A1   fill-first Wiener                  no training   physics prior alone
    S0   Y + M                              supervised
    S1   Y + M + X0                         supervised
    S2   Y + M + X0 + U                     supervised
    SS0  Y + M                              self-supervised
    SS1  Y + M + X0                         self-supervised
    SS2  Y + M + X0 + U                     self-supervised   candidate main
    SS2g SS2 with gated prior fusion        self-supervised

``g`` suffixes add the gated fusion stem instead of plain concatenation. The
equal-parameter controls required by the plan (zero prior / random prior / plain
concat) are produced by overriding ``warm_start`` or ``fusion`` on a base arm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

__all__ = ["Arm", "ARMS", "get_arm", "build_model", "TRAINED_ARMS", "PR2_ARMS",
           "BASELINE_FEATURES", "eval_features"]

_BASE_FEATURES = ("y", "mask")


@dataclass(frozen=True)
class Arm:
    name: str
    features: tuple[str, ...]
    warm_start: str
    supervision: str          # dense | self | none
    fusion: bool = False
    needs_training: bool = True
    description: str = ""
    # Two branches on complementary halves of the acquisition (E012). Only the
    # network and the objective change; the data pipeline stays shared.
    dual: bool = False
    # --- PR2 (dual prior) fields; empty/False = legacy behaviour, so every old
    # arm record keeps its meaning and its checkpoints keep loading.
    #: Features the observation branch reads. Non-empty => the PR2 model.
    obs_features: tuple[str, ...] = ()
    #: Features the physics branch reads (empty => single-branch residual).
    phys_features: tuple[str, ...] = ()
    #: ``D_raw = I0 + R`` (True) or ``D_raw = R`` (False, S0 semantics).
    residual: bool = False
    #: Gate the physics branch by (mask, u_sampling, u_physics).
    use_gate: bool = False
    #: ``off`` | ``hard`` observation consistency, identical in train/eval/infer.
    consistency: str = "off"
    #: Which P2 experiment family this arm belongs to (provenance for figures).
    family: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "features": list(self.features),
                "warm_start": self.warm_start, "supervision": self.supervision,
                "fusion": self.fusion, "needs_training": self.needs_training,
                "description": self.description, "dual": self.dual,
                "obs_features": list(self.obs_features),
                "phys_features": list(self.phys_features),
                "residual": self.residual, "use_gate": self.use_gate,
                "consistency": self.consistency, "family": self.family}


ARMS: dict[str, Arm] = {
    "A0": Arm("A0", _BASE_FEATURES, "bilinear", "none", needs_training=False,
              description="bilinear/axial interpolation: the no-learning floor"),
    "A1": Arm("A1", _BASE_FEATURES, "wiener_fill_first", "none", needs_training=False,
              description="FillFirst-Wiener: physics prior with no network"),
    "S0": Arm("S0", _BASE_FEATURES, "bilinear", "dense",
              description="supervised direct baseline (Y,M only)"),
    "S1": Arm("S1", _BASE_FEATURES + ("x0",), "wiener", "dense",
              description="supervised + physics warm-start"),
    "S2": Arm("S2", _BASE_FEATURES + ("x0", "uncertainty"), "wiener", "dense",
              description="supervised + warm-start + U"),
    "SS0": Arm("SS0", _BASE_FEATURES, "bilinear", "self",
               description="self-supervised basic capacity"),
    "SS1": Arm("SS1", _BASE_FEATURES + ("x0",), "wiener", "self",
               description="self-supervised + physics warm-start"),
    "SS2": Arm("SS2", _BASE_FEATURES + ("x0", "uncertainty"), "wiener", "self",
               description="candidate main model: self-supervised + X0 + U"),
    "SS2g": Arm("SS2g", _BASE_FEATURES + ("x0", "uncertainty"), "wiener", "self",
                fusion=True, description="SS2 with gated prior fusion"),
    "S1g": Arm("S1g", _BASE_FEATURES + ("x0",), "wiener", "dense", fusion=True,
               description="supervised + warm-start, gated fusion"),
    "S2g": Arm("S2g", _BASE_FEATURES + ("x0", "uncertainty"),
                "wiener", "dense", fusion=True,
                description="supervised + warm-start + U + gated fusion"),
    # E012: the arm that answers "MMS only ever looks at half the acquisition".
    # Same features, warm start and objective as SS1 per branch; two branches on
    # disjoint halves, each scored on the other's real measurements.
    "SS1c": Arm("SS1c", _BASE_FEATURES + ("x0",), "wiener", "self", dual=True,
                description="dual-branch complementary MMS: each branch rebuilds "
                            "the rows the other one withheld"),

    # ------------------------------------------------------------------
    # PR2: observation branch + physics-prior branch (plan §6).
    #
    # These coexist with the arms above and change none of them. The names are
    # P2_* precisely so that a result can be traced to a method generation: an
    # S1 number and a P2_DualW number were produced by different code, on
    # different definitions of the prior, and are not interchangeable.
    # ------------------------------------------------------------------
    "P2_I0": Arm("P2_I0", _BASE_FEATURES, "bilinear", "none", needs_training=False,
                 family="pr2",
                 description="pure linear interpolation: I0 with no FFT and no "
                             "network (the reference re-computation of A0)"),
    "P2_W0": Arm("P2_W0", _BASE_FEATURES, "wiener_fill_all", "none",
                 needs_training=False, family="pr2",
                 description="fill-first Wiener W0 with no network; needs a "
                             "labelled H to mean anything beyond interpolation"),
    "P2_Direct": Arm("P2_Direct", _BASE_FEATURES, "bilinear", "dense",
                     obs_features=("y", "mask"), family="pr2",
                     description="Y+M -> image, no residual, no physics branch "
                                 "(the S0 semantics under the PR2 model)"),
    "P2_ResidualI": Arm("P2_ResidualI", _BASE_FEATURES + ("interp",), "bilinear",
                        "dense", obs_features=("y", "mask", "interp"),
                        residual=True, family="pr2",
                        description="D_raw = I0 + R(Y, M, I0): what the residual "
                                    "parameterisation alone buys"),
    "P2_DualI": Arm("P2_DualI", _BASE_FEATURES + ("interp",), "bilinear", "dense",
                    obs_features=("y", "mask"), phys_features=("interp",),
                    residual=True, family="pr2",
                    description="two branches, physics branch sees I0 only: the "
                                "capacity/structure control for P2_DualW"),
    "P2_DualW": Arm("P2_DualW",
                    _BASE_FEATURES + ("interp", "wiener", "phys_delta"),
                    "wiener_fill_all", "dense",
                    obs_features=("y", "mask"),
                    phys_features=("interp", "wiener", "phys_delta", "mask"),
                    residual=True, family="pr2",
                    description="two branches, physics branch sees I0, W0 and "
                                "W0-I0: does the Wiener candidate add anything "
                                "beyond I0 and capacity"),
    "P2_DualWU": Arm("P2_DualWU",
                     _BASE_FEATURES + ("interp", "wiener", "phys_delta",
                                       "u_sampling", "u_physics"),
                     "wiener_fill_all", "dense",
                     obs_features=("y", "mask", "u_sampling"),
                     phys_features=("interp", "wiener", "phys_delta", "mask",
                                    "u_physics"),
                     residual=True, use_gate=True, family="pr2",
                     description="P2_DualW with the fusion gated by M, u_sampling "
                                 "and u_physics (conditional: only meaningful "
                                 "once u_physics is not identically zero, i.e. "
                                 "under a non-identity H)"),
}

#: Arms whose numbers come from the PR2 model generation.
PR2_ARMS = tuple(a.name for a in ARMS.values() if a.family == "pr2")

TRAINED_ARMS = tuple(a.name for a in ARMS.values() if a.needs_training)
BASELINE_ARMS = tuple(a.name for a in ARMS.values() if not a.needs_training)


def get_arm(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[name]


def build_model(arm: Arm | str, context_depth: int, base: int = 48,
                window: int = 8, depths: tuple[int, ...] = (2, 2, 2),
                heads: tuple[int, ...] = (4, 8, 16),
                phys_control: str = "none",
                consistency: str | None = None,
                collect_aux: bool = False) -> torch.nn.Module:
    """Build the network for an arm.

    Inputs are ``len(features) * context_depth`` channels; output is
    ``context_depth`` channels (one per plane) and the caller selects the
    centre plane. With fusion on, the stem replaces the first conv and the
    backbone width stays at ``base``, so parameter counts stay comparable.

    ``phys_control`` (``none`` | ``zeros`` | ``shuffle``) builds the same
    parameter-matched PR2 model with the physics candidate's information
    destroyed; ``consistency`` overrides the arm's observation-consistency mode
    (it must be the same value in training, evaluation and inference -- the
    default reads it from the arm so there is one place to get it wrong).
    """
    from spur_mpm.models.prior_fusion import PriorFusionStem
    from spur_mpm.models.swin_unet_2d import SwinUNet2D

    if isinstance(arm, str):
        arm = get_arm(arm)
    if not arm.needs_training:
        raise ValueError(f"arm {arm.name} has no network; use the baseline path")
    if arm.obs_features:
        from spur_mpm.models.dual_prior_reconstructor import (
            DualPriorReconstructor,
        )
        return DualPriorReconstructor(
            features=tuple(arm.features), obs_features=tuple(arm.obs_features),
            phys_features=tuple(arm.phys_features), t=context_depth, base=base,
            window=window, depths=tuple(depths), heads=tuple(heads),
            residual=bool(arm.residual), use_gate=bool(arm.use_gate),
            consistency=arm.consistency if consistency is None else consistency,
            phys_control=phys_control, collect_aux=collect_aux)
    if arm.fusion and "x0" not in arm.features:
        raise ValueError(f"arm {arm.name}: fusion requires x0 in the feature set")
    if arm.dual:
        # Two full backbones, built from the same record with ``dual`` cleared.
        # Anything shared between the branches (features, warm start, fusion)
        # therefore stays identical by construction.
        from spur_mpm.models.complement import ComplementaryDual
        single = replace(arm, dual=False)
        return ComplementaryDual(
            build_model(single, context_depth, base=base, window=window,
                        depths=depths, heads=heads),
            build_model(single, context_depth, base=base, window=window,
                        depths=depths, heads=heads))

    n_feat = len(arm.features)
    in_ch = n_feat * context_depth
    unet_in = base if arm.fusion else in_ch
    unet = SwinUNet2D(unet_in, context_depth, base=base, window=window,
                      depths=depths, heads=heads)
    if arm.fusion:
        return torch.nn.Sequential(
            PriorFusionStem(arm.features, context_depth, base), unet)
    return unet


#: Channels a parameter-free arm is scored on. A0/A1 have no network, so their
#: prediction is the physics warm start itself; the item therefore has to carry
#: ``x0`` even though the arm's own ``features`` tuple does not list it.
BASELINE_FEATURES = ("y", "mask", "x0")


def eval_features(arm: Arm) -> tuple[str, ...]:
    """Features an arm needs to produce a prediction, trained or not.

    One definition, used by both ``evaluate.py`` and ``inference.py``: if the two
    disagreed, a full-volume figure could be drawn from a different input than
    the table that is supposed to describe it.
    """
    if arm.needs_training:
        return tuple(arm.features)
    return BASELINE_FEATURES


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def with_overrides(arm: Arm, **kwargs) -> Arm:
    """Copy of ``arm`` with fields replaced (equal-parameter control arms)."""
    return replace(arm, **kwargs)

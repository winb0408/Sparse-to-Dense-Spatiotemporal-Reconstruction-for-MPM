"""Observation branch + physics-prior branch, fused into a residual image.

    obs  : (Y, M[, u_sampling])          what the scanner actually measured
    phys : (I0, W0, W0-I0[, M, u_phys])  what the optical model suggests
    f_i  = E_obs,i(obs) + g_i . P_i(E_phys,i(phys))        per scale i
    R    = D(f_1..f_n)
    D_raw = I0 + R                        (or R, when residual=False)

Why a residual on ``I0`` and not on ``W0``: ``I0`` is the operator that is
always defined. Under the identity PSF used on this cohort ``W0 == I0`` exactly,
so anchoring on ``W0`` would make the architecture look like it depends on a
deconvolution that, on this data, does not exist. The network is asked to
predict a *correction* to interpolation, which is the quantity that is either
hard or easy and is comparable between arms that do and do not see ``W0``.

The head is zero-initialised, so at step 0 ``R == 0`` and ``D_raw == I0``
exactly; every arm that shares the decoder therefore starts from the same
predictor the parameter-free floor would produce. The fusion projections are
zero-initialised too, so the physics branch is admitted by learning rather than
by construction -- but note the gate variant starts at ``sigmoid(0) = 0.5``, not
0, because a gate that starts closed learns more slowly (see
``PriorFusionStem``, whose convention this matches).

``phys_control`` exists so "the physics candidate helped" can be separated from
"the second encoder helped": ``zeros`` keeps the architecture and the parameter
count and removes the information; ``shuffle`` keeps the marginal statistics and
removes the spatial correspondence. A gain that survives neither control came
from the information, not from capacity.

Nothing here is vessel-specific. The output is an image; segmentation,
skeletons and topology confidence are separate task adapters (PR2 §0, §5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spur_mpm.common.tensors import as_reconstruction
from spur_mpm.models.swin_unet_2d import _Down, _Stage, _Up

__all__ = ["DualPriorReconstructor", "PHYS_CONTROLS", "CONSISTENCY_MODES",
           "as_reconstruction"]

#: What to do to the physics-branch input, for the parameter-matched controls.
PHYS_CONTROLS = ("none", "zeros", "shuffle", "interp_only")
#: Observation consistency at the measured positions.
CONSISTENCY_MODES = ("off", "hard")


class _Branch(nn.Module):
    """Encoder stack returning one feature map per scale (shallowest first)."""

    def __init__(self, in_ch: int, base: int, window: int,
                 depths: tuple[int, ...], heads: tuple[int, ...]):
        super().__init__()
        n = len(depths)
        dims = [base * (2 ** i) for i in range(n)]
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.stages = nn.ModuleList(
            [_Stage(dims[i], heads[i], window, depths[i]) for i in range(n)])
        self.downs = nn.ModuleList([_Down(dims[i], dims[i + 1])
                                    for i in range(n - 1)])
        self.dims = dims

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return feats


class _Fusion(nn.Module):
    """Per-scale merge of the two branches, optionally gated.

    ``gate_cond`` is ``None`` for the ungated control, which must be as close to
    the gated version as possible in everything but the gate. Both variants
    start as "observation only follows the observation path": the projection is
    zero-initialised, so ``fused = f_obs`` at step 0 whether or not a gate
    exists.
    """

    def __init__(self, dims: list[int], cond_dim: int, use_gate: bool):
        super().__init__()
        self.use_gate = bool(use_gate)
        self.proj = nn.ModuleList([nn.Conv2d(d, d, 1) for d in dims])
        for p in self.proj:
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)
        if self.use_gate:
            self.gate = nn.ModuleList(
                [nn.Conv2d(2 * d + cond_dim, d, 3, padding=1) for d in dims])
            for g in self.gate:
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)

    def forward(self, f_obs: list[torch.Tensor], f_phys: list[torch.Tensor] | None,
                cond: torch.Tensor | None) -> tuple[list[torch.Tensor], list]:
        gates = []
        if f_phys is None:
            return f_obs, gates
        out = []
        for i, (fo, fp) in enumerate(zip(f_obs, f_phys)):
            p = self.proj[i](fp)
            if self.use_gate:
                c = cond
                if c is None:
                    c = torch.zeros_like(fo)
                if c.shape[-2:] != fo.shape[-2:]:
                    c = F.adaptive_avg_pool2d(c, fo.shape[-2:])
                g = torch.sigmoid(self.gate[i](torch.cat([fo, p, c], dim=1)))
                gates.append(g)
                out.append(fo + g * p)
            else:
                out.append(fo + p)
        return out, gates


class _Decoder(nn.Module):
    """Upsample-and-fuse decoder over the fused features."""

    def __init__(self, dims: list[int], window: int, depths: tuple[int, ...],
                 heads: tuple[int, ...], out_ch: int):
        super().__init__()
        n = len(dims)
        self.ups = nn.ModuleList([_Up(dims[n - 1 - i], dims[n - 2 - i])
                                  for i in range(n - 1)])
        self.stages = nn.ModuleList(
            [_Stage(dims[n - 2 - i], heads[n - 2 - i], window,
                    depths[n - 2 - i]) for i in range(n - 1)])
        self.head = nn.Conv2d(dims[0], out_ch, 3, padding=1)
        # Zero head => the residual is exactly zero at initialisation.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for i, up in enumerate(self.ups):
            x = up(x, feats[len(feats) - 2 - i])
            x = self.stages[i](x)
        return self.head(x)


class DualPriorReconstructor(nn.Module):
    """Two encoders on different views of the same observation.

    Parameters
    ----------
    features     : channel order the dataset stacks (``ItemConfig.features``).
    obs_features : feature names the observation branch reads.
    phys_features: feature names the physics branch reads; empty for the
                   single-branch residual arm.
    t            : number of stacked Z planes (channels per feature).
    residual     : ``D_raw = I0 + R`` when True, ``D_raw = R`` when False. The
                   latter is the "direct" arm (S0 semantics) and needs ``interp``
                   in ``features`` only if the caller wants it reported.
    consistency  : ``off`` | ``hard``; ``hard`` returns
                   ``reconstruction = M*Y + (1-M)*D_raw``, applied identically in
                   training, evaluation and inference because it lives here.
    phys_control : ``none`` | ``zeros`` | ``shuffle`` (parameter-matched control
                   arms; see the module docstring).
    """

    def __init__(self, features: tuple[str, ...], obs_features: tuple[str, ...],
                 phys_features: tuple[str, ...], t: int, base: int = 48,
                 window: int = 8, depths: tuple[int, ...] = (2, 2, 2),
                 heads: tuple[int, ...] = (4, 8, 16), residual: bool = True,
                 use_gate: bool = False, consistency: str = "off",
                 phys_control: str = "none", collect_aux: bool = False,
                 branch_auxiliary: bool = False):
        super().__init__()
        features = tuple(features)
        obs_features = tuple(obs_features)
        phys_features = tuple(phys_features)
        if not obs_features:
            raise ValueError("the observation branch needs at least one feature")
        missing = [f for f in (*obs_features, *phys_features) if f not in features]
        if missing:
            raise ValueError(f"branch features {missing} are not in the stacked "
                             f"feature set {features}")
        if phys_control not in PHYS_CONTROLS:
            raise ValueError(f"phys_control must be one of {PHYS_CONTROLS}, got "
                             f"{phys_control!r}")
        if consistency not in CONSISTENCY_MODES:
            raise ValueError(f"consistency must be one of {CONSISTENCY_MODES}, "
                             f"got {consistency!r}")
        if residual and "interp" not in features:
            raise ValueError("a residual model needs 'interp' in features: D_raw "
                             "= I0 + R")
        if use_gate and not phys_features:
            raise ValueError("a gate needs a physics branch to gate")
        if consistency == "hard" and not {"y", "mask"} <= set(features):
            raise ValueError("hard observation consistency needs 'y' and 'mask' "
                             "in features")

        self.features = features
        self.obs_features = obs_features
        self.phys_features = phys_features
        self.t = int(t)
        self.residual = bool(residual)
        self.consistency = consistency
        self.phys_control = phys_control
        self.collect_aux = bool(collect_aux)
        self.branch_auxiliary = bool(branch_auxiliary)
        self.idx = {f: i for i, f in enumerate(features)}

        dims = [base * (2 ** i) for i in range(len(depths))]
        self.obs = _Branch(len(obs_features) * self.t, base, window, depths, heads)
        self.phys = (_Branch(len(phys_features) * self.t, base, window, depths, heads)
                     if phys_features else None)
        cond_names = tuple(f for f in ("mask", "u_sampling", "u_physics")
                           if f in features)
        self.cond_names = cond_names
        cond_dim = len(cond_names) * self.t if cond_names else 0
        self.fusion = _Fusion(dims, cond_dim, use_gate)
        self.decoder = _Decoder(dims, window, depths, heads, out_ch=self.t)
        # PR3: lightweight per-branch diagnostics/auxiliary objectives. Each
        # branch can learn independently from the shared dense reference.
        # These heads NEVER receive GT as input and are disabled by default
        # for historical PR2 arms; they are read only by PR3 training.
        self.obs_head = (nn.Conv2d(dims[0], self.t, 1)
                         if self.branch_auxiliary else None)
        self.phys_head = (nn.Conv2d(dims[0], self.t, 1)
                          if self.phys is not None and self.branch_auxiliary else None)
        if self.obs_head is not None:
            nn.init.zeros_(self.obs_head.weight)
            nn.init.zeros_(self.obs_head.bias)
        if self.phys_head is not None:
            nn.init.zeros_(self.phys_head.weight)
            nn.init.zeros_(self.phys_head.bias)

    # -- input slicing ------------------------------------------------------
    def _split(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        n = len(self.features)
        if c != n * self.t:
            raise ValueError(f"expected {n * self.t} input channels "
                             f"({n} features x {self.t} planes), got {c}")
        v = x.view(b, n, self.t, h, w)
        return {f: v[:, i] for i, f in enumerate(self.features)}

    def _stack(self, v: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
        return torch.cat([v[f] for f in names], dim=1)

    def forward(self, x: torch.Tensor) -> dict:
        v = self._split(x)
        obs = self._stack(v, self.obs_features)

        # Parameter-matched control: the physics branch must see I0 where it
        # would have seen W0, and therefore a zero delta. It must NOT lose
        # anything else -- u_physics is a function of (M, H, lam), not of W0,
        # so zeroing it would hand the control less information than the
        # treatment for a reason that has nothing to do with the deconvolved
        # candidate, and would flatter the treatment.
        if self.phys_control == "interp_only":
            if "wiener" in v and "interp" in v:
                v["wiener"] = v["interp"]
            if "phys_delta" in v:
                v["phys_delta"] = torch.zeros_like(v["phys_delta"])
        phys = None
        if self.phys is not None:
            phys = self._stack(v, self.phys_features)
            if self.phys_control == "zeros":
                phys = torch.zeros_like(phys)
            elif self.phys_control == "shuffle":
                # Permute spatial positions within the batch element: keeps the
                # histogram, destroys the correspondence with the measurement.
                flat = phys.flatten(2)
                perm = torch.randperm(flat.shape[-1], device=phys.device)
                phys = flat[..., perm].view_as(phys)

        cond = (self._stack(v, self.cond_names) if self.cond_names else None)
        f_obs = self.obs(obs)
        f_phys = self.phys(phys) if phys is not None else None
        fused, gates = self.fusion(f_obs, f_phys, cond)
        r = self.decoder(fused)

        i0 = v["interp"] if "interp" in v else None
        d_raw = (i0 + r) if self.residual else r
        if self.consistency == "hard":
            m = v["mask"]
            d_hat = m * v["y"] + (1.0 - m) * d_raw
        else:
            d_hat = d_raw

        aux = {"residual": r}
        # The two independent views are supervised with small, optional losses
        # on the same dense target. Observation estimates I0 correction;
        # physics estimates a W0 correction. Neither branch is forced to copy
        # Wiener artifacts; final fused output remains the deployed output.
        if self.branch_auxiliary:
            aux["observation_prediction"] = (
                (i0 if i0 is not None else v["y"]) + self.obs_head(f_obs[0]))
            if self.phys_head is not None:
                w0 = v.get("wiener", i0)
                aux["physics_prediction"] = w0 + self.phys_head(f_phys[0])
        if i0 is not None:
            aux["interp"] = i0
        if self.collect_aux:
            aux["gate"] = [g.detach() for g in gates]
            if gates:
                aux["gate_mean"] = float(torch.stack(
                    [g.mean().detach() for g in gates]).mean())
        return {"reconstruction_raw": d_raw, "reconstruction": d_hat,
                "reconstruction_uncertainty": None, "aux": aux}

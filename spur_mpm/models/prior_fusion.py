"""Sampling-aware prior fusion stem.

Adapted from PSRAS/models/prior_fusion.py, generalised so the input feature set
is an arm parameter rather than a hard-coded 5-channel layout. The stem
separates three information streams and gates the physics prior by the sampling
condition::

    F_Y = E_Y(Y)                     measurement appearance
    F_W = E_W(X0)                    physics warm-start
    F_S = E_S([M, C_s, U, ...])      sampling condition
    G   = sigma(Conv([F_S, F_W]))    gate
    F_in = F_Y + G . F_W             -> UNet

Zero-initialising the gate's final conv gives ``G = sigmoid(0) = 0.5`` at step
0, not an exact identity: fusion starts by admitting half of ``E_W(X0)`` and
learns to amplify or suppress. Every fused arm shares this init so the
comparison between arms stays fair.

Why gate instead of concatenate: concatenation changes the number of input
channels, which changes parameter count and optimisation dynamics at the same
time as it adds information. A gated stem keeps the backbone width fixed, so
"fusion helps" can be separated from "more parameters help" (plan §4.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PriorFusionStem"]


class PriorFusionStem(nn.Module):
    """Map arm-selected input channels to one fused feature map for the UNet.

    Parameters
    ----------
    features : feature names, in the channel order the dataset stacks them.
    t        : number of Z planes stacked as channels.
    out_ch   : backbone width (UNet base).
    """

    def __init__(self, features: tuple[str, ...] | list[str], t: int, out_ch: int):
        super().__init__()
        features = tuple(features)
        if "y" not in features or "x0" not in features:
            raise ValueError(f"prior fusion needs both 'y' and 'x0', got {features}")
        self.features = features
        self.t = int(t)
        self.idx_y = features.index("y")
        self.idx_x0 = features.index("x0")
        self.cond = tuple(i for i, f in enumerate(features)
                          if i not in (self.idx_y, self.idx_x0))

        self.e_y = nn.Conv2d(self.t, out_ch, 3, padding=1)
        self.e_w = nn.Conv2d(self.t, out_ch, 3, padding=1)
        self.e_s = nn.Conv2d(self.t * max(1, len(self.cond)), out_ch, 3, padding=1)
        self.gate = nn.Conv2d(2 * out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F*t, H, W], feature-major (see ModelInputs.stack).
        b, _, h, w = x.shape
        v = x.view(b, len(self.features), self.t, h, w)

        f_y = self.e_y(v[:, self.idx_y].reshape(b, self.t, h, w))
        f_w = self.e_w(v[:, self.idx_x0].reshape(b, self.t, h, w))
        if self.cond:
            cond = v[:, list(self.cond)].reshape(b, self.t * len(self.cond), h, w)
        else:
            cond = torch.zeros(b, self.t, h, w, device=x.device, dtype=x.dtype)
        f_s = self.e_s(cond)

        g = torch.sigmoid(self.gate(torch.cat([f_s, f_w], dim=1)))
        return f_y + g * f_w

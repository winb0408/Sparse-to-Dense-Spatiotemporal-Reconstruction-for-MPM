"""Sampling-aware prior fusion stem (PLAN.md §5, B4).

Instead of concatenating every input channel into the first conv, the stem
separates three information streams and gates the physics prior by the
sampling condition:

    F_Y = E_Y(Y)                        measurement appearance
    F_W = E_W(X0)                       physics warm-start
    F_S = E_S([M, C_s, U_p])            sampling condition
    G   = sigma(Conv([F_S, F_W]))       gate
    F_in = F_Y + G . F_W                -> UNet

Zero-initialising the gate's final conv gives g = sigmoid(0) = 0.5 at
start (NOT exact identity -- DEC-013 note): the stem starts by admitting
half of e_w(X0) and the gate learns to amplify or suppress from there.
All fusion runs (B4/B5/B6) share this init, so comparisons are consistent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PriorFusionStem"]


class PriorFusionStem(nn.Module):
    """Maps raw input channels to a fused feature map for the UNet.

    Parameters
    ----------
    in_ch   : total input channels (dataset `features` x T slices).
    t       : number of slices stacked as channels.
    out_ch  : stem output channels (UNet base width).
    """

    def __init__(self, in_ch: int, t: int, out_ch: int):
        super().__init__()
        c = in_ch // t                       # channels per slice (5 = y,m,x0,c,u)
        assert c == 5, f"expected 5 features per slice, got {c}"
        self.t = t
        self.e_y = nn.Conv2d(t * 1, out_ch, 3, padding=1)      # Y channels (1/slice)
        self.e_w = nn.Conv2d(t * 1, out_ch, 3, padding=1)      # X0 channels
        self.e_s = nn.Conv2d(t * 3, out_ch, 3, padding=1)      # M, C_s, U_p
        self.gate = nn.Conv2d(2 * out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, t*5, H, W].  Dataset layout is FEATURE-major (DEC-013): the
        # channels are [y(t0..t), m(t0..t), x0(t0..t), c_s(t0..t), u_p(t0..t)],
        # so the view splits on features first, time second.
        B, _, H, W = x.shape
        v = x.view(B, 5, self.t, H, W)
        y = v[:, 0]
        x0 = v[:, 2]
        cond = v[:, [1, 3, 4]].reshape(B, self.t * 3, H, W)
        f_y, f_w, f_s = self.e_y(y), self.e_w(x0), self.e_s(cond)
        g = torch.sigmoid(self.gate(torch.cat([f_s, f_w], dim=1)))
        return f_y + g * f_w

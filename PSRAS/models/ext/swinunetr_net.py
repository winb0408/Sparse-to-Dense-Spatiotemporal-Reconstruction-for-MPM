"""EXP-020 sunetr adapter: official MONAI SwinUNETR (author-upstreamed).

Pre-registered DEC-023 adaptation:
- input per-frame channels (B, F, H, W, T) in MONAI's (B,C,H,W,D) layout,
  in_chans=F, out_chans=T; plane j read on the (channel, depth) diagonal.
- T padded to a multiple of 16 (patch 2 x 4 pool stages) when 3D.
- 2D situation (T=1) uses spatial_dims=2 with in_chans=F, out_chans=1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from monai.networks.nets import SwinUNETR


class SunetrNet(nn.Module):
    def __init__(self, n_feats: int, t: int, feature_size: int = 48):
        super().__init__()
        self.n_feats, self.t = n_feats, t
        self.spatial3d = t > 1
        if self.spatial3d:
            # img_size must itself pass MONAI's %32 check -> use padded depth
            self.net = SwinUNETR(img_size=(256, 256, (t + 31) // 32 * 32),
                                 in_channels=n_feats, out_channels=t,
                                 feature_size=feature_size, spatial_dims=3)
        else:
            self.net = SwinUNETR(img_size=(256, 256), in_channels=n_feats,
                                 out_channels=1, feature_size=feature_size,
                                 spatial_dims=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if not self.spatial3d:
            return self.net(x)[:, 0]
        v = x.view(B, self.n_feats, self.t, H, W).permute(0, 1, 3, 4, 2)
        # MONAI SwinUNETR requires every spatial dim divisible by 32
        # (patch 2 x 4 downsamples); pad depth 15 -> 32 and crop back.
        t_pad = (self.t + 31) // 32 * 32
        if t_pad != self.t:
            v = F.pad(v, (0, t_pad - self.t))
        o = self.net(v)                          # (B, T, H, W, Tp)
        o = o[..., :self.t]
        diag = o.diagonal(dim1=1, dim2=4)        # (B, H, W, T)
        return diag.permute(0, 3, 1, 2).contiguous()

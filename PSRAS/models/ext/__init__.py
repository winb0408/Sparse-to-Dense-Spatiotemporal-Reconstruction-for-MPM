"""External original-repo baselines for EXP-020 (DEC-023).

All three adapters wrap PRISTINE third-party repos (PSRAS/third_party/,
never modified) behind the protocol's model interface:

    forward(x: [B, F*T, H, W]) -> [B, T, H, W]

Adaptation choices pre-registered in PROJECT_LOG.md DEC-023/EXP-020:
- vswin   : official SwinTransformer3D backbone (classification) + a
            pre-registered lightweight 3D deconv decoder (the repo has no
            dense head); T padded to even for the (2,4,4) patch embed.
- sunetr  : official MONAI implementation (author-upstreamed code);
            in_chans=F per-frame channels, out_chans=T, plane j read on
            the (channel, depth) diagonal; 2D situation uses spatial_dims=2.
- rvrt    : official RVRT (upscale=1 deblurring path); input channels
            padded to the repo's hard-coded 3; recurrent over T with
            clip_size=2 (T=1 -> clip 1); 3-channel output averaged.
"""

import torch

from .vswin_net import VSwinNet
from .swinunetr_net import SunetrNet
from .rvrt_net import RvrtNet

__all__ = ["VSwinNet", "SunetrNet", "RvrtNet", "build_ext_model"]


def build_ext_model(kind: str, n_feats: int, t: int) -> torch.nn.Module:
    if kind == "vswin":
        return VSwinNet(n_feats, t)
    if kind == "sunetr":
        return SunetrNet(n_feats, t)
    if kind == "rvrt":
        return RvrtNet(n_feats, t)
    raise ValueError(f"unknown external model kind: {kind}")

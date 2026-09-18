"""EXP-020 vswin adapter: official Video-Swin-Transformer backbone.

The pristine repo (third_party/Video-Swin-Transformer) is a classification
codebase; its backbone file imports mmcv.runner / mmaction.utils / ..builder
only for checkpoint loading, logging and registry decoration.  We inject
stub modules into sys.modules so the untouched repo file imports standalone
(EXP-020 pre-registration: no third_party edits).

Adaptation (pre-registered DEC-023):
- official backbone config: patch (2,4,4), embed 96, depths (2,2,6,2),
  heads (3,6,12,24), window (2,7,7) -- the canonical video-swin-S.
- pre-registered lightweight 3D deconv decoder (repo has no dense head):
  768 -> ConvT(2,4,4) -> ConvT(1,4,4) -> 1x1x1 conv, x4 spatial upsample.
- T padded to even for the (2,4,4) temporal patch, cropped back on output.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

VSWIN_FILE = (Path(__file__).resolve().parents[2] / "third_party"
              / "Video-Swin-Transformer" / "mmaction" / "models" / "backbones"
              / "swin_transformer.py")
_BACKBONE_CLS = None


def _load_backbone_cls():
    global _BACKBONE_CLS
    if _BACKBONE_CLS is not None:
        return _BACKBONE_CLS

    # --- stub mmcv / mmaction (repo stays pristine) ---
    runner = types.ModuleType("mmcv.runner")

    def _no_checkpoint(*a, **k):  # pragma: no cover - pretrained unused
        raise RuntimeError("vswin pretrained loading not used in EXP-020")

    runner.load_checkpoint = _no_checkpoint
    mmcv = types.ModuleType("mmcv")
    mmcv.runner = runner
    utils = types.ModuleType("mmaction.utils")
    utils.get_root_logger = lambda **k: logging.getLogger("vswin")
    builder = types.ModuleType("mmaction.models.builder")

    class _Registry:
        def register_module(self, **k):
            def deco(cls):
                return cls
            return deco

    builder.BACKBONES = _Registry()
    models = types.ModuleType("mmaction.models")
    models.builder = builder
    mmaction = types.ModuleType("mmaction")
    mmaction.utils = utils
    mmaction.models = models
    bbs = types.ModuleType("mmaction.models.backbones")
    for name, mod in [("mmcv", mmcv), ("mmcv.runner", runner),
                      ("mmaction", mmaction), ("mmaction.utils", utils),
                      ("mmaction.models", models),
                      ("mmaction.models.builder", builder),
                      ("mmaction.models.backbones", bbs)]:
        sys.modules.setdefault(name, mod)

    spec = importlib.util.spec_from_file_location(
        "mmaction.models.backbones.swin_transformer", VSWIN_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _BACKBONE_CLS = mod.SwinTransformer3D
    return _BACKBONE_CLS


class VSwinNet(nn.Module):
    """Official Video-Swin backbone + pre-registered deconv decoder."""

    def __init__(self, n_feats: int, t: int, embed_dim: int = 96):
        super().__init__()
        self.n_feats, self.t = n_feats, t
        self.backbone = _load_backbone_cls()(
            patch_size=(2, 4, 4), in_chans=n_feats, embed_dim=embed_dim,
            depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            window_size=(2, 7, 7), patch_norm=True, drop_path_rate=0.2)
        c_end = embed_dim * 8
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(c_end, 256, (2, 4, 4), stride=(2, 4, 4)),
            nn.GELU(),
            nn.ConvTranspose3d(256, 128, (1, 4, 4), stride=(1, 4, 4)),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(64, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        v = x.view(B, self.n_feats, self.t, H, W)
        t_pad = self.t + (self.t % 2)          # (2,4,4) patch needs even D
        if t_pad != self.t:
            v = F.pad(v, (0, 0, 0, 0, 0, t_pad - self.t))
        feat = self.backbone(v.contiguous())    # (B, 768, Tp/2, H/4, W/4)
        out = self.decoder(feat)                # (B, 1, Tp, H, W)
        return out[:, 0, :self.t]

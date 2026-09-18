"""EXP-020 rvrt adapter: official RVRT (JingyunLiang/RVRT, pristine).

Pre-registered DEC-023 adaptation:
- upscale=1 deblurring path (output size == input size); input channels
  padded to the repo's hard-coded 3 (b1's 2 feats -> 1 zero channel);
  3-channel output averaged to 1.
- recurrent over T with clip_size=2; T padded to a multiple of the clip
  size, cropped back (T=1 uses clip_size=1).
- SpyNet from scratch (spynet_path=None; no pretrained weights are used
  anywhere in EXP-020 -- every model trains from scratch on CIDC25).
- The repo's deformable-attention CUDA extension is JIT-compiled at import;
  the THC shim (third_party/_shim) is injected via CPATH so no repo file
  is modified for torch>=1.11 compatibility.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

RVRT_REPO = Path(__file__).resolve().parents[2] / "third_party" / "RVRT"
_SHIM = Path(__file__).resolve().parents[2] / "third_party" / "_shim"
_RVRT_CLS = None


def _import_rvrt():
    # Load the pristine repo under a dedicated namespace ("_rvrt_repo") so
    # its top-level "models" package cannot collide with PSRAS's own.
    global _RVRT_CLS
    if _RVRT_CLS is not None:
        return _RVRT_CLS
    os.environ["CPATH"] = str(_SHIM) + os.pathsep + os.environ.get("CPATH", "")
    root = types.ModuleType("_rvrt_repo")
    root.__path__ = [str(RVRT_REPO)]
    pkg = types.ModuleType("_rvrt_repo.models")
    pkg.__path__ = [str(RVRT_REPO / "models")]
    sys.modules.setdefault("_rvrt_repo", root)
    sys.modules.setdefault("_rvrt_repo.models", pkg)
    spec = importlib.util.spec_from_file_location(
        "_rvrt_repo.models.network_rvrt", RVRT_REPO / "models" / "network_rvrt.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _RVRT_CLS = mod.RVRT
    return _RVRT_CLS


class RvrtNet(nn.Module):
    def __init__(self, n_feats: int, t: int, embed_dims=(144, 144, 144)):
        super().__init__()
        self.n_feats, self.t = n_feats, t
        self.clip = 2 if t > 1 else 1
        self.pad_c = 3 - n_feats               # repo hard-codes 3 input ch
        cls = _import_rvrt()
        self.net = cls(
            upscale=1, clip_size=self.clip, img_size=[2, 64, 64],
            window_size=[2, 8, 8], num_blocks=[1, 2, 1], depths=[2, 2, 2],
            embed_dims=list(embed_dims), num_heads=[6, 6, 6],
            spynet_path=None, use_checkpoint_attn=True,
            use_checkpoint_ffn=True, cpu_cache_length=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The repo's deformable-attention CUDA extension is fp32-only and
        # the fp16 path crashes under autocast; run RVRT in fp32 (declared
        # in the EXP-020 addendum).
        with torch.autocast("cuda", enabled=False):
            return self._forward_fp32(x.float())

    def _forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        v = x.view(B, self.n_feats, self.t, H, W).permute(0, 2, 1, 3, 4)
        t_pad = (self.t + self.clip - 1) // self.clip * self.clip
        if t_pad != self.t:
            v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, t_pad - self.t))
        if self.pad_c > 0:
            v = torch.cat([v, v.new_zeros(B, t_pad, self.pad_c, H, W)], dim=2)
        o = self.net(v)                         # (B, Tp, 3, H, W)
        return o[:, :self.t].mean(dim=2)

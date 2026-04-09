"""
models/video_swin_unet.py
=========================
Video Swin-UNet for the spatiotemporal_3d task.

The encoder uses Video Swin Transformer blocks that perform 3-D window attention
across (T, H, W) — enabling the model to exploit temporal correlations between
consecutive frames captured under complementary sparse scanning patterns.

Tensor shape convention (5-D):
    [B, C, T, H, W]   — standard spatiotemporal format
Internally, after patch embedding, we work with:
    [B, T'·H'·W', C]  — sequence of spatiotemporal tokens

All shape annotations use:
    T' = T // window_t   (temporal patch grid)
    H' = H // patch_size (spatial patch grid, H-dim)
    W' = W // patch_size (spatial patch grid, W-dim)
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 3-D Windowing helpers
# ─────────────────────────────────────────────

def window_partition_3d(
    x:  torch.Tensor,          # [B, T, H, W, C]
    wt: int, wh: int, ww: int, # temporal and spatial window sizes
) -> torch.Tensor:
    """
    Partition a spatiotemporal feature map into non-overlapping 3-D windows.

    Input : [B, T, H, W, C]
    Output: [B·nT·nH·nW, wt·wh·ww, C]

    nT = T//wt, nH = H//wh, nW = W//ww
    Window volume = wt*wh*ww tokens
    """
    B, T, H, W, C = x.shape
    x = x.view(B,
               T // wt, wt,
               H // wh, wh,
               W // ww, ww, C)
    # Permute so window indices are contiguous
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    # [B, nT, nH, nW, wt, wh, ww, C]
    return x.view(-1, wt * wh * ww, C)


def window_reverse_3d(
    wins: torch.Tensor,         # [B·nT·nH·nW, wt*wh*ww, C]
    wt: int, wh: int, ww: int,
    T: int, H: int, W: int,
) -> torch.Tensor:
    """
    Reverse window_partition_3d.

    Output: [B, T, H, W, C]
    """
    nT, nH, nW = T // wt, H // wh, W // ww
    B = int(wins.shape[0] / (nT * nH * nW))
    x = wins.view(B, nT, nH, nW, wt, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, T, H, W, -1)


# ─────────────────────────────────────────────
# 3-D Window Multi-Head Self-Attention
# ─────────────────────────────────────────────

class WindowAttention3D(nn.Module):
    """
    3-D window attention with relative positional bias over (t, h, w).

    Tokens: [B·nW, wt*wh*ww, C]
    """

    def __init__(
        self,
        dim:         int,
        window_size: Tuple[int, int, int],   # (wt, wh, ww)
        num_heads:   int,
        qkv_bias:    bool  = True,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ):
        super().__init__()
        self.dim        = dim
        self.wt, self.wh, self.ww = window_size
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = head_dim ** -0.5

        N = self.wt * self.wh * self.ww   # tokens per window

        # Relative position bias table: [(2wt-1)·(2wh-1)·(2ww-1), nHeads]
        self.rel_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.wt - 1) * (2 * self.wh - 1) * (2 * self.ww - 1),
                num_heads,
            )
        )
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # Pre-compute 3-D relative position indices
        ct = torch.arange(self.wt)
        ch = torch.arange(self.wh)
        cw = torch.arange(self.ww)
        grid = torch.stack(torch.meshgrid(ct, ch, cw, indexing="ij"))     # [3, wt, wh, ww]
        flat = grid.flatten(1)                                             # [3, N]
        rel  = flat[:, :, None] - flat[:, None, :]                        # [3, N, N]
        rel  = rel.permute(1, 2, 0).contiguous()                          # [N, N, 3]
        rel[..., 0] += self.wt - 1
        rel[..., 1] += self.wh - 1
        rel[..., 2] += self.ww - 1
        rel[..., 0] *= (2 * self.wh - 1) * (2 * self.ww - 1)
        rel[..., 1] *= (2 * self.ww - 1)
        self.register_buffer("rel_pos_idx", rel.sum(-1).flatten())        # [N²]

        self.qkv       = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : [B·nW, N, C]      N = wt*wh*ww
        mask : [nW,   N, N]  or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.rel_bias_table[self.rel_pos_idx].view(N, N, self.num_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ─────────────────────────────────────────────
# Video Swin Block
# ─────────────────────────────────────────────

class VideoSwinBlock(nn.Module):
    """
    One Video Swin Transformer block.

    Even index: W-MSA (shift_size = (0,0,0))
    Odd  index: SW-MSA (shift_size = (wt//2, wh//2, ww//2))

    Input / output: [B, T*H*W, C]   (sequence of spatiotemporal tokens)
    Caller must pass T, H, W for reshaping.
    """

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: Tuple[int, int, int] = (2, 8, 8),
        shift_size:  Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio:   float = 4.0,
        drop:        float = 0.0,
        attn_drop:   float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size
        self.norm1       = nn.LayerNorm(dim)
        self.norm2       = nn.LayerNorm(dim)
        self.attn        = WindowAttention3D(dim, window_size, num_heads,
                                             attn_drop=attn_drop, proj_drop=drop)
        mlp_hidden       = int(dim * mlp_ratio)
        self.mlp         = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        x : [B, T*H*W, C]
        """
        B, _, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, T, H, W, C)

        wt, wh, ww = self.window_size
        st, sh, sw = self.shift_size

        # Cyclic shift across (T, H, W)
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-st, -sh, -sw), dims=(1, 2, 3))

        # Partition → attend → reverse
        wins = window_partition_3d(x, wt, wh, ww)          # [B·nW, N, C]
        wins = self.attn(wins)
        x    = window_reverse_3d(wins, wt, wh, ww, T, H, W)  # [B, T, H, W, C]

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(st, sh, sw), dims=(1, 2, 3))

        x = x.view(B, T * H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Patch Merging / Expanding (3-D, spatial only)
# ─────────────────────────────────────────────

class PatchMerging3D(nn.Module):
    """
    Merge 2×2 spatial neighbours; keep temporal dimension unchanged.

    Input : [B, T*H*W, C]       → spatial dims H,W (T unchanged)
    Output: [B, T*(H/2)*(W/2), 2C]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) \
            -> Tuple[torch.Tensor, int, int, int]:
        B, _, C = x.shape
        x  = x.view(B, T, H, W, C)
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        x  = self.reduction(self.norm(x))
        return x, T, H // 2, W // 2


class PatchExpanding3D(nn.Module):
    """
    Inverse of PatchMerging3D: pixel-shuffle across spatial dims.

    Input : [B, T*H*W, C]
    Output: [B, T*(2H)*(2W), C/2]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) \
            -> Tuple[torch.Tensor, int, int, int]:
        B, _, C = x.shape
        x = self.expand(x)                          # [B, T*H*W, 2C]
        x = x.view(B, T, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, T, 2 * H, 2 * W, C // 2)
        x = self.norm(x).view(B, -1, C // 2)
        return x, T, 2 * H, 2 * W


# ─────────────────────────────────────────────
# Encoder / Decoder stages
# ─────────────────────────────────────────────

class VideoEncoderStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: Tuple[int, int, int], downsample: bool):
        super().__init__()
        wt, wh, ww = window_size
        self.blocks = nn.ModuleList([
            VideoSwinBlock(
                dim, num_heads, window_size,
                shift_size=(0, 0, 0) if i % 2 == 0
                           else (wt // 2, wh // 2, ww // 2),
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x: torch.Tensor, T: int, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, T, H, W)
        skip = (x, T, H, W)
        if self.downsample is not None:
            x, T, H, W = self.downsample(x, T, H, W)
        return x, T, H, W, skip


class VideoDecoderStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: Tuple[int, int, int]):
        super().__init__()
        wt, wh, ww = window_size
        self.upsample  = PatchExpanding3D(dim)
        self.skip_proj = nn.Linear(dim, dim // 2, bias=False)
        self.blocks    = nn.ModuleList([
            VideoSwinBlock(
                dim // 2, num_heads, window_size,
                shift_size=(0, 0, 0) if i % 2 == 0
                           else (wt // 2, wh // 2, ww // 2),
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int,
                skip: Tuple) -> Tuple[torch.Tensor, int, int, int]:
        x, T, H, W = self.upsample(x, T, H, W)
        sx, sT, sH, sW = skip
        x = x + self.skip_proj(sx)
        for blk in self.blocks:
            x = blk(x, T, H, W)
        return self.norm(x), T, H, W


# ─────────────────────────────────────────────
# Full Video Swin-UNet
# ─────────────────────────────────────────────

class VideoSwinUNet(nn.Module):
    """
    3-D UNet with Video Swin Transformer encoder for spatiotemporal_3d task.

    Input  : [B, in_channels,  T, H, W]   (in_channels = 1 or 2)
    Output : [B, out_channels, T, H, W]

    Shape flow (example: patch_size=4, T=8, H=W=256, embed_dim=96):
      Input   [B,  2,  8, 256, 256]
      ↓ 3-D patch embed (spatial only, stride=4)
              [B, 96,  8,  64,  64]
      → token [B, 8·64·64, 96]  = [B, 32768, 96]
      Encoder stage 0 → skip0 [B, 32768, 96]   (T=8, H=64, W=64)
               merge  → [B, 8192, 192]          (T=8, H=32, W=32)
      Encoder stage 1 → skip1 [B, 8192,  192]
               merge  → [B, 2048, 384]          (T=8, H=16, W=16)
      Encoder stage 2 → skip2 [B, 2048,  384]
               merge  → [B,  512, 768]          (T=8, H= 8, W= 8)
      Encoder stage 3 → bottleneck [B, 512, 768]
      Decoder stage 0 up → [B, 2048, 384] + skip2
      Decoder stage 1 up → [B, 8192, 192] + skip1
      Decoder stage 2 up → [B, 32768, 96] + skip0
      reshape → [B, 96, 8, 64, 64]
      head    → [B,  1, 8, 256, 256]
    """

    def __init__(
        self,
        in_channels:  int   = 2,
        out_channels: int   = 1,
        img_size:     int   = 256,
        n_frames:     int   = 8,
        patch_size:   int   = 4,          # spatial-only patch embed stride
        embed_dim:    int   = 96,
        depths:       List[int]            = [2, 2, 6, 2],
        num_heads:    List[int]            = [3, 6, 12, 24],
        window_size:  Tuple[int, int, int] = (2, 8, 8),
    ):
        super().__init__()
        self.n_stages = len(depths)
        dims          = [embed_dim * (2 ** i) for i in range(self.n_stages)]

        # ── Spatial patch embedding (applied independently to each frame) ─────
        # [B, in_ch, T, H, W] → [B, embed_dim, T, H/ps, W/ps]
        # Implement as a 3-D conv with kernel (1, ps, ps) and stride (1, ps, ps)
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride     =(1, patch_size, patch_size),
        )
        self.patch_norm  = nn.LayerNorm(embed_dim)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.ModuleList([
            VideoEncoderStage(
                dim        = dims[i],
                depth      = depths[i],
                num_heads  = num_heads[i],
                window_size= window_size,
                downsample = (i < self.n_stages - 1),
            )
            for i in range(self.n_stages)
        ])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.ModuleList([
            VideoDecoderStage(
                dim        = dims[self.n_stages - 1 - i],
                depth      = depths[self.n_stages - 2 - i],
                num_heads  = num_heads[self.n_stages - 2 - i],
                window_size= window_size,
            )
            for i in range(self.n_stages - 1)
        ])

        # ── Final head: upsample spatial dims back ×patch_size ────────────────
        # Use 3-D transposed conv with temporal kernel=1 to preserve T
        self.head = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2,
                               kernel_size=(1, 2, 2), stride=(1, 2, 2)), nn.GELU(),
            nn.ConvTranspose3d(embed_dim // 2, embed_dim // 4,
                               kernel_size=(1, 2, 2), stride=(1, 2, 2)), nn.GELU(),
            nn.Conv3d(embed_dim // 4, out_channels, kernel_size=1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, in_ch, T, H, W]
        """
        B, C, T, H, W = x.shape

        # ── Patch embed ──
        # [B, in_ch, T, H, W] → [B, embed_dim, T, H/ps, W/ps]
        x = self.patch_embed(x)
        T_, pH, pW = x.shape[-3], x.shape[-2], x.shape[-1]

        # Flatten to token sequence: [B, T'*H'*W', embed_dim]
        x = x.permute(0, 2, 3, 4, 1).contiguous()         # [B, T', H', W', C]
        x = self.patch_norm(x.view(B, -1, x.shape[-1]))   # [B, T'*H'*W', C]

        curT, curH, curW = T_, pH, pW
        skips = []

        # ── Encode ──
        for stage in self.encoder:
            x, curT, curH, curW, skip = stage(x, curT, curH, curW)
            skips.append(skip)

        # ── Decode ──
        for i, stage in enumerate(self.decoder):
            skip = skips[-(i + 2)]
            x, curT, curH, curW = stage(x, curT, curH, curW, skip)

        # ── Reshape to 5-D and apply head ──
        x = x.view(B, curT, curH, curW, -1)               # [B, T', H', W', embed_dim]
        x = x.permute(0, 4, 1, 2, 3).contiguous()         # [B, embed_dim, T', H', W']
        x = self.head(x)                                   # [B, 1, T, H, W]
        return x

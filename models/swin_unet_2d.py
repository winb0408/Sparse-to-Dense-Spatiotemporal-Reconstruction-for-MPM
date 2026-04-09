"""
models/swin_unet_2d.py
======================
2-D Swin-UNet for baseline and spatial_2d tasks.

Encoder: 4 stages of (Swin-Transformer blocks + PatchMerging)
Decoder: mirrored, with PatchExpanding + residual skip connections
Head   : ConvTranspose2d chain back to full spatial resolution

All tensor shapes are annotated at each major step.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Windowing helpers
# ─────────────────────────────────────────────

def window_partition_2d(x: torch.Tensor, ws: int) -> torch.Tensor:
    """
    Partition a feature map into non-overlapping windows.

    Input : [B, H, W, C]
    Output: [B*nH*nW, ws, ws, C]   where nH=H//ws, nW=W//ws
    """
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse_2d(wins: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window_partition_2d.

    Input : [B*nH*nW, ws, ws, C]
    Output: [B, H, W, C]
    """
    B = int(wins.shape[0] / (H * W / ws / ws))
    x = wins.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ─────────────────────────────────────────────
# Window Multi-Head Self-Attention (W-MSA / SW-MSA)
# ─────────────────────────────────────────────

class WindowAttention2D(nn.Module):
    """
    Window-based multi-head self-attention with relative positional bias.

    Works on flattened window tokens: [nW*B, ws², C]
    """

    def __init__(
        self,
        dim:         int,
        window_size: int,
        num_heads:   int,
        qkv_bias:    bool  = True,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ):
        super().__init__()
        self.dim         = dim
        self.ws          = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # Relative position bias table: [(2ws-1)², nHeads]
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # Pre-compute relative position indices
        coords   = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size) # , indexing="ij"
        ))                                                              # [2, ws, ws]
        flat     = coords.flatten(1)                                   # [2, ws²]
        rel      = flat[:, :, None] - flat[:, None, :]                 # [2, ws², ws²]
        rel      = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += window_size - 1
        rel[..., 1] += window_size - 1
        rel[..., 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_idx", rel.sum(-1).flatten())     # [ws²·ws²]

        self.qkv       = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : [nW*B, ws², C]
        mask : [nW, ws², ws²]  (cyclic-shift attention mask) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        #   q,k,v : [B_, nH, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        #   attn : [B_, nH, N, N]

        # Add relative position bias: [nH, N, N]
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
# Swin Transformer Block (W-MSA + SW-MSA alternation)
# ─────────────────────────────────────────────

class SwinBlock2D(nn.Module):
    """
    One Swin Transformer block.
    Even-index blocks use W-MSA (shift_size=0).
    Odd-index blocks use SW-MSA (shift_size=ws//2).
    """

    def __init__(
        self,
        dim:        int,
        num_heads:  int,
        window_size:int  = 8,
        shift_size: int  = 0,
        mlp_ratio:  float= 4.0,
        drop:       float= 0.0,
        attn_drop:  float= 0.0,
    ):
        super().__init__()
        self.ws         = window_size
        self.shift      = shift_size
        self.norm1      = nn.LayerNorm(dim)
        self.norm2      = nn.LayerNorm(dim)
        self.attn       = WindowAttention2D(dim, window_size, num_heads,
                                            attn_drop=attn_drop, proj_drop=drop)
        mlp_hidden      = int(dim * mlp_ratio)
        self.mlp        = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x : [B, H*W, C]   (sequence of spatial tokens)
        """
        B, _, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Cyclic shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # Partition → attend → reverse
        wins = window_partition_2d(x, self.ws)          # [nW*B, ws, ws, C]
        wins = wins.view(-1, self.ws ** 2, C)
        wins = self.attn(wins)
        wins = wins.view(-1, self.ws, self.ws, C)
        x    = window_reverse_2d(wins, self.ws, H, W)  # [B, H, W, C]

        # Reverse cyclic shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Patch Merging (down-sample ×2)
# ─────────────────────────────────────────────

class PatchMerging2D(nn.Module):
    """
    Merge 2×2 neighbouring patches → halve spatial dims, double channels.

    Input : [B, H*W, C]     H,W
    Output: [B, H/2*W/2, 2C]  H/2, W/2
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) \
            -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = self.reduction(self.norm(
            torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        ))
        return x, H // 2, W // 2


# ─────────────────────────────────────────────
# Patch Expanding (up-sample ×2)
# ─────────────────────────────────────────────

class PatchExpanding2D(nn.Module):
    """
    Inverse of PatchMerging: double spatial dims, halve channels (pixel-shuffle).

    Input : [B, H*W, C]
    Output: [B, 4H*W, C/2]    i.e. spatial dims doubled, channels halved
    """

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int) \
            -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = self.expand(x)                              # [B, H*W, 2C]
        x = x.view(B, H, W, 2 * C)
        # Pixel-shuffle: [B, H, W, 2, 2, C/2] → [B, 2H, 2W, C/2]
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)
        x = self.norm(x).view(B, -1, C // 2)
        return x, 2 * H, 2 * W


# ─────────────────────────────────────────────
# Encoder / Decoder stages
# ─────────────────────────────────────────────

class EncoderStage2D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int, downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock2D(dim, num_heads, window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])
        self.downsample = PatchMerging2D(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        skip = (x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, skip


class DecoderStage2D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.upsample  = PatchExpanding2D(dim)
        # After concat with skip (same dim // 2), project back to dim // 2
        self.skip_proj = nn.Linear(dim, dim // 2, bias=False)
        self.blocks    = nn.ModuleList([
            SwinBlock2D(dim // 2, num_heads, window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int,
                skip: Tuple) -> Tuple[torch.Tensor, int, int]:
        x, H, W = self.upsample(x, H, W)  # x 维度变为 dim // 2
        sx, sH, sW = skip                 # sx 维度已经是 dim // 2
        
        # 1. 在通道维度（dim=-1）将上采样的 x 与跳跃连接的 sx 拼接起来
        # 拼接后维度变为 (dim // 2) + (dim // 2) = dim
        concat_x = torch.cat([x, sx], dim=-1)
        
        # 2. 通过 skip_proj 将维度从 dim 重新投影/降维回 dim // 2
        x = self.skip_proj(concat_x)
        
        for blk in self.blocks:
            x = blk(x, H, W)
        return self.norm(x), H, W


# ─────────────────────────────────────────────
# Full Swin-UNet 2D
# ─────────────────────────────────────────────

class SwinUNet2D(nn.Module):
    """
    2-D Swin-UNet for baseline / spatial_2d tasks.

    Input  : [B, in_channels,  H, W]   (in_channels = 1 or 2)
    Output : [B, out_channels, H, W]   (out_channels = 1)
    """

    def __init__(
        self,
        in_channels:  int   = 2,
        out_channels: int   = 1,
        img_size:     int   = 256,
        patch_size:   int   = 4,        # initial patch embedding stride
        embed_dim:    int   = 96,
        depths:       List[int] = [2, 2, 6, 2],
        num_heads:    List[int] = [3, 6, 12, 24],
        window_size:  int   = 8,
    ):
        super().__init__()
        self.n_stages  = len(depths)
        dims           = [embed_dim * (2 ** i) for i in range(self.n_stages)]

        # ── Patch embedding ──────────────────────────────────────────────────
        # [B, in_ch, H, W] → [B, embed_dim, H/ps, W/ps]
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.patch_norm  = nn.LayerNorm(embed_dim)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.ModuleList([
            EncoderStage2D(
                dim        = dims[i],
                depth      = depths[i],
                num_heads  = num_heads[i],
                window_size= window_size,
                downsample = (i < self.n_stages - 1),
            )
            for i in range(self.n_stages)
        ])

        # ── Decoder ──────────────────────────────────────────────────────────
        # Decoder stage i up-samples from dims[n-1-i] → dims[n-2-i]
        self.decoder = nn.ModuleList([
            DecoderStage2D(
                dim        = dims[self.n_stages - 1 - i],
                depth      = depths[self.n_stages - 2 - i],
                num_heads  = num_heads[self.n_stages - 2 - i],
                window_size= window_size,
            )
            for i in range(self.n_stages - 1)
        ])

        # ── Final head: upsample back to full resolution ──────────────────────
        # patch_size = 4 → need 2 × ConvTranspose2d(stride=2)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2), nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2), nn.GELU(),
            nn.Conv2d(embed_dim // 4, out_channels, 1),
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
        x : [B, in_ch, H, W]

        Shape flow (example: patch_size=4, img_size=256, embed_dim=96):
          [B,  2, 256, 256]
          [B, 96,  64,  64]   patch embed
          → sequence [B, 4096, 96]
          encoder stage 0 → [B, 4096, 96]   skip0    (H=64, W=64)
                  down   → [B, 1024, 192]             (H=32, W=32)
          encoder stage 1 → [B, 1024, 192]  skip1
                  down   → [B,  256, 384]             (H=16, W=16)
          encoder stage 2 → [B,  256, 384]  skip2
                  down   → [B,   64, 768]             (H= 8, W= 8)
          encoder stage 3 → [B,   64, 768]  (bottleneck)
          decoder stage 0: up → [B,  256, 384] + skip2
          decoder stage 1: up → [B, 1024, 192] + skip1
          decoder stage 2: up → [B, 4096,  96] + skip0
          reshape  → [B, 96, 64, 64]
          head     → [B,  1, 256, 256]
        """
        B, C, H, W = x.shape

        # Patch embed
        x = self.patch_embed(x)                    # [B, embed_dim, H/ps, W/ps]
        ph, pw = x.shape[-2], x.shape[-1]
        x = self.patch_norm(x.flatten(2).transpose(1, 2))  # [B, ph*pw, embed_dim]

        cur_H, cur_W = ph, pw
        skips = []

        # Encode
        for stage in self.encoder:
            x, cur_H, cur_W, skip = stage(x, cur_H, cur_W)
            skips.append(skip)

        # Decode (use all but the last skip which is the bottleneck itself)
        for i, stage in enumerate(self.decoder):
            skip = skips[-(i + 2)]
            x, cur_H, cur_W = stage(x, cur_H, cur_W, skip)

        # Reshape to spatial feature map
        x = x.transpose(1, 2).view(B, -1, cur_H, cur_W)   # [B, embed_dim, ph, pw]
        return self.head(x)                                 # [B, 1, H, W]

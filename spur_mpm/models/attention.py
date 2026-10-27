"""Window attention with correct shifted-window (SW-MSA) masks.

Adapted verbatim from PSRAS/models/attention.py (branch review-v4-20261019),
which already fixed the legacy bug of applying roll+partition without the
non-adjacency mask. 2-D and 3-D variants are kept so a future axial model can
reuse the same kernels.

Fixes the legacy issue (PLAN.md C3): the old implementations applied
torch.roll + window partition but never restricted attention across
non-adjacent regions, i.e. they were not standard SW-MSA.  Here the
attention mask is built exactly as in the reference Swin / Video Swin
implementations: region ids over the padded map, rolled, then
per-window pairwise comparisons.

Provides 2D (H, W) and 3D (T, H, W) variants so the future unified
axial-3D network (PLAN.md Phase 3) can reuse the same kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["window_partition_2d", "window_reverse_2d",
           "window_partition_3d", "window_reverse_3d",
           "shifted_window_mask_2d", "shifted_window_mask_3d",
           "WindowAttention2D", "WindowAttention3D"]


# ---------------------------------------------------------------------------
# partition helpers
# ---------------------------------------------------------------------------

def window_partition_2d(x: torch.Tensor, ws: int) -> torch.Tensor:
    # x: [B, H, W, C] -> [B*nH*nW, ws*ws, C]
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def window_reverse_2d(x: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    B = x.shape[0] // ((H // ws) * (W // ws))
    C = x.shape[-1]
    x = x.view(B, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


def window_partition_3d(x: torch.Tensor, wt: int, ws: int) -> torch.Tensor:
    # x: [B, T, H, W, C] -> [B*nT*nH*nW, wt*ws*ws, C]
    B, T, H, W, C = x.shape
    x = x.view(B, T // wt, wt, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5, 6, 7).reshape(-1, wt * ws * ws, C)


def window_reverse_3d(x: torch.Tensor, wt: int, ws: int, T: int, H: int, W: int) -> torch.Tensor:
    B = x.shape[0] // ((T // wt) * (H // ws) * (W // ws))
    C = x.shape[-1]
    x = x.view(B, T // wt, H // ws, wt, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5, 6, 7).reshape(B, T, H, W, C)


# ---------------------------------------------------------------------------
# shifted-window attention masks (the legacy code was missing these)
# ---------------------------------------------------------------------------

def shifted_window_mask_2d(Hp: int, Wp: int, ws: int, shift: int,
                           device=None) -> torch.Tensor:
    """SW-MSA attention mask for 2D windows.

    Returns float tensor [nW, ws*ws, ws*ws]: 0 where attention is allowed,
    -100 across non-adjacent rolled regions.  shift == 0 (W-MSA) allows
    everything inside each window.
    """
    if shift <= 0:
        return torch.zeros((Hp // ws) * (Wp // ws), ws * ws, ws * ws, device=device)
    img = torch.zeros(1, Hp, Wp, 1, device=device)
    cnt = 0
    for h in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for w in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, h, w, :] = cnt
            cnt += 1
    wins = window_partition_2d(img, ws)              # [nW, ws*ws, 1]
    mask = wins - wins.transpose(1, 2)               # [nW, N, N]
    return mask.masked_fill(mask != 0, -100.0)


def shifted_window_mask_3d(Tp: int, Hp: int, Wp: int, wt: int, ws: int,
                           st: int, shift: int, device=None) -> torch.Tensor:
    """SW-MSA attention mask for 3D (T, H, W) windows -> [nW, wt*ws*ws, ...]."""
    if st <= 0 and shift <= 0:
        n_w = (Tp // wt) * (Hp // ws) * (Wp // ws)
        return torch.zeros(n_w, wt * ws * ws, wt * ws * ws, device=device)
    img = torch.zeros(1, Tp, Hp, Wp, 1, device=device)
    cnt = 0
    for t in (slice(0, -wt), slice(-wt, -st), slice(-st, None)):
        for h in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            for w in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                img[:, t, h, w, :] = cnt
                cnt += 1
    wins = window_partition_3d(img, wt, ws)
    mask = wins - wins.transpose(1, 2)
    return mask.masked_fill(mask != 0, -100.0)


def _pair_rel_index(window: tuple[int, ...]) -> torch.Tensor:
    """Relative-position index for all token pairs inside a window.

    Tokens are ordered row-major over ``window`` (per-axis sizes).
    Returns [N*N] indices into a table of size prod_d (2*s_d - 1).
    """
    grids = torch.meshgrid(*[torch.arange(s) for s in window], indexing="ij")
    grid = torch.stack(grids).reshape(len(window), -1)           # [D, N]
    rel = grid[:, :, None] - grid[:, None, :]                    # [D, N, N]
    rel = rel + torch.tensor([s - 1 for s in window])[:, None, None]
    idx = rel[0]
    for d in range(1, len(window)):
        idx = idx * (2 * window[d] - 1) + rel[d]
    return idx.reshape(-1).long()


def _bias_for_window(wins: torch.Tensor, n_w: int, table: torch.Tensor,
                     rel_index: torch.Tensor, n: int, heads: int,
                     mask: torch.Tensor) -> torch.Tensor:
    """Combine relative-position bias with the SW-MSA mask, expanded to batch.

    wins: [G, N, C]; mask: [n_w, N, N]; returns [G, heads, N, N].
    """
    rpb = table[rel_index].reshape(n, n, heads).permute(2, 0, 1)   # [heads, N, N]
    bias = rpb[None] + mask[:, None]                               # [n_w, heads, N, N]
    b = wins.shape[0] // n_w
    return bias[None].expand(b, -1, -1, -1, -1).reshape(wins.shape[0], heads, n, n)


# ---------------------------------------------------------------------------
# attention modules
# ---------------------------------------------------------------------------

class _MHA(nn.Module):
    """Multi-head self-attention over a fixed-size token set."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        # x: [G, N, C]; attn_bias: [G, heads, N, N]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # [B, h, N, d]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn + attn_bias).softmax(dim=-1) @ v      # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class WindowAttention2D(nn.Module):
    """Standard Swin block attention (W-MSA / SW-MSA) for [B, C, H, W]."""

    def __init__(self, dim: int, heads: int = 4, window_size: int = 8, shifted: bool = False):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.ws = window_size
        self.shift = window_size // 2 if shifted else 0
        self.norm = nn.LayerNorm(dim)
        self.attn = _MHA(dim, heads)
        n = self.ws * self.ws
        self.register_buffer("rpb_table", torch.zeros((2 * self.ws - 1) ** 2, heads))
        nn.init.zeros_(self.rpb_table)
        self.register_buffer("rel_index", _pair_rel_index((self.ws, self.ws)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Hp = ((H + self.ws - 1) // self.ws) * self.ws
        Wp = ((W + self.ws - 1) // self.ws) * self.ws
        xp = F.pad(x, (0, Wp - W, 0, Hp - H)).permute(0, 2, 3, 1)   # [B, H, W, C]
        if self.shift:
            xp = torch.roll(xp, shifts=(-self.shift, -self.shift), dims=(1, 2))
        mask = shifted_window_mask_2d(Hp, Wp, self.ws, self.shift, x.device)
        wins = window_partition_2d(self.norm(xp), self.ws)          # [G, N, C]
        n = self.ws * self.ws
        bias = _bias_for_window(wins, mask.shape[0], self.rpb_table,
                                self.rel_index, n, self.heads, mask)
        out = self.attn(wins, bias)
        xp = window_reverse_2d(out, self.ws, Hp, Wp)
        if self.shift:
            xp = torch.roll(xp, shifts=(self.shift, self.shift), dims=(1, 2))
        return xp[:, :H, :W, :].permute(0, 3, 1, 2)


class WindowAttention3D(nn.Module):
    """Video-Swin-style block attention for [B, C, T, H, W].

    The temporal window extent ``wt`` is a constructor parameter; with
    wt=1 the block reduces to per-frame spatial window attention.
    """

    def __init__(self, dim: int, heads: int = 4,
                 window_size: tuple[int, int] = (2, 8), shifted: bool = False):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.wt, self.ws = window_size
        self.st = self.wt // 2 if shifted else 0
        self.shift = self.ws // 2 if shifted else 0
        self.norm = nn.LayerNorm(dim)
        self.attn = _MHA(dim, heads)
        n = self.wt * self.ws * self.ws
        n_rel = (2 * self.wt - 1) * (2 * self.ws - 1) ** 2
        self.register_buffer("rpb_table", torch.zeros(n_rel, heads))
        nn.init.zeros_(self.rpb_table)
        self.register_buffer("rel_index", _pair_rel_index((self.wt, self.ws, self.ws)),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        Tp = ((T + self.wt - 1) // self.wt) * self.wt
        Hp = ((H + self.ws - 1) // self.ws) * self.ws
        Wp = ((W + self.ws - 1) // self.ws) * self.ws
        xp = F.pad(x, (0, Wp - W, 0, Hp - H, 0, Tp - T)).permute(0, 2, 3, 4, 1)
        if self.shift:
            xp = torch.roll(xp, shifts=(-self.st, -self.shift, -self.shift),
                            dims=(1, 2, 3))
        mask = shifted_window_mask_3d(Tp, Hp, Wp, self.wt, self.ws,
                                      self.st, self.shift, x.device)
        wins = window_partition_3d(self.norm(xp), self.wt, self.ws)
        n = self.wt * self.ws * self.ws
        bias = _bias_for_window(wins, mask.shape[0], self.rpb_table,
                                self.rel_index, n, self.heads, mask)
        out = self.attn(wins, bias)
        xp = window_reverse_3d(out, self.wt, self.ws, Tp, Hp, Wp)
        if self.shift:
            xp = torch.roll(xp, shifts=(self.st, self.shift, self.shift),
                            dims=(1, 2, 3))
        return xp[:, :T, :H, :W, :].permute(0, 4, 1, 2, 3)

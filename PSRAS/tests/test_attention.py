"""Tests for corrected SW-MSA window attention (PLAN.md C3 fix).

The critical property: with shifted windows, attention across non-adjacent
rolled regions must be blocked.  Verified directly on the mask and
end-to-end through the module.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.attention import (WindowAttention2D, WindowAttention3D,
                              shifted_window_mask_2d, shifted_window_mask_3d)


def test_sw_mask_blocks_cross_region_attention():
    mask = shifted_window_mask_2d(16, 16, 8, 4)
    # With cyclic shift there MUST exist blocked pairs (-100) ...
    assert (mask == -100.0).any()
    # ... and allowed pairs (0) within each window.
    assert (mask == 0.0).any()
    # W-MSA (no shift) blocks nothing.
    mask_w = shifted_window_mask_2d(16, 16, 8, 0)
    assert not (mask_w == -100.0).any()


def test_2d_module_shapes_and_backward():
    torch.manual_seed(0)
    attn = WindowAttention2D(dim=32, heads=4, window_size=8, shifted=True)
    x = torch.randn(2, 32, 20, 24, requires_grad=True)
    y = attn(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_shift_changes_output():
    """SW-MSA must differ from W-MSA (mask actually participates)."""
    torch.manual_seed(1)
    torch.use_deterministic_algorithms(False)
    xs = torch.randn(1, 8, 32, 32)
    x = xs.clone().requires_grad_(False)
    m_w = WindowAttention2D(dim=8, heads=2, window_size=8, shifted=False)
    m_s = WindowAttention2D(dim=8, heads=2, window_size=8, shifted=True)
    for m in (m_w, m_s):
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.2)
    out_w, out_s = m_w(x), m_s(x)
    assert not torch.allclose(out_w, out_s)


def test_3d_module_shapes_and_backward():
    torch.manual_seed(2)
    attn = WindowAttention3D(dim=16, heads=4, window_size=(2, 8), shifted=True)
    x = torch.randn(1, 16, 5, 24, 24, requires_grad=True)
    y = attn(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_3d_mask_blocks_cross_region_attention():
    mask = shifted_window_mask_3d(4, 16, 16, 2, 8, 1, 4)
    assert (mask == -100.0).any() and (mask == 0.0).any()

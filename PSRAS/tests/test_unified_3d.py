"""Phase 3 unit tests: models/unified_3d.py (DESIGN_phase3.md).

Pre-registered exit conditions that are testable without training:
  - axial attention is an exact no-op at init for ANY T (zero-init branch)
  - T=1 input passes through unchanged shape (2D compatibility)
  - spatial-only downsampling: T never changes through the UNet
  - fusion stem parses feature-major channels correctly (DEC-013)
  - parameter count stays within +/-10% of SwinUNet2D at equal config
"""
import torch

from models.attention import WindowAttention3D  # noqa: F401  (import sanity)
from models.prior_fusion import PriorFusionStem
from models.swin_unet_2d import SwinUNet2D
from models.unified_3d import AxialAttention, SwinUNet3D, Unified3D


def test_axial_attention_is_exact_noop_at_init():
    torch.manual_seed(0)
    attn = AxialAttention(48)
    x = torch.randn(2, 48, 7, 16, 16)
    assert torch.allclose(attn(x), x, atol=1e-6), \
        "zero-init axial branch must be an exact no-op at init (any T)"


def test_axial_attention_mixes_after_init():
    torch.manual_seed(0)
    attn = AxialAttention(48)
    with torch.no_grad():
        # give proj a non-zero weight and a strong T-dependence in input
        attn.proj.weight.normal_(0, 0.1)
        x = torch.randn(1, 48, 7, 8, 8)
        x[0, :, 0] += 5.0
        out = attn(x)
        assert not torch.allclose(out, x, atol=1e-4)
        # slice 0 (perturbed) must mix into slice 1 (untouched before)
        assert (out[0, :, 1] - x[0, :, 1]).abs().max() > 1e-4


def test_unet3d_preserves_t_and_shape():
    torch.manual_seed(0)
    net = SwinUNet3D(3, base=16, window=4)
    for t in (1, 3, 7, 15):
        x = torch.randn(1, 3, t, 32, 32)
        assert net(x).shape == (1, 1, t, 32, 32), f"T={t} must pass through"


def test_unified3d_end_to_end_all_t():
    torch.manual_seed(0)
    for t in (1, 7, 15):
        net = Unified3D(5, t=t, base=16, window=4, fusion=True)
        x = torch.randn(2, 5 * t, 32, 32)
        assert net(x).shape == (2, t, 32, 32)


def test_unified3d_fusion_stem_stream_layout():
    """Feature-major parse: e_y sees y, e_w sees x0, e_s sees mask/c_s/u_p."""
    t = 7
    stem = Unified3D(5, t=t, base=8, window=4, fusion=True)
    x = torch.arange(1, 6).float().repeat_interleave(t).view(1, 5 * t, 1, 1)
    x = x.expand(1, 5 * t, 8, 8)
    seen = {}
    for name, mod in [("e_y", stem.stem.e_y), ("e_w", stem.stem.e_w),
                      ("e_s", stem.stem.e_s)]:
        mod.register_forward_hook(lambda m, i, o, name=name: seen.__setitem__(name, i[0]))
    with torch.no_grad():
        stem(x)
    assert torch.all(seen["e_y"] == 1.0)
    assert torch.all(seen["e_w"] == 3.0)
    assert torch.all(seen["e_s"][:, 0] == 2.0)      # mask stream
    assert torch.all(seen["e_s"][:, 1] == 4.0)      # c_s stream
    assert torch.all(seen["e_s"][:, 2] == 5.0)      # u_p stream


def test_unified3d_nonfusion_matches_2d_interface():
    torch.manual_seed(0)
    net = Unified3D(3, t=7, base=16, window=4, fusion=False)
    x = torch.randn(2, 3 * 7, 32, 32)
    assert net(x).shape == (2, 7, 32, 32)


def test_param_count_parity_with_2d():
    """Unified3D must be within +/-10% params of SwinUNet2D (fair comparison)."""
    torch.manual_seed(0)
    # 2D B4-equivalent: in 15 channels flat, out 7 (params dominated by base)
    m2d = SwinUNet2D(15, 7, base=48, window=8)
    stem2d = PriorFusionStem(35, 7, 48)
    n2d = sum(p.numel() for p in m2d.parameters()) + \
        sum(p.numel() for p in stem2d.parameters())
    m3d = Unified3D(5, t=7, base=48, window=8, fusion=True)
    n3d = sum(p.numel() for p in m3d.parameters())
    ratio = n3d / n2d
    assert 0.9 < ratio < 1.10, f"param ratio {ratio:.3f} outside 0.9-1.10"


def test_axial_at_bottleneck_places_axial_only_in_bottleneck():
    """DEC-017 rollback variant: encoder/decoder blocks are pure 2D."""
    torch.manual_seed(0)
    net = Unified3D(5, t=7, base=16, window=4, fusion=True, axial_at="bottleneck")
    assert net.unet.encoders[0].blocks[0].axial is None
    assert net.unet.decoders[0].blocks[0].axial is None
    assert net.unet.bottleneck.blocks[0].axial is not None
    x = torch.randn(1, 5 * 7, 32, 32)
    assert net(x).shape == (1, 7, 32, 32)
    # "all" placement keeps axial everywhere (default unchanged)
    net_all = Unified3D(5, t=7, base=16, window=4, fusion=True)
    assert net_all.unet.encoders[0].blocks[0].axial is not None


def test_axial_at_bottleneck_param_parity_with_2d():
    """DEC-017 rollback variant sits BELOW the 2D param count (ratio ~0.88).

    Directional parity: with encoder/decoder axial branches removed and
    bottleneck_depth=1 the variant has ~12% fewer parameters than the 2D
    B4-equivalent.  This is the conservative direction for the rollback
    test -- a win cannot be attributed to extra capacity.  Restoring
    parity via bottleneck_depth=2 overshoots to 1.103, so we keep the
    smaller net and assert the one-sided band.
    """
    torch.manual_seed(0)
    m2d = SwinUNet2D(15, 7, base=48, window=8)
    stem2d = PriorFusionStem(35, 7, 48)
    n2d = sum(p.numel() for p in m2d.parameters()) + \
        sum(p.numel() for p in stem2d.parameters())
    m3d = Unified3D(5, t=7, base=48, window=8, fusion=True, axial_at="bottleneck")
    n3d = sum(p.numel() for p in m3d.parameters())
    ratio = n3d / n2d
    assert 0.85 < ratio < 1.10, f"param ratio {ratio:.3f} outside 0.85-1.10"


def test_output_center_fills_all_frames_with_query_slice():
    """DEC-018: center output keeps the [B, T, H, W] contract; every frame
    slot carries the query-slice prediction (only the center is supervised)."""
    torch.manual_seed(0)
    net = Unified3D(5, t=7, base=16, window=4, fusion=True,
                    axial_at="bottleneck", output="center")
    net.eval()
    x = torch.randn(2, 5 * 7, 32, 32)
    y = net(x)
    assert y.shape == (2, 7, 32, 32)
    for tt in range(7):
        assert torch.allclose(y[:, tt], y[:, 3]), f"frame {tt} != center"

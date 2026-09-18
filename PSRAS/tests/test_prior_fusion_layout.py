"""DEC-013 regression: channel layout must be parsed feature-major.

The dataset builds inputs as cat([feat_f.reshape(-1,H,W) for f in features]),
i.e. channels are [y(t0..t), m(t0..t), x0(t0..t), c_s(t0..t), u_p(t0..t)].
PriorFusionStem (and the residual-X0 extraction) must split on features
first; a t-major view scrambles streams whenever T > 1.
"""
import torch

from models.prior_fusion import PriorFusionStem


def test_stem_streams_feature_major() -> None:
    """With each stream set to a distinct constant, e_y/e_w/e_s must receive
    exactly their own stream's constant."""
    t = 7
    stem = PriorFusionStem(5 * t, t, out_ch=8)
    # channels [y*7, m*7, x0*7, c_s*7, u_p*7] with stream f filled by value f+1
    x = torch.arange(1, 6).float().repeat_interleave(t).view(1, 5 * t, 1, 1)
    x = x.expand(1, 5 * t, 8, 8)

    seen = {}
    for name, mod in [("e_y", stem.e_y), ("e_w", stem.e_w), ("e_s", stem.e_s)]:
        mod.register_forward_hook(lambda m, i, o, name=name: seen.__setitem__(name, i[0]))

    with torch.no_grad():
        stem(x)
    assert torch.all(seen["e_y"] == 1.0), "e_y must receive only the y stream"
    assert torch.all(seen["e_w"] == 3.0), "e_w must receive only the x0 stream"
    assert torch.all(seen["e_s"][:, 0:t] == 2.0), "e_s y-slot must be mask"
    assert torch.all(seen["e_s"][:, t:2 * t] == 4.0), "e_s middle slot must be c_s"
    assert torch.all(seen["e_s"][:, 2 * t:3 * t] == 5.0), "e_s last slot must be u_p"


def test_stem_forward_runs_and_gate_is_half_at_init() -> None:
    """Zero-init gate -> g = sigmoid(0) = 0.5 at start (DEC-013 note):
    F_in == e_y(Y) + 0.5 * e_w(X0)."""
    t = 7
    torch.manual_seed(0)
    stem = PriorFusionStem(5 * t, t, out_ch=8)
    x = torch.randn(2, 5 * t, 16, 16)
    with torch.no_grad():
        f = stem(x)
        v = x.view(2, 5, t, 16, 16)
        expected = stem.e_y(v[:, 0]) + 0.5 * stem.e_w(v[:, 2])
    assert f.shape == (2, 8, 16, 16)
    assert torch.allclose(f, expected, atol=1e-6), "init gate admits exactly half of e_w(X0)"

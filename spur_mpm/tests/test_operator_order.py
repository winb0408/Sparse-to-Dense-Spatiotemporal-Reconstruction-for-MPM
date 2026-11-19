"""The two Wiener orderings coincide on the centre plane at T = 3, and only there.

`docs/PHASE1_PR3_E026_OPERATOR_ORDER.md` records that E026's `P2_W0`
(`wiener_fill_all`) and `P3_Wlegacy` (`wiener_fill_first`) cannot be two
independent floors at the protocol's 3-plane context, and that the ordering
starts to matter from T = 5. The document first explained the T = 3 equality as
"the axial OTF is sampled at DC and the Nyquist pair", which is wrong for a
length-3 transform: the bins are {0, +1/3, -1/3}, and the equality holds because
those two non-DC coefficients are equal and opposite on the filled profile while
the Gaussian OTF is even in f_z.

These tests pin both halves of that statement, for every lattice phase, so the
document may state a derivation instead of a measurement alone.
"""

from __future__ import annotations

import numpy as np
import torch

from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import warm_start

#: The E026 kernel, in pixels: sigma (1.0, 1.5) px.
PSF = PSFConfig.from_pixels(1.0, 1.5, voxel_xy_um=0.98, voxel_z_um=5.0)
LAM = 0.01


def _window(depth: int, phase_z: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    dense = torch.from_numpy(rng.random((depth, 48, 48)).astype("float32"))
    spec = SamplingSpec(mode="yz", ky=2, kz=2)
    mask = spec.mask(dense.shape, phase=(phase_z, 0, 0)).bool()
    y = torch.where(mask, dense, torch.zeros_like(dense))
    return y, mask


def _centre(a: torch.Tensor, b: torch.Tensor) -> float:
    c = a.shape[0] // 2
    return float((a[c] - b[c]).abs().mean())


def test_the_two_wiener_orders_agree_on_the_centre_plane_at_t3():
    for phase_z in (0, 1):
        y, m = _window(3, phase_z)
        a = warm_start(y, m, PSF, LAM, mode="wiener_fill_all")
        b = warm_start(y, m, PSF, LAM, mode="wiener_fill_first")
        assert _centre(a, b) < 1e-6, (
            f"phase {phase_z}: centre-plane disagreement {_centre(a, b):.2e} at "
            "T=3; the operator-order note is no longer true")


def test_the_orders_differ_once_the_window_admits_interior_frequencies():
    # At T = 5 the ±1/5 and ±2/5 bins no longer cancel, and the ordering shows.
    y, m = _window(5, 0)
    a = warm_start(y, m, PSF, LAM, mode="wiener_fill_all")
    b = warm_start(y, m, PSF, LAM, mode="wiener_fill_first")
    assert _centre(a, b) > 1e-4, (
        "the two orders agree at T=5 as well, so the sweep in the operator-order "
        "note cannot be explained by window length alone")


def test_a_measured_centre_is_where_the_ordering_first_separates():
    """The phase split that the single-mean table used to hide."""
    y, m = _window(5, 0)
    centre = y.shape[0] // 2
    assert bool(m[centre].any()), "phase 0 makes the centre a measured plane"
    a = warm_start(y, m, PSF, LAM, mode="wiener_fill_all")
    b = warm_start(y, m, PSF, LAM, mode="wiener_fill_first")
    # A missing centre in the same window is still an exact agreement: its
    # profile is the two-knot interpolation in both orders.
    a_mix = warm_start(y, m, PSF, LAM, mode="wiener_fill_all")
    y2, m2 = _window(5, 1)
    assert bool(~m2[m2.shape[0] // 2].any()), "phase 1 makes the centre missing"
    a2 = warm_start(y2, m2, PSF, LAM, mode="wiener_fill_all")
    b2 = warm_start(y2, m2, PSF, LAM, mode="wiener_fill_first")
    assert _centre(a2, b2) < 1e-6
    assert _centre(a_mix, b) > 1e-4

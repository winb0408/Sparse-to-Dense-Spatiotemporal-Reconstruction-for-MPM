"""Sampling-conditioned reliability fields.

Three fields, each carrying a different claim (plan §4.2). They stay separate
in the code because they must stay separate in any claim:

``C_s``
    Sampling confidence: Gaussian-smoothed measurement support. Descriptive
    only -- "how much measured neighbourhood surrounds this voxel".
``U_geo``
    Geometric unreliability from the distance to the nearest measured sample,
    computed per axis with that axis's own sampling pitch, so a 5 um Z step is
    never treated like a 1 um lateral pixel.
``U_phys``
    Physics unreliability: the mask pushed through the optical transfer
    function's reliable-frequency support. Needs a calibrated (non-identity)
    PSF; with an identity PSF it degenerates to ``1 - M`` and should not be
    used as a physics claim.

None of these is a calibrated error estimate. Whether any of them predicts
reconstruction error is an experiment (SPUR-E006), not an assumption.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spur_mpm.physics.psf import PSFConfig, gaussian_otf_3d

__all__ = ["sampling_confidence", "geometric_uncertainty", "physics_uncertainty",
           "GEO_MODES"]


def sampling_confidence(mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """``C_s = G_sigma * M``, normalised so a fully sampled region maps to ~1.

    Replicate padding keeps the "fully sampled == 1" convention at the volume
    borders instead of fading to 0 there.
    """
    m = mask.float()
    if sigma <= 0:
        return m
    r = max(1, int(round(3 * sigma)))
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=mask.device)
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    x = F.pad(m[None, None], (r, r, r, r, r, r), mode="replicate")
    x = F.conv3d(x, g.view(1, 1, -1, 1, 1))     # along Z
    x = F.conv3d(x, g.view(1, 1, 1, -1, 1))     # along H
    x = F.conv3d(x, g.view(1, 1, 1, 1, -1))     # along W
    return x[0, 0].clamp(0.0, 1.0)


def _distance_to_measured(flags: torch.Tensor) -> torch.Tensor:
    """Distance, in samples, from every index to the nearest True in ``[n]``.

    Positions with no measurement anywhere get ``n`` so the field stays finite.
    """
    n = flags.numel()
    idx = torch.arange(n, dtype=torch.float32, device=flags.device)
    far = torch.full_like(idx, float(n))

    # Sentinels must be *outside* the index range in the direction each scan
    # propagates: -1 for the forward scan (max of last measured index) and +n
    # for the backward one (min of next measured index). A single large sentinel
    # for both would be absorbed by cummax and every index past the first
    # measurement would read distance n.
    pos_f = torch.where(flags, idx, torch.full_like(idx, -1.0))
    last = torch.cummax(pos_f, dim=0).values
    dist_fwd = torch.where(last >= 0, idx - last, far)

    pos_b = torch.where(flags, idx, far)
    nxt = torch.flip(torch.cummin(torch.flip(pos_b, [0]), dim=0).values, [0])
    dist_bwd = torch.where(nxt < n, nxt - idx, far)

    return torch.minimum(dist_fwd, dist_bwd)


#: ``per_axis``  distance expressed as a fraction of that axis's own pitch
#:               (isolates sampling geometry from optics);
#: ``physical``  distance in micrometres over the window's largest pitch
#:               (lets a 20 um axial gap weigh far more than a 2 um row gap).
GEO_MODES = ("per_axis", "physical")


def geometric_uncertainty(mask: torch.Tensor, ky: int = 2, kz: int = 1,
                          voxel_xy_um: float = 1.0, voxel_z_um: float = 1.0,
                          mode: str = "per_axis",
                          combine: str = "union") -> torch.Tensor:
    """``U_geo`` in [0, 1]: how far each voxel sits from any measurement.

    Parameters
    ----------
    mask   : ``[Z, H, W]`` bool measurement support.
    ky, kz : nominal row / plane spacing of the sampling protocol, used as the
             per-axis normalisation pitch.
    mode   : see :data:`GEO_MODES`.
    combine: ``union`` -- ``1 - prod(1 - u_ax)``, unreliable if *any* axis is
             far; ``mean`` -- arithmetic mean of the three axis fields.

    Both modes are monotone in distance and reach 1 at half a sampling period,
    the point farthest from any measured sample inside a gap.
    """
    if mode not in GEO_MODES:
        raise ValueError(f"unknown geo mode {mode!r}; expected {GEO_MODES}")
    d, h, w = mask.shape
    row_flags = mask.any(dim=(0, 2))     # [H]
    col_flags = mask.any(dim=(0, 1))     # [W]
    plane_flags = mask.any(dim=(1, 2))   # [Z]

    # Half-period is the worst-case distance inside a lattice gap; a dense axis
    # (pitch 1) has no gap and yields 0 everywhere.
    lateral_half = max(0.5, ky / 2.0)
    axial_half = max(0.5, kz / 2.0)

    if mode == "per_axis":
        u_row = (_distance_to_measured(row_flags) / lateral_half).clamp(0, 1)
        u_col = (_distance_to_measured(col_flags) / lateral_half).clamp(0, 1)
        u_pln = (_distance_to_measured(plane_flags) / axial_half).clamp(0, 1)
    else:
        # Physical distance in micrometres, normalised by the largest nominal
        # pitch present in this window, so axial and lateral gaps are directly
        # comparable and an empty axis saturates earliest.
        scale = max(ky * voxel_xy_um, kz * voxel_z_um, 1e-6)
        u_row = ((_distance_to_measured(row_flags) * voxel_xy_um) / scale).clamp(0, 1)
        u_col = ((_distance_to_measured(col_flags) * voxel_xy_um) / scale).clamp(0, 1)
        u_pln = ((_distance_to_measured(plane_flags) * voxel_z_um) / scale).clamp(0, 1)

    uy = u_row.view(1, h, 1)
    ux = u_col.view(1, 1, w)
    uz = u_pln.view(d, 1, 1)

    if combine == "mean":
        field = (uy.expand(d, h, w) + ux.expand(d, h, w) + uz.expand(d, h, w)) / 3.0
    elif combine == "union":
        field = 1.0 - (1 - uy) * (1 - ux) * (1 - uz)
    else:
        raise ValueError(f"unknown combine {combine!r}; expected union|mean")

    # A measured voxel carries its own sample, so its geometric unreliability is
    # exactly zero by definition -- the per-axis lattice fields above cannot see
    # this on their own (e.g. a measured voxel in an odd column of a row-only
    # lattice would otherwise inherit the column gap's distance).
    return field * (~mask.bool()).float()


def physics_uncertainty(mask: torch.Tensor, psf: PSFConfig,
                        lam: float = 0.01) -> torch.Tensor:
    """``U_phys = 1 - C_p`` with ``C_p = F^-1[A(f) F(M)] / A(0)``.

    ``A(f) = |H|^2 / (|H|^2 + lam)`` is the fraction of each frequency the optics
    preserve; filtering the mask through it says which voxels are in principle
    recoverable even from dense sampling. A(0) normalisation maps a fully
    sampled mask to exactly 1.
    """
    if psf.is_identity:
        return (~mask.bool()).float()
    d, h, w = mask.shape
    m = mask.float()[None]
    otf = gaussian_otf_3d((d, h, w), psf, device=mask.device)
    a = otf ** 2 / (otf ** 2 + lam)
    a_dc = float(a[0, 0, 0])
    mf = torch.fft.fftn(m, dim=(-3, -2, -1))
    cp = torch.fft.ifftn(a * mf, dim=(-3, -2, -1)).real / a_dc
    return (1.0 - cp[0]).clamp(0.0, 1.0)

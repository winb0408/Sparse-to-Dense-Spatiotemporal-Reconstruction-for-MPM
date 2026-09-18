"""Unified observation builder.

Every consumer (training dataset, evaluation, inference) obtains its model
inputs exclusively through :func:`build_observation`, guaranteeing one
observation model everywhere (PLAN.md P0-2).

Output contract (PLAN.md §5 naming):

    Y    network-grid observation (zero-filled sparse measurement)
    M    measurement support mask          (NOT uncertainty)
    X0   physics warm-start (Wiener/bilinear)
    C_s  sampling confidence               (geometry of M: distance to
          nearest measured sample, via Gaussian smoothing)
    U_p  physics-derived uncertainty       (OTF attenuation applied to M:
          how much of the local neighborhood survives the optics)
    Q    target/query mask                 (set by the dataset, not here)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from physics.acquisition import Observation, blur_3d_fft
from physics.psf import PSFConfig, gaussian_otf_3d
from physics.wiener import warm_start

__all__ = ["ModelInputs", "build_observation"]


@dataclass
class ModelInputs:
    y: torch.Tensor            # [T, H, W] (or [B, T, H, W])
    mask: torch.Tensor         # float, same shape
    x0: torch.Tensor
    confidence: torch.Tensor   # C_s
    uncertainty: torch.Tensor  # U_p
    metadata: dict = field(default_factory=dict)


def _confidence_from_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """C_s = G_sigma * M, normalized so a fully-sampled region maps to ~1.

    Uses replicate padding at the volume borders so the "fully sampled == 1"
    convention also holds at the edges (support continues beyond the FOV).
    """
    if sigma <= 0:
        return mask.float()
    m = mask.float()[None, None]
    r = max(1, int(3 * sigma))
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=mask.device)
    g = torch.exp(-ax**2 / (2 * sigma**2))
    g = g / g.sum()
    kw = g.view(1, 1, 1, 1, -1)
    kh = g.view(1, 1, 1, -1, 1)
    kt = g.view(1, 1, -1, 1, 1)
    m = torch.nn.functional.pad(m, (r, r, r, r, r, r), mode="replicate")
    m = torch.nn.functional.conv3d(m, kw)
    m = torch.nn.functional.conv3d(m, kh)
    m = torch.nn.functional.conv3d(m, kt)
    return m[0, 0].clamp(0.0, 1.0)


def _physics_uncertainty(mask: torch.Tensor, psf: PSFConfig, lam: float) -> torch.Tensor:
    """U_p = 1 - C_p, C_p = reliability-filtered mask.

    C_p = F^-1[ A(f) * F(M) ] / A(0),  A(f) = |H|^2 / (|H|^2 + lam).

    Dividing by the DC value A(0) makes a fully-sampled mask map to exactly 1
    (its spectrum lives only at DC).  Where the optics preserve no reliable
    frequency content or no nearby samples exist, C_p -> 0 and U_p -> 1.
    """
    T, H, W = mask.shape
    m = mask.float()[None]
    if psf.is_identity:
        cp = m.clone()
    else:
        otf = gaussian_otf_3d((T, H, W), psf, device=mask.device)
        A = otf**2 / (otf**2 + lam)          # frequency-domain reliability
        a_dc = float(A[0, 0, 0])
        Mf = torch.fft.fftn(m, dim=(-3, -2, -1))
        cp = torch.fft.ifftn(A * Mf, dim=(-3, -2, -1)).real / a_dc
    return (1.0 - cp[0]).clamp(0.0, 1.0)


def build_observation(obs: Observation, psf: PSFConfig | None = None,
                      lam: float = 0.01, warm_start_kind: str = "wiener",
                      confidence_sigma: float | None = None) -> ModelInputs:
    """Build model inputs from a raw sparse observation.

    Parameters
    ----------
    obs               : physics.acquisition.Observation (raw sparse Y + mask M).
    psf               : optical model used for warm-start deconvolution and
                        physics uncertainty. None / identity PSF => the
                        warm-start falls back to interpolation-only and U_p
                        reduces to the sampling-geometry term.
    lam               : Wiener regularization.
    warm_start_kind   : "wiener" | "bilinear".
    confidence_sigma  : Gaussian sigma (px) for C_s; default K/2.
    """
    y, mask = obs.y, obs.mask
    if psf is None:
        psf = PSFConfig()
    K = obs.metadata.get("acquisition", {}).get("K", 2)
    sigma_c = confidence_sigma if confidence_sigma is not None else max(1.0, K / 2.0)

    x0 = warm_start(y, mask, psf, lam, mode=warm_start_kind)
    c_s = _confidence_from_mask(mask, sigma_c)
    u_p = _physics_uncertainty(mask, psf, lam)

    return ModelInputs(
        y=y, mask=mask.float(), x0=x0, confidence=c_s, uncertainty=u_p,
        metadata={"warm_start": warm_start_kind, "lam": lam,
                  "acquisition": obs.metadata.get("acquisition", {})},
    )

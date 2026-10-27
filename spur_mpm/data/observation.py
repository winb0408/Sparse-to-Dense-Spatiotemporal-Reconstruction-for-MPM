"""The single observation builder: raw sparse measurement -> model inputs.

Every consumer (training dataset, evaluation, inference) builds model inputs
through :func:`build_model_inputs`, so there is exactly one definition of what
the network sees.

Output contract::

    Y    zero-filled sparse measurement        (measured positions only)
    M    measurement support, float {0,1}       (NOT uncertainty)
    X0   physics warm-start (Wiener / bilinear / fill-first)
    C_s  sampling confidence (smoothed support)
    U    sampling/physics unreliability in [0,1]

**Ordering rule (plan §5.2).** In self-supervised training the holdout split
must happen *before* this function is called: ``Y``/``M`` passed here are the
reduced ``Y_in``/``M_in``, so ``X0``, ``C_s`` and ``U`` can only ever be derived
from positions the network is allowed to see. Building the prior from the full
acquisition and then masking afterwards is the leak this ordering prevents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from spur_mpm.models.uncertainty import (
    geometric_uncertainty,
    physics_uncertainty,
    sampling_confidence,
)
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.wiener import warm_start

__all__ = ["FEATURES", "ModelInputs", "build_model_inputs"]

#: Input channel vocabulary; arms select a subset (plan §4.3).
FEATURES = ("y", "mask", "x0", "confidence", "uncertainty")

U_KINDS = ("geo", "phys", "mean")


@dataclass
class ModelInputs:
    y: torch.Tensor            # [Z, H, W]
    mask: torch.Tensor         # [Z, H, W] float
    x0: torch.Tensor           # [Z, H, W]
    confidence: torch.Tensor   # [Z, H, W]
    uncertainty: torch.Tensor  # [Z, H, W]
    metadata: dict = field(default_factory=dict)

    def stack(self, features: tuple[str, ...] | list[str]) -> torch.Tensor:
        """Stack selected features as channels, feature-major ``[F*Z, H, W]``.

        Feature-major (not slice-major) so a caller can always recover one
        feature with ``x.view(B, F, Z, H, W)[:, i]`` regardless of window depth.
        """
        return torch.cat([getattr(self, f).reshape(-1, *self.y.shape[1:])
                          for f in features], dim=0)


def build_model_inputs(y: torch.Tensor, mask: torch.Tensor,
                       psf: PSFConfig | None = None, lam: float = 0.01,
                       warm_start_kind: str = "wiener",
                       confidence_sigma: float | None = None,
                       u_kind: str = "geo", geo_mode: str = "per_axis",
                       ky: int = 1, kz: int = 1,
                       voxel_xy_um: float = 1.0, voxel_z_um: float = 1.0
                       ) -> ModelInputs:
    """Build ``(Y, M, X0, C_s, U)`` from a sparse observation.

    Parameters
    ----------
    y, mask            : zero-filled measurement and its support, ``[Z, H, W]``.
    psf                : optical model for the warm-start and ``U_phys``;
                         identity (default) means "the data already carries the
                         optics" and reduces the Wiener branches to gap filling.
    warm_start_kind    : ``bilinear`` | ``wiener`` | ``wiener_fill_first`` |
                         ``iterdc:<lam>:<steps>``.
    confidence_sigma   : Gaussian sigma (px) for ``C_s``; default half the row pitch.
    u_kind             : ``geo`` | ``phys`` | ``mean`` -- which unreliability
                         field fills the ``U`` channel.
    geo_mode           : ``per_axis`` | ``physical`` (see models.uncertainty).
    ky, kz             : sampling pitches, needed for the geometric field.
    """
    if psf is None:
        psf = PSFConfig()
    if u_kind not in U_KINDS:
        raise ValueError(f"unknown u_kind {u_kind!r}; expected {U_KINDS}")

    mask_bool = mask.bool()
    # Enforce blindness here rather than trusting the caller to have zeroed the
    # unsupported positions: if a held-out value ever rides along in ``y`` it
    # would reach the network. Everything below is derived from ``y_in`` only.
    y_in = y * mask_bool
    x0 = warm_start(y_in, mask_bool, psf, lam, mode=warm_start_kind)

    sigma_c = confidence_sigma if confidence_sigma is not None else max(1.0, ky / 2.0)
    c_s = sampling_confidence(mask_bool, sigma_c)

    if u_kind == "geo":
        u = geometric_uncertainty(mask_bool, ky=ky, kz=kz, voxel_xy_um=voxel_xy_um,
                                  voxel_z_um=voxel_z_um, mode=geo_mode)
    elif u_kind == "phys":
        u = physics_uncertainty(mask_bool, psf, lam)
    else:
        u_geo = geometric_uncertainty(mask_bool, ky=ky, kz=kz, voxel_xy_um=voxel_xy_um,
                                      voxel_z_um=voxel_z_um, mode=geo_mode)
        u = 0.5 * (u_geo + physics_uncertainty(mask_bool, psf, lam))

    return ModelInputs(
        y=y_in, mask=mask_bool.float(), x0=x0, confidence=c_s, uncertainty=u,
        metadata={"warm_start": warm_start_kind, "lam": lam, "u_kind": u_kind,
                  "geo_mode": geo_mode, "psf": psf.to_dict()},
    )

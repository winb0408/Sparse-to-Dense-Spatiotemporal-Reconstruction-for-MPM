"""The single observation builder: raw sparse measurement -> model inputs.

Every consumer (training dataset, evaluation, inference) builds model inputs
through :func:`build_model_inputs`, so there is exactly one definition of what
the network sees.

Output contract::

    Y          zero-filled sparse measurement      (measured positions only)
    M          measurement support, float {0,1}     (NOT uncertainty)
    X0         arm's warm-start (bilinear / wiener / fill-first / fill-all)
    C_s        sampling confidence (smoothed support)
    U          sampling/physics unreliability in [0,1]
    I0         Interp(Y, M) -- pure interpolation, no FFT          (PR2 §3)
    W0         Wiener(I0; H, lam), filled first                    (PR2 §3)
    dW         W0 - I0                                             (PR2 §3)
    u_s        sampling-geometry uncertainty (the old ``U`` under geo)
    u_p        relative candidate disagreement |W0-I0| / (|I0|+..)  -- a
               feature, not a calibrated probability                 (PR2 §5)
    valid_z    which stacked planes exist in the volume (padding is not a plane)

``X0`` is kept and is *not* redefined: old arms read it, old checkpoints were
trained on it, and ``I0``/``W0`` are new names for the two operators the plan
wants separated. ``x0`` equals ``interp`` or ``wiener`` only when the arm's
warm-start mode says so -- that equality is a property of the arm, not of this
builder.

**Ordering rule (plan §5.2).** In self-supervised training the holdout split
must happen *before* this function is called: ``Y``/``M`` passed here are the
reduced ``Y_in``/``M_in``, so every prior below can only ever be derived from
positions the network is allowed to see. Building the prior from the full
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
from spur_mpm.physics.prior import physics_prior
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.wiener import warm_start

__all__ = ["FEATURES", "STACKABLE", "ModelInputs", "build_model_inputs"]

#: Input channel vocabulary; arms select a subset (plan §4.3).
FEATURES = ("y", "mask", "x0", "confidence", "uncertainty",
            "interp", "wiener", "phys_delta", "u_sampling", "u_physics")

#: ``valid_z`` is per-plane, not per-voxel, so it cannot be a stacked channel.
STACKABLE = FEATURES

U_KINDS = ("geo", "phys", "mean")


@dataclass
class ModelInputs:
    y: torch.Tensor            # [Z, H, W]
    mask: torch.Tensor         # [Z, H, W] float
    x0: torch.Tensor           # [Z, H, W]
    confidence: torch.Tensor   # [Z, H, W]
    uncertainty: torch.Tensor  # [Z, H, W]
    interp: torch.Tensor = None      # [Z, H, W]  I0
    wiener: torch.Tensor = None      # [Z, H, W]  W0
    phys_delta: torch.Tensor = None  # [Z, H, W]  W0 - I0
    u_sampling: torch.Tensor = None  # [Z, H, W]
    u_physics: torch.Tensor = None   # [Z, H, W]
    valid_z: torch.Tensor = None     # [Z] bool
    metadata: dict = field(default_factory=dict)

    def feature(self, name: str) -> torch.Tensor:
        """One feature, resolved through the old-to-new aliases."""
        if name not in STACKABLE:
            raise KeyError(f"{name!r} is not a stackable feature; "
                           f"expected one of {STACKABLE}")
        alias = {"x0": "x0", "u_sampling": "u_sampling", "u_physics": "u_physics",
                 "phys_delta": "phys_delta", "interp": "interp",
                 "wiener": "wiener"}.get(name, name)
        tensor = getattr(self, alias, None)
        if tensor is None:
            raise KeyError(f"feature {name!r} was not computed by this build; "
                           "the caller asked for a channel it did not request")
        return tensor

    def stack(self, features: tuple[str, ...] | list[str]) -> torch.Tensor:
        """Stack selected features as channels, feature-major ``[F*Z, H, W]``.

        Feature-major (not slice-major) so a caller can always recover one
        feature with ``x.view(B, F, Z, H, W)[:, i]`` regardless of window depth.
        """
        return torch.cat([self.feature(f).reshape(-1, *self.y.shape[1:])
                          for f in features], dim=0)


def build_model_inputs(y: torch.Tensor, mask: torch.Tensor,
                       psf: PSFConfig | None = None, lam: float = 0.01,
                       warm_start_kind: str = "wiener",
                       confidence_sigma: float | None = None,
                       u_kind: str = "geo", geo_mode: str = "per_axis",
                       ky: int = 1, kz: int = 1,
                       voxel_xy_um: float = 1.0, voxel_z_um: float = 1.0,
                       psf_source: str | None = None,
                       valid_z: torch.Tensor | None = None,
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
    psf_source         : provenance label for a non-identity PSF (PR2 §2). A
                         non-identity PSF without one is refused by
                         :func:`spur_mpm.physics.prior.physics_prior`.
    valid_z            : ``[Z]`` bool, False where the window is padded beyond
                         the physical volume. Recorded so a downstream consumer
                         can tell "no measurement" from "no plane".

    Returns
    -------
    :class:`ModelInputs` with both the old channels (``x0``, ``confidence``,
    ``uncertainty``) and the PR2 ones (``interp``, ``wiener``, ``phys_delta``,
    ``u_sampling``, ``u_physics``). The PR2 channels cost one extra
    interpolation pass; under an identity PSF the Wiener step is a copy, so the
    overhead is the gap filling the builder was doing anyway.
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
    prior = physics_prior(y_in, mask_bool, psf, lam, psf_source=psf_source)

    sigma_c = confidence_sigma if confidence_sigma is not None else max(1.0, ky / 2.0)
    c_s = sampling_confidence(mask_bool, sigma_c)

    u_geo = geometric_uncertainty(mask_bool, ky=ky, kz=kz, voxel_xy_um=voxel_xy_um,
                                  voxel_z_um=voxel_z_um, mode=geo_mode)
    u_phys = physics_uncertainty(mask_bool, psf, lam)
    if u_kind == "geo":
        u = u_geo
    elif u_kind == "phys":
        u = u_phys
    else:
        u = 0.5 * (u_geo + u_phys)

    if valid_z is None:
        valid_z = torch.ones(y_in.shape[0], dtype=torch.bool)
    else:
        valid_z = valid_z.bool().reshape(-1)
        if valid_z.shape != (y_in.shape[0],):
            raise ValueError(f"valid_z shape {tuple(valid_z.shape)} != "
                             f"({y_in.shape[0]},)")

    metadata = {"warm_start": warm_start_kind, "lam": lam, "u_kind": u_kind,
                "geo_mode": geo_mode, "psf": psf.to_dict(),
                "psf_source": prior["metadata"]["psf_source"],
                "voxel_xy_um": voxel_xy_um, "voxel_z_um": voxel_z_um,
                "ky": ky, "kz": kz,
                "phys_delta_zero": prior["metadata"]["phys_delta_zero"]}
    return ModelInputs(
        y=y_in, mask=mask_bool.float(), x0=x0, confidence=c_s, uncertainty=u,
        interp=prior["interp"], wiener=prior["wiener"],
        phys_delta=prior["phys_delta"], u_sampling=u_geo,
        u_physics=prior["u_physics"], valid_z=valid_z, metadata=metadata,
    )

"""The physics-prior bundle handed to the reconstruction network.

Three tensors and their provenance, computed once and shared by the dataset,
the evaluation and the inference path (PR2 §3)::

    I0          = Interp(Y, M)                     pure interpolation
    W0          = Wiener(I0; H, lambda)            fill-first deconvolution
    phys_delta  = W0 - I0                          what the optical model changed

plus ``u_physics``, the *relative disagreement* between the two candidates.
That last one is a feature, not a calibrated probability: it says "the physics
candidate moved a lot here", which is a reason to look, not a reason to trust.

Two separations this module exists to keep straight (PR2 §2, §6):

* ``H == I`` (the real 3P stacks already carry their optics) makes
  ``W0 == I0`` **exactly**, and ``phys_delta`` identically zero. That is the
  measured situation on this cohort, so no result here may be read as evidence
  for or against Wiener deconvolution -- it is evidence about gap filling.
* A non-identity ``H`` must carry a source label saying where it came from
  (``calibrated_optical`` or ``synthetic_extra_degradation``). An unlabelled
  non-identity PSF is refused, because the two sources license different
  conclusions and mixing them is how a stress test becomes a headline.

:func:`mask_aware_inverse` is the optional fourth operator from the plan: it
solves ``min_X 0.5||M(HX - Y)||^2 + lam/2||X - I0||^2``, i.e. it pulls
*unmeasured* voxels toward the **interpolated prior**, not toward zero. The old
``iterative_dc_inverse`` is deliberately left untouched (its ``+ lam*x`` term
shrinks toward zero, which its docstring once described as shrinking toward the
prior -- a documentation bug with a numerically visible consequence).
"""

from __future__ import annotations

import torch

from spur_mpm.physics.psf import PSFConfig, gaussian_otf_3d
from spur_mpm.physics.wiener import (
    _check_prior_args,
    interp_linear,
    wiener_deconv_3d,
)

__all__ = [
    "PSF_SOURCES",
    "mask_aware_inverse",
    "physics_prior",
    "prior_summary",
    "psf_source_label",
]

#: Where a non-identity ``H`` may legitimately come from (PR2 §2).
PSF_SOURCES = ("identity", "calibrated_optical", "synthetic_extra_degradation",
               "unlabelled_identity_assumed")


def _source_for(psf: PSFConfig, psf_source: str | None) -> str:
    """Name the optical model, and refuse to invent one."""
    if psf_source is not None and psf_source not in PSF_SOURCES:
        raise ValueError(f"unknown psf_source {psf_source!r}; expected one of "
                         f"{PSF_SOURCES}")
    if psf.is_identity:
        return "identity"
    if psf_source in (None, "identity"):
        raise ValueError(
            "a non-identity PSF needs an explicit psf_source "
            "('calibrated_optical' or 'synthetic_extra_degradation'): the two "
            "license different conclusions, and an unlabelled H is how a "
            "degradation stress test ends up in the main table")
    return psf_source


def psf_source_label(psf: PSFConfig | None = None,
                     psf_source: str | None = None) -> str:
    """Resolved provenance label for ``H``, refused exactly as the prior refuses.

    Public so a run summary can record *which* optical model produced the number
    without computing a prior for the sake of a string, and without a second
    copy of the labelling rule that could drift from this one.
    """
    return _source_for(PSFConfig() if psf is None else psf, psf_source)


def physics_prior(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig | None = None,
                  lam: float = 0.01, psf_source: str | None = None,
                  ) -> dict:
    """Compute ``I0``, ``W0``, ``phys_delta`` and their provenance.

    ``y`` must already be the *blind* observation (measured positions only);
    nothing here reads a dense reference, and the function would have no way
    to: it is not passed one.
    """
    if psf is None:
        psf = PSFConfig()
    _check_prior_args(y, mask, psf, lam)
    source = _source_for(psf, psf_source)

    mask_b = mask.bool()
    y_in = torch.where(mask_b, y, torch.zeros_like(y))
    i0 = interp_linear(y_in, mask_b)
    # ``wiener_fill_all`` re-derives I0 from (Y, M); computing it once here and
    # branching keeps the two candidates from drifting apart through a second
    # interpolation pass, and makes the identity case an exact copy.
    w0 = i0.clone() if psf.is_identity else wiener_deconv_3d(i0, psf, lam)
    if not bool(torch.isfinite(w0).all()):
        raise ValueError("Wiener fill-all produced non-finite values; check the "
                         "PSF sigmas, lambda and the input range")
    delta = w0 - i0

    # Relative disagreement: |W0 - I0| against the magnitude already present.
    # Zero when the two candidates agree *including at zero*, which is the
    # identity-PSF case -- and then there is nothing for a gate to weigh.
    denom = i0.abs() + delta.abs() + 1e-6
    u_physics = torch.clamp(delta.abs() / denom, 0.0, 1.0)

    meta = {
        "psf_source": source,
        "psf": psf.to_dict(),
        "lam": float(lam),
        "lambda_note": "Wiener regularisation, applied to the filled volume",
        "phys_delta_zero": bool(not bool(delta.any())),
        "u_physics_kind": "relative candidate disagreement in [0,1]; NOT a "
                          "calibrated error probability",
        "operator": "I0=Interp(Y,M); W0=Wiener(I0;H,lam) (fill first, then FFT)",
    }
    return {"interp": i0, "wiener": w0, "phys_delta": delta,
            "u_physics": u_physics, "metadata": meta}


def mask_aware_inverse(y: torch.Tensor, mask: torch.Tensor, interp: torch.Tensor,
                       psf: PSFConfig | None = None, lam: float = 0.01,
                       steps: int = 16, step_size: float | None = None) -> torch.Tensor:
    """Solve ``min_X 0.5||M(HX - Y)||^2 + lam/2 ||X - I0||^2`` by gradient descent.

    Differences from :func:`spur_mpm.physics.wiener.iterative_dc_inverse`:

    * the regulariser is centred on ``I0`` (``X - I0``), so unmeasured voxels
      are *held at the interpolated value* rather than shrunk toward zero. With
      ``H = I`` and no measurement, the fixed point is exactly ``I0``;
    * measured voxels are driven toward ``Y`` with weight ``1/(1+lam)``, so the
      operator does not silently zero out what the scanner actually saw;
    * the old function is left alone, so old checkpoints keep their semantics.

    The step is ``1/(1+lam)``: the Lipschitz constant of the gradient is at most
    ``||M|| * ||H||^2 + lam <= 1 + lam`` for a Gaussian OTF (``|H| <= 1``).
    """
    if psf is None:
        psf = PSFConfig()
    _check_prior_args(y, mask, psf, lam)
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if y.shape != interp.shape:
        raise ValueError(f"y {tuple(y.shape)} and interp {tuple(interp.shape)} "
                         "disagree")
    mask_b = mask.bool()
    m = mask_b.to(y.dtype)
    y_in = torch.where(mask_b, y, torch.zeros_like(y))

    x = interp.clone()
    eta = float(step_size) if step_size is not None else 1.0 / (1.0 + float(lam))

    if psf.is_identity:
        for _ in range(int(steps)):
            x = x - eta * (m * (x - y_in) + lam * (x - interp))
        return x

    otf = gaussian_otf_3d(tuple(y.shape), psf, device=y.device)
    otf_c = torch.conj(otf)
    for _ in range(int(steps)):
        hx = torch.fft.ifftn(otf * torch.fft.fftn(x, dim=(-3, -2, -1)),
                             dim=(-3, -2, -1)).real
        res = torch.fft.ifftn(otf_c * torch.fft.fftn(m * (hx - y_in),
                                                     dim=(-3, -2, -1)),
                              dim=(-3, -2, -1)).real
        x = x - eta * (res + lam * (x - interp))
    if not bool(torch.isfinite(x).all()):
        raise ValueError("mask-aware inverse diverged (non-finite); lower the "
                         "step or raise lambda")
    return x


def prior_summary(prior: dict) -> dict:
    """JSON-able summary for a run log: what was computed, not the tensors."""
    i0, w0, d = prior["interp"], prior["wiener"], prior["phys_delta"]
    return {
        **prior["metadata"],
        "interp_range": [float(i0.min()), float(i0.max())],
        "wiener_range": [float(w0.min()), float(w0.max())],
        "phys_delta_abs_mean": float(d.abs().mean()),
        "phys_delta_abs_max": float(d.abs().max()),
        "u_physics_mean": float(prior["u_physics"].mean()),
    }

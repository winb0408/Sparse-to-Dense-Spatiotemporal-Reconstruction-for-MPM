"""Physics warm-start ``X0``: anisotropic Wiener deconvolution + gap filling.

Ported from PSRAS/physics/wiener.py (depth axis renamed, adaptive-lambda tuned
for the old dataset removed). The estimate is

    X_W = F^-1 [ H* / (|H|^2 + lambda) * F(Y) ]

with ``H`` the anisotropic OTF from :mod:`spur_mpm.physics.psf`.

**Read this before interpreting any result.** In image domain A the PSF is the
identity (the real stack already carries the optics). With ``H = 1`` the Wiener
estimate collapses to interpolation alone, and every "Wiener" arm degenerates
to the bilinear arm. That collapse is a *finding to report*, not a bug to hide:
it means the physics prior added nothing beyond gap filling on this data
(plan §8.2). A genuine Wiener gain requires either a calibrated PSF (domain B)
or an explicit, labelled degradation stress test.

Warm-start policies per sampling mode:

``y``   lateral interpolation between measured rows, then optional lateral
        deconvolution (X stays dense, so rows are the natural knots).
``z``   deconvolve measured planes, then interpolate across the depth gap:
        deconvolution cannot invent a missing plane.
``yz``  lateral deconvolution of measured planes first, then axial
        interpolation between deconvolved knots.
"""

from __future__ import annotations

import torch

from spur_mpm.physics.psf import PSFConfig

__all__ = ["wiener_deconv_3d", "interp_linear", "WARM_START_MODES", "warm_start"]

#: ``wiener_fill_all`` is the *new* fill-first branch (PR2 §T1): every missing
#: axis is interpolated before any FFT. ``wiener_fill_first`` is the old one,
#: which only fills lateral holes, and is kept unchanged because checkpoints and
#: published numbers were produced with it. For a Y-only mask the two agree; for
#: Z / YZ they do not, and the difference is the whole point of the fix.
WARM_START_MODES = ("bilinear", "wiener", "wiener_fill_first", "wiener_fill_all",
                    "iterdc")


def wiener_deconv_3d(y: torch.Tensor, psf: PSFConfig, lam: float = 0.01) -> torch.Tensor:
    """Anisotropic Wiener deconvolution of ``[Z, H, W]`` or ``[B, Z, H, W]``.

    Unmeasured positions must already be filled (zero-filled input rings badly).
    """
    if psf.is_identity:
        return y
    squeeze = y.ndim == 3
    if squeeze:
        y = y[None]
    d, h, w = y.shape[-3:]
    yf = torch.fft.fftn(y.float(), dim=(-3, -2, -1))
    from spur_mpm.physics.psf import gaussian_otf_3d
    otf = gaussian_otf_3d((d, h, w), psf, device=y.device)[None]
    x = torch.fft.ifftn(torch.conj(otf) / (otf.abs() ** 2 + lam) * yf,
                        dim=(-3, -2, -1)).real
    return x[0] if squeeze else x


def _depth_interp(vol: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    """Linear interpolation along Z using measured planes as knots.

    ``vol`` [Z, H, W] holds arbitrary values on unmeasured planes;
    ``measured`` is a bool ``[Z]``. Outside the measured range the nearest
    knot is replicated.
    """
    z = vol.shape[0]
    idx = torch.nonzero(measured, as_tuple=False).flatten()
    out = vol.clone()
    if idx.numel() == 0:
        return out
    first, last = int(idx[0]), int(idx[-1])
    for t in range(first, last + 1):
        if measured[t]:
            continue
        t0 = int(idx[idx <= t][-1])
        t1 = int(idx[idx >= t][0])
        if t1 == t0:
            continue
        w = (t - t0) / (t1 - t0)
        out[t] = (1 - w) * vol[t0] + w * vol[t1]
    if first > 0:
        out[:first] = vol[first]
    if last < z - 1:
        out[last + 1:] = vol[last]
    return out


def _lateral_interp(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill lateral holes by linear interpolation between fully measured rows.

    Under row-structured masks (``y``/``grid``) the fully measured rows form a
    global lattice, so the fill vectorises over planes and columns. If no row
    is complete (scattered masks), rows with any measurement act as knots and
    their missing pixels count as zeros -- a documented approximation.
    Measured pixels are never overwritten by a fill.
    """
    h = vol.shape[1]
    row_knot = mask.all(dim=(0, -1))
    if not bool(row_knot.any()):
        row_knot = mask.amax(dim=(0, -1))
    idx = torch.nonzero(row_knot, as_tuple=False).flatten()
    out = vol.clone()
    if idx.numel() == 0:
        return out
    n = idx.numel()
    pos = torch.arange(h, device=vol.device)
    right = torch.searchsorted(idx, pos, right=True)
    sel = (right > 0) & (right < n) & ~row_knot.to(torch.bool)
    if sel.any():
        r0 = idx[(right[sel] - 1).clamp(min=0)]
        r1 = idx[right[sel].clamp(max=n - 1)]
        w = ((pos[sel] - r0).float() / (r1 - r0).float()).view(1, -1, 1)
        out[:, sel, :] = (1 - w) * vol[:, r0, :] + w * vol[:, r1, :]
    first, last = int(idx[0]), int(idx[-1])
    if first > 0:
        out[:, :first, :] = vol[:, first:first + 1, :]
    if last < h - 1:
        out[:, last + 1:, :] = vol[:, last:last + 1, :]
    return torch.where(mask.to(torch.bool), vol, out)


def iterative_dc_inverse(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
                         lam: float = 0.01, steps: int = 5,
                         step_size: float | None = None) -> torch.Tensor:
    """Gradient descent on ``0.5||M(Hx - Y)||^2 + 0.5*lam*||x||^2``.

    Initialised from the bilinear warm-start so missing regions carry
    interpolated structure before the first step. With an identity PSF the
    problem still constrains measured voxels and shrinks the rest toward the
    interpolated prior.
    """
    from spur_mpm.physics.psf import gaussian_otf_3d

    squeeze = y.ndim == 3
    if squeeze:
        y = y[None]
    d, h, w = y.shape[-3:]
    otf = gaussian_otf_3d((d, h, w), psf, device=y.device)[None]
    m = mask[None].float() if mask.ndim == 3 else mask.float()

    x = torch.stack([warm_start(y[b], mask if mask.ndim == 3 else mask[b],
                                psf, lam, mode="bilinear") for b in range(y.shape[0])])

    eta = step_size if step_size is not None else 1.0 / (1.0 + lam)
    for _ in range(steps):
        xf = torch.fft.fftn(x.float(), dim=(-3, -2, -1))
        hx = torch.fft.ifftn(otf * xf, dim=(-3, -2, -1)).real
        r = m * (hx - y.float())
        rf = torch.fft.fftn(r, dim=(-3, -2, -1))
        grad = torch.fft.ifftn(torch.conj(otf) * rf, dim=(-3, -2, -1)).real + lam * x
        x = x - eta * grad
    return x[0] if squeeze else x


def warm_start(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
               lam: float = 0.01, mode: str = "wiener") -> torch.Tensor:
    """Physics warm-start ``X0 = W(Y, M; H)`` for a ``[Z, H, W]`` observation.

    Parameters
    ----------
    y     : zero-filled sparse measurement ``[Z, H, W]``.
    mask  : bool measurement support, same shape.
    psf   : optical model. Identity => the Wiener branches reduce to filling.
    mode  : ``bilinear`` | ``wiener`` | ``wiener_fill_first`` | ``iterdc:<lam>:<steps>``.
    """
    if mode == "bilinear":
        return _bilinear(y, mask)

    if mode.startswith("iterdc"):
        parts = mode.split(":")
        lam_i = float(parts[1]) if len(parts) > 1 else lam
        steps_i = int(parts[2]) if len(parts) > 2 else 5
        return iterative_dc_inverse(y, mask, psf, lam=lam_i, steps=steps_i)

    if mode not in ("wiener", "wiener_fill_first", "wiener_fill_all"):
        raise ValueError(f"unknown warm-start mode {mode!r}; expected one of "
                         f"{WARM_START_MODES}")

    if mode == "wiener_fill_all":
        return wiener_fill_all(y, mask, psf, lam)

    if mode == "wiener_fill_first":
        # Deconvolving zero-filled holes rings (Gibbs), so fill the
        # measurement-domain holes first and deconvolve the filled volume.
        filled = _lateral_interp(y, mask)
        return _wiener_pipeline(filled, mask, psf, lam, lateral_holes=False)

    return _wiener_pipeline(y, mask, psf, lam)


def interp_linear(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``I0 = Interp(Y, M)``: pure linear interpolation, no FFT, no network.

    This is the public name for the operator the ``bilinear`` arm has always
    used. It fills **every** missing axis (Z first, then laterally), so the
    result has no zero-valued hole left for a later FFT to ring on -- which is
    what makes it usable as the prior ``W0`` is computed from.

    Measured voxels are never overwritten, so ``I0[M] == Y[M]`` exactly.
    """
    if y.shape != mask.shape:
        raise ValueError(f"y {tuple(y.shape)} and mask {tuple(mask.shape)} disagree")
    mask_b = mask.bool()
    mask_z = mask_b.amax(dim=(-2, -1))
    out = _depth_interp(y, mask_z) if not bool(mask_z.all()) else y
    if bool((~mask_b).any()):
        out = _lateral_interp(out, mask_b)
    return out


def _bilinear(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return interp_linear(y, mask)


def wiener_fill_all(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
                    lam: float = 0.01) -> torch.Tensor:
    """``W0 = Wiener(I0; H, lam)``: fill every hole first, then deconvolve.

    The order matters for more than numerics. Convolving a zero-filled hole
    with a blur kernel pulls zeros *into* measured neighbourhoods (Gibbs
    ringing plus a systematic darkening at the edge of the support), and the
    damage is largest exactly where the arm is supposed to be inventing
    structure. Filling first means the FFT only ever sees a complete image, so
    the only thing the deconvolution can do is redistribute intensity that is
    already there.

    With an identity PSF this returns ``I0`` bit-for-bit (no FFT is run at
    all): there is no optical model to invert, so claiming a "Wiener gain"
    under ``H = I`` would be claiming that gap filling beats gap filling.
    """
    _check_prior_args(y, mask, psf, lam)
    i0 = interp_linear(y, mask)
    if psf.is_identity:
        return i0.clone()
    out = wiener_deconv_3d(i0, psf, lam)
    if not bool(torch.isfinite(out).all()):
        raise ValueError("Wiener fill-all produced non-finite values; check the "
                         "PSF sigmas, lambda and the input range")
    return out


def _check_prior_args(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
                      lam: float) -> None:
    """Reject the inputs that make the inverse problem meaningless.

    ``lam <= 0`` divides by ``|H|^2 + lam``, which is zero wherever the OTF has
    a null (every real PSF has them) -- that is an Inf, not a small number. A
    negative or non-finite sigma is not a PSF at all. Both are configuration
    errors and belong at the top of the stack trace, not in a NaN loss.
    """
    if not isinstance(lam, (int, float)) or not float(lam) > 0:
        raise ValueError(f"lambda must be a positive finite number, got {lam!r}")
    if not float(lam) < float("inf"):
        raise ValueError(f"lambda must be finite, got {lam!r}")
    for name in ("sigma_xy_um", "sigma_z_um"):
        v = getattr(psf, name)
        if not (float(v) >= 0.0) or not float(v) < float("inf"):
            raise ValueError(f"psf.{name} must be finite and >= 0, got {v!r}")
    if y.shape != mask.shape:
        raise ValueError(f"y {tuple(y.shape)} and mask {tuple(mask.shape)} disagree")
    if not bool(torch.isfinite(y).all()):
        raise ValueError("the measurement contains non-finite values")


def _wiener_pipeline(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
                     lam: float, lateral_holes: bool | None = None) -> torch.Tensor:
    mask_z = mask.amax(dim=(-2, -1))
    axial_gaps = not bool(mask_z.all())
    if lateral_holes is None:
        lateral_holes = bool((~mask).any())

    if psf.is_identity:
        # Nothing to deconvolve: the physics prior is exactly gap filling.
        out = _depth_interp(y, mask_z) if axial_gaps else y
        if lateral_holes:
            out = _lateral_interp(out, mask)
        return out

    if axial_gaps:
        # Deconvolve measured planes laterally first, then interpolate between
        # deconvolved knots: interpolating raw data would blur twice.
        lat = psf.lateral_only()
        out = y.clone()
        for t in range(y.shape[0]):
            if mask_z[t]:
                out[t] = wiener_deconv_3d(out[t][None], lat, lam)[0]
        out = _depth_interp(out, mask_z)
        if lateral_holes:
            out = _lateral_interp(out, mask)
        return out

    out = wiener_deconv_3d(y, psf, lam)
    if lateral_holes:
        out = _lateral_interp(out, mask)
    return out

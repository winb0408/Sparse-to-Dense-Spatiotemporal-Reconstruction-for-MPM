"""True anisotropic 3D Wiener deconvolution (PLAN.md P0-4).

Given the observation model Y = M . (H*X + n), the Wiener initial estimate is

    X_W = F^-1 [ H* / (|H|^2 + lambda) * F(Y) ]

with H the anisotropic 3D OTF (psf.gaussian_otf_3d).  lambda plays the role of
the noise-to-signal regularization K; it is selected on validation data from
{0.001, 0.005, 0.01, 0.05, 0.1}.

Warm-start policy per sampling mode (PLAN.md DEC-003):
- xy  : per-slice lateral deconvolution (sigma_z = 0 OTF) -- the 3D Wiener
        with sigma_z = 0 reduces exactly to independent 2D Wiener per slice.
- z   : lateral deconvolution of measured slices, then axial linear
        interpolation across missing slices (axial gaps are an inpainting
        problem; deconvolution does not address them).
- xyz : axial zero-fill/interp first, then full anisotropic 3D Wiener.
"""

from __future__ import annotations

import torch

from .psf import PSFConfig, gaussian_otf_3d

__all__ = [
    "wiener_deconv_3d",
    "reliability_kernel",
    "warm_start",
]


def wiener_deconv_3d(y: torch.Tensor, psf: PSFConfig, lam: float = 0.01) -> torch.Tensor:
    """Anisotropic Wiener deconvolution of a volume or batch of volumes.

    Parameters
    ----------
    y   : [T, H, W] or [B, T, H, W] observation (zero-filled where unmeasured).
    psf : PSFConfig; sigma=0 on all axes gives the identity operator.
    lam : Wiener regularization (noise-to-signal ratio K).

    Returns
    -------
    Same shape as ``y`` (real float32).
    """
    if psf.is_identity:
        return y
    squeeze = y.ndim == 3
    if squeeze:
        y = y[None]
    B, T, H, W = y.shape
    # FFT over the last three (spatial+z) dims.
    Yf = torch.fft.fftn(y.float(), dim=(-3, -2, -1))
    otf = gaussian_otf_3d((T, H, W), psf, device=y.device, dtype=torch.float32)
    Hs = otf[None]  # [1, T, H, W]
    Xf = torch.conj(Hs) / (Hs.abs() ** 2 + lam) * Yf
    x = torch.fft.ifftn(Xf, dim=(-3, -2, -1)).real
    return x[0] if squeeze else x


def reliability_kernel(psf: PSFConfig, lam: float,
                       shape: tuple[int, int, int], device=None) -> torch.Tensor:
    """Spatial kernel of reliable frequency support, from OTF attenuation.

    A(f) = |H(f)|^2 / (|H(f)|^2 + lam); its inverse FFT is the kernel that
    maps the sampling mask to a physics-derived confidence field
    (data/observation.py).  Frequencies the optics never preserve are
    excluded, so the kernel reflects what is actually recoverable.
    """
    T, H, W = shape
    otf = gaussian_otf_3d((T, H, W), psf, device=device)
    A = otf**2 / (otf**2 + lam)
    k = torch.fft.ifftn(A).real
    return torch.roll(k, shifts=(T // 2, H // 2, W // 2), dims=(-3, -2, -1))


def _axial_interp(vol: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    """Linear interpolation along T using measured slices as knots.

    vol : [T, H, W] with arbitrary values on unmeasured slices.
    measured : [T] bool.
    """
    T = vol.shape[0]
    idx = torch.nonzero(measured, as_tuple=False).flatten()
    out = vol.clone()
    if idx.numel() == 0:
        return out
    first, last = int(idx[0]), int(idx[-1])
    # Interior gaps: for each unmeasured slice between two measured knots.
    for t in range(first, last + 1):
        if measured[t]:
            continue
        below = idx[idx <= t]
        above = idx[idx >= t]
        t0 = int(below[-1])
        t1 = int(above[0])
        if t1 == t0:
            continue
        w = (t - t0) / (t1 - t0)
        out[t] = (1 - w) * vol[t0] + w * vol[t1]
    # Outside the measured range: nearest measured slice.
    if first > 0:
        out[:first] = vol[first]
    if last < T - 1:
        out[last + 1:] = vol[last]
    return out


def _lateral_interp(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill lateral holes by linear interpolation along H between knot rows.

    Knot rows are rows measured in full (y / grid patterns: kept rows carry
    complete measurements). Under these row-structured masks the knot pattern
    is global, so the interpolation vectorizes over slices and columns.
    If no row is fully measured (e.g. random point masks), rows with any
    measurement act as knots and their unmeasured pixels count as zero knots
    (documented approximation for unstructured masks).
    """
    H = vol.shape[1]
    row_knot = mask.all(dim=(0, -1))                  # [H] fully-measured rows
    if not bool(row_knot.any()):
        row_knot = mask.amax(dim=(0, -1))             # fallback: any-measurement
    idx = torch.nonzero(row_knot, as_tuple=False).flatten()
    out = vol.clone()
    if idx.numel() == 0:
        return out
    n = idx.numel()
    pos = torch.arange(H, device=vol.device)
    right = torch.searchsorted(idx, pos, right=True)  # knots above each row
    sel = (right > 0) & (right < n) & ~row_knot.to(torch.bool)
    if sel.any():
        t0 = idx[(right[sel] - 1).clamp(min=0)]
        t1 = idx[right[sel].clamp(max=n - 1)]
        w = ((pos[sel] - t0).float() / (t1 - t0).float()).view(1, -1, 1)
        out[:, sel, :] = (1 - w) * vol[:, t0, :] + w * vol[:, t1, :]
    # Outside the measured range: nearest knot row.
    first, last = int(idx[0]), int(idx[-1])
    if first > 0:
        out[:, :first, :] = vol[:, first:first + 1, :]
    if last < H - 1:
        out[:, last + 1:, :] = vol[:, last:last + 1, :]
    # Measured pixels always keep their value -- fills never overwrite data.
    return torch.where(mask.to(torch.bool), vol, out)


def _adaptive_lambda(y: torch.Tensor, mask: torch.Tensor, lam0: float) -> torch.Tensor:
    """Per-slice noise-adaptive Wiener lambda (EXP-023 / DEC-026).

    lam_eff = lam0 * (1 - rho_t) with rho_t = measured fraction of slice t.
    Densely measured slices get near-frozen deconvolution (rho -> 1) while
    mostly missing slices get heavy regularization -- the EXP-022 lesson
    (frozen z-tuned lambda collapsed to Gibbs ringing on zero-filled
    lateral holes) made quantitative: holes have rho = 0, so lam_eff = lam0.
    """
    rho = mask.float().mean(dim=(-2, -1))              # [T]
    return lam0 * (1.0 - rho)


def iterative_dc_inverse(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
                         lam: float = 0.01, steps: int = 5,
                         step_size: float | None = None) -> torch.Tensor:
    """Iterative data-consistent inversion prior (EXP-023 P4).

    Gradient descent on 0.5*|| M.(H*X - Y) ||^2 + 0.5*lam*||X||^2 with the
    analytic Gaussian OTF (H*X and H^T*X both = FFT convolution since the
    Gaussian PSF is symmetric).  Initialization: bilinear warm-start, so
    missing regions carry interpolated structure before the first step.

    Parameters
    ----------
    y    : [T, H, W] zero-filled observation.
    mask : [T, H, W] bool measurement support.
    psf  : optical model (identity => gradient steps still apply data
           consistency on the mask, deconvolution term becomes identity).
    steps: number of gradient steps.
    """
    squeeze = y.ndim == 3
    if squeeze:
        y = y[None]
    B, T, H, W = y.shape
    m = mask[None].float() if mask.ndim == 3 else mask.float()
    dev = y.device
    otf = gaussian_otf_3d((T, H, W), psf, device=dev, dtype=torch.float32)[None]
    Hs = otf
    x = _axial_interp(y if squeeze else y, mask.amax(dim=(-2, -1)) > 0.5) if False else None
    # init: bilinear warm start (fills both axial gaps and lateral holes)
    init = torch.zeros_like(y.float())
    for b in range(B):
        mb = mask if mask.ndim == 3 else mask[b]
        init[b] = warm_start(y[b], mb, psf, lam, mode="bilinear")
    x = init
    # Lipschitz bound of M.H^T.H.M + lam*I: at most ||H||^2 + lam = 1 + lam.
    L = 1.0 + lam
    eta = step_size if step_size is not None else 1.0 / L
    for _ in range(steps):
        # grad = H^T M (M H x - M y) + lam x ; H^T = conj(OTF) mult, H = OTF mult
        Xf = torch.fft.fftn(x.float(), dim=(-3, -2, -1))
        hx = torch.fft.ifftn(Hs * Xf, dim=(-3, -2, -1)).real
        r = m * (hx - y.float())
        Rf = torch.fft.fftn(r, dim=(-3, -2, -1))
        grad = torch.fft.ifftn(torch.conj(Hs) * Rf, dim=(-3, -2, -1)).real + lam * x
        x = x - eta * grad
    return x[0] if squeeze else x


def _adaptive_wiener(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
                     lam0: float) -> torch.Tensor:
    """Fill-first Wiener with per-slice adaptive lambda (EXP-023 P3)."""
    filled = _lateral_interp(y, mask)
    lat_psf = PSFConfig(psf.sigma_xy_um, 0.0, psf.voxel_xy_um, psf.voxel_z_um)
    lam_eff = _adaptive_lambda(filled, mask, lam0)      # [T]
    out = filled.clone()
    for t in range(y.shape[0]):
        if lam_eff[t] < lam0:                            # measured content exists
            out[t] = wiener_deconv_3d(filled[t][None], lat_psf, float(max(lam_eff[t], 1e-8)))[0]
    return out


def warm_start(y: torch.Tensor, mask: torch.Tensor, psf: PSFConfig,
               lam: float = 0.01, mode: str = "wiener") -> torch.Tensor:
    """Physics warm-start X0 = W(Y; H, M) with per-mode axial handling.

    Parameters
    ----------
    y    : [T, H, W] zero-filled observation (M . acquisition).
    mask : [T, H, W] measurement support (True = measured).
    psf  : optical model; sigma_z must be 0 for xy mode, > 0 when axial
           information is used.
    mode : "wiener", "bilinear" or "wiener_fill_first" (DEC-015).
    """
    T, H, W = y.shape
    mask_z = mask.amax(dim=(-2, -1)) > 0.5            # slice has any measurement
    axial_gaps = not bool(mask_z.all())
    # Lateral holes on slices that were actually measured.
    lateral_holes = bool((~mask[mask_z]).any()) if axial_gaps else bool((~mask).any())

    if mode == "bilinear":
        out = _axial_interp(y, mask_z) if axial_gaps else y
        if lateral_holes:
            out = _lateral_interp(out, mask)
        return out

    if mode.startswith("adaptive:"):
        # EXP-023 P3: noise-adaptive lambda, lam0 parsed from the mode string
        # ("adaptive:1e-2"). Fill lateral holes FIRST (EXP-022 lesson), then
        # per-slice deconvolution with lam_eff = lam0*(1-rho).
        lam0 = float(mode.split(":")[1])
        return _adaptive_wiener(y, mask, psf, lam0)

    if mode.startswith("iterdc:"):
        # EXP-023 P4: iterative data-consistent inversion. "iterdc:lam:steps"
        lam_i, steps_i = mode.split(":")[1:]
        return iterative_dc_inverse(y, mask, psf, lam=float(lam_i),
                                    steps=int(steps_i))

    if mode == "wiener_fill_first":
        # DEC-015 (EXP-011(三)): for laterally-gapped protocols (xy grid / y),
        # deconvolving zero-filled holes rings (Gibbs).  Fill the
        # measurement-domain holes FIRST, then run the standard wiener
        # pipeline on the filled volume (no lateral holes remain).
        if lateral_holes:
            y = _lateral_interp(y, mask)
            lateral_holes = False
        mode = "wiener"

    # Wiener family.
    if axial_gaps:
        # DEC-003: deconvolve the measured slices laterally FIRST, then fill
        # the axial gaps by interpolating between the deconvolved knots.
        # (Interpolating raw data first would propagate blurred values into
        # the knots and blur the estimate twice.)
        if psf.is_identity:
            out = _axial_interp(y, mask_z)
            if lateral_holes:
                out = _lateral_interp(out, mask)
            return out
        lat_psf = PSFConfig(psf.sigma_xy_um, 0.0, psf.voxel_xy_um, psf.voxel_z_um)
        out = y.clone()
        for t in range(T):
            if mask_z[t]:
                out[t] = wiener_deconv_3d(out[t][None], lat_psf, lam)[0]
        # _axial_interp keeps measured slices and interpolates the gaps from
        # them (also fills outside the measured range with nearest knots).
        out = _axial_interp(out, mask_z)
        if lateral_holes:
            out = _lateral_interp(out, mask)
        return out

    # No axial gaps: plain deconvolution (xy mode => sigma_z=0 => per-slice),
    # then fill lateral holes between deconvolved rows/cols (X0 is a dense
    # estimate: PSF inversion is physics, gap filling is sample reconstruction).
    out = wiener_deconv_3d(y, psf, lam)
    if lateral_holes:
        out = _lateral_interp(out, mask)
    return out

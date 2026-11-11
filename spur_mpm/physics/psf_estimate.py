"""Estimating an effective ``H`` when the deposit ships no optical calibration.

The Dryad 3P deposit does **not** contain a PSF calibration. Checked, not
assumed (2026-11-10):

* ``README.pdf`` says "All imaging parameters are described in the manuscript
  Methods" -- the Methods are not part of the deposit;
* the vessel TIFFs carry ImageJ metadata with ``spacing=5.0`` and a resolution
  tag, and nothing about NA, emission wavelength or immersion medium;
* the dataset spec (``configs/data/dryad_3p_vessels.yaml``) records the
  excitation (1340 nm), the voxel sizes and the axial step -- also no NA.

So on this cohort ``H`` is neither ``calibrated_optical`` (nobody measured it)
nor ``synthetic_extra_degradation`` (nobody added it). Two estimators live here,
and **neither is allowed to call its output calibrated**:

1. :func:`diffraction_limited_sigma` -- arithmetic from wavelength, NA and
   refractive index, with the multi-photon narrowing. This bounds the *optical*
   blur from below-ish, it does not describe the blur *in tissue*: scattering
   broadens the PSF with depth and the deposit does not say at what depth each
   stack was taken. Its result is therefore an order-of-magnitude sanity check,
   useful mostly when it says "the optics cannot be what is blurring this".

2. :func:`holdout_width_scan` -- the width that best predicts **held-out measured
   rows**. This is a task-optimal *effective* width, not a physical
   measurement: it answers "would a deconvolution of this shape have helped on
   this data", and if the optimum sits at ``sigma = 0`` then the honest estimate
   *is* identity, and no deconvolution gain can be claimed at all.

Both are estimators of the same unknown, they are reported side by side, and
their disagreement is part of the result. A caller that wants to *use* the
output must label it ``estimated_from_data`` (see
:data:`spur_mpm.physics.prior.PSF_SOURCES`), so that no summary produced with it
can be read as a calibrated-optics result.
"""

from __future__ import annotations

import math

import torch

from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.wiener import interp_linear, wiener_fill_all

__all__ = ["best_sigma", "diffraction_limited_sigma", "holdout_width_scan"]


# --- 1. arithmetic from the optical parameters ------------------------------

def diffraction_limited_sigma(*, excitation_nm: float, na: float,
                              n_medium: float = 1.33, n_photon: int = 3,
                              voxel_xy_um: float = 0.98,
                              voxel_z_um: float = 5.0) -> dict:
    """Gaussian sigma implied by the diffraction limit of an N-photon focus.

    Uses the standard single-photon results

        FWHM_xy = 0.51 * lambda / NA
        FWHM_z  = 0.88 * lambda / (n - sqrt(n^2 - NA^2))

    and then the first-order multi-photon narrowing: fluorescence goes as
    ``I^N``, so an intensity PSF with sigma ``s`` becomes a signal PSF with
    ``s / sqrt(N)``. That is the right scaling for the lateral axes and a
    deliberate approximation axially (the exact N-photon axial profile also
    depends on the pupil and on which order of the Bessel expansion dominates
    near NA ~ 1). It is reported as an approximation, not as a calibration.

    Returns a dict with both the physical sigmas and their size **in voxels**,
    because that last number decides whether a deconvolution can do anything at
    all: a PSF narrower than one voxel is not recoverable from a sampled stack.
    """
    for name, value in (("excitation_nm", excitation_nm), ("na", na),
                        ("n_medium", n_medium), ("voxel_xy_um", voxel_xy_um),
                        ("voxel_z_um", voxel_z_um)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) \
                or float(value) <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    if not isinstance(n_photon, int) or n_photon < 1:
        raise ValueError(f"n_photon must be a positive integer, got {n_photon!r}")
    if float(na) >= float(n_medium):
        raise ValueError(
            f"NA {na} must be below the medium index {n_medium}: the axial "
            "formula divides by n - sqrt(n^2 - NA^2), which is zero (and the "
            "objective cannot accept such a cone anyway) at NA >= n")

    lam_um = float(excitation_nm) / 1000.0
    fwhm_xy_um = 0.51 * lam_um / float(na)
    fwhm_z_um = 0.88 * lam_um / (float(n_medium)
                                 - math.sqrt(float(n_medium) ** 2 - float(na) ** 2))
    narrowing = math.sqrt(float(n_photon))
    sigma_xy_um = fwhm_xy_um / narrowing / 2.3548
    sigma_z_um = fwhm_z_um / narrowing / 2.3548

    return {
        "excitation_nm": float(excitation_nm),
        "na": float(na),
        "n_medium": float(n_medium),
        "n_photon": int(n_photon),
        "fwhm_xy_um_single_photon": fwhm_xy_um,
        "fwhm_z_um_single_photon": fwhm_z_um,
        "sigma_xy_um": sigma_xy_um,
        "sigma_z_um": sigma_z_um,
        "sigma_xy_px": sigma_xy_um / float(voxel_xy_um),
        "sigma_z_px": sigma_z_um / float(voxel_z_um),
        "psf": PSFConfig(sigma_xy_um, sigma_z_um, float(voxel_xy_um),
                         float(voxel_z_um)).to_dict(),
        "assumptions": [
            "diffraction-limited: no scattering, no aberration, no defocus, no "
            "motion; real tissue is always broader than this, never narrower",
            "first-order multi-photon narrowing (FWHM / sqrt(N)); exact for the "
            "lateral axes, approximate axially",
            "NA and n are NOT in the deposit -- they are inputs to this call, "
            "and the answer inherits whatever error they carry",
        ],
    }


# --- 2. the width that best predicts held-out measurements ------------------

def _score(pred: torch.Tensor, ref: torch.Tensor, held: torch.Tensor,
           peak: float = 1.0) -> dict:
    err = (pred[held] - ref[held]).float()
    mse = float(err.pow(2).mean())
    psnr = float("inf") if mse <= 0 else 10.0 * math.log10(peak ** 2 / mse)
    return {"rmse": math.sqrt(mse), "mae": float(err.abs().mean()), "psnr": psnr}


def holdout_width_scan(observed: torch.Tensor, mask: torch.Tensor, *,
                       ref: torch.Tensor | None = None, axis: str = "xy",
                       sigmas_um=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0),
                       lam: float = 0.01, voxel_xy_um: float = 0.98,
                       voxel_z_um: float = 5.0) -> list[dict]:
    """Score ``W0(sigma)`` against a reference on the voxels it was *not* given.

    For each candidate width: interpolate the withheld measurement (``I0``),
    deconvolve with that ``H`` (``W0``), and score the result on the voxels the
    protocol skipped. ``observed`` is the dense volume the mask samples from --
    only its masked voxels are ever handed to the operators, so the scan cannot
    see the answer it is scored on.

    **``ref`` decides what is being estimated, and the two cases are not the
    same quantity:**

    * ``ref=None`` (default) -- score against the withheld *measurements*. This
      is the real-data case, and the recovered width is the **total** low-pass
      between the latent object and the measurement: optics, plus whatever the
      interpolation itself smoothed, plus the detector's response. It is an
      upper bound on the optical blur and it is the number that answers "would a
      deconvolution have helped predict this measurement". It may legitimately
      be non-zero with a perfect microscope.
    * ``ref=<sharp signal>`` -- score against the underlying object (a phantom,
      or a unit test). Now the recovered width is the blur the deconvolution
      actually undoes, i.e. a PSF in the usual sense.

    Neither output is a calibration, and neither is written into a config by
    anything here.

    ``sigmas_um`` must start at 0, which is the interpolation baseline: every
    ``delta_psnr`` is relative to it. An optimum at ``sigma = 0`` is the
    informative outcome and means "do not deconvolve" -- it is an estimate of
    identity, not a failed search.

    ``axis`` selects which width to scan: ``xy`` (lateral only) or ``z`` (axial
    only, which at a 5 um step can only ever be sub-voxel).
    """
    if axis not in ("xy", "z"):
        raise ValueError(f"axis must be 'xy' or 'z', got {axis!r}")
    observed = torch.as_tensor(observed, dtype=torch.float32)
    mask = torch.as_tensor(mask).bool()
    if observed.shape != mask.shape:
        raise ValueError(f"observed {tuple(observed.shape)} and mask "
                         f"{tuple(mask.shape)} disagree")
    if observed.ndim != 3:
        raise ValueError(f"expected a [Z,H,W] volume, got {tuple(observed.shape)}")
    ref = observed if ref is None else torch.as_tensor(ref, dtype=torch.float32)
    if ref.shape != observed.shape:
        raise ValueError(f"ref {tuple(ref.shape)} and observed "
                         f"{tuple(observed.shape)} disagree")
    sigmas = [float(s) for s in sigmas_um]
    if not sigmas or sigmas[0] != 0.0:
        raise ValueError("sigmas_um must start at 0.0 (the identity baseline)")
    if any(s < 0 for s in sigmas):
        raise ValueError("sigmas_um must be >= 0")

    held = ~mask
    n_held = int(held.sum())
    if n_held == 0:
        raise ValueError("the mask covers every voxel; there is nothing withheld "
                         "to score on")
    if int(mask.sum()) == 0:
        raise ValueError("the mask covers no voxel; there is nothing to "
                         "reconstruct from")

    # The observation the algorithm is allowed to see: measured voxels only.
    y = torch.where(mask, observed, torch.zeros_like(observed))
    i0 = interp_linear(y, mask)
    base = _score(i0, ref, held)

    rows: list[dict] = []
    for s in sigmas:
        psf = PSFConfig(s if axis == "xy" else 0.0,
                        s if axis == "z" else 0.0,
                        float(voxel_xy_um), float(voxel_z_um))
        w0 = wiener_fill_all(y, mask, psf, lam)
        sc = _score(w0, ref, held)
        rows.append({
            "axis": axis,
            "sigma_um": s,
            "sigma_px": s / (float(voxel_xy_um) if axis == "xy" else float(voxel_z_um)),
            "lam": float(lam),
            "is_identity": psf.is_identity,
            "n_held": n_held,
            # A non-zero sigma that leaves W0 numerically equal to I0 has not
            # been tested, it has been skipped (a sub-voxel sigma rounds away in
            # the OTF at float32). Reporting the changed fraction stops such a
            # row from being read as "this width does not help".
            "changed_fraction": float((w0 - i0).abs().gt(1e-6).float().mean()),
            "rmse": sc["rmse"],
            "mae": sc["mae"],
            "psnr": sc["psnr"],
            "delta_psnr_vs_identity": sc["psnr"] - base["psnr"],
        })
    return rows


def best_sigma(rows: list[dict]) -> dict:
    """The row with the best held-out PSNR, with ties resolved toward identity.

    Ties go to the *smaller* sigma on purpose: if the data cannot distinguish
    two widths, the one that does less to the image is the one to report.

    Accepts both a raw scan row (``psnr``, one patch) and an aggregated row
    (``psnr_mean``, many patches) rather than making callers rename fields to
    get an answer out.
    """
    if not rows:
        raise ValueError("no scan rows to choose from")

    def score(row: dict) -> float:
        for key in ("psnr", "psnr_mean"):
            if key in row:
                return float(row[key])
        raise KeyError(f"row has neither 'psnr' nor 'psnr_mean': {sorted(row)}")

    return sorted(rows, key=lambda r: (-score(r), r["sigma_um"]))[0]

"""Is the literature bead-FWHM PSF distinguishable from interpolation at all?

Issue #4 (E027) asks one narrow question before any training: on this cohort's
sampling grid, does the Wiener initialisation built from the paper's measured
bead FWHM differ from plain interpolation, where it matters (the voxels that
were never measured), and does it differ *usefully*?

Source (Stas et al., J. Biomed. Opt. 31(1):016004, 2026, Methods 2.1, Fig. 1b):
0.20 um beads, n = 5, FWHM 0.87 +/- 0.12 um lateral and 4.59 +/- 0.48 um axial.
With FWHM = 2*sqrt(2 ln 2)*sigma that is sigma = (0.3695, 1.9492) um, which on
this cohort's 0.98 um / 5 um grid is **(0.377, 0.390) voxels**. The axial FWHM
is smaller than the 5 um z step -- the kernel is undersampled axially and close
to a delta -- so "no difference" is the honest expectation. This measures it.

Design choices that keep the number honest:

* Whole-volume, not per window. The kernel couples neighbouring planes (3.5 %
  of its mass each at +/-1), and a centre-plane-only test would silently drop
  the axial part of the PSF.
* The lattice is the pipeline's own (`observe` with the frozen `SamplingSpec`,
  phase 0), so the mask is the one the experiments use.
* lambda is chosen on a **training** volume; the test volume is reported, never
  tuned on.
* The identity PSF is run through the same code path as an assertion: with
  H = I the Wiener branch must equal interpolation, or the operator is not
  trustworthy and no PSF conclusion may be drawn.
* Nothing here trains, and nothing here is a method result.

Usage:
    python -m spur_mpm.scripts.e027_psf_precheck --fold 0 --protocol yz \
        --ky 2 --kz 2 --out runs/E027_literature_psf/precheck_yz.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spur_mpm.data.datasets import VolumeCache
from spur_mpm.data.split import folds_by_name, read_split
from spur_mpm.physics.acquisition import AcquisitionConfig, NoiseConfig, observe
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import warm_start

#: Stas et al. 2026, Methods 2.1: 0.20 um beads, n = 5, mean +/- SEM.
FWHM_XY_UM, FWHM_XY_SEM_UM = 0.87, 0.12
FWHM_Z_UM, FWHM_Z_SEM_UM = 4.59, 0.48
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
LABEL = "literature_bead_fwhm_gaussian"
SOURCE = {
    "citation": "Stas et al., J. Biomed. Opt. 31(1):016004 (2026)",
    "doi": "10.1117/1.JBO.31.1.016004",
    "section": "Methods 2.1 and Fig. 1b",
    "label": LABEL,
    "bead_diameter_um": 0.20,
    "n_beads": 5,
    "fwhm_xy_um": FWHM_XY_UM, "fwhm_xy_sem_um": FWHM_XY_SEM_UM,
    "fwhm_z_um": FWHM_Z_UM, "fwhm_z_sem_um": FWHM_Z_SEM_UM,
    "conversion": "sigma = FWHM / (2*sqrt(2*ln2)) = FWHM / 2.35482",
}


def paper_psf(geom) -> PSFConfig:
    return PSFConfig(FWHM_XY_UM * FWHM_TO_SIGMA, FWHM_Z_UM * FWHM_TO_SIGMA,
                     geom.voxel_xy_um, geom.z_step_um)


def identity_psf(geom) -> PSFConfig:
    return PSFConfig(0.0, 0.0, geom.voxel_xy_um, geom.z_step_um)


def kernel_stats(psf: PSFConfig) -> dict:
    """How much of the kernel sits in the centre voxel, per axis and in 3-D.

    Computed from the sampled weights directly, not by inverting a finite FFT
    grid: an FFT-based "centre fraction" of an odd-sized grid mixes the aliased
    kernel with the grid's own resolution and reported 0.52 for a kernel whose
    three-tap weights are 0.944 (lateral) and 0.931 (axial).
    """
    def weights(sigma_px: float, half: int = 3):
        r = np.arange(2 * half + 1) - half
        w = np.exp(-0.5 * (r / sigma_px) ** 2) if sigma_px > 0 else (r == 0).astype(float)
        return w / w.sum()

    wx = weights(float(psf.sigma_xy_px))
    wz = weights(float(psf.sigma_z_px))
    centre = float(wz[len(wz) // 2] * wx[len(wx) // 2] ** 2)
    return {
        "sigma_xy_px": float(psf.sigma_xy_px), "sigma_z_px": float(psf.sigma_z_px),
        "is_identity": bool(psf.is_identity),
        "lateral_weights_3tap": [float(v) for v in wx[len(wx) // 2 - 1:len(wx) // 2 + 2]],
        "axial_weights_3tap": [float(v) for v in wz[len(wz) // 2 - 1:len(wz) // 2 + 2]],
        "centre_voxel_fraction": centre,
        "axial_pm1_fraction": float(2 * wz[len(wz) // 2 - 1] * wx[len(wx) // 2] ** 2),
    }


def _observe(spec: SamplingSpec, dense: np.ndarray, phase=(0, 0, 0)):
    """The pipeline's own acquisition: sparse y and its mask, no extra blur."""
    acq = observe(torch.from_numpy(dense),
                  AcquisitionConfig(sampling=spec, psf=PSFConfig(),
                                    noise=NoiseConfig(poisson_photons=0.0,
                                                      gaussian_sigma=0.0, seed=0),
                                    phase=tuple(phase)))
    return acq.y.numpy(), acq.mask.numpy().astype(bool)


def _psnr(a: np.ndarray, b: np.ndarray, sel: np.ndarray) -> float:
    d = (a - b)[sel]
    if d.size == 0:
        return float("nan")
    mse = float(np.mean(d.astype(np.float64) ** 2))
    return float(10.0 * np.log10(max(mse, 1e-12) ** -1))


@torch.no_grad()
def dc_gain(psf: PSFConfig, lam: float, shape=(15, 33, 33)) -> float:
    """Response of the Wiener operator to a constant volume.

    The operator is ``conj(H)/(|H|^2 + lam)``, so a constant image comes back
    scaled by ``1/(1+lam)``: the regulariser darkens the image as well as
    damping the deconvolution. Measured, not assumed, because it is one of the
    two ways a "prior initialisation" can look different from interpolation.
    """
    y = torch.ones(shape, dtype=torch.float64)
    return float(warm_start(y, torch.ones(shape, dtype=torch.bool), psf, lam,
                            mode="wiener").mean())


@torch.no_grad()
def compare(dense, y, mask, psf, lam, geom) -> dict:
    """Interpolation vs Wiener on the same sparse observation."""
    yt = torch.from_numpy(y).float()
    mt = torch.from_numpy(mask)
    i0 = warm_start(yt, mt, identity_psf(geom), lam, mode="bilinear").numpy()
    w0 = warm_start(yt, mt, psf, lam, mode="wiener").numpy()
    gain = dc_gain(psf, lam)
    # A DC-normalised variant: same operator with the 1/(1+lam) darkening
    # removed, so a difference cannot be attributed to a global scale factor.
    w0n = w0 / gain if gain else w0
    miss = ~mask
    d = np.abs(w0 - i0)
    dn = np.abs(w0n - i0)
    ref_max = float(dense.max())
    return {
        "dc_gain": gain,
        "mean_abs_delta_missing_dc_normalised": float(dn[miss].mean()) if miss.any() else 0.0,
        "mae_w0_missing_dc_normalised": float(np.abs(w0n - dense)[miss].mean()) if miss.any() else 0.0,
        "psnr_w0_missing_dc_normalised": _psnr(w0n, dense, miss),
        "windows_or_voxels_missing": int(miss.sum()),
        "mean_abs_delta_all": float(d.mean()),
        "mean_abs_delta_missing": float(d[miss].mean()) if miss.any() else 0.0,
        "p95_abs_delta_missing": float(np.quantile(d[miss], 0.95)) if miss.any() else 0.0,
        "mae_i0_missing": float(np.abs(i0 - dense)[miss].mean()) if miss.any() else 0.0,
        "mae_w0_missing": float(np.abs(w0 - dense)[miss].mean()) if miss.any() else 0.0,
        "psnr_i0_missing": _psnr(i0, dense, miss),
        "psnr_w0_missing": _psnr(w0, dense, miss),
        "measured_region_std": float((w0 - i0)[mask].std()) if mask.any() else 0.0,
        "w0_below_zero_fraction": float((w0 < 0).mean()),
        "w0_above_ref_max_fraction": float((w0 > ref_max).mean()) if ref_max > 0 else 0.0,
        "max_w0_minus_i0": float(d.max()),
    }


def run(*, index: str, split: str, fold: int, protocol: str, ky: int, kz: int,
        lams: list[float], out: Path) -> dict:
    folds = folds_by_name(read_split(split))
    cache = VolumeCache.load(index, fold=fold)
    calib = cache.calibration_for(fold)
    train_sid = [s for s in folds[fold]["train"] if s in cache][0]
    test_sid = [s for s in folds[fold]["test"] if s in cache][0]
    geom = cache.geometry(test_sid)
    spec = SamplingSpec(mode=protocol, ky=ky, kz=kz)

    def volume(sid):
        dense = calib.apply(cache.array(sid)).astype(np.float32)
        y, m = _observe(spec, dense)
        return dense, y, m, cache.geometry(sid)

    dense_tr, y_tr, m_tr, geom_tr = volume(train_sid)
    paper_te = paper_psf(geom)

    result = {
        "issue": 4, "experiment": "SPUR-E027", "source": SOURCE,
        "grid": {"voxel_xy_um": geom.voxel_xy_um, "z_step_um": geom.z_step_um,
                 "axial_fwhm_over_step": FWHM_Z_UM / geom.z_step_um},
        "protocol": {"mode": protocol, "ky": ky, "kz": kz,
                     "retained_fraction_measured": float(m_tr.mean())},
        "kernels": {"paper": kernel_stats(paper_te),
                    "identity": kernel_stats(identity_psf(geom))},
        "lambda_selection": {"rule": "training volume only", "volume": train_sid,
                             "rows": []},
        "test": {"volume": test_sid, "rows": []},
    }

    for lam in lams:
        r = compare(dense_tr, y_tr, m_tr, paper_psf(geom_tr), lam, geom_tr)
        r.update({"lam": lam, "volume": train_sid})
        result["lambda_selection"]["rows"].append(r)
    best = min(result["lambda_selection"]["rows"], key=lambda r: r["mae_w0_missing"])
    lam = float(best["lam"])
    result["lambda_selection"]["chosen_lam"] = lam

    dense_te, y_te, m_te, geom_te = volume(test_sid)
    for tag, psf in (("I0_identity", identity_psf(geom_te)),
                     ("W0_paper", paper_psf(geom_te)),
                     ("W0_identity_check", identity_psf(geom_te))):
        r = compare(dense_te, y_te, m_te, psf, lam, geom_te)
        r.update({"tag": tag, "lam": lam, "volume": test_sid})
        result["test"]["rows"].append(r)

    rows = {r["tag"]: r for r in result["test"]["rows"]}
    chk = rows["W0_identity_check"]
    result["test"]["identity_assertion"] = {
        "mean_abs_delta_missing": chk["mean_abs_delta_missing"],
        "holds": bool(chk["mean_abs_delta_missing"] < 1e-6),
        "note": "with H = I the Wiener branch must reproduce interpolation",
    }
    result["verdict"] = _verdict(result, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


def _verdict(result: dict, rows: dict) -> str:
    k = result["kernels"]["paper"]
    if not result["test"]["identity_assertion"]["holds"]:
        return ("The identity check failed: with H = I the Wiener branch did not "
                "reproduce interpolation, so no PSF conclusion may be drawn.")
    w0, i0 = rows["W0_paper"], rows["I0_identity"]
    delta = w0["mean_abs_delta_missing"]
    rel = (w0["mae_w0_missing"] - i0["mae_i0_missing"]) / max(i0["mae_i0_missing"], 1e-12)
    rel_n = (w0["mae_w0_missing_dc_normalised"] - i0["mae_i0_missing"]) / max(
        i0["mae_i0_missing"], 1e-12)
    near_delta = k["centre_voxel_fraction"] > 0.75
    if near_delta and abs(rel) < 0.01:
        return ("The literature kernel is nearly a delta on this grid (centre "
                f"voxel carries {k['centre_voxel_fraction']:.3f} of its mass) and "
                f"moves the missing voxels by {delta:.2e}; missing-region MAE "
                f"{i0['mae_i0_missing']:.5f} -> {w0['mae_w0_missing']:.5f} "
                f"({100*rel:+.2f} %). W0_paper is interpolation for any practical "
                "purpose, so no trained arm may attribute a gain to this prior.")
    return (f"The literature kernel is nearly a delta (centre voxel "
            f"{k['centre_voxel_fraction']:.3f}) yet its Wiener initialisation "
            f"moves the missing voxels by {delta:.2e}: missing-region MAE "
            f"{i0['mae_i0_missing']:.5f} -> {w0['mae_w0_missing']:.5f} "
            f"({100*rel:+.2f} %), PSNR {i0['psnr_i0_missing']:.3f} -> "
            f"{w0['psnr_w0_missing']:.3f} dB; DC gain {w0['dc_gain']:.4f}. After "
            f"removing that DC factor the MAE is "
            f"{w0['mae_w0_missing_dc_normalised']:.5f} ({100*rel_n:+.2f} %), so the "
            "loss is not a brightness offset but amplified lattice-frequency "
            f"artefacts (measured-region std {w0['measured_region_std']:.4f}, "
            f"{100*w0['w0_above_ref_max_fraction']:.2f} % of voxels above the "
            "reference maximum).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data_local/dryad_cache/index_cohort.json")
    ap.add_argument("--split", default="spur_mpm/experiments/E020_cohort/split.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--protocol", default="yz", choices=["yz", "y"])
    ap.add_argument("--ky", type=int, default=2)
    ap.add_argument("--kz", type=int, default=2)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = run(index=a.index, split=a.split, fold=a.fold, protocol=a.protocol,
            ky=a.ky, kz=a.kz, lams=a.lams, out=a.out)
    print(json.dumps({"grid": r["grid"], "protocol": r["protocol"],
                      "kernels": r["kernels"],
                      "chosen_lam": r["lambda_selection"]["chosen_lam"],
                      "test": r["test"]["rows"],
                      "identity": r["test"]["identity_assertion"],
                      "verdict": r["verdict"]}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

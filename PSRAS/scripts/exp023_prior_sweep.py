"""EXP-023 (pre-registered): standalone prior-quality sweep for YZ-sparse.

Question A protocol (no network): build yz observations with INDEPENDENT
ky/kz sparsity (make_mask_yz2), apply each candidate prior recipe, and
measure the prior's own PSNR/SSIM (and measured/missing split errors via
the explicit mask) against the dense GT on the deterministic val patch grid.

Recipes
-------
P0 bilinear          : axial interp + lateral interp (legacy b2 path)
P1 wiener frozen     : lateral Wiener on measured slices + axial interp
                       (legacy b3 path, lambda frozen -- expected collapse)
P2 wiener_fill_first : fill holes first, then Wiener (DEC-015 variant)
P3 adaptive:lam0     : fill-first + per-slice lam_eff = lam0*(1-rho)
P4 iterdc:lam:steps  : iterative data-consistent inversion (bilinear init)

Outputs runs_v3/exp023_prior_sweep/summary.csv (+ console table).

Usage:
    python scripts/exp023_prior_sweep.py [--device cuda] [--n_items 24]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.datasets import CIDC25SparseDataset  # noqa: E402
from metrics.image_metrics import psnr, ssim, split_errors  # noqa: E402
from physics.acquisition import blur_3d_fft, NoiseConfig  # noqa: E402
from physics.psf import PSFConfig  # noqa: E402
from physics.sampling import make_mask_yz2  # noqa: E402
from physics.wiener import warm_start  # noqa: E402

DATA = Path(__file__).parents[1] / "runs_v3/data/cidc25_v3"
OUT_DIR = Path(__file__).parents[1] / "runs_v3/exp023_prior_sweep"

# (ky, kz) matrix: joint + asymmetric + fixed-budget B=16 triple {(4,4),(8,2),(2,8)}
K_COMBOS = [(2, 2), (2, 4), (4, 2), (4, 4), (8, 2), (2, 8)]

RECIPES = [
    ("P0_bilinear", "bilinear"),
    ("P1_wiener_frozen", "wiener"),
    ("P2_fill_first", "wiener_fill_first"),
    ("P3_adaptive_1e-3", "adaptive:1e-3"),
    ("P3_adaptive_1e-2", "adaptive:1e-2"),
    ("P3_adaptive_1e-1", "adaptive:1e-1"),
    ("P4_iterdc_1e-3_5", "iterdc:1e-3:5"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_items", type=int, default=24,
                   help="deterministic val patch grid items per (ky,kz)")
    p.add_argument("--psf_xy", type=float, default=1.0)
    p.add_argument("--psf_z", type=float, default=1.5)
    args = p.parse_args()

    psf = PSFConfig(sigma_xy_um=args.psf_xy, sigma_z_um=args.psf_z,
                    voxel_xy_um=1.0, voxel_z_um=1.0)
    ds = CIDC25SparseDataset(DATA, split="val", regime="2.5d", mode="z", K=2,
                              psf=psf, patch_hw=(256, 256),
                              features=("y", "mask", "x0"))
    # Use the dataset only for its deterministic patch grid + GT: we draw
    # items directly from its fixed_items machinery via _draw_patch seeds.
    ds.fixed_items = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    rows = []

    # Deterministic patch grid: reuse the dataset's rng derivation with
    # distinct index offsets so patches differ per combo (seeded, stable).
    for ky, kz in K_COMBOS:
        T = 2 * kz - 1 if kz > 1 else 1
        if T > 31:
            T = 31
        for i in range(args.n_items):
            # draw a GT patch from the val split via the dataset's machinery
            index = (ky * 100 + kz) * 1000 + i
            gt, _, phases = ds._draw_patch(index % ds.patches_per_epoch)
            gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32))
            th = gt_t.shape[0]
            # our T window may differ from the drawn patch's T: redraw with
            # the right T by temporarily patching ds.T
            ds_T_backup = ds.T
            ds.T = T
            gt, _, phases = ds._draw_patch(index % ds.patches_per_epoch)
            gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32))
            ds.T = ds_T_backup
            th = gt_t.shape[0]
            t0g, y0g, x0g = phases

            mask = make_mask_yz2(ky, kz, th, *gt_t.shape[1:],
                                 seed=i, phase_t=t0g % kz if kz > 0 else 0,
                                 phase_y=y0g % ky if ky > 0 else 0)
            clean = blur_3d_fft(gt_t.to(dev), psf)
            y = torch.where(mask.to(dev), clean, torch.zeros_like(clean))
            # noiseless mainline (DEC-008): no additional noise term

            for rname, mode in RECIPES:
                x0 = warm_start(y, mask.to(dev), psf, lam=0.01, mode=mode)
                x0 = x0.clamp(0.0, 1.0)
                tgt = gt_t.to(dev)
                q = torch.ones_like(tgt, dtype=torch.bool)  # full-window prior quality
                pv = psnr(x0, tgt).item()
                # per-slice SSIM (2D window, DEC-025): mean over T slices
                if x0.ndim == 3:
                    sv = float(ssim(x0[:, None], tgt[:, None]))
                else:
                    sv = float("nan")
                em, es = split_errors(x0, tgt, mask.to(dev))
                rows.append({"ky": ky, "kz": kz, "item": i, "recipe": rname,
                             "psnr": round(pv, 4), "ssim": round(sv, 4),
                             "err_measured": round(em, 6),
                             "err_missing": round(es, 6)})
        done = [r for r in rows if r["ky"] == ky and r["kz"] == kz]
        print(f"--- (ky={ky}, kz={kz}) mean PSNR over {args.n_items} items ---")
        for rname, _ in RECIPES:
            vals = [r["psnr"] for r in done if r["recipe"] == rname]
            print(f"  {rname:22s} {sum(vals)/len(vals):7.2f} dB")

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} rows -> {OUT_DIR/'summary.csv'}")


if __name__ == "__main__":
    main()

"""EXP-011: Wiener lambda sweep on the z-mode validation protocol.

No training -- computes X0 = W(Y; H, M) for each lambda and measures PSNR
against GT under the 2.5D z-K4 validation protocol. Answers whether the
fixed lam=0.01 used in every Wiener-X0 run (B3/B5/B6) is near-optimal.
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, ".")

from physics.psf import PSFConfig
from physics.wiener import warm_start
from data.datasets import make_val_dataset, window_T
from metrics.image_metrics import psnr, ssim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="2.5d")
    ap.add_argument("--mode", default="z")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="runs_v3/data/cidc25_v3")
    ap.add_argument("--n-items", type=int, default=8)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--psf_xy", type=float, default=1.0)
    ap.add_argument("--psf_z", type=float, default=1.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = window_T(args.regime, args.mode, args.K)
    psf = PSFConfig(sigma_xy_um=args.psf_xy, sigma_z_um=args.psf_z,
                    voxel_xy_um=1.0, voxel_z_um=1.0)
    ds = make_val_dataset(args.data_dir, "val", args.regime, psf=psf,
                          mode=args.mode, K=args.K, seed=args.seed)
    n = min(args.n_items, len(ds))

    rows = []
    for lam in args.lams:
        ps, ss = [], []
        for i in range(n):
            it = ds[i]
            gt = it["target"].float().to(device)                  # [T,H,W]
            x = it["inputs"].float().to(device)                   # [F*T,H,W]
            # DEC-013: feature-major channels -> y is feature 0, all T slices.
            # X0 is computed on the full window (no query slicing: the query
            # one-hot only selects the centre slice for model evaluation).
            y = x.view(-1, t, *gt.shape[-2:])[0]                  # [T,H,W]
            mask = y != 0
            x0 = warm_start(y, mask, psf, lam=lam, mode="wiener")
            ps.append(psnr(x0[None], gt[None]).item())
            ss.append(ssim(x0[None][:, None], gt[None][:, None]).item())
        rows.append({"lam": lam, "psnr": sum(ps) / n, "ssim": sum(ss) / n})
        print(f"lam={lam:<8g} X0 PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.4f}")

    out = "runs_v3/lambda_sweep_z.json"
    json.dump({"protocol": {"regime": args.regime, "mode": args.mode,
                            "K": args.K, "n_items": n, "t": t},
               "rows": rows}, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

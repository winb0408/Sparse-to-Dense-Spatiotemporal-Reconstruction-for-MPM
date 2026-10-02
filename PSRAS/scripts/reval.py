"""Re-evaluate a trained run under its own acquisition protocol.

Motivation: train.py used to build the val dataset without mode/K, so runs
trained outside xy-grid-K4 were validated under the wrong protocol. This
script loads a run's config.json + checkpoint and recomputes val metrics
under the protocol the run was actually trained on.

Usage:
    python scripts/reval.py runs_v3/b0_2d_y_K8 [--ckpt best|last]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.datasets import make_val_dataset            # noqa: E402
from metrics.image_metrics import psnr, split_errors, ssim  # noqa: E402
from physics.psf import PSFConfig                     # noqa: E402
from scripts.train import PRESETS, build_model        # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--ckpt", choices=["best", "last"], default="last")
    p.add_argument("--split", default="val",
                   help="split key in split.json: val | test | test_temporal ...")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="rvrt's CUDA-only deformable attention needs gpu")
    p.add_argument("--data_dir", type=Path,
                   default=Path(__file__).parents[1] / "runs_v3/data/cidc25_v3")
    # EXP-025: evaluate a (mixed-regime) run at a specific acquisition point
    # that may differ from the training config (unified/specialist runs are
    # scored on the full Y + YZ suite).  Overrides the protocol below.
    p.add_argument("--eval_mode", choices=["y", "yz2", "xyz2"], default=None)
    p.add_argument("--eval_K", type=int, default=0)
    p.add_argument("--eval_kz", type=int, default=0)
    args = p.parse_args()

    cfg = json.loads((args.run_dir / "config.json").read_text())
    psf = PSFConfig(sigma_xy_um=cfg["psf"]["sigma_xy_px"],
                    sigma_z_um=cfg["psf"]["sigma_z_px"],
                    voxel_xy_um=1.0, voxel_z_um=1.0)
    t, feats = cfg["t"], tuple(cfg["features"])
    preset = PRESETS[cfg["preset"]]

    # yz2 runs carry kz in config.json; pass through as K2 so the mask matches training
    k2 = cfg.get("kz", 0) or 0
    eval_tag = ""
    if args.eval_mode is not None:
        # EXP-025 point override: rebuild the protocol for one acquisition
        # point (y: regime 2d, T=1 / yz2 & xyz2: regime 2.5d, T=2*kz-1).
        from scripts.train import point_params
        from physics.psf import PSFConfig as _P
        ev_regime, ev_K, ev_kz, ev_t = point_params(
            args.eval_mode, args.eval_K if args.eval_mode == "y"
            else (args.eval_K, args.eval_kz), "2d", "2.5d")
        psf = _P(sigma_xy_um=cfg["psf"]["sigma_xy_px"],
                 sigma_z_um=cfg["psf"]["sigma_z_px"] if ev_regime != "2d" else 0.0,
                 voxel_xy_um=1.0, voxel_z_um=1.0)
        regime, mode, K, k2, t = ev_regime, args.eval_mode, ev_K, ev_kz, ev_t
        t_override = None
        eval_tag = f"_{mode}_K{K}x{k2}" if mode in ("yz2", "xyz2") else f"_{mode}_K{K}"
    else:
        regime, mode, K, t_override = cfg["regime"], cfg["mode"], cfg["K"], cfg["t"]
    val_ds = make_val_dataset(args.data_dir, args.split, regime,
                              mode=mode, K=K, K2=k2, psf=psf,
                              patch_hw=(cfg["patch"], cfg["patch"]),
                              warm_start_kind=cfg["warm_start"],
                              lam=cfg["lam_wiener"], features=feats,
                              t_override=t_override)

    ckpt = torch.load(args.run_dir / f"{args.ckpt}.pt", map_location="cpu",
                      weights_only=False)
    model = build_model(preset, t, cfg["model"]["base"], cfg["model"]["window"],
                        cfg["model"].get("kind", "swin2d"),
                        axial_at=cfg["model"].get("axial_at", "all"),
                        output=cfg["model"].get("output", "all"),
                        use_az=cfg["model"].get("use_az", True))
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()

    psnrs, ssims = [], []
    e_meas, e_miss = [], []
    with torch.no_grad():
        for it in val_ds.fixed_items:
            x, target = it["inputs"], it["target"]
            x = x.to(args.device)
            pred = model(x[None]).float().view(t, *x.shape[-2:])
            if preset["residual"]:
                # DEC-013: inputs are feature-major [F*t, H, W]; X0 is the
                # 3rd feature (y, mask, x0, ...) -- same convention as train.py.
                x0 = x.view(len(feats), t, *x.shape[-2:])[2]
                pred = x0 + pred
            q = it["query"].bool()
            pred_sel, tgt_sel = pred[q], target.view(t, *x.shape[-2:]).to(pred.device)[q]
            psnrs.append(psnr(pred_sel, tgt_sel).item())
            ssims.append(ssim(pred_sel[:, None], tgt_sel[:, None]).item())
            # DEC-025 (protocol v4): split-region errors must use the real
            # measurement mask feature, never Y != 0 (a measured position can
            # legitimately hold a zero value).  Runs without a mask feature
            # (e.g. b0 = y only) record NaN split errors.
            fmaps = x.view(len(feats), t, *x.shape[-2:])
            if "mask" in feats:
                em, es = split_errors(pred_sel, tgt_sel, fmaps[feats.index("mask")][q])
            else:
                em, es = float("nan"), float("nan")
            e_meas.append(em)
            e_miss.append(es)

    out = {"psnr": sum(psnrs) / len(psnrs), "ssim": sum(ssims) / len(ssims),
           "err_measured": sum(e_meas) / len(e_meas),
           "err_missing": sum(e_miss) / len(e_miss),
           "n_items": len(psnrs), "ckpt": args.ckpt, "split": args.split,
           "metrics_version": "v4: 2D gaussian SSIM window; split errors from M",
           "protocol": {"mode": mode, "K": K, "K2": k2}}
    out_name = "val_metrics_corrected.json" if args.split == "val" else \
        f"metrics_{args.split}.json"
    if eval_tag:
        out_name = out_name.replace(".json", f"{eval_tag}.json")
    (args.run_dir / out_name).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Protocol v3 training entry point (PLAN.md Phase 2, ablation matrix B0-B6).

Ablation presets (PLAN.md §4 Phase 2):
    B0  raw sparse Y only                  (no mask prior)
    B1  + sampling mask M
    B2  + bilinear warm-start X0
    B3  + true 3D Wiener X0
    B4  + sampling-aware prior fusion (C_s, U_p, gated)
    B5  + residual learning  X_hat = X0 + dX
    B6  + data consistency loss (PSF line only)

Every run writes runs_v3/<name>/: config.json (full protocol), log.csv,
best.pt / last.pt, and val_metrics.json.  Nothing here touches legacy code.

Usage:
    python scripts/train.py --preset b3 --regime 2d --K 4 --steps 3000
    python scripts/train.py --preset b0 --regime 2.5d --K 2 --steps 3000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import CIDC25SparseDataset, make_val_dataset, window_T  # noqa: E402
from physics.acquisition import NoiseConfig                                # noqa: E402
from data.normalization import IntensityCalibration  # noqa: E402  (loaded via split.json)
from losses.data_consistency import DataConsistencyLoss  # noqa: E402
from losses.reconstruction import ReconstructionLoss  # noqa: E402
from metrics.image_metrics import psnr, ssim  # noqa: E402
from models.prior_fusion import PriorFusionStem  # noqa: E402
from models.swin_unet_2d import SwinUNet2D  # noqa: E402
from physics.psf import PSFConfig  # noqa: E402

# --------------------------------------------------------------------------
# Ablation presets: which inputs enter the model, warm-start kind, residual,
# and whether the DC loss is active.  Features per slice: y, mask, x0, c_s, u_p.
# --------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "b0": {"features": ("y",), "warm": "bilinear", "fusion": False, "residual": False, "dc": 0.0},
    "b1": {"features": ("y", "mask"), "warm": "bilinear", "fusion": False, "residual": False, "dc": 0.0},
    "b2": {"features": ("y", "mask", "x0"), "warm": "bilinear", "fusion": False, "residual": False, "dc": 0.0},
    "b3": {"features": ("y", "mask", "x0"), "warm": "wiener", "fusion": False, "residual": False, "dc": 0.0},
    "b4": {"features": ("y", "mask", "x0", "confidence", "uncertainty"),
           "warm": "wiener", "fusion": True, "residual": False, "dc": 0.0},
    "b5": {"features": ("y", "mask", "x0", "confidence", "uncertainty"),
           "warm": "wiener", "fusion": True, "residual": True, "dc": 0.0},
    "b6": {"features": ("y", "mask", "x0", "confidence", "uncertainty"),
           "warm": "wiener", "fusion": True, "residual": True, "dc": 0.05},
}


def build_model(preset: dict, t: int, base: int, window: int,
                model_kind: str = "swin2d",
                axial_at: str = "all", output: str = "all",
                window_t: int = 1) -> torch.nn.Module:
    feats_per_slice = len(preset["features"])
    if model_kind in ("vswin", "sunetr", "rvrt"):
        # EXP-020 external original-repo baselines (DEC-023).
        from models.ext import build_ext_model
        return build_ext_model(model_kind, feats_per_slice, t)
    in_ch = t * feats_per_slice
    out_ch = t
    if model_kind in ("unified3d", "video3d"):
        # unified3d: factorised (wt=1 per-frame windows + zero-init axial
        # branch).  video3d (EXP-018): full 3D shifted-window attention
        # (Video-Swin style), 3D mixing active from step 0.
        from models.unified_3d import Unified3D
        return Unified3D(feats_per_slice, t, base=base, window=window,
                         fusion=preset["fusion"], axial_at=axial_at,
                         output=output,
                         window_t=3 if model_kind == "video3d" else window_t)
    unet_in = base if preset["fusion"] else in_ch
    unet = SwinUNet2D(unet_in, out_ch if not preset["residual"] else out_ch,
                      base=base, window=window)
    if preset["fusion"]:
        stem = PriorFusionStem(in_ch, t, base)
        return torch.nn.Sequential(stem, unet)
    return unet


def make_psf(psf_xy: float, psf_z: float, regime: str) -> PSFConfig:
    """sigma in PIXELS (provisional until physical voxel sizes are known)."""
    return PSFConfig(sigma_xy_um=psf_xy, sigma_z_um=0.0 if regime == "2d" else psf_z,
                     voxel_xy_um=1.0, voxel_z_um=1.0)


@torch.no_grad()
def validate(model, loader, device, t: int, *, residual: bool = False,
             n_features: int = 0, x0_idx: int = 2) -> dict:
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            target = batch["target"].to(device)
            pred = model(x).float().view(-1, t, *x.shape[-2:])
            if residual:
                # DEC-013: inputs are feature-major [B, F*t, H, W].
                x0 = x.view(-1, n_features, t, *x.shape[-2:])[:, x0_idx]
                pred = x0 + pred
            q = batch["query"].to(device).bool()
            pred_sel = pred[q]      # [N, H, W], N = selected slices
            tgt_sel = target[q]
            psnrs.append(psnr(pred_sel, tgt_sel).item())
            ssims.append(ssim(pred_sel[:, None], tgt_sel[:, None]).item())
    model.train()
    return {"psnr": sum(psnrs) / len(psnrs), "ssim": sum(ssims) / len(ssims)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--regime", choices=["2d", "2.5d"], default="2d")
    p.add_argument("--mode", choices=["xy", "y", "z", "yz", "xyz"], default=None,
                   help="default: xy for 2d, z for 2.5d. y/yz = line skipping "
                        "(physically time-saving); xy grid = line lattice")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--t", type=int, default=None,
                   help="override window depth (odd; default window_T(regime,"
                        " mode, K)); EXP-017 context scaling")
    p.add_argument("--poisson_photons", type=float, default=0.0,
                   help="peak photon budget for shot noise on measured px "
                        "(0 = off, DEC-008 default; Phase 4 photon budget)")
    p.add_argument("--data_dir", type=Path, default=Path(__file__).parents[1] / "runs_v3/data/cidc25_v3")
    p.add_argument("--out_root", type=Path, default=Path(__file__).parents[1] / "runs_v3")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--psf_xy", type=float, default=1.0, help="sigma in pixels")
    p.add_argument("--psf_z", type=float, default=1.5, help="sigma in pixels")
    p.add_argument("--lam_wiener", type=float, default=0.01)
    p.add_argument("--model", choices=["swin2d", "unified3d", "video3d",
                                       "vswin", "sunetr", "rvrt"],
                   default="swin2d",
                   help="network family: swin2d (main line) / unified3d "
                        "(factorised axial, EXP-012) / video3d (full 3D "
                        "shifted-window, EXP-018) / vswin+sunetr+rvrt "
                        "(external original-repo baselines, EXP-020)")
    p.add_argument("--warm", choices=["wiener", "bilinear", "wiener_fill_first"],
                   default=None,
                   help="override the preset's warm-start kind (DEC-015)")
    p.add_argument("--axial_at", choices=["all", "bottleneck"], default="all",
                   help="unified3d only: axial attention placement (DEC-017)")
    p.add_argument("--output", choices=["all", "center"], default="all",
                   help="unified3d only: emit all T frames or query center only "
                        "(DEC-018)")
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation micro-batches "
                        "(effective batch = batch * accum)")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="linear LR warmup steps before cosine decay")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    mode = args.mode or ("xy" if args.regime == "2d" else "z")
    preset = PRESETS[args.preset]
    t = args.t or window_T(args.regime, mode, args.K)
    assert t % 2 == 1, f"window depth must be odd, got {t}"
    psf = make_psf(args.psf_xy, args.psf_z, args.regime)
    name = args.name or f"{args.preset}_{args.regime}_{mode}_K{args.K}"
    out_dir = args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "preset": args.preset, "regime": args.regime, "mode": mode, "K": args.K,
        "psf": {"sigma_xy_px": args.psf_xy, "sigma_z_px": args.psf_z if args.regime != "2d" else 0.0},
        "noise": {"poisson_photons": args.poisson_photons, "gaussian_sigma": 0.0,
                  "note": "CIDC25 stacks carry their own simulated noise; extra "
                          "noise is off by default (see PROJECT_LOG DEC-008)"},
        "lam_wiener": args.lam_wiener,
        "warm_start": args.warm or preset["warm"],
        "fusion": preset["fusion"], "residual": preset["residual"], "dc_weight": preset["dc"],
        "features": list(preset["features"]), "t": t,
        "model": {"base": args.base, "window": args.window, "kind": args.model,
                  "axial_at": args.axial_at, "output": args.output,
                  "window_t": 3 if args.model == "video3d" else 1},
        "optim": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                  "accum": args.accum, "warmup_steps": args.warmup_steps},
        "patch": args.patch, "seed": args.seed,
        "data_dir": str(args.data_dir), "protocol": "v3",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_ds = CIDC25SparseDataset(
        args.data_dir, "train", args.regime, mode=mode, K=args.K, psf=psf,
        patch_hw=(args.patch, args.patch), seed=args.seed,
        warm_start_kind=args.warm or preset["warm"], lam=args.lam_wiener,
        features=preset["features"], t_override=args.t,
        noise=NoiseConfig(args.poisson_photons, 0.0))
    val_ds = make_val_dataset(
        args.data_dir, "val", args.regime, mode=mode, K=args.K, psf=psf,
        patch_hw=(args.patch, args.patch),
        warm_start_kind=args.warm or preset["warm"], lam=args.lam_wiener,
        features=preset["features"], t_override=args.t,
        noise=NoiseConfig(args.poisson_photons, 0.0))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    # eval-mode validation runs without gradient checkpointing, so its
    # memory per sample is much higher than training -- cap the val batch
    # by the training micro-batch (EXP-018: video3d OOM'd at val with 4).
    val_loader = DataLoader(val_ds, batch_size=min(4, args.batch), num_workers=2)

    model = build_model(preset, t, args.base, args.window, args.model,
                        axial_at=args.axial_at, output=args.output,
                        window_t=1).to(device)
    recon_loss = ReconstructionLoss()
    dc_loss = (DataConsistencyLoss(psf) if preset["dc"] > 0 and not psf.is_identity else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # DEC-017: linear warmup then cosine decay (warmup stabilises the 3D
    # variants' slow start); gradient accumulation matches swin2d's effective
    # batch of 8 when the 3D model only fits micro-batches of 4.
    def _lr_lambda(step):
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "l1", "ms_ssim", "dc", "lr", "elapsed_s"])

    best_psnr, t0 = -math.inf, time.time()
    step = 0
    model.train()
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            x = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            q = batch["query"].to(device).bool()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(x).float().view(-1, t, *x.shape[-2:])
                if preset["residual"]:
                    # DEC-013: inputs are feature-major [B, F*t, H, W];
                    # X0 is the 3rd feature (y, mask, x0, ...)
                    x0 = x.view(-1, len(preset["features"]), t, *x.shape[-2:])[:, 2]
                    pred = x0 + pred
                pred_sel = pred[q]      # [N, H, W], N = selected slices
                tgt_sel = target[q]
                out = recon_loss(pred_sel, tgt_sel)
                loss = out["loss"]
                if dc_loss is not None:
                    y_meas = x.view(-1, t, len(preset["features"]), *x.shape[-2:])[:, :, 0]
                    mask = x.view(-1, t, len(preset["features"]), *x.shape[-2:])[:, :, 1]
                    dc = dc_loss(pred, y_meas, mask)
                    loss = loss + preset["dc"] * dc
            if step % args.accum == 0:
                opt.zero_grad(set_to_none=True)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

            if step % args.log_every == 0:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [step, f"{loss.item():.5f}", f"{out['l1'].item():.5f}",
                         f"{out['ms_ssim'].item():.5f}",
                         f"{dc.item():.5f}" if dc_loss is not None else "",
                         f"{sched.get_last_lr()[0]:.2e}", f"{time.time() - t0:.0f}"])
            if (step + 1) % args.val_every == 0 or step == args.steps - 1:
                vm = validate(model, val_loader, device, t,
                              residual=preset["residual"],
                              n_features=len(preset["features"]))
                vm["step"] = step + 1
                with open(out_dir / "val_metrics.jsonl", "a") as f:
                    f.write(json.dumps(vm) + "\n")
                if vm["psnr"] > best_psnr:
                    best_psnr = vm["psnr"]
                    torch.save({"model": model.state_dict(), "config": config,
                                "val": vm}, out_dir / "best.pt")
                print(f"[{name}] step {step + 1}: loss={loss.item():.4f} "
                      f"val PSNR={vm['psnr']:.2f} SSIM={vm['ssim']:.4f}", flush=True)
            step += 1

    torch.save({"model": model.state_dict(), "config": config},
               out_dir / "last.pt")
    final = validate(model, val_loader, device, t,
                     residual=preset["residual"],
                     n_features=len(preset["features"]))
    (out_dir / "final_val.json").write_text(json.dumps(final, indent=2))
    print(f"[{name}] done. best val PSNR={best_psnr:.2f}, final={final}")


if __name__ == "__main__":
    main()

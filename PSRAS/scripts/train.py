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
import os
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
    # EXP-024 (Phase 2 YZ mainline): physics prior variants selected by
    # EXP-023 -- b4i = iterative DC inversion x0, b4f = fill-first Wiener x0.
    "b4i": {"features": ("y", "mask", "x0", "confidence", "uncertainty"),
            "warm": "iterdc:1e-3:5", "fusion": True, "residual": False, "dc": 0.0},
    "b4f": {"features": ("y", "mask", "x0", "confidence", "uncertainty"),
            "warm": "wiener_fill_first", "fusion": True, "residual": False, "dc": 0.0},
}


def build_model(preset: dict, t: int, base: int, window: int,
                model_kind: str = "swin2d",
                axial_at: str = "all", output: str = "all",
                window_t: int = 1, use_az: bool = True) -> torch.nn.Module:
    feats_per_slice = len(preset["features"])
    if model_kind == "univ4":
        # EXP-025 (Phase 4 Unified Framework, v4-draft §7): T-agnostic
        # per-slice shared backbone + zero-init axial adapter.  T is only a
        # shape hint -- the module accepts any T at forward time.
        from models.unified_v4 import UnifiedV4
        return UnifiedV4(feats_per_slice, base=base, window=window,
                         use_az=use_az)
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


# --------------------------------------------------------------------------
# EXP-025 (Phase 4 Unified Framework): mixed-regime training.  A "point" is
# (mode, severity): ('y', K) for the Y regime (2d), ('yz2', (ky, kz)) for YZ,
# ('xyz2', (klat, kz)) for XYZ (EXP-026, Phase 5).
# --------------------------------------------------------------------------
def parse_regime_mix(spec: str) -> list[tuple[str, object]]:
    points: list[tuple[str, object]] = []
    for part in spec.split(";"):
        mode, sev = part.strip().split(":")
        if mode == "y":
            for k in sev.split(","):
                points.append(("y", int(k)))
        elif mode in ("yz2", "xyz2"):
            for c in sev.split(","):
                ka, kb = c.split("x")
                points.append((mode, (int(ka), int(kb))))
        else:
            raise ValueError(f"regime_mix: unsupported mode {mode!r}")
    if not points:
        raise ValueError("regime_mix: empty spec")
    return points


def point_params(mode: str, sev, regime_2d: str, regime_25d: str):
    """(regime, K, kz, t) for a mix point."""
    if mode == "y":
        return regime_2d, sev, 0, window_T(regime_2d, "y", sev)
    # yz2: (ky, kz); xyz2: (klat, kz).  Both are 2.5d with plane spacing kz.
    kz = sev[1]
    t = 1 if kz == 1 else 2 * kz - 1
    return regime_25d, sev[0], kz, t


def make_dataset_for_point(data_dir, split, mode, sev, psf_xy, psf_z, *,
                           warm_start_kind, lam, features, patch, seed,
                           noise, patch_hw=None):
    from data.datasets import CIDC25SparseDataset
    regime, K, kz, t = point_params(mode, sev, "2d", "2.5d")
    psf = make_psf(psf_xy, psf_z, regime)
    kw = dict(mode=mode, K=K, psf=psf, patch_hw=patch_hw or (patch, patch),
              warm_start_kind=warm_start_kind, lam=lam, features=features,
              noise=noise)
    if split == "train":
        return CIDC25SparseDataset(data_dir, "train", regime, seed=seed,
                                   t_override=None, K2=kz, **kw), t
    from data.datasets import make_val_dataset
    ds = make_val_dataset(data_dir, "val", regime, K2=kz, **kw)
    return ds, t


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
    p.add_argument("--mode", choices=["xy", "y", "z", "yz", "xyz", "yz2"], default=None,
                   help="default: xy for 2d, z for 2.5d. y/yz = line skipping "
                        "(physically time-saving); xy grid = line lattice")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--kz", type=int, default=0,
                   help="yz2 mode: independent plane spacing kz (K stays ky)")
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
                                       "univ4", "vswin", "sunetr", "rvrt"],
                   default="swin2d",
                   help="network family: swin2d (main line) / unified3d "
                        "(factorised axial, EXP-012) / video3d (full 3D "
                        "shifted-window, EXP-018) / univ4 (Phase 4 unified: "
                        "per-slice shared backbone + zero-init A_Z adapter, "
                        "EXP-025) / vswin+sunetr+rvrt "
                        "(external original-repo baselines, EXP-020)")
    p.add_argument("--regime_mix", type=str, default=None,
                   help="EXP-025 mixed-regime training: semicolon-separated "
                        "regimes, each 'mode:spec' with comma-separated "
                        "severities.  y uses K values (regime 2d); yz2 uses "
                        "kyxkz pairs.  Example: "
                        "'y:2,4,8;yz2:2x2,2x4,2x8,4x2,4x4,8x2'.  Batches "
                        "alternate uniformly across regimes/severities.")
    p.add_argument("--regime_weights", type=str, default=None,
                   help="EXP-028: per-regime sampling weights, comma-separated "
                        "'mode:w' (e.g. 'y:1,yz2:1,xyz2:2' -> batch shares "
                        "25/25/50%%).  None = point-uniform (EXP-025/026 "
                        "behaviour).  Within a regime points stay uniform.")
    p.add_argument("--resume", action="store_true",
                   help="DEC-027 C: resume from resume.pt in the run dir "
                        "(model/opt/sched/scaler/step restored; dataloader "
                        "batch sequence restarts -- stated limitation)")
    p.add_argument("--ckpt_every", type=int, default=500,
                   help="steps between resume.pt writes (atomic replace)")
    p.add_argument("--no_az", action="store_true",
                   help="univ4 only: disable the A_Z axial adapter "
                        "(U-noAZ / SYZ-noAZ ablation arms, EXP-025)")
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
    mix_points = parse_regime_mix(args.regime_mix) if args.regime_mix else None
    if mix_points is not None:
        assert args.model == "univ4", "regime_mix requires --model univ4"
        mode = "mixed"
        t = max(point_params(m, s, "2d", "2.5d")[3] for m, s in mix_points)
    elif mode == "yz2":
        # EXP-024: T follows plane spacing kz; K means ky.
        kz = args.kz or args.K
        t = args.t or (1 if kz == 1 else 2 * kz - 1)
    else:
        t = args.t or window_T(args.regime, mode, args.K)
    assert t % 2 == 1, f"window depth must be odd, got {t}"
    psf = make_psf(args.psf_xy, args.psf_z, args.regime)
    name = args.name or f"{args.preset}_{args.regime}_{mode}_K{args.K}"
    if mix_points is not None and args.name is None:
        name = f"{args.preset}_univ4_mix"
    out_dir = args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "preset": args.preset, "regime": args.regime, "mode": mode, "K": args.K,
        "kz": args.kz,
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
                  "window_t": 3 if args.model == "video3d" else 1,
                  "use_az": (args.model != "univ4") or (not args.no_az)},
        **({"regime_mix": args.regime_mix} if args.regime_mix else {}),
        **({"regime_weights": args.regime_weights} if args.regime_weights else {}),
        "optim": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                  "accum": args.accum, "warmup_steps": args.warmup_steps},
        "patch": args.patch, "seed": args.seed,
        "data_dir": str(args.data_dir), "protocol": "v3",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if mix_points is not None:
        # EXP-025 mixed-regime training: one train loader + one val loader
        # per (regime, severity) point; batches alternate uniformly.
        noise = NoiseConfig(args.poisson_photons, 0.0)
        train_loaders, val_specs = [], []
        for m, sev in mix_points:
            ds, t_pt = make_dataset_for_point(
                args.data_dir, "train", m, sev, args.psf_xy, args.psf_z,
                warm_start_kind=args.warm or preset["warm"],
                lam=args.lam_wiener, features=preset["features"],
                patch=args.patch, seed=args.seed, noise=noise)
            train_loaders.append(DataLoader(ds, batch_size=args.batch,
                                            shuffle=True, num_workers=4,
                                            pin_memory=True, drop_last=True))
            vds, _ = make_dataset_for_point(
                args.data_dir, "val", m, sev, args.psf_xy, args.psf_z,
                warm_start_kind=args.warm or preset["warm"],
                lam=args.lam_wiener, features=preset["features"],
                patch=args.patch, seed=args.seed, noise=noise)
            tag = f"{m}_{sev[0]}x{sev[1]}" if m in ("yz2", "xyz2") else f"{m}_K{sev}"
            val_specs.append((tag, DataLoader(vds, batch_size=min(4, args.batch),
                                              num_workers=2), t_pt))
        val_loader, val_t = val_specs[0][1], val_specs[0][2]  # placeholders

        mix_rng = torch.Generator().manual_seed(args.seed)

        # EXP-028: per-regime sampling weights.  A point in regime r draws
        # P(point) = w_r / sum(w) / n_points_r -- regime-level shares are
        # w-normalised, points stay uniform within a regime.
        if args.regime_weights:
            wmap = {}
            for part in args.regime_weights.split(","):
                m, w = part.strip().split(":")
                wmap[m] = float(w)
            missing = {m for m, _ in mix_points} - set(wmap)
            assert not missing, f"regime_weights missing regimes: {missing}"
            counts: dict[str, int] = {}
            for m, _ in mix_points:
                counts[m] = counts.get(m, 0) + 1
            wsum = sum(wmap[m] for m in counts)
            point_probs = torch.tensor(
                [wmap[m] / wsum / counts[m] for m, _ in mix_points],
                dtype=torch.float64)
        else:
            point_probs = None

        def _next_mixed_batch():
            if point_probs is not None:
                i = int(torch.multinomial(point_probs, 1, generator=mix_rng))
            else:
                i = int(torch.randint(len(train_loaders), (1,), generator=mix_rng))
            loader = train_loaders[i]
            if not hasattr(loader, "_mix_iter"):
                loader._mix_iter = iter(loader)
            try:
                return next(loader._mix_iter)
            except StopIteration:
                loader._mix_iter = iter(loader)
                return next(loader._mix_iter)

        def validate_mixed(model) -> dict:
            per, ps = {}, []
            for tag, loader, t_pt in val_specs:
                vm = validate(model, loader, device, t_pt,
                              residual=preset["residual"],
                              n_features=len(preset["features"]))
                per[tag] = vm
                ps.append(vm["psnr"])
            return {"per_regime": per, "psnr": sum(ps) / len(ps),
                    "ssim": sum(v["ssim"] for v in per.values()) / len(per)}
    else:
        train_ds = CIDC25SparseDataset(
            args.data_dir, "train", args.regime, mode=mode, K=args.K, psf=psf,
            patch_hw=(args.patch, args.patch), seed=args.seed,
            warm_start_kind=args.warm or preset["warm"], lam=args.lam_wiener,
            features=preset["features"], t_override=args.t, K2=args.kz,
            noise=NoiseConfig(args.poisson_photons, 0.0))
        val_ds = make_val_dataset(
            args.data_dir, "val", args.regime, mode=mode, K=args.K, psf=psf,
            patch_hw=(args.patch, args.patch),
            warm_start_kind=args.warm or preset["warm"], lam=args.lam_wiener,
            features=preset["features"], t_override=args.t, K2=args.kz,
            noise=NoiseConfig(args.poisson_photons, 0.0))
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        # eval-mode validation runs without gradient checkpointing, so its
        # memory per sample is much higher than training -- cap the val batch
        # by the training micro-batch (EXP-018: video3d OOM'd at val with 4).
        val_loader = DataLoader(val_ds, batch_size=min(4, args.batch), num_workers=2)
        _next_mixed_batch, validate_mixed, val_t, mix_rng = None, None, t, None

    model = build_model(preset, t, args.base, args.window, args.model,
                        axial_at=args.axial_at, output=args.output,
                        window_t=1,
                        use_az=(args.model != "univ4") or (not args.no_az)
                        ).to(device)
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
    # DEC-027 C (checkpoint-resume): --resume restores model/opt/sched/
    # scaler/step/best_psnr/mix_rng from the last periodic snapshot.
    # Stated limitation: dataloader worker RNG is not snapshotted, so the
    # batch sequence after a resume differs from an uninterrupted run.
    resume_path = out_dir / "resume.pt"
    start_step, best_psnr = 0, -math.inf
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_step, best_psnr = ck["step"], ck["best_psnr"]
        if ck.get("mix_rng") is not None and mix_rng is not None:
            mix_rng.set_state(ck["mix_rng"])
        print(f"[{name}] resumed from step {start_step} "
              f"(best_psnr={best_psnr:.2f})", flush=True)
    if start_step == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "l1", "ms_ssim", "dc",
                                    "lr", "elapsed_s"])
    else:
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "l1", "ms_ssim", "dc",
                                    "lr", "elapsed_s", "resumed_from",
                                    start_step])

    t0 = time.time()
    step = start_step
    model.train()
    n_feats = len(preset["features"])

    def _batch_source():
        if _next_mixed_batch is not None:
            while True:
                yield _next_mixed_batch()
        else:
            while True:
                for batch in train_loader:
                    yield batch

    for batch in _batch_source():
        if step >= args.steps:
            break
        x = batch["inputs"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        q = batch["query"].to(device).bool()
        t_b = x.shape[1] // n_feats      # mixed training: T varies per batch
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(x).float().view(-1, t_b, *x.shape[-2:])
            if preset["residual"]:
                # DEC-013: inputs are feature-major [B, F*t, H, W];
                # X0 is the 3rd feature (y, mask, x0, ...)
                x0 = x.view(-1, n_feats, t_b, *x.shape[-2:])[:, 2]
                pred = x0 + pred
            pred_sel = pred[q]      # [N, H, W], N = selected slices
            tgt_sel = target[q]
            out = recon_loss(pred_sel, tgt_sel)
            loss = out["loss"]
            if dc_loss is not None:
                xg = x.view(-1, n_feats, t_b, *x.shape[-2:])
                y_meas = xg[:, 0].transpose(1, 2)
                mask = xg[:, 1].transpose(1, 2)
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
            if validate_mixed is not None:
                vm = validate_mixed(model)
            else:
                vm = validate(model, val_loader, device, t,
                              residual=preset["residual"],
                              n_features=n_feats)
            vm["step"] = step + 1
            with open(out_dir / "val_metrics.jsonl", "a") as f:
                f.write(json.dumps(vm) + "\n")
            if vm["psnr"] > best_psnr:
                best_psnr = vm["psnr"]
                torch.save({"model": model.state_dict(), "config": config,
                            "val": vm}, out_dir / "best.pt")
            print(f"[{name}] step {step + 1}: loss={loss.item():.4f} "
                  f"val PSNR={vm['psnr']:.2f} SSIM={vm['ssim']:.4f}", flush=True)
        if (step + 1) % args.ckpt_every == 0:
            # DEC-027 C: atomic periodic snapshot (tmp + os.replace) so a
            # crash mid-write never corrupts the resume point.  Placed after
            # the val block so best_psnr is current when snapshotted.
            snap = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step + 1, "best_psnr": best_psnr,
                    "mix_rng": mix_rng.get_state() if mix_rng is not None else None}
            torch.save(snap, out_dir / "resume_tmp.pt")
            os.replace(out_dir / "resume_tmp.pt", resume_path)
        step += 1

    torch.save({"model": model.state_dict(), "config": config},
               out_dir / "last.pt")
    final = (validate_mixed(model) if validate_mixed is not None
             else validate(model, val_loader, device, t,
                           residual=preset["residual"], n_features=n_feats))
    (out_dir / "final_val.json").write_text(json.dumps(final, indent=2))
    print(f"[{name}] done. best val PSNR={best_psnr:.2f}, final={final}")


if __name__ == "__main__":
    main()

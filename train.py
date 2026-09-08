"""
train.py  (v2 — Noisy Nuclei edition)
=======================================
Training loop for the 3PM reconstruction pipeline.
Updated to work with NucleiDenoisingDataset and the new options.py.

Key changes from v1
--------------------
- Uses ``build_dataloaders`` from the updated dataset.py which expects
  ``opts.data_root`` (paired layout) instead of opts.train_dir / opts.val_dir.
- Validation DataLoader returns (inp, target) for mode='test' full images,
  or (inp, target, meta) for NucleiGridDataset patches; both are handled.
- PSNR metric is computed on [0,1]-rescaled tensors (not raw [-1,1]) so
  values are directly comparable to published fluorescence denoising benchmarks.
- Added SSIM metric logging alongside PSNR.

Usage
-----
# Baseline denoising — Noisy Nuclei:
python train.py --task baseline --model swin_unet_2d \
    --data_root /data/Denoising_Dataset \
    --norm_mode minmax --patches_per_image 50 \
    --batch_size 8 --epochs 100 --run_name baseline_minmax

# Spatial 2D sparse reconstruction:
python train.py --task spatial_2d --model swin_unet_2d \
    --data_root /data/Denoising_Dataset \
    --sparse_source low --sparsity_factor 4 \
    --run_name spatial2d_grid_k4

# Ablation — MSE only:
python train.py --task baseline --model swin_unet_2d \
    --data_root /data/Denoising_Dataset \
    --no_dino --no_msssim --run_name ablation_mse_only

# Resume from checkpoint:
python train.py --task baseline --model swin_unet_2d \
    --data_root /data/Denoising_Dataset \
    --resume --checkpoint runs/baseline_minmax/best.pth
"""

import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from options import get_opts
from dataset import build_dataloaders
from models  import build_model
from loss    import ReconstructionLoss

def _torch_load_compat(path: Path):
    """
    torch.load compatibility across old/new PyTorch versions.
    - Newer versions may default to weights_only=True
    - Older versions may not accept weights_only kwarg
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═════════════════════════════════════════════════════════════════════════════
# LR schedule: linear warm-up → cosine annealing
# ═════════════════════════════════════════════════════════════════════════════

def build_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs:  int,
) -> LambdaLR:
    """Linear warm-up for `warmup_epochs`, then cosine decay to eta_min = lr/100."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR in dB computed on [0,1]-rescaled tensors.

    Inputs are expected in model range [-1, 1].
    Rescaling:  x_01 = (x + 1) / 2.
    Data range is 1.0 (max - min in [0,1]).
    """
    p = (pred.detach()   + 1.0) / 2.0
    t = (target.detach() + 1.0) / 2.0
    mse = float(torch.mean((p - t) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


# ═════════════════════════════════════════════════════════════════════════════
# One-epoch loop  (shared between train and val)
# ═════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    criterion:    ReconstructionLoss,
    optimizer:    Optional[torch.optim.Optimizer],
    device:       torch.device,
    scaler:       Optional[torch.cuda.amp.GradScaler],
    opts,
    is_train:     bool = True,
    writer:       Optional[SummaryWriter] = None,
    global_step:  int  = 0,
) -> Dict[str, float]:
    """
    Run one full epoch.  Returns a dict of mean metrics over all batches.

    Handles both 2-item  (inp, target)
    and       3-item  (inp, target, meta)  DataLoader outputs so that
    NucleiGridDataset and NucleiDenoisingDataset are both supported.
    """
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    totals: Dict[str, float] = {}
    n = len(loader)
    t0 = time.time()

    with ctx:
        for batch_idx, batch in enumerate(loader):
            # Unpack (2 or 3 items)
            inp, target = batch[0], batch[1]
            inp    = inp.to(device,    non_blocking=True)
            target = target.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # ── Forward ──────────────────────────────────────────────────────
            if is_train and scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(inp)
                    loss, metrics = criterion(pred, target)
            else:
                pred = model(inp)
                loss, metrics = criterion(pred, target)

            psnr = compute_psnr(pred, target)
            metrics["metric/psnr"] = psnr

            # ── Backward ─────────────────────────────────────────────────────
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if opts.clip_grad > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if opts.clip_grad > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
                    optimizer.step()

            # ── Accumulate ───────────────────────────────────────────────────
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v

            # ── TensorBoard per-step (train only, every 20 batches) ───────────
            if is_train and writer is not None and batch_idx % 20 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"step/{k}", v, global_step + batch_idx)

            # ── Console progress ──────────────────────────────────────────────
            log_freq = max(1, n // 5)
            if (batch_idx + 1) % log_freq == 0:
                mode_str = "TRAIN" if is_train else "VAL  "
                psf_str = ""
                if "loss/psf" in metrics and metrics["loss/psf"] > 0:
                    psf_str = f"  psf={metrics['loss/psf']:.4f}"
                print(
                    f"  [{mode_str}] {batch_idx+1:4d}/{n}  "
                    f"loss={metrics['loss/total']:.4f}  "
                    f"mse={metrics['loss/mse']:.4f}  "
                    f"ssim={metrics['loss/ms_ssim']:.4f}  "
                    f"dino={metrics['loss/dino']:.4f}{psf_str}  "
                    f"psnr={psnr:.2f} dB  "
                    f"[{time.time()-t0:.0f}s]"
                )

    return {k: v / max(1, n) for k, v in totals.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path:        Path,
    model:       nn.Module,
    optimizer:   torch.optim.Optimizer,
    scheduler:   LambdaLR,
    epoch:       int,
    val_metrics: Dict[str, float],
    opts,
) -> None:
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "val_metrics": val_metrics,
        "opts":        vars(opts),
    }, str(path))
    print(f"  ✓ Checkpoint → {path}")


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR]              = None,
) -> Tuple[int, Dict[str, float]]:
    ckpt = _torch_load_compat(path)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler is not None and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt.get("epoch", 0), ckpt.get("val_metrics", {})


# ═════════════════════════════════════════════════════════════════════════════
# TensorBoard image logging
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def log_images(
    writer:  SummaryWriter,
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    epoch:   int,
    n_vis:   int = 4,
) -> None:
    """
    Log a side-by-side [input | prediction | GT] grid to TensorBoard.
    Rescales from model range [-1,1] → [0,1] for display.
    For 3-D inputs [B,C,T,H,W] the middle frame is used.
    """
    model.eval()
    batch = next(iter(loader))
    inp    = batch[0][:n_vis].to(device)
    target = batch[1][:n_vis].to(device)
    pred   = model(inp)

    def _vis(t: torch.Tensor) -> torch.Tensor:
        """→ [N, 1, H, W] in [0,1]."""
        if t.ndim == 5:                                 # [B, C, T, H, W]
            t = t[:, :, t.shape[2] // 2]               # middle frame
        return ((t[:, :1].clamp(-1.0, 1.0) + 1.0) / 2.0)

    try:
        import torchvision
        grid = torchvision.utils.make_grid(
            torch.cat([_vis(inp), _vis(pred), _vis(target)], dim=0),
            nrow=n_vis,
        )
        try:
            writer.add_image("epoch/input_pred_gt", grid, epoch)
        except Exception as exc:
            # Pillow/tensorboard version mismatches should not crash training.
            print(f"  [WARN] Skip TensorBoard image logging at epoch {epoch}: {exc}")
    except ImportError:
        pass   # torchvision not available — skip image logging


# ═════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    opts   = get_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  3PM Reconstruction  |  task={opts.task}  model={opts.model}")
    print(f"  Device  : {device}")
    print(f"  Run dir : {opts.log_path}")
    print(f"{'='*65}\n")

    set_seed(opts.seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(opts)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(opts).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,}\n")

    # ── Loss, Optimiser, Scheduler ────────────────────────────────────────────
    criterion = ReconstructionLoss(opts).to(device)
    optimizer = AdamW(model.parameters(), lr=opts.lr,
                      weight_decay=opts.weight_decay)
    scheduler = build_scheduler(optimizer, opts.warmup_epochs, opts.epochs)

    scaler = (
        torch.cuda.amp.GradScaler()
        if opts.use_amp and device.type == "cuda" else None
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    if opts.resume and opts.checkpoint:
        start_epoch, prev_metrics = load_checkpoint(
            Path(opts.checkpoint), model, optimizer, scheduler
        )
        print(f"  Resumed from epoch {start_epoch}  "
              f"(val_loss={prev_metrics.get('loss/total', '?'):.4f})\n")
    elif getattr(opts, "init_checkpoint", None):
        init_path = Path(opts.init_checkpoint)
        ckpt = _torch_load_compat(init_path)
        model.load_state_dict(ckpt["model_state"], strict=True)
        print(f"  Loaded init weights from {init_path} (fine-tuning mode)\n")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer       = SummaryWriter(log_dir=str(opts.log_path))
    best_val     = float("inf")
    best_epoch   = 0
    epochs_without_improvement = 0
    global_step  = start_epoch * len(train_loader)
    has_validation = val_loader is not None

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, opts.epochs + 1):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch:4d}/{opts.epochs}    LR = {lr_now:.3e}")
        print(f"{'─'*65}")

        # Train
        train_m = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            opts, is_train=True, writer=writer, global_step=global_step,
        )
        global_step += len(train_loader)
        scheduler.step()

        # Validate (only when validation set is available)
        val_m: Optional[Dict[str, float]] = None
        if has_validation:
            val_m = run_epoch(
                model, val_loader, criterion, None, device, None,
                opts, is_train=False,
            )
            for k, v in val_m.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # ── Epoch-level TensorBoard ──────────────────────────────────────────
        for k, v in train_m.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        if has_validation and val_m is not None:
            print(
                f"\n  SUMMARY  train→ loss={train_m['loss/total']:.4f}  "
                f"psnr={train_m['metric/psnr']:.2f} dB\n"
                f"           val  → loss={val_m['loss/total']:.4f}  "
                f"psnr={val_m['metric/psnr']:.2f} dB"
            )
            if epoch % 5 == 0:
                log_images(writer, model, val_loader, device, epoch)
        else:
            print(
                f"\n  SUMMARY  train→ loss={train_m['loss/total']:.4f}  "
                f"psnr={train_m['metric/psnr']:.2f} dB  (no validation)"
            )

        # ── Checkpointing ────────────────────────────────────────────────────
        if has_validation and val_m is not None:
            metrics_for_ckpt = val_m
            monitored_loss = val_m["loss/total"]
            monitored_name = "val/loss/total"
        else:
            metrics_for_ckpt = train_m
            monitored_loss = train_m["loss/total"]
            monitored_name = "train/loss/total"

        improved = monitored_loss < (best_val - opts.early_stop_min_delta)
        if improved:
            best_val = monitored_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                opts.log_path / "best.pth",
                model, optimizer, scheduler, epoch, metrics_for_ckpt, opts,
            )
        else:
            epochs_without_improvement += 1
            if opts.early_stop_patience > 0:
                print(
                    f"  Early-stop monitor {monitored_name}: "
                    f"no improvement for {epochs_without_improvement}/"
                    f"{opts.early_stop_patience} epochs "
                    f"(best={best_val:.6f} at epoch {best_epoch})"
                )

        if epoch % opts.save_every == 0:
            save_checkpoint(
                opts.log_path / f"epoch_{epoch:04d}.pth",
                model, optimizer, scheduler, epoch, metrics_for_ckpt or train_m, opts,
            )

        if opts.early_stop_patience > 0 and epochs_without_improvement >= opts.early_stop_patience:
            print(
                f"  Early stopping at epoch {epoch}: {monitored_name} "
                f"did not improve by at least {opts.early_stop_min_delta:g} "
                f"for {opts.early_stop_patience} consecutive epochs."
            )
            break

    writer.close()
    # Save last epoch as last.pth when no validation (so user can eval on test)
    if not has_validation:
        save_checkpoint(
            opts.log_path / "last.pth",
            model, optimizer, scheduler, opts.epochs, train_m, opts,
        )
        print(f"\n  Training complete.  No validation; use eval.py on test set.")
        print(f"  Saved last → {opts.log_path / 'last.pth'}, best (by train loss) → {opts.log_path / 'best.pth'}")
    else:
        print(f"\n  Training complete.  Best val loss = {best_val:.4f} (epoch {best_epoch})")
        print(f"  Best model → {opts.log_path / 'best.pth'}")


if __name__ == "__main__":
    main()

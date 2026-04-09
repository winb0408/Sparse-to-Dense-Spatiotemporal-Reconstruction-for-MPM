"""
eval.py  (v2 — publication-grade)
==================================
Full evaluation pipeline for the 3PM sparse-to-dense reconstruction pipeline.

Design goals
------------
  1. Metric completeness   — PSNR, SSIM, MS-SSIM, LPIPS, NCC, MAE, CNR,
                             Spectral SNR (sSNR), Frequency-weighted PSNR (FwPSNR),
                             plus Δ-variants relative to the noisy input.
  2. Statistical rigour    — Per-image values + bootstrap 95 % CI on means.
  3. Ablation support      — Compare N checkpoints; Wilcoxon signed-rank test
                             between every run pair; Bonferroni-corrected p-values.
  4. I/O robustness        — Hann-window blending for sliding-window reassembly;
                             reflect-padded patches; graceful LPIPS/skimage fallback.
  5. Reproducibility       — Seeded RNG; float32 .npy + uint16 .tif outputs.

Metrics
-------
  PSNR     (dB)   Peak SNR                                    ↑
  SSIM            Structural Similarity (11×11 Gaussian win)  ↑
  MS-SSIM         Multi-Scale SSIM, 5 scales                  ↑
  LPIPS           AlexNet perceptual distance                 ↓
  NCC             Normalised Cross-Correlation                 ↑
  MAE             Mean Absolute Error                          ↓
  CNR             Contrast-to-Noise Ratio                      ↑
  sSNR    (dB)    Spectral SNR (Weigert et al. NatMeth 2018)  ↑
  FwPSNR  (dB)    Frequency-weighted PSNR                     ↑
  delta_psnr / delta_ssim — improvement over noisy input

Outputs  (written to  <output_dir>/eval/)
-----------------------------------------
  metrics_{run}.csv        per-image × metric table
  summary_{run}.csv        mean / std / median / ci95_lo / ci95_hi
  ablation_all.csv         all runs concatenated  (consumed by visualize.py)
  ablation_summary.csv     run × metric mean ± std pivot
  ablation_pvalues.csv     Wilcoxon p-value matrix (Bonferroni-corrected)
  images/{run}/{stem}_pred.npy   float32 [H,W] ∈ [0,1]
  images/{run}/{stem}_gt.npy
  images/{run}/{stem}_noisy.npy
  images/{run}/{stem}_pred.tif   uint16

Usage
-----
# Single checkpoint:
python eval.py --task baseline --model swin_unet_2d \\
    --data_root /data/Denoising_Dataset \\
    --checkpoint runs/full/best.pth \\
    --output_dir eval_results --patch_size 256

# Ablation suite (multiple checkpoints):
python eval.py --task baseline --model swin_unet_2d \\
    --data_root /data/Denoising_Dataset \\
    --eval_checkpoints runs/full/best.pth runs/no_dino/best.pth \\
    --eval_run_names   full              no_dino \\
    --output_dir eval_results/ablation
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage, stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    import tifffile as _tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False

try:
    from skimage.metrics import structural_similarity as _ski_ssim
    from skimage.metrics import peak_signal_noise_ratio as _ski_psnr
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    warnings.warn("scikit-image not found — SSIM uses torch fallback.", stacklevel=2)

try:
    import lpips as _lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False
    warnings.warn(
        "lpips not installed — LPIPS metric disabled.\n"
        "Install with:  pip install lpips",
        stacklevel=2,
    )

# ── Local project imports ──────────────────────────────────────────────────────
from options import get_opts
from dataset import (NucleiGridDataset, NucleiDenoisingDataset, NucleiTemporal2p5DDataset,
                     load_tiff_float32, normalise_pair)
from models  import build_model

def _torch_load_compat(path: Path):
    """
    Backward-compatible torch.load wrapper.
    Older torch versions do not support weights_only argument.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")

def _pad_to_multiple_2d(x: torch.Tensor, multiple: int = 32) -> Tuple[torch.Tensor, int, int]:
    """Reflect-pad [B,C,H,W] so H and W are multiples of `multiple`."""
    h, w = x.shape[-2], x.shape[-1]
    h_pad = int(math.ceil(h / multiple) * multiple)
    w_pad = int(math.ceil(w / multiple) * multiple)
    pad_h = h_pad - h
    pad_w = w_pad - w
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    # F.pad order: (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, h, w


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Individual metric functions
#     Input convention: float32 numpy arrays [H, W]  ∈ [0, 1]
# ══════════════════════════════════════════════════════════════════════════════

def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR in dB, data_range = 1.0."""
    if _HAS_SKIMAGE:
        return float(_ski_psnr(gt, pred, data_range=1.0))
    mse = float(np.mean((pred - gt) ** 2))
    return 100.0 if mse < 1e-12 else float(10.0 * math.log10(1.0 / mse))


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM, 11×11 Gaussian window, data_range = 1.0."""
    if _HAS_SKIMAGE:
        return float(_ski_ssim(gt, pred, data_range=1.0,
                               win_size=11, gaussian_weights=True, sigma=1.5))
    # Pure-torch fallback
    p = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    g = torch.from_numpy(gt  ).unsqueeze(0).unsqueeze(0)
    C1, C2, k = 0.01**2, 0.03**2, 11
    mu_p = F.avg_pool2d(p, k, 1, k // 2)
    mu_g = F.avg_pool2d(g, k, 1, k // 2)
    s_pp = F.avg_pool2d(p * p, k, 1, k // 2) - mu_p ** 2
    s_gg = F.avg_pool2d(g * g, k, 1, k // 2) - mu_g ** 2
    s_pg = F.avg_pool2d(p * g, k, 1, k // 2) - mu_p * mu_g
    num  = (2 * mu_p * mu_g + C1) * (2 * s_pg  + C2)
    den  = (mu_p**2 + mu_g**2 + C1) * (s_pp + s_gg + C2)
    return float((num / den).mean())


def _ms_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    MS-SSIM (Wang et al. 2003), 5 scales, weighted product.
    Returns ∈ [0, 1].
    """
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    p, g = pred.copy(), gt.copy()
    val = 1.0
    for i, w in enumerate(weights):
        val *= max(_ssim(p, g), 0.0) ** w
        if i < 4:
            p = ndimage.uniform_filter(p, 2)[::2, ::2]
            g = ndimage.uniform_filter(g, 2)[::2, ::2]
    return float(val)


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _ncc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Normalised Cross-Correlation ∈ [-1, 1]."""
    pf = pred.flatten() - pred.mean()
    gf = gt.flatten()   - gt.mean()
    return float(np.dot(pf, gf) / (np.linalg.norm(pf) * np.linalg.norm(gf) + 1e-10))


def _cnr(pred: np.ndarray, thresh: float = 0.5) -> float:
    """
    Contrast-to-Noise Ratio of the prediction.
    signal  = pixels > thresh  (foreground nuclei)
    background = pixels ≤ thresh
    Returns NaN if either region has < 10 pixels.
    """
    sig = pred[pred > thresh]
    bg  = pred[pred <= thresh]
    if len(sig) < 10 or len(bg) < 10:
        return float("nan")
    return float(abs(sig.mean() - bg.mean()) / (bg.std() + 1e-10))


def _spectral_snr(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Spectral SNR (sSNR) in dB  (Weigert et al., Nature Methods 2018).

    PSD_noise = |PSD_gt − PSD_pred|
    sSNR = 10 log10(sum(PSD_pred) / sum(PSD_noise))

    Positive values → prediction captures the GT spectral content.
    """
    psd_gt   = np.abs(np.fft.fft2(gt  )) ** 2
    psd_pred = np.abs(np.fft.fft2(pred)) ** 2
    noise    = np.abs(psd_gt - psd_pred)
    return float(10.0 * math.log10(psd_pred.sum() / (noise.sum() + 1e-30)))


def _fw_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Frequency-weighted PSNR (FwPSNR).

    Per-pixel squared error is weighted by the local spectral power in the GT:
        weight_map = |IFFT(|FFT(gt)|)|        (spatial spectral-power map)
        FwPSNR = 10 log10(1 / weighted_MSE)

    High-information frequency bands contribute more to the score.
    """
    err      = (pred - gt) ** 2
    wmap     = np.abs(np.fft.ifft2(np.abs(np.fft.fft2(gt))))
    wnorm    = wmap / (wmap.sum() + 1e-30)
    wmse     = float((wnorm * err).sum())
    return 100.0 if wmse < 1e-12 else float(10.0 * math.log10(1.0 / wmse))


def _lpips_score(
    pred:   np.ndarray,
    gt:     np.ndarray,
    net,                         # lpips.LPIPS instance
    device: torch.device,
) -> float:
    """LPIPS (AlexNet).  Lower = perceptually more similar."""
    def _t(x: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1)
                .unsqueeze(0).to(device) * 2.0 - 1.0)
    with torch.no_grad():
        return float(net(_t(pred), _t(gt)).item())


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Compound metric computation
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = [
    "psnr", "ssim", "ms_ssim", "mae", "ncc", "cnr",
    "s_snr", "fw_psnr", "lpips",
    "psnr_noisy", "ssim_noisy", "delta_psnr", "delta_ssim",
]


def compute_all_metrics(
    pred:       np.ndarray,
    gt:         np.ndarray,
    noisy:      Optional[np.ndarray],
    lpips_net:  Optional[object],
    device:     torch.device,
    cnr_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    Full metric suite for one (pred, gt) pair.
    All inputs are float32 [H, W] in [0, 1].
    """
    m: Dict[str, float] = {}
    m["psnr"]    = _psnr   (pred, gt)
    m["ssim"]    = _ssim   (pred, gt)
    m["ms_ssim"] = _ms_ssim(pred, gt)
    m["mae"]     = _mae    (pred, gt)
    m["ncc"]     = _ncc    (pred, gt)
    m["cnr"]     = _cnr    (pred, cnr_thresh)
    m["s_snr"]   = _spectral_snr(pred, gt)
    m["fw_psnr"] = _fw_psnr(pred, gt)
    m["lpips"]   = (
        _lpips_score(pred, gt, lpips_net, device)
        if (lpips_net is not None and _HAS_LPIPS)
        else float("nan")
    )
    if noisy is not None:
        m["psnr_noisy"] = _psnr(noisy, gt)
        m["ssim_noisy"] = _ssim(noisy, gt)
        m["delta_psnr"] = m["psnr"] - m["psnr_noisy"]
        m["delta_ssim"] = m["ssim"] - m["ssim_noisy"]
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Statistical utilities
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values:      np.ndarray,
    n_bootstrap: int   = 2000,
    ci:          float = 0.95,
    seed:        int   = 0,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap (ci*100) % confidence interval for the mean.

    Returns (ci_lo, ci_hi).  NaN-safe.
    """
    rng  = np.random.default_rng(seed)
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def wilcoxon_pvalue_matrix(
    dfs:       List[pd.DataFrame],
    run_names: List[str],
    metrics:   List[str],
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test between every pair of runs for every metric.
    Bonferroni correction for the number of pairwise comparisons per metric.

    Returns a long-form DataFrame with columns:
        metric | comparison (A vs B) | p_value | significant (α = 0.05 after correction)
    """
    pairs = [
        (i, j, f"{run_names[i]} vs {run_names[j]}")
        for i in range(len(run_names))
        for j in range(i + 1, len(run_names))
    ]
    n_pairs = len(pairs)
    rows = []
    for metric in metrics:
        for i, j, label in pairs:
            a = dfs[i][metric].dropna().values if metric in dfs[i] else np.array([])
            b = dfs[j][metric].dropna().values if metric in dfs[j] else np.array([])
            n = min(len(a), len(b))
            if n < 5:
                pval = float("nan")
            else:
                try:
                    _, raw_p = stats.wilcoxon(a[:n], b[:n], alternative="two-sided")
                    pval = min(float(raw_p) * n_pairs, 1.0)   # Bonferroni
                except Exception:
                    pval = float("nan")
            rows.append({
                "metric":       metric,
                "comparison":   label,
                "p_value":      pval,
                "significant":  pval < 0.05 if not math.isnan(pval) else False,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Hann-window sliding-window assembler
# ══════════════════════════════════════════════════════════════════════════════

def _hann2d(size: int, device: torch.device) -> torch.Tensor:
    """2-D Hann blending window [1, 1, size, size]."""
    h = torch.hann_window(size, periodic=False, device=device)
    return (h.unsqueeze(1) * h.unsqueeze(0)).view(1, 1, size, size)


@torch.no_grad()
def assemble_full_image(
    model:       torch.nn.Module,
    grid_loader: DataLoader,
    patch_size:  int,
    H_pad:       int,
    W_pad:       int,
    device:      torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window inference with Hann-window blending.

    model output convention: [-1, 1]  →  rescaled to [0, 1] before return.

    Returns
    -------
    pred_full : [H_pad, W_pad]  float32 ∈ [0, 1]
    gt_full   : [H_pad, W_pad]  float32 ∈ [0, 1]
    (Caller is responsible for cropping to original unpadded H × W.)
    """
    model.eval()
    ps  = patch_size
    win = _hann2d(ps, device)

    acc_pred = torch.zeros(1, 1, H_pad, W_pad, device=device)
    acc_gt   = torch.zeros_like(acc_pred)
    w_pred   = torch.zeros_like(acc_pred)
    w_gt     = torch.zeros_like(acc_pred)

    for batch in grid_loader:
        inp, target, meta = batch[0], batch[1], batch[2]
        inp    = inp.to(device)
        target = target.to(device)
        pred   = model(inp)                              # [B, 1, ps, ps]

        B = pred.shape[0]
        for b in range(B):
            y0 = int(meta["y0"][b]) if torch.is_tensor(meta["y0"]) else int(meta["y0"])
            x0 = int(meta["x0"][b]) if torch.is_tensor(meta["x0"]) else int(meta["x0"])
            vy = min(ps, H_pad - y0)
            vx = min(ps, W_pad - x0)
            if vy <= 0 or vx <= 0:
                continue
            w = win[:, :, :vy, :vx]
            acc_pred[:, :, y0:y0+vy, x0:x0+vx] += pred  [b:b+1, :, :vy, :vx] * w
            acc_gt  [:, :, y0:y0+vy, x0:x0+vx] += target[b:b+1, :, :vy, :vx] * w
            w_pred  [:, :, y0:y0+vy, x0:x0+vx] += w
            w_gt    [:, :, y0:y0+vy, x0:x0+vx] += w

    def _to_np(acc, w):
        t = (acc / w.clamp(min=1e-6)).squeeze().cpu().numpy()
        return ((t + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)

    return _to_np(acc_pred, w_pred), _to_np(acc_gt, w_gt)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Single-checkpoint evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(
    opts,
    checkpoint_path: Path,
    output_dir:      Path,
    run_name:        str   = "run",
    save_images:     bool  = True,
    cnr_thresh:      float = 0.5,
    n_bootstrap:     int   = 2000,
) -> pd.DataFrame:
    """
    Evaluate one checkpoint over the full Noisy Nuclei test set.

    Returns a pd.DataFrame with one row per test image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = output_dir / "images" / run_name
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(opts).to(device)
    # #region agent log
    try:
        with open("/data2/wjb/threePM/.cursor/debug-e242fc.log", "a") as _f:
            import json
            _f.write(json.dumps({"id":"log_load","timestamp":__import__("time").time()*1000,"location":"eval.py:429","message":"torch.load checkpoint","data":{"checkpoint_path":str(checkpoint_path)},"hypothesisId":"H1"}) + "\n")
    except Exception:
        pass
    # #endregion
    ckpt  = _torch_load_compat(checkpoint_path)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")

    print(f"\n{'━'*62}")
    print(f"  Run: {run_name:<20s}  epoch={epoch}  device={device}")
    print(f"{'━'*62}")

    # ── LPIPS network ─────────────────────────────────────────────────────────
    lpips_net = None
    if _HAS_LPIPS:
        try:
            import lpips as _lp
            lpips_net = _lp.LPIPS(net="alex").to(device)
            lpips_net.eval()
            print("  LPIPS   : AlexNet ✓")
        except Exception as exc:
            warnings.warn(f"LPIPS init failed: {exc}")
    else:
        print("  LPIPS   : disabled (pip install lpips to enable)")

    eval_split_dir = getattr(opts, "eval_split_dir", None)
    if eval_split_dir:
        noisy_dir = Path(eval_split_dir) / "low"
    else:
        noisy_dir = Path(opts.data_root) / "Test" / "low"
    records: List[Dict] = []

    if opts.task == "spatial_2p5d" and not (getattr(opts, "eval_full_image", False) or eval_split_dir):
        raise ValueError(
            "spatial_2p5d evaluation currently supports full-image mode only. "
            "Use --eval_full_image (and optional --eval_split_dir)."
        )

    if getattr(opts, "eval_full_image", False) or eval_split_dir:
        # ── Full-image evaluation: one 1024×1024 forward per test image ──────
        if opts.task == "spatial_2p5d":
            full_ds = NucleiTemporal2p5DDataset(
                root=opts.data_root,
                split="test",
                mode="test",
                patch_size=opts.patch_size,
                patches_per_image=1,
                norm_mode=getattr(opts, "norm_mode", "minmax"),
                temporal_window=getattr(opts, "temporal_window", 3),
                split_dir_override=eval_split_dir,
                stack_policy=getattr(opts, "stack_policy", "error"),
                stack_frame_index=getattr(opts, "stack_frame_index", 0),
            )
        else:
            full_ds = NucleiDenoisingDataset(
                root              = opts.data_root,
                split             = "test",
                task              = opts.task,
                mode              = "test",          # full 1024×1024, no augmentation
                patch_size        = opts.patch_size,
                patches_per_image = 1,
                norm_mode         = getattr(opts, "norm_mode",   "minmax"),
                sparsity_factor   = opts.sparsity_factor,
                scan_pattern      = opts.scan_pattern,
                sparse_source     = getattr(opts, "sparse_source", "low"),
                use_mask_channel  = not getattr(opts, "no_mask_ch", False),
                apply_psf         = getattr(opts, "apply_psf",    False),
                psf_sigma_lateral = opts.psf_sigma_lateral,
                split_dir_override= eval_split_dir,
                stack_policy=getattr(opts, "stack_policy", "error"),
                stack_frame_index=getattr(opts, "stack_frame_index", 0),
            )
        if hasattr(full_ds, "centers"):
            stems = list(full_ds.centers)
        else:
            stems = [p[0].stem for p in full_ds.pairs]
        print(f"  Test set: {len(stems)} images  |  full-image forward\n")

        loader = DataLoader(
            full_ds, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        for (inp, target), stem in tqdm(
            zip(loader, stems), desc=f"  [{run_name}]", ncols=72, leave=True,
            total=len(stems),
        ):
            # inp/target are in model range [-1,1]
            inp    = inp.to(device)
            target = target.to(device)
            with torch.no_grad():
                # Swin window partition requires token map divisibility; for 490x490
                # CIDC25 frames we reflect-pad to multiples of 32 then crop back.
                inp_pad, H_orig, W_orig = _pad_to_multiple_2d(inp, multiple=32)
                pred = model(inp_pad)
                target_pad, _, _ = _pad_to_multiple_2d(target, multiple=32)

            pred_full = ((pred.squeeze().cpu().numpy()       + 1.0) / 2.0).clip(0.0, 1.0)
            gt_full   = ((target_pad.squeeze().cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0)
            pred_full = pred_full[:H_orig, :W_orig]
            gt_full   = gt_full[:H_orig, :W_orig]

            # Load noisy image for delta metrics
            noisy_np = None
            for ext in (".tif", ".tiff", ".png", ".jpg"):
                p = noisy_dir / f"{stem}{ext}"
                if p.exists():
                    raw = load_tiff_float32(
                        p,
                        stack_policy=getattr(opts, "stack_policy", "error"),
                        frame_index=getattr(opts, "stack_frame_index", 0),
                    )[:H_orig, :W_orig]
                    noisy_np, _ = normalise_pair(
                        raw, gt_full, mode=getattr(opts, "norm_mode", "minmax")
                    )
                    break

            m = compute_all_metrics(pred_full, gt_full, noisy_np, lpips_net, device, cnr_thresh)
            m.update(image=stem, run=run_name, epoch=epoch)
            records.append(m)

            if save_images:
                np.save(str(img_dir / f"{stem}_pred.npy"),  pred_full)
                np.save(str(img_dir / f"{stem}_gt.npy"),    gt_full)
                if noisy_np is not None:
                    np.save(str(img_dir / f"{stem}_noisy.npy"), noisy_np)
                if _HAS_TIFFFILE:
                    _tifffile.imwrite(
                        str(img_dir / f"{stem}_pred.tif"),
                        (pred_full * 65535).astype(np.uint16),
                    )

            tqdm.write(
                f"  {stem:30s}  PSNR={m['psnr']:5.2f} dB  "
                f"SSIM={m['ssim']:.4f}  sSNR={m['s_snr']:+.2f} dB  "
                f"FwPSNR={m['fw_psnr']:.2f} dB"
            )
    else:
        # ── Sliding-window grid evaluation (default, memory-efficient) ───────
        grid_ds = NucleiGridDataset(
            root              = opts.data_root,
            task              = opts.task,
            patch_size        = opts.patch_size,
            norm_mode         = getattr(opts, "norm_mode",         "minmax"),
            sparsity_factor   = opts.sparsity_factor,
            scan_pattern      = opts.scan_pattern,
            sparse_source     = getattr(opts, "sparse_source",     "low"),
            use_mask_channel  = not getattr(opts, "no_mask_ch",    False),
            apply_psf         = getattr(opts, "apply_psf",         False),
            psf_sigma_lateral = opts.psf_sigma_lateral,
        )

        stems = sorted({item["stem"] for item in grid_ds._items})
        print(f"  Test set: {len(stems)} images  |  {len(grid_ds)} grid patches\n")

        for stem in tqdm(stems, desc=f"  [{run_name}]", ncols=72, leave=True):
            # All patches for this image
            indices = [i for i, it in enumerate(grid_ds._items) if it["stem"] == stem]
            meta0   = grid_ds._items[indices[0]]
            H_orig, W_orig = meta0["H"],     meta0["W"]
            H_pad,  W_pad  = meta0["H_pad"], meta0["W_pad"]

            loader = DataLoader(Subset(grid_ds, indices),
                                batch_size=8, shuffle=False,
                                num_workers=0, pin_memory=True)

            pred_full, gt_full = assemble_full_image(
                model, loader, opts.patch_size, H_pad, W_pad, device
            )
            # Crop back to original size
            pred_full = pred_full[:H_orig, :W_orig]
            gt_full   = gt_full  [:H_orig, :W_orig]

            # Load noisy image for delta metrics
            noisy_np = None
            for ext in (".tif", ".tiff", ".png", ".jpg"):
                p = noisy_dir / f"{stem}{ext}"
                if p.exists():
                    raw = load_tiff_float32(
                        p,
                        stack_policy=getattr(opts, "stack_policy", "error"),
                        frame_index=getattr(opts, "stack_frame_index", 0),
                    )[:H_orig, :W_orig]
                    noisy_np, _ = normalise_pair(
                        raw, gt_full, mode=getattr(opts, "norm_mode", "minmax")
                    )
                    break

            # ── Metrics ───────────────────────────────────────────────────────────
            m = compute_all_metrics(
                pred_full, gt_full, noisy_np, lpips_net, device, cnr_thresh
            )
            m.update(image=stem, run=run_name, epoch=epoch)
            records.append(m)

            # ── Save outputs ───────────────────────────────────────────────────────
            if save_images:
                np.save(str(img_dir / f"{stem}_pred.npy"),  pred_full)
                np.save(str(img_dir / f"{stem}_gt.npy"),    gt_full)
                if noisy_np is not None:
                    np.save(str(img_dir / f"{stem}_noisy.npy"), noisy_np)
                if _HAS_TIFFFILE:
                    _tifffile.imwrite(
                        str(img_dir / f"{stem}_pred.tif"),
                        (pred_full * 65535).astype(np.uint16),
                    )

            tqdm.write(
                f"  {stem:30s}  PSNR={m['psnr']:5.2f} dB  "
                f"SSIM={m['ssim']:.4f}  sSNR={m['s_snr']:+.2f} dB  "
                f"FwPSNR={m['fw_psnr']:.2f} dB"
            )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    df = pd.DataFrame(records)
    metric_cols = [
        c for c in df.columns
        if c not in ("image", "run", "epoch")
        and pd.api.types.is_float_dtype(df[c])
    ]

    summary_rows = []
    for col in metric_cols:
        vals = df[col].dropna().values
        lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap)
        summary_rows.append({
            "metric":  col,
            "mean":    float(np.nanmean(vals)) if len(vals) else float("nan"),
            "std":     float(np.nanstd(vals))  if len(vals) else float("nan"),
            "median":  float(np.nanmedian(vals)) if len(vals) else float("nan"),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n":       len(vals),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("metric")

    # Write CSV outputs
    df.to_csv(str(output_dir / f"metrics_{run_name}.csv"), index=False)
    summary_df.to_csv(str(output_dir / f"summary_{run_name}.csv"))

    # Console table
    print(f"\n  {'─'*55}")
    print(f"  Summary — {run_name}")
    print(f"  {'─'*55}")
    for idx_val, row in summary_df.iterrows():
        print(
            f"  {str(idx_val):12s}  {row['mean']:8.4f} ± {row['std']:.4f}"
            f"   95% CI [{row['ci95_lo']:.4f}, {row['ci95_hi']:.4f}]"
        )
    print(f"  → {output_dir / f'metrics_{run_name}.csv'}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Ablation evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ablation(
    opts,
    checkpoint_paths: List[Path],
    run_names:        List[str],
    output_dir:       Path,
    cnr_thresh:       float = 0.5,
    n_bootstrap:      int   = 2000,
) -> pd.DataFrame:
    """
    Evaluate multiple checkpoints and produce:
      ablation_all.csv       — concatenated per-image metrics (for visualize.py)
      ablation_summary.csv   — mean ± std per run
      ablation_pvalues.csv   — Wilcoxon p-value matrix
    """
    dfs: List[pd.DataFrame] = []
    for ckpt, name in zip(checkpoint_paths, run_names):
        df = evaluate_checkpoint(
            opts, ckpt, output_dir,
            run_name=name, cnr_thresh=cnr_thresh, n_bootstrap=n_bootstrap,
        )
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(str(output_dir / "ablation_all.csv"), index=False)

    # Summary pivot
    metric_cols = [
        c for c in merged.columns
        if c not in ("image", "run", "epoch")
        and pd.api.types.is_float_dtype(merged[c])
    ]
    rows = []
    for name in run_names:
        sub = merged[merged["run"] == name]
        row: Dict = {"run": name}
        for col in metric_cols:
            vals = sub[col].dropna().values
            row[f"{col}_mean"]    = float(np.nanmean(vals)) if len(vals) else float("nan")
            row[f"{col}_std"]     = float(np.nanstd(vals))  if len(vals) else float("nan")
            lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap)
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(str(output_dir / "ablation_summary.csv"), index=False)

    # Wilcoxon p-values
    core = [m for m in ["psnr", "ssim", "ms_ssim", "lpips", "fw_psnr", "s_snr"]
            if m in merged.columns]
    pval_df = wilcoxon_pvalue_matrix(dfs, run_names, core)
    pval_df.to_csv(str(output_dir / "ablation_pvalues.csv"), index=False)

    # Console summary
    print(f"\n{'═'*62}")
    print("  ABLATION SUMMARY")
    print(f"{'═'*62}")
    display = [f"{m}_mean" for m in ["psnr", "ssim", "ms_ssim", "lpips"]
               if f"{m}_mean" in summary.columns]
    if display:
        print(summary[["run"] + display].to_string(
            index=False, float_format="{:.4f}".format
        ))
    sigs = pval_df[pval_df["significant"]]
    if len(sigs):
        print(f"\n  Significant differences (Bonferroni α=0.05):")
        for _, r in sigs.iterrows():
            print(f"    {r['metric']:10s}  {r['comparison']}  p={r['p_value']:.4f}")
    print(f"\n  Outputs → {output_dir}/")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_eval_extras() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--eval_checkpoints",  nargs="+", type=str,   default=None)
    p.add_argument("--eval_run_names",    nargs="+", type=str,   default=None)
    p.add_argument("--no_save_images",    action="store_true")
    p.add_argument("--cnr_signal_thresh", type=float, default=0.5)
    p.add_argument("--n_bootstrap",       type=int,   default=2000)
     # When True, run evaluation with a single 1024×1024 forward per image
     # instead of sliding-window patch tiling (NucleiGridDataset + assemble_full_image).
    p.add_argument("--eval_full_image",   action="store_true",
                   help="Use full-image forward (one pass per 1024×1024 image) "
                        "instead of sliding-window grid inference.")
    p.add_argument("--eval_split_dir", type=str, default=None,
                   help="Optional directory with high/ and low/ for direct evaluation "
                        "(e.g., CIDC25/Training or CIDC25/Validation).")
    extra, _ = p.parse_known_args()
    return extra


def main():
    opts  = get_opts()
    extra = _parse_eval_extras()
    for k, v in vars(extra).items():
        setattr(opts, k, v)

    out = Path(opts.output_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)

    if opts.eval_checkpoints:
        assert opts.eval_run_names, "--eval_run_names required with --eval_checkpoints"
        assert len(opts.eval_run_names) == len(opts.eval_checkpoints), \
            "--eval_run_names and --eval_checkpoints must be equal length"
        evaluate_ablation(
            opts,
            checkpoint_paths = [Path(p) for p in opts.eval_checkpoints],
            run_names        = opts.eval_run_names,
            output_dir       = out,
            cnr_thresh       = opts.cnr_signal_thresh,
            n_bootstrap      = opts.n_bootstrap,
        )
    else:
        assert opts.checkpoint, "Provide --checkpoint for single-run evaluation."
        run_name = Path(opts.checkpoint).parent.name
        evaluate_checkpoint(
            opts,
            checkpoint_path = Path(opts.checkpoint),
            output_dir      = out,
            run_name        = run_name,
            save_images     = not opts.no_save_images,
            cnr_thresh      = opts.cnr_signal_thresh,
            n_bootstrap      = opts.n_bootstrap,
        )


if __name__ == "__main__":
    main()

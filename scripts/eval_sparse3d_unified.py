#!/usr/bin/env python3
"""
Unified evaluation for Phase 1 + Phase 2 sparse 3D reconstruction models.

Evaluation protocol — reproducing the EXACT training data pipeline:
  - 2D  (spatial_2d):     Load pre-generated high/low paired frames from
                          CIDC25_sparse3d/K4_{mode}/Validation/.
                          Apply the same on-the-fly mask+warmstart as
                          NucleiDenoisingDataset._build_spatial2d.
  - 2.5D (spatial_2p5d):  Load pre-generated high/low paired frames.
                          Stack T low neighbours, normalise per-pair.
                          Matches NucleiTemporal2p5DDataset.
  - 3D  (spatiotemporal_3d): Load F0.tif directly, per-patch minmax
                          normalisation + on-the-fly 3D sparsification.
                          Matches CIDC25Volume3DDataset.__getitem__.

Usage:
  # Step 0: generate validation test set (if not yet done)
  python scripts/prepare_cidc25_sparse3d.py \
    --cidc_root .../AI4Life-CIDC25 --output_root .../CIDC25_sparse3d \
    --gt_stem F0 --split Validation --sparsity_factors 4 --modes xy z xyz

  # Step 1: evaluate all models
  python scripts/eval_sparse3d_unified.py \
    --cidc_root .../AI4Life-CIDC25 \
    --sparse_root .../CIDC25_sparse3d \
    --gt_stem F0 --split Validation \
    --runs_dir runs --output_dir eval_sparse3d
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F_torch
import tifffile
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import (
    load_tiff_float32,
    normalise_pair,
    make_mask_2d,
    make_mask_3d,
    warm_start_2d,
    warm_start_3d,
    wiener_init_2d,
    wiener_init_3d,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim_2d(pred: np.ndarray, gt: np.ndarray, win: int = 7) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = uniform_filter(pred, size=win)
    mu_g = uniform_filter(gt, size=win)
    sig_pp = uniform_filter(pred * pred, size=win) - mu_p ** 2
    sig_gg = uniform_filter(gt * gt, size=win) - mu_g ** 2
    sig_pg = uniform_filter(pred * gt, size=win) - mu_p * mu_g
    num = (2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)
    den = (mu_p ** 2 + mu_g ** 2 + C1) * (sig_pp + sig_gg + C2)
    return float(np.mean(num / den))


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def parse_run_config(run_name: str) -> Dict:
    cfg = {"run_name": run_name, "algorithm": "?", "sparse_mode": "?",
           "K": 4, "extras": ""}
    m = re.match(r"sparse3d_(2d|2p5d|3d)_(xy|z|xyz)_K(\d+)(.*)", run_name)
    if m:
        cfg["algorithm"] = m.group(1)
        cfg["sparse_mode"] = m.group(2)
        cfg["K"] = int(m.group(3))
        cfg["extras"] = m.group(4).strip("_") if m.group(4) else ""
    return cfg


def load_model(run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "best.pth"
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    opts = argparse.Namespace(**ckpt.get("opts", {}))
    state = ckpt["model_state"]
    w = state.get("patch_embed.weight")
    if w is not None:
        opts.force_in_channels = int(w.shape[1])
    from models import build_model
    model = build_model(opts)
    model.load_state_dict(state)
    return model.to(device).eval(), opts


def build_psf_map_2d(H: int, W: int, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k2d = torch.exp(-0.5 * (x.unsqueeze(0)**2 + x.unsqueeze(1)**2) / sigma**2)
    k2d /= k2d.max()
    psf_map = torch.zeros(1, H, W)
    cy, cx = H // 2, W // 2
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    ky0, kx0 = r - (cy - y0), r - (cx - x0)
    psf_map[0, y0:y1, x0:x1] = k2d[ky0:ky0+(y1-y0), kx0:kx0+(x1-x0)]
    return psf_map


# ═══════════════════════════════════════════════════════════════════════════════
#  2D evaluation — mirrors NucleiDenoisingDataset._build_spatial2d
# ═══════════════════════════════════════════════════════════════════════════════

def load_paired_frames(val_dir: Path) -> List[Tuple[Path, Path]]:
    """Load matched (high, low) frame paths from a Validation directory."""
    high_dir = val_dir / "high"
    low_dir = val_dir / "low"
    high_map = {p.stem: p for p in sorted(high_dir.iterdir())
                if p.suffix.lower() in {".tif", ".tiff"}}
    low_map = {p.stem: p for p in sorted(low_dir.iterdir())
               if p.suffix.lower() in {".tif", ".tiff"}}
    common = sorted(set(high_map) & set(low_map))
    return [(high_map[s], low_map[s]) for s in common]


@torch.no_grad()
def evaluate_2d(
    model, opts, val_dir: Path,
    device: torch.device, max_frames: int, patch_hw: int,
) -> List[Dict]:
    """Mirrors _build_spatial2d: mask → warmstart/Wiener on 'low' source."""
    pairs = load_paired_frames(val_dir)
    K = opts.sparsity_factor
    pattern = opts.scan_pattern
    use_mask = not getattr(opts, "no_mask_ch", False)
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    norm_mode = getattr(opts, "norm_mode", "minmax")
    rows = []

    for i, (hp, lp) in enumerate(pairs[:max_frames]):
        high_np = load_tiff_float32(hp, stack_policy="first")
        low_np = load_tiff_float32(lp, stack_policy="first")
        low_n, high_n = normalise_pair(low_np, high_np, mode=norm_mode)

        H, W = high_n.shape
        low_t = torch.from_numpy(low_n.copy()).unsqueeze(0)   # [1,H,W]
        high_t = torch.from_numpy(high_n.copy()).unsqueeze(0)  # [1,H,W]
        high_m = high_t * 2.0 - 1.0

        # _build_spatial2d logic (sparse_source='low')
        source = low_t
        mask = make_mask_2d(H, W, K, pattern)
        sparse = source * mask
        if use_wiener:
            interp = wiener_init_2d(
                sparse.squeeze(0), psf_sig
            ).unsqueeze(0)
        else:
            interp = warm_start_2d(sparse, K)
        interp_m = interp * 2.0 - 1.0
        if use_mask:
            inp = torch.cat([interp_m, mask], dim=0)
        else:
            inp = interp_m

        # Full-image inference (avoid tile seam artifacts).
        pred_frame = _infer_2d_full(model, inp, device)
        gt_01 = high_n
        rows.append({"frame": i, "psnr": psnr(pred_frame, gt_01),
                      "ssim": ssim_2d(pred_frame, gt_01)})
        if i % 50 == 0:
            print(f"    frame {i}/{min(len(pairs), max_frames)}  "
                  f"PSNR={rows[-1]['psnr']:.2f}")
    return rows


@torch.no_grad()
def _infer_2d_full(
    model, full_inp: torch.Tensor, device: torch.device,
) -> np.ndarray:
    """Run 2D model on full image (no tiling), with adaptive auto-padding."""
    C, H, W = full_inp.shape
    x0 = full_inp.unsqueeze(0)  # [1,C,H,W]
    last_err = None
    # Try increasing align factors to satisfy different Swin window/downsample configs.
    for align in (32, 64, 96, 128, 160, 192, 224, 256):
        H_pad = ((H + align - 1) // align) * align
        W_pad = ((W + align - 1) // align) * align
        x = x0
        if H_pad != H or W_pad != W:
            pad_h = H_pad - H
            pad_w = W_pad - W
            # Use replicate padding to avoid reflect-mode constraint when pad >= input size.
            x = F_torch.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        try:
            out = model(x.to(device))
            out_np = ((out[0, 0].cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            return out_np[:H, :W]
        except RuntimeError as e:
            last_err = e
            if "shape" not in str(e) and "view" not in str(e):
                raise
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("Full-image inference failed without RuntimeError details.")


# ═══════════════════════════════════════════════════════════════════════════════
#  2.5D evaluation — mirrors NucleiTemporal2p5DDataset
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_2p5d(
    model, opts, val_dir: Path,
    device: torch.device, max_frames: int, patch_hw: int,
) -> List[Dict]:
    """Evaluate 2.5D in both legacy and mask-aware CIDC25 modes."""
    pairs = load_paired_frames(val_dir)
    tw = getattr(opts, "temporal_window", 3)
    radius = tw // 2
    norm_mode = getattr(opts, "norm_mode", "minmax")
    use_mask = not getattr(opts, "no_mask_ch", False)
    use_wiener = getattr(opts, "wiener_init", False)
    K = getattr(opts, "sparsity_factor", 4)
    Kz = getattr(opts, "sparsity_factor_z", 4)
    sparse_mode = getattr(opts, "sparse_mode", "xy")
    pattern = getattr(opts, "scan_pattern", "grid")
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    rows = []

    # IMPORTANT: decide input construction from model's actual expected channels.
    # This avoids mismatch for legacy checkpoints where opts.no_mask_ch=False
    # but model was trained with in_ch=temporal_window (no mask channels).
    expected_in_ch = None
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "in_channels"):
        expected_in_ch = int(model.patch_embed.in_channels)
    if expected_in_ch is None:
        # Conservative fallback to legacy behavior.
        expected_in_ch = tw

    use_legacy_input = expected_in_ch == tw
    use_masked_input = expected_in_ch == (2 * tw)
    if not use_legacy_input and not use_masked_input:
        raise ValueError(
            f"Unsupported 2.5D in_ch={expected_in_ch}; expected {tw} or {2 * tw}."
        )

    # Build sequence index (same as NucleiTemporal2p5DDataset)
    _PAT = re.compile(r"^(?P<prefix>.+)_f(?P<idx>\d+)$")
    stem_to_idx = {}
    idx_to_stem = {}
    for hp, lp in pairs:
        m = _PAT.match(hp.stem)
        if m:
            idx = int(m.group("idx"))
            stem_to_idx[hp.stem] = idx
            idx_to_stem[idx] = hp.stem

    # high/low maps
    high_map = {hp.stem: hp for hp, _ in pairs}
    low_map = {hp.stem: lp for hp, lp in pairs}

    sorted_idxs = sorted(idx_to_stem.keys())
    eval_count = 0

    for c_idx in sorted_idxs:
        if eval_count >= max_frames:
            break
        # Check all T neighbors exist
        ok = all((c_idx + d) in idx_to_stem
                 for d in range(-radius, radius + 1))
        if not ok:
            continue

        center_stem = idx_to_stem[c_idx]
        high_np = load_tiff_float32(high_map[center_stem], stack_policy="first")
        center_low_np = load_tiff_float32(low_map[center_stem], stack_policy="first")

        # Legacy 2.5D path: input [T,H,W] from pre-generated low frames.
        if use_legacy_input:
            _, high_n = normalise_pair(center_low_np, high_np, mode=norm_mode)
            if norm_mode == "minmax":
                g_min = min(float(center_low_np.min()), float(high_np.min()))
                g_max = max(float(center_low_np.max()), float(high_np.max()))
                denom = g_max - g_min + 1e-8

            low_frames = []
            for d in range(-radius, radius + 1):
                ns = idx_to_stem[c_idx + d]
                lf = load_tiff_float32(low_map[ns], stack_policy="first")
                if norm_mode == "minmax":
                    lf_n = ((lf - g_min) / denom).clip(0, 1).astype(np.float32)
                else:
                    lf_n = lf
                low_frames.append(lf_n)

            low_stack = np.stack(low_frames, axis=0)  # [T, H, W]
            H, W = high_n.shape
            low_t = torch.from_numpy(low_stack.copy())  # [T, H, W]
            inp_2p5d = low_t * 2.0 - 1.0  # [-1, 1]
        else:
            # New CIDC25 2.5D path: build mask-aware input [2T,H,W].
            frames_raw = []
            for d in range(-radius, radius + 1):
                ns = idx_to_stem[c_idx + d]
                hf = load_tiff_float32(high_map[ns], stack_policy="first")
                frames_raw.append(hf.astype(np.float32))
            frame_stack = np.stack(frames_raw, axis=0)  # [T, H, W]

            if norm_mode == "minmax":
                vmin = float(frame_stack.min())
                vmax = float(frame_stack.max())
                denom = vmax - vmin + 1e-8
                frame_stack = ((frame_stack - vmin) / denom).clip(0, 1).astype(np.float32)

            H, W = frame_stack.shape[-2:]
            high_n = frame_stack[radius]
            mask_3d = make_mask_3d(
                tw, H, W, K, pattern,
                sparse_mode=sparse_mode,
                sparsity_factor_z=Kz,
            )[0]

            interp_list = []
            mask_list = []
            for t in range(tw):
                ft = torch.from_numpy(frame_stack[t]).unsqueeze(0)  # [1,H,W]
                mt = mask_3d[t].unsqueeze(0)
                sp = ft * mt
                if use_wiener:
                    it = wiener_init_2d(sp.squeeze(0), psf_sig).unsqueeze(0)
                else:
                    it = warm_start_2d(sp, K)
                interp_list.append(it)
                mask_list.append(mt)

            interp_all = torch.cat(interp_list, dim=0)  # [T,H,W]
            inp_2p5d = interp_all * 2.0 - 1.0
            mask_all = torch.cat(mask_list, dim=0)
            inp_2p5d = torch.cat([inp_2p5d, mask_all], dim=0)  # [2T,H,W]

        pred_frame = _infer_2d_full(model, inp_2p5d, device)

        rows.append({"frame": c_idx, "psnr": psnr(pred_frame, high_n),
                      "ssim": ssim_2d(pred_frame, high_n)})
        eval_count += 1
        if eval_count % 50 == 0:
            print(f"    center={c_idx}  #{eval_count}/{max_frames}  "
                  f"PSNR={rows[-1]['psnr']:.2f}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  3D evaluation — mirrors CIDC25Volume3DDataset.__getitem__
# ═══════════════════════════════════════════════════════════════════════════════

def _read_volume_for_3d(cidc_root: Path, split: str, gt_stem: str) -> np.ndarray:
    """Load stack using the same normalisation as CIDC25Volume3DDataset.__init__."""
    gt_path = cidc_root / split / f"{gt_stem}.tif"
    raw = tifffile.imread(str(gt_path))
    while raw.ndim > 3:
        raw = raw[:, 0]
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    elif raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    arr = raw.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    return ((arr - vmin) / (vmax - vmin + 1e-8)).clip(0, 1)


@torch.no_grad()
def evaluate_3d(
    model, opts, volume: np.ndarray,
    device: torch.device, max_frames: int, patch_hw: int,
) -> List[Dict]:
    """Per-patch minmax normalise → on-the-fly 3D sparse → model."""
    sparse_mode = getattr(opts, "sparse_mode", "xy")
    K = opts.sparsity_factor
    Kz = getattr(opts, "sparsity_factor_z", 4)
    vol_T = getattr(opts, "volume_patch_t", 16)
    use_mask = not getattr(opts, "no_mask_ch", False)
    has_psf_ch = getattr(opts, "psf_input_channel", False)
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    psf_sig_ax = getattr(opts, "psf_sigma_axial", 3.0)

    F_total, H, W = volume.shape
    n = min(F_total, max_frames)
    all_pred = np.zeros((n, H, W), dtype=np.float32)
    gt_normed = np.zeros((n, H, W), dtype=np.float32)

    for t0 in range(0, n, vol_T):
        t1 = min(t0 + vol_T, n)
        actual_t = t1 - t0

        chunk_raw = volume[t0:t1].copy()
        # Per-patch minmax (same as CIDC25Volume3DDataset)
        vmin, vmax = chunk_raw.min(), chunk_raw.max()
        denom = vmax - vmin + 1e-8
        chunk_n = ((chunk_raw - vmin) / denom).clip(0, 1).astype(np.float32)
        gt_normed[t0:t1] = chunk_n

        if actual_t < vol_T:
            pad = np.zeros((vol_T - actual_t, H, W), dtype=np.float32)
            chunk_n = np.concatenate([chunk_n, pad], axis=0)

        for y0 in range(0, H, patch_hw):
            for x0 in range(0, W, patch_hw):
                y1, x1 = min(y0 + patch_hw, H), min(x0 + patch_hw, W)
                ph, pw = y1 - y0, x1 - x0
                tile = chunk_n[:, y0:y1, x0:x1]
                if ph < patch_hw or pw < patch_hw:
                    tp = np.zeros((vol_T, patch_hw, patch_hw), dtype=np.float32)
                    tp[:, :ph, :pw] = tile
                    tile = tp

                TT, HH, WW = tile.shape
                gt_t = torch.from_numpy(tile).unsqueeze(0)  # [1,T,H,W]
                mask = make_mask_3d(TT, HH, WW, K, "grid",
                                    sparse_mode=sparse_mode,
                                    sparsity_factor_z=Kz)
                sparse = gt_t * mask

                if use_wiener:
                    interp = wiener_init_3d(
                        sparse.squeeze(0), psf_sig, psf_sig_ax
                    ).unsqueeze(0)
                else:
                    fz = Kz if sparse_mode in ("z", "xyz") else 1
                    fxy = K if sparse_mode in ("xy", "xyz") else 1
                    interp = warm_start_3d(sparse, factor=fxy, factor_z=fz)

                interp_m = interp * 2.0 - 1.0
                channels = [interp_m]
                if use_mask:
                    channels.append(mask)
                if has_psf_ch:
                    psf_ch = build_psf_map_2d(
                        HH, WW, psf_sig
                    ).unsqueeze(1).expand(1, TT, HH, WW)
                    channels.append(psf_ch)

                inp = torch.cat(channels, dim=0).unsqueeze(0).to(device)
                out = model(inp)
                out_np = ((out[0, 0].cpu().numpy() + 1.0) / 2.0).clip(0, 1)
                all_pred[t0:t1, y0:y1, x0:x1] = out_np[:actual_t, :ph, :pw]

        print(f"    chunk [{t0}:{t1}] done")

    rows = []
    for f in range(n):
        rows.append({
            "frame": f,
            "psnr": psnr(all_pred[f], gt_normed[f]),
            "ssim": ssim_2d(all_pred[f], gt_normed[f]),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_run(
    run_dir: Path, sparse_root: Path, cidc_root: Path,
    gt_stem: str, split: str, vol_3d: Optional[np.ndarray],
    device: torch.device, max_frames: int, patch_hw: int,
) -> Tuple[Dict, List[Dict]]:
    run_name = run_dir.name
    cfg = parse_run_config(run_name)
    algo, smode, K = cfg["algorithm"], cfg["sparse_mode"], cfg["K"]

    print(f"\n{'='*60}")
    print(f"  {run_name}  (algo={algo}, sparse={smode}, K={K}, "
          f"extras={cfg['extras'] or 'none'})")
    print(f"{'='*60}")

    model, opts = load_model(run_dir, device)
    t0 = time.time()

    val_dir = sparse_root / f"K{K}_{smode}" / split

    if algo == "3d":
        per_frame = evaluate_3d(model, opts, vol_3d, device, max_frames, patch_hw)
    elif algo == "2p5d":
        per_frame = evaluate_2p5d(model, opts, val_dir, device, max_frames, patch_hw)
    else:
        per_frame = evaluate_2d(model, opts, val_dir, device, max_frames, patch_hw)

    elapsed = time.time() - t0
    del model
    torch.cuda.empty_cache()

    psnr_v = [r["psnr"] for r in per_frame]
    ssim_v = [r["ssim"] for r in per_frame]
    return {
        "run_name": run_name, "algorithm": algo, "sparse_mode": smode,
        "extras": cfg["extras"], "n_frames": len(per_frame),
        "psnr_mean": float(np.mean(psnr_v)),
        "psnr_std": float(np.std(psnr_v)),
        "ssim_mean": float(np.mean(ssim_v)),
        "ssim_std": float(np.std(ssim_v)),
        "time_s": round(elapsed, 1),
    }, per_frame


def compute_baseline_2d(val_dir: Path, max_frames: int, norm_mode: str = "minmax"):
    """Baseline: degraded low vs GT high (= input quality before model)."""
    pairs = load_paired_frames(val_dir)
    ps_list, ss_list = [], []
    for hp, lp in pairs[:max_frames]:
        high_np = load_tiff_float32(hp, stack_policy="first")
        low_np = load_tiff_float32(lp, stack_policy="first")
        low_n, high_n = normalise_pair(low_np, high_np, mode=norm_mode)
        ps_list.append(psnr(low_n, high_n))
        ss_list.append(ssim_2d(low_n, high_n))
    return float(np.mean(ps_list)), float(np.std(ps_list)), \
        float(np.mean(ss_list)), float(np.std(ss_list)), len(ps_list)


def main():
    p = argparse.ArgumentParser(
        description="Unified evaluation for sparse 3D reconstruction models."
    )
    p.add_argument("--cidc_root", type=str, required=True)
    p.add_argument("--sparse_root", type=str, required=True,
                   help="Root of pre-generated sparse data (CIDC25_sparse3d/).")
    p.add_argument("--gt_stem", type=str, default="F0")
    p.add_argument("--split", type=str, default="Validation")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--patch_hw", type=int, default=256)
    p.add_argument("--run_filter", type=str, default="sparse3d_")
    p.add_argument(
        "--sparsity_k",
        type=int,
        default=None,
        help="If set, only evaluate runs/baselines for this K (e.g. 2 or 4).",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cidc_root = Path(args.cidc_root)
    sparse_root = Path(args.sparse_root)

    # ── Baseline: low vs high for each sparse mode / each available K ────────
    print("\nBaseline input quality (low vs high, no model):")
    baselines = []
    baseline_dirs = sorted([
        d for d in sparse_root.glob("K*_*/")
        if d.is_dir() and (d / args.split / "high").is_dir()
    ])
    for d in baseline_dirs:
        name = d.name
        m = re.match(r"K(\d+)_(xy|z|xyz)$", name)
        if not m:
            continue
        k = int(m.group(1))
        if args.sparsity_k is not None and k != args.sparsity_k:
            continue
        smode = m.group(2)
        vd = d / args.split
        pm, ps, sm, ss, nf = compute_baseline_2d(vd, args.max_frames)
        print(f"  K{k}_{smode}: PSNR={pm:.2f}±{ps:.2f}  SSIM={sm:.4f}±{ss:.4f}  "
              f"({nf} frames)")
        baselines.append({
            "run_name": f"baseline_input_{smode}_K{k}",
            "algorithm": "input", "sparse_mode": smode, "extras": f"K{k}",
            "n_frames": nf,
            "psnr_mean": pm, "psnr_std": ps,
            "ssim_mean": sm, "ssim_std": ss,
            "time_s": 0.0,
        })

    # ── Load 3D volume for spatiotemporal_3d models ───────────────────────────
    vol_3d = None
    runs_dir = Path(args.runs_dir)
    run_dirs = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(args.run_filter)
        and (d / "best.pth").exists()
    ])
    if args.sparsity_k is not None:
        sk = args.sparsity_k
        run_dirs = [
            d for d in run_dirs
            if re.search(rf"_K{sk}(_|$)", d.name)
        ]
    has_3d = any(parse_run_config(d.name)["algorithm"] == "3d" for d in run_dirs)
    if has_3d:
        print(f"\nLoading 3D volume for spatiotemporal_3d evaluation...")
        vol_3d = _read_volume_for_3d(cidc_root, args.split, args.gt_stem)
        print(f"  shape={vol_3d.shape}, range=[{vol_3d.min():.4f}, {vol_3d.max():.4f}]")

    # ── Discover and evaluate ─────────────────────────────────────────────────
    print(f"\nFound {len(run_dirs)} runs:")
    for d in run_dirs:
        print(f"  {d.name}")

    all_summaries = list(baselines)
    for rd in run_dirs:
        try:
            summary, per_frame = evaluate_run(
                rd, sparse_root, cidc_root,
                args.gt_stem, args.split, vol_3d,
                device, args.max_frames, args.patch_hw,
            )
            all_summaries.append(summary)

            run_out = output_dir / rd.name
            run_out.mkdir(parents=True, exist_ok=True)
            with open(run_out / "metrics.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["frame", "psnr", "ssim"])
                w.writeheader()
                w.writerows(per_frame)

            s = summary
            print(f"  => PSNR={s['psnr_mean']:.2f}±{s['psnr_std']:.2f}  "
                  f"SSIM={s['ssim_mean']:.4f}±{s['ssim_std']:.4f}  "
                  f"({s['n_frames']} frames, {s['time_s']}s)")
        except Exception:
            import traceback
            traceback.print_exc()
            all_summaries.append({
                "run_name": rd.name, "algorithm": "?", "sparse_mode": "?",
                "extras": "", "n_frames": 0,
                "psnr_mean": 0.0, "psnr_std": 0.0,
                "ssim_mean": 0.0, "ssim_std": 0.0,
                "time_s": 0.0,
            })

    # ── Write summary tables ──────────────────────────────────────────────────
    cols = ["run_name", "algorithm", "sparse_mode", "extras",
            "n_frames", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "time_s"]

    csv_path = output_dir / "summary_all.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_summaries)

    ranked = sorted(all_summaries, key=lambda x: -x["psnr_mean"])
    md_path = output_dir / "summary_all.md"
    with open(md_path, "w") as fh:
        fh.write("# Sparse 3D Reconstruction — Unified Evaluation\n\n")
        fh.write(f"**GT**: `{args.split}/{args.gt_stem}.tif` | "
                 f"max_frames={args.max_frames}\n\n")
        fh.write("| # | Run | Algo | Sparse | Extras | Frames "
                 "| PSNR (dB) | SSIM | Time |\n")
        fh.write("|---|-----|------|--------|--------|--------"
                 "|-----------|------|------|\n")
        for i, s in enumerate(ranked, 1):
            fh.write(
                f"| {i} | {s['run_name']} | {s['algorithm']} "
                f"| {s['sparse_mode']} | {s['extras']} "
                f"| {s['n_frames']} "
                f"| {s['psnr_mean']:.2f}±{s['psnr_std']:.2f} "
                f"| {s['ssim_mean']:.4f}±{s['ssim_std']:.4f} "
                f"| {s['time_s']}s |\n"
            )

    print(f"\n{'='*70}")
    print(f"Summary → {csv_path}")
    print(f"Markdown → {md_path}")
    print(f"{'='*70}\n")
    print("Results ranked by PSNR:\n")
    header = (f"{'#':>2}  {'Run':<42} {'Algo':>5} {'Sparse':>6}  "
              f"{'PSNR':>12}  {'SSIM':>14}  {'Time':>6}")
    print(header)
    print("-" * len(header))
    for i, s in enumerate(ranked, 1):
        print(f"{i:>2}  {s['run_name']:<42} {s['algorithm']:>5} "
              f"{s['sparse_mode']:>6}  "
              f"{s['psnr_mean']:>6.2f}±{s['psnr_std']:<5.2f}  "
              f"{s['ssim_mean']:>7.4f}±{s['ssim_std']:<6.4f}  "
              f"{s['time_s']:>5.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate per-sparse-mode overview figures for eval_sparse3d_k2_fullimg:

  - One figure per mode (xy / z / xyz)
  - 5 columns: Sparse input | 2D recon | 2.5D recon | 3D recon | GT
  - Recon models use mask + Wiener-init checkpoints
  - Row 1: full FOV; Row 2: ROI zoom
  - Colormap / colorbar / robust_range: same as plot_pollen_structural_comparison.py (magma)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F_torch
import tifffile
from scipy.ndimage import gaussian_filter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import (
    load_tiff_float32,
    make_mask_2d,
    make_mask_3d,
    normalise_pair,
    warm_start_2d,
    warm_start_3d,
    wiener_init_2d,
    wiener_init_3d,
)
from scripts.eval_sparse3d_unified import load_model
from scripts.plot_pollen_structural_comparison import robust_range

_BIO_PURPLE_CMAP: LinearSegmentedColormap | None = None


def get_bio_purple_cmap() -> LinearSegmentedColormap:
    global _BIO_PURPLE_CMAP
    if _BIO_PURPLE_CMAP is None:
        stops = [
            (0.00, (0.00, 0.00, 0.00)),
            (0.18, (0.12, 0.00, 0.22)),
            (0.42, (0.38, 0.00, 0.58)),
            (0.68, (0.62, 0.15, 0.92)),
            (0.88, (0.88, 0.45, 1.00)),
            (1.00, (1.00, 0.92, 1.00)),
        ]
        _BIO_PURPLE_CMAP = LinearSegmentedColormap.from_list("bio_purple", stops, N=256)
    return _BIO_PURPLE_CMAP


def enhance_detail_for_pseudocolor(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1.0, 99.0])
    x = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    blur = gaussian_filter(x, sigma=0.8)
    sharp = np.clip(x + 0.6 * (x - blur), 0.0, 1.0)
    return np.power(sharp, 0.92).astype(np.float32)


def load_sparse_and_gt(
    mode: str,
    k: int,
    frame_idx: int,
    sparse_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse simulated input (low) and GT (high) for one validation frame."""
    val_dir = sparse_root / f"K{k}_{mode}" / "Validation"
    hp = val_dir / "high" / f"F0_f{frame_idx:04d}.tif"
    lp = val_dir / "low" / f"F0_f{frame_idx:04d}.tif"
    high_np = load_tiff_float32(hp, stack_policy="first")
    low_np = load_tiff_float32(lp, stack_policy="first")
    low_n, high_n = normalise_pair(low_np, high_np, mode="minmax")
    return low_n, high_n


def read_volume(path: Path) -> np.ndarray:
    raw = tifffile.imread(str(path))
    while raw.ndim > 3:
        raw = raw[:, 0]
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    arr = raw.astype(np.float32)
    return ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8)).clip(0, 1)


def load_maps(val_dir: Path) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    high_map, low_map = {}, {}
    for p in sorted((val_dir / "high").glob("*.tif")):
        m = re.search(r"_f(\d+)$", p.stem)
        if m:
            high_map[int(m.group(1))] = p
    for p in sorted((val_dir / "low").glob("*.tif")):
        m = re.search(r"_f(\d+)$", p.stem)
        if m:
            low_map[int(m.group(1))] = p
    return high_map, low_map


def resolve_run(runs_dir: Path, mode: str, k: int, algo: str) -> Path:
    if algo == "2d":
        candidates = [
            f"sparse3d_2d_{mode}_K{k}_wiener",
            f"sparse3d_2d_{mode}_K{k}_wiener_mdc25",
            f"sparse3d_2d_{mode}_K{k}",
        ]
    elif algo == "2p5d":
        candidates = [
            f"sparse3d_2p5d_{mode}_K{k}_mask_wiener",
            f"sparse3d_2p5d_{mode}_K{k}_mask_wiener_mdc25",
            f"sparse3d_2p5d_{mode}_K{k}_mask",
            f"sparse3d_2p5d_{mode}_K{k}",
        ]
    else:
        candidates = [
            f"sparse3d_3d_{mode}_K{k}_wiener",
            f"sparse3d_3d_{mode}_K{k}_wiener_mdc25",
            f"sparse3d_3d_{mode}_K{k}",
        ]
    for name in candidates:
        p = runs_dir / name
        if (p / "best.pth").exists():
            return p
    raise FileNotFoundError(f"No checkpoint for {algo}/{mode}/K{k}: {candidates}")


@torch.no_grad()
def infer_2d_full(model, inp: torch.Tensor, device: torch.device) -> np.ndarray:
    _, h, w = inp.shape
    last_err = None
    for align in (32, 64, 96, 128, 160, 192, 224, 256):
        hp = ((h + align - 1) // align) * align
        wp = ((w + align - 1) // align) * align
        x = inp.unsqueeze(0)
        if hp != h or wp != w:
            x = F_torch.pad(x, (0, wp - w, 0, hp - h), mode="replicate")
        try:
            out = model(x.to(device))
            return ((out[0, 0].cpu().numpy() + 1.0) / 2.0).clip(0, 1)[:h, :w]
        except RuntimeError as e:
            last_err = e
    raise RuntimeError("2D full-image inference failed.") from last_err


def _interp_2d_frame(
    source: np.ndarray,
    mask: np.ndarray,
    k: int,
    use_wiener: bool,
    psf_sig: float,
) -> np.ndarray:
    src_t = torch.from_numpy(source.astype(np.float32)).unsqueeze(0)
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    sparse = src_t * mask_t
    if use_wiener:
        interp = wiener_init_2d(sparse.squeeze(0), psf_sig).unsqueeze(0)
    else:
        interp = warm_start_2d(sparse, k)
    return interp.squeeze(0).numpy().clip(0, 1)


@torch.no_grad()
def pipeline_2d(
    run_dir: Path,
    mode: str,
    k: int,
    frame_idx: int,
    sparse_root: Path,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, opts = load_model(run_dir, device)
    val_dir = sparse_root / f"K{k}_{mode}" / "Validation"
    hp = val_dir / "high" / f"F0_f{frame_idx:04d}.tif"
    lp = val_dir / "low" / f"F0_f{frame_idx:04d}.tif"
    high_np = load_tiff_float32(hp, stack_policy="first")
    low_np = load_tiff_float32(lp, stack_policy="first")
    low_n, high_n = normalise_pair(low_np, high_np, mode=getattr(opts, "norm_mode", "minmax"))
    h, w = high_n.shape
    mask = make_mask_2d(h, w, k, getattr(opts, "scan_pattern", "grid"))[0].numpy()
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    inp_vis = _interp_2d_frame(low_n, mask, k, use_wiener, psf_sig)

    low_t = torch.from_numpy(low_n).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    sparse = low_t * mask_t
    if use_wiener:
        interp = wiener_init_2d(sparse.squeeze(0), psf_sig).unsqueeze(0)
    else:
        interp = warm_start_2d(sparse, k)
    inp = interp * 2.0 - 1.0
    if not getattr(opts, "no_mask_ch", False):
        inp = torch.cat([inp, mask_t], dim=0)
    pred = infer_2d_full(model, inp, device)
    return inp_vis, pred, high_n


@torch.no_grad()
def pipeline_2p5d(
    run_dir: Path,
    mode: str,
    k: int,
    frame_idx: int,
    sparse_root: Path,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, opts = load_model(run_dir, device)
    val_dir = sparse_root / f"K{k}_{mode}" / "Validation"
    high_map, low_map = load_maps(val_dir)
    tw = getattr(opts, "temporal_window", 3)
    radius = tw // 2
    neigh = [frame_idx + d for d in range(-radius, radius + 1)]
    if any(n not in high_map for n in neigh):
        raise ValueError(f"Frame {frame_idx} lacks neighbors for T={tw}")

    center_high = load_tiff_float32(high_map[frame_idx], stack_policy="first")
    center_low = load_tiff_float32(low_map[frame_idx], stack_policy="first")
    _, high_n = normalise_pair(center_low, center_high, mode=getattr(opts, "norm_mode", "minmax"))
    h, w = high_n.shape

    expected_ch = int(model.patch_embed.in_channels)
    use_legacy = expected_ch == tw
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    kz = getattr(opts, "sparsity_factor_z", k)
    sparse_mode = getattr(opts, "sparse_mode", mode)

    if use_legacy:
        g_min = min(float(center_low.min()), float(center_high.min()))
        g_max = max(float(center_low.max()), float(center_high.max()))
        denom = g_max - g_min + 1e-8
        frames = []
        for n in neigh:
            lf = load_tiff_float32(low_map[n], stack_policy="first")
            frames.append(((lf - g_min) / denom).clip(0, 1).astype(np.float32))
        stack = np.stack(frames, axis=0)
        mask = np.ones((tw, h, w), dtype=np.float32)
        inp_vis = stack[radius]
        inp = torch.from_numpy(stack) * 2.0 - 1.0
    else:
        frames = []
        for n in neigh:
            hf = load_tiff_float32(high_map[n], stack_policy="first").astype(np.float32)
            frames.append(hf)
        stack = np.stack(frames, axis=0)
        stack = ((stack - stack.min()) / (stack.max() - stack.min() + 1e-8)).clip(0, 1)
        mask_3d = make_mask_3d(
            tw, h, w, k, getattr(opts, "scan_pattern", "grid"),
            sparse_mode=sparse_mode, sparsity_factor_z=kz,
        )[0].numpy()
        interp_list, mask_list = [], []
        for t in range(tw):
            interp_list.append(
                _interp_2d_frame(stack[t], mask_3d[t], k, use_wiener, psf_sig)
            )
            mask_list.append(mask_3d[t])
        inp_vis = interp_list[radius]
        interp_all = torch.from_numpy(np.stack(interp_list)).unsqueeze(1) * 2.0 - 1.0
        interp_all = interp_all.squeeze(1)
        mask_all = torch.from_numpy(np.stack(mask_list))
        inp = torch.cat([interp_all, mask_all], dim=0)

    pred = infer_2d_full(model, inp, device)
    return inp_vis, pred, high_n


@torch.no_grad()
def pipeline_3d(
    run_dir: Path,
    mode: str,
    k: int,
    frame_idx: int,
    gt_vol: np.ndarray,
    device: torch.device,
    patch_hw: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, opts = load_model(run_dir, device)
    vol_t = int(getattr(opts, "volume_patch_t", 16))
    kz = int(getattr(opts, "sparsity_factor_z", k))
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    psf_sig_ax = getattr(opts, "psf_sigma_axial", 3.0)

    f, h, w = gt_vol.shape
    t0 = max(0, min(frame_idx - vol_t // 2, f - vol_t))
    t1 = min(f, t0 + vol_t)
    chunk = gt_vol[t0:t1].copy()
    vmin, vmax = chunk.min(), chunk.max()
    chunk_n = ((chunk - vmin) / (vmax - vmin + 1e-8)).clip(0, 1).astype(np.float32)
    act_t = chunk_n.shape[0]
    if act_t < vol_t:
        pad = np.zeros((vol_t - act_t, h, w), dtype=np.float32)
        chunk_n = np.concatenate([chunk_n, pad], axis=0)

    tt, hh, ww = chunk_n.shape
    mask = make_mask_3d(tt, hh, ww, k, "grid", sparse_mode=mode, sparsity_factor_z=kz)[0].numpy()
    gt_t = torch.from_numpy(chunk_n).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    sparse = gt_t * mask_t
    if use_wiener:
        interp = wiener_init_3d(sparse.squeeze(0), psf_sig, psf_sig_ax).unsqueeze(0)
    else:
        fz = kz if mode in ("z", "xyz") else 1
        fxy = k if mode in ("xy", "xyz") else 1
        interp = warm_start_3d(sparse, factor=fxy, factor_z=fz)
    local_idx = min(max(frame_idx - t0, 0), act_t - 1)
    inp_vis = ((interp[0, local_idx].cpu().numpy())).clip(0, 1)

    pred_vol = np.zeros((tt, h, w), dtype=np.float32)
    for y0 in range(0, h, patch_hw):
        for x0 in range(0, w, patch_hw):
            y1, x1 = min(y0 + patch_hw, h), min(x0 + patch_hw, w)
            ph, pw = y1 - y0, x1 - x0
            tile = chunk_n[:, y0:y1, x0:x1]
            mask_tile = mask[:, y0:y1, x0:x1]
            if ph < patch_hw or pw < patch_hw:
                tp = np.zeros((tt, patch_hw, patch_hw), dtype=np.float32)
                mp = np.zeros((tt, patch_hw, patch_hw), dtype=np.float32)
                tp[:, :ph, :pw] = tile
                mp[:, :ph, :pw] = mask_tile
                tile, mask_tile = tp, mp
            gt_tile = torch.from_numpy(tile).unsqueeze(0)
            m_tile = torch.from_numpy(mask_tile).unsqueeze(0)
            sp = gt_tile * m_tile
            if use_wiener:
                it = wiener_init_3d(sp.squeeze(0), psf_sig, psf_sig_ax).unsqueeze(0)
            else:
                fz = kz if mode in ("z", "xyz") else 1
                fxy = k if mode in ("xy", "xyz") else 1
                it = warm_start_3d(sp, factor=fxy, factor_z=fz)
            channels = [it * 2.0 - 1.0]
            if not getattr(opts, "no_mask_ch", False):
                channels.append(m_tile)
            x = torch.cat(channels, dim=0).unsqueeze(0).to(device)
            o = model(x)[0, 0].detach().cpu().numpy()
            o = ((o + 1.0) / 2.0).clip(0, 1)
            pred_vol[:, y0:y1, x0:x1] = o[:, :ph, :pw]

    gt_frame = chunk_n[local_idx]
    return inp_vis, pred_vol[local_idx], gt_frame


def auto_roi(gt: np.ndarray, size: int = 160) -> Tuple[int, int, int, int]:
    sm = gaussian_filter(gt.astype(np.float32), sigma=2.0)
    y, x = np.unravel_index(int(np.argmax(sm)), sm.shape)
    h, w = gt.shape
    half = size // 2
    y0 = int(np.clip(y - half, 0, max(0, h - size)))
    x0 = int(np.clip(x - half, 0, max(0, w - size)))
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    return y0, x0, y1 - y0, x1 - x0


def draw_panel_magma(
    fig: plt.Figure,
    ax,
    img: np.ndarray,
    title: str,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Same as plot_pollen_structural_comparison.save_panel (magma + colorbar)."""
    display = img
    if crop is not None:
        y0, x0, rh, rw = crop
        display = img[y0 : y0 + rh, x0 : x0 + rw]
    if vmin is None or vmax is None:
        vmin, vmax = robust_range(img)
    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    if roi is not None and crop is None:
        y0, x0, rh, rw = roi
        rect = mpatches.Rectangle(
            (x0, y0), rw, rh, linewidth=2.0, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)


def draw_panel_purple(
    ax,
    img: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Legacy bio-purple LUT (no colorbar)."""
    display = img
    if crop is not None:
        y0, x0, rh, rw = crop
        display = img[y0 : y0 + rh, x0 : x0 + rw]
    enh = enhance_detail_for_pseudocolor(display)
    norm = np.clip((enh - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    rgb = get_bio_purple_cmap()(norm)[..., :3]
    ax.imshow(rgb)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if roi is not None and crop is None:
        y0, x0, rh, rw = roi
        rect = mpatches.Rectangle(
            (x0, y0), rw, rh, linewidth=2.0, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)


def make_mode_figure(
    mode: str,
    k: int,
    frame_idx: int,
    runs_dir: Path,
    sparse_root: Path,
    gt_vol: np.ndarray,
    device: torch.device,
    out_dir: Path,
    roi_size: int,
    display_style: str,
    cmap: str,
) -> None:
    run_2d = resolve_run(runs_dir, mode, k, "2d")
    run_2p5d = resolve_run(runs_dir, mode, k, "2p5d")
    run_3d = resolve_run(runs_dir, mode, k, "3d")

    print(f"[{mode}] 2D  <- {run_2d.name}")
    print(f"[{mode}] 2.5D<- {run_2p5d.name}")
    print(f"[{mode}] 3D  <- {run_3d.name}")

    sparse_in, gt = load_sparse_and_gt(mode, k, frame_idx, sparse_root)
    _, pred2d, _ = pipeline_2d(run_2d, mode, k, frame_idx, sparse_root, device)
    _, pred25, _ = pipeline_2p5d(run_2p5d, mode, k, frame_idx, sparse_root, device)
    _, pred3d, _ = pipeline_3d(run_3d, mode, k, frame_idx, gt_vol, device)

    panels = [
        ("Sparse Input", sparse_in),
        ("2D Recon", pred2d),
        ("2.5D Recon", pred25),
        ("3D Recon", pred3d),
        ("Ground Truth", gt),
    ]

    roi = auto_roi(gt, size=roi_size)

    ncols = len(panels)
    w_scale = 4.2 if display_style == "magma" else 3.6
    fig, axes = plt.subplots(2, ncols, figsize=(w_scale * ncols, 7.2), dpi=220)
    for c, (title, img) in enumerate(panels):
        if display_style == "magma":
            vmin, vmax = robust_range(img)
            draw_panel_magma(
                fig, axes[0, c], img, title, cmap=cmap, vmin=vmin, vmax=vmax, roi=roi
            )
            draw_panel_magma(
                fig, axes[1, c], img, title, cmap=cmap, vmin=vmin, vmax=vmax, crop=roi
            )
        else:
            enh = enhance_detail_for_pseudocolor(img)
            lo, hi = np.percentile(enh, [1.0, 99.0])
            draw_panel_purple(axes[0, c], img, title, lo, hi, roi=roi)
            draw_panel_purple(axes[1, c], img, title, lo, hi, crop=roi)

    axes[0, 0].set_ylabel("Global", fontsize=11)
    axes[1, 0].set_ylabel("ROI zoom", fontsize=11)
    style_tag = "magma+colorbar" if display_style == "magma" else "bio-purple"
    fig.suptitle(
        f"Sparse mode={mode.upper()} | K={k} | frame={frame_idx} | {style_tag}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if display_style == "magma" else "_purple"
    stem = f"overview_{mode}_K{k}_f{frame_idx:04d}_pseudocolor{suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    fig.clear()
    plt.close(fig)
    print(f"  saved {stem}.png/pdf")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sparse-mode overview pseudocolor (K=2/4 full-image qualitative)."
    )
    p.add_argument(
        "--cidc_root",
        type=str,
        default="/data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25",
    )
    p.add_argument(
        "--sparse_root",
        type=str,
        default="/data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d",
    )
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument(
        "--output_dir",
        type=str,
        default="eval_sparse3d_k2_fullimg/qualitative",
    )
    p.add_argument("--gt_stem", type=str, default="F0")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--modes", nargs="+", default=["xy", "z", "xyz"])
    p.add_argument("--roi_size", type=int, default=160)
    p.add_argument(
        "--display-style",
        type=str,
        default="magma",
        choices=["magma", "purple"],
        help="magma: pollen-style + colorbar; purple: legacy bio-purple LUT.",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib cmap when display-style=magma.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_vol = read_volume(Path(args.cidc_root) / "Validation" / f"{args.gt_stem}.tif")
    out_dir = Path(args.output_dir)
    runs_dir = Path(args.runs_dir)
    sparse_root = Path(args.sparse_root)

    print(f"Device: {device}")
    print(f"GT volume: {gt_vol.shape}")
    for mode in args.modes:
        make_mode_figure(
            mode=mode,
            k=args.k,
            frame_idx=args.frame,
            runs_dir=runs_dir,
            sparse_root=sparse_root,
            gt_vol=gt_vol,
            device=device,
            out_dir=out_dir,
            roi_size=args.roi_size,
            display_style=args.display_style,
            cmap=args.cmap,
        )
    print(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()

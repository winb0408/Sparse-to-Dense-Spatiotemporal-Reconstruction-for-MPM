#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile
from scipy.ndimage import rotate

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import (
    load_tiff_float32,
    normalise_pair,
    make_mask_2d,
    make_mask_3d,
    warm_start_2d,
    warm_start_3d,
    wiener_init_2d,
)
from scripts.eval_sparse3d_unified import load_model


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
    high_dir = val_dir / "high"
    low_dir = val_dir / "low"
    high_map = {}
    low_map = {}
    for p in sorted(high_dir.glob("*.tif")):
        idx = int(p.stem.split("_f")[-1])
        high_map[idx] = p
    for p in sorted(low_dir.glob("*.tif")):
        idx = int(p.stem.split("_f")[-1])
        low_map[idx] = p
    return high_map, low_map


@torch.no_grad()
def infer_2d_full(model, inp: torch.Tensor, device: torch.device) -> np.ndarray:
    """Full-image inference with padding for Swin compatibility."""
    C, H, W = inp.shape
    align = 32
    H_pad = ((H + align - 1) // align) * align
    W_pad = ((W + align - 1) // align) * align
    x = inp.unsqueeze(0)
    if H_pad != H or W_pad != W:
        import torch.nn.functional as F
        x = F.pad(x, (0, W_pad - W, 0, H_pad - H), mode="reflect")
    out = model(x.to(device))
    out_np = ((out[0, 0].cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    return out_np[:H, :W]


@torch.no_grad()
def infer_2d(run_dir: Path, mode: str, k: int, frame_idx: int, sparse_root: Path, device: torch.device) -> np.ndarray:
    model, opts = load_model(run_dir, device)
    val_dir = sparse_root / f"K{k}_{mode}" / "Validation"
    hp = val_dir / "high" / f"F0_f{frame_idx:04d}.tif"
    lp = val_dir / "low" / f"F0_f{frame_idx:04d}.tif"
    high_np = load_tiff_float32(hp, stack_policy="first")
    low_np = load_tiff_float32(lp, stack_policy="first")
    low_n, high_n = normalise_pair(low_np, high_np, mode=getattr(opts, "norm_mode", "minmax"))
    H, W = high_n.shape
    low_t = torch.from_numpy(low_n.copy()).unsqueeze(0)
    mask = make_mask_2d(H, W, opts.sparsity_factor, opts.scan_pattern)
    sparse = low_t * mask
    if getattr(opts, "wiener_init", False):
        interp = wiener_init_2d(sparse.squeeze(0), getattr(opts, "psf_sigma_lateral", 1.0)).unsqueeze(0)
    else:
        interp = warm_start_2d(sparse, opts.sparsity_factor)
    inp = interp * 2.0 - 1.0
    if not getattr(opts, "no_mask_ch", False):
        inp = torch.cat([inp, mask], dim=0)
    return infer_2d_full(model, inp, device)


@torch.no_grad()
def infer_2p5d(run_dir: Path, mode: str, k: int, frame_idx: int, sparse_root: Path, device: torch.device) -> np.ndarray:
    model, opts = load_model(run_dir, device)
    val_dir = sparse_root / f"K{k}_{mode}" / "Validation"
    high_map, low_map = load_maps(val_dir)
    tw = getattr(opts, "temporal_window", 3)
    radius = tw // 2
    neigh = [frame_idx + d for d in range(-radius, radius + 1)]
    if any(n not in high_map for n in neigh):
        raise ValueError(f"Frame {frame_idx} lacks neighbors for T={tw}")

    # decide by model in_ch
    expected_ch = int(model.patch_embed.in_channels)
    use_legacy = expected_ch == tw

    center_high = load_tiff_float32(high_map[frame_idx], stack_policy="first")
    center_low = load_tiff_float32(low_map[frame_idx], stack_policy="first")
    _, high_n = normalise_pair(center_low, center_high, mode=getattr(opts, "norm_mode", "minmax"))
    H, W = high_n.shape

    if use_legacy:
        g_min = min(float(center_low.min()), float(center_high.min()))
        g_max = max(float(center_low.max()), float(center_high.max()))
        denom = g_max - g_min + 1e-8
        frames = []
        for n in neigh:
            lf = load_tiff_float32(low_map[n], stack_policy="first")
            frames.append(((lf - g_min) / denom).clip(0, 1).astype(np.float32))
        inp = torch.from_numpy(np.stack(frames, axis=0)) * 2.0 - 1.0
    else:
        frames = []
        for n in neigh:
            hf = load_tiff_float32(high_map[n], stack_policy="first").astype(np.float32)
            frames.append(hf)
        frame_stack = np.stack(frames, axis=0)
        frame_stack = ((frame_stack - frame_stack.min()) / (frame_stack.max() - frame_stack.min() + 1e-8)).clip(0, 1)
        mask_3d = make_mask_3d(tw, H, W, k, getattr(opts, "scan_pattern", "grid"),
                               sparse_mode=getattr(opts, "sparse_mode", mode),
                               sparsity_factor_z=getattr(opts, "sparsity_factor_z", k))[0]
        interp_list, mask_list = [], []
        for t in range(tw):
            ft = torch.from_numpy(frame_stack[t]).unsqueeze(0)
            mt = mask_3d[t].unsqueeze(0)
            sp = ft * mt
            if getattr(opts, "wiener_init", False):
                it = wiener_init_2d(sp.squeeze(0), getattr(opts, "psf_sigma_lateral", 1.0)).unsqueeze(0)
            else:
                it = warm_start_2d(sp, k)
            interp_list.append(it)
            mask_list.append(mt)
        interp = torch.cat(interp_list, dim=0) * 2.0 - 1.0
        masks = torch.cat(mask_list, dim=0)
        inp = torch.cat([interp, masks], dim=0)
    return infer_2d_full(model, inp, device)


@torch.no_grad()
def infer_3d(run_dir: Path, mode: str, k: int, frame_idx: int, gt_vol: np.ndarray, device: torch.device) -> np.ndarray:
    model, opts = load_model(run_dir, device)
    vol_T = int(getattr(opts, "volume_patch_t", 16))
    Kz = int(getattr(opts, "sparsity_factor_z", k))
    F, H, W = gt_vol.shape
    t0 = max(0, min(frame_idx - vol_T // 2, F - vol_T))
    t1 = min(F, t0 + vol_T)
    chunk = gt_vol[t0:t1].copy()
    vmin, vmax = chunk.min(), chunk.max()
    chunk_n = ((chunk - vmin) / (vmax - vmin + 1e-8)).clip(0, 1).astype(np.float32)
    actual_t = chunk_n.shape[0]
    if actual_t < vol_T:
        pad = np.zeros((vol_T - actual_t, H, W), dtype=np.float32)
        chunk_n = np.concatenate([chunk_n, pad], axis=0)
    TT = chunk_n.shape[0]
    patch_hw = 256
    out = np.zeros((TT, H, W), dtype=np.float32)
    for y0 in range(0, H, patch_hw):
        for x0 in range(0, W, patch_hw):
            y1, x1 = min(y0 + patch_hw, H), min(x0 + patch_hw, W)
            ph, pw = y1 - y0, x1 - x0
            tile = chunk_n[:, y0:y1, x0:x1]
            if ph < patch_hw or pw < patch_hw:
                tp = np.zeros((TT, patch_hw, patch_hw), dtype=np.float32)
                tp[:, :ph, :pw] = tile
                tile = tp
            HH, WW = tile.shape[-2:]
            gt_t = torch.from_numpy(tile).unsqueeze(0)  # [1,T,H,W]
            mask = make_mask_3d(TT, HH, WW, k, "grid", sparse_mode=mode, sparsity_factor_z=Kz)
            sparse = gt_t * mask
            if getattr(opts, "wiener_init", False):
                from dataset import wiener_init_3d
                interp = wiener_init_3d(
                    sparse.squeeze(0),
                    getattr(opts, "psf_sigma_lateral", 1.0),
                    getattr(opts, "psf_sigma_axial", 3.0),
                ).unsqueeze(0)
            else:
                fz = Kz if mode in ("z", "xyz") else 1
                fxy = k if mode in ("xy", "xyz") else 1
                interp = warm_start_3d(sparse, factor=fxy, factor_z=fz)
            inp = [interp * 2.0 - 1.0]
            if not getattr(opts, "no_mask_ch", False):
                inp.append(mask)
            x = torch.cat(inp, dim=0).unsqueeze(0).to(device)
            o = model(x)[0, 0].detach().cpu().numpy()
            o = ((o + 1.0) / 2.0).clip(0, 1)
            out[:, y0:y1, x0:x1] = o[:, :ph, :pw]
    local_idx = min(max(frame_idx - t0, 0), actual_t - 1)
    return out[local_idx]


def oblique_mip(vol: np.ndarray, angle: float = 35.0) -> np.ndarray:
    # rotate in (T,H) plane then max-project over T
    rot = rotate(vol, angle=angle, axes=(0, 1), reshape=False, order=1, mode="nearest")
    mip = rot.max(axis=0)
    mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
    return mip


def main() -> None:
    p = argparse.ArgumentParser(description="Generate qualitative comparison panels.")
    p.add_argument("--cidc_root", type=str, required=True)
    p.add_argument("--sparse_root", type=str, required=True)
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--output_dir", type=str, default="eval_sparse3d_k2/qualitative")
    p.add_argument("--frames", type=int, nargs="+", default=[40, 100, 160])
    p.add_argument("--modes", type=str, nargs="+", default=["xy", "z", "xyz"])
    p.add_argument("--gt_stem", type=str, default="F0")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    sparse_root = Path(args.sparse_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_vol = read_volume(Path(args.cidc_root) / "Validation" / f"{args.gt_stem}.tif")
    modes = args.modes
    ks = [2, 4]

    for mode in modes:
        # build predictions cache: method->k->list(frame images)
        preds = {
            "gt": {},
            "sparse": {},
            "2d": {},
            "2p5d": {},
            "3d": {},
        }
        for k in ks:
            preds["gt"][k] = []
            preds["sparse"][k] = []
            preds["2d"][k] = []
            preds["2p5d"][k] = []
            preds["3d"][k] = []

            high_dir = sparse_root / f"K{k}_{mode}" / "Validation" / "high"
            low_dir = sparse_root / f"K{k}_{mode}" / "Validation" / "low"
            run2d = runs_dir / f"sparse3d_2d_{mode}_K{k}"
            run2p5d = runs_dir / f"sparse3d_2p5d_{mode}_K{k}"
            run3d = runs_dir / f"sparse3d_3d_{mode}_K{k}"
            for fidx in args.frames:
                hp = high_dir / f"F0_f{fidx:04d}.tif"
                lp = low_dir / f"F0_f{fidx:04d}.tif"
                gt = load_tiff_float32(hp, stack_policy="first")
                sp = load_tiff_float32(lp, stack_policy="first")
                # normalise for display
                gt_d = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
                sp_d = (sp - sp.min()) / (sp.max() - sp.min() + 1e-8)
                preds["gt"][k].append(gt_d)
                preds["sparse"][k].append(sp_d)
                preds["2d"][k].append(infer_2d(run2d, mode, k, fidx, sparse_root, device))
                preds["2p5d"][k].append(infer_2p5d(run2p5d, mode, k, fidx, sparse_root, device))
                preds["3d"][k].append(infer_3d(run3d, mode, k, fidx, gt_vol, device))

        # panel: 6 rows x 5 cols
        # rows 0-2: K2 (3 representative frames), rows 3-5: K4 (same 3 frames)
        fig, axes = plt.subplots(6, 5, figsize=(12, 14))
        col_titles = ["GT", "Sparse input", "2D", "2.5D", "3D"]
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t, fontsize=11)
        for r, fidx in enumerate(args.frames):
            ims_k2 = [preds["gt"][2][r], preds["sparse"][2][r], preds["2d"][2][r], preds["2p5d"][2][r], preds["3d"][2][r]]
            ims_k4 = [preds["gt"][4][r], preds["sparse"][4][r], preds["2d"][4][r], preds["2p5d"][4][r], preds["3d"][4][r]]
            for c in range(5):
                axes[r, c].imshow(ims_k2[c], cmap="gray", vmin=0, vmax=1)
                axes[r + 3, c].imshow(ims_k4[c], cmap="gray", vmin=0, vmax=1)
                axes[r, c].axis("off")
                axes[r + 3, c].axis("off")
            axes[r, 0].set_ylabel(f"K2 f{fidx}", fontsize=10)
            axes[r + 3, 0].set_ylabel(f"K4 f{fidx}", fontsize=10)
        plt.suptitle(f"Mode={mode.upper()} | Full-image qualitative comparison", fontsize=14)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(out_dir / f"qual_panel_{mode}.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig)

        # oblique view panel using lightweight pseudo-volumes from representative frames
        fig2, ax2 = plt.subplots(len(args.frames), 3, figsize=(9, 3 * len(args.frames)))
        if len(args.frames) == 1:
            ax2 = np.expand_dims(ax2, axis=0)
        for c, t in enumerate(["GT oblique", "3D K2 oblique", "3D K4 oblique"]):
            ax2[0, c].set_title(t, fontsize=10)
        for r, fidx in enumerate(args.frames):
            t0 = max(0, fidx - 8)
            t1 = min(gt_vol.shape[0], fidx + 8)
            vol_gt = gt_vol[t0:t1]
            # Build pseudo-volumes from already inferred representative 3D frames
            # (fast and sufficient for qualitative oblique comparison).
            rec_k2_center = preds["3d"][2][r]
            rec_k4_center = preds["3d"][4][r]
            rec_k2 = np.stack([rec_k2_center] * max(3, vol_gt.shape[0]), axis=0)
            rec_k4 = np.stack([rec_k4_center] * max(3, vol_gt.shape[0]), axis=0)
            ims = [oblique_mip(vol_gt), oblique_mip(rec_k2), oblique_mip(rec_k4)]
            for c in range(3):
                ax2[r, c].imshow(ims[c], cmap="gray", vmin=0, vmax=1)
                ax2[r, c].axis("off")
            ax2[r, 0].set_ylabel(f"Frame {fidx}", fontsize=10)
        plt.suptitle(f"Mode={mode.upper()} | 3D oblique MIP comparison", fontsize=13)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(out_dir / f"oblique_panel_{mode}.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig2)

    print(f"Saved qualitative figures to: {out_dir}")


if __name__ == "__main__":
    main()

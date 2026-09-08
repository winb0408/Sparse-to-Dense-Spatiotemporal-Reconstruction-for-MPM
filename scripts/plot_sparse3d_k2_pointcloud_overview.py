#!/usr/bin/env python3
"""
Generate full-volume point-cloud comparison figures for eval_sparse3d_k2.

Per sparse mode (xy / z / xyz), create one 5-column figure:
  Sparse input | 2D recon | 2.5D recon | 3D recon | Ground Truth

Model selection prefers Wiener / mask+Wiener checkpoints:
  2D   : sparse3d_2d_{mode}_K{k}_wiener
  2.5D : sparse3d_2p5d_{mode}_K{k}_mask_wiener
  3D   : sparse3d_3d_{mode}_K{k}_wiener

Fallback to *_mdc25 variants if local CIDC25 checkpoints are unavailable.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import load_tiff_float32, make_mask_3d, normalise_pair
from scripts.eval_sparse3d_unified import (
    _infer_2d_full,
    load_model,
    load_paired_frames,
    wiener_init_2d,
    wiener_init_3d,
)
from scripts.plot_cidc25_sparse3d_pointcloud_overview import (
    draw_pointcloud_axis,
    normalize_with_reference,
)


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


def resolve_run(runs_dir: Path, mode: str, k: int, algo: str) -> Path:
    if algo == "2d":
        candidates = [
            f"sparse3d_2d_{mode}_K{k}_wiener",
            f"sparse3d_2d_{mode}_K{k}_wiener_mdc25",
        ]
    elif algo == "2p5d":
        candidates = [
            f"sparse3d_2p5d_{mode}_K{k}_mask_wiener",
            f"sparse3d_2p5d_{mode}_K{k}_mask_wiener_mdc25",
        ]
    else:
        candidates = [
            f"sparse3d_3d_{mode}_K{k}_wiener",
            f"sparse3d_3d_{mode}_K{k}_wiener_mdc25",
        ]
    for name in candidates:
        run_dir = runs_dir / name
        if (run_dir / "best.pth").exists():
            return run_dir
    raise FileNotFoundError(f"No preferred checkpoint found for {algo}/{mode}/K{k}")


def extract_frame_idx(stem: str) -> int:
    m = re.search(r"_f(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse frame index from {stem}")
    return int(m.group(1))


def load_validation_volume(val_dir: Path, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    pairs = load_paired_frames(val_dir)
    high_frames: List[np.ndarray] = []
    low_frames: List[np.ndarray] = []
    for hp, lp in pairs[:max_frames]:
        high_np = load_tiff_float32(hp, stack_policy="first")
        low_np = load_tiff_float32(lp, stack_policy="first")
        low_n, high_n = normalise_pair(low_np, high_np, mode="minmax")
        low_frames.append(low_n.astype(np.float32))
        high_frames.append(high_n.astype(np.float32))
    return np.stack(low_frames, axis=0), np.stack(high_frames, axis=0)


@torch.no_grad()
def infer_volume_2d(run_dir: Path, val_dir: Path, max_frames: int, device: torch.device) -> np.ndarray:
    model, opts = load_model(run_dir, device)
    pairs = load_paired_frames(val_dir)
    preds: List[np.ndarray] = []
    for hp, lp in pairs[:max_frames]:
        high_np = load_tiff_float32(hp, stack_policy="first")
        low_np = load_tiff_float32(lp, stack_policy="first")
        low_n, high_n = normalise_pair(low_np, high_np, mode=getattr(opts, "norm_mode", "minmax"))
        sparse = torch.from_numpy(low_n.copy()).unsqueeze(0)
        if getattr(opts, "wiener_init", False):
            interp = wiener_init_2d(sparse.squeeze(0), getattr(opts, "psf_sigma_lateral", 1.0)).unsqueeze(0)
        else:
            interp = sparse
        inp = interp * 2.0 - 1.0
        if not getattr(opts, "no_mask_ch", False):
            mask = (torch.from_numpy(low_n) > 1e-6).float().unsqueeze(0)
            inp = torch.cat([inp, mask], dim=0)
        pred = _infer_2d_full(model, inp, device)
        preds.append(pred.astype(np.float32))
    return np.stack(preds, axis=0)


@torch.no_grad()
def infer_volume_2p5d(run_dir: Path, val_dir: Path, max_frames: int, device: torch.device) -> np.ndarray:
    model, opts = load_model(run_dir, device)
    pairs = load_paired_frames(val_dir)
    tw = getattr(opts, "temporal_window", 3)
    radius = tw // 2
    stems = [hp.stem for hp, _ in pairs[:max_frames]]
    high_map = {hp.stem: hp for hp, _ in pairs}
    low_map = {lp.stem: lp for _, lp in pairs}
    idx_to_stem = {extract_frame_idx(stem): stem for stem in stems}
    sorted_idxs = sorted(idx_to_stem.keys())
    preds: List[np.ndarray] = []
    expected_in_ch = int(model.patch_embed.in_channels)
    use_masked_input = expected_in_ch == (2 * tw)
    for c_idx in sorted_idxs:
        if len(preds) >= max_frames:
            break
        if any((c_idx + d) not in idx_to_stem for d in range(-radius, radius + 1)):
            continue
        frames = []
        masks = []
        center_stem = idx_to_stem[c_idx]
        center_high = load_tiff_float32(high_map[center_stem], stack_policy="first")
        center_low = load_tiff_float32(low_map[center_stem], stack_policy="first")
        _, center_gt = normalise_pair(center_low, center_high, mode=getattr(opts, "norm_mode", "minmax"))
        if use_masked_input:
            for d in range(-radius, radius + 1):
                ns = idx_to_stem[c_idx + d]
                lf = load_tiff_float32(low_map[ns], stack_policy="first")
                hf = load_tiff_float32(high_map[ns], stack_policy="first")
                lf_n, _ = normalise_pair(lf, hf, mode=getattr(opts, "norm_mode", "minmax"))
                mask_t = (torch.from_numpy(lf_n) > 1e-6).float().unsqueeze(0)
                sparse = torch.from_numpy(lf_n).unsqueeze(0) * mask_t
                interp = wiener_init_2d(sparse.squeeze(0), getattr(opts, "psf_sigma_lateral", 1.0)).unsqueeze(0)
                frames.append(interp.squeeze(0).numpy())
                masks.append(mask_t.squeeze(0).numpy())
            inp = torch.cat(
                [
                    torch.from_numpy(np.stack(frames, axis=0)) * 2.0 - 1.0,
                    torch.from_numpy(np.stack(masks, axis=0)),
                ],
                dim=0,
            )
        else:
            for d in range(-radius, radius + 1):
                ns = idx_to_stem[c_idx + d]
                lf = load_tiff_float32(low_map[ns], stack_policy="first")
                hf = load_tiff_float32(high_map[ns], stack_policy="first")
                lf_n, _ = normalise_pair(lf, hf, mode=getattr(opts, "norm_mode", "minmax"))
                frames.append(lf_n)
            inp = torch.from_numpy(np.stack(frames, axis=0)) * 2.0 - 1.0
        pred = _infer_2d_full(model, inp, device)
        preds.append(pred.astype(np.float32))
    return np.stack(preds, axis=0)


@torch.no_grad()
def infer_volume_3d(
    run_dir: Path,
    gt_vol: np.ndarray,
    mode: str,
    k: int,
    max_frames: int,
    patch_hw: int,
    device: torch.device,
) -> np.ndarray:
    model, opts = load_model(run_dir, device)
    vol_t = int(getattr(opts, "volume_patch_t", 16))
    kz = int(getattr(opts, "sparsity_factor_z", k))
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    psf_sig_ax = getattr(opts, "psf_sigma_axial", 3.0)
    use_mask = not getattr(opts, "no_mask_ch", False)

    f_total, h, w = gt_vol.shape
    n = min(f_total, max_frames)
    out_vol = np.zeros((n, h, w), dtype=np.float32)
    for t0 in range(0, n, vol_t):
        t1 = min(t0 + vol_t, n)
        chunk = gt_vol[t0:t1].copy()
        if chunk.shape[0] < vol_t:
            pad = np.zeros((vol_t - chunk.shape[0], h, w), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        tt = chunk.shape[0]
        mask = make_mask_3d(tt, h, w, k, "grid", sparse_mode=mode, sparsity_factor_z=kz)[0].numpy()
        for y0 in range(0, h, patch_hw):
            for x0 in range(0, w, patch_hw):
                y1, x1 = min(y0 + patch_hw, h), min(x0 + patch_hw, w)
                ph, pw = y1 - y0, x1 - x0
                tile = chunk[:, y0:y1, x0:x1]
                mask_tile = mask[:, y0:y1, x0:x1]
                if ph < patch_hw or pw < patch_hw:
                    tp = np.zeros((tt, patch_hw, patch_hw), dtype=np.float32)
                    mp = np.zeros((tt, patch_hw, patch_hw), dtype=np.float32)
                    tp[:, :ph, :pw] = tile
                    mp[:, :ph, :pw] = mask_tile
                    tile, mask_tile = tp, mp
                sparse = torch.from_numpy(tile).unsqueeze(0) * torch.from_numpy(mask_tile).unsqueeze(0)
                interp = wiener_init_3d(sparse.squeeze(0), psf_sig, psf_sig_ax).unsqueeze(0)
                chans = [interp * 2.0 - 1.0]
                if use_mask:
                    chans.append(torch.from_numpy(mask_tile).unsqueeze(0))
                inp = torch.cat(chans, dim=0).unsqueeze(0).to(device)
                pred = model(inp)[0, 0].detach().cpu().numpy()
                pred = ((pred + 1.0) / 2.0).clip(0, 1)
                out_vol[t0:t1, y0:y1, x0:x1] = pred[: t1 - t0, :ph, :pw]
    return out_vol


def make_figure(
    mode: str,
    sparse_vol: np.ndarray,
    pred2d: np.ndarray,
    pred25: np.ndarray,
    pred3d: np.ndarray,
    gt_vol: np.ndarray,
    out_path: Path,
    threshold: float,
    max_points: int,
) -> None:
    panels = [
        ("Sparse Input", sparse_vol),
        ("2D + Wiener", pred2d),
        ("2.5D + Mask+Wiener", pred25),
        ("3D + Wiener", pred3d),
        ("Ground Truth", gt_vol),
    ]
    fig = plt.figure(figsize=(23, 5.4), constrained_layout=True)
    for j, (title, vol) in enumerate(panels, start=1):
        vol_norm = normalize_with_reference(vol, gt_vol)
        ax = fig.add_subplot(1, 5, j, projection="3d")
        draw_pointcloud_axis(
            ax,
            vol_norm,
            title,
            threshold=threshold,
            max_points=max_points,
            seed=100 + j,
            elev=26.0,
            azim=42.0,
            cmap="magma_r",
            point_size=0.8,
            point_alpha=0.28,
        )
    fig.suptitle(f"eval_sparse3d_k2 | mode={mode.upper()} | point-cloud overview", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 5-column point-cloud overviews for eval_sparse3d_k2.")
    p.add_argument("--cidc_root", type=Path, default=Path("/data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25"))
    p.add_argument("--sparse_root", type=Path, default=Path("/data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d"))
    p.add_argument("--runs_dir", type=Path, default=Path("/data2/wjb/threePM/runs"))
    p.add_argument("--output_dir", type=Path, default=Path("/data2/wjb/threePM/eval_sparse3d_k2/qualitative"))
    p.add_argument("--gt_stem", type=str, default="F0")
    p.add_argument("--split", type=str, default="Validation")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--modes", nargs="+", default=["xy", "z", "xyz"])
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--patch_hw", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--max_points", type=int, default=120000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_full = read_volume(args.cidc_root / args.split / f"{args.gt_stem}.tif")[: args.max_frames]

    for mode in args.modes:
        val_dir = args.sparse_root / f"K{args.k}_{mode}" / args.split
        sparse_vol, gt_pair_vol = load_validation_volume(val_dir, args.max_frames)
        run2d = resolve_run(args.runs_dir, mode, args.k, "2d")
        run25 = resolve_run(args.runs_dir, mode, args.k, "2p5d")
        run3d = resolve_run(args.runs_dir, mode, args.k, "3d")
        print(f"[{mode}] 2D   <- {run2d.name}")
        print(f"[{mode}] 2.5D <- {run25.name}")
        print(f"[{mode}] 3D   <- {run3d.name}")
        pred2d = infer_volume_2d(run2d, val_dir, args.max_frames, device)
        pred25 = infer_volume_2p5d(run25, val_dir, args.max_frames, device)
        n = min(len(gt_pair_vol), len(pred2d), len(pred25), len(gt_full))
        pred3d = infer_volume_3d(run3d, gt_full[:n], mode, args.k, n, args.patch_hw, device)
        out_path = args.output_dir / f"pointcloud_overview_{mode}_K{args.k}.png"
        make_figure(
            mode,
            sparse_vol[:n],
            pred2d[:n],
            pred25[:n],
            pred3d[:n],
            gt_pair_vol[:n],
            out_path,
            threshold=args.threshold,
            max_points=args.max_points,
        )
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()

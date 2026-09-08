#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tifffile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import wiener_init_2d, wiener_init_3d
from models import build_model


def parse_case(case_name: str) -> Tuple[int, str]:
    m = re.match(r".*-(\d+)x(XY|XYZ|X|Y|Z)$", case_name)
    if not m:
        raise ValueError(f"Unrecognized case name: {case_name}")
    k = int(m.group(1))
    mode = m.group(2)
    return k, mode


def map_mode_to_model(mode: str) -> str:
    if mode in ("X", "Y", "XY"):
        return "xy"
    if mode == "Z":
        return "z"
    if mode == "XYZ":
        return "xyz"
    raise ValueError(f"Unsupported sparse mode: {mode}")


def load_model(run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    opts = argparse.Namespace(**ckpt.get("opts", {}))
    model = build_model(opts)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    return model, opts


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vmin = float(vol.min())
    vmax = float(vol.max())
    return ((vol - vmin) / (vmax - vmin + 1e-8)).clip(0.0, 1.0).astype(np.float32)


@torch.no_grad()
def infer_2d_frame(model, frame_low: np.ndarray, frame_mask: np.ndarray, device: torch.device, psf_sigma: float) -> np.ndarray:
    low_t = torch.from_numpy(frame_low).unsqueeze(0)   # [1,H,W]
    mask_t = torch.from_numpy(frame_mask).unsqueeze(0)  # [1,H,W]
    sparse = low_t * mask_t
    interp = wiener_init_2d(sparse.squeeze(0), psf_sigma).unsqueeze(0)
    interp_m = interp * 2.0 - 1.0
    inp = torch.cat([interp_m, mask_t], dim=0).unsqueeze(0)  # [1,2,H,W]

    _, _, h, w = inp.shape
    for align in (32, 64, 96, 128, 160, 192, 224, 256):
        hp = ((h + align - 1) // align) * align
        wp = ((w + align - 1) // align) * align
        x = inp
        if hp != h or wp != w:
            x = F.pad(inp, (0, wp - w, 0, hp - h), mode="replicate")
        try:
            out = model(x.to(device))
            pred = ((out[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            return pred[:h, :w]
        except RuntimeError:
            continue
    raise RuntimeError("2D inference failed after adaptive padding trials.")


@torch.no_grad()
def infer_2d_volume(model, low_vol: np.ndarray, mask_vol: np.ndarray, device: torch.device, psf_sigma: float) -> np.ndarray:
    t = low_vol.shape[0]
    pred = np.zeros_like(low_vol, dtype=np.float32)
    for i in range(t):
        pred[i] = infer_2d_frame(model, low_vol[i], mask_vol[i], device, psf_sigma)
    return pred


@torch.no_grad()
def infer_2p5d_volume(
    model,
    low_vol: np.ndarray,
    mask_vol: np.ndarray,
    device: torch.device,
    temporal_window: int,
    psf_sigma: float,
) -> np.ndarray:
    t, h, w = low_vol.shape
    radius = temporal_window // 2
    pred = np.zeros_like(low_vol, dtype=np.float32)
    expected_in_ch = int(model.patch_embed.in_channels)
    if expected_in_ch != temporal_window * 2:
        raise ValueError(f"2.5D model expects in_ch={expected_in_ch}, not mask-aware 2T={temporal_window*2}.")

    for c in range(t):
        idxs = [min(max(c + d, 0), t - 1) for d in range(-radius, radius + 1)]
        interps = []
        masks = []
        for ii in idxs:
            low_t = torch.from_numpy(low_vol[ii]).unsqueeze(0)
            mask_t = torch.from_numpy(mask_vol[ii]).unsqueeze(0)
            sparse = low_t * mask_t
            interp = wiener_init_2d(sparse.squeeze(0), psf_sigma).unsqueeze(0)
            interps.append(interp)
            masks.append(mask_t)
        interp_all = torch.cat(interps, dim=0) * 2.0 - 1.0
        mask_all = torch.cat(masks, dim=0)
        inp = torch.cat([interp_all, mask_all], dim=0)  # [2T,H,W]
        # Direct forward for 2.5D input with adaptive padding.
        x = inp.unsqueeze(0)
        done = False
        for align in (32, 64, 96, 128, 160, 192, 224, 256):
            hp = ((h + align - 1) // align) * align
            wp = ((w + align - 1) // align) * align
            xx = x
            if hp != h or wp != w:
                xx = F.pad(x, (0, wp - w, 0, hp - h), mode="replicate")
            try:
                out = model(xx.to(device))
                pred[c] = ((out[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)[:h, :w]
                done = True
                break
            except RuntimeError:
                continue
        if not done:
            raise RuntimeError("2.5D inference failed after adaptive padding trials.")
    return pred


@torch.no_grad()
def infer_3d_volume(
    model,
    low_vol: np.ndarray,
    mask_vol: np.ndarray,
    device: torch.device,
    volume_patch_t: int,
    patch_hw: int,
    psf_sigma_xy: float,
    psf_sigma_z: float,
) -> np.ndarray:
    t, h, w = low_vol.shape
    pred = np.zeros_like(low_vol, dtype=np.float32)
    for t0 in range(0, t, volume_patch_t):
        t1 = min(t0 + volume_patch_t, t)
        ct = t1 - t0
        low_chunk = low_vol[t0:t1]
        mask_chunk = mask_vol[t0:t1]
        if ct < volume_patch_t:
            pad = np.zeros((volume_patch_t - ct, h, w), dtype=np.float32)
            low_chunk = np.concatenate([low_chunk, pad], axis=0)
            mask_chunk = np.concatenate([mask_chunk, pad], axis=0)
        for y0 in range(0, h, patch_hw):
            for x0 in range(0, w, patch_hw):
                y1 = min(y0 + patch_hw, h)
                x1 = min(x0 + patch_hw, w)
                ph = y1 - y0
                pw = x1 - x0
                low_tile = low_chunk[:, y0:y1, x0:x1]
                mask_tile = mask_chunk[:, y0:y1, x0:x1]
                if ph < patch_hw or pw < patch_hw:
                    lp = np.zeros((volume_patch_t, patch_hw, patch_hw), dtype=np.float32)
                    mp = np.zeros((volume_patch_t, patch_hw, patch_hw), dtype=np.float32)
                    lp[:, :ph, :pw] = low_tile
                    mp[:, :ph, :pw] = mask_tile
                    low_tile, mask_tile = lp, mp
                sparse = torch.from_numpy(low_tile).unsqueeze(0) * torch.from_numpy(mask_tile).unsqueeze(0)
                interp = wiener_init_3d(sparse.squeeze(0), psf_sigma_xy, psf_sigma_z).unsqueeze(0)
                inp = torch.cat([interp * 2.0 - 1.0, torch.from_numpy(mask_tile).unsqueeze(0)], dim=0).unsqueeze(0).to(device)
                out = model(inp)
                out_np = ((out[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
                pred[t0:t1, y0:y1, x0:x1] = out_np[:ct, :ph, :pw]
    return pred


def save_tif(path: Path, vol: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), (vol.clip(0, 1) * 65535).astype(np.uint16), imagej=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Infer new sparse sample using MDC25 pretrained 2D/2.5D/3D Wiener+mask-aware models.")
    p.add_argument("--sample_root", type=Path, default=Path("/data2/wjb/threePM/ThreePM_openDatasets/200um-f10-z5_sparse"))
    p.add_argument("--runs_dir", type=Path, default=Path("/data2/wjb/threePM/runs"))
    p.add_argument("--output_root", type=Path, default=Path("/data2/wjb/threePM/eval_newsample_mdc25_preds"))
    p.add_argument("--patch_hw", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_path = args.sample_root / "200um-f10-z5-dense" / "200um-f10-z5-dense.tif"
    if not dense_path.exists():
        raise FileNotFoundError(f"Missing dense reference: {dense_path}")
    dense_vol = normalize_volume(tifffile.imread(str(dense_path)).astype(np.float32))

    cases = sorted([d for d in args.sample_root.iterdir() if d.is_dir() and d.name.endswith(tuple(["X", "Y", "XY", "Z", "XYZ"])) and "dense" not in d.name])
    if not cases:
        raise ValueError(f"No sparse case directories found under: {args.sample_root}")

    for case_dir in cases:
        k, raw_mode = parse_case(case_dir.name)
        model_mode = map_mode_to_model(raw_mode)
        sparse_path = case_dir / f"{case_dir.name}.tif"
        mask_path = case_dir / f"{case_dir.name}_mask.tif"
        if not sparse_path.exists() or not mask_path.exists():
            print(f"[Skip] missing tif or mask in {case_dir.name}")
            continue

        low_vol = normalize_volume(tifffile.imread(str(sparse_path)).astype(np.float32))
        mask_vol = (tifffile.imread(str(mask_path)) > 0).astype(np.float32)
        n = min(low_vol.shape[0], mask_vol.shape[0], dense_vol.shape[0])
        low_vol = low_vol[:n]
        mask_vol = mask_vol[:n]
        gt_vol = dense_vol[:n]

        run_2d = args.runs_dir / f"sparse3d_2d_{model_mode}_K2_wiener_mdc25"
        run_2p5d = args.runs_dir / f"sparse3d_2p5d_{model_mode}_K2_mask_wiener_mdc25"
        run_3d = args.runs_dir / f"sparse3d_3d_{model_mode}_K2_wiener_mdc25"

        print(f"\n=== Case {case_dir.name} | mapped mode={model_mode} | K_in_name={k} ===")
        print(f"Using runs:\n  2D   {run_2d.name}\n  2.5D {run_2p5d.name}\n  3D   {run_3d.name}")

        model2d, opts2d = load_model(run_2d, device)
        pred2d = infer_2d_volume(
            model2d, low_vol, mask_vol, device,
            psf_sigma=float(getattr(opts2d, "psf_sigma_lateral", 1.0)),
        )
        save_tif(args.output_root / case_dir.name / "pred_2d_wiener_maskaware.tif", pred2d)
        del model2d
        torch.cuda.empty_cache()

        model25, opts25 = load_model(run_2p5d, device)
        pred25 = infer_2p5d_volume(
            model25, low_vol, mask_vol, device,
            temporal_window=int(getattr(opts25, "temporal_window", 3)),
            psf_sigma=float(getattr(opts25, "psf_sigma_lateral", 1.0)),
        )
        save_tif(args.output_root / case_dir.name / "pred_2p5d_wiener_maskaware.tif", pred25)
        del model25
        torch.cuda.empty_cache()

        model3d, opts3d = load_model(run_3d, device)
        pred3d = infer_3d_volume(
            model3d, low_vol, mask_vol, device,
            volume_patch_t=int(getattr(opts3d, "volume_patch_t", 16)),
            patch_hw=args.patch_hw,
            psf_sigma_xy=float(getattr(opts3d, "psf_sigma_lateral", 1.0)),
            psf_sigma_z=float(getattr(opts3d, "psf_sigma_axial", 3.0)),
        )
        save_tif(args.output_root / case_dir.name / "pred_3d_wiener_maskaware.tif", pred3d)
        del model3d
        torch.cuda.empty_cache()

        save_tif(args.output_root / case_dir.name / "input_sparse.tif", low_vol)
        save_tif(args.output_root / case_dir.name / "gt_dense.tif", gt_vol)
        save_tif(args.output_root / case_dir.name / "mask.tif", mask_vol)
        print(f"[Done] Saved predictions to: {args.output_root / case_dir.name}")

    print(f"\nAll done. Outputs at: {args.output_root}")


if __name__ == "__main__":
    main()


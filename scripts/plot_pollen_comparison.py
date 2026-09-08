#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_torch
import tifffile

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import warm_start_2d, warm_start_3d, wiener_init_2d, wiener_init_3d


def load_volume(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    while arr.ndim > 3:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def minmax_pair(low: np.ndarray, high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vmin = float(min(low.min(), high.min()))
    vmax = float(max(low.max(), high.max()))
    denom = vmax - vmin + 1e-8
    low_n = ((low - vmin) / denom).clip(0, 1).astype(np.float32)
    high_n = ((high - vmin) / denom).clip(0, 1).astype(np.float32)
    return low_n, high_n


def parse_run_config(run_name: str) -> Dict[str, str]:
    m = re.match(r"sparse3d_(2d|2p5d|3d)_(xy|z|xyz)_K(\d+)(.*)", run_name)
    if not m:
        return {"algorithm": "?", "sparse_mode": "?", "K": "?", "extras": ""}
    extras = m.group(4).strip("_") if m.group(4) else ""
    return {
        "algorithm": m.group(1),
        "sparse_mode": m.group(2),
        "K": m.group(3),
        "extras": extras,
    }


def load_model(run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "best.pth"
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    opts = argparse.Namespace(**ckpt.get("opts", {}))
    from models import build_model

    model = build_model(opts)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), opts


@torch.no_grad()
def infer_2d_full(model, inp: torch.Tensor, device: torch.device) -> np.ndarray:
    _, h, w = inp.shape
    last_err = None
    for align in (32, 64, 96, 128, 160, 192, 224, 256):
        hp = ((h + align - 1) // align) * align
        wp = ((w + align - 1) // align) * align
        x = inp.unsqueeze(0)
        if hp != h or wp != w:
            x = F_torch.pad(x, (0, wp - w, 0, hp - h), mode="reflect")
        try:
            out = model(x.to(device))
            out_np = ((out[0, 0].cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            return out_np[:h, :w]
        except RuntimeError as e:
            last_err = e
    raise RuntimeError("2D/2.5D full-image inference failed.") from last_err


@torch.no_grad()
def predict_frame(
    model,
    opts,
    algorithm: str,
    low_vol: np.ndarray,
    high_vol: np.ndarray,
    mask_vol: np.ndarray,
    frame_idx: int,
    patch_hw: int,
    device: torch.device,
) -> np.ndarray:
    if algorithm == "2d":
        use_mask = not getattr(opts, "no_mask_ch", False)
        use_wiener = getattr(opts, "wiener_init", False)
        k = int(getattr(opts, "sparsity_factor", 4))
        psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
        low_n, _ = minmax_pair(low_vol[frame_idx], high_vol[frame_idx])
        low_t = torch.from_numpy(low_n).unsqueeze(0)
        mask_t = torch.from_numpy(mask_vol[frame_idx].astype(np.float32)).unsqueeze(0)
        sparse = low_t * mask_t
        interp = wiener_init_2d(sparse.squeeze(0), psf_sig).unsqueeze(0) if use_wiener else warm_start_2d(sparse, k)
        inp = interp * 2.0 - 1.0
        if use_mask:
            inp = torch.cat([inp, mask_t], dim=0)
        return infer_2d_full(model, inp, device)

    if algorithm == "2p5d":
        tw = int(getattr(opts, "temporal_window", 3))
        radius = tw // 2
        use_wiener = getattr(opts, "wiener_init", False)
        k = int(getattr(opts, "sparsity_factor", 4))
        psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
        expected_in = int(getattr(getattr(model, "patch_embed", None), "in_channels", tw))
        legacy_input = expected_in == tw
        idxs = [min(max(frame_idx + d, 0), low_vol.shape[0] - 1) for d in range(-radius, radius + 1)]
        low_chunk = np.stack([low_vol[i] for i in idxs], axis=0)
        high_chunk = np.stack([high_vol[i] for i in idxs], axis=0)
        mask_chunk = np.stack([mask_vol[i] for i in idxs], axis=0).astype(np.float32)

        if legacy_input:
            vmin = float(min(low_chunk[radius].min(), high_chunk[radius].min()))
            vmax = float(max(low_chunk[radius].max(), high_chunk[radius].max()))
            denom = vmax - vmin + 1e-8
            low_stack_n = ((low_chunk - vmin) / denom).clip(0, 1).astype(np.float32)
            inp = torch.from_numpy(low_stack_n) * 2.0 - 1.0
            return infer_2d_full(model, inp, device)

        vmin = float(high_chunk.min())
        vmax = float(high_chunk.max())
        denom = vmax - vmin + 1e-8
        interp_list = []
        mask_list = []
        for t in range(tw):
            low_t = ((low_chunk[t] - vmin) / denom).clip(0, 1).astype(np.float32)
            low_tt = torch.from_numpy(low_t).unsqueeze(0)
            mask_tt = torch.from_numpy(mask_chunk[t]).unsqueeze(0)
            sparse = low_tt * mask_tt
            interp = wiener_init_2d(sparse.squeeze(0), psf_sig).unsqueeze(0) if use_wiener else warm_start_2d(sparse, k)
            interp_list.append(interp)
            mask_list.append(mask_tt)
        interp_all = torch.cat(interp_list, dim=0) * 2.0 - 1.0
        mask_all = torch.cat(mask_list, dim=0)
        inp = torch.cat([interp_all, mask_all], dim=0)
        return infer_2d_full(model, inp, device)

    # 3d
    vol_t = int(getattr(opts, "volume_patch_t", 16))
    use_mask = not getattr(opts, "no_mask_ch", False)
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    psf_sig_ax = getattr(opts, "psf_sigma_axial", 3.0)
    k = int(getattr(opts, "sparsity_factor", 4))
    h, w = high_vol.shape[1:]
    t0 = max(0, frame_idx - vol_t // 2)
    t1 = min(t0 + vol_t, low_vol.shape[0])
    t0 = max(0, t1 - vol_t)
    low_chunk = low_vol[t0:t1].astype(np.float32)
    high_chunk = high_vol[t0:t1].astype(np.float32)
    mask_chunk = mask_vol[t0:t1].astype(np.float32)
    act_t = low_chunk.shape[0]

    vmin = float(high_chunk.min())
    vmax = float(high_chunk.max())
    denom = vmax - vmin + 1e-8
    low_n = ((low_chunk - vmin) / denom).clip(0, 1).astype(np.float32)
    if act_t < vol_t:
        pad_t = vol_t - act_t
        low_n = np.concatenate([low_n, np.zeros((pad_t, h, w), dtype=np.float32)], axis=0)
        mask_chunk = np.concatenate([mask_chunk, np.zeros((pad_t, h, w), dtype=np.float32)], axis=0)

    pred_all = np.zeros((vol_t, h, w), dtype=np.float32)
    for y0 in range(0, h, patch_hw):
        for x0 in range(0, w, patch_hw):
            y1, x1 = min(y0 + patch_hw, h), min(x0 + patch_hw, w)
            ph, pw = y1 - y0, x1 - x0
            low_tile = low_n[:, y0:y1, x0:x1]
            mask_tile = mask_chunk[:, y0:y1, x0:x1]
            if ph < patch_hw or pw < patch_hw:
                lp = np.zeros((vol_t, patch_hw, patch_hw), dtype=np.float32)
                mp = np.zeros((vol_t, patch_hw, patch_hw), dtype=np.float32)
                lp[:, :ph, :pw] = low_tile
                mp[:, :ph, :pw] = mask_tile
                low_tile, mask_tile = lp, mp
            sparse = torch.from_numpy(low_tile).unsqueeze(0) * torch.from_numpy(mask_tile).unsqueeze(0)
            interp = wiener_init_3d(sparse.squeeze(0), psf_sig, psf_sig_ax).unsqueeze(0) if use_wiener else warm_start_3d(sparse, factor=k, factor_z=1)
            inp = interp * 2.0 - 1.0
            if use_mask:
                inp = torch.cat([inp, torch.from_numpy(mask_tile).unsqueeze(0)], dim=0)
            out = model(inp.unsqueeze(0).to(device))
            out_np = ((out[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            pred_all[:, y0:y1, x0:x1] = out_np[:, :ph, :pw]
    pick = min(max(frame_idx - t0, 0), act_t - 1)
    return pred_all[pick]


def choose_best_runs(summary_csv: Path) -> Dict[str, Dict[str, Dict]]:
    best: Dict[str, Dict[str, Dict]] = {}
    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            case = r["case"]
            algo = r["algorithm"]
            p = float(r["psnr_mean"])
            best.setdefault(case, {})
            if algo not in best[case] or p > best[case][algo]["psnr_mean"]:
                best[case][algo] = {"run_name": r["run_name"], "psnr_mean": p}
    return best


def main():
    ap = argparse.ArgumentParser(description="Generate pollen qualitative comparison panels.")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--mask_root", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--summary_csv", type=str, default="eval_pollen_sparse/summary_pollen.csv")
    ap.add_argument("--output_dir", type=str, default="eval_pollen_sparse/qualitative")
    ap.add_argument("--patch_hw", type=int, default=256)
    ap.add_argument("--frame_idx", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_path = next(data_root.glob("*-dense-*.tif"))
    gt_vol = load_volume(dense_path)
    best = choose_best_runs(Path(args.summary_csv))

    sparse_paths = sorted([p for p in data_root.glob("*.tif") if "dense" not in p.name])
    for sp in sparse_paths:
        case = sp.stem
        if case not in best or not all(k in best[case] for k in ("2d", "2p5d", "3d")):
            continue
        mask_path = mask_root / sp.name.replace(".tif", "_mask.tif")
        if not mask_path.exists():
            continue
        low_vol = load_volume(sp)
        mask_vol = (load_volume(mask_path) > 0).astype(np.float32)
        n = min(low_vol.shape[0], gt_vol.shape[0], mask_vol.shape[0])
        idx = min(max(args.frame_idx, 0), n - 1)

        low_n, gt_n = minmax_pair(low_vol[idx], gt_vol[idx])
        sparse_n = (low_n * mask_vol[idx]).clip(0, 1)

        preds = {}
        for algo in ("2d", "2p5d", "3d"):
            run_name = best[case][algo]["run_name"]
            model, opts = load_model(runs_dir / run_name, device)
            pred = predict_frame(model, opts, algo, low_vol[:n], gt_vol[:n], mask_vol[:n], idx, args.patch_hw, device)
            preds[algo] = (pred, run_name, best[case][algo]["psnr_mean"])
            del model
            torch.cuda.empty_cache()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))
        imgs = [gt_n, sparse_n, preds["2d"][0], preds["2p5d"][0], preds["3d"][0]]
        titles = [
            "GT (dense)",
            "Sparse input",
            f"Best 2D\n{preds['2d'][1]}\nPSNR={preds['2d'][2]:.2f}",
            f"Best 2.5D\n{preds['2p5d'][1]}\nPSNR={preds['2p5d'][2]:.2f}",
            f"Best 3D\n{preds['3d'][1]}\nPSNR={preds['3d'][2]:.2f}",
        ]
        for ax, im, tt in zip(axes, imgs, titles):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_title(tt, fontsize=9)
            ax.axis("off")
        fig.suptitle(case, fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"{case}_best5.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

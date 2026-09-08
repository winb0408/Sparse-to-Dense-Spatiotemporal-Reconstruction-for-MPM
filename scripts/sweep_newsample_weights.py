#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from scipy.ndimage import uniform_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.infer_new_sparse_sample_mdc25 import (
    parse_case,
    map_mode_to_model,
    load_model,
    normalize_volume,
    infer_2d_frame,
    infer_3d_volume,
)
from dataset import wiener_init_2d


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim_2d(pred: np.ndarray, gt: np.ndarray, win: int = 7) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_p = uniform_filter(pred, size=win)
    mu_g = uniform_filter(gt, size=win)
    sig_pp = uniform_filter(pred * pred, size=win) - mu_p ** 2
    sig_gg = uniform_filter(gt * gt, size=win) - mu_g ** 2
    sig_pg = uniform_filter(pred * gt, size=win) - mu_p * mu_g
    num = (2 * mu_p * mu_g + c1) * (2 * sig_pg + c2)
    den = (mu_p ** 2 + mu_g ** 2 + c1) * (sig_pp + sig_gg + c2)
    return float(np.mean(num / (den + 1e-12)))


def parse_run_name(run_name: str) -> Tuple[str, str, int]:
    m = re.match(r"sparse3d_(2d|2p5d|3d)_(xy|z|xyz)_K(\d+).*", run_name)
    if not m:
        return "?", "?", -1
    return m.group(1), m.group(2), int(m.group(3))


@torch.no_grad()
def infer_2p5d_frame(
    model,
    low_vol: np.ndarray,
    mask_vol: np.ndarray,
    frame_idx: int,
    temporal_window: int,
    psf_sigma: float,
    device: torch.device,
) -> np.ndarray:
    t, h, w = low_vol.shape
    radius = temporal_window // 2
    expected_in_ch = int(model.patch_embed.in_channels)
    if expected_in_ch != temporal_window * 2:
        raise ValueError(f"2.5D in_ch mismatch: expected {temporal_window*2}, got {expected_in_ch}")
    idxs = [min(max(frame_idx + d, 0), t - 1) for d in range(-radius, radius + 1)]
    interps = []
    masks = []
    for ii in idxs:
        low_t = torch.from_numpy(low_vol[ii]).unsqueeze(0)
        mask_t = torch.from_numpy(mask_vol[ii]).unsqueeze(0)
        sparse = low_t * mask_t
        interp = wiener_init_2d(sparse.squeeze(0), psf_sigma).unsqueeze(0)
        interps.append(interp)
        masks.append(mask_t)
    inp = torch.cat([torch.cat(interps, dim=0) * 2.0 - 1.0, torch.cat(masks, dim=0)], dim=0).unsqueeze(0)
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
    raise RuntimeError("2.5D inference failed after adaptive padding trials.")


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep sparse3d checkpoints on new 200um sample and compare weights.")
    p.add_argument("--sample_root", type=Path, default=Path("/data2/wjb/threePM/ThreePM_openDatasets/200um-f10-z5_sparse"))
    p.add_argument("--runs_dir", type=Path, default=Path("/data2/wjb/threePM/runs"))
    p.add_argument("--output_dir", type=Path, default=Path("/data2/wjb/threePM/eval_newsample_mdc25_preds/weight_sweep"))
    p.add_argument("--run_prefix", type=str, default="sparse3d_")
    p.add_argument("--strict_match", action="store_true", help="Only evaluate runs with same sparse mode and same K as case.")
    p.add_argument("--eval_frames", type=int, default=3, help="How many frames per case to evaluate (uniformly sampled).")
    p.add_argument("--patch_hw", type=int, default=256)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dense_path = args.sample_root / "200um-f10-z5-dense" / "200um-f10-z5-dense.tif"
    dense_vol = normalize_volume(tifffile.imread(str(dense_path)).astype(np.float32))

    cases = sorted([d for d in args.sample_root.iterdir() if d.is_dir() and "dense" not in d.name and d.name.endswith(("X", "Y", "XY", "Z", "XYZ"))])
    runs = sorted([d for d in args.runs_dir.iterdir() if d.is_dir() and d.name.startswith(args.run_prefix) and (d / "best.pth").exists()])

    rows: List[Dict] = []
    for run_dir in runs:
        run_name = run_dir.name
        algo, train_mode, train_k = parse_run_name(run_name)
        if algo not in ("2d", "2p5d", "3d"):
            continue
        try:
            model, opts = load_model(run_dir, device)
        except Exception as e:
            print(f"[Skip run] {run_name}: {e}")
            continue

        for case_dir in cases:
            k_case, raw_mode = parse_case(case_dir.name)
            case_mode = map_mode_to_model(raw_mode)
            if args.strict_match and not (train_mode == case_mode and train_k == k_case):
                continue

            sparse_path = case_dir / f"{case_dir.name}.tif"
            mask_path = case_dir / f"{case_dir.name}_mask.tif"
            low_vol = normalize_volume(tifffile.imread(str(sparse_path)).astype(np.float32))
            mask_vol = (tifffile.imread(str(mask_path)) > 0).astype(np.float32)
            n = min(low_vol.shape[0], mask_vol.shape[0], dense_vol.shape[0])
            low_vol = low_vol[:n]
            mask_vol = mask_vol[:n]
            gt_vol = dense_vol[:n]

            if n <= 0:
                continue
            if args.eval_frames <= 1:
                frame_ids = [n // 2]
            else:
                frame_ids = sorted(set(np.linspace(0, n - 1, num=args.eval_frames, dtype=int).tolist()))

            psnrs: List[float] = []
            ssims: List[float] = []
            ok = True
            for fid in frame_ids:
                try:
                    if algo == "2d":
                        pred = infer_2d_frame(
                            model,
                            low_vol[fid],
                            mask_vol[fid],
                            device,
                            psf_sigma=float(getattr(opts, "psf_sigma_lateral", 1.0)),
                        )
                    elif algo == "2p5d":
                        pred = infer_2p5d_frame(
                            model,
                            low_vol,
                            mask_vol,
                            frame_idx=fid,
                            temporal_window=int(getattr(opts, "temporal_window", 3)),
                            psf_sigma=float(getattr(opts, "psf_sigma_lateral", 1.0)),
                            device=device,
                        )
                    else:
                        pred_vol = infer_3d_volume(
                            model,
                            low_vol,
                            mask_vol,
                            device=device,
                            volume_patch_t=int(getattr(opts, "volume_patch_t", 16)),
                            patch_hw=args.patch_hw,
                            psf_sigma_xy=float(getattr(opts, "psf_sigma_lateral", 1.0)),
                            psf_sigma_z=float(getattr(opts, "psf_sigma_axial", 3.0)),
                        )
                        pred = pred_vol[fid]
                    gt = gt_vol[fid]
                    psnrs.append(psnr(pred, gt))
                    ssims.append(ssim_2d(pred, gt))
                except Exception:
                    ok = False
                    break

            if not ok or not psnrs:
                continue
            rows.append({
                "case": case_dir.name,
                "case_mode": case_mode,
                "case_k": k_case,
                "run_name": run_name,
                "algorithm": algo,
                "train_mode": train_mode,
                "train_k": train_k,
                "k_match": str(train_k == k_case),
                "mode_match": str(train_mode == case_mode),
                "n_eval_frames": len(frame_ids),
                "psnr_mean": float(np.mean(psnrs)),
                "psnr_std": float(np.std(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "ssim_std": float(np.std(ssims)),
            })
            print(f"[{case_dir.name}] {run_name} => PSNR={np.mean(psnrs):.2f}, SSIM={np.mean(ssims):.4f}")

        del model
        torch.cuda.empty_cache()

    csv_path = args.output_dir / "newsample_weight_sweep.csv"
    cols = [
        "case", "case_mode", "case_k", "run_name", "algorithm",
        "train_mode", "train_k", "k_match", "mode_match", "n_eval_frames",
        "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # best per case overall and strict(match k+mode)
    best_overall: Dict[str, Dict] = {}
    best_strict: Dict[str, Dict] = {}
    for r in rows:
        c = r["case"]
        if c not in best_overall or r["psnr_mean"] > best_overall[c]["psnr_mean"]:
            best_overall[c] = r
        if r["k_match"] == "True" and r["mode_match"] == "True":
            if c not in best_strict or r["psnr_mean"] > best_strict[c]["psnr_mean"]:
                best_strict[c] = r

    md_path = args.output_dir / "best_per_case.md"
    with open(md_path, "w") as f:
        f.write("# New Sample Weight Sweep Best Runs\n\n")
        f.write(f"- Source CSV: `{csv_path}`\n")
        f.write(f"- Eval frames per case: `{args.eval_frames}`\n\n")
        f.write("## Best Overall (by PSNR)\n\n")
        f.write("| Case | Best Run | Algo | TrainMode/K | PSNR | SSIM |\n")
        f.write("|---|---|---|---|---:|---:|\n")
        for c in sorted(best_overall):
            r = best_overall[c]
            f.write(f"| {c} | {r['run_name']} | {r['algorithm']} | {r['train_mode']}/K{r['train_k']} | {r['psnr_mean']:.3f} | {r['ssim_mean']:.4f} |\n")
        f.write("\n## Best Strict Match (same mode and same K)\n\n")
        f.write("| Case | Best Run | Algo | TrainMode/K | PSNR | SSIM |\n")
        f.write("|---|---|---|---|---:|---:|\n")
        for c in sorted(best_overall):
            if c in best_strict:
                r = best_strict[c]
                f.write(f"| {c} | {r['run_name']} | {r['algorithm']} | {r['train_mode']}/K{r['train_k']} | {r['psnr_mean']:.3f} | {r['ssim_mean']:.4f} |\n")
            else:
                f.write(f"| {c} | - | - | - | - | - |\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()


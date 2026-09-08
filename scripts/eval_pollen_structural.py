#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import tifffile
from scipy.ndimage import binary_dilation, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_pollen_comparison import (  # noqa: E402
    load_volume,
    minmax_pair,
    parse_run_config,
    predict_frame,
)
from models import build_model  # noqa: E402


def psnr_from_mse(mse: float) -> float:
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim_map(pred: np.ndarray, gt: np.ndarray, win: int = 7) -> np.ndarray:
    c1, c2 = 0.01**2, 0.03**2
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mu_p = uniform_filter(pred, size=win)
    mu_g = uniform_filter(gt, size=win)
    sig_pp = uniform_filter(pred * pred, size=win) - mu_p**2
    sig_gg = uniform_filter(gt * gt, size=win) - mu_g**2
    sig_pg = uniform_filter(pred * gt, size=win) - mu_p * mu_g
    num = (2 * mu_p * mu_g + c1) * (2 * sig_pg + c2)
    den = (mu_p**2 + mu_g**2 + c1) * (sig_pp + sig_gg + c2)
    return num / (den + 1e-8)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        return float(np.mean(values))
    return float(np.mean(values[m]))


def make_foreground_mask(
    gt: np.ndarray,
    percentile: float = 70.0,
    dilation_iter: int = 3,
    min_fraction: float = 0.02,
) -> np.ndarray:
    gt = gt.astype(np.float32)
    finite = gt[np.isfinite(gt)]
    if finite.size == 0:
        return np.ones_like(gt, dtype=bool)
    thr = float(np.percentile(finite, percentile))
    mask = gt > thr
    if mask.mean() < min_fraction:
        thr = float(np.percentile(finite, 100.0 * (1.0 - min_fraction)))
        mask = gt > thr
    if dilation_iter > 0:
        mask = binary_dilation(mask, iterations=dilation_iter)
    return mask.astype(bool)


def downsample_mean(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    h2, w2 = h // 2, w // 2
    if h2 < 8 or w2 < 8:
        return x
    x = x[: h2 * 2, : w2 * 2]
    return x.reshape(h2, 2, w2, 2).mean(axis=(1, 3))


def downsample_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    h2, w2 = h // 2, w // 2
    if h2 < 8 or w2 < 8:
        return mask
    m = mask[: h2 * 2, : w2 * 2].astype(np.float32)
    return m.reshape(h2, 2, w2, 2).mean(axis=(1, 3)) > 0.25


def ms_ssim_fg(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    weights = [0.5, 0.3, 0.2]
    scores: List[float] = []
    used_weights: List[float] = []
    p, g, m = pred.copy(), gt.copy(), mask.copy()
    for weight in weights:
        scores.append(masked_mean(ssim_map(p, g), m))
        used_weights.append(weight)
        if min(p.shape) < 32:
            break
        p, g, m = downsample_mean(p), downsample_mean(g), downsample_mask(m)
    w = np.asarray(used_weights, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    return float(np.sum(s * w) / np.sum(w))


def ncc_fg(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        m = np.ones_like(mask, dtype=bool)
    p = pred[m].astype(np.float64)
    g = gt[m].astype(np.float64)
    p = p - p.mean()
    g = g - g.mean()
    denom = float(np.sqrt(np.sum(p * p) * np.sum(g * g)) + 1e-12)
    return float(np.sum(p * g) / denom)


def foreground_linear_match(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not np.any(m):
        return pred.copy()
    p = pred[m].astype(np.float64)
    g = gt[m].astype(np.float64)
    var = float(np.var(p))
    if var < 1e-12:
        return pred.copy()
    scale = float(np.cov(p, g, bias=True)[0, 1] / var)
    bias = float(g.mean() - scale * p.mean())
    return (pred * scale + bias).clip(0, 1).astype(np.float32)


def structural_metrics(pred: np.ndarray, gt: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.float32).clip(0, 1)
    gt = gt.astype(np.float32).clip(0, 1)
    pred_cal = foreground_linear_match(pred, gt, fg_mask)

    mse_full = float(np.mean((pred - gt) ** 2))
    mse_fg = float(np.mean((pred[fg_mask] - gt[fg_mask]) ** 2)) if np.any(fg_mask) else mse_full
    mae_fg = float(np.mean(np.abs(pred[fg_mask] - gt[fg_mask]))) if np.any(fg_mask) else float(np.mean(np.abs(pred - gt)))
    mse_fg_cal = (
        float(np.mean((pred_cal[fg_mask] - gt[fg_mask]) ** 2)) if np.any(fg_mask) else float(np.mean((pred_cal - gt) ** 2))
    )
    mae_fg_cal = (
        float(np.mean(np.abs(pred_cal[fg_mask] - gt[fg_mask]))) if np.any(fg_mask) else float(np.mean(np.abs(pred_cal - gt)))
    )

    ssim_full = float(np.mean(ssim_map(pred, gt)))
    ssim_fg = masked_mean(ssim_map(pred, gt), fg_mask)
    ssim_fg_cal = masked_mean(ssim_map(pred_cal, gt), fg_mask)
    ms_ssim = ms_ssim_fg(pred, gt, fg_mask)
    ms_ssim_cal = ms_ssim_fg(pred_cal, gt, fg_mask)
    ncc = ncc_fg(pred, gt, fg_mask)

    psnr_fg_cal = psnr_from_mse(mse_fg_cal)
    # Structure-first score. NCC is mapped from [-1, 1] to [0, 1].
    ncc01 = (ncc + 1.0) / 2.0
    psnr01 = float(np.clip((psnr_fg_cal - 10.0) / 25.0, 0.0, 1.0))
    composite = 0.40 * ssim_fg_cal + 0.25 * ms_ssim_cal + 0.25 * ncc01 + 0.10 * psnr01

    return {
        "psnr_full": psnr_from_mse(mse_full),
        "ssim_full": ssim_full,
        "psnr_fg": psnr_from_mse(mse_fg),
        "mae_fg": mae_fg,
        "ssim_fg": ssim_fg,
        "ms_ssim_fg": ms_ssim,
        "ncc_fg": ncc,
        "psnr_fg_cal": psnr_fg_cal,
        "mae_fg_cal": mae_fg_cal,
        "ssim_fg_cal": ssim_fg_cal,
        "ms_ssim_fg_cal": ms_ssim_cal,
        "composite_score": float(composite),
        "fg_fraction": float(fg_mask.mean()),
    }


def mean_std(rows: Iterable[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = [float(r[key]) for r in rows]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def collect_cases(data_root: Path, mask_root: Path):
    dense = next(data_root.glob("*-dense-*.tif"))
    dense_vol = load_volume(dense)
    cases = []
    for p in sorted(data_root.glob("*.tif")):
        if "dense" in p.name:
            continue
        parts = p.stem.split("-")
        mode_part = next((part for part in parts if "x" in part and part[0].isdigit()), "")
        if not mode_part:
            continue
        k = int(mode_part.split("x")[0])
        mode = mode_part.split("x")[1].lower()
        mask_p = mask_root / p.name.replace(".tif", "_mask.tif")
        if not mask_p.exists():
            continue
        cases.append(
            {
                "name": p.stem,
                "K": k,
                "mode": mode,
                "low_path": p,
                "mask_path": mask_p,
                "gt_vol": dense_vol,
            }
        )
    return cases


def load_model_adaptive(run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "best.pth"
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    opts = argparse.Namespace(**ckpt.get("opts", {}))
    state = ckpt["model_state"]
    patch_weight = state.get("patch_embed.weight")
    if patch_weight is not None:
        checkpoint_in_ch = int(patch_weight.shape[1])
        task = getattr(opts, "task", "")
        if task == "spatial_2p5d":
            tw = int(getattr(opts, "temporal_window", 3))
            if checkpoint_in_ch == tw:
                # Older 2.5D checkpoints used [T] channels, while newer build logic
                # may infer [T interp + T mask] from cidc25_root/data_root.
                setattr(opts, "cidc25_root", None)
                setattr(opts, "data_root", "")
                setattr(opts, "no_mask_ch", True)
            elif checkpoint_in_ch == 2 * tw:
                setattr(opts, "cidc25_root", getattr(opts, "cidc25_root", "legacy_masked"))
                setattr(opts, "no_mask_ch", False)
        elif checkpoint_in_ch == 1:
            setattr(opts, "no_mask_ch", True)
        elif checkpoint_in_ch == 2:
            setattr(opts, "no_mask_ch", False)

    model = build_model(opts)
    model.load_state_dict(state)
    return model.to(device).eval(), opts


def write_case_metrics(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out_dir: Path, rows: List[Dict]) -> None:
    cols = [
        "case",
        "K",
        "mode",
        "run_name",
        "algorithm",
        "train_sparse_mode",
        "extras",
        "n_frames",
        "fg_fraction_mean",
        "fg_fraction_std",
        "psnr_full_mean",
        "psnr_full_std",
        "ssim_full_mean",
        "ssim_full_std",
        "psnr_fg_mean",
        "psnr_fg_std",
        "ssim_fg_mean",
        "ssim_fg_std",
        "ms_ssim_fg_mean",
        "ms_ssim_fg_std",
        "ncc_fg_mean",
        "ncc_fg_std",
        "psnr_fg_cal_mean",
        "psnr_fg_cal_std",
        "ssim_fg_cal_mean",
        "ssim_fg_cal_std",
        "ms_ssim_fg_cal_mean",
        "ms_ssim_fg_cal_std",
        "mae_fg_cal_mean",
        "mae_fg_cal_std",
        "composite_score_mean",
        "composite_score_std",
        "passes_ssim85",
    ]
    summary_csv = out_dir / "summary_structural.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "summary_structural.md"
    with open(md_path, "w") as f:
        f.write("# Pollen Structure-Aware Reconstruction Evaluation\n\n")
        f.write(
            "Metrics are foreground-aware. `*_cal` values use foreground linear intensity matching "
            "before metric computation, so they emphasize recovered structure rather than global brightness scale.\n\n"
        )
        ranked = sorted(rows, key=lambda x: (x["case"], -float(x["composite_score_mean"])))
        current = None
        for r in ranked:
            if current != r["case"]:
                current = r["case"]
                f.write(f"\n## {current}\n\n")
                f.write("| Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal | SSIM>=0.85 |\n")
                f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
            f.write(
                f"| {r['run_name']} | {r['algorithm']} | {r['train_sparse_mode']} | "
                f"{r['composite_score_mean']:.4f} | {r['ssim_fg_cal_mean']:.4f} | "
                f"{r['ms_ssim_fg_cal_mean']:.4f} | {r['ncc_fg_mean']:.4f} | "
                f"{r['psnr_fg_cal_mean']:.2f} | {r['passes_ssim85']} |\n"
            )
    print(f"Done. Summary CSV: {summary_csv}")
    print(f"Done. Summary MD:  {md_path}")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structure-aware evaluation for pollen sparse reconstruction.")
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets"),
        help="Pollen 2D sparse dataset root.",
    )
    p.add_argument(
        "--mask_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets/mask"),
        help="Pollen 2D mask dataset root.",
    )
    p.add_argument("--runs_dir", type=Path, default=Path("/data2/wjb/threePM/runs"))
    p.add_argument("--output_dir", type=Path, default=Path("/data2/wjb/threePM/eval_pollen_sparse_structural"))
    p.add_argument("--max_frames", type=int, default=1)
    p.add_argument("--patch_hw", type=int, default=256)
    p.add_argument("--run_filter", type=str, default="sparse3d_")
    p.add_argument("--fg_percentile", type=float, default=70.0)
    p.add_argument("--fg_dilation", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_root = args.output_dir / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(args.data_root, args.mask_root)
    run_dirs = sorted(
        d for d in args.runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(args.run_filter) and (d / "best.pth").exists()
    )

    summary_rows: List[Dict] = []
    for case in cases:
        low_vol = load_volume(case["low_path"])
        high_vol = case["gt_vol"]
        mask_vol = (load_volume(case["mask_path"]) > 0).astype(np.float32)
        n = min(low_vol.shape[0], high_vol.shape[0], mask_vol.shape[0], args.max_frames)
        low_vol, high_vol, mask_vol = low_vol[:n], high_vol[:n], mask_vol[:n]
        case_out = args.output_dir / case["name"]
        case_out.mkdir(parents=True, exist_ok=True)
        case_pred_out = pred_root / case["name"]
        case_pred_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Case {case['name']} | K={case['K']} | mode={case['mode']} | frames={n} ===")

        for rd in run_dirs:
            cfg = parse_run_config(rd.name)
            if str(case["K"]) != cfg["K"]:
                continue
            try:
                model, opts = load_model_adaptive(rd, device)
            except RuntimeError as exc:
                print(f"  [Skip] {rd.name}: incompatible checkpoint ({exc})")
                continue
            per_frame: List[Dict] = []
            pred_stack: List[np.ndarray] = []

            for i in range(n):
                try:
                    pred = predict_frame(model, opts, cfg["algorithm"], low_vol, high_vol, mask_vol, i, args.patch_hw, device)
                except RuntimeError as exc:
                    print(f"  [Skip] {rd.name}: inference failed ({exc})")
                    per_frame = []
                    pred_stack = []
                    break
                _, gt_n_pair = minmax_pair(low_vol[i], high_vol[i])
                gt_n = ((high_vol[i] - high_vol[i].min()) / (high_vol[i].max() - high_vol[i].min() + 1e-8)).astype(np.float32)
                # Use high-only GT normalization for metrics, but keep pair-normalized sparse display compatible elsewhere.
                _ = gt_n_pair
                fg = make_foreground_mask(gt_n, percentile=args.fg_percentile, dilation_iter=args.fg_dilation)
                metrics = structural_metrics(pred, gt_n, fg)
                row = {"frame": i, **metrics}
                per_frame.append(row)
                pred_stack.append(pred.astype(np.float32))

            if not per_frame:
                del model
                torch.cuda.empty_cache()
                continue

            np.save(case_pred_out / f"{rd.name}_pred.npy", np.stack(pred_stack, axis=0))
            write_case_metrics(case_out / f"{rd.name}_metrics_structural.csv", per_frame)

            summary = {
                "case": case["name"],
                "K": case["K"],
                "mode": case["mode"],
                "run_name": rd.name,
                "algorithm": cfg["algorithm"],
                "train_sparse_mode": cfg["sparse_mode"],
                "extras": cfg["extras"],
                "n_frames": len(per_frame),
            }
            for key in [
                "fg_fraction",
                "psnr_full",
                "ssim_full",
                "psnr_fg",
                "ssim_fg",
                "ms_ssim_fg",
                "ncc_fg",
                "psnr_fg_cal",
                "ssim_fg_cal",
                "ms_ssim_fg_cal",
                "mae_fg_cal",
                "composite_score",
            ]:
                mean, std = mean_std(per_frame, key)
                summary[f"{key}_mean"] = mean
                summary[f"{key}_std"] = std
            summary["passes_ssim85"] = bool(summary["ssim_fg_cal_mean"] >= 0.85)
            summary_rows.append(summary)
            print(
                f"  {rd.name}: Score={summary['composite_score_mean']:.4f} "
                f"SSIMfg_cal={summary['ssim_fg_cal_mean']:.4f} "
                f"NCCfg={summary['ncc_fg_mean']:.4f} PSNRfg_cal={summary['psnr_fg_cal_mean']:.2f}"
            )

            del model
            torch.cuda.empty_cache()

    write_summary(args.output_dir, summary_rows)


if __name__ == "__main__":
    main()

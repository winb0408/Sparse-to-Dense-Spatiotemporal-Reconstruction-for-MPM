#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F_torch
import tifffile
from scipy.ndimage import uniform_filter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import warm_start_2d, warm_start_3d, wiener_init_2d, wiener_init_3d


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
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
    return float(np.mean(num / (den + 1e-8)))


def minmax_pair(low: np.ndarray, high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vmin = float(min(low.min(), high.min()))
    vmax = float(max(low.max(), high.max()))
    denom = vmax - vmin + 1e-8
    low_n = ((low - vmin) / denom).clip(0, 1).astype(np.float32)
    high_n = ((high - vmin) / denom).clip(0, 1).astype(np.float32)
    return low_n, high_n


def minmax_pair_255(low: np.ndarray, high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    low_n, high_n = minmax_pair(low, high)
    return low_n * 255.0, high_n * 255.0


def to_model_range(x: torch.Tensor, norm_mode: str) -> torch.Tensor:
    if norm_mode in ("minmax255", "fixed255"):
        return x / 127.5 - 1.0
    return x * 2.0 - 1.0


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
    c, h, w = inp.shape
    # Keep strictly full-image inference. If a checkpoint has stricter shape
    # constraints, retry with larger global padding alignment (still one-shot).
    last_err: Optional[RuntimeError] = None
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
            continue
    raise RuntimeError(
        f"Full-image inference failed after multi-align padding retries for shape ({h}, {w})."
    ) from last_err


def evaluate_2d(
    model,
    opts,
    low_vol: np.ndarray,
    high_vol: np.ndarray,
    mask_vol: np.ndarray,
    max_frames: int,
    device: torch.device,
) -> List[Dict]:
    use_mask = not getattr(opts, "no_mask_ch", False)
    use_wiener = getattr(opts, "wiener_init", False)
    norm_mode = getattr(opts, "norm_mode", "minmax")
    k = getattr(opts, "sparsity_factor", 4)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    rows: List[Dict] = []

    n = min(max_frames, low_vol.shape[0], high_vol.shape[0], mask_vol.shape[0])
    for i in range(n):
        if norm_mode in ("minmax255", "fixed255"):
            low_n, high_n = minmax_pair_255(low_vol[i], high_vol[i])
        else:
            low_n, high_n = minmax_pair(low_vol[i], high_vol[i])
        low_t = torch.from_numpy(low_n).unsqueeze(0)
        mask_t = torch.from_numpy(mask_vol[i].astype(np.float32)).unsqueeze(0)
        sparse = low_t * mask_t
        if use_wiener:
            interp = wiener_init_2d(sparse.squeeze(0), psf_sig).unsqueeze(0)
        else:
            interp = warm_start_2d(sparse, k)
        inp = to_model_range(interp, norm_mode)
        if use_mask:
            inp = torch.cat([inp, mask_t], dim=0)

        pred = infer_2d_full(model, inp, device)
        gt_eval = (high_n / 255.0).clip(0, 1) if norm_mode in ("minmax255", "fixed255") else high_n
        rows.append({"frame": i, "psnr": psnr(pred, gt_eval), "ssim": ssim_2d(pred, gt_eval)})
    return rows


def evaluate_2p5d(
    model,
    opts,
    low_vol: np.ndarray,
    high_vol: np.ndarray,
    mask_vol: np.ndarray,
    max_frames: int,
    device: torch.device,
) -> List[Dict]:
    tw = int(getattr(opts, "temporal_window", 3))
    radius = tw // 2
    use_wiener = getattr(opts, "wiener_init", False)
    k = getattr(opts, "sparsity_factor", 4)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    norm_mode = getattr(opts, "norm_mode", "minmax")
    expected_in = int(getattr(getattr(model, "patch_embed", None), "in_channels", tw))
    legacy_input = expected_in == tw
    mask_input = expected_in == (2 * tw)
    rows: List[Dict] = []

    n = min(max_frames, low_vol.shape[0], high_vol.shape[0], mask_vol.shape[0])
    if n <= 0:
        return rows
    for c in range(n):
        # Edge-replication so single-frame / short stacks can still be evaluated.
        idxs = [min(max(c + d, 0), n - 1) for d in range(-radius, radius + 1)]
        low_chunk = np.stack([low_vol[i] for i in idxs], axis=0)
        high_chunk = np.stack([high_vol[i] for i in idxs], axis=0)
        mask_chunk = np.stack([mask_vol[i] for i in idxs], axis=0).astype(np.float32)

        if legacy_input:
            if norm_mode in ("minmax255", "fixed255"):
                center_low_n, center_high_n = minmax_pair_255(low_chunk[radius], high_chunk[radius])
            else:
                center_low_n, center_high_n = minmax_pair(low_chunk[radius], high_chunk[radius])
            # Legacy 2.5D uses the same min-max for neighbor frames.
            vmin = float(min(low_chunk[radius].min(), high_chunk[radius].min()))
            vmax = float(max(low_chunk[radius].max(), high_chunk[radius].max()))
            denom = vmax - vmin + 1e-8
            low_stack_n = ((low_chunk - vmin) / denom).clip(0, 1).astype(np.float32)
            if norm_mode in ("minmax255", "fixed255"):
                low_stack_n = low_stack_n * 255.0
            inp = to_model_range(torch.from_numpy(low_stack_n), norm_mode)
            gt = (center_high_n / 255.0).clip(0, 1) if norm_mode in ("minmax255", "fixed255") else center_high_n
        elif mask_input:
            # Mask-aware 2.5D path.
            vmin = float(high_chunk.min())
            vmax = float(high_chunk.max())
            denom = vmax - vmin + 1e-8
            high_stack_n = ((high_chunk - vmin) / denom).clip(0, 1).astype(np.float32)
            if norm_mode in ("minmax255", "fixed255"):
                high_stack_n = high_stack_n * 255.0
            interp_list = []
            mask_list = []
            for t in range(tw):
                low_t = ((low_chunk[t] - vmin) / denom).clip(0, 1).astype(np.float32)
                if norm_mode in ("minmax255", "fixed255"):
                    low_t = low_t * 255.0
                low_tt = torch.from_numpy(low_t).unsqueeze(0)
                mask_tt = torch.from_numpy(mask_chunk[t]).unsqueeze(0)
                sparse = low_tt * mask_tt
                if use_wiener:
                    interp = wiener_init_2d(sparse.squeeze(0), psf_sig).unsqueeze(0)
                else:
                    interp = warm_start_2d(sparse, k)
                interp_list.append(interp)
                mask_list.append(mask_tt)
            interp_all = to_model_range(torch.cat(interp_list, dim=0), norm_mode)
            mask_all = torch.cat(mask_list, dim=0)
            inp = torch.cat([interp_all, mask_all], dim=0)
            gt = high_stack_n[radius]
            if norm_mode in ("minmax255", "fixed255"):
                gt = (gt / 255.0).clip(0, 1)
        else:
            continue

        pred = infer_2d_full(model, inp, device)
        rows.append({"frame": c, "psnr": psnr(pred, gt), "ssim": ssim_2d(pred, gt)})
    return rows


def evaluate_3d(
    model,
    opts,
    low_vol: np.ndarray,
    high_vol: np.ndarray,
    mask_vol: np.ndarray,
    max_frames: int,
    patch_hw: int,
    device: torch.device,
) -> List[Dict]:
    vol_t = int(getattr(opts, "volume_patch_t", 16))
    use_mask = not getattr(opts, "no_mask_ch", False)
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = getattr(opts, "psf_sigma_lateral", 1.0)
    psf_sig_ax = getattr(opts, "psf_sigma_axial", 3.0)
    k = int(getattr(opts, "sparsity_factor", 4))
    norm_mode = getattr(opts, "norm_mode", "minmax")

    n = min(max_frames, low_vol.shape[0], high_vol.shape[0], mask_vol.shape[0])
    h, w = high_vol.shape[1:]
    pred_all = np.zeros((n, h, w), dtype=np.float32)
    gt_all = np.zeros((n, h, w), dtype=np.float32)

    for t0 in range(0, n, vol_t):
        t1 = min(t0 + vol_t, n)
        act_t = t1 - t0
        high_chunk = high_vol[t0:t1].astype(np.float32)
        low_chunk = low_vol[t0:t1].astype(np.float32)
        mask_chunk = mask_vol[t0:t1].astype(np.float32)

        vmin = float(high_chunk.min())
        vmax = float(high_chunk.max())
        denom = vmax - vmin + 1e-8
        high_n = ((high_chunk - vmin) / denom).clip(0, 1).astype(np.float32)
        low_n = ((low_chunk - vmin) / denom).clip(0, 1).astype(np.float32)
        if norm_mode in ("minmax255", "fixed255"):
            high_n = high_n * 255.0
            low_n = low_n * 255.0
            gt_all[t0:t1] = (high_n[:act_t] / 255.0).clip(0, 1)
        else:
            gt_all[t0:t1] = high_n

        if act_t < vol_t:
            pad_t = vol_t - act_t
            high_n = np.concatenate([high_n, np.zeros((pad_t, h, w), dtype=np.float32)], axis=0)
            low_n = np.concatenate([low_n, np.zeros((pad_t, h, w), dtype=np.float32)], axis=0)
            mask_chunk = np.concatenate([mask_chunk, np.zeros((pad_t, h, w), dtype=np.float32)], axis=0)

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
                if use_wiener:
                    interp = wiener_init_3d(sparse.squeeze(0), psf_sig, psf_sig_ax).unsqueeze(0)
                else:
                    interp = warm_start_3d(sparse, factor=k, factor_z=1)
                inp = to_model_range(interp, norm_mode)
                if use_mask:
                    inp = torch.cat([inp, torch.from_numpy(mask_tile).unsqueeze(0)], dim=0)
                out = model(inp.unsqueeze(0).to(device))
                out_np = ((out[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
                pred_all[t0:t1, y0:y1, x0:x1] = out_np[:act_t, :ph, :pw]

    rows: List[Dict] = []
    for i in range(n):
        rows.append({"frame": i, "psnr": psnr(pred_all[i], gt_all[i]), "ssim": ssim_2d(pred_all[i], gt_all[i])})
    return rows


def load_volume(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    while arr.ndim > 3:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def collect_cases(data_root: Path, mask_root: Path):
    dense = next(data_root.glob("*-dense-*.tif"))
    dense_vol = load_volume(dense)
    cases = []
    pat = re.compile(r".*-(\d+)x([A-Z]+)-.*\.tif$")
    for p in sorted(data_root.glob("*.tif")):
        if "dense" in p.name:
            continue
        m = pat.match(p.name)
        if not m:
            continue
        k = int(m.group(1))
        mode = m.group(2).lower()
        mask_p = mask_root / p.name.replace(".tif", "_mask.tif")
        if not mask_p.exists():
            continue
        cases.append({
            "name": p.stem,
            "K": k,
            "mode": mode,
            "low_path": p,
            "mask_path": mask_p,
            "gt_vol": dense_vol,
        })
    return cases


def main():
    ap = argparse.ArgumentParser(description="Evaluate pretrained sparse models on pollen sparse dataset.")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--mask_root", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--output_dir", type=str, default="eval_pollen_sparse")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--patch_hw", type=int, default=256)
    ap.add_argument("--run_filter", type=str, default="sparse3d_")
    ap.add_argument(
        "--allow_cross_dim_transfer",
        action="store_true",
        default=False,
        help="Evaluate 2.5D/3D checkpoints on the 2D pollen protocol as transfer probes.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root)
    cases = collect_cases(data_root, mask_root)
    if not cases:
        raise RuntimeError("No valid sparse pollen cases found.")

    runs_dir = Path(args.runs_dir)
    run_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(args.run_filter) and (d / "best.pth").exists()
    )

    summary_rows: List[Dict] = []
    for case in cases:
        low_vol = load_volume(case["low_path"])
        mask_vol = load_volume(case["mask_path"])
        high_vol = case["gt_vol"]
        n = min(low_vol.shape[0], high_vol.shape[0], mask_vol.shape[0], args.max_frames)
        low_vol = low_vol[:n]
        high_vol = high_vol[:n]
        mask_vol = (mask_vol[:n] > 0).astype(np.float32)

        case_out = out_dir / case["name"]
        case_out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Case {case['name']} | K={case['K']} | mode={case['mode']} | frames={n} ===")

        for rd in run_dirs:
            cfg = parse_run_config(rd.name)
            if cfg["algorithm"] in {"2p5d", "3d"} and not args.allow_cross_dim_transfer:
                continue
            # Keep comparable sparsity level. Cross-dimensional runs require
            # --allow_cross_dim_transfer because the current pollen protocol is
            # a 2D mean-projection dataset, not a 3D volume benchmark.
            if str(case["K"]) != cfg["K"]:
                continue
            model, opts = load_model(rd, device)
            algo = cfg["algorithm"]
            if algo == "2d":
                per_frame = evaluate_2d(model, opts, low_vol, high_vol, mask_vol, n, device)
            elif algo == "2p5d":
                per_frame = evaluate_2p5d(model, opts, low_vol, high_vol, mask_vol, n, device)
            elif algo == "3d":
                per_frame = evaluate_3d(model, opts, low_vol, high_vol, mask_vol, n, args.patch_hw, device)
            else:
                del model
                torch.cuda.empty_cache()
                continue

            ps = [r["psnr"] for r in per_frame]
            ss = [r["ssim"] for r in per_frame]
            row = {
                "case": case["name"],
                "K": case["K"],
                "mode": case["mode"],
                "run_name": rd.name,
                "algorithm": cfg["algorithm"],
                "train_sparse_mode": cfg["sparse_mode"],
                "extras": cfg["extras"],
                "n_frames": len(per_frame),
                "psnr_mean": float(np.mean(ps)) if ps else 0.0,
                "psnr_std": float(np.std(ps)) if ps else 0.0,
                "ssim_mean": float(np.mean(ss)) if ss else 0.0,
                "ssim_std": float(np.std(ss)) if ss else 0.0,
            }
            summary_rows.append(row)
            print(f"  {rd.name}: PSNR={row['psnr_mean']:.2f} SSIM={row['ssim_mean']:.4f}")

            per_csv = case_out / f"{rd.name}_metrics.csv"
            with open(per_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "psnr", "ssim"])
                w.writeheader()
                w.writerows(per_frame)

            del model
            torch.cuda.empty_cache()

    summary_csv = out_dir / "summary_pollen.csv"
    cols = [
        "case",
        "K",
        "mode",
        "run_name",
        "algorithm",
        "train_sparse_mode",
        "extras",
        "n_frames",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
    ]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)

    md_path = out_dir / "summary_pollen.md"
    with open(md_path, "w") as f:
        f.write("# Pollen Sparse Reconstruction Evaluation\n\n")
        ranked = sorted(summary_rows, key=lambda x: (x["case"], -x["psnr_mean"]))
        current = None
        for r in ranked:
            if current != r["case"]:
                current = r["case"]
                f.write(f"\n## {current}\n\n")
                f.write("| Run | Algo | TrainSparse | PSNR(dB) | SSIM |\n")
                f.write("|---|---|---|---:|---:|\n")
            f.write(
                f"| {r['run_name']} | {r['algorithm']} | {r['train_sparse_mode']} | "
                f"{r['psnr_mean']:.2f}±{r['psnr_std']:.2f} | "
                f"{r['ssim_mean']:.4f}±{r['ssim_std']:.4f} |\n"
            )

    print(f"\nDone. Summary CSV: {summary_csv}")
    print(f"Done. Summary MD:  {md_path}")


if __name__ == "__main__":
    main()

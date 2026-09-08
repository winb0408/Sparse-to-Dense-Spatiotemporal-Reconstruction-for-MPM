#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_torch
import tifffile

from dataset import normalise_pair, make_mask_2d, wiener_init_2d, warm_start_2d, to_model_range, wiener_init_3d, warm_start_3d
from scripts.eval_sparse3d_unified import load_model


def infer_2d_full(model, inp_chw: torch.Tensor, device: torch.device) -> np.ndarray:
    c, h, w = inp_chw.shape
    x0 = inp_chw.unsqueeze(0)
    last_err = None
    for align in (32, 64, 96, 128, 160, 192, 224, 256):
        hp = ((h + align - 1) // align) * align
        wp = ((w + align - 1) // align) * align
        x = x0
        if hp != h or wp != w:
            x = F_torch.pad(x, (0, wp - w, 0, hp - h), mode="replicate")
        try:
            out = model(x.to(device))
            y = ((out[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            return y[:h, :w]
        except RuntimeError as e:
            last_err = e
            if "shape" not in str(e) and "view" not in str(e):
                raise
    raise last_err if last_err is not None else RuntimeError("2D full inference failed.")


def infer_2d_pred(model, opts, low: np.ndarray, gt: np.ndarray) -> np.ndarray:
    low_n, _ = normalise_pair(low.astype(np.float32), gt.astype(np.float32), mode=getattr(opts, "norm_mode", "minmax"))
    h, w = low_n.shape
    low_t = torch.from_numpy(low_n).unsqueeze(0)
    mask = make_mask_2d(h, w, getattr(opts, "sparsity_factor", 2), getattr(opts, "scan_pattern", "grid"))
    sparse = low_t * mask
    if getattr(opts, "wiener_init", False):
        interp = wiener_init_2d(sparse.squeeze(0), getattr(opts, "psf_sigma_lateral", 1.0)).unsqueeze(0)
    else:
        interp = warm_start_2d(sparse, getattr(opts, "sparsity_factor", 2))
    inp = interp * 2.0 - 1.0
    if not getattr(opts, "no_mask_ch", False):
        inp = torch.cat([inp, mask], dim=0)
    return infer_2d_full(model, inp, next(model.parameters()).device)


def infer_2p5d_pred(model, opts, sparse_stack: np.ndarray, gt_stack: np.ndarray) -> np.ndarray:
    tw = int(getattr(opts, "temporal_window", 3))
    radius = tw // 2
    z = int(sparse_stack.shape[0] // 2)
    idxs = [min(max(z + d, 0), sparse_stack.shape[0] - 1) for d in range(-radius, radius + 1)]
    low_chunk = sparse_stack[idxs].astype(np.float32)
    gt_chunk = gt_stack[idxs].astype(np.float32)
    vmin = float(gt_chunk.min())
    vmax = float(gt_chunk.max())
    denom = vmax - vmin + 1e-8
    low_n = ((low_chunk - vmin) / denom).clip(0, 1)

    expected_in_ch = int(model.patch_embed.in_channels)
    use_masked_input = expected_in_ch == (2 * tw)
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = float(getattr(opts, "psf_sigma_lateral", 1.0))

    if use_masked_input:
        interps, masks = [], []
        for t in range(tw):
            low_t = torch.from_numpy(low_n[t]).unsqueeze(0)
            mask_t = (low_t > 1e-6).float()
            sparse_t = low_t * mask_t
            if use_wiener:
                interp_t = wiener_init_2d(sparse_t.squeeze(0), psf_sig).unsqueeze(0)
            else:
                interp_t = sparse_t
            interps.append(interp_t)
            masks.append(mask_t)
        inp = torch.cat([torch.cat(interps, dim=0) * 2.0 - 1.0, torch.cat(masks, dim=0)], dim=0)
    else:
        inp = torch.from_numpy(low_n.copy()) * 2.0 - 1.0

    return infer_2d_full(model, inp, next(model.parameters()).device)


def infer_3d_center_pred(model, opts, sparse_stack: np.ndarray, gt_stack: np.ndarray) -> np.ndarray:
    vol_t = int(getattr(opts, "volume_patch_t", 16))
    patch_hw = int(getattr(opts, "patch_size", 256))
    z, h, w = sparse_stack.shape
    low = sparse_stack.astype(np.float32)
    gt = gt_stack.astype(np.float32)
    mask = (low > 1e-6).astype(np.float32)
    pred = np.zeros_like(low, dtype=np.float32)
    use_mask = not getattr(opts, "no_mask_ch", False)
    use_wiener = getattr(opts, "wiener_init", False)
    psf_sig = float(getattr(opts, "psf_sigma_lateral", 1.0))
    psf_sig_ax = float(getattr(opts, "psf_sigma_axial", 3.0))
    k = int(getattr(opts, "sparsity_factor", 2))
    for t0 in range(0, z, vol_t):
        t1 = min(t0 + vol_t, z)
        act_t = t1 - t0
        gt_c = gt[t0:t1]
        low_c = low[t0:t1]
        m_c = mask[t0:t1]
        vmin = float(gt_c.min())
        vmax = float(gt_c.max())
        denom = vmax - vmin + 1e-8
        gt_n = ((gt_c - vmin) / denom).clip(0, 1)
        low_n = ((low_c - vmin) / denom).clip(0, 1)
        if act_t < vol_t:
            pad = vol_t - act_t
            low_n = np.concatenate([low_n, np.zeros((pad, h, w), np.float32)], axis=0)
            m_c = np.concatenate([m_c, np.zeros((pad, h, w), np.float32)], axis=0)
        for y0 in range(0, h, patch_hw):
            for x0 in range(0, w, patch_hw):
                y1, x1 = min(y0 + patch_hw, h), min(x0 + patch_hw, w)
                ph, pw = y1 - y0, x1 - x0
                lt = low_n[:, y0:y1, x0:x1]
                mt = m_c[:, y0:y1, x0:x1]
                if ph < patch_hw or pw < patch_hw:
                    lpad = np.zeros((vol_t, patch_hw, patch_hw), np.float32)
                    mpad = np.zeros((vol_t, patch_hw, patch_hw), np.float32)
                    lpad[:, :ph, :pw] = lt
                    mpad[:, :ph, :pw] = mt
                    lt, mt = lpad, mpad
                sparse = torch.from_numpy(lt).unsqueeze(0) * torch.from_numpy(mt).unsqueeze(0)
                if use_wiener:
                    interp = wiener_init_3d(sparse.squeeze(0), psf_sig, psf_sig_ax).unsqueeze(0)
                else:
                    interp = warm_start_3d(sparse, factor=k, factor_z=1)
                inp = to_model_range(interp, getattr(opts, "norm_mode", "minmax"))
                if use_mask:
                    inp = torch.cat([inp, torch.from_numpy(mt).unsqueeze(0)], dim=0)
                out = model(inp.unsqueeze(0).to(next(model.parameters()).device))
                out_np = ((out[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
                pred[t0:t1, y0:y1, x0:x1] = out_np[:act_t, :ph, :pw]
    return pred[z // 2]


def load01_tif(path: Path) -> np.ndarray:
    x = tifffile.imread(str(path)).astype(np.float32)
    if x.ndim > 2:
        x = x[0]
    vmin, vmax = float(x.min()), float(x.max())
    return ((x - vmin) / (vmax - vmin + 1e-8)).clip(0, 1)


def main():
    ap = argparse.ArgumentParser(description="MDC25 K2 representative GT/Sparse/best-2D/2.5D/3D figure")
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--sparse_root", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--output_path", type=str, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.summary_csv)))
    runs_dir = Path(args.runs_dir)
    sparse_root = Path(args.sparse_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def best_run(mode: str, algo: str) -> str:
        cand = [r for r in rows if f"_{mode}_" in r["run_name"] and algo in r["run_name"]]
        cand = sorted(cand, key=lambda x: -float(x["psnr_mean"]))
        if not cand:
            raise RuntimeError(f"No run found for mode={mode}, algo={algo}")
        return cand[0]["run_name"]

    fig, axes = plt.subplots(3, 5, figsize=(18, 10), dpi=200)
    col_titles = ["GT", "Sparse", "Best 2D", "Best 2.5D", "Best 3D"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11)

    for i, mode in enumerate(("xy", "z", "xyz")):
        v3 = sparse_root / f"3d/K2_{mode}/Validation"
        gt_tif = sorted((v3 / "high").glob("*.tif"))[0]
        sp_tif = sorted((v3 / "low").glob("*.tif"))[0]
        gt2d = load01_tif(gt_tif)
        sp2d = load01_tif(sp_tif)

        stack_gt = sorted((v3 / "stacks").glob("*_gt.npy"))[0]
        stack_sp = Path(str(stack_gt).replace("_gt.npy", "_sparse.npy"))
        gt_stack = np.load(stack_gt).astype(np.float32)
        sp_stack = np.load(stack_sp).astype(np.float32)

        rn_2d = best_run(mode, "_2d_")
        rn_25 = best_run(mode, "_2p5d_")
        rn_3d = best_run(mode, "_3d_")

        m2d, o2d = load_model(runs_dir / rn_2d, device)
        m25, o25 = load_model(runs_dir / rn_25, device)
        m3d, o3d = load_model(runs_dir / rn_3d, device)
        p2d = infer_2d_pred(m2d, o2d, sp2d, gt2d)
        p25 = infer_2p5d_pred(m25, o25, sp_stack, gt_stack)
        p3d = infer_3d_center_pred(m3d, o3d, sp_stack, gt_stack)

        imgs = [gt2d, sp2d, p2d, p25, p3d]
        for j, im in enumerate(imgs):
            axes[i, j].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"Mode={mode}", fontsize=11)

    plt.tight_layout()
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

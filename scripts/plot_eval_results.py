#!/usr/bin/env python3
"""
Generate visualization figures for sparse 3D reconstruction evaluation results.

Produces:
  fig1_phase1_grouped_bar.pdf/png   — Phase 1: 2D/2.5D/3D × xy/z/xyz grouped bar
  fig2_phase2_psf_ablation.pdf/png  — Phase 2: PSF ablation on xy mode
  fig3_perframe_psnr.pdf/png        — Per-frame PSNR curves (key models)
  fig4_summary_heatmap.pdf/png      — PSNR / SSIM heatmap overview
  fig6_k_sweep_psnr.pdf/png         — PSNR vs K curves
  fig7_optimization_gain.pdf/png    — gains over same-K same-mode baseline

Usage:
  python scripts/plot_eval_results.py \
    --eval_dir eval_sparse3d \
    --output_dir eval_sparse3d/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

ALGO_COLOR = {
    "input":  "#aaaaaa",
    "2d":     "#4878cf",
    "2p5d":   "#6acc65",
    "3d":     "#d65f5f",
    "3d+psf": "#ee8844",
}
MODE_HATCH = {"xy": "", "z": "////", "xyz": "xxxx"}
ALGO_LABEL = {"input": "Input", "2d": "2D", "2p5d": "2.5D", "3d": "3D"}


def load_summary(eval_dir: Path) -> List[Dict]:
    rows = []
    with open(eval_dir / "summary_all.csv") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in {"run_name","algorithm","sparse_mode","extras"} else v)
                         for k, v in r.items()})
    return rows


def load_perframe(eval_dir: Path, run_name: str):
    p = eval_dir / run_name / "metrics.csv"
    frames, psnrs, ssims = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            frames.append(int(r["frame"]))
            psnrs.append(float(r["psnr"]))
            ssims.append(float(r["ssim"]))
    return frames, psnrs, ssims


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Phase 1: grouped bar (algo × sparse_mode)
# ══════════════════════════════════════════════════════════════════════════════

def fig_phase1_grouped_bar(rows: List[Dict], out_dir: Path):
    algos = ["input", "2d", "2p5d", "3d"]
    modes = ["xy", "z", "xyz"]

    data = {}
    for algo in algos:
        for mode in modes:
            hits = [r for r in rows
                    if r["algorithm"] == algo and r["sparse_mode"] == mode
                    and r["extras"] in ("", None)]
            if hits:
                data[(algo, mode)] = hits[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bar_w = 0.18
    x = np.arange(len(modes))

    for ax_idx, metric in enumerate(["psnr_mean", "ssim_mean"]):
        ax = axes[ax_idx]
        ylabel = "PSNR (dB)" if metric == "psnr_mean" else "SSIM"
        err_key = "psnr_std" if metric == "psnr_mean" else "ssim_std"

        for i, algo in enumerate(algos):
            vals, errs = [], []
            for mode in modes:
                d = data.get((algo, mode))
                vals.append(d[metric] if d else 0.0)
                errs.append(d[err_key] if d else 0.0)
            offset = (i - len(algos) / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=ALGO_LABEL.get(algo, algo),
                          color=ALGO_COLOR.get(algo, "#888888"),
                          yerr=errs, capsize=3,
                          error_kw={"elinewidth": 1.2, "alpha": 0.8},
                          alpha=0.85)

        ax.set_xlabel("Sparse Mode")
        ax.set_ylabel(ylabel)
        title = ("Phase 1 — PSNR Comparison\n(Validation/F0, K=4, 200 frames)"
                 if metric == "psnr_mean"
                 else "Phase 1 — SSIM Comparison\n(Validation/F0, K=4, 200 frames)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["XY sparse", "Z sparse", "XYZ sparse"])
        ax.legend(loc="upper right" if metric == "psnr_mean" else "lower right")

        # Annotate top bars
        for i, algo in enumerate(algos):
            for j, mode in enumerate(modes):
                d = data.get((algo, mode))
                if d and d[metric] > 0:
                    v = d[metric]
                    offset = (i - len(algos) / 2 + 0.5) * bar_w
                    fmt = f"{v:.1f}" if metric == "psnr_mean" else f"{v:.3f}"
                    ax.text(x[j] + offset, v + (d[err_key] + 0.3),
                            fmt, ha="center", va="bottom", fontsize=6.5, rotation=90)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig1_phase1_grouped_bar.{ext}")
    plt.close()
    print(f"  fig1 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Phase 2: PSF ablation (xy mode only)
# ══════════════════════════════════════════════════════════════════════════════

def fig_phase2_psf_ablation(rows: List[Dict], out_dir: Path):
    # Models to compare (all xy mode, 3D)
    targets = [
        ("baseline_input_xy_K4",    "input", "Input (XY)"),
        ("baseline_input_xy",       "input_old", "Input (XY, old tag)"),
        ("sparse3d_3d_xy_K4",        "3d_base", "3D (baseline)"),
        ("sparse3d_3d_xy_K4_psf_ch", "3d_psf_ch", "3D + PSF channel"),
        ("sparse3d_3d_xy_K4_psf_loss","3d_psf_loss","3D + PSF loss"),
        ("sparse3d_3d_xy_K4_psf_full","3d_psf_full","3D + PSF full"),
        ("sparse3d_3d_xy_K4_wiener",  "3d_wiener","3D + Wiener init"),
        ("sparse3d_2d_xy_K4",         "2d", "2D (ref)"),
        ("sparse3d_2p5d_xy_K4",       "2p5d", "2.5D (ref)"),
    ]
    row_map = {r["run_name"]: r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#aaaaaa", "#d65f5f", "#f0a070", "#f0c840", "#e05050",
              "#2080c0", "#4878cf", "#6acc65"]

    for ax_idx, metric in enumerate(["psnr_mean", "ssim_mean"]):
        ax = axes[ax_idx]
        err_key = "psnr_std" if metric == "psnr_mean" else "ssim_std"
        ylabel = "PSNR (dB)" if metric == "psnr_mean" else "SSIM"

        names, vals, errs, clrs = [], [], [], []
        for run_name, _, label in targets:
            if run_name in row_map:
                d = row_map[run_name]
                names.append(label)
                vals.append(d[metric])
                errs.append(d[err_key])
        clrs = colors[:len(names)]

        y = np.arange(len(names))
        bars = ax.barh(y, vals, xerr=errs, height=0.6, color=clrs,
                       capsize=3, error_kw={"elinewidth": 1.2}, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel(ylabel)
        ax.set_title(f"Phase 2 PSF Ablation — {ylabel}\n(XY sparse, K=4, 200 frames)")

        # Value labels
        for i, (v, e) in enumerate(zip(vals, errs)):
            fmt = f"{v:.2f}" if metric == "psnr_mean" else f"{v:.4f}"
            ax.text(v + e + 0.2, i, fmt, va="center", fontsize=8)

        # Vertical reference line at input baseline
        input_val = next((d[metric] for rn, _, _ in targets
                          if rn == "baseline_input_xy" and rn in row_map
                          for d in [row_map[rn]]), None)
        if input_val is None:
            input_val = next((d[metric] for rn, _, _ in targets
                              if rn == "baseline_input_xy_K4" and rn in row_map
                              for d in [row_map[rn]]), None)
        if input_val:
            ax.axvline(input_val, color="gray", linestyle="--", alpha=0.6, linewidth=1)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig2_phase2_psf_ablation.{ext}")
    plt.close()
    print(f"  fig2 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-frame PSNR curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_perframe_psnr(eval_dir: Path, out_dir: Path):
    key_runs = [
        ("sparse3d_2p5d_z_K4",        "#2ca02c", "2.5D + Z sparse"),
        ("sparse3d_3d_xy_K4_wiener",   "#1f77b4", "3D + Wiener (XY)"),
        ("sparse3d_2d_z_K4",           "#9467bd", "2D + Z sparse"),
        ("sparse3d_2d_xy_K4",          "#ff7f0e", "2D + XY sparse"),
        ("sparse3d_3d_xy_K4",          "#d62728", "3D (XY baseline)"),
        ("sparse3d_2p5d_xy_K4",        "#8c564b", "2.5D + XY sparse"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for (run_name, color, label) in key_runs:
        try:
            frames, psnrs, ssims = load_perframe(eval_dir, run_name)
        except FileNotFoundError:
            continue
        smooth_p = np.convolve(psnrs, np.ones(5)/5, mode="same")
        smooth_s = np.convolve(ssims, np.ones(5)/5, mode="same")
        axes[0].plot(frames, smooth_p, color=color, label=label, linewidth=1.4, alpha=0.85)
        axes[1].plot(frames, smooth_s, color=color, label=label, linewidth=1.4, alpha=0.85)

    for ax, ylabel, title in zip(axes,
                                  ["PSNR (dB)", "SSIM"],
                                  ["Per-frame PSNR (moving avg 5)",
                                   "Per-frame SSIM (moving avg 5)"]):
        ax.set_xlabel("Frame index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("Per-frame Reconstruction Quality — Validation/F0.tif", y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig3_perframe_curves.{ext}")
    plt.close()
    print(f"  fig3 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Summary heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_summary_heatmap(rows: List[Dict], out_dir: Path):
    # Filter model runs only (exclude baselines)
    model_rows = [r for r in rows if r["algorithm"] not in ("input",)]
    model_rows.sort(key=lambda r: -r["psnr_mean"])

    run_labels = [r["run_name"].replace("sparse3d_", "") for r in model_rows]
    psnr_vals = np.array([r["psnr_mean"] for r in model_rows])
    ssim_vals = np.array([r["ssim_mean"] for r in model_rows])

    fig, axes = plt.subplots(1, 2, figsize=(9, 0.55 * len(model_rows) + 1.5))

    for ax, vals, vmin_v, vmax_v, cmap, label in [
        (axes[0], psnr_vals[:, None], 18, 45, "RdYlGn", "PSNR (dB)"),
        (axes[1], ssim_vals[:, None], 0.5, 1.0, "RdYlGn", "SSIM"),
    ]:
        im = ax.imshow(vals, cmap=cmap, aspect="auto",
                       vmin=vmin_v, vmax=vmax_v)
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        ax.set_yticks(range(len(run_labels)))
        ax.set_yticklabels(run_labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        # Annotate cells
        for i, v in enumerate(vals[:, 0]):
            fmt = f"{v:.1f}" if label == "PSNR (dB)" else f"{v:.3f}"
            ax.text(0, i, fmt, ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < (v - vmin_v)/(vmax_v - vmin_v) < 0.8 else "white",
                    fontweight="bold")

    plt.suptitle("Model Comparison Heatmap\n(sorted by PSNR, Validation/F0, 200 frames)",
                 y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig4_summary_heatmap.{ext}")
    plt.close()
    print(f"  fig4 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Stage 2/3 historical: 2D vs 2.5D pilot results
# ══════════════════════════════════════════════════════════════════════════════

def fig_historical_2d_2p5d(out_dir: Path):
    """Reproduce Stage 2/3 comparison from known results."""
    # From STAGE2_EXTENDED_REPORT.md and STAGE3_REPORT.md
    hist_data = {
        "Baseline 2D\n(Stage-2 pilot)": {"psnr": 27.41, "ssim": 0.719, "ms_ssim": 0.688, "delta_psnr": 1.00},
        "2.5D w=3\n(Stage-2 pilot)":    {"psnr": 29.70, "ssim": 0.785, "ms_ssim": 0.750, "delta_psnr": 3.33},
        "2.5D w=5\n(Stage-2 pilot)":    {"psnr": 28.94, "ssim": 0.769, "ms_ssim": 0.731, "delta_psnr": 2.59},
        "Baseline 2D\n(Stage-1,DD)":    {"psnr": 33.82, "ssim": 0.945, "ms_ssim": 0.963, "delta_psnr": 14.30},
        "2.5D w=3\n(Stage-3 full)":     {"psnr": 25.13, "ssim": 0.639, "ms_ssim": 0.594, "delta_psnr": -0.53},
    }

    models = list(hist_data.keys())
    colors_hist = ["#4878cf", "#6acc65", "#a8d878", "#d65f5f", "#e0b040"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("delta_psnr", "Δ PSNR (dB)")]

    for ax, (key, ylabel) in zip(axes, metrics):
        vals = [hist_data[m][key] for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors_hist, alpha=0.85, width=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Historical Results — {ylabel}")
        if key == "delta_psnr":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        for i, v in enumerate(vals):
            ax.text(i, v + (abs(v) * 0.01 + 0.1), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    plt.suptitle("Stage 1–3 Historical Results: 2D Baseline and 2.5D Temporal Context",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig5_historical_2d_2p5d.{ext}")
    plt.close()
    print(f"  fig5 saved")


def fig_k_sweep_psnr(rows: List[Dict], out_dir: Path):
    """PSNR-vs-K curves for sparse modes and algorithm variants."""
    series = {}
    for r in rows:
        algo = r["algorithm"]
        if algo == "input":
            continue
        mode = r["sparse_mode"]
        if mode not in ("xy", "z", "xyz"):
            continue
        run = r["run_name"]
        k = None
        if "_K" in run:
            try:
                k = int(run.split("_K")[1].split("_")[0])
            except Exception:
                k = None
        if k is None:
            continue
        extras = r.get("extras", "") or ""
        tag = f"{algo}"
        if extras:
            tag = f"{algo}+{extras}"
        key = (mode, tag)
        series.setdefault(key, []).append((k, r["psnr_mean"]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    mode_order = ["xy", "z", "xyz"]
    for ax, mode in zip(axes, mode_order):
        keys = sorted([k for k in series.keys() if k[0] == mode], key=lambda x: x[1])
        for _, tag in keys:
            pts = sorted(series[(mode, tag)], key=lambda x: x[0])
            ks = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if len(ks) < 2:
                continue
            ax.plot(ks, ys, marker="o", linewidth=1.6, label=tag)
        ax.set_title(f"{mode.upper()} sparse")
        ax.set_xlabel("Sparsity factor K")
        ax.set_xticks([2, 3, 4])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[-1].legend(fontsize=7, loc="best")
    plt.suptitle("K Sweep: PSNR vs K", y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig6_k_sweep_psnr.{ext}")
    plt.close()
    print("  fig6 saved")


def fig_optimization_gain(rows: List[Dict], out_dir: Path):
    """Show gains versus same algo/mode/K baseline (extras='')."""
    base = {}
    gains = []
    for r in rows:
        if r["algorithm"] == "input":
            continue
        run = r["run_name"]
        if "_K" not in run:
            continue
        try:
            k = int(run.split("_K")[1].split("_")[0])
        except Exception:
            continue
        key = (r["algorithm"], r["sparse_mode"], k)
        extras = r.get("extras", "") or ""
        if extras == "":
            base[key] = r["psnr_mean"]
    for r in rows:
        if r["algorithm"] == "input":
            continue
        run = r["run_name"]
        extras = r.get("extras", "") or ""
        if extras == "" or "_K" not in run:
            continue
        try:
            k = int(run.split("_K")[1].split("_")[0])
        except Exception:
            continue
        key = (r["algorithm"], r["sparse_mode"], k)
        if key not in base:
            continue
        gains.append({
            "label": f"{r['algorithm']} {r['sparse_mode']} K{k} +{extras}",
            "gain": r["psnr_mean"] - base[key],
        })
    gains = sorted(gains, key=lambda x: x["gain"], reverse=True)[:12]
    if not gains:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(gains))))
    y = np.arange(len(gains))
    vals = [g["gain"] for g in gains]
    labels = [g["label"] for g in gains]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("PSNR gain (dB) over same algo/mode/K baseline")
    ax.set_title("Optimization Gain Ranking")
    for i, v in enumerate(vals):
        ax.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"fig7_optimization_gain.{ext}")
    plt.close()
    print("  fig7 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, default="eval_sparse3d")
    p.add_argument("--output_dir", type=str, default="eval_sparse3d/figures")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading summary data...")
    rows = load_summary(eval_dir)
    print(f"  {len(rows)} rows loaded")

    print("\nGenerating figures...")
    fig_phase1_grouped_bar(rows, out_dir)
    fig_phase2_psf_ablation(rows, out_dir)
    fig_perframe_psnr(eval_dir, out_dir)
    fig_summary_heatmap(rows, out_dir)
    fig_historical_2d_2p5d(out_dir)
    fig_k_sweep_psnr(rows, out_dir)
    fig_optimization_gain(rows, out_dir)

    print(f"\nAll figures saved to: {out_dir}")
    print("Files:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

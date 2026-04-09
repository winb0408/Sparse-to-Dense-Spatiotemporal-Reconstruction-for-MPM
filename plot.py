"""
visualize.py  (v2 — publication-grade)
========================================
Publication-quality composite figure generator for the 3PM reconstruction
pipeline.  Targets Nature sub-journals, Small Methods, Photonix.

Physical constraints observed
------------------------------
  Single column  : 88  mm  = 3.46 in
  Double column  : 180 mm  = 7.09 in  (Nature family standard)
  Font           : Arial (Helvetica fallback), 7 pt body, 8 pt panel labels
  Line weight    : axes 0.5 pt, data lines 0.8–1.2 pt
  Resolution     : 300 DPI raster; PDF/SVG for vector press-ready output
  Colourmap      : inferno  (fluorescence);  RdBu_r  (signed error)
  Colour palette : Wong / IBM colorblind-safe 8-colour cycle

Figure panels
-------------
Fig 1  Qualitative reconstruction
       (a) Noisy input  |  (b) Reconstruction  |  (c) GT
       Magnified ROI insets with connecting lines (zoom-indicator style).
       Shared joint colorscale; scale bar; per-panel PSNR / SSIM annotation.

Fig 2  Quantitative error analysis
       (a) Absolute |Pred−GT| map
       (b) Signed   Pred−GT   map  (diverging RdBu)
       (c) Error histogram with MAE / RMSE callout
       (d) Horizontal line profile: Input / Pred / GT with shaded ±1σ band

Fig 3  Frequency-domain fidelity
       (a–c) 2-D log-power spectrum: Input / Pred / GT
       (d)   Radially-averaged PSD (log-log)  with noise-floor annotation
       (e)   FwPSNR / sSNR bar summary (single numbers, formatted for panel)

Fig 4  Ablation comparison  (requires ablation_all.csv from eval.py)
       (a) Box + violin for PSNR across runs
       (b) Box + violin for SSIM across runs
       (c) Bar chart (mean ± 95 % CI) for all core metrics
       Significance brackets with Bonferroni-corrected p-values sourced from
       ablation_pvalues.csv if present, else re-computed inline.

Fig 5  Training dynamics  (reads TensorBoard event files)
       (a) Loss curves (train / val) with epoch of best val highlighted
       (b) PSNR curves (train / val)
       (c) LR schedule

Usage
-----
# Figures 1–3 (single-image, qualitative + error + spectral):
python visualize.py --mode single \\
    --pred_npy  eval/images/full/00044628_pred.npy \\
    --gt_npy    eval/images/full/00044628_gt.npy \\
    --noisy_npy eval/images/full/00044628_noisy.npy \\
    --roi 200 300 128 128 \\
    --pixel_size_um 0.103 \\
    --output_dir figures/ --fmt pdf

# Figure 4 (ablation comparison):
python visualize.py --mode ablation \\
    --ablation_csv  eval/ablation_all.csv \\
    --pvalues_csv   eval/ablation_pvalues.csv \\
    --output_dir figures/

# Figure 5 (training curves):
python visualize.py --mode training \\
    --tensorboard_dir runs/baseline_minmax \\
    --output_dir figures/

# All figures:
python visualize.py --mode all \\
    --pred_npy ... --gt_npy ... --noisy_npy ... \\
    --ablation_csv ... --tensorboard_dir ... \\
    --output_dir figures/
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                           # headless / cluster-safe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage, signal as _signal

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


# ══════════════════════════════════════════════════════════════════════════════
# Style constants
# ══════════════════════════════════════════════════════════════════════════════

# Nature double-column width in inches
_DW  = 7.09    # 180 mm
_SW  = 3.46    # 88  mm
_CM  = 1 / 2.54

# Wong + IBM colorblind-safe 8-colour cycle  (in order of use)
_PALETTE = [
    "#0072B2",   # blue
    "#E69F00",   # orange
    "#009E73",   # green
    "#D55E00",   # vermillion
    "#CC79A7",   # reddish purple
    "#56B4E9",   # sky blue
    "#F0E442",   # yellow
    "#999999",   # grey
]

# Colormaps
_CM_FLUOR   = "inferno"     # fluorescence signal
_CM_ERR_ABS = "hot"         # absolute error
_CM_ERR_SGN = "RdBu_r"     # signed error (diverging)
_CM_PSD     = "viridis"     # power spectrum


def _nature_rcparams() -> Dict:
    """Return a dict of matplotlib rcParams targeting Nature submission standards."""
    return {
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "Helvetica Neue", "Helvetica",
                                  "Liberation Sans", "DejaVu Sans"],
        "font.size":             7,
        "axes.labelsize":        7,
        "axes.titlesize":        7,
        "axes.titlepad":         3,
        "axes.labelpad":         2,
        "axes.linewidth":        0.5,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "xtick.labelsize":       6,
        "ytick.labelsize":       6,
        "xtick.major.width":     0.5,
        "ytick.major.width":     0.5,
        "xtick.major.size":      2.5,
        "ytick.major.size":      2.5,
        "xtick.minor.width":     0.3,
        "ytick.minor.width":     0.3,
        "lines.linewidth":       0.9,
        "patch.linewidth":       0.5,
        "legend.fontsize":       6,
        "legend.title_fontsize": 6,
        "legend.frameon":        False,
        "legend.handlelength":   1.5,
        "figure.dpi":            300,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.02,
        "pdf.fonttype":          42,      # embed fonts for press-ready PDF
        "ps.fonttype":           42,
        "axes.prop_cycle":       plt.cycler(color=_PALETTE),
    }


def apply_style():
    """Apply Nature-standard rcParams globally."""
    plt.rcParams.update(_nature_rcparams())


# ══════════════════════════════════════════════════════════════════════════════
# Shared axis utilities
# ══════════════════════════════════════════════════════════════════════════════

_PANEL_ALPHA = list("abcdefghijklmnopqrstuvwxyz")


def label_panel(ax, label: str, x: float = -0.08, y: float = 1.08,
                fontsize: int = 8, fontweight: str = "bold"):
    """
    Add bold lower-case panel label slightly outside the top-left corner.

    The default position is tuned to reduce overlap with image content,
    especially for tight layouts.
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va="top", ha="right", clip_on=False)


def remove_axes(ax):
    """Remove all ticks, labels, and spines from an image axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def add_colorbar(ax, im, label: str = "", fontsize: int = 6,
                 width: str = "3%", pad: float = 0.03,
                 n_ticks: int = 3):
    """Attach a minimal colorbar to the right of an image axis."""
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size=width, pad=pad)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fontsize, labelpad=2)
    cb.ax.tick_params(labelsize=fontsize - 1, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.locator = mticker.MaxNLocator(nbins=n_ticks)
    cb.update_ticks()


def add_scalebar(ax, pixel_size_um: float, bar_um: float = 50,
                 loc: str = "lower right", color: str = "white",
                 fontsize: int = 5.5):
    """
    Draw a physical scale bar directly onto an image axis.

    Parameters
    ----------
    pixel_size_um : µm per pixel (spatial calibration)
    bar_um        : desired bar length in µm
    loc           : 'lower right' | 'lower left' | 'upper right' | 'upper left'
    """
    bar_px = bar_um / pixel_size_um
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()          # may be inverted for images
    W = abs(xr - xl)
    H = abs(yt - yb)
    mx, my = 0.04 * W, 0.05 * H
    th = max(1.5, H * 0.012)        # bar thickness in pixels

    if "right" in loc:
        x1 = max(xl, xr) - mx;  x0 = x1 - bar_px
    else:
        x0 = min(xl, xr) + mx;  x1 = x0 + bar_px

    yref = (min(yb, yt) + my) if "lower" in loc else (max(yb, yt) - my)
    # Bars are drawn in data coordinates so they scale correctly on zoom
    ax.add_patch(Rectangle((x0, yref - th / 2), bar_px, th,
                            linewidth=0, facecolor=color, zorder=5,
                            transform=ax.transData))
    # Place the text above (for lower loc) or below (for upper loc) the bar
    if "lower" in loc:
        text_y = yref - th * 3.0
        va = "top"
    else:
        text_y = yref + th * 3.0
        va = "bottom"
    ax.text((x0 + x1) / 2, text_y, f"{bar_um:.0f} µm",
            color=color, fontsize=fontsize, ha="center", va=va,
            transform=ax.transData, zorder=5)


def _annotate_metric(ax, text: str, loc: str = "lower right",
                     fontsize: int = 5.5):
    """Overlay a metric value string on an image axis (dark semi-transparent box)."""
    ha = "right" if "right" in loc else "left"
    va = "bottom" if "lower" in loc else "top"
    x  = 0.97 if "right" in loc else 0.03
    y  = 0.03 if "lower" in loc else 0.97
    ax.text(x, y, text, color="white", fontsize=fontsize,
            ha=ha, va=va, transform=ax.transAxes, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="black",
                      alpha=0.5, lw=0))


def _draw_zoom_connector(ax_full, ax_inset, roi: Tuple[int, int, int, int],
                         fig, color: str = "yellow", lw: float = 0.6):
    """
    Draw a ROI rectangle on ax_full and two connecting lines to ax_inset,
    replicating the zoom-indicator style used in Nature Methods imaging papers.

    roi = (y0, x0, h, w)  in pixel coordinates of ax_full.
    """
    y0, x0, rh, rw = roi
    # Rectangle on full image
    ax_full.add_patch(Rectangle((x0, y0), rw, rh,
                                 linewidth=lw, edgecolor=color,
                                 facecolor="none", zorder=4))

    # Two diagonal connectors (top-right corner of ROI → inset)
    for (src_x, src_y), (dst_relx, dst_rely) in [
        ((x0 + rw, y0),      (0.0, 1.0)),   # top-right → inset top-left
        ((x0 + rw, y0 + rh), (0.0, 0.0)),   # bot-right → inset bot-left
    ]:
        con = ConnectionPatch(
            xyA=(src_x, src_y), coordsA=ax_full.transData,
            xyB=(dst_relx, dst_rely), coordsB=ax_inset.transAxes,
            color=color, lw=lw, linestyle="--", zorder=3,
        )
        fig.add_artist(con)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Qualitative reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def figure_qualitative(
    noisy:         np.ndarray,      # [H, W]  float32 ∈ [0,1]
    pred:          np.ndarray,
    gt:            np.ndarray,
    roi:           Optional[Tuple[int, int, int, int]] = None,  # (y0, x0, h, w)
    pixel_size_um: float = 0.1,
    sample_name:   str   = "",
    psnr_pred:     Optional[float] = None,
    ssim_pred:     Optional[float] = None,
    psnr_noisy:    Optional[float] = None,
    output_path:   Optional[Path]  = None,
) -> plt.Figure:
    """
    Three-panel (Input | Prediction | GT) qualitative reconstruction figure.

    When roi is provided, a second row shows magnified insets with
    zoom-indicator connecting lines (Nature Methods style).

    Layout (with ROI):
    ┌──────────┬──────────┬──────────┐   Row 0: full images
    │ (a) Inp  │ (b) Pred │ (c) GT   │
    ├──────────┼──────────┼──────────┤   Row 1: ROI insets
    │ (d) inp↑ │ (e) pre↑ │ (f) gt↑  │
    └──────────┴──────────┴──────────┘
    """
    apply_style()

    has_roi = roi is not None
    n_rows  = 2 if has_roi else 1
    # Double-column figure; height scales with rows and aspect ratio
    aspect  = gt.shape[0] / gt.shape[1]
    fig_w   = _DW
    panel_h = fig_w / 3 * aspect   # height of one image panel
    fig_h   = panel_h * n_rows + (0.25 if has_roi else 0.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_rows, 3, figure=fig,
        wspace=0.04, hspace=0.12 if has_roi else 0.04,
    )

    images = [noisy, pred, gt]
    titles = ["Input (noisy)", "Reconstruction", "Ground truth"]
    vmin   = min(a.min() for a in images)
    vmax   = max(a.max() for a in images)

    top_axes = []
    last_im = None
    for col, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(img, cmap=_CM_FLUOR, vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        remove_axes(ax)
        ax.set_title(title, pad=3)
        label_panel(ax, _PANEL_ALPHA[col])
        top_axes.append(ax)
        last_im = im

    # Single shared colorbar for the top row, placed outside the grid on the right,
    # so that all three image panels retain identical sizes.
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.15 if has_roi else 0.12, 0.015, 0.7 if has_roi else 0.75])
        cb  = plt.colorbar(last_im, cax=cax)
        cb.set_label("Intensity (a.u.)", fontsize=6, labelpad=2)
        cb.ax.tick_params(labelsize=5, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)

        # Metric annotation: PSNR on pred, PSNR on noisy
        if col == 1 and psnr_pred is not None:
            txt = f"PSNR = {psnr_pred:.2f} dB"
            if ssim_pred is not None:
                txt += f"\nSSIM = {ssim_pred:.4f}"
            _annotate_metric(ax, txt, "lower right")
        if col == 0 and psnr_noisy is not None:
            _annotate_metric(ax, f"PSNR = {psnr_noisy:.2f} dB", "lower right")

    # Scale bar on GT (col 2) — move to lower-left to avoid overlapping metrics
    add_scalebar(top_axes[2], pixel_size_um=pixel_size_um,
                 bar_um=50, loc="lower left")

    # ── ROI insets ─────────────────────────────────────────────────────────
    if has_roi:
        y0, x0, rh, rw = roi
        inset_axes_list = []
        for col, img in enumerate(images):
            ax_ins = fig.add_subplot(gs[1, col])
            ax_ins.imshow(img[y0:y0+rh, x0:x0+rw],
                          cmap=_CM_FLUOR, vmin=vmin, vmax=vmax,
                          interpolation="nearest", aspect="equal")
            remove_axes(ax_ins)
            label_panel(ax_ins, _PANEL_ALPHA[3 + col])
            zoom = gt.shape[1] / rw
            ax_ins.set_title(f"×{zoom:.0f}", pad=2, fontsize=6)
            inset_axes_list.append(ax_ins)

            _draw_zoom_connector(top_axes[col], ax_ins, roi, fig)

        # Scale bar in inset (physical length adjusted for magnification)
        add_scalebar(inset_axes_list[2],
                     pixel_size_um=pixel_size_um,
                     bar_um=max(5, 50 / zoom),
                     loc="lower left")

    if sample_name:
        fig.suptitle(sample_name, fontsize=6.5, y=1.005)

    if output_path:
        fig.savefig(str(output_path), dpi=300)
        # Also save a PNG alongside the requested format for convenience
        if output_path.suffix.lower() != ".png":
            png_path = output_path.with_suffix(".png")
            fig.savefig(str(png_path), dpi=300)
        print(f"[fig1] → {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Quantitative error analysis
# ══════════════════════════════════════════════════════════════════════════════

def figure_error_analysis(
    pred:          np.ndarray,
    gt:            np.ndarray,
    noisy:         Optional[np.ndarray] = None,
    profile_row:   Optional[int]        = None,
    pixel_size_um: float                = 0.1,
    sample_name:   str                  = "",
    output_path:   Optional[Path]       = None,
    compute_fwhm:  bool                 = False,
) -> plt.Figure:
    """
    Four-panel error analysis.

    (a) |Pred − GT| absolute error map
    (b)  Pred − GT  signed error map (diverging RdBu)
    (c) Error histogram (density) with MAE / RMSE callout
    (d) Horizontal line profile through the image with ±1 std noise band
    """
    apply_style()

    err_sgn = (pred - gt).astype(np.float32)
    err_abs = np.abs(err_sgn)

    if profile_row is None:
        profile_row = gt.shape[0] // 2

    # Arrange panels in a 2×2 grid to avoid horizontally stretched plots
    fig = plt.figure(figsize=(_DW, _DW * 0.60))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35)

    # ── (a) Absolute error ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    e99  = float(np.percentile(err_abs, 99.5))
    im_a = ax_a.imshow(err_abs, cmap=_CM_ERR_ABS, vmin=0, vmax=e99,
                       interpolation="nearest")
    remove_axes(ax_a)
    ax_a.set_title("|Pred − GT|")
    label_panel(ax_a, "a")
    add_colorbar(ax_a, im_a, label="Abs. error")
    ax_a.axhline(profile_row, color="cyan", lw=0.7, ls="--", zorder=3)

    # ── (b) Signed error ──────────────────────────────────────────────────────
    ax_b  = fig.add_subplot(gs[0, 1])
    sym   = float(np.percentile(np.abs(err_sgn), 99.5))
    im_b  = ax_b.imshow(err_sgn, cmap=_CM_ERR_SGN, vmin=-sym, vmax=sym,
                        interpolation="nearest")
    remove_axes(ax_b)
    ax_b.set_title("Pred − GT")
    label_panel(ax_b, "b")
    add_colorbar(ax_b, im_b, label="Signed error")
    ax_b.axhline(profile_row, color="white", lw=0.7, ls="--", zorder=3)

    # ── (c) Error histogram ───────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    counts, bin_edges = np.histogram(err_sgn.flatten(), bins=150, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax_c.fill_between(centers, counts, alpha=0.6,
                      color=_PALETTE[0], step="mid")
    ax_c.plot(centers, counts, color=_PALETTE[0], lw=0.8)
    ax_c.axvline(0, color="black", lw=0.7, ls="--")
    ax_c.set_xlabel("Signed error")
    ax_c.set_ylabel("Density")
    ax_c.set_title("Error distribution")
    label_panel(ax_c, "c")
    mae_val  = float(err_abs.mean())
    rmse_val = float(np.sqrt(np.mean(err_sgn**2)))
    ax_c.text(0.97, 0.95,
              f"MAE  = {mae_val:.4f}\nRMSE = {rmse_val:.4f}",
              transform=ax_c.transAxes, fontsize=5.5, ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, lw=0.4))

    # ── (d) Horizontal line profile ──────────────────────────────────────────
    ax_d  = fig.add_subplot(gs[1, 1])
    x_um  = np.arange(gt.shape[1]) * pixel_size_um

    def _plot_profile(arr, label, color, ls="-", alpha=1.0):
        profile = arr[profile_row].astype(float)
        ax_d.plot(x_um, profile, color=color, ls=ls, lw=0.85,
                  alpha=alpha, label=label)

    if noisy is not None:
        _plot_profile(noisy, "Input", _PALETTE[7], ls="--", alpha=0.65)
    _plot_profile(gt,   "GT",   _PALETTE[2])
    _plot_profile(pred, "Pred", _PALETTE[0])

    # Shaded region: ±1 std of abs error along the profile row
    err_row = err_abs[profile_row].astype(float)
    gt_row  = gt[profile_row].astype(float)
    ax_d.fill_between(x_um, gt_row - err_row, gt_row + err_row,
                      color=_PALETTE[0], alpha=0.15, label="±|err|")

    # ── Optional: FWHM estimation for main peaks (GT vs Pred) ────────────────
    if compute_fwhm:
        try:
            gt_row  = gt[profile_row].astype(float)
            pred_row = pred[profile_row].astype(float)
            # Detect peaks on GT profile
            if gt_row.max() > 0:
                peaks, _ = _signal.find_peaks(gt_row, prominence=0.25 * gt_row.max())
            else:
                peaks = np.array([], dtype=int)
            if peaks.size > 0:
                # Limit to top 3 peaks by height for stability
                heights = gt_row[peaks]
                order   = np.argsort(heights)[::-1][:3]
                peaks   = peaks[order]
                # widths[0] are FWHM in samples; widths[2:], [3:] are left/right bases
                w_gt = _signal.peak_widths(gt_row, peaks, rel_height=0.5)
                w_pr = _signal.peak_widths(pred_row, peaks, rel_height=0.5)
                # Convert to µm and average over peaks
                fwhm_gt_um   = float(np.mean(w_gt[0]) * pixel_size_um)
                fwhm_pred_um = float(np.mean(w_pr[0]) * pixel_size_um)

                # Visual markers for the first GT / Pred peak FWHM
                for (lo, hi) in zip(w_gt[2], w_gt[3]):
                    ax_d.axvspan(
                        x_um[int(lo)], x_um[min(int(hi), len(x_um)-1)],
                        color=_PALETTE[2], alpha=0.10, lw=0
                    )
                for (lo, hi) in zip(w_pr[2], w_pr[3]):
                    ax_d.axvspan(
                        x_um[int(lo)], x_um[min(int(hi), len(x_um)-1)],
                        color=_PALETTE[0], alpha=0.08, lw=0
                    )

                ax_d.text(
                    0.03, 0.97,
                    f"FWHM GT   ≈ {fwhm_gt_um:.2f} µm\n"
                    f"FWHM Pred ≈ {fwhm_pred_um:.2f} µm",
                    transform=ax_d.transAxes,
                    fontsize=5.5,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", alpha=0.8, lw=0.4),
                )
        except Exception as exc:  # noqa: F841
            # FWHM is optional; failures should not break the figure
            pass

    ax_d.set_xlabel("Position (µm)")
    ax_d.set_ylabel("Intensity (a.u.)")
    ax_d.set_title(f"Line profile  (y = {profile_row} px)")
    ax_d.legend(loc="upper right", fontsize=5.5)
    label_panel(ax_d, "d")

    if sample_name:
        fig.suptitle(sample_name, fontsize=6.5, y=1.005)
    if output_path:
        fig.savefig(str(output_path), dpi=300)
        if output_path.suffix.lower() != ".png":
            png_path = output_path.with_suffix(".png")
            fig.savefig(str(png_path), dpi=300)
        print(f"[fig2] → {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Test-set statistical distributions & tolerance curves
# ══════════════════════════════════════════════════════════════════════════════

def figure_statistics_single_run(
    metrics_csv: Path,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Test-set statistical distributions for a single checkpoint (Figure 3).

    Panels:
      (a) PSNR across images — violin + box
      (b) SSIM across images — violin + box

    The input CSV is metrics_{run}.csv produced by eval.py, containing one
    row per test image with at least 'psnr' and 'ssim' columns.
    """
    apply_style()

    df = pd.read_csv(str(metrics_csv))
    if "psnr" not in df.columns or "ssim" not in df.columns:
        raise ValueError(f"'psnr' or 'ssim' missing in {metrics_csv}")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(_DW, _DW * 0.30),
        gridspec_kw={"wspace": 0.35},
    )

    def _one_metric(ax, values: np.ndarray, metric: str, ylabel: str, panel_label: str):
        vals = pd.Series(values).dropna()
        if not len(vals):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=6)
            return

        if _HAS_SNS:
            # Build a small DataFrame to satisfy seaborn API
            dd = pd.DataFrame({metric: vals.values, "dummy": ["run"] * len(vals)})
            sns.violinplot(
                x="dummy", y=metric, data=dd,
                color=_PALETTE[0], inner="box",
                cut=0, saturation=0.85, linewidth=0.6, ax=ax,
            )
            ax.set_xticks([])
        else:
            ax.violinplot(vals.values, showmeans=False, showmedians=False)
            ax.boxplot(vals.values, widths=0.25,
                       medianprops=dict(color="k", lw=1.0),
                       whiskerprops=dict(lw=0.6, color="#444"),
                       capprops=dict(lw=0.6, color="#444"))
            ax.set_xticks([])

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Test images")
        label_panel(ax, panel_label)
        ax.set_title("Distribution across test set")

    _one_metric(axes[0], df["psnr"].values, "psnr", "PSNR (dB)", "a")
    _one_metric(axes[1], df["ssim"].values, "ssim", "SSIM", "b")

    if output_path:
        fig.savefig(str(output_path), dpi=300)
        print(f"[fig3_stats] → {output_path}")
    return fig


def figure_tolerance_curve(
    sparsity_factors: Sequence[float],
    psnr_mean: Sequence[float],
    psnr_std: Optional[Sequence[float]] = None,
    ssim_mean: Optional[Sequence[float]] = None,
    ssim_std: Optional[Sequence[float]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Tolerance limit curves (error-bar line plots) for PSNR / SSIM vs sparsity.

    Parameters
    ----------
    sparsity_factors : list-like
        Sparsity factors, e.g. [2, 4, 8, 16].
    psnr_mean, psnr_std : list-like
        Mean PSNR (dB) and optional standard deviation or CI half-width.
    ssim_mean, ssim_std : list-like, optional
        Mean SSIM and optional standard deviation or CI half-width.

    This function is intentionally decoupled from eval.py — you can compute
    the statistics however you like (e.g. from multiple runs), then pass the
    aggregated values here to obtain a clean publication-ready panel.
    """
    apply_style()

    x = np.asarray(sparsity_factors, dtype=float)
    psnr_mean = np.asarray(psnr_mean, dtype=float)
    psnr_std  = np.zeros_like(psnr_mean) if psnr_std is None else np.asarray(psnr_std)

    fig, ax1 = plt.subplots(figsize=(_SW, _SW * 0.70))

    # PSNR on left y-axis
    ax1.errorbar(
        x, psnr_mean, yerr=psnr_std,
        marker="o", ms=3.0,
        color=_PALETTE[0], lw=0.9,
        capsize=2.0, label="PSNR",
    )
    ax1.set_xlabel("Sparsity factor")
    ax1.set_ylabel("PSNR (dB)", color=_PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=_PALETTE[0])
    ax1.set_xscale("log", basex=2)
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):d}×")
    )

    # Optional SSIM on right y-axis
    if ssim_mean is not None:
        ssim_mean = np.asarray(ssim_mean, dtype=float)
        ssim_std  = np.zeros_like(ssim_mean) if ssim_std is None else np.asarray(ssim_std)
        ax2 = ax1.twinx()
        ax2.errorbar(
            x, ssim_mean, yerr=ssim_std,
            marker="s", ms=2.8,
            color=_PALETTE[2], lw=0.9,
            capsize=2.0, label="SSIM",
        )
        ax2.set_ylabel("SSIM", color=_PALETTE[2])
        ax2.tick_params(axis="y", labelcolor=_PALETTE[2])

        # Shared legend
        lines  = ax1.get_lines() + ax2.get_lines()
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, fontsize=6, loc="best")
    else:
        ax1.legend(fontsize=6, loc="best")

    label_panel(ax1, "a")
    ax1.set_title("Tolerance limit curve (PSNR / SSIM vs sparsity)")

    if output_path:
        fig.savefig(str(output_path), dpi=300)
        print(f"[fig3_tolerance] → {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Frequency-domain fidelity
# ══════════════════════════════════════════════════════════════════════════════

def _radial_psd(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectral density.  Returns (freq, power)."""
    H, W   = img.shape
    fft    = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    psd2d  = np.abs(fft) ** 2
    cy, cx = H // 2, W // 2
    y, x   = np.mgrid[-cy:H-cy, -cx:W-cx]
    r      = np.sqrt(x**2 + y**2).astype(int)
    r_max  = min(cy, cx)
    power  = np.array([psd2d[r == ri].mean() if (r == ri).any() else 0
                       for ri in range(r_max)])
    freq   = np.arange(r_max) / max(H, W)
    return freq, power


def figure_frequency(
    noisy:       np.ndarray,
    pred:        np.ndarray,
    gt:          np.ndarray,
    sample_name: str          = "",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Five-panel frequency-domain analysis.

    (a–c) 2-D log-power spectrum for Input / Pred / GT
    (d)   Radially-averaged PSD  (log-log)  with noise-floor annotation
    (e)   sSNR and FwPSNR bar summary (Input vs Pred baseline)
    """
    apply_style()

    # Use a 2×3 layout: three 2-D PSDs on the first row,
    # radial PSD and bar chart on the second row to avoid crowding.
    fig = plt.figure(figsize=(_DW, _DW * 0.50))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.35)

    def _log_psd(img: np.ndarray) -> np.ndarray:
        fft = np.fft.fftshift(np.fft.fft2(img - img.mean()))
        return np.log10(np.abs(fft) ** 2 + 1e-10)

    # (a–c): 2-D PSD panels (first row)
    for col, (img, lbl) in enumerate([(noisy, "Input"), (pred, "Pred"), (gt, "GT")]):
        ax  = fig.add_subplot(gs[0, col])
        psd = _log_psd(img)
        vlo = float(np.percentile(psd, 2))
        vhi = float(np.percentile(psd, 99))
        ax.imshow(psd, cmap=_CM_PSD, vmin=vlo, vmax=vhi, interpolation="nearest")
        remove_axes(ax)
        ax.set_title(lbl)
        label_panel(ax, _PANEL_ALPHA[col])

    # (d): Radial PSD curves (second row, spanning two columns)
    ax_d = fig.add_subplot(gs[1, 0:2])
    for img, lbl, col, ls, alpha in [
        (noisy, "Input", _PALETTE[7], "-", 0.6),
        (pred,  "Pred",  _PALETTE[0], "-", 0.85),
        (gt,    "GT",    _PALETTE[2], "--", 0.85),
    ]:
        freq, power = _radial_psd(img)
        ax_d.semilogy(freq[1:], power[1:], color=col, lw=0.9, ls=ls, alpha=alpha, label=lbl)

    ax_d.set_xlabel("Spatial freq. (cyc/px)")
    ax_d.set_ylabel("Power (a.u.)")
    ax_d.set_title("Radial PSD")
    ax_d.legend(fontsize=5.5)
    label_panel(ax_d, "d")

    # (e): sSNR / FwPSNR bar chart (second row, right column)
    from eval import _spectral_snr, _fw_psnr   # reuse from eval.py
    ax_e  = fig.add_subplot(gs[1, 2])
    labs  = ["sSNR\n(dB)", "FwPSNR\n(dB)"]
    inp_v = [_spectral_snr(noisy, gt), _fw_psnr(noisy, gt)]
    pre_v = [_spectral_snr(pred,  gt), _fw_psnr(pred,  gt)]
    x     = np.arange(2)
    w     = 0.3
    ax_e.bar(x - w/2, inp_v, width=w, label="Input", color=_PALETTE[7])
    ax_e.bar(x + w/2, pre_v, width=w, label="Pred",  color=_PALETTE[0])
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(labs)
    ax_e.set_ylabel("Score (dB)")
    ax_e.set_title("Spectral metrics")
    ax_e.legend(fontsize=5.5)
    ax_e.axhline(0, color="k", lw=0.5, ls="--")
    label_panel(ax_e, "e")

    if sample_name:
        fig.suptitle(sample_name, fontsize=6.5, y=1.005)
    if output_path:
        fig.savefig(str(output_path), dpi=300)
        if output_path.suffix.lower() != ".png":
            png_path = output_path.with_suffix(".png")
            fig.savefig(str(png_path), dpi=300)
        print(f"[fig3] → {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Ablation comparison
# ══════════════════════════════════════════════════════════════════════════════

def _sig_bracket(ax, x1: float, x2: float, y: float, pval: float,
                 h: float = 0.01, fontsize: int = 5.5, color: str = "k"):
    """
    Draw a significance bracket between x1 and x2 at height y.

    p-value display:
        p < 0.001  → ***
        p < 0.01   → **
        p < 0.05   → *
        else       → n.s.
    """
    if math.isnan(pval):
        return
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=0.6)
    ax.text((x1 + x2) / 2, y + h * 1.2, stars, ha="center",
            va="bottom", fontsize=fontsize, color=color)


def figure_ablation(
    ablation_csv:  Path,
    pvalues_csv:   Optional[Path] = None,
    metrics:       Sequence[str]  = ("psnr", "ssim", "ms_ssim", "lpips"),
    output_path:   Optional[Path] = None,
) -> plt.Figure:
    """
    Two-row ablation comparison.

    Row 0: box + violin plots for PSNR (left) and SSIM (right).
    Row 1: bar chart (mean ± 95% CI bootstrap) for all core metrics.
           Significance brackets sourced from pvalues_csv if present.
    """
    apply_style()

    df    = pd.read_csv(str(ablation_csv))
    runs  = df["run"].unique().tolist()
    n     = len(runs)
    pal   = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(runs)}

    # Load p-values if available
    pval_lookup: Dict[Tuple[str, str, str], float] = {}
    if pvalues_csv and Path(pvalues_csv).exists():
        pv = pd.read_csv(str(pvalues_csv))
        for _, row in pv.iterrows():
            pval_lookup[(row["metric"], row["comparison"])] = float(row["p_value"])

    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError(f"None of the requested metrics found in {ablation_csv}.")

    metric_labels = {
        "psnr":       "PSNR (dB) ↑",
        "ssim":       "SSIM ↑",
        "ms_ssim":    "MS-SSIM ↑",
        "lpips":      "LPIPS ↓",
        "mae":        "MAE ↓",
        "ncc":        "NCC ↑",
        "cnr":        "CNR ↑",
        "s_snr":      "sSNR (dB) ↑",
        "fw_psnr":    "FwPSNR (dB) ↑",
        "delta_psnr": "ΔPSNR (dB) ↑",
    }

    fig = plt.figure(figsize=(_DW, _DW * 0.70))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.50)

    # ── Row 0: violin + box for top two metrics ────────────────────────────
    for col, metric in enumerate(metrics[:2]):
        ax = fig.add_subplot(gs[0, col])
        _draw_violin_box(ax, df, runs, metric, pal)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_xlabel("")
        label_panel(ax, _PANEL_ALPHA[col])
        if col == 0:
            ax.set_title("Per-sample distribution")

        # Add significance bracket between first run and best run
        means = [df[df["run"] == r][metric].mean() for r in runs]
        best  = int(np.argmax(means)) if "lpips" not in metric else int(np.argmin(means))
        if best != 0:
            ymax  = ax.get_ylim()[1]
            span  = ax.get_ylim()[1] - ax.get_ylim()[0]
            label = f"{runs[0]} vs {runs[best]}"
            pval  = pval_lookup.get((metric, label),
                    pval_lookup.get((metric, f"{runs[best]} vs {runs[0]}"), float("nan")))
            _sig_bracket(ax, 0, best, ymax + 0.02 * span, pval, h=0.025 * span)
            ax.set_ylim(top=ymax + 0.12 * span)

    # ── Row 1: bar chart for all metrics ──────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])
    _draw_metric_bars(ax_bar, df, runs, metrics, pal, metric_labels, pval_lookup)
    label_panel(ax_bar, "c")

    if output_path:
        fig.savefig(str(output_path), dpi=300)
        print(f"[fig4] → {output_path}")
    return fig


def _draw_violin_box(ax, df: pd.DataFrame, runs: List[str],
                     metric: str, pal: Dict):
    """Draw a violin + embedded box plot for one metric."""
    data = [df[df["run"] == r][metric].dropna().values for r in runs]

    if _HAS_SNS:
        sns.violinplot(data=df, x="run", y=metric, order=runs,
                       palette=pal, ax=ax, linewidth=0.5,
                       inner="box", scale="width", cut=0, saturation=0.85)
    else:
        parts = ax.violinplot(data, positions=range(len(runs)),
                              showmedians=False, showextrema=False)
        for pc, r in zip(parts["bodies"], runs):
            pc.set_facecolor(pal[r])
            pc.set_alpha(0.72)
        ax.boxplot(data, positions=range(len(runs)), widths=0.14,
                   patch_artist=False,
                   medianprops=dict(color="k", lw=1.0),
                   whiskerprops=dict(lw=0.5, color="#444"),
                   capprops=dict(lw=0.5, color="#444"),
                   flierprops=dict(marker="o", ms=1.5, alpha=0.35,
                                   markeredgewidth=0, color="#666"),
                   showfliers=True)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=30, ha="right", fontsize=5.5)


def _draw_metric_bars(
    ax, df: pd.DataFrame, runs: List[str], metrics: List[str],
    pal: Dict, labels: Dict, pval_lookup: Dict,
):
    """
    Grouped bar chart: runs (hue) × metrics (x-axis), mean ± 95 % CI.
    """
    from eval import bootstrap_ci as _bci   # reuse bootstrap CI

    n_metrics = len(metrics)
    n_runs    = len(runs)
    x_base    = np.arange(n_metrics)
    total_w   = 0.75
    w_single  = total_w / n_runs

    for ri, run in enumerate(runs):
        sub    = df[df["run"] == run]
        means, errs_lo, errs_hi = [], [], []
        for m in metrics:
            vals = sub[m].dropna().values
            mean = float(np.nanmean(vals)) if len(vals) else float("nan")
            lo, hi = _bci(vals)
            means.append(mean)
            errs_lo.append(abs(mean - lo))
            errs_hi.append(abs(hi - mean))

        x_pos = x_base + (ri - (n_runs - 1) / 2) * w_single
        ax.bar(x_pos, means,
               yerr=[errs_lo, errs_hi],
               width=w_single * 0.88,
               color=pal[run], label=run,
               linewidth=0,
               error_kw=dict(elinewidth=0.55, capthick=0.55, capsize=2.0, ecolor="k"))

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [labels.get(m, m) for m in metrics], rotation=28, ha="right", fontsize=5.5
    )
    ax.set_ylabel("Score")
    ax.set_title("Mean ± 95 % CI (bootstrap)")
    ax.legend(fontsize=5.5, loc="upper right",
              ncol=min(n_runs, 4), handlelength=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Training dynamics
# ══════════════════════════════════════════════════════════════════════════════

def _read_tb(log_dir: Path, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a scalar tag from TensorBoard event files.  Returns (steps, values)."""
    if not _HAS_TB:
        return np.array([]), np.array([])
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    evts = ea.Scalars(tag)
    return (np.array([e.step for e in evts]),
            np.array([e.value for e in evts]))


def figure_training(
    log_dir:     Path,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Three-panel training dynamics figure.

    (a) Total loss — train (solid) and val (dashed), best-val epoch marked.
    (b) PSNR — train and val.
    (c) Learning-rate schedule.
    """
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(_DW, _DW * 0.28),
                              gridspec_kw={"wspace": 0.40})

    tag_specs = [
        ("train/loss/total",  "val/loss/total",  "Total loss",  "a"),
        ("train/metric/psnr", "val/metric/psnr", "PSNR (dB)",  "b"),
        ("train/lr",          None,               "Learning rate", "c"),
    ]

    for ax, (train_tag, val_tag, ylabel, lbl) in zip(axes, tag_specs):
        ts, tv = _read_tb(log_dir, train_tag)
        if ts.size:
            ax.plot(ts, tv, color=_PALETTE[0], lw=0.9, label="Train")

        if val_tag:
            vs, vv = _read_tb(log_dir, val_tag)
            if vs.size:
                ax.plot(vs, vv, color=_PALETTE[1], lw=0.9, ls="--", label="Val")
                # Mark best validation epoch
                best_ep = vs[np.argmin(vv) if "loss" in val_tag else np.argmax(vv)]
                ax.axvline(best_ep, color="#999", lw=0.5, ls=":")
                ax.text(best_ep, ax.get_ylim()[1],
                        f" ep {best_ep}", fontsize=5, color="#666",
                        va="top", ha="left")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=5.5)
        label_panel(ax, lbl)

        if train_tag == "train/lr":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.1e}")
            )

    if output_path:
        fig.savefig(str(output_path), dpi=300)
        print(f"[fig5] → {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _npy(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not Path(path).exists():
        return None
    return np.load(path).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3PM publication-quality figure generator"
    )
    p.add_argument("--mode", choices=["single", "ablation", "training", "all", "stats"],
                   default="single")
    # Single-image sources
    p.add_argument("--pred_npy",       type=str, default=None)
    p.add_argument("--gt_npy",         type=str, default=None)
    p.add_argument("--noisy_npy",      type=str, default=None)
    p.add_argument("--sample_name",    type=str, default="")
    p.add_argument("--pixel_size_um",  type=float, default=0.1)
    p.add_argument("--roi", type=int, nargs=4, default=None,
                   metavar=("Y0", "X0", "H", "W"))
    p.add_argument("--profile_row",    type=int, default=None)
    p.add_argument("--metrics_csv",    type=str, default=None,
                   help="metrics_{run}.csv produced by eval.py (for mode=stats)")
    # Ablation
    p.add_argument("--ablation_csv",   type=str, default=None)
    p.add_argument("--pvalues_csv",    type=str, default=None)
    p.add_argument("--ablation_metrics", nargs="+",
                   default=["psnr", "ssim", "ms_ssim", "lpips"])
    # Training curves
    p.add_argument("--tensorboard_dir", type=str, default=None)
    # Output
    p.add_argument("--output_dir",     type=str, default="figures")
    p.add_argument("--fmt",            type=str, default="pdf",
                   choices=["pdf", "svg", "png", "tif"],
                   help="pdf/svg for press-ready vector output (recommended)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt  = args.fmt

    # ── Figures 1–3: single-image ────────────────────────────────────────────
    if args.mode in ("single", "all"):
        pred  = _npy(args.pred_npy)
        gt    = _npy(args.gt_npy)
        noisy = _npy(args.noisy_npy)

        if pred is not None and gt is not None:
            from eval import _psnr, _ssim
            psnr_p = _psnr(pred, gt)
            ssim_p = _ssim(pred, gt)
            psnr_n = _psnr(noisy, gt) if noisy is not None else None

            noisy_img = noisy if noisy is not None else np.zeros_like(gt)

            figure_qualitative(
                noisy_img, pred, gt,
                roi=tuple(args.roi) if args.roi else None,
                pixel_size_um=args.pixel_size_um,
                sample_name=args.sample_name,
                psnr_pred=psnr_p, ssim_pred=ssim_p,
                psnr_noisy=psnr_n,
                output_path=out / f"fig1_qualitative.{fmt}",
            )
            figure_error_analysis(
                pred, gt, noisy,
                profile_row=args.profile_row,
                pixel_size_um=args.pixel_size_um,
                sample_name=args.sample_name,
                output_path=out / f"fig2_error.{fmt}",
                compute_fwhm=True,
            )
            figure_frequency(
                noisy_img, pred, gt,
                sample_name=args.sample_name,
                output_path=out / f"fig3_frequency.{fmt}",
            )
        else:
            print("[visualize] --pred_npy and --gt_npy required for mode=single/all")

    # ── Figure 4: ablation ───────────────────────────────────────────────────
    if args.mode in ("ablation", "all") and args.ablation_csv:
        figure_ablation(
            ablation_csv=Path(args.ablation_csv),
            pvalues_csv=Path(args.pvalues_csv) if args.pvalues_csv else None,
            metrics=args.ablation_metrics,
            output_path=out / f"fig4_ablation.{fmt}",
        )

    # ── Figure 5: training curves ────────────────────────────────────────────
    if args.mode in ("training", "all") and args.tensorboard_dir:
        figure_training(
            log_dir=Path(args.tensorboard_dir),
            output_path=out / f"fig5_training.{fmt}",
        )

    # ── Figure 3 (stats-only): single-run PSNR/SSIM distribution ──────────────
    if args.mode == "stats":
        if not args.metrics_csv:
            print("[visualize] --metrics_csv is required for mode=stats")
        else:
            figure_statistics_single_run(
                metrics_csv=Path(args.metrics_csv),
                output_path=out / f"fig3_stats.{fmt}",
            )

    print(f"\n[visualize] All figures written to {out}/")


if __name__ == "__main__":
    main()

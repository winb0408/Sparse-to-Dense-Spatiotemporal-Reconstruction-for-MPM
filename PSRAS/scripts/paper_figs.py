"""Paper-level figures for the PSRAS v4 innovation claims (quant + qual).

Claims mapped to figures:
  C1 fixed-budget Y/Z allocation law (ky,kz)=(2,8)   -> fig1_alloc_law
  C2 Wiener prior + gated fusion beats direct          -> fig3_gain_over_direct
  C3 prior quality does not decide network quality     -> fig2_prior_vs_network
  C4 unified model closes most of the specialist gap   -> fig4/fig5
  qualitative evidence                                 -> qual_*, xz_*

Sub-commands
  quant          -- data-only figures from metrics_test*.json / EXP-023 csv
  qual_yz        -- EXP-024 arm panels + XZ cross-sections (GPU)
  qual_unified   -- EXP-025/026 unified panels across regimes (GPU)
  all            -- quant + qual_yz + qual_unified

Style: serif, thin open spines, no grid, muted palette, PNG 300 dpi + PDF,
grayscale panels vmin=0/vmax=1 with axes off (matches the threePM figure
convention used by visualize_v3.py).

Examples
  CUDA_VISIBLE_DEVICES=1 python scripts/paper_figs.py quant
  CUDA_VISIBLE_DEVICES=1 python scripts/paper_figs.py qual_yz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.datasets import _derive_seed, make_val_dataset  # noqa: E402
from data.observation import build_observation             # noqa: E402
from metrics.image_metrics import psnr                     # noqa: E402
from physics.acquisition import (                          # noqa: E402
    AcquisitionConfig, NoiseConfig, simulate_acquisition)
from physics.psf import PSFConfig                          # noqa: E402
from scripts.train import PRESETS, build_model             # noqa: E402

ROOT = Path(__file__).parents[1]
RUNS = ROOT / "runs_v3"
DATA = RUNS / "data/cidc25_v3"
OUT = ROOT / "eval_viz/paper_figures"

PTS = [(2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (8, 2)]
PT_LAB = [f"{a}×{b}" for a, b in PTS]

# Arm -> (colour, marker, label).  b1 Direct, b2 bilinear-WS, b4i IterDC-WS,
# b4f FillFirst-Wiener-WS + gated fusion.
ARMS = {
    "b1": ("#8a8a8a", "o", "-",  "Direct (Y only)"),
    "b2": ("#4c72b0", "s", "--", "Bilinear prior"),
    "b4i": ("#dd8452", "^", "-.", "Iter-DC prior"),
    "b4f": ("#55a868", "D", "-",  "FillFirst-Wiener + gated fusion"),
}
ARM_FULL = {
    "b1": "b1 = Direct / Y-only input (y, mask)",
    "b2": "b2 = bilinear warm-start prior (y, mask, x0)",
    "b4i": "b4i = Iter-DC warm-start prior (y, mask, x0, C_s, U_p)",
    "b4f": "b4f = FillFirst-Wiener warm-start prior (y, mask, x0, C_s, U_p)",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, stem: str, out_dir: Path = OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_dir / stem}.png")


# --------------------------------------------------------------------------
# metrics helpers
# --------------------------------------------------------------------------

def _mn(run: str, tag: str | None = None) -> float | None:
    """Test-split PSNR of a run dir (metrics_test[_tag].json)."""
    p = RUNS / run / (f"metrics_test_{tag}.json" if tag else "metrics_test.json")
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["psnr"])


def _arm_stats(arm: str) -> tuple[list[float], list[float], list[int]]:
    """Mean/sd/n over seeds for one arm at every YZ point."""
    means, sds, ns = [], [], []
    for ky, kz in PTS:
        vs = [v for v in (_mn(f"YZ_{ky}x{kz}_{arm}{s}")
                          for s in ("", "_s1", "_s2")) if v is not None]
        means.append(float(np.mean(vs)) if vs else np.nan)
        sds.append(float(np.std(vs)) if vs else 0.0)
        ns.append(len(vs))
    return means, sds, ns


def _prior_stats() -> dict[str, list[float]]:
    """EXP-023 prior-sweep PSNR (mean over items) per recipe per point."""
    by = defaultdict(lambda: defaultdict(list))
    with open(RUNS / "exp023_prior_sweep/summary.csv") as fh:
        for row in csv.DictReader(fh):
            by[(int(row["ky"]), int(row["kz"]))][row["recipe"]].append(
                float(row["psnr"]))
    out = defaultdict(list)
    for pt in PTS:
        for recipe, vals in by[pt].items():
            out[recipe].append(float(np.mean(vals)))
    return out


# --------------------------------------------------------------------------
# quant figures
# --------------------------------------------------------------------------

def fig_alloc_law(out_dir: Path) -> None:
    """C1: fixed-budget allocation law.  (a) PSNR vs the 6 YZ points for the
    4 arms; (b) equal-budget slice ky*kz=16 -- (2,8) vs (4,4) vs (8,2)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    x = np.arange(len(PTS))
    for arm, (c, m, ls, lab) in ARMS.items():
        mu, sd, _ = _arm_stats(arm)
        ax.errorbar(x, mu, yerr=sd, color=c, marker=m, ls=ls, lw=1.3,
                    ms=4.2, capsize=2.5, label=lab, zorder=3)
        if arm == "b4f":
            for xi, v in zip(x, mu):
                ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=6.8, color=c)
    ax.set_xticks(x, PT_LAB)
    ax.set_xlabel(r"YZ sampling point $(k_y \times k_z)$")
    ax.set_ylabel("PSNR (dB), test split")
    ax.set_title("(a) YZ arms over the sampling grid")
    ax.legend(frameon=False, loc="lower left", handlelength=2.4)
    ax.set_ylim(27.5, 43.0)
    ax.annotate("error bars = seed spread (3 seeds at 2×8, 4×4)",
                (0.02, 0.02), xycoords="axes fraction", fontsize=6.5,
                color="0.35")

    ax = axes[1]
    eq_pts = [(2, 8), (4, 4), (8, 2)]
    w = 0.2
    xi = np.arange(len(eq_pts))
    for j, (arm, (c, _, _, lab)) in enumerate(ARMS.items()):
        mu, sd, _ = _arm_stats(arm)
        vals = [mu[PTS.index(p)] for p in eq_pts]
        errs = [sd[PTS.index(p)] for p in eq_pts]
        ax.bar(xi + (j - 1.5) * w, vals, w, yerr=errs, color=c, alpha=0.9,
               edgecolor="black", linewidth=0.4, capsize=2,
               error_kw={"elinewidth": 0.7}, label=arm)
        for a, b in zip(xi + (j - 1.5) * w, vals):
            ax.text(a, b + 0.25, f"{b:.1f}", ha="center", fontsize=6,
                    rotation=90, color="0.2")
    ax.set_xticks(xi, [f"{a}×{b}" for a, b in eq_pts])
    ax.set_xlabel(r"equal budget $k_y k_z = 16$ samples / 16-plane volume")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(b) same sampling cost, different allocation")
    ax.set_ylim(27.0, 44.5)
    ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig1_alloc_law", out_dir)


def fig_prior_vs_network(out_dir: Path) -> None:
    """C3: prior-side ranking vs network-side ranking.
    (a) analytic prior PSNR, (b) trained network PSNR, (c) the deltas --
    prior says IterDC >= FillFirst at 4 of 6 points, network says otherwise."""
    prior = _prior_stats()
    recipes = [("P0_bilinear", "#8a8a8a", "o", "-", "P0 bilinear"),
               ("P1_wiener_frozen", "#c44e52", "v", ":", "P1 Wiener (frozen)"),
               ("P4_iterdc_1e-3_5", "#dd8452", "^", "-.", "P4 Iter-DC"),
               ("P2_fill_first", "#55a868", "D", "-", "P2 FillFirst-Wiener")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    x = np.arange(len(PTS))
    ax = axes[0]
    for key, c, m, ls, lab in recipes:
        ax.plot(x, prior[key], color=c, marker=m, ls=ls, lw=1.2, ms=4,
                label=lab)
    ax.set_xticks(x, PT_LAB, rotation=20)
    ax.set_ylabel("PSNR (dB) of the prior volume")
    ax.set_title("(a) prior side: Iter-DC ≥ FillFirst at 4/6")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_ylim(26, 40)

    ax = axes[1]
    for arm in ("b2", "b4i", "b4f"):
        c, m, ls, lab = ARMS[arm]
        mu, sd, _ = _arm_stats(arm)
        ax.errorbar(x, mu, yerr=sd, color=c, marker=m, ls=ls, lw=1.3, ms=4,
                    capsize=2.5, label=lab)
    ax.set_xticks(x, PT_LAB, rotation=20)
    ax.set_ylabel("PSNR (dB) of the network output")
    ax.set_title("(b) network side: FillFirst is best at 5/6")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_ylim(27.5, 42)

    ax = axes[2]
    d_prior = np.array(prior["P4_iterdc_1e-3_5"]) - np.array(prior["P2_fill_first"])
    mu_i, _, _ = _arm_stats("b4i")
    mu_f, _, _ = _arm_stats("b4f")
    d_net = np.array(mu_i) - np.array(mu_f)
    w = 0.38
    ax.bar(x - w / 2, d_prior, w, color="#dd8452", edgecolor="black",
           linewidth=0.4, label=r"prior: IterDC $-$ FillFirst")
    ax.bar(x + w / 2, d_net, w, color="#55a868", edgecolor="black",
           linewidth=0.4, label=r"network: IterDC $-$ FillFirst")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, PT_LAB, rotation=20)
    ax.set_ylabel(r"$\Delta$ PSNR (dB)")
    ax.set_title("(c) the two rankings disagree")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_ylim(-1.7, 1.7)
    fig.tight_layout()
    _save(fig, "fig2_prior_vs_network", out_dir)


def fig_gain(out_dir: Path) -> None:
    """C2: gain over the Direct arm, per sampling point."""
    base, _, _ = _arm_stats("b1")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(PTS))
    w = 0.26
    for j, arm in enumerate(("b2", "b4i", "b4f")):
        c, _, _, lab = ARMS[arm]
        mu, _, _ = _arm_stats(arm)
        d = np.array(mu) - np.array(base)
        ax.bar(x + (j - 1) * w, d, w, color=c, alpha=0.92,
               edgecolor="black", linewidth=0.4, label=lab)
        for a, b in zip(x + (j - 1) * w, d):
            ax.text(a, b + 0.06, f"{b:+.2f}", ha="center", fontsize=6.1,
                    rotation=90, color="0.2")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, PT_LAB)
    ax.set_xlabel(r"YZ sampling point $(k_y \times k_z)$")
    ax.set_ylabel("PSNR gain over Direct (dB)")
    ax.set_title("Prior + gated fusion gain over the Y-only Direct baseline")
    ax.set_ylim(0, 4.3)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig3_gain_over_direct", out_dir)


def fig_unified(out_dir: Path) -> None:
    """C4: unified vs specialist.  (a) cross-transfer matrix as grouped
    bars; (b) unified variants (AZ branch on/off, regime weighting)."""
    suites = [("Y (K=4)", "y_K4"), ("YZ (4×4)", "yz2_K4x4"),
              ("XYZ (4×4)", "xyz2_K4x4")]
    models = [("UNIV4_SY", "#4c72b0", "Specialist Y"),
              ("UNIV4_SYZ", "#dd8452", "Specialist YZ"),
              ("UNIV4_SXYZ", "#8172b2", "Specialist XYZ"),
              ("UNIV4_U15", "#55a868", "Unified U15")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    w = 0.2
    xi = np.arange(len(suites))
    for j, (run, c, lab) in enumerate(models):
        vals, errs = [], []
        for _, tag in suites:
            vs = [v for v in (_mn(f"{run}_s{s}", tag) for s in (0, 1, 2))
                  if v is not None]
            vals.append(float(np.mean(vs)))
            errs.append(float(np.std(vs)))
        ax.bar(xi + (j - 1.5) * w, vals, w, yerr=errs, color=c, alpha=0.9,
               edgecolor="black", linewidth=0.4, capsize=1.8,
               error_kw={"elinewidth": 0.6}, label=lab)
        for a, b in zip(xi + (j - 1.5) * w, vals):
            ax.text(a, b + 0.5, f"{b:.1f}", ha="center", fontsize=5.8,
                    rotation=90, color="0.15")
    ax.set_xticks(xi, [s for s, _ in suites])
    ax.set_xlabel("evaluation regime (test split)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) specialists collapse off-regime, U15 does not")
    ax.set_ylim(0, 42)
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", ncol=2)
    ax.annotate("mean ± sd over 3 seeds", (0.02, 0.96),
                xycoords="axes fraction", fontsize=6.4, color="0.35",
                va="top")

    ax = axes[1]
    variants = [("UNIV4_U15", "#55a868", "U15 (AZ prior on)"),
                ("UNIV4_U_noAZ", "#8a8a8a", "U-noAZ (AZ prior off)"),
                ("UNIV4_U15W", "#4c72b0", "U15W (1:1:2 weighting)")]
    for j, (run, c, lab) in enumerate(variants):
        vals, errs = [], []
        for _, tag in suites:
            vs = [v for v in (_mn(f"{run}_s{s}", tag) for s in (0, 1, 2))
                  if v is not None]
            vals.append(float(np.mean(vs)))
            errs.append(float(np.std(vs)))
        ax.bar(xi + (j - 1) * w, vals, w, yerr=errs, color=c, alpha=0.9,
               edgecolor="black", linewidth=0.4, capsize=1.8,
               error_kw={"elinewidth": 0.6}, label=lab)
        for a, b in zip(xi + (j - 1) * w, vals):
            ax.text(a, b + 0.5, f"{b:.1f}", ha="center", fontsize=5.8,
                    rotation=90, color="0.15")
    ax.set_xticks(xi, [s for s, _ in suites])
    ax.set_xlabel("evaluation regime (test split)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(b) the axial prior is what carries the XYZ regime")
    ax.set_ylim(0, 42)
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", ncol=2)
    fig.tight_layout()
    _save(fig, "fig4_unified_cross_transfer", out_dir)


# --------------------------------------------------------------------------
# model / dataset helpers for qualitative figures
# --------------------------------------------------------------------------

def _load_model(run: str, device: torch.device):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    m = cfg["model"]
    model = build_model(PRESETS[cfg["preset"]], cfg["t"], m["base"],
                        m["window"], m.get("kind", "swin2d"),
                        axial_at=m.get("axial_at", "all"),
                        output=m.get("output", "all"),
                        window_t=m.get("window_t", 1),
                        use_az=m.get("use_az", True))
    sd = torch.load(RUNS / run / "best.pt", map_location="cpu",
                    weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval().to(device)
    return cfg, model


def _val_ds(cfg: dict, mode: str, K: int, K2: int = 0, *,
            sigma_z: float | None = None, features=None,
            warm: str | None = None, t_override: int | None = None):
    """Validation grid for an arbitrary protocol (unified models are
    T-agnostic, so the protocol is decoupled from the run config)."""
    sz = cfg["psf"]["sigma_z_px"] if sigma_z is None else sigma_z
    psf = PSFConfig(sigma_xy_um=cfg["psf"]["sigma_xy_px"], sigma_z_um=sz,
                    voxel_xy_um=1.0, voxel_z_um=1.0)
    return make_val_dataset(
        DATA, "val", "2.5d" if mode != "y" else "2d", mode=mode, K=K, K2=K2,
        psf=psf, patch_hw=(cfg["patch"], cfg["patch"]),
        warm_start_kind=warm or cfg["warm_start"], lam=cfg["lam_wiener"],
        features=tuple(features or cfg["features"]), t_override=t_override)


def _point_of(cfg: dict, mode: str, K: int, K2: int) -> dict:
    """Protocol parameters actually used at the eval point."""
    if mode == "y":
        return {"mode": "y", "K": K, "K2": 0, "t": 1, "sigma_z": 0.0}
    return {"mode": mode, "K": K, "K2": K2,
            "t": 1 if K2 == 1 else 2 * K2 - 1, "sigma_z": 1.5}


@torch.no_grad()
def _predict(model, x: torch.Tensor, t: int, device: torch.device):
    return model(x[None].to(device)).float().view(t, *x.shape[-2:])


def _query_index(t: int) -> int:
    return t // 2


def _draw(img, ax, title: str | None = None, fontsize: float = 9):
    if img is not None:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize, pad=3)


def _annot(ax, val: float, color="0.15") -> None:
    ax.text(0.5, -0.04, f"{val:.2f} dB", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.2, color=color)


# --------------------------------------------------------------------------
# EXP-024 qualitative
# --------------------------------------------------------------------------

def qual_yz(ky: int, kz: int, out_dir: Path, device, n_items: int = 3,
            with_err: bool = True) -> None:
    arms = ["b1", "b2", "b4i", "b4f"]
    runs = {a: f"YZ_{ky}x{kz}_{a}" for a in arms}
    loaded = {a: _load_model(r, device) for a, r in runs.items()}

    # display streams come from b4f (it carries x0 / confidence features);
    # every arm reads its own dataset because the input channels differ
    # (b1 = 2 feats, b2 = 3, b4i/b4f = 5).
    cfg0, _ = loaded["b4f"]
    dss = {a: _val_ds(loaded[a][0], "yz2", ky, kz, sigma_z=1.5) for a in arms}
    ds = dss["b4f"]
    t = ds.T
    q = ds.query[0] if t == 1 else _query_index(t)

    gts, ys, x0s, preds, errs = [], [], [], {a: [] for a in arms}, {a: [] for a in arms}
    for i in range(n_items):
        it = ds.fixed_items[i]
        x, target = it["inputs"], it["target"]
        gt = target[q].numpy()
        gts.append(gt)
        v = x.view(len(cfg0["features"]), t, *x.shape[-2:])
        # the query plane is usually NOT measured along the axial axis --
        # display the nearest measured plane of Y instead of a black frame
        cand = [j for j in range(t) if float(v[1][j].sum()) > 0]
        m = min(cand, key=lambda j: abs(j - q)) if cand else q
        ys.append(v[0][m].numpy())
        x0s.append(v[2][q].numpy())
        for a in arms:
            cfg, model = loaded[a]
            ai = dss[a].fixed_items[i]
            assert torch.equal(ai["target"], target), "arm item misaligned"
            pred = _predict(model, ai["inputs"], cfg["t"], device)[q].cpu().numpy()
            preds[a].append(pred)
            errs[a].append(np.abs(pred - gt))

    cols = [("GT", gts), ("Sparse Y\n(measured plane)", ys), ("X0 prior", x0s)]
    cols += [(ARMS[a][3], preds[a]) for a in arms]
    fig, axes = plt.subplots(n_items, len(cols),
                             figsize=(1.55 * len(cols), 1.55 * n_items + 0.9))
    if n_items == 1:
        axes = axes[None, :]
    for c, (title, imgs) in enumerate(cols):
        for r in range(n_items):
            ax = axes[r, c]
            _draw(imgs[r].clip(0, 1), ax,
                  title if r == 0 else None, fontsize=8.2)
            if title.startswith(("Direct", "Bilinear", "Iter-DC", "FillFirst")):
                _annot(ax, psnr(torch.from_numpy(imgs[r]),
                                torch.from_numpy(gts[r])).item(),
                       color=ARMS[[a for a in arms if ARMS[a][3] == title][0]][0])
    for r in range(n_items):
        axes[r, 0].set_ylabel(f"item {r}", fontsize=8)
    fig.suptitle(f"EXP-024 YZ (k_y,k_z)=({ky},{kz}) — query plane between "
                 f"measured planes (object u, val split)", fontsize=10)
    fig.text(0.5, 0.005,
             " | ".join(ARM_FULL.values()), ha="center", fontsize=6.2,
             color="0.3")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    _save(fig, f"qual_yz_{ky}x{kz}", out_dir)
    np.savez_compressed(out_dir / f"qual_yz_{ky}x{kz}_data.npz",
                        gt=np.stack(gts), sparse_y=np.stack(ys),
                        x0=np.stack(x0s),
                        **{f"pred_{a}": np.stack(preds[a]) for a in arms})

    if not with_err:
        return
    fig, axes = plt.subplots(n_items, len(arms) + 1,
                             figsize=(1.55 * (len(arms) + 1), 1.55 * n_items + 0.9))
    if n_items == 1:
        axes = axes[None, :]
    vmax = 0.12
    for r in range(n_items):
        _draw(gts[r].clip(0, 1), axes[r, 0],
              "GT" if r == 0 else None, fontsize=8.2)
        for c, a in enumerate(arms):
            im = axes[r, c + 1].imshow(errs[a][r], cmap="magma", vmin=0,
                                       vmax=vmax, interpolation="nearest")
            axes[r, c + 1].set_xticks([])
            axes[r, c + 1].set_yticks([])
            for sp in axes[r, c + 1].spines.values():
                sp.set_visible(False)
            if r == 0:
                axes[r, c + 1].set_title(ARMS[a][3], fontsize=8.2, pad=3)
            axes[r, c + 1].text(0.5, -0.04, f"MAE {errs[a][r].mean():.4f}",
                                transform=axes[r, c + 1].transAxes,
                                ha="center", va="top", fontsize=7)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label(r"$|$pred $-$ GT$|$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"EXP-024 YZ (k_y,k_z)=({ky},{kz}) — absolute error maps",
                 fontsize=10)
    _save(fig, f"qual_yz_{ky}x{kz}_err", out_dir)


def _long_item(ds, item_index: int, n_knots: int = 6) -> dict:
    """Lattice-anchored long volume; axial spacing k_ax = K2 for yz2."""
    n_per_stack = 8
    si, j = divmod(item_index, n_per_stack)
    stack, (f0, f1) = ds.stacks[si], ds.frame_ranges[si]
    k_ax = ds.acq.K2 if ds.acq.mode in ("yz2", "xyz2") and ds.acq.K2 else ds.acq.K
    t_win = ds.T
    l_disp = (n_knots - 1) * k_ax + 1
    l_total = l_disp + 2 * (k_ax - 1)
    t0s = np.linspace(f0, max(f0, f1 - t_win), n_per_stack, dtype=int)
    t0 = (int(t0s[j]) // k_ax) * k_ax
    t0 = max(0, min(t0, f1 - l_total))
    y0 = (stack.shape[1] - ds.patch_hw[0]) // 2
    x0 = (stack.shape[2] - ds.patch_hw[1]) // 2
    gt = np.asarray(stack[t0:t0 + l_total, y0:y0 + ds.patch_hw[0],
                          x0:x0 + ds.patch_hw[1]], dtype=np.float32)
    acq = AcquisitionConfig(
        mode=ds.acq.mode, K=ds.acq.K, K2=ds.acq.K2, pattern=ds.acq.pattern,
        psf=ds.acq.psf,
        noise=NoiseConfig(ds.acq.noise.poisson_photons,
                          ds.acq.noise.gaussian_sigma,
                          _derive_seed(ds.seed, 7777, si, t0)),
        mask_seed=_derive_seed(ds.seed, 8888, si, t0),
        phase_t=t0 % k_ax, phase_y=y0 % ds.acq.K, phase_x=x0 % ds.acq.K)
    obs = simulate_acquisition(torch.from_numpy(gt), acq)
    mi = build_observation(obs, psf=ds.acq.psf, lam=ds.lam,
                           warm_start_kind=ds.warm_start_kind)
    return {"gt": torch.from_numpy(gt), "y": mi.y, "mask": mi.mask,
            "x0": mi.x0, "confidence": mi.confidence,
            "uncertainty": mi.uncertainty}


@torch.no_grad()
def _slide(model, feats: dict, order: tuple[str, ...], k_ax: int, t_win: int,
           device, chunk: int = 8) -> torch.Tensor:
    l_total = feats["gt"].shape[0]
    l_disp = l_total - 2 * (k_ax - 1)
    half = t_win // 2
    outs, buf = [], []

    def _flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        outs.extend(t.cpu() for t in
                    model(x).float().view(-1, t_win, *x.shape[-2:])[:, half])
        buf.clear()

    for p in range(k_ax - 1, k_ax - 1 + l_disp):
        buf.append(torch.cat([feats[f][p - k_ax + 1: p + k_ax] for f in order], 0))
        if len(buf) >= chunk:
            _flush()
    _flush()
    return torch.stack(outs)


def qual_xz_yz(ky: int, kz: int, out_dir: Path, device, n_items: int = 2,
               arms: tuple[str, ...] = ("b1", "b4f")) -> None:
    """XZ cross-section (brightest GT column); measured planes = stripes."""
    loaded = {a: _load_model(f"YZ_{ky}x{kz}_{a}", device) for a in arms}
    cfg0, _ = loaded["b4f"]
    ds = _val_ds(cfg0, "yz2", ky, kz, sigma_z=1.5)
    k_ax, t_win = kz, ds.T
    order = tuple(cfg0["features"])
    cols = ["GT (XZ)", "Sparse Y (XZ)", "X0 (XZ)"] + [ARMS[a][3] for a in arms]
    fig, axes = plt.subplots(n_items, len(cols),
                             figsize=(1.7 * len(cols), 1.35 * n_items + 0.9))
    if n_items == 1:
        axes = axes[None, :]
    stats = []
    for r in range(n_items):
        feats = _long_item(ds, r)
        gt = feats["gt"]
        col = int(gt.sum(dim=(0, 1)).argmax().item()) if n_items == 1 or r == 0 \
            else stats[r - 1][-1]
        l_disp = gt.shape[0] - 2 * (k_ax - 1)
        sl = slice(k_ax - 1, k_ax - 1 + l_disp)
        gv = gt[sl][:, :, col].numpy()
        axes[r, 0].imshow(gv.T, cmap="gray", vmin=0, vmax=1, aspect="auto",
                          interpolation="nearest")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values():
            sp.set_visible(False)
        if r == 0:
            axes[r, 0].set_title("GT (XZ)", fontsize=8.2)
        yv = feats["y"][sl][:, :, col].numpy()
        axes[r, 1].imshow(yv.T, cmap="gray", vmin=0, vmax=1, aspect="auto",
                          interpolation="nearest")
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        for sp in axes[r, 1].spines.values():
            sp.set_visible(False)
        if r == 0:
            axes[r, 1].set_title("Sparse Y (XZ)", fontsize=8.2)
        xv = feats["x0"][sl][:, :, col].numpy()
        axes[r, 2].imshow(xv.T, cmap="gray", vmin=0, vmax=1, aspect="auto",
                          interpolation="nearest")
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
        for sp in axes[r, 2].spines.values():
            sp.set_visible(False)
        if r == 0:
            axes[r, 2].set_title("X0 (XZ)", fontsize=8.2)
        row_vals = []
        for c, a in enumerate(arms):
            cfg, model = loaded[a]
            pv = _slide(model, feats, tuple(cfg["features"]), k_ax, t_win,
                        device)[:, :, col].numpy()
            ax = axes[r, 3 + c]
            ax.imshow(pv.T, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            p = psnr(torch.from_numpy(pv), torch.from_numpy(gv)).item()
            row_vals.append(p)
            ax.text(0.5, -0.04, f"{p:.2f} dB", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.2, color=ARMS[a][0])
            if r == 0:
                ax.set_title(ARMS[a][3], fontsize=8.2)
        stats.append(row_vals + [col])
        print(f"  xz ({ky}x{kz}) item {r} col {col}: "
              + ", ".join(f"{a}={v:.2f}" for a, v in zip(arms, row_vals)))
    for r in range(n_items):
        axes[r, 0].set_ylabel(f"item {r}", fontsize=8)
    fig.suptitle(f"EXP-024 YZ (k_y,k_z)=({ky},{kz}) — XZ cross-section "
                 f"(column {stats[0][-1]}); horizontal stripes = unmeasured "
                 f"planes (spacing {k_ax})", fontsize=9.5)
    fig.tight_layout()
    _save(fig, f"xz_yz_{ky}x{kz}", out_dir)


# --------------------------------------------------------------------------
# EXP-025/026 unified qualitative
# --------------------------------------------------------------------------

SUITE_SPEC = {
    "y": ("y_K4", "y", 4, 0, ("UNIV4_SY", "UNIV4_U15")),
    "yz2": ("yz2_K4x4", "yz2", 4, 4, ("UNIV4_SYZ", "UNIV4_U15")),
    "xyz2": ("xyz2_K4x4", "xyz2", 4, 4, ("UNIV4_SXYZ", "UNIV4_U15")),
}


def qual_unified(kind: str, out_dir: Path, device, n_items: int = 3) -> None:
    tag, mode, K, K2, runs = SUITE_SPEC[kind]
    loaded = {r: _load_model(r + "_s0", device) for r in runs}
    cfg0, _ = loaded["UNIV4_U15"]
    pt = _point_of(cfg0, mode, K, K2)
    ds = _val_ds(cfg0, mode, K, K2, sigma_z=pt["sigma_z"])
    t = ds.T
    q = _query_index(t) if t > 1 else 0

    disp_cfg = cfg0
    gts, sp, x0s = [], [], []
    preds = {r: [] for r in runs}
    for i in range(n_items):
        it = ds.fixed_items[i]
        x, target = it["inputs"], it["target"]
        gts.append(target[q].numpy())
        v = x.view(len(disp_cfg["features"]), t, *x.shape[-2:])
        cand = [j for j in range(t) if float(v[1][j].sum()) > 0]
        m = min(cand, key=lambda j: abs(j - q)) if cand else q
        sp.append(v[0][m].numpy())
        x0s.append(v[2][q].numpy())
        for r in runs:
            _, model = loaded[r]
            preds[r].append(
                _predict(model, it["inputs"], t, device)[q].cpu().numpy())

    names = {"UNIV4_SY": "Specialist Y", "UNIV4_SYZ": "Specialist YZ",
             "UNIV4_SXYZ": "Specialist XYZ", "UNIV4_U15": "Unified U15"}
    cols = [("GT", gts), ("Sparse input\n(measured plane)", sp), ("X0 prior", x0s)]
    cols += [(names[r], preds[r]) for r in runs]
    fig, axes = plt.subplots(n_items, len(cols),
                             figsize=(1.55 * len(cols), 1.55 * n_items + 0.9))
    if n_items == 1:
        axes = axes[None, :]
    for c, (title, imgs) in enumerate(cols):
        for r in range(n_items):
            ax = axes[r, c]
            _draw(None, ax, title if r == 0 else None, fontsize=8.2)
            ax.imshow(imgs[r].clip(0, 1), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if title in ("Specialist Y", "Specialist YZ", "Specialist XYZ", "Unified U15"):
                key = [r_ for r_ in runs if names[r_] == title][0]
                _annot(ax, psnr(torch.from_numpy(imgs[r]),
                                torch.from_numpy(gts[r])).item(),
                       color="#55a868" if key == "UNIV4_U15" else "0.3")
    for r in range(n_items):
        axes[r, 0].set_ylabel(f"item {r}", fontsize=8)
    fig.suptitle(f"EXP-025/026 unified model on {tag} — specialist vs "
                 f"one unified network (object u, val split)", fontsize=10)
    fig.tight_layout()
    _save(fig, f"qual_univ_{tag}", out_dir)
    np.savez_compressed(out_dir / f"qual_univ_{tag}_data.npz",
                        gt=np.stack(gts), sparse=np.stack(sp),
                        x0=np.stack(x0s),
                        **{f"pred_{r}": np.stack(preds[r]) for r in runs})


def xz_unified_yz(out_dir: Path, device, n_items: int = 2) -> None:
    """XZ cross-section for the unified model on YZ 4x4 (no specialist)."""
    cfg, model = _load_model("UNIV4_U15_s0", device)
    ds = _val_ds(cfg, "yz2", 4, 4, sigma_z=1.5)
    k_ax, t_win = 4, ds.T
    order = tuple(cfg["features"])
    fig, axes = plt.subplots(n_items, 4, figsize=(6.8, 1.35 * n_items + 0.9))
    if n_items == 1:
        axes = axes[None, :]
    for r in range(n_items):
        feats = _long_item(ds, r)
        gt = feats["gt"]
        col = int(gt.sum(dim=(0, 1)).argmax().item())
        l_disp = gt.shape[0] - 2 * (k_ax - 1)
        sl = slice(k_ax - 1, k_ax - 1 + l_disp)
        gv = gt[sl][:, :, col].numpy()
        pv = _slide(model, feats, order, k_ax, t_win, device)[:, :, col]
        for c, (name, img) in enumerate([
                ("GT (XZ)", gv),
                ("Sparse Y (XZ)", feats["y"][sl][:, :, col].numpy()),
                ("X0 (XZ)", feats["x0"][sl][:, :, col].numpy()),
                ("Unified U15 (XZ)", pv.numpy())]):
            ax = axes[r, c]
            ax.imshow(img.T, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(name, fontsize=8.2)
            if c == 3:
                ax.text(0.5, -0.04,
                        f"{psnr(torch.from_numpy(img), torch.from_numpy(gv)).item():.2f} dB",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.2, color="#55a868")
    for r in range(n_items):
        axes[r, 0].set_ylabel(f"item {r}", fontsize=8)
    fig.suptitle("Unified U15 on YZ (4×4) — XZ cross-section; "
                 "stripes = unmeasured planes (spacing 4)", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "xz_univ_yz2_K4x4", out_dir)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quant", "qual_yz", "qual_unified", "all"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--items", type=int, default=3)
    args = ap.parse_args()
    _style()
    out_dir = Path(args.out)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.cmd in ("quant", "all"):
        fig_alloc_law(out_dir)
        fig_prior_vs_network(out_dir)
        fig_gain(out_dir)
        fig_unified(out_dir)
    if args.cmd in ("qual_yz", "all"):
        for ky, kz in ((2, 8), (8, 2)):
            qual_yz(ky, kz, out_dir, device, n_items=args.items)
            qual_xz_yz(ky, kz, out_dir, device, n_items=2)
    if args.cmd in ("qual_unified", "all"):
        for kind in ("y", "yz2", "xyz2"):
            qual_unified(kind, out_dir, device, n_items=args.items)
        xz_unified_yz(out_dir, device)


if __name__ == "__main__":
    main()

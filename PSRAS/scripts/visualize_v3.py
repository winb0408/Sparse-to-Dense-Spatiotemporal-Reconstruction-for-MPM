"""Unified qualitative + curve visualization for PSRAS Protocol v3 (DEC-020).

Style is deliberately inherited from threePM's
scripts/plot_sparse_recon_case_studies.py: grayscale panels with
vmin=0/vmax=1, axes off, column titles on the top row, row labels on the
left, saved as both PNG (240 dpi) and PDF.

Sub-commands:
  panel  -- qualitative grid for a group of runs sharing one protocol:
           columns = GT | Sparse Y | X0 | <runs...>, rows = val items.
           For z-mode protocols a second XZ-cross-section figure is
           produced (missing planes show as horizontal stripes).
  curve  -- severity curve: PSNR vs K, one line per preset label,
           mean +- per-seed spread, from each run's
           val_metrics_corrected.json (reval output).

Examples:
  python scripts/visualize_v3.py panel --out eval_viz/severity_z_K2 \
      --runs B0=runs_v3/b0_2p5d_z_K2_p8k_lam3e6 B2=... B3=... B4=...
  python scripts/visualize_v3.py curve --out eval_viz/severity_curve \
      --series B4=runs_v3/b4_2p5d_z_K{K}_p8k_lam3e6 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.datasets import _derive_seed, make_val_dataset  # noqa: E402
from data.observation import build_observation        # noqa: E402
from metrics.image_metrics import psnr                # noqa: E402
from physics.acquisition import (                     # noqa: E402
    AcquisitionConfig, NoiseConfig, simulate_acquisition)
from physics.psf import PSFConfig                     # noqa: E402
from scripts.train import PRESETS, build_model        # noqa: E402

FEAT = {"y": 0, "mask": 1, "x0": 2}


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):                 # legacy convention: png + pdf
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _load_run(run_dir: Path, ckpt: str, device: torch.device):
    cfg = json.loads((run_dir / "config.json").read_text())
    psf = PSFConfig(sigma_xy_um=cfg["psf"]["sigma_xy_px"],
                    sigma_z_um=cfg["psf"]["sigma_z_px"],
                    voxel_xy_um=1.0, voxel_z_um=1.0)
    feats = tuple(cfg["features"])
    val_ds = make_val_dataset(Path(__file__).parents[1] / "runs_v3/data/cidc25_v3",
                              "val", cfg["regime"], mode=cfg["mode"], K=cfg["K"],
                              psf=psf, patch_hw=(cfg["patch"], cfg["patch"]),
                              warm_start_kind=cfg["warm_start"],
                              lam=cfg["lam_wiener"], features=feats,
                              t_override=cfg["t"])
    preset = PRESETS[cfg["preset"]]
    model = build_model(preset, cfg["t"], cfg["model"]["base"],
                        cfg["model"]["window"], cfg["model"].get("kind", "swin2d"),
                        axial_at=cfg["model"].get("axial_at", "all"),
                        output=cfg["model"].get("output", "all"))
    sd = torch.load(run_dir / f"{ckpt}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval().to(device)
    return cfg, val_ds, model


@torch.no_grad()
def _predict(model, x: torch.Tensor, t: int, device: torch.device) -> torch.Tensor:
    """Center (query) slice prediction, [H, W]."""
    pred = model(x[None].to(device)).float().view(t, *x.shape[-2:])
    return pred


def _build_long_item(ds, item_index: int, n_knots: int) -> dict[str, torch.Tensor]:
    """Lattice-anchored long volume for sliding-query inference (DEC-021 A).

    Knots sit every K planes (phase 0).  The displayed range spans
    (n_knots-1)*K + 1 planes between the first and last knot; K-1 margin
    planes are appended on each side so every displayed plane's full 2K-1
    window (T = t of the run) fits inside the built volume.  Patch geometry
    and seeds mirror make_val_dataset so the volume is a superset of the
    corresponding val item's physics.
    """
    n_per_stack = 8
    si, j = divmod(item_index, n_per_stack)
    stack, (f0, f1) = ds.stacks[si], ds.frame_ranges[si]
    k, t_win = ds.acq.K, ds.T
    l_disp = (n_knots - 1) * k + 1
    l_total = l_disp + 2 * (k - 1)
    t0s = np.linspace(f0, max(f0, f1 - t_win), n_per_stack, dtype=int)
    t0 = (int(t0s[j]) // k) * k                    # lattice phase 0
    t0 = max(0, min(t0, f1 - l_total))
    y0 = (stack.shape[1] - ds.patch_hw[0]) // 2
    x0 = (stack.shape[2] - ds.patch_hw[1]) // 2
    gt = np.asarray(stack[t0:t0 + l_total, y0:y0 + ds.patch_hw[0],
                          x0:x0 + ds.patch_hw[1]], dtype=np.float32)
    acq = AcquisitionConfig(
        mode=ds.acq.mode, K=k, pattern=ds.acq.pattern, psf=ds.acq.psf,
        noise=NoiseConfig(ds.acq.noise.poisson_photons,
                          ds.acq.noise.gaussian_sigma,
                          _derive_seed(ds.seed, 7777, si, t0)),
        mask_seed=_derive_seed(ds.seed, 8888, si, t0),
        phase_t=t0 % k, phase_y=y0 % k, phase_x=x0 % k)
    obs = simulate_acquisition(torch.from_numpy(gt), acq)
    mi = build_observation(obs, psf=ds.acq.psf, lam=ds.lam,
                           warm_start_kind=ds.warm_start_kind)
    return {"gt": torch.from_numpy(gt), "y": mi.y, "mask": mi.mask,
            "x0": mi.x0, "confidence": mi.confidence,
            "uncertainty": mi.uncertainty}


@torch.no_grad()
def _sliding_query_volume(model, feats_long: dict[str, torch.Tensor],
                          feat_order: tuple[str, ...], k: int, t_win: int,
                          device: torch.device, chunk: int = 8,
                          residual: bool = False) -> torch.Tensor:
    """Full-volume prediction by sliding the query plane (DEC-021 option A).

    For every displayed plane p, the model is fed the 2K-1 window centred on
    p (the training geometry) and only the centre output is kept.  Returns
    [L_disp, H, W]; every plane is now a supervised-geometry prediction, so
    the XZ cross-section is meaningful everywhere.
    """
    l_total = feats_long["gt"].shape[0]
    l_disp = l_total - 2 * (k - 1)
    half = t_win // 2
    assert not residual, "residual presets not supported in sliding-query viz"
    outs: list[torch.Tensor] = []
    buf: list[torch.Tensor] = []

    def _flush() -> None:
        if not buf:
            return
        x = torch.stack(buf).to(device)
        pred = model(x).float().view(-1, t_win, *x.shape[-2:])[:, half]
        outs.extend(t.cpu() for t in pred)
        buf.clear()

    for p in range(k - 1, k - 1 + l_disp):
        buf.append(torch.cat([feats_long[f][p - k + 1: p + k]
                              for f in feat_order], 0))
        if len(buf) >= chunk:
            _flush()
    _flush()
    return torch.stack(outs)


def cmd_panel(args) -> None:
    device = torch.device(args.device)
    runs = {}
    for spec in args.runs:                      # "B0=runs_v3/..."
        name, path = spec.split("=", 1)
        runs[name] = Path(path)

    cfg0, ds0, _ = _load_run(runs[next(iter(runs))], args.ckpt, device)
    t, mode = cfg0["t"], cfg0["mode"]
    items = args.items
    idxs = items if items else list(range(min(3, len(ds0.fixed_items))))

    # display streams (GT / Y / X0) must come from a run whose feature list
    # actually contains x0 (B0 carries only [y]); fall back to no X0 column.
    display_src = next((n for n in runs
                        if len(json.loads((runs[n] / "config.json").read_text())["features"]) >= 3),
                       None)

    preds: dict[str, list] = {name: [] for name in runs}
    gts, ys, x0s = [], [], []
    for name, path in runs.items():
        cfg, ds, model = _load_run(path, args.ckpt, device)
        assert (cfg["mode"], cfg["K"]) == (cfg0["mode"], cfg0["K"]), \
            f"{name} protocol mismatch: {cfg['mode']}/K{cfg['K']}"
        for i in idxs:
            it = ds.fixed_items[i]
            x, target = it["inputs"], it["target"]
            pred = _predict(model, x, cfg["t"], device)
            q = it["query"].bool()
            preds[name].append((pred[q][0].cpu(), target.view(t, *x.shape[-2:])[q][0].cpu()))
            if name == display_src:
                v = x.view(len(cfg["features"]), t, *x.shape[-2:])
                gts.append(target.view(t, *x.shape[-2:])[q][0].cpu())
                ys.append(v[0][q][0].cpu())
                x0s.append(v[2][q][0].cpu())

    col_specs = [("GT", gts), ("Sparse Y", ys)]
    if display_src is not None:
        col_specs.append(("X0", x0s))
    col_specs += [(n, [p for p, _ in preds[n]]) for n in runs]
    n_rows, n_cols = len(idxs), len(col_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 1.9 * n_rows + 1))
    if n_rows == 1:
        axes = axes[None, :]
    for c, (title, imgs) in enumerate(col_specs):
        axes[0, c].set_title(title, fontsize=11)
        for r in range(n_rows):
            ax = axes[r, c]
            img = imgs[r]
            ax.imshow(img.numpy().clip(0, 1), cmap="gray", vmin=0, vmax=1)
            tgt = col_specs[0][1][r]         # GT column
            if title not in ("GT", "Sparse Y"):
                p = psnr(img, tgt).item()
                ax.text(0.5, -0.06, f"{p:.2f} dB", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8)
            ax.axis("off")
    for r, i in enumerate(idxs):
        axes[r, 0].set_ylabel(f"item {i}", fontsize=10)
    fig.suptitle(f"PSRAS v3 | mode={mode} K={cfg0['K']} lam={cfg0['lam_wiener']:g} "
                 f"| {args.ckpt}.pt", fontsize=13)
    fig.tight_layout()
    _save(fig, Path(args.out), f"qual_panel_{mode}_K{cfg0['K']}")

    if mode == "z":                             # XZ cross-section figure
        if args.xz_col is None:                 # brightest GT column
            col = int(gts[0].sum(dim=0).argmax().item())
        else:
            col = args.xz_col
        n_cols_xz = 2 + len(runs)              # GT | X0 | runs (Y is 1-slice)
        fig2, axes2 = plt.subplots(n_rows, n_cols_xz,
                                   figsize=(2.4 * n_cols_xz, 2.4 * n_rows + 1))
        if n_rows == 1:
            axes2 = axes2[None, :]
        vols: dict[str, list] = {}
        for name, path in runs.items():
            cfg, ds, model = _load_run(path, args.ckpt, device)
            vols[name] = []
            for i in idxs:
                it = ds.fixed_items[i]
                pred = _predict(model, it["inputs"], cfg["t"], device)
                vols[name].append(pred[:, :, col].cpu())
        first = display_src if display_src is not None else next(iter(runs))
        cfg, ds, _ = _load_run(runs[first], args.ckpt, device)
        gt_vols, x0_vols = [], []
        for i in idxs:
            it = ds.fixed_items[i]
            x, target = it["inputs"], it["target"]
            v = x.view(len(cfg["features"]), t, *x.shape[-2:])
            gt_vols.append(target.view(t, *x.shape[-2:])[:, :, col].cpu())
            x0_vols.append(v[2][:, :, col].cpu() if len(cfg["features"]) >= 3
                           else None)
        xz_specs = [("GT (XZ)", gt_vols)]
        if display_src is not None:
            xz_specs.append(("X0 (XZ)", x0_vols))
        xz_specs += [(n, vols[n]) for n in runs]
        for c, (title, vv) in enumerate(xz_specs):
            axes2[0, c].set_title(title, fontsize=11)
            for r in range(n_rows):
                axes2[r, c].imshow(vv[r].numpy().T.clip(0, 1), cmap="gray",
                                   vmin=0, vmax=1, aspect="auto")
                axes2[r, c].axis("off")
        for r, i in enumerate(idxs):
            axes2[r, 0].set_ylabel(f"item {i}", fontsize=10)
        fig2.suptitle(f"PSRAS v3 | XZ cross-section (col {col}) | mode=z "
                      f"K={cfg0['K']} | stripes = missing planes", fontsize=12)
        fig2.tight_layout()
        _save(fig2, Path(args.out), f"qual_xz_K{cfg0['K']}")
    print(f"saved -> {args.out}")


def cmd_tcurve(args) -> None:
    """EXP-017: PSNR vs axial context depth T (z mode, K fixed)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for spec in args.series:                    # "B3=runs_v3/b3_..._T{T}_..."
        name, pattern = spec.split("=", 1)
        ts, vals = [], []
        for t in args.ts:
            d = Path(pattern.format(T=t))
            if t == 2 * args.K - 1 and not d.exists():
                # anchor point: the severity-matrix run (no T tag)
                d = Path(pattern.format(T=t)).with_name(
                    Path(pattern.format(T=t)).name.replace(f"_T{t}", ""))
            f = d / "val_metrics_corrected.json"
            if not f.exists():
                print(f"[warn] no reval for {name} T={t}: {f}")
                continue
            ts.append(t)
            vals.append(json.loads(f.read_text())["psnr"])
        if not ts:
            continue
        ax.plot(ts, vals, marker="o", label=name)
        for t, v in zip(ts, vals):
            ax.annotate(f"{v:.1f}", (t, v), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    # shading: fraction of z-phases whose window contains a knot (single-
    # knot reachability = min(T, K)/K) as a light band under the curves.
    for t in args.ts:
        frac = min(t, args.K) / args.K
        ax.axvspan(t - 0.4, t + 0.4, ymax=frac, color="0.85", zorder=0)
    ax.axvline(2 * args.K - 1, color="k", ls="--", lw=0.8)
    ax.annotate("T = 2K-1 (DEC-012 well-posed)",
                (2 * args.K - 1, ax.get_ylim()[1]), rotation=90,
                textcoords="offset points", xytext=(4, -4), fontsize=7,
                va="top")
    ax.annotate("single-knot reach ≈ T=7", (7, min(ax.get_ylim())),
                textcoords="offset points", xytext=(6, 6), fontsize=7,
                color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.ts)
    ax.set_xticklabels([str(t) for t in args.ts])
    ax.minorticks_off()
    ax.set_xlabel("axial context depth T (planes per window; K=8)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSRAS v3 | EXP-017 axial context scaling (val, reval best.pt, seed 0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, Path(args.out), "exp017_T_curve")
    print(f"saved -> {args.out}")


def cmd_pcurve(args) -> None:
    """EXP-019: PSNR vs peak photon budget P (z mode, K fixed)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for spec in args.series:                    # "B3=runs_v3/b3_..._ph{P}_..."
        name, pattern = spec.split("=", 1)
        ps, vals = [], []
        for pnt in args.ps:
            d = Path(pattern.format(P=pnt))
            if pnt == 0 and not d.exists():
                # P=0 anchor: the noiseless severity-matrix run (no ph tag)
                d = Path(pattern.format(P=pnt)).with_name(
                    Path(pattern.format(P=pnt)).name.replace("_ph0", ""))
            f = d / "val_metrics_corrected.json"
            if not f.exists():
                print(f"[warn] no reval for {name} P={pnt}: {f}")
                continue
            ps.append(pnt)
            vals.append(json.loads(f.read_text())["psnr"])
        if not ps:
            continue
        ax.plot(ps, vals, marker="o", label=name)
        for pnt, v in zip(ps, vals):
            ax.annotate(f"{v:.1f}", (pnt, v), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    if args.collapse:
        # shaded band: bare-Wiener collapse plateau (EXP-019（四）: λ-mismatch)
        lo, hi = args.collapse
        ax.axhspan(lo, hi, color="0.90", zorder=0)
        ax.annotate("b3 collapse plateau (λ-mismatch, cf. λ sweep 34.5/35.3)",
                    (args.ps[0], (lo + hi) / 2), textcoords="offset points",
                    xytext=(6, 0), fontsize=7, color="0.35", va="center")
    ax.set_xscale("log")
    ax.set_xticks(args.ps)
    ax.set_xticklabels(["∞" if p == 0 else str(p) for p in args.ps])
    ax.minorticks_off()
    ax.set_xlabel("peak photon budget P per measurement (P=∞: noiseless)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSRAS v3 | EXP-019 photon budget dose-response (val, reval best.pt, seed 0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, Path(args.out), "exp019_photon_curve")
    print(f"saved -> {args.out}")


def cmd_curve(args) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for spec in args.series:                    # "B4=runs_v3/b4_2p5d_z_K{K}_..."
        name, pattern = spec.split("=", 1)
        ks, means, los, his = [], [], [], []
        for k in args.ks:
            vals = []
            for seed_tag in ("", "_s1", "_s2"):
                d = Path(pattern.format(K=k)) .with_name(
                    Path(pattern.format(K=k)).name + seed_tag)
                f = d / "val_metrics_corrected.json"
                if f.exists():
                    vals.append(json.loads(f.read_text())["psnr"])
            if vals:
                ks.append(k)
                m = sum(vals) / len(vals)
                means.append(m)
                los.append(m - min(vals))
                his.append(max(vals) - m)
        if not ks:
            print(f"[warn] no data for {name}: {pattern}")
            continue
        ax.errorbar(ks, means, yerr=[los, his], marker="o", capsize=3,
                    label=name)
        for k, m in zip(ks, means):
            ax.annotate(f"{m:.1f}", (k, m), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.ks)
    ax.set_xticklabels([str(k) for k in args.ks])
    ax.minorticks_off()
    ax.set_xlabel("K (keep 1 of K planes; r = 1/K)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSRAS v3 | z-axis sampling severity (val, reval best.pt)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, Path(args.out), "severity_curve_z")
    print(f"saved -> {args.out}")


# --------------------------------------------------------------------------
# Pseudocolor mode (DEC-020 amendment): style inherited from threePM's
# scripts/plot_sparse3d_k2_overview_pseudocolor.py +
# plot_pollen_structural_comparison.py -- two rows (Global full FOV / ROI
# zoom), magma + per-panel colorbar, robust_range (0.5/99.5 pct), yellow ROI
# rectangle picked by brightest gaussian-smoothed GT pixel.
# --------------------------------------------------------------------------

_BIO_PURPLE = None


def _bio_purple_cmap():
    global _BIO_PURPLE
    if _BIO_PURPLE is None:
        from matplotlib.colors import LinearSegmentedColormap
        stops = [(0.00, (0.00, 0.00, 0.00)), (0.18, (0.12, 0.00, 0.22)),
                 (0.42, (0.38, 0.00, 0.58)), (0.68, (0.62, 0.15, 0.92)),
                 (0.88, (0.88, 0.45, 1.00)), (1.00, (1.00, 0.92, 1.00))]
        _BIO_PURPLE = LinearSegmentedColormap.from_list("bio_purple", stops, N=256)
    return _BIO_PURPLE


def _robust_range(img: np.ndarray):
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, [0.5, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def _auto_roi(gt: np.ndarray, size: int = 160):
    sm = gaussian_filter(gt.astype(np.float32), sigma=2.0)
    y, x = np.unravel_index(int(np.argmax(sm)), sm.shape)
    h, w = gt.shape
    half = size // 2
    y0 = int(np.clip(y - half, 0, max(0, h - size)))
    x0 = int(np.clip(x - half, 0, max(0, w - size)))
    return y0, x0, min(h, y0 + size) - y0, min(w, x0 + size) - x0


def _draw_pcolor(fig, ax, img: np.ndarray, title: str, style: str, cmap: str,
                 rng=None, roi=None, crop=None) -> None:
    display = img
    if crop is not None:
        y0, x0, rh, rw = crop
        display = img[y0:y0 + rh, x0:x0 + rw]
    if style == "magma":
        vmin, vmax = rng if rng is not None else _robust_range(img)
        im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    else:                                        # legacy bio-purple, no colorbar
        lo, hi = np.percentile(display, [1.0, 99.0])
        blur = gaussian_filter(np.clip((display - lo) / (hi - lo + 1e-8), 0, 1), 0.8)
        enh = np.clip(np.clip((display - lo) / (hi - lo + 1e-8), 0, 1)
                      + 0.6 * (np.clip((display - lo) / (hi - lo + 1e-8), 0, 1) - blur),
                      0, 1) ** 0.92
        ax.imshow(_bio_purple_cmap()(enh)[..., :3], aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if roi is not None and crop is None:
        y0, x0, rh, rw = roi
        ax.add_patch(mpatches.Rectangle((x0, y0), rw, rh, linewidth=2.0,
                                        edgecolor="yellow", facecolor="none"))


def cmd_pcolor(args) -> None:
    device = torch.device(args.device)
    runs = {}
    for spec in args.runs:                      # "B0=runs_v3/..."
        name, path = spec.split("=", 1)
        runs[name] = Path(path)

    cfg0 = json.loads((runs[next(iter(runs))] / "config.json").read_text())
    t, mode = cfg0["t"], cfg0["mode"]
    idxs = args.items if args.items else [0]
    style = args.style
    tag = "magma+colorbar" if style == "magma" else "bio-purple"

    for i in idxs:
        # GT / Y / X0 come from the first run carrying x0; Y and GT exist in all.
        src = next((n for n in runs
                    if len(json.loads((runs[n] / "config.json").read_text())
                        ["features"]) >= 3), next(iter(runs)))
        specs: list[tuple[str, np.ndarray]] = []
        pred_specs: list[tuple[str, np.ndarray]] = []
        vol_specs: dict[str, torch.Tensor] = {}
        x0_vol: torch.Tensor | None = None
        loaded: dict[str, tuple] = {}
        for name, path in runs.items():
            cfg, ds, model = _load_run(path, args.ckpt, device)
            assert (cfg["mode"], cfg["K"]) == (cfg0["mode"], cfg0["K"]), \
                f"{name} protocol mismatch"
            it = ds.fixed_items[i]
            x, target = it["inputs"], it["target"]
            pred = _predict(model, x, cfg["t"], device)
            q = it["query"].bool()
            if name == src:
                v = x.view(len(cfg["features"]), t, *x.shape[-2:])
                specs.append(("GT", target.view(t, *x.shape[-2:])[q][0].cpu().numpy()))
                specs.append(("Sparse Y", v[0][q][0].cpu().numpy()))
                if len(cfg["features"]) >= 3:
                    specs.append(("X0", v[2][q][0].cpu().numpy()))
                    x0_vol = v[2]
                vol_specs["GT"] = target.view(t, *x.shape[-2:])
                vol_specs["Y"] = v[0]
            vol_specs[name] = pred
            pred_specs.append((name, pred[q][0].cpu().numpy().clip(0, 1)))
            loaded[name] = (cfg, ds, model)
        gt = specs[0][1]
        specs += [(n, im) for n, im in pred_specs]

        gt = specs[0][1]
        # query plane unsampled (r=1/K grid) -> Sparse Y is genuinely all-zero;
        # annotate so the black column reads as "missing plane", not a rendering bug
        y_title = "Sparse Y (unsampled here)" if specs[1][1].max() == 0 else "Sparse Y"
        specs = [(tt if tt in ("GT", "Sparse Y") else
                  f"{tt}  ({psnr(torch.from_numpy(im), torch.from_numpy(gt)).item():.2f} dB)",
                  im) for tt, im in specs]
        specs[1] = (y_title, specs[1][1])
        roi = _auto_roi(gt, args.roi_size)
        rng = _robust_range(gt) if args.share_scale else None

        ncols = len(specs)
        w_scale = 4.2 if style == "magma" else 3.6
        fig, axes = plt.subplots(2, ncols, figsize=(w_scale * ncols, 7.2), dpi=220)
        if ncols == 1:
            axes = axes[None, :]
        for c, (title, img) in enumerate(specs):
            _draw_pcolor(fig, axes[0, c], img, title, style, args.cmap,
                         rng=rng, roi=roi)
            _draw_pcolor(fig, axes[1, c], img, title, style, args.cmap,
                         rng=rng, crop=roi)
        axes[0, 0].text(-0.30, 0.5, "Global", transform=axes[0, 0].transAxes,
                        rotation=90, va="center", fontsize=11)
        axes[1, 0].text(-0.30, 0.5, "ROI zoom", transform=axes[1, 0].transAxes,
                        rotation=90, va="center", fontsize=11)
        fig.suptitle(f"PSRAS v3 | mode={mode} K={cfg0['K']} "
                     f"lam={cfg0['lam_wiener']:g} | {args.ckpt}.pt | item {i} "
                     f"| {tag}", fontsize=13, y=1.02)
        fig.tight_layout()
        _save(fig, Path(args.out), f"pcolor_{mode}_K{cfg0['K']}_item{i}")
        print(f"saved pcolor_{mode}_K{cfg0['K']}_item{i}")

        if mode == "z":                         # XZ cross-section, pseudocolor
            col = args.xz_col if args.xz_col is not None \
                else int(gt.sum(axis=0).argmax().item())
            display_n = len(specs) - len(pred_specs)
            if args.xz_knots > 0:
                # DEC-021 option A: sliding-query volume inference.  Every
                # displayed plane is a supervised-geometry prediction from
                # its own 2K-1 window, so the XZ section is meaningful on
                # all planes (unlike the raw single-pass [T,H,W] output,
                # whose non-query planes are unsupervised).  Titles carry
                # the full-volume PSNR as an additive metric.
                ds_src = next((loaded[n][1] for n in runs
                               if loaded[n][0]["warm_start"] == "wiener"
                               and "x0" in loaded[n][0]["features"]),
                              next(iter(loaded.values()))[1])
                long_i = _build_long_item(ds_src, i, args.xz_knots)
                kd = cfg0["K"] - 1
                l_disp = long_i["gt"].shape[0] - 2 * kd
                sl = slice(kd, kd + l_disp)
                gt_disp = long_i["gt"][sl]
                xz_specs = [("GT (XZ)", gt_disp[:, :, col].numpy().T)]
                xz_specs.append(("Sparse Y (XZ)",
                                 long_i["y"][sl, :, col].numpy().T))
                xz_specs.append(("X0 (XZ)", long_i["x0"][sl, :, col].numpy().T))
                for (tt, _), (name, _) in zip(specs[display_n:], pred_specs):
                    cfg, _, model = loaded[name]
                    pv = _sliding_query_volume(
                        model, long_i, cfg["features"], cfg["K"], cfg["t"],
                        device, residual=cfg["residual"])
                    p_vol = psnr(pv, gt_disp).item()
                    xz_specs.append((f"{tt} (XZ) PSNR(vol)={p_vol:.2f}",
                                     pv[:, :, col].numpy().T))
                xz_note = (f"sliding-query volume, {args.xz_knots} knots "
                           f"({l_disp} planes) | PSNR(vol) over shown planes")
            else:
                xz_specs = [("GT (XZ)", vol_specs["GT"][:, :, col].cpu().numpy().T)]
                if "Y" in vol_specs:
                    xz_specs.append(("Sparse Y (XZ)",
                                     vol_specs["Y"][:, :, col].cpu().numpy().T))
                if x0_vol is not None:
                    xz_specs.append(("X0 (XZ)",
                                     x0_vol[:, :, col].cpu().numpy().T))
                xz_specs += [(tt, vol_specs[n][:, :, col].cpu().numpy().T)
                             for (tt, _), (n, _) in zip(specs[display_n:], pred_specs)]
                xz_note = "stripes = missing planes"
            xz_roi = _auto_roi(xz_specs[0][1], args.roi_size)
            fig2, axes2 = plt.subplots(2, len(xz_specs),
                                       figsize=(w_scale * len(xz_specs), 7.2),
                                       dpi=220)
            if len(xz_specs) == 1:
                axes2 = axes2[None, :]
            for c, (title, img) in enumerate(xz_specs):
                _draw_pcolor(fig2, axes2[0, c], img, title, style,
                             args.cmap, rng=rng, roi=xz_roi)
                _draw_pcolor(fig2, axes2[1, c], img, title, style,
                             args.cmap, rng=rng, crop=xz_roi)
            axes2[0, 0].text(-0.30, 0.5, "Global", transform=axes2[0, 0].transAxes,
                             rotation=90, va="center", fontsize=11)
            axes2[1, 0].text(-0.30, 0.5, "ROI zoom", transform=axes2[1, 0].transAxes,
                             rotation=90, va="center", fontsize=11)
            fig2.suptitle(f"PSRAS v3 | XZ cross-section (col {col}) | mode=z "
                          f"K={cfg0['K']} | {args.ckpt}.pt | item {i} | "
                          f"{xz_note} | {tag}", fontsize=12, y=1.02)
            fig2.tight_layout()
            _save(fig2, Path(args.out), f"pcolor_xz_K{cfg0['K']}_item{i}")
            print(f"saved pcolor_xz_K{cfg0['K']}_item{i}")
    print(f"saved -> {args.out}")


# EXP-020 baseline-comparison matrix (DEC-023): rows in display order;
# runs missing best.pt (e.g. still training) are skipped with a note.
# rvrt b3 xyz/2d are training collapses (EXP-020 结果14) -- rendered with
# an explicit "collapse" tag, excluded from gain reading.
_COLLAPSE_RUNS = {"b3_2d_xy_K8_p8k_rvrt", "b3_2p5d_xyz_K8_p8k_lam3e6_rvrt"}
_COMPARE_MATRIX = {
    "2d_xy": {
        "suptitle": "EXP-020 | 2D situation (xy grid K=8, T=1)",
        "xz": False,
        "rows": [
            ("Ours swin2d", "b1", "b1_2d_xy_K8_p8k"),
            ("Ours swin2d", "b3", "b3_2d_xy_K8_p8k_lam3e6"),
            ("Ours swin2d", "b4", "b4_2d_xy_K8_p8k_lam3e6"),
            ("Video Swin (official)", "b1", "b1_2d_xy_K8_p8k_vswin"),
            ("Video Swin (official)", "b3", "b3_2d_xy_K8_p8k_vswin"),
            ("SwinUNETR (official)", "b1", "b1_2d_xy_K8_p8k_sunetr"),
            ("SwinUNETR (official)", "b3", "b3_2d_xy_K8_p8k_sunetr"),
            ("RVRT (official)", "b1", "b1_2d_xy_K8_p8k_rvrt"),
            ("RVRT (official)", "b3", "b3_2d_xy_K8_p8k_rvrt"),
        ],
    },
    "2p5d_z": {
        "suptitle": "EXP-020 | 2.5D situation (axial knots, z K=8, T=15)",
        "xz": True,
        "rows": [
            ("Ours swin2d", "b1", "b1_2p5d_z_K8_p8k"),
            ("Ours swin2d", "b3", "b3_2p5d_z_K8_p8k_lam3e6"),
            ("Ours swin2d", "b4", "b4_2p5d_z_K8_p8k_lam3e6"),
            ("Video Swin (official)", "b1", "b1_2p5d_z_K8_p8k_vswin"),
            ("Video Swin (official)", "b3", "b3_2p5d_z_K8_p8k_lam3e6_vswin"),
            ("SwinUNETR (official)", "b1", "b1_2p5d_z_K8_p8k_sunetr"),
            ("SwinUNETR (official)", "b3", "b3_2p5d_z_K8_p8k_lam3e6_sunetr"),
            ("RVRT (official)", "b1", "b1_2p5d_z_K8_p8k_rvrt"),
            ("RVRT (official)", "b3", "b3_2p5d_z_K8_p8k_lam3e6_rvrt"),
        ],
    },
    "2p5d_xyz": {
        "suptitle": "EXP-020 | 3D situation (axial gaps + lateral grid, xyz K=8, T=15)",
        "xz": True,
        "rows": [
            ("Ours swin2d", "b1", "b1_2p5d_xyz_K8_p8k"),
            ("Ours swin2d", "b3", "b3_2p5d_xyz_K8_p8k_lam3e6"),
            ("Ours swin2d", "b4", "b4_2p5d_xyz_K8_p8k_lam3e6"),
            ("Video Swin (official)", "b1", "b1_2p5d_xyz_K8_p8k_vswin"),
            ("Video Swin (official)", "b3", "b3_2p5d_xyz_K8_p8k_lam3e6_vswin"),
            ("SwinUNETR (official)", "b1", "b1_2p5d_xyz_K8_p8k_sunetr"),
            ("SwinUNETR (official)", "b3", "b3_2p5d_xyz_K8_p8k_lam3e6_sunetr"),
            ("RVRT (official)", "b1", "b1_2p5d_xyz_K8_p8k_rvrt"),
            ("RVRT (official)", "b3", "b3_2p5d_xyz_K8_p8k_lam3e6_rvrt"),
        ],
    },
}


def _reval_of(run_dir: Path):
    f = run_dir / "val_metrics_corrected.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return float(d["psnr"]), float(d["ssim"])


def _compare_tables(sit: str, rows: list[dict], out_dir: Path) -> None:
    """Quantitative deliverable: CSV (long) + MD (pivot with Wiener-gain col)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table.csv", "w") as f:
        f.write("arch,arm,run,reval_psnr,reval_ssim,note\n")
        for r in rows:
            if r["reval"] is None:
                p, s, note = "", "", "pending"
            else:
                p, s = f'{r["reval"][0]:.4f}', f'{r["reval"][1]:.6f}'
                note = "training collapse" if r["collapse"] else ""
            f.write(f'{r["arch"]},{r["arm"]},{r["name"]},{p},{s},{note}\n')

    archs: list[str] = []
    for r in rows:
        if r["arch"] not in archs:
            archs.append(r["arch"])
    lines = [f"# EXP-020 quantitative comparison — {sit}", "",
             "reval (per-item PSNR/SSIM over 8 val items, best.pt). "
             "gain = b3 − b1 (Wiener x0 vs no warm start).",
             "Single-seed (seed 0) protocol throughout, matching the external "
             "baselines; EXP-017 severity-matrix two-seed means (e.g. ours z "
             "b3 = 44.73) differ slightly from these seed-0 values.", "",
             "| Arch | b1 (y+mask) | b3 (Wiener x0) | gain (b3−b1) | b4 (fusion) |",
             "|---|---|---|---|---|"]
    for a in archs:
        cells = {r["arm"]: r for r in rows if r["arch"] == a}

        def fmt(arm: str) -> str:
            r = cells.get(arm)
            if not r or r["reval"] is None:
                return "—"
            return f'{r["reval"][0]:.2f} / {r["reval"][1]:.4f}'

        b1, b3 = cells.get("b1"), cells.get("b3")
        if (b1 and b3 and b1["reval"] is not None and b3["reval"] is not None
                and not (b1["collapse"] or b3["collapse"])):
            gtxt = f'**{b3["reval"][0] - b1["reval"][0]:+.2f}**'
        elif b1 and b3 and (b1["collapse"] or b3["collapse"]):
            gtxt = "collapse (excluded)"
        else:
            gtxt = "—"
        lines.append(f"| {a} | {fmt('b1')} | {fmt('b3')} | {gtxt} | {fmt('b4')} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")


def _chunk_for(cfg: dict) -> int:
    return {"rvrt": 1, "sunetr": 2}.get(cfg["model"].get("kind", "swin2d"), 8)


def _compare_figure(sit: str, spec: dict, rows: list[dict], out_dir: Path,
                    item: int, device: torch.device, args) -> None:
    avail = [r for r in rows
             if (r["dir"] / f"{args.ckpt}.pt").exists() and r["reval"] is not None]
    skipped = [r["name"] for r in rows if r not in avail]
    if skipped:
        print(f"[compare:{sit}] skipped (no ckpt): {skipped}")
    if not avail:
        return

    # Shared GT / Sparse Y source: first run carrying x0 (same protocol, seeds).
    src = next((r for r in avail if "x0" in json.loads(
        (r["dir"] / "config.json").read_text())["features"]), avail[0])
    cfg_s, ds_s, _ = _load_run(src["dir"], args.ckpt, device)

    if spec["xz"]:
        long_i = _build_long_item(ds_s, item, args.xz_knots)
        kd = cfg_s["K"] - 1
        gt_disp = long_i["gt"][kd:kd + (long_i["gt"].shape[0] - 2 * kd)]
        y_disp = long_i["y"][kd:kd + gt_disp.shape[0]].numpy()
        col = int(gt_disp.sum(axis=(1, 2)).argmax().item())  # brightest XZ col
        gt_img = gt_disp[:, :, col].numpy().T
        y_img = y_disp[:, :, col].T
        rng = _robust_range(gt_img)
    else:
        it = ds_s.fixed_items[item]
        x_s, tgt = it["inputs"], it["target"]
        t_s = cfg_s["t"]
        v = x_s.view(len(cfg_s["features"]), t_s, *x_s.shape[-2:])
        q = it["query"].bool()
        gt_img = tgt.view(t_s, *x_s.shape[-2:])[q][0].cpu().numpy()
        y_img = v[0][q][0].cpu().numpy()
        rng = _robust_range(gt_img)

    n = len(avail)
    fig, axes = plt.subplots(n, 4, figsize=(13.2, 2.35 * n), dpi=200)
    if n == 1:
        axes = axes[None, :]
    err_vmax = 0.0
    recs = []
    for r in avail:
        cfg, ds_r, model = _load_run(r["dir"], args.ckpt, device)
        if spec["xz"]:
            pv = _sliding_query_volume(model, long_i, tuple(cfg["features"]),
                                       cfg["K"], cfg["t"], device,
                                       chunk=_chunk_for(cfg),
                                       residual=cfg["residual"])
            rec = pv[:, :, col].numpy().T
        else:
            it_r = ds_r.fixed_items[item]
            x2 = it_r["inputs"]
            pred = _predict(model, x2, cfg["t"], device)
            if cfg["residual"]:
                pred = x2.view(len(cfg["features"]), cfg["t"],
                               *x2.shape[-2:])[2] + pred
            rec = pred[it_r["query"].bool()][0].cpu().numpy()
        recs.append(rec.clip(0, 1))
        err_vmax = max(err_vmax, float(np.percentile(np.abs(rec - gt_img), 99.5)))

    for i, (r, rec) in enumerate(zip(avail, recs)):
        err = np.abs(rec - gt_img)
        label = f'{r["arch"]} {r["arm"]}'
        if r["reval"] is not None:
            label += f'\nreval {r["reval"][0]:.2f} dB'
        if r["collapse"]:
            label += "\n(training collapse)"
        _draw_pcolor(fig, axes[i, 0], y_img, "", "magma", "magma", rng=rng)
        _draw_pcolor(fig, axes[i, 1], rec, "", "magma", "magma", rng=rng)
        _draw_pcolor(fig, axes[i, 2], gt_img, "", "magma", "magma", rng=rng)
        _draw_pcolor(fig, axes[i, 3], err, "", "magma", "inferno",
                     rng=(0.0, max(err_vmax, 1e-6)))
        axes[i, 0].text(-0.04, 0.5, label, transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=7)
        if spec["xz"]:
            ax_in = axes[i, 1].inset_axes([0.58, 0.04, 0.40, 0.26])
            ax_in.plot(gt_img.mean(axis=1), color="0.35", ls="--", lw=0.8,
                       label="GT")
            ax_in.plot(rec.mean(axis=1), color="#4C72B0", lw=0.9, label="recon")
            ax_in.set_xticks([]), ax_in.set_yticks([])
            for sp in ax_in.spines.values():
                sp.set_color("yellow")
    for c, t in enumerate(["Sparse Y", "Recon (+z-profile)", "GT", "|err|"]):
        axes[0, c].set_title(t, fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    loc = f"XZ col {col} | " if spec["xz"] else ""
    view = "sliding-query volume" if spec["xz"] else "query slice"
    fig.suptitle(f'{spec["suptitle"]} | {loc}item {item} | '
                 f"{args.ckpt}.pt | {view}", fontsize=12, y=1.005)
    fig.tight_layout()
    _save(fig, out_dir, f"compare_{sit}_item{item}")
    print(f"saved compare_{sit}_item{item} ({n} rows)")


_FUSION_RUNS = {
    "2d_xy": ("b4_2d_xy_K8_p8k_lam3e6", False),
    "2p5d_z": ("b4_2p5d_z_K8_p8k_lam3e6", True),
    "2p5d_xyz": ("b4_2p5d_xyz_K8_p8k_lam3e6", True),
}


@torch.no_grad()
def _gate_single(model, x: torch.Tensor, t: int, device: torch.device):
    """Query-slice prediction + mean-channel gate map for a 2D-situation item."""
    stem = model[0]
    cap: list[torch.Tensor] = []
    h = stem.gate.register_forward_hook(
        lambda m, i, o: cap.append(o.detach().cpu()))
    pred = model(x[None].to(device)).float().view(t, *x.shape[-2:])
    h.remove()
    g = torch.sigmoid(cap[0][0]).mean(0)             # [base,H,W] -> [H,W]
    return pred.cpu(), g


@torch.no_grad()
def _sliding_query_gate(model, feats_long, feat_order, k, t_win, device,
                        chunk: int = 8):
    """_sliding_query_volume + per-query-plane gate maps (b4 fusion stem).

    The 2D fusion stem produces ONE [base, H, W] gate map per window
    (channels e_y/e_w/e_s mix all t planes), so every query plane keeps the
    gate of its own centred window.  Returns (pred [L,H,W], gate [L,H,W]).
    """
    l_total = feats_long["gt"].shape[0]
    l_disp = l_total - 2 * (k - 1)
    half = t_win // 2
    stem = model[0]
    caps: list[torch.Tensor] = []
    h = stem.gate.register_forward_hook(
        lambda m, i, o: caps.append(o.detach().cpu()))
    outs: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    buf: list[torch.Tensor] = []

    def _flush() -> None:
        if not buf:
            return
        x = torch.stack(buf).to(device)
        pred = model(x).float().view(-1, t_win, *x.shape[-2:])[:, half]
        g = torch.sigmoid(caps.pop(0)).mean(1)       # [B,base,H,W] -> [B,H,W]
        outs.extend(t.cpu() for t in pred)
        gates.extend(g[i] for i in range(g.shape[0]))
        buf.clear()

    for p in range(k - 1, k - 1 + l_disp):
        buf.append(torch.cat([feats_long[f][p - k + 1: p + k]
                              for f in feat_order], 0))
        if len(buf) >= chunk:
            _flush()
    _flush()
    h.remove()
    assert len(gates) == l_disp, (len(gates), l_disp)
    return torch.stack(outs), torch.stack(gates)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _fusion_figure(sit: str, run_name: str, has_xz: bool, out_dir: Path,
                   item: int, xz_knots: int, ckpt: str,
                   device: torch.device) -> None:
    run_dir = Path(__file__).parents[1] / "runs_v3" / run_name
    cfg, ds, model = _load_run(run_dir, ckpt, device)
    feat_order = tuple(cfg["features"])
    n_feat = len(feat_order)

    if has_xz:
        long_i = _build_long_item(ds, item, xz_knots)
        kd = cfg["K"] - 1
        gt_disp = long_i["gt"][kd:kd + (long_i["gt"].shape[0] - 2 * kd)]
        up_disp = long_i["uncertainty"][kd:kd + gt_disp.shape[0]].numpy()
        cs_disp = long_i["confidence"][kd:kd + gt_disp.shape[0]].numpy()
        y_disp = long_i["y"][kd:kd + gt_disp.shape[0]].numpy()
        pred, gate = _sliding_query_gate(model, long_i, feat_order,
                                         cfg["K"], cfg["t"], device,
                                         chunk=_chunk_for(cfg))
        col = int(gt_disp.sum(axis=(1, 2)).argmax().item())
        views = {k2: v[:, :, col].T for k2, v in
                 [("gt", gt_disp.numpy()), ("pred", pred.numpy()),
                  ("up", up_disp), ("cs", cs_disp), ("y", y_disp)]}
        views["gate"] = gate.numpy()[:, :, col].T
        view = f"XZ col {col} | sliding-query"
    else:
        it = ds.fixed_items[item]
        x = it["inputs"]
        q = it["query"].bool()
        v = x.view(n_feat, cfg["t"], *x.shape[-2:])
        pred, gate = _gate_single(model, x, cfg["t"], device)
        if cfg["residual"]:
            pred = v[2] + pred
        views = {"gt": it["target"].view(cfg["t"], *x.shape[-2:])[q][0].numpy(),
                 "pred": pred[q][0].numpy(),
                 "up": v[4][q][0].numpy(), "cs": v[3][q][0].numpy(),
                 "y": v[0][q][0].numpy(), "gate": gate.numpy()}
        view = "query slice"

    err = np.abs(views["pred"] - views["gt"])
    views["err"] = err
    r_gate = _corr(views["gate"], err)
    r_up = _corr(views["up"], err)
    r_gup = _corr(views["gate"], views["up"])
    reval = _reval_of(run_dir)
    label_ps = (f"reval {reval[0]:.2f} dB | corr(gate,|err|)={r_gate:+.3f} | "
                f"corr(U_p,|err|)={r_up:+.3f} | corr(gate,U_p)={r_gup:+.3f}")

    cols = [("Recon", "pred", "magma", "magma", None),
            ("Gate g̅ (fusion prior)", "gate", "viridis", "viridis", (0, 1)),
            ("U_p (input physics)", "up", "magma", "magma", (0, 1)),
            ("C_s (input confidence)", "cs", "magma", "magma", (0, 1)),
            ("|err|", "err", "magma", "inferno", None)]
    err_vmax = max(float(np.percentile(err, 99.5)), 1e-6)
    rng_gt = _robust_range(views["gt"])
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.4), dpi=200)
    for c, (title, key, cmap, cb, rng) in enumerate(cols):
        if rng is None:
            rng = (0.0, err_vmax) if key == "err" else rng_gt
        _draw_pcolor(fig, axes[c], views[key], "", cmap, cb, rng=rng)
        axes[c].set_title(title, fontsize=10)
        axes[c].axis("off")
    fig.suptitle(f"EXP-020 fusion interpretability | Ours b4 {sit} | {view}"
                 f" | item {item} | {label_ps}", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f"fusion_{sit}_item{item}")
    print(f"saved fusion_{sit}_item{item}  ({label_ps})")


def cmd_fusion(args) -> None:
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sit, (run_name, has_xz) in _FUSION_RUNS.items():
        run_dir = Path(__file__).parents[1] / "runs_v3" / run_name
        if not (run_dir / f"{args.ckpt}.pt").exists():
            print(f"[fusion:{sit}] skipped (no ckpt): {run_name}")
            continue
        for item in args.items:
            _fusion_figure(sit, run_name, has_xz, out_dir, item,
                           args.xz_knots, args.ckpt, device)


def cmd_compare(args) -> None:
    device = torch.device(args.device)
    root = Path(__file__).parents[1] / "runs_v3"
    out_root = Path(args.out)
    for sit, spec in _COMPARE_MATRIX.items():
        rows = [dict(arch=a, arm=arm, name=name, dir=root / name,
                     reval=_reval_of(root / name), collapse=name in _COLLAPSE_RUNS)
                for a, arm, name in spec["rows"]]
        _compare_tables(sit, rows, out_root / sit)
        print(f"[compare:{sit}] table.csv / table.md written")
        if args.tables_only:
            continue
        for item in args.items:
            _compare_figure(sit, spec, rows, out_root / sit, item, device, args)
    print(f"saved -> {out_root}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("panel")
    pp.add_argument("--runs", nargs="+", required=True,
                    help="name=run_dir pairs, display order preserved")
    pp.add_argument("--out", required=True)
    pp.add_argument("--items", type=int, nargs="*", default=None)
    pp.add_argument("--ckpt", choices=["best", "last"], default="best")
    pp.add_argument("--xz_col", type=int, default=None,
                    help="XZ cross-section column (default: brightest GT col)")
    pp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pp.set_defaults(func=cmd_panel)
    pq = sub.add_parser("pcolor",
                        help="pseudocolor overview: Global row + ROI zoom row")
    pq.add_argument("--runs", nargs="+", required=True,
                    help="name=run_dir pairs, display order preserved")
    pq.add_argument("--out", required=True)
    pq.add_argument("--items", type=int, nargs="*", default=None,
                    help="val item indices (default: [0], one figure per item)")
    pq.add_argument("--ckpt", choices=["best", "last"], default="best")
    pq.add_argument("--roi_size", type=int, default=160)
    pq.add_argument("--style", choices=["magma", "purple"], default="magma")
    pq.add_argument("--cmap", default="magma")
    pq.add_argument("--share_scale", action="store_true",
                    help="use GT robust_range for all panels instead of per-panel")
    pq.add_argument("--xz_col", type=int, default=None)
    pq.add_argument("--xz_knots", type=int, default=0,
                    help="z mode: >0 enables sliding-query full-volume XZ "
                         "(DEC-021 A); value = number of knots in the shown "
                         "range, K-1 margin planes added each side")
    pq.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pq.set_defaults(func=cmd_pcolor)
    pc = sub.add_parser("curve")
    pc.add_argument("--series", nargs="+", required=True,
                    help="name=pattern with {K} placeholder")
    pc.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    pc.add_argument("--out", required=True)
    pc.set_defaults(func=cmd_curve)
    pp2 = sub.add_parser("pcurve")
    pp2.add_argument("--series", nargs="+", required=True,
                     help="name=pattern with {P} placeholder; P=0 point "
                          "falls back to the noiseless run without the ph tag")
    pp2.add_argument("--ps", type=int, nargs="+", default=[0, 50, 20, 10, 5])
    pp2.add_argument("--collapse", type=float, nargs=2, default=None,
                     metavar=("LO", "HI"),
                     help="shade a horizontal band [LO,HI] as collapse plateau")
    pp2.add_argument("--out", required=True)
    pp2.set_defaults(func=cmd_pcurve)
    pt = sub.add_parser("tcurve")
    pt.add_argument("--series", nargs="+", required=True,
                    help="name=pattern with {T} placeholder; T=2K-1 point "
                         "falls back to the anchor run without the T tag")
    pt.add_argument("--ts", type=int, nargs="+", default=[1, 3, 7, 15, 31])
    pt.add_argument("--K", type=int, default=8)
    pt.add_argument("--out", required=True)
    pt.set_defaults(func=cmd_tcurve)
    cm = sub.add_parser("compare",
                        help="EXP-020 baseline comparison: 3 situations, "
                             "quantitative tables + paper-grade figures")
    cm.add_argument("--out", default="eval_viz/baseline_compare")
    cm.add_argument("--items", type=int, nargs="*", default=[0],
                    help="val item indices, one figure per item")
    cm.add_argument("--xz_knots", type=int, default=3,
                    help="knots in shown z/xyz range (display planes = "
                         "(n-1)*K+1); K-1 margin planes added each side")
    cm.add_argument("--ckpt", choices=["best", "last"], default="best")
    cm.add_argument("--tables_only", action="store_true")
    cm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cm.set_defaults(func=cmd_compare)
    fu = sub.add_parser("fusion",
                        help="b4 fusion interpretability: gate map g vs "
                             "input U_p/C_s fields vs |err|, with corr stats")
    fu.add_argument("--out", default="eval_viz/fusion_uncertainty")
    fu.add_argument("--items", type=int, nargs="*", default=[0],
                    help="val item indices, one figure per item")
    fu.add_argument("--xz_knots", type=int, default=3)
    fu.add_argument("--ckpt", choices=["best", "last"], default="best")
    fu.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    fu.set_defaults(func=cmd_fusion)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""EXP-027 noise_stress visualization (paired F1/F2/F3 input, clean F0 GT).

Style inherited from visualize_v3.py (DEC-020/021): grayscale panels
vmin=0/vmax=1, axes off, column titles on the top row, PSNR annotation
under each reconstruction panel, PNG (240 dpi) + PDF double format.

Sub-commands:
  panel  -- one figure per evaluation point; rows = noise stacks
           (F1/F2/F3 = residual std .072/.180/.309), columns = GT (clean
           F0, query slice) | Noisy raw frame | Sparse Y | X0 | U15_s0
           | U_noAZ_s0.  For 2.5d modes Sparse Y shows the window slice
           with the most measured pixels (the query plane itself is
           unmeasured and would render black).
  curve  -- PSNR vs noise level, one subplot per point; U15/U_noAZ mean
            +- per-seed spread, x0 prior dashed, noisy-input PSNR dotted.

Outputs to eval_viz/exp027_noise_stress/ (paths recorded in PROJECT_LOG).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.datasets import make_val_dataset          # noqa: E402
from metrics.image_metrics import psnr              # noqa: E402
from physics.psf import PSFConfig                   # noqa: E402
from scripts.train import PRESETS, build_model      # noqa: E402

DATA = Path(__file__).parents[1] / "runs_v3/data/cidc25_v3"
OUT = Path(__file__).parents[1] / "eval_viz/exp027_noise_stress"
POINTS = [("y", 2, 0), ("y", 4, 0), ("y", 8, 0), ("yz2", 4, 4), ("xyz2", 4, 4)]
NOISE = ["F1", "F2", "F3"]
STD = {"F1": 0.072, "F2": 0.180, "F3": 0.309}
RAW_PSNR = {"F1": 22.8, "F2": 14.8, "F3": 10.1}   # noisy stack vs clean F0 (measured)
RUNS = {"U15_s0": "runs_v3/UNIV4_U15_s0", "U_noAZ_s0": "runs_v3/UNIV4_U_noAZ_s0"}
FEATS = ("y", "mask", "x0", "confidence", "uncertainty")


def tag_of(mode: str, K: int, kz: int) -> str:
    return f"{mode}_K{K}" if mode == "y" else f"{mode}_K{K}x{kz}"


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _point_psf(mode: str, run_cfg: dict) -> PSFConfig:
    """reval.py point-override convention (verified insensitive: query-plane
    y is unmeasured and the matched Wiener restores measured planes)."""
    sz = run_cfg["psf"]["sigma_z_px"] if mode != "y" else 0.0
    return PSFConfig(sigma_xy_um=run_cfg["psf"]["sigma_xy_px"], sigma_z_um=sz,
                     voxel_xy_um=1.0, voxel_z_um=1.0)


def _build_items(mode: str, K: int, kz: int, F: str):
    """Paired val items + model inputs for one noise stack at one point."""
    regime = "2d" if mode == "y" else "2.5d"
    t = 1 if mode == "y" else (2 * kz - 1)
    cfg = json.loads((Path(RUNS["U15_s0"]) / "config.json").read_text())
    ds = make_val_dataset(DATA, "noise_stress", regime, mode=mode, K=K, K2=kz,
                          psf=_point_psf(mode, cfg), patch_hw=(256, 256),
                          warm_start_kind=cfg["warm_start"], lam=cfg["lam_wiener"],
                          features=FEATS, gt_stack="F0", stacks_override=[F])
    # noisy raw frame at the item-2 coordinates (make_val_dataset geometry)
    stack = np.load(DATA / "stacks" / f"{F}.npy", mmap_mode="r")
    th = ds.T
    t0s = np.linspace(750, max(750, 1500 - th), 8, dtype=int)
    y0 = (stack.shape[1] - 256) // 2
    x0 = (stack.shape[2] - 256) // 2
    return ds, t, int(t0s[2]), y0, x0, np.asarray(stack[t0s[2]:t0s[2] + th,
                                                       y0:y0 + 256, x0:x0 + 256],
                                                  dtype=np.float32)


def _load_model(run_dir: str, t: int):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    model = build_model(PRESETS[cfg["preset"]], t, cfg["model"]["base"],
                        cfg["model"]["window"], cfg["model"].get("kind", "swin2d"),
                        axial_at=cfg["model"].get("axial_at", "all"),
                        output=cfg["model"].get("output", "all"),
                        use_az=cfg["model"].get("use_az", True))
    sd = torch.load(Path(run_dir) / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"])
    return model.eval()


@torch.no_grad()
def cmd_panel(args) -> None:
    device = torch.device(args.device)
    for mode, K, kz in POINTS:
        tag = tag_of(mode, K, kz)
        qi = None                        # query slice index set per point below
        rows = []
        for F in NOISE:
            ds, t, t0, y0, x0, raw = _build_items(mode, K, kz, F)
            it = ds.fixed_items[2]
            x, target = it["inputs"], it["target"]
            q = it["query"].bool()
            qi = int(np.argmax(it["query"].numpy()))
            v = x.view(len(FEATS), t, 256, 256)
            # Sparse Y: query slice for y; most-measured slice for 2.5d
            mi = int(v[1].sum(dim=(1, 2)).argmax().item())
            y_img = v[0][mi] if mode != "y" else v[0][qi]
            row = {"gt": target.view(t, 256, 256)[qi], "raw": torch.from_numpy(raw[qi]),
                   "y": y_img, "x0": v[2][qi]}
            for name, rd in RUNS.items():
                model = _load_model(rd, t).to(device)
                pred = model(x[None].to(device)).float().view(t, 256, 256)
                row[name] = pred[qi].cpu()
                del model
            rows.append(row)
        cols = ["GT (F0 clean)", "Noisy raw", "Sparse Y", "X0"] + list(RUNS)
        col_key = {"GT (F0 clean)": "gt", "Noisy raw": "raw",
                   "Sparse Y": "y", "X0": "x0"}
        fig, axes = plt.subplots(3, len(cols), figsize=(1.9 * len(cols), 1.9 * 3 + 1))
        for c, title in enumerate(cols):
            axes[0, c].set_title(title, fontsize=10)
            for r, F in enumerate(NOISE):
                ax = axes[r, c]
                img = rows[r][col_key.get(title, title)]
                ax.imshow(img.numpy().clip(0, 1), cmap="gray", vmin=0, vmax=1)
                if title not in ("GT (F0 clean)", "Noisy raw", "Sparse Y"):
                    p = psnr(img, rows[r]["gt"]).item()
                    ax.text(0.5, -0.06, f"{p:.2f} dB", transform=ax.transAxes,
                            ha="center", va="top", fontsize=8)
                ax.axis("off")
            axes[r, 0].set_ylabel(f"{F}\n(std {STD[F]:.3f})", fontsize=9)
        fig.suptitle(f"EXP-027 noise_stress | {tag} | item 2 | last.pt, seed 0",
                     fontsize=12)
        fig.tight_layout()
        _save(fig, f"panel_{tag}")
        print(f"[ok] panel_{tag}")


def cmd_curve(args) -> None:
    import statistics as st
    from pathlib import Path as P
    x0 = json.loads((P(__file__).parents[1] / "runs_v3/exp027_x0_prior_noise.json").read_text())
    fig, axes = plt.subplots(1, len(POINTS), figsize=(3.1 * len(POINTS), 3.2))
    for ax, (mode, K, kz) in zip(axes, POINTS):
        tag = tag_of(mode, K, kz)
        xs = np.arange(4)
        labels = ["F0\n(clean)"] + [f"{F}\nstd {STD[F]:.3f}" for F in NOISE]
        for arm, pat in (("U15", "-o"), ("U_noAZ", "-s")):
            runs = [f"runs_v3/UNIV4_{arm}_s{i}" for i in range(3)]
            mean, spread = [], []
            for F in ["F0"] + NOISE:
                name = f"metrics_test_{tag}.json" if F == "F0" else \
                    f"metrics_noise_stress_{F}_{tag}.json"
                vals = [json.load(open(P(r) / name))["psnr"] for r in runs]
                mean.append(st.mean(vals))
                spread.append(max(abs(v - st.mean(vals)) for v in vals))
            mean, spread = np.array(mean), np.array(spread)
            ax.plot(xs, mean, pat, label=arm, ms=4)
            ax.fill_between(xs, mean - spread, mean + spread, alpha=0.15)
        ax.plot(xs, [x0[tag][F] for F in ["F0"] + NOISE], "--", color="gray",
                label="x0 prior", lw=1)
        ax.plot(xs[1:], [RAW_PSNR[F] for F in NOISE], ":", color="black",
                label="noisy input", lw=1)
        ax.set_title(tag, fontsize=11)
        ax.set_xticks(xs, labels, fontsize=7)
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("EXP-027 noise_stress degradation curves (mean ± max per-seed dev)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "noise_curve")
    print("[ok] noise_curve")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("panel")
    sub.add_parser("curve")
    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    {"panel": cmd_panel, "curve": cmd_curve}[args.cmd](args)


if __name__ == "__main__":
    main()

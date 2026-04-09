#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


PROJECT_DIR = Path(
    "/data2/wjb/threePM/projects/three-photon-spatiotemporal-sparse-sampling-fast-high-quality-reconstruction"
)
FIG_DIR = PROJECT_DIR / "figures_stage3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILES = {
    "Baseline-2D (pilot)": Path(
        "/data2/wjb/threePM/eval_results/stage2_baseline2d_pilot_eval/eval/summary_stage2_baseline2d_pilot.csv"
    ),
    "2.5D w=3 (pilot)": Path(
        "/data2/wjb/threePM/eval_results/stage2_2p5d_w3_pilot_eval/eval/summary_stage2_2p5d_w3_pilot.csv"
    ),
    "2.5D w=5 (pilot)": Path(
        "/data2/wjb/threePM/eval_results/stage2_2p5d_w5_e6_eval/eval/summary_stage2_2p5d_w5_e6.csv"
    ),
    "2.5D w=3 (stage3 full)": Path(
        "/data2/wjb/threePM/eval_results/stage3_2p5d_w3_e180_full_eval/eval/summary_stage3_2p5d_w3_e180_full.csv"
    ),
}

METRICS_STAGE3 = Path(
    "/data2/wjb/threePM/eval_results/stage3_2p5d_w3_e180_full_eval/eval/metrics_stage3_2p5d_w3_e180_full.csv"
)
IMG_DIR_STAGE3 = Path(
    "/data2/wjb/threePM/eval_results/stage3_2p5d_w3_e180_full_eval/eval/images/stage3_2p5d_w3_e180_full"
)


def read_summary(path: Path):
    out = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r["metric"]
            try:
                out[m] = float(r["mean"]) if r["mean"] else np.nan
            except Exception:
                out[m] = np.nan
    return out


def plot_summary_bars(all_data):
    labels = list(all_data.keys())
    psnr = [all_data[k].get("psnr", np.nan) for k in labels]
    ssim = [all_data[k].get("ssim", np.nan) for k in labels]
    lpips = [all_data[k].get("lpips", np.nan) for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5), dpi=180)
    plt.bar(x - width, psnr, width=width, label="PSNR (dB)")
    plt.bar(x, ssim, width=width, label="SSIM")
    plt.bar(x + width, lpips, width=width, label="LPIPS (lower better)")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Metric value")
    plt.title("Cross-stage metric comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_metric_comparison.png")
    plt.close()


def plot_stage3_distribution():
    psnr_vals, ssim_vals, dpsnr_vals = [], [], []
    with METRICS_STAGE3.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            psnr_vals.append(float(r["psnr"]))
            ssim_vals.append(float(r["ssim"]))
            dpsnr_vals.append(float(r["delta_psnr"]))

    plt.figure(figsize=(12, 4), dpi=180)
    plt.subplot(1, 3, 1)
    plt.hist(psnr_vals, bins=40)
    plt.title("PSNR distribution")
    plt.xlabel("PSNR")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.hist(ssim_vals, bins=40)
    plt.title("SSIM distribution")
    plt.xlabel("SSIM")

    plt.subplot(1, 3, 3)
    plt.hist(dpsnr_vals, bins=40)
    plt.title("Delta PSNR distribution")
    plt.xlabel("Delta PSNR")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_stage3_distributions.png")
    plt.close()


def _norm_joint(a, b, c):
    cat = np.concatenate([a.ravel(), b.ravel(), c.ravel()])
    lo, hi = np.percentile(cat, [0.5, 99.5])
    den = hi - lo + 1e-8
    return (
        np.clip((a - lo) / den, 0, 1),
        np.clip((b - lo) / den, 0, 1),
        np.clip((c - lo) / den, 0, 1),
    )


def plot_qualitative_examples():
    # Pick representative samples by quantiles on PSNR, but only from saved image files
    records = []
    with METRICS_STAGE3.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            records.append((r["image"], float(r["psnr"])))
    available = {p.name.replace("_noisy.npy", "") for p in IMG_DIR_STAGE3.glob("*_noisy.npy")}
    records = [x for x in records if x[0] in available]
    if len(records) < 3:
        raise RuntimeError("Not enough saved qualitative samples for figure generation.")
    records.sort(key=lambda x: x[1])
    picks = [
        records[len(records) // 10],
        records[len(records) // 2],
        records[len(records) * 9 // 10],
    ]

    plt.figure(figsize=(12, 9), dpi=180)
    for i, (name, psnr) in enumerate(picks):
        noisy = np.load(IMG_DIR_STAGE3 / f"{name}_noisy.npy")
        pred = np.load(IMG_DIR_STAGE3 / f"{name}_pred.npy")
        gt = np.load(IMG_DIR_STAGE3 / f"{name}_gt.npy")
        n, p, g = _norm_joint(noisy, pred, gt)

        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(n, cmap="gray")
        plt.title(f"{name} noisy")
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(p, cmap="gray")
        plt.title(f"pred (PSNR={psnr:.2f})")
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(g, cmap="gray")
        plt.title("ground truth")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_stage3_qualitative_examples.png")
    plt.close()


def main():
    all_data = {name: read_summary(path) for name, path in SUMMARY_FILES.items()}
    plot_summary_bars(all_data)
    plot_stage3_distribution()
    plot_qualitative_examples()
    print(f"[OK] Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()

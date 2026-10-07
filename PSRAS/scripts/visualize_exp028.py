"""EXP-028 regime-weighting visualization: 15-point suite curves.

One subplot per suite (Y / YZ / XYZ); lines = U15W (weighted) vs U15
(point-uniform, EXP-026) vs SXYZ (specialist, EXP-026), mean +- max
per-seed deviation.  Style inherited from visualize_v3.py (DEC-020/021):
PNG 240 dpi + PDF.

    python scripts/visualize_exp028.py curve
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

OUT = Path(__file__).parents[1] / "eval_viz/exp028_regime_weight"
TAGS = {
    "Y suite (y)": [("y_K2", "K2"), ("y_K4", "K4"), ("y_K8", "K8")],
    "YZ suite (yz2, ky x kz)": [(f"yz2_K{a}x{b}", f"{a}x{b}")
                                for a, b in [(2,2),(2,4),(2,8),(4,2),(4,4),(8,2)]],
    "XYZ suite (xyz2, klat x kz)": [(f"xyz2_K{a}x{b}", f"{a}x{b}")
                                    for a, b in [(2,2),(2,4),(2,8),(4,2),(4,4),(8,2)]],
}
ARMS = {"U15W (1:1:2)": "UNIV4_U15W", "U15 (point-uniform)": "UNIV4_U15",
        "SXYZ (specialist)": "UNIV4_SXYZ"}


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, (suite, tags) in zip(axes, TAGS.items()):
        for label, prefix in ARMS.items():
            xs = np.arange(len(tags))
            mean, spread = [], []
            for tag, _ in tags:
                vals = [json.load(open(Path("runs_v3") / f"{prefix}_s{i}" /
                                        f"metrics_test_{tag}.json"))["psnr"]
                        for i in range(3)]
                mean.append(st.mean(vals))
                spread.append(max(abs(v - st.mean(vals)) for v in vals))
            mean, spread = np.array(mean), np.array(spread)
            ax.plot(xs, mean, "-o" if "U15W" in label else ("-s" if "U15" in label else "--^"),
                    label=label, ms=4)
            ax.fill_between(xs, mean - spread, mean + spread, alpha=0.15)
        ax.set_title(suite, fontsize=11)
        ax.set_xticks(np.arange(len(tags)), [lab for _, lab in tags], fontsize=8)
        ax.set_ylabel("test PSNR (dB)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("EXP-028 regime weighting (XYZ 1:1:2) vs point-uniform vs specialist "
                 "(3-seed mean ± max dev)", fontsize=12)
    fig.tight_layout()
    _save(fig, "exp028_suite_curves")
    print("[ok] exp028_suite_curves")


if __name__ == "__main__":
    main()

"""Fig. 2 — the real-data qualitative comparison, in pseudocolour.

Rows are the three acquisition protocols (Y-50 %, Y-75 %, YZ-75 %); columns are
the methods that actually exist for that protocol:

    Dense reference | Sparse (mask visible) | Interpolation (I0)
    | W0 literature-PSF Wiener | ResidualI (strongest simple learned baseline)
    | Proposed (P3_DualAdaptive)

A column whose checkpoint does not exist for that protocol is drawn as an empty
panel marked ``NOT RUN`` -- Issue #4 forbids filling a cell from another
protocol or another prior. The literature WIener column is kept even though M1
shows it makes things worse: that is the finding, and a figure that omitted it
would be choosing the data.

Everything is read from `cache_rois.py`'s `.npz` cache, so the figure can be
re-rendered without a GPU or a checkpoint.

Usage:
    python -m spur_mpm.scripts.paper_figures.fig2_qualitative \
        --rois reports/paper_structure_v1/rois \
        --out figures/paper_structure_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from spur_mpm.scripts.paper_figures.style import (ERROR_CMAP, MISSING,
                                                  VESSEL_CMAP, add_scalebar,
                                                  apply_style, mm, panel_letter,
                                                  save_figure)

#: Column order. The keys match the `.npz` arrays; `arm` keys come from the
#: manifest, and a missing arm is drawn as NOT RUN rather than skipped.
COLUMNS = [
    ("dense", "Dense reference"),
    ("sparse", "Sparse sampled"),
    ("i0", "Interpolation"),
    ("w0_paper", "W0 literature PSF"),
    ("ResidualI", "ResidualI"),
    ("Proposed", "Proposed"),
]
PROTOCOL_TITLES = {
    "Y50": "Y-50 %  ($k_y$=2, $k_z$=1)",
    "Y75": "Y-75 %  ($k_y$=4, $k_z$=1)",
    "YZ75": "YZ-75 %  ($k_y$=2, $k_z$=2)",
}
PROTOCOL_ORDER = ["Y50", "Y75", "YZ75"]


def _lut(ref: np.ndarray) -> tuple[float, float]:
    """One intensity window per protocol, from the reference only."""
    return float(np.percentile(ref, 0.5)), float(np.percentile(ref, 99.5))


def _show(ax, img, rng, *, cmap=VESSEL_CMAP, mask=None):
    """Draw one panel; `mask is False` voxels are painted cyan, not black."""
    im = ax.imshow(img, cmap=cmap, vmin=rng[0], vmax=rng[1],
                   interpolation="nearest")
    if mask is not None:
        holes = (~np.asarray(mask)).astype(float)
        holes[holes == 0] = np.nan      # NaN is transparent, so only holes paint
        ax.imshow(holes, cmap=_flat_cmap(MISSING), vmin=0, vmax=1,
                  interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    return im


def _flat_cmap(color):
    import matplotlib.colors as mcolors
    return mcolors.ListedColormap([color])


def _crop_xy(vol, half=160, centre=None):
    c = vol.shape[1] // 2 if centre is None else centre
    return vol[:, c - half:c + half, c - half:c + half]


def _xz(vol, y=None):
    y = vol.shape[1] // 2 if y is None else y
    return vol[:, y, :]


def _block(fig, gs, key, data, manifest, cols, *, zoom_um=96.0):
    """One protocol: an XY row and an XZ row."""
    vox_xy = manifest["voxel_um"]["xy"]
    vox_z = manifest["voxel_um"]["z"]
    dense = data["dense"]
    rng = _lut(dense)
    arms = manifest["protocols"][key].get("arms", {})
    n = len(cols)
    for i, (k, title) in enumerate(cols):
        # -- XY -------------------------------------------------------------
        ax = fig.add_subplot(gs[0, i])
        if k in data:
            img = _crop_xy(data[k])
            _show(ax, img, rng, mask=(_crop_xy(data["mask"]) if k == "sparse" else None))
        elif k in arms and arms[k]["state"] == "NOT_RUN":
            _not_run(ax)
        else:
            _not_run(ax)
        if i == 0:
            panel_letter(ax, "a" if key == "Y50" else "b" if key == "Y75" else "c")
            add_scalebar(ax, vox_xy, 50, color="white")
        ax.set_title(title, pad=2)
        if i == 0:
            ax.set_ylabel("XY", fontsize=8)
        # -- XZ -------------------------------------------------------------
        ax = fig.add_subplot(gs[1, i])
        if k in data:
            img = _xz(data[k])
            _show(ax, img, rng, mask=(_xz(data["mask"]) if k == "sparse" else None))
        else:
            _not_run(ax)
        ax.set_aspect(vox_xy / vox_z)
        if i == 0:
            ax.set_ylabel("XZ", fontsize=8)
            add_scalebar(ax, vox_xy, 50, color="white")
    return rng


def _not_run(ax):
    ax.set_facecolor("#f2f2f2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "NOT RUN", ha="center", va="center", fontsize=7,
            color="#666666", transform=ax.transAxes)
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def render(rois: Path, out: Path, *, dpi: int) -> list[Path]:
    manifest = json.loads((rois / "manifest.json").read_text())
    data = {k: dict(np.load(rois / f"roi_{k}.npz")) for k in PROTOCOL_ORDER
            if (rois / f"roi_{k}.npz").exists()}
    written = []
    config = manifest.get("figure", {})

    # --- Fig. 2a: XY + XZ per protocol ------------------------------------
    fig = plt.figure(figsize=(mm(183), mm(150)))
    outer = fig.add_gridspec(len(PROTOCOL_ORDER), 1, hspace=0.32,
                             left=0.045, right=0.99, top=0.97, bottom=0.02)
    for r, key in enumerate(PROTOCOL_ORDER):
        inner = outer[r].subgridspec(2, len(COLUMNS), hspace=0.05, wspace=0.05)
        _block(fig, inner, key, data[key], manifest, COLUMNS)
        fig.text(0.005, 1 - (r + 0.5) / len(PROTOCOL_ORDER),
                 PROTOCOL_TITLES[key], rotation=90, va="center", fontsize=8)
    fig.suptitle("E027 Fig. 2 — sparse 3PM reconstruction of the same vessel ROI",
                 fontsize=9, y=0.995)
    written += save_figure(fig, out / "fig2a_protocols", dpi_print=dpi)
    plt.close(fig)

    # --- Fig. 2b: missing-region error, one shared scale per protocol ------
    keys = ["i0", "w0_paper", "ResidualI", "Proposed"]
    titles = ["Interpolation", "W0 literature PSF", "ResidualI", "Proposed"]
    fig = plt.figure(figsize=(mm(183), mm(78)))
    grid = fig.add_gridspec(len(PROTOCOL_ORDER), len(keys) + 1, width_ratios=[0.03] + [1] * len(keys),
                            hspace=0.22, wspace=0.06, left=0.09, right=0.99,
                            top=0.92, bottom=0.02)
    for r, key in enumerate(PROTOCOL_ORDER):
        d = data[key]
        miss = ~d["mask"]
        errs = {}
        for k in keys:
            if k in d:
                e = np.abs(_crop_xy(d[k]) - _crop_xy(d["dense"]))
                m = _crop_xy(miss.astype(np.float32))
                errs[k] = np.where(m > 0, e, np.nan)
        vmax = np.nanpercentile([np.nanpercentile(v, 99) for v in errs.values()], 90) if errs else 1.0
        for c, (k, title) in enumerate(zip(keys, titles)):
            ax = fig.add_subplot(grid[r, c + 1])
            if k in errs:
                im = ax.imshow(errs[k], cmap=ERROR_CMAP, vmin=0, vmax=vmax,
                               interpolation="nearest")
            else:
                _not_run(ax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8, pad=2)
            if c == 0:
                panel_letter(ax, "def"[r])
        # One colour bar per protocol, on the left.
        cax = fig.add_subplot(grid[r, 0])
        if errs:
            plt.colorbar(plt.cm.ScalarMappable(
                norm=plt.Normalize(0, vmax), cmap=ERROR_CMAP), cax=cax)
            cax.tick_params(labelsize=6)
            cax.set_ylabel(f"|error|, {key}", fontsize=6)
    fig.suptitle("E027 Fig. 2 — |prediction - reference| on voxels never measured "
                 "(shared scale within each protocol)", fontsize=8, y=0.995)
    written += save_figure(fig, out / "fig2b_error_missing", dpi_print=dpi)
    plt.close(fig)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rois", type=Path,
                    default=Path("reports/paper_structure_v1/rois"))
    ap.add_argument("--out", type=Path, default=Path("figures/paper_structure_v1"))
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    apply_style(8)
    written = render(a.rois, a.out, dpi=a.dpi)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()

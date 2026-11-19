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
    """One XY plane (the slab's centre) cropped to a square ROI."""
    z = vol.shape[0] // 2
    c = vol.shape[1] // 2 if centre is None else centre
    plane = vol[z]
    return plane[c - half:c + half, c - half:c + half]


def _xz(vol, y=None, half=160):
    """One XZ section: depth against y, cropped to the same in-plane ROI.

    Displayed with ``aspect = vox_z / vox_xy``: a 5 µm z step must look ~5x
    taller than a 0.98 µm pixel is wide, otherwise 24 planes collapse into a
    line and the figure hides exactly the axial truncation it is meant to show.
    """
    y = vol.shape[1] // 2 if y is None else y
    c = vol.shape[2] // 2
    return vol[:, y, c - half:c + half]


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
            panel_letter(ax, "a" if key == "Y50" else "b" if key == "Y75" else "c",
                         color="white", outline="black")
            add_scalebar(ax, vox_xy, 50, color="white")
        ax.set_title(title, pad=2)
        if i == 0:
            ax.set_ylabel("XY", fontsize=8)
        # -- XZ -------------------------------------------------------------
        ax = fig.add_subplot(gs[1, i])
        if k in data:
            img = _xz(data[k])
            _show(ax, img, rng, mask=(_xz(data["mask"]) if k == "sparse" else None))
            # Only a panel that holds an image gets the physical aspect; giving
            # it to an empty cell shrank the NOT RUN box to a sliver.
            ax.set_aspect(vox_z / vox_xy)
        else:
            _not_run(ax)
        if i == 0:
            ax.set_ylabel("XZ", fontsize=8)
            add_scalebar(ax, vox_xy, 50, color="white")
    return rng


def _not_run(ax):
    """An empty cell that says so. Never fills a gap with another protocol."""
    ax.set_facecolor("#f2f2f2")
    ax.set_xticks([])
    ax.set_yticks([])
    # An axes that is about to receive no image must not inherit a data aspect
    # from the panels around it: doing so shrank the XZ NOT RUN cells into a
    # sliver with a vertical bar through the text.
    ax.set_aspect("auto")
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
    fig = plt.figure(figsize=(mm(183), mm(160)))
    outer = fig.add_gridspec(len(PROTOCOL_ORDER), 1, hspace=0.30,
                             left=0.06, right=0.995, top=0.90, bottom=0.015)
    for r, key in enumerate(PROTOCOL_ORDER):
        # XY row is square; the XZ row is a short, physically scaled strip.
        inner = outer[r].subgridspec(2, len(COLUMNS), height_ratios=[1, 0.42],
                                     hspace=0.08, wspace=0.05)
        _block(fig, inner, key, data[key], manifest, COLUMNS)
        fig.text(0.008, 0.90 - (r + 0.42) * (0.885 / len(PROTOCOL_ORDER)),
                 PROTOCOL_TITLES[key], rotation=90, va="center", fontsize=7.5)
    fig.suptitle("E027 Fig. 2 — the same vessel ROI under three sampling protocols "
                 "(magenta = intensity, cyan = never measured)", fontsize=8, y=0.985)
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
        errs, maes = {}, {}
        for k in keys:
            if k in d:
                e = np.abs(_crop_xy(d[k]) - _crop_xy(d["dense"]))
                m = _crop_xy(miss.astype(np.float32))
                errs[k] = np.where(m > 0, e, np.nan)
                # The number a reader would otherwise have to infer from colour:
                # MAE on the missing voxels of exactly this crop.
                maes[k] = float(np.nanmean(errs[k]))
        vmax = np.nanpercentile([np.nanpercentile(v, 99) for v in errs.values()], 90) if errs else 1.0
        for c, (k, title) in enumerate(zip(keys, titles)):
            ax = fig.add_subplot(grid[r, c + 1])
            if k in errs:
                ax.imshow(errs[k], cmap=ERROR_CMAP, vmin=0, vmax=vmax,
                          interpolation="nearest")
                txt = ax.text(0.97, 0.04, f"MAE {maes[k]:.4f}",
                              transform=ax.transAxes, ha="right", va="bottom",
                              fontsize=6, color="white")
                import matplotlib.patheffects as pe
                txt.set_path_effects([pe.withStroke(linewidth=1.6,
                                                    foreground="black")])
            else:
                _not_run(ax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8, pad=2)
            if c == 0:
                panel_letter(ax, "def"[r])
        # The protocol name belongs to the row, not to a colour bar's axis.
        fig.text(0.005, 1 - (r + 0.55) * (0.90 / len(PROTOCOL_ORDER)),
                 PROTOCOL_TITLES[key], rotation=90, va="center", fontsize=7)
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

    # --- Fig. 2c: the fine-vessel zoom the Issue asks for -------------------
    # 96 µm across, same columns and the same frozen LUT as the block it comes
    # from. One protocol (YZ-75 %) because that is where the z-skipping and the
    # fine branches interact.
    key = "YZ75"
    d = data[key]
    vox = manifest["voxel_um"]["xy"]
    half = max(8, int(round(48.0 / vox)))            # 96 µm full width
    rng = _lut(d["dense"])
    fig = plt.figure(figsize=(mm(183), mm(42)))
    grid = fig.add_gridspec(1, len(COLUMNS), wspace=0.06,
                            left=0.01, right=0.99, top=0.88, bottom=0.02)
    z = d["dense"].shape[0] // 2
    cy = cx = d["dense"].shape[1] // 2
    for i, (k, title) in enumerate(COLUMNS):
        ax = fig.add_subplot(grid[0, i])
        if k in d:
            img = d[k][z][cy - half:cy + half, cx - half:cx + half]
            m = (d["mask"][z][cy - half:cy + half, cx - half:cx + half]
                 if k == "sparse" else None)
            _show(ax, img, rng, mask=m)
            panel_letter(ax, "abcdef"[i], color="white", outline="black")
        else:
            _not_run(ax)
        ax.set_title(title, fontsize=7.5, pad=2)
    fig.text(0.5, 0.945, PROTOCOL_TITLES[key] +
             f" — 96 µm zoom on fine vessels, same frozen LUT as panel c",
             ha="center", fontsize=8)
    written += save_figure(fig, out / "fig2c_zoom", dpi_print=dpi)
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

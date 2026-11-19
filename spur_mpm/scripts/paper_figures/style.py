"""Publication styling for the E027 structure figures.

One place for the rules Issue #4 fixes, so no two figures can disagree:

* **Vessels are magenta.** A sequential colormap from black through deep
  magenta to hot pink and warm white. Not grey (a single-channel micrograph in
  grey reads as raw data, and the task is to show structure), and not rainbow
  (it invents an ordering the intensity does not have).
* **Missing voxels are cyan** (`#22d3ee`). No point of the magenta ramp reaches
  cyan, so a hole can never be mistaken for a dark vessel; the mask is drawn as
  an overlay, never implied by a dark background.
* **Errors are magma**, continuous, with a scale shared by every row of a panel.
* **One LUT window per panel**, taken from the reference. No per-method
  contrast stretching, ever: that is how a figure wins an argument the numbers
  do not.
* Physical scale bars in micrometres, panel letters a-f, and vector text.

The figure sizes are in millimetres and the fonts in points because the target
is a two-column journal page (~183 mm), not a screen.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["VESSEL_CMAP", "ERROR_CMAP", "MISSING", "apply_style",
           "add_scalebar", "panel_letter", "save_figure", "mm"]

#: Black -> deep magenta -> hot pink -> warm white. Monotonic in luminance so it
#: survives greyscale printing and every common colour-vision deficiency.
VESSEL_CMAP = LinearSegmentedColormap.from_list(
    "vessel_magenta",
    [(0.00, "#050005"), (0.18, "#3d0033"), (0.38, "#8a0a63"),
     (0.60, "#d81e8a"), (0.80, "#ff6ab5"), (1.00, "#fff2f8")],
)
ERROR_CMAP = matplotlib.colormaps["magma"].copy()
# Measured voxels are neutral grey, not white: white washes the map out
# and reads as "no error" when the truth is "not scored here".
ERROR_CMAP.set_bad("#d9d9d9")

#: Never measured. Far from every colour the vessel ramp produces.
MISSING = "#22d3ee"


def mm(x: float) -> float:
    """Millimetres to inches (matplotlib's unit)."""
    return x / 25.4


def apply_style(font_pt: float = 8.0) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font_pt,
        "axes.titlesize": font_pt,
        "axes.labelsize": font_pt,
        "xtick.labelsize": font_pt - 1,
        "ytick.labelsize": font_pt - 1,
        "legend.fontsize": font_pt - 1,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 1.5,
        "ytick.major.size": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        # Vector text in the PDF/SVG; the raster panels stay raster.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_letter(ax, letter: str, *, x: float = 0.012, y: float = 0.985,
                 size: float = 9.0, color: str = "black",
                 outline: str | None = None) -> None:
    """Lowercase panel letter, top-left, drawn in axes coordinates.

    Over a micrograph the letter needs an outline, not a second colour choice:
    white-on-magenta disappears wherever the vessel is bright, black-on-cyan
    disappears where the mask is dense. A stroked glyph survives both.
    """
    import matplotlib.patheffects as pe
    text = ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
                   fontsize=size, fontweight="bold", color=color)
    if outline:
        text.set_path_effects([pe.withStroke(linewidth=2.0, foreground=outline)])


def add_scalebar(ax, um_per_px: float, bar_um: float, *, label: str | None = None,
                 color: str = "white", lw: float = 2.0, pad_frac: float = 0.04) -> None:
    """Physical scale bar in the lower right of an image axis."""
    h, w = ax.images[0].get_array().shape[-2:]
    bar_px = bar_um / um_per_px
    x0 = w * (1.0 - pad_frac) - bar_px
    y0 = h * (1.0 - pad_frac)
    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=lw,
            solid_capstyle="butt")
    ax.text(x0 + bar_px / 2, y0 - h * 0.02,
            label or (f"{bar_um:g} µm" if bar_um >= 1 else f"{bar_um*1000:g} nm"),
            color=color, ha="center", va="bottom", fontsize=7)


def save_figure(fig, out_stem: Path, *, dpi_print: int = 600) -> list[Path]:
    """Write the same figure as PNG (preview), PDF and SVG (vector text).

    The three files must show the same image; a paper figure that exists only as
    a re-stretched PNG cannot be re-laid-out by the journal.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".png", {"dpi": dpi_print}),
                           (".pdf", {}),
                           (".svg", {})):
        path = out_stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        written.append(path)
    return written

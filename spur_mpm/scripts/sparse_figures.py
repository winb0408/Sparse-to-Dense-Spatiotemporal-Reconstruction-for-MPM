"""Figures for the sparse-acquisition ladder: what was measured, and what was not.

    python -m spur_mpm.scripts.sparse_figures \
        --index data_local/dryad_cache/index_cohort.json \
        --manifest3d data_local/sparse/cohort3d/manifest.json \
        --manifest2d data_local/sparse/cohort2d/manifest.json \
        --sample M2_FOVI_S1_vessels --out spur_mpm/figures/sparse

Three figures, each answering one question a reader should not have to take on
trust:

``sparse_1_construction.png``
    One volume, every level, in two orthogonal views. Unmeasured voxels are drawn
    in cyan rather than black, so the support is visible instead of being
    indistinguishable from dark tissue. The XY row shows that a kept row is kept
    across the whole fast axis; the YZ row shows that Z-skipping removes entire
    planes. The XY row is drawn on a plane every one of those levels measured --
    on a skipped plane a Z-sparse panel would be blank for display reasons alone.
``sparse_2_ladder_2d.png``
    The 2-D (Y-direction only) ladder, with the row pattern drawn underneath as a
    binary raster -- the "dense in, sparse out" statement made literal.
``sparse_3_coverage.png``
    Which (Z, Y) scan lines were acquired, for every 3-D level. This is the honest
    cost picture: X is the fast axis, so a line that is scanned at all costs a
    full line, and the fraction of white here is the fraction of scan time.

Every panel re-derives the mask from the spec and compares it with the mask
stored in the manifest; a mismatch is raised, not drawn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spur_mpm.data.datasets import VolumeCache
from spur_mpm.data.sparse import measured_mask, read_manifest
from spur_mpm.physics.sampling import SamplingSpec

__all__ = ["render"]

CMAP = "magma"
MISSING = "#22d3ee"          # bright cyan: no point of magma reaches it
_NOTE = dict(fontsize=5.6, color="white", ha="right", va="bottom",
             bbox=dict(facecolor="black", alpha=0.55, lw=0.0, pad=1.3))
_HEAD = dict(fontsize=7.0, color="black", ha="center", va="bottom")


def _show(ax, img, rng, cmap: str = CMAP, bad: str | None = MISSING):
    im = ax.imshow(img, cmap=cmap, vmin=rng[0], vmax=rng[1],
                   interpolation="nearest", aspect="auto", resample=False)
    if bad is not None:
        im.cmap.set_bad(bad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def _note(ax, text: str) -> None:
    ax.text(0.972, 0.028, text, transform=ax.transAxes, zorder=6, **_NOTE)


def _axes_hint(ax, text: str) -> None:
    """Which way each displayed axis runs, in the panel's own corner."""
    ax.text(0.028, 0.972, text, transform=ax.transAxes, zorder=6,
            fontsize=5.6, color="white", ha="left", va="top",
            bbox=dict(facecolor="black", alpha=0.55, lw=0.0, pad=1.3))


def _entries(manifest: dict, sample_id: str) -> dict[str, dict]:
    return {e["level"]: e for e in manifest["entries"] if e["sample_id"] == sample_id}


def _cross_check(sample_id: str, entry: dict, shape: tuple[int, ...], plane: int | None) -> np.ndarray:
    """Rebuild the mask from its spec and require it to equal the stored one."""
    spec = SamplingSpec.from_dict(entry["spec"])
    m = measured_mask(shape, spec, phase=tuple(entry["phase"]))
    if plane is not None:
        m = m[plane]
    stored = _load_mask(entry, plane=plane)
    if not np.array_equal(m, stored):
        raise RuntimeError(
            f"{sample_id}/{entry['level']}: the mask in the manifest is not the "
            "mask its spec produces; the figure would show the wrong protocol")
    return m


def _load_mask(entry: dict, plane: int | None) -> np.ndarray:
    with np.load(entry["mask_path"]) as z:
        packed = z["packed"]
        shape = tuple(int(v) for v in z["shape"])
    flat = np.unpackbits(packed)[:int(np.prod(shape))]
    m = flat.reshape(shape).astype(bool)
    return m[plane] if plane is not None else m


def _common_plane(entries: dict[str, dict], shape: tuple[int, ...]) -> int:
    """A plane that *every* one of these levels measured, nearest the centre.

    If the panel row is drawn on a plane that some level skipped, that panel is
    entirely cyan and reads as "this protocol measures nothing here" -- a
    display artefact, not a property of the protocol.
    """
    shared = None
    for entry in entries.values():
        planes = _load_mask(entry, plane=None).any(axis=(1, 2))
        shared = planes if shared is None else (shared & planes)
    if shared is None or not shared.any():
        raise RuntimeError("no single plane is measured by every level; "
                           "draw the panels on separate planes instead")
    idx = np.flatnonzero(shared)
    return int(idx[np.argmin(np.abs(idx - shape[0] // 2))])


def _shape_of(entry: dict) -> tuple[int, ...]:
    with np.load(entry["mask_path"]) as z:
        return tuple(int(v) for v in z["shape"])


def _crop(img: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[-2:]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return img[..., y0:y0 + size, x0:x0 + size], (y0, x0)


def render(index: Path, manifest3d: Path | None, manifest2d: Path | None,
           out_dir: Path, sample_id: str, dpi: int = 220,
           crop: int = 300) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = VolumeCache.load(index, fold=0)
    dense = cache.calibration.apply(np.asarray(cache.array(sample_id))).astype(np.float32)
    out: list[Path] = []

    # one intensity window for the whole figure set: a dim measurement must read
    # as a dim measurement, not be re-normalised into looking dense
    rng = (float(np.percentile(dense, 1.0)), float(np.percentile(dense, 99.5)))
    zmid = dense.shape[0] // 2

    # ---- figure 1: one volume, every 3-D level, two orthogonal views --------
    if manifest3d is not None:
        man = read_manifest(manifest3d)
        ent = _entries(man, sample_id)
        levels = [name for name in ("y_k2", "y_k4", "z_k2", "z_k4", "yz_k2x2",
                                   "yz_k2x4") if name in ent]
        # draw the XY row on a plane that every rendered level measured, else the
        # Z-sparse panels would be entirely cyan for display reasons alone
        z1 = _common_plane({n: ent[n] for n in levels}, _shape_of(ent[levels[0]]))
        xy0, (y0, x0) = _crop(dense[z1], crop)
        xc = xy0.shape[-1] // 2
        yz0 = dense[:, y0:y0 + crop, x0 + xc]         # [Z, y]: z down, y right

        fig, axes = plt.subplots(2, len(levels) + 1, figsize=(1.62 * (len(levels) + 1), 3.9),
                                 gridspec_kw=dict(wspace=0.03, hspace=0.06,
                                                  left=0.035, right=0.995,
                                                  top=0.86, bottom=0.02))
        _show(axes[0, 0], xy0, rng)
        axes[0, 0].set_title("dense reference", **_HEAD)
        _note(axes[0, 0], f"z = {z1}")
        _show(axes[1, 0], yz0, rng)
        axes[1, 0].set_title("dense reference", **_HEAD)
        _note(axes[1, 0], f"x = {x0 + xc}")

        for j, name in enumerate(levels, start=1):
            m = _cross_check(sample_id, ent[name], _shape_of(ent[name]), plane=None)
            y = np.where(m, dense, np.nan).astype(np.float32)
            _show(axes[0, j], y[z1, y0:y0 + crop, x0:x0 + crop], rng)
            axes[0, j].set_title(name.replace("_", " "), **_HEAD)
            _note(axes[0, j], f"{ent[name]['measured_fraction'] * 100:.1f} % measured")
            _show(axes[1, j], y[:, y0:y0 + crop, x0 + xc], rng)
            _note(axes[1, j], f"{ent[name]['scan_line_fraction'] * 100:.0f} % scan time")

        _axes_hint(axes[0, 0], "y↓  x→")
        _axes_hint(axes[1, 0], "z↓  y→")

        fig.text(0.005, 0.955, "sparse acquisition, one volume",
                 fontsize=9, ha="left", va="top")
        fig.text(0.005, 0.925,
                 f"{sample_id}  |  top: XY plane at z = {z1} (measured by every level "
                 f"shown)  |  bottom: YZ slice at x = {x0 + xc}  |  y increases "
                 "downwards, x and y increase rightwards, z increases downwards  |  "
                 "cyan = not measured, drawn as no-data rather than black",
                 fontsize=5.8, ha="left", va="top", color="#333333")
        p = out_dir / "sparse_1_construction.png"
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        out.append(p)

        # ---- figure 3: the (Z, Y) coverage picture -------------------------
        names = [n for n in ("y_k2", "y_k4", "z_k2", "z_k4", "yz_k2x2", "yz_k2x4")
                 if n in ent]
        fig, axes = plt.subplots(1, len(names), figsize=(1.55 * len(names), 1.95),
                                 gridspec_kw=dict(wspace=0.05, left=0.02, right=0.995,
                                                  top=0.80, bottom=0.05))
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            m = _cross_check(sample_id, ent[name], _shape_of(ent[name]), plane=None)
            lines = m.any(axis=-1)                    # [Z, Y]: z down, y right
            ax.imshow(lines, cmap="binary", vmin=0, vmax=1, aspect="auto",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(name.replace("_", " "), **_HEAD)
            ax.text(0.97, 0.04, f"{ent[name]['scan_line_fraction'] * 100:.0f} % of scan time",
                    transform=ax.transAxes, fontsize=5.4, ha="right", va="bottom",
                    color="white" if lines.mean() > 0.5 else "black")
        _axes_hint(axes[0], "z↓  y→")
        fig.text(0.004, 0.955, "which (Z, Y) scan lines were acquired",
                 fontsize=9, ha="left", va="top")
        fig.text(0.004, 0.925,
                 f"{sample_id}  |  rows = z (z = 0 at the top, deepest plane at the "
                 "bottom), columns = y  |  white = line scanned (all of X), black = line "
                 "skipped  |  a full line costs a full line of dwell time, so white "
                 "area = scan time",
                 fontsize=5.8, ha="left", va="top", color="#333333")
        p = out_dir / "sparse_3_coverage.png"
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        out.append(p)

    # ---- figure 2: the 2-D ladder ------------------------------------------
    if manifest2d is not None:
        man2 = read_manifest(manifest2d)
        key = f"{sample_id}#z{zmid}"
        if not any(e["sample_id"] == key for e in man2["entries"]):
            key = next((e["sample_id"] for e in man2["entries"]           # another plane
                        if e["sample_id"].startswith(sample_id)), None)
            if key is None:
                raise SystemExit(f"{sample_id} not in {manifest2d}")
        ent2 = _entries(man2, key)
        # the masks were cut from the plane named in their key; show that plane,
        # not "the middle of the volume", so the picture and the mask agree
        z2 = int(key.rsplit("#z", 1)[1]) if "#z" in key else zmid
        names = [n for n in ("y_k2", "y_k3", "y_k4", "y_k8") if n in ent2]
        ref, off = _crop(dense[z2], crop)
        ncol = len(names) + 1
        fig, axes = plt.subplots(2, ncol, figsize=(1.62 * ncol, 3.5),
                                 gridspec_kw=dict(wspace=0.03, hspace=0.06,
                                                  left=0.035, right=0.995,
                                                  top=0.84, bottom=0.02,
                                                  height_ratios=[6, 1]))
        _show(axes[0, 0], ref, rng)
        axes[0, 0].set_title("dense reference", **_HEAD)
        _note(axes[0, 0], "100 % measured")
        _axes_hint(axes[0, 0], "y↓  x→")
        _axes_hint(axes[1, 0], "y↓")
        mask_ref = np.ones_like(ref, dtype=bool)
        _show(axes[1, 0], mask_ref.astype(np.float32), (0, 1), cmap="gray", bad=None)

        for j, name in enumerate(names, start=1):
            m2 = _cross_check(ent2[name]["sample_id"], ent2[name],
                              _shape_of(ent2[name]), plane=None)
            y = np.where(m2, dense[z2], np.nan).astype(np.float32)
            m2c = m2[off[0]:off[0] + crop, off[1]:off[1] + crop]
            yc = y[off[0]:off[0] + crop, off[1]:off[1] + crop]
            _show(axes[0, j], yc, rng)
            axes[0, j].set_title(name.replace("_", " "), **_HEAD)
            rows = int(m2.any(axis=-1).sum())
            _note(axes[0, j], f"{ent2[name]['measured_fraction'] * 100:.1f} %  "
                              f"({rows}/{m2.shape[0]} rows)")
            # the row pattern, drawn at the crop's vertical scale
            prof = m2.any(axis=-1)[off[0]:off[0] + crop][:, None]
            axes[1, j].imshow(prof, cmap="binary", vmin=0, vmax=1, aspect="auto",
                              interpolation="nearest")
            axes[1, j].set_xticks([])
            axes[1, j].set_yticks([])
            for sp in axes[1, j].spines.values():
                sp.set_visible(False)

        fig.text(0.005, 0.955, "2-D ladder: Y-direction sparsity only",
                 fontsize=9, ha="left", va="top")
        fig.text(0.005, 0.925,
                 f"{sample_id}, plane z = {z2}  |  top: measurement (cyan = row not "
                 "scanned)  |  bottom: scanned / skipped row pattern at true scale  |  "
                 "y increases downwards, x increases rightwards",
                 fontsize=5.8, ha="left", va="top", color="#333333")
        p = out_dir / "sparse_2_ladder_2d.png"
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        out.append(p)

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--manifest3d", type=Path, default=None)
    p.add_argument("--manifest2d", type=Path, default=None)
    p.add_argument("--sample", default="M2_FOVI_S1_vessels")
    p.add_argument("--out", type=Path, default=Path("spur_mpm/figures/sparse"))
    p.add_argument("--crop", type=int, default=300)
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args(argv)
    for pth in render(args.index, args.manifest3d, args.manifest2d, args.out,
                      args.sample, dpi=args.dpi, crop=args.crop):
        print(pth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

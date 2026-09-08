#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import colormaps


def to_rgb_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        if np.nanmax(img) <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgbf = rgb.astype(np.float32)
    return 0.299 * rgbf[..., 0] + 0.587 * rgbf[..., 1] + 0.114 * rgbf[..., 2]


def detect_content_rows(gray01: np.ndarray) -> tuple[int, int]:
    row_occ = (gray01 < 0.985).mean(axis=1)
    active = row_occ > 0.02
    idx = np.where(active)[0]
    if idx.size == 0:
        return 0, gray01.shape[0]
    return int(idx[0]), int(idx[-1] + 1)


def detect_panel_columns(gray01_crop: np.ndarray) -> list[tuple[int, int]]:
    col_occ = (gray01_crop < 0.985).mean(axis=0)
    active = col_occ > 0.02

    runs: list[tuple[int, int]] = []
    i, n = 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            if (j - i) >= 40:
                runs.append((i, j))
            i = j
        else:
            i += 1

    # Fallback: default 5 columns for these qualitative panels.
    if len(runs) < 3:
        w = gray01_crop.shape[1]
        k = 5
        step = w // k
        runs = [(t * step, (t + 1) * step if t < k - 1 else w) for t in range(k)]
    return runs


def stretch_and_colorize(
    region_gray: np.ndarray, cmap_name: str = "magma", gamma: float = 0.85
) -> tuple[np.ndarray, float, float]:
    vals = region_gray[np.isfinite(region_gray)]
    if vals.size == 0:
        out = np.zeros((*region_gray.shape, 3), dtype=np.uint8)
        return out, 0.0, 1.0

    vmin, vmax = np.percentile(vals, [0.5, 99.5])
    if vmax <= vmin:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = np.clip((region_gray - vmin) / (vmax - vmin), 0.0, 1.0)
    # gamma < 1 highlights dim cellular details.
    norm = np.power(norm, gamma, dtype=np.float32)

    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    return rgb, float(vmin), float(vmax)


def process_one_image(src_path: Path, dst_path: Path, cmap_name: str, gamma: float) -> None:
    arr = mpimg.imread(str(src_path))
    rgb = to_rgb_u8(arr)
    gray = rgb_to_gray(rgb)
    gray01 = gray / 255.0

    y0, y1 = detect_content_rows(gray01)
    panel_runs = detect_panel_columns(gray01[y0:y1, :])

    out = rgb.copy()
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)

    for x0_rel, x1_rel in panel_runs:
        x0, x1 = x0_rel, x1_rel
        panel_w = x1 - x0
        if panel_w < 20:
            continue

        cbw = max(12, min(22, panel_w // 11))
        gap = 3
        data_w = max(1, panel_w - cbw - gap)

        region = gray[y0:y1, x0 : x0 + data_w]
        colorized, vmin, vmax = stretch_and_colorize(region, cmap_name=cmap_name, gamma=gamma)
        out[y0:y1, x0 : x0 + data_w] = colorized

        cb_x0 = x0 + data_w + gap
        cb_x1 = min(x1, cb_x0 + cbw)
        if cb_x1 <= cb_x0:
            continue

        h = y1 - y0
        grad = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
        cmap = colormaps.get_cmap(cmap_name)
        cb_rgb = (cmap(grad)[..., :3] * 255.0).astype(np.uint8)
        cb_rgb = np.repeat(cb_rgb, cb_x1 - cb_x0, axis=1)
        out[y0:y1, cb_x0:cb_x1] = cb_rgb

        out[y0:y1, cb_x0 : cb_x0 + 1] = 255
        out[y0:y1, cb_x1 - 1 : cb_x1] = 255

        # redraw each iteration to keep text on latest array
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        draw.text((cb_x0, max(0, y0 - 12)), f"{vmax:.0f}", fill=(255, 255, 255))
        draw.text((cb_x0, min(pil.height - 12, y1)), f"{vmin:.0f}", fill=(255, 255, 255))

    pil.save(dst_path)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Regenerate qualitative panel PNGs with per-panel adaptive contrast, "
            "pseudocolor, and per-panel colorbars."
        )
    )
    p.add_argument(
        "--src_dir",
        type=Path,
        default=Path("/data2/wjb/threePM/eval_pollen_sparse/qualitative"),
        help="Source qualitative PNG directory.",
    )
    p.add_argument(
        "--dst_dir",
        type=Path,
        default=Path("/data2/wjb/threePM/eval_pollen_sparse/qualitative_pseudocolor_v3"),
        help="Output directory (non-overwrite of originals).",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap name, e.g. magma/inferno/viridis.",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.85,
        help="Gamma for normalized intensities; <1 highlights dim details.",
    )
    return p.parse_args()


def main() -> None:
    args = build_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {args.src_dir}")

    for f in files:
        out = args.dst_dir / f.name
        process_one_image(f, out, cmap_name=args.cmap, gamma=args.gamma)
        print(f"[OK] {out}")
    print(f"\n[Done] Generated {len(files)} files -> {args.dst_dir}")


if __name__ == "__main__":
    main()

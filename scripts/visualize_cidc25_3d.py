#!/usr/bin/env python3
"""
3D visualization for AI4Life-CIDC25 stack TIFF data.

Features:
1) Orthogonal slices (XY/XZ/YZ)
2) 3D point-cloud rendering from high-intensity voxels

Example:
  python scripts/visualize_cidc25_3d.py \
    --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
    --split Training \
    --stack A1 \
    --output_dir /data2/wjb/threePM/figures_cidc25_3d
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import tifffile


def read_stack(stack_path: Path) -> np.ndarray:
    arr = tifffile.imread(str(stack_path))
    if arr.ndim < 3:
        raise ValueError(f"Expected stack [F,H,W], but got shape={arr.shape} from {stack_path}")
    while arr.ndim > 3:
        arr = arr[:, 0]
    return arr.astype(np.float32, copy=False)


def robust_normalize(x: np.ndarray, low_q: float = 0.5, high_q: float = 99.5) -> np.ndarray:
    lo = np.percentile(x, low_q)
    hi = np.percentile(x, high_q)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32, copy=False)


def make_orth_views(
    vol: np.ndarray,
    title: str,
    save_path: Path,
    z_idx: int | None = None,
    y_idx: int | None = None,
    x_idx: int | None = None,
) -> Tuple[int, int, int]:
    z = vol.shape[0] // 2 if z_idx is None else int(np.clip(z_idx, 0, vol.shape[0] - 1))
    y = vol.shape[1] // 2 if y_idx is None else int(np.clip(y_idx, 0, vol.shape[1] - 1))
    x = vol.shape[2] // 2 if x_idx is None else int(np.clip(x_idx, 0, vol.shape[2] - 1))

    xy = vol[z, :, :]
    xz = vol[:, y, :]
    yz = vol[:, :, x]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axs[0].imshow(xy, cmap="gray")
    axs[0].set_title(f"XY (z={z})")
    axs[0].axis("off")

    axs[1].imshow(xz, cmap="gray", aspect="auto")
    axs[1].set_title(f"XZ (y={y})")
    axs[1].axis("off")

    axs[2].imshow(yz, cmap="gray", aspect="auto")
    axs[2].set_title(f"YZ (x={x})")
    axs[2].axis("off")

    fig.suptitle(title)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return z, y, x


def make_point_cloud(
    vol: np.ndarray,
    title: str,
    save_path: Path,
    threshold: float = 0.85,
    max_points: int = 120000,
) -> int:
    mask = vol >= threshold
    coords = np.argwhere(mask)  # (N, 3) -> z,y,x
    if coords.size == 0:
        raise RuntimeError(
            f"No voxels above threshold={threshold}. "
            "Try a smaller threshold, e.g. 0.7 or 0.6."
        )

    if coords.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[idx]

    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    c = vol[z, y, x]

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=c, cmap="inferno", s=1.0, alpha=0.35, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=28, azim=40)

    ax.set_box_aspect((vol.shape[2], vol.shape[1], vol.shape[0]))
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    return int(coords.shape[0])


def _sample_points(vol: np.ndarray, threshold: float, max_points: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    mask = vol >= threshold
    coords = np.argwhere(mask)  # (N, 3) -> z,y,x
    if coords.size == 0:
        return coords, np.array([], dtype=np.float32)
    if coords.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[idx]
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    c = vol[z, y, x]
    return coords, c


def interactive_viewer(
    vol: np.ndarray,
    base_name: str,
    threshold: float = 0.85,
    max_points: int = 60000,
) -> None:
    # Keep interactive refresh smooth by limiting sampled points.
    max_points = int(max(1000, max_points))
    z_init = vol.shape[0] // 2
    y_init = vol.shape[1] // 2
    x_init = vol.shape[2] // 2

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[20, 1.8])

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_3d = fig.add_subplot(gs[0, 3], projection="3d")

    img_xy = ax_xy.imshow(vol[z_init], cmap="gray", vmin=0, vmax=1)
    ax_xy.set_title(f"XY (z={z_init})")
    ax_xy.axis("off")

    img_xz = ax_xz.imshow(vol[:, y_init, :], cmap="gray", aspect="auto", vmin=0, vmax=1)
    ax_xz.set_title(f"XZ (y={y_init})")
    ax_xz.axis("off")

    img_yz = ax_yz.imshow(vol[:, :, x_init], cmap="gray", aspect="auto", vmin=0, vmax=1)
    ax_yz.set_title(f"YZ (x={x_init})")
    ax_yz.axis("off")

    coords, colors = _sample_points(vol, threshold=threshold, max_points=max_points)
    if coords.size > 0:
        sc = ax_3d.scatter(
            coords[:, 2], coords[:, 1], coords[:, 0],
            c=colors, cmap="inferno", s=1.0, alpha=0.35, linewidths=0
        )
    else:
        sc = ax_3d.scatter([], [], [], s=1.0)
    ax_3d.set_title(f"3D points (thr={threshold:.2f}, n={coords.shape[0]})")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.view_init(elev=28, azim=40)
    ax_3d.set_box_aspect((vol.shape[2], vol.shape[1], vol.shape[0]))

    ax_slider_z = fig.add_subplot(gs[1, 0])
    ax_slider_y = fig.add_subplot(gs[1, 1])
    ax_slider_x = fig.add_subplot(gs[1, 2])
    ax_slider_t = fig.add_subplot(gs[1, 3])

    s_z = Slider(ax_slider_z, "z", 0, vol.shape[0] - 1, valinit=z_init, valfmt="%0.0f")
    s_y = Slider(ax_slider_y, "y", 0, vol.shape[1] - 1, valinit=y_init, valfmt="%0.0f")
    s_x = Slider(ax_slider_x, "x", 0, vol.shape[2] - 1, valinit=x_init, valfmt="%0.0f")
    s_t = Slider(ax_slider_t, "thr", 0.10, 0.99, valinit=threshold, valfmt="%0.2f")

    def redraw_slices() -> None:
        z = int(round(s_z.val))
        y = int(round(s_y.val))
        x = int(round(s_x.val))
        img_xy.set_data(vol[z])
        img_xz.set_data(vol[:, y, :])
        img_yz.set_data(vol[:, :, x])
        ax_xy.set_title(f"XY (z={z})")
        ax_xz.set_title(f"XZ (y={y})")
        ax_yz.set_title(f"YZ (x={x})")

    def redraw_points() -> None:
        nonlocal sc
        thr = float(s_t.val)
        coords_new, colors_new = _sample_points(vol, threshold=thr, max_points=max_points)
        sc.remove()
        if coords_new.size > 0:
            sc = ax_3d.scatter(
                coords_new[:, 2], coords_new[:, 1], coords_new[:, 0],
                c=colors_new, cmap="inferno", s=1.0, alpha=0.35, linewidths=0
            )
        else:
            sc = ax_3d.scatter([], [], [], s=1.0)
        ax_3d.set_title(f"3D points (thr={thr:.2f}, n={coords_new.shape[0]})")

    def on_slider_change(_val: float) -> None:
        redraw_slices()
        redraw_points()
        fig.canvas.draw_idle()

    def on_scroll(event) -> None:
        if event.inaxes is ax_xy:
            step = 1 if event.button == "up" else -1
            cur = int(round(s_z.val))
            s_z.set_val(np.clip(cur + step, 0, vol.shape[0] - 1))

    s_z.on_changed(on_slider_change)
    s_y.on_changed(on_slider_change)
    s_x.on_changed(on_slider_change)
    s_t.on_changed(on_slider_change)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    fig.suptitle(
        f"{base_name} | Interactive 3D Viewer\n"
        "Mouse wheel on XY: switch z-slice; drag sliders: adjust x/y/z and threshold"
    )
    plt.tight_layout()
    plt.show()


def resolve_stack_path(cidc_root: Path, split: str, stack: str) -> Path:
    split_dir = cidc_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    stack_path = split_dir / f"{stack}.tif"
    if not stack_path.exists():
        tif_list = sorted(p.name for p in split_dir.glob("*.tif"))
        hint = ", ".join(tif_list[:20]) if tif_list else "No .tif files found"
        raise FileNotFoundError(
            f"Stack not found: {stack_path}\nAvailable examples: {hint}"
        )
    return stack_path


def main() -> None:
    parser = argparse.ArgumentParser(description="3D visualization for CIDC25 stacks.")
    parser.add_argument("--cidc_root", type=str, required=True, help="Path to AI4Life-CIDC25 root.")
    parser.add_argument("--split", type=str, default="Training", choices=["Training", "Validation"])
    parser.add_argument("--stack", type=str, default="A1", help="Stack stem, e.g. A1/B1/C2/D2/F0/F1.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save figures.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Normalized intensity threshold for 3D points.")
    parser.add_argument("--max_points", type=int, default=120000, help="Maximum number of points in 3D scatter.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive viewer (sliders + wheel to browse volume).",
    )
    parser.add_argument(
        "--web_interactive",
        action="store_true",
        help="Save browser-based interactive 3D HTML (works in headless environments).",
    )
    args = parser.parse_args()

    cidc_root = Path(args.cidc_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack_path = resolve_stack_path(cidc_root, args.split, args.stack)
    vol_raw = read_stack(stack_path)
    vol = robust_normalize(vol_raw)

    base_name = f"{args.split}_{args.stack}"
    orth_path = output_dir / f"{base_name}_orth_views.png"
    cloud_path = output_dir / f"{base_name}_pointcloud3d.png"

    z, y, x = make_orth_views(
        vol=vol,
        title=f"{base_name} | Orthogonal Views",
        save_path=orth_path,
    )
    n_pts = make_point_cloud(
        vol=vol,
        title=f"{base_name} | 3D Point Cloud",
        save_path=cloud_path,
        threshold=args.threshold,
        max_points=args.max_points,
    )

    if args.interactive:
        has_display = bool(os.environ.get("DISPLAY"))
        if not has_display:
            print("Interactive window skipped: DISPLAY is empty (headless environment).")
            print("Use --web_interactive to generate an HTML interactive view.")
        else:
            interactive_viewer(
                vol=vol,
                base_name=base_name,
                threshold=args.threshold,
                max_points=min(args.max_points, 60000),
            )

    if args.web_interactive:
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError(
                "plotly is required for --web_interactive. Install with: pip install plotly"
            ) from exc

        coords_html, colors_html = _sample_points(
            vol, threshold=args.threshold, max_points=min(args.max_points, 150000)
        )
        html_path = output_dir / f"{base_name}_interactive3d.html"
        if coords_html.size == 0:
            fig = go.Figure()
            fig.update_layout(
                title=f"{base_name} | No points above threshold={args.threshold:.2f}",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                ),
            )
            fig.write_html(str(html_path), include_plotlyjs="cdn")
        else:
            xh = coords_html[:, 2]
            yh = coords_html[:, 1]
            zh = coords_html[:, 0]
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=xh,
                        y=yh,
                        z=zh,
                        mode="markers",
                        marker=dict(
                            size=1.5,
                            color=colors_html,
                            colorscale="Inferno",
                            opacity=0.45,
                            showscale=True,
                            colorbar=dict(title="Intensity"),
                        ),
                    )
                ]
            )
            fig.update_layout(
                title=f"{base_name} | Interactive 3D Point Cloud",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="manual",
                    aspectratio=dict(x=vol.shape[2], y=vol.shape[1], z=vol.shape[0]),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )
            fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"  saved       : {html_path}")

    print("3D visualization complete.")
    print(f"  stack       : {stack_path}")
    print(f"  shape       : {tuple(vol_raw.shape)}")
    print(f"  center idx  : z={z}, y={y}, x={x}")
    print(f"  points used : {n_pts}")
    print(f"  saved       : {orth_path}")
    print(f"  saved       : {cloud_path}")


if __name__ == "__main__":
    main()

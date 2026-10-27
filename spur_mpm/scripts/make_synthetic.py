"""Generate a SYNTHETIC vessel-like ZYX dataset for pipeline smoke tests.

**This is not evidence.** It exists so the code path (audit -> split -> prepare
-> baselines -> train -> evaluate) can be exercised end-to-end before the real
Dryad stacks arrive. Any number produced on these volumes is a test of the
plumbing, never a scientific result, and results are tagged
``SYNTHETIC_SMOKE`` everywhere they are written.

The generator draws a few hundred random 3-D tubes with Poisson-like noise and
an axial sensitivity falloff, then writes 16-bit TIFFs so the loader, the axis
audit and the calibration path are all exercised exactly as they will be on real
data.

    python -m spur_mpm.scripts.make_synthetic --out data_local/synthetic \
        --spec spur_mpm/configs/data/synthetic_smoke.yaml --mice 3 --fovs 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

__all__ = ["make_volume", "write_dataset"]


def make_volume(depth: int = 32, height: int = 256, width: int = 256,
                n_vessels: int = 18, seed: int = 0, z_falloff: float = 0.6,
                photons: float = 900.0) -> np.ndarray:
    """A synthetic vessel volume in uint16, shaped ``[Z, H, W]``."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((depth, height, width), dtype=np.float32)

    zz, yy, xx = np.mgrid[0:depth, 0:height, 0:width]
    for _ in range(n_vessels):
        # A tube is a smooth 3-D parametric curve with a finite radius.
        p0 = rng.uniform([0, 0.05, 0.05], [1, 0.95, 0.95]) * np.array([depth, height, width])
        amp = rng.uniform(0.05, 0.25, size=3) * np.array([depth, height, width])
        freq = rng.uniform(0.5, 2.5, size=3)
        phase = rng.uniform(0, 2 * np.pi, size=3)
        radius = rng.uniform(1.2, 3.5)
        intensity = rng.uniform(0.5, 1.0)
        t = np.linspace(0, 1, max(24, depth * 2))
        pts = p0[None] + amp[None] * np.sin(2 * np.pi * freq[None] * t[:, None] + phase[None])
        pts = np.clip(pts, 0, np.array([depth - 1, height - 1, width - 1]))

        r = int(np.ceil(radius * 2))
        for zc, yc, xc in pts:
            zi, yi, xi = int(round(zc)), int(round(yc)), int(round(xc))
            z0, z1 = max(0, zi - r), min(depth, zi + r + 1)
            y0, y1 = max(0, yi - r), min(height, yi + r + 1)
            x0, x1 = max(0, xi - r), min(width, xi + r + 1)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue
            d2 = ((zz[z0:z1, y0:y1, x0:x1] - zc) ** 2 +
                  (yy[z0:z1, y0:y1, x0:x1] - yc) ** 2 +
                  (xx[z0:z1, y0:y1, x0:x1] - xc) ** 2)
            vol[z0:z1, y0:y1, x0:x1] += intensity * np.exp(-d2 / (2 * radius ** 2))

    vol /= max(vol.max(), 1e-6)
    # Axial sensitivity falloff, as a real deep-tissue stack would show.
    profile = np.exp(-z_falloff * np.arange(depth) / max(1, depth - 1))
    vol *= profile[:, None, None]
    vol += 0.03 * rng.random(vol.shape)          # background haze

    scaled = vol / max(vol.max(), 1e-6) * photons
    noisy = rng.poisson(np.clip(scaled, 0, None)).astype(np.float32)
    noisy += rng.normal(0, 2.0, size=noisy.shape)  # readout noise
    return np.clip(noisy, 0, 65535).astype(np.uint16)


def write_dataset(out_root: Path, spec_path: Path, mice: int = 3, fovs: int = 2,
                  depth: int = 32, height: int = 256, width: int = 256,
                  z_step_um: float = 5.0, voxel_xy_um: float = 0.98) -> Path:
    import tifffile

    tiff_dir = out_root / "tiff"
    tiff_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    seed = 0
    for m in range(1, mice + 1):
        for f in range(1, fovs + 1):
            seed += 1
            sid = f"SYN_M{m}_FOV{f}"
            vol = make_volume(depth, height, width, seed=seed)
            path = tiff_dir / f"{sid}.tif"
            tifffile.imwrite(path, vol, imagej=True,
                             metadata={"axes": "ZYX", "spacing": z_step_um,
                                       "unit": "um"})
            entries.append({
                "glob": f"tiff/{sid}.tif",
                "metadata": {
                    "sample_id": sid, "mouse_id": f"M{m}", "session_id": "S1",
                    "fov_id": f"FOV{f}", "channel": "vessels", "axes": "ZYX",
                    "voxel_xy_um": voxel_xy_um, "z_step_um": z_step_um,
                    "excitation_nm": 1340, "depth_origin_um": 0.0,
                },
            })
            print(f"wrote {path} {vol.shape} {vol.dtype}", flush=True)

    spec = {
        "name": "synthetic_smoke",
        "root": str(out_root.resolve()),
        "defaults": {"axes": "ZYX", "channel": "vessels",
                     "voxel_xy_um": voxel_xy_um, "z_step_um": z_step_um},
        "files": entries,
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"wrote spec {spec_path}")
    return spec_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data_local/synthetic"))
    p.add_argument("--spec", type=Path, default=Path("spur_mpm/configs/data/synthetic_smoke.yaml"))
    p.add_argument("--mice", type=int, default=3)
    p.add_argument("--fovs", type=int, default=2)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    args = p.parse_args()
    write_dataset(args.out, args.spec, mice=args.mice, fovs=args.fovs,
                  depth=args.depth, height=args.height, width=args.width)


if __name__ == "__main__":
    main()

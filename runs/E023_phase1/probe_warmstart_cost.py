"""What the physics warm start costs, with and without a PSF.

The cost table now times A0/A1's operator, but under the experiment's identity
PSF those two are the *same* operator (``_wiener_pipeline`` returns the filled
volume unchanged when ``psf.is_identity``), which is why A1 has always been
numerically identical to A0. This probe records both that degeneracy and what a
real, calibrated PSF would cost -- the number behind "a genuine Wiener gain needs
a measured PSF" (runbook §6.6).

    PYTHONPATH=. python runs/E023_phase1/probe_warmstart_cost.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.wiener import warm_start

DEVICE = "cuda:2"
Z, H, W = 200, 256, 256          # the volume convention of the cost table
N_ITERS = 10
CASES = {
    "identity": dict(sigma_xy_um=0.0, sigma_z_um=0.0),
    "sigma_xy_0.98um": dict(sigma_xy_um=0.98, sigma_z_um=0.0),
    "sigma_xy_0.98um_z_5um": dict(sigma_xy_um=0.98, sigma_z_um=5.0),
}


def main() -> int:
    dev = torch.device(DEVICE)
    y = torch.rand(Z, H, W, device=dev)
    mask = torch.zeros(Z, H, W, dtype=torch.bool, device=dev)
    mask[:, ::2, :] = True
    y = y * mask

    rows = []
    for case, sig in CASES.items():
        psf = PSFConfig(voxel_xy_um=0.98, voxel_z_um=5.0, **sig)
        for mode in ("bilinear", "wiener_fill_first"):
            for _ in range(3):
                with torch.no_grad():
                    warm_start(y, mask, psf, lam=0.01, mode=mode)
            torch.cuda.synchronize(dev)
            samples = []
            for _ in range(N_ITERS):
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                with torch.no_grad():
                    warm_start(y, mask, psf, lam=0.01, mode=mode)
                torch.cuda.synchronize(dev)
                samples.append((time.perf_counter() - t0) * 1e3)
            samples.sort()
            rows.append({"psf": case, "mode": mode,
                         "median_ms_per_volume": round(samples[len(samples) // 2], 3),
                         "p10_ms": round(samples[0], 3),
                         "p90_ms": round(samples[-1], 3),
                         "device": DEVICE,
                         "volume": [Z, H, W], "n_iters": N_ITERS})
            print(f"{case:24s} {mode:18s} {rows[-1]['median_ms_per_volume']:7.2f} ms")

    out = Path("runs/E021_ladder/warmstart_cost.json")
    out.write_text(json.dumps({
        "what": "warm-start operator cost per volume, as a function of the PSF",
        "why": "the identity PSF makes A1 collapse onto A0; this quantifies both "
               "that degeneracy and the cost a calibrated PSF would add",
        "device": DEVICE, "note": "operator only: no dataset, no disk I/O",
        "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

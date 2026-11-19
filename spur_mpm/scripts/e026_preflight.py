"""E026 preflight: look at Y, M, the target and the candidate operators before GPU.

The historical recheck asks for this step by name ("Run ONLY after full test and
visual check of Y/M/GT and 2D vs 3D candidate"). It is not an experiment: it
scores nothing on test animals and reports no method result. It answers four
questions on **training** volumes only:

1. Is the mask what the protocol claims (25 % retained, Z and Y both skipped)?
2. Is the target the dense recording D, and is Y = M*(H*D) for the synthetic
   group while Y = M*D for the real group?
3. Do the three prior operators (interp, full-3D fill-then-Wiener, historical
   lateral-Wiener-then-Z-fill) differ, and by how much on the *missing* voxels?
4. Does any of them ring or go negative where there is nothing to recover?

Writes one JSON and one PNG panel per group under ``--out``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spur_mpm.data.datasets import ItemConfig, VolumeCache, _make_item
from spur_mpm.data.split import folds_by_name, read_split
from spur_mpm.physics.acquisition import apply_otf
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.wiener import (interp_linear, warm_start,
                                     wiener_fill_all)

DEFAULT_INDEX = "data_local/dryad_cache/index_cohort.json"
DEFAULT_SPLIT = "spur_mpm/experiments/E020_cohort/split.json"


def _panels(gt, y, m, cands, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = gt.shape[0] // 2
    cols = [("target D", gt[z]), ("Y (masked)", y[z]),
            ("M", np.asarray(m[z], dtype=np.float32))] + \
           [(k, np.asarray(v[z])) for k, v in cands.items()]
    errs = [(f"{k} |error| on missing",
             np.asarray(np.abs(v - gt) * (~np.asarray(m, dtype=bool)))[z])
            for k, v in cands.items()]
    n = max(len(cols), len(errs))
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4), squeeze=False)
    for j in range(n):
        for i, row in enumerate((cols, errs)):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(row):
                ax.axis("off"); continue
            name, img = row[j]
            vmax = np.percentile(np.abs(img), 99.5) or 1.0
            ax.imshow(img, cmap="gray", vmin=None if i else 0.0, vmax=vmax)
            ax.set_title(name, fontsize=7)
    fig.suptitle(title + f"  (plane z={z})", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(*, index: str, split: str, fold: int, source_kind: str,
        ky: int, kz: int, known_xy_um: float, known_z_um: float,
        lam: float, seed: int, out: Path, z_slice: int | None) -> dict:
    folds = folds_by_name(read_split(split))
    train = list(folds[fold]["train"])
    cache = VolumeCache.load(index, fold=fold)
    sid = [s for s in train if s in cache][0]
    geom = cache.geometry(sid)
    raw = cache.array(sid)
    # Slice before calibrating: the calibration is per-voxel, so this is the
    # same array and it avoids a full-volume float32 temporary (200 MB on this
    # cohort) that the box's current memory pressure makes expensive.
    if z_slice is not None:
        lo = max(0, min(raw.shape[0] - 16, z_slice))
        raw = raw[lo:lo + 16]
    else:
        raw = raw[:max(16, min(48, raw.shape[0]))]
    dense = np.ascontiguousarray(
        cache.calibration.apply(np.ascontiguousarray(raw))).astype("float32")

    known = PSFConfig(sigma_xy_um=known_xy_um, sigma_z_um=known_z_um,
                      voxel_xy_um=geom.voxel_xy_um, voxel_z_um=geom.z_step_um)
    if source_kind == "matched_synthetic":
        prior, forward, src = known, known, "synthetic_extra_degradation"
    else:
        prior, forward, src = PSFConfig(), PSFConfig(), None
    cfg = ItemConfig(sampling=SamplingSpec(mode="yz", ky=ky, kz=kz),
                     psf=prior, psf_source=src,
                     forward_psf=forward, forward_psf_source=src,
                     warm_start_kind="wiener_fill_all", lam=lam,
                     features=("y", "mask"), check_leakage=False)
    center = dense.shape[0] // 2
    item = _make_item(dense, geom, cfg, 0, 0, 0, seed, center=center)
    n = dense.shape[0]
    inp = item["inputs"]
    assert inp.shape[0] % n == 0, (
        f"inputs has {inp.shape[0]} channel-planes, expected a multiple of {n}")
    assert inp.shape[0] == 2 * n, f"expected features (y, mask), got {inp.shape}"
    y = inp[0:n]
    m = inp[n:2 * n].bool()
    assert y.shape == dense.shape and m.shape == dense.shape
    gt = item["target"]
    assert torch.equal(gt, torch.from_numpy(dense[center]))

    ref = torch.from_numpy(dense)
    expect_forward = apply_otf(ref, forward) if not forward.is_identity else ref
    forward_ok = torch.allclose(y[m], expect_forward[m], atol=1e-6)
    target_ok = torch.allclose(gt, ref[center])
    no_blur = torch.allclose(y[m], ref[m], atol=1e-6)

    cands = {
        "I0 interp": interp_linear(torch.where(m, y, torch.zeros_like(y)), m),
        "W0 full 3D": wiener_fill_all(torch.where(m, y, torch.zeros_like(y)), m,
                                      prior, lam),
        "W0 legacy": warm_start(torch.where(m, y, torch.zeros_like(y)), m,
                                prior, lam, mode="wiener_fill_first"),
    }
    miss = ~m
    rows = {}
    for k, v in cands.items():
        d = (v - ref)[miss]
        rows[k] = {"mse_missing": float((d ** 2).mean()),
                   "mae_missing": float(d.abs().mean()),
                   "min": float(v.min()), "max": float(v.max()),
                   "changed_vs_interp": float((v - cands["I0 interp"]).abs().mean()),
                   "finite": bool(torch.isfinite(v).all())}
    pair = float((cands["W0 full 3D"] - cands["W0 legacy"]).abs().mean())
    summary = {
        "kind": "preflight", "group": source_kind, "train_volume": sid,
        "protocol": {"mode": "yz", "ky": ky, "kz": kz, "fold": fold,
                     "sampled_fraction": float(m.float().mean()),
                     "planes_sampled": int(m.any(dim=(-2, -1)).sum()),
                     "planes_total": int(m.shape[0]),
                     "rows_sampled_per_plane": int(m[0].any(dim=-1).sum()),
                     "rows_total": int(m.shape[1]),
                     "context_planes": int(n)},
        "checks": {"target_is_dense_recording": bool(target_ok),
                   "y_equals_forward_H_of_D_once": bool(forward_ok),
                   "y_equals_D_on_measured": bool(no_blur),
                   "prior_kernel_used": prior.to_dict(),
                   "forward_kernel_used": forward.to_dict()},
        "candidates": rows,
        "full3d_vs_legacy_mean_abs_diff": pair,
        "note": ("Preflight runs on a TRAIN volume and reports no method result. "
                 "In the matched_synthetic group the numbers describe a known-"
                 "kernel synthetic inverse problem, not the recorded stacks."),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"preflight_{source_kind}.json").write_text(json.dumps(summary, indent=2))
    # Full volumes in, one plane shown: passing the centre plane *and* indexing
    # it inside would show a 1-D row and matplotlib would refuse the image.
    _panels(ref.numpy(), y.numpy(), m.numpy(),
            {k: v.numpy() for k, v in cands.items()},
            out / f"preflight_{source_kind}.png", f"E026 {source_kind} ({sid})")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--source-kind", choices=("real", "matched_synthetic"),
                   default="matched_synthetic")
    p.add_argument("--ky", type=int, default=2)
    p.add_argument("--kz", type=int, default=2)
    p.add_argument("--known-xy-um", type=float, default=0.98)
    p.add_argument("--known-z-um", type=float, default=7.5)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z-slice", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    res = run(index=a.index, split=a.split, fold=a.fold,
              source_kind=a.source_kind, ky=a.ky, kz=a.kz,
              known_xy_um=a.known_xy_um, known_z_um=a.known_z_um, lam=a.lam,
              seed=a.seed, out=a.out, z_slice=a.z_slice)
    print(json.dumps({k: res[k] for k in
                      ("group", "train_volume", "protocol", "checks",
                       "full3d_vs_legacy_mean_abs_diff")}, indent=2))
    for k, v in res["candidates"].items():
        print(f"  {k:12s} mse_missing={v['mse_missing']:.6f} "
              f"mae_missing={v['mae_missing']:.6f} "
              f"range=({v['min']:.3f},{v['max']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

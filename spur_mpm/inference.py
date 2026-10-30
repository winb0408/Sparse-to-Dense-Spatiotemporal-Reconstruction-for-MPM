"""Full-volume inference with the complete real acquisition mask.

    python -m spur_mpm.inference --config <exp>.yaml --arm SS1 \
        --ckpt runs/<run>/best.pt --sample SYN_M1_FOV1 \
        --out data_local/recon/SYN_M1_FOV1_SS1.npy

Differences from evaluation, and why they matter (plan §5.4):

* no holdout is carved out -- every position the scanner actually measured is
  available, because at deployment nothing is being withheld for training;
* the depth context is rebuilt from the same fixed physical range used in
  training, so the model does not silently gain access to neighbouring planes
  that would not exist in a real sparse scan;
* the output is a full ``ZYX`` volume obtained by stacking per-plane
  predictions, plus the per-plane acquisition mask, so any later figure can say
  exactly which voxels were measured and which were inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spur_mpm.common.config import apply_overrides, deep_merge, load_config
from spur_mpm.common.tensors import select_centre
from spur_mpm.data.datasets import ItemConfig, VolumeCache, VolumeWindowDataset
from spur_mpm.models.registry import build_model, eval_features, get_arm
from spur_mpm.physics.acquisition import NoiseConfig
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.ssl.holdout import HoldoutSpec
from spur_mpm.physics.sampling import SamplingSpec

__all__ = ["reconstruct_volume", "main"]

DEFAULTS = Path(__file__).parent / "configs" / "base.yaml"


@torch.no_grad()
def reconstruct_volume(cfg: dict, arm_name: str, ckpt: str | Path,
                       sample_id: str, out_path: str | Path,
                       device: str | None = None) -> dict:
    device = torch.device(device or (cfg["run"].get("device", "cuda")
                                     if torch.cuda.is_available() else "cpu"))
    arm = get_arm(arm_name)
    cache = VolumeCache.load(cfg["data"]["index"],
                              fold=int(cfg["data"].get("fold", 0)))
    geom = cache.geometry(sample_id)

    sampling = SamplingSpec.from_dict(cfg["sampling"])
    psf = PSFConfig(**cfg["physics"]["psf"])
    noise = NoiseConfig(**cfg.get("noise", {"poisson_photons": 0.0,
                                            "gaussian_sigma": 0.0, "seed": 0}))
    context = int(cfg["data"].get("context_slices", 1))
    if cfg["data"].get("context_z_um"):
        context = geom.context_depth(context_z_um=float(cfg["data"]["context_z_um"]))

    item_cfg = ItemConfig(
        sampling=sampling, psf=psf, noise=noise, warm_start_kind=arm.warm_start,
        lam=float(cfg["physics"]["lam"]), features=eval_features(arm),
        supervision="dense", holdout=(
            HoldoutSpec(ratio=float(cfg["selfsupervision"]["ratio"]),
                        geometry=str(cfg["selfsupervision"]["geometry"]),
                        seed=0) if arm.dual else None),
        dual=arm.dual,
        u_kind=cfg["physics"].get("u_kind", "geo"),
        geo_mode=cfg["physics"].get("geo_mode", "per_axis"),
    )
    ds = VolumeWindowDataset(cache, [sample_id], item_cfg, patch_hw=None,
                             context_depth=context, z_stride=1)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    model = None
    if arm.needs_training:
        model = build_model(arm, context, base=int(cfg["model"]["base"]),
                            window=int(cfg["model"]["window"]),
                            depths=tuple(cfg["model"]["depths"]),
                            heads=tuple(cfg["model"]["heads"])).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()

    volume = None
    masks, x0s = None, None
    with torch.no_grad():
        for batch in loader:
            if model is not None:
                pred = select_centre(model(batch["inputs"].to(device)).float(),
                                     batch["center_index"])
                pred = pred[0].cpu().numpy()
            else:
                # Baseline arm: the physics warm start for the centre plane.
                f = list(item_cfg.features).index("x0")
                pred = batch["inputs"][0, f * context + context // 2].numpy()

            zq = int(batch["query_z"][0])
            if volume is None:
                h, w = pred.shape
                volume = np.zeros((geom.depth, h, w), dtype=np.float32)
                masks = np.zeros((geom.depth, h, w), dtype=bool)
                x0s = np.zeros((geom.depth, h, w), dtype=np.float32)
            volume[zq] = pred
            masks[zq] = batch["acq_mask"][0].numpy().astype(bool)
            f_x0 = list(item_cfg.features).index("x0")
            x0s[zq] = batch["inputs"][0, f_x0 * context + context // 2].numpy()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, volume)
    np.save(out_path.with_suffix(".mask.npy"), masks)
    np.save(out_path.with_suffix(".warmstart.npy"), x0s)

    meta = {
        "sample_id": sample_id,
        "arm": arm.name,
        "checkpoint": str(ckpt) if ckpt else None,
        "shape": list(volume.shape),
        "geometry": geom.to_dict(),
        "sampling": sampling.to_dict(),
        "context_depth": context,
        "physics": {"psf": psf.to_dict(), "lam": cfg["physics"]["lam"]},
        "notes": "virtual sparse input; not a hardware acquisition. Volume built "
                 "by stacking centre-plane predictions; see *.mask.npy for the "
                 "measured support of each plane.",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return {"volume": str(out_path), "meta": meta}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--base", type=Path, default=DEFAULTS)
    p.add_argument("--arm", required=True)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--sample", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    cfg = load_config(args.base)
    if args.config:
        cfg = deep_merge(cfg, load_config(args.config))
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    result = reconstruct_volume(cfg, args.arm, args.ckpt, args.sample, args.out)
    print(json.dumps(result["meta"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

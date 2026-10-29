"""Training entry point.

    python -m spur_mpm.train --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
        --arm S1 --fold 0 --seed 0

One code path for every trained arm. What the arm changes is a *config*, never
a separate pipeline (plan §4.3), so a difference between two arms cannot come
from a difference in their data handling.

Model selection (plan §5.3), which is where self-supervision is most often
faked:

* supervised arms (``S*``) may select on dense-reference validation PSNR --
  they are explicitly the labelled upper bound;
* self-supervised arms (``SS*``) select on the **held-out measurement loss** of
  a separate validation group. The dense reference is never read while a
  self-supervised model is being trained or selected; it is only computed by
  ``evaluate.py`` after the checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spur_mpm.common.config import apply_overrides, config_hash, load_config
from spur_mpm.common.provenance import collect_provenance
from spur_mpm.common.runlog import RunDir
from spur_mpm.common.seeding import seed_worker, set_seed
from spur_mpm.common.tensors import select_centre as _select_centre
from spur_mpm.data.datasets import (
    ItemConfig,
    SparseVolumeDataset,
    VolumeCache,
    VolumeWindowDataset,
)
from spur_mpm.data.split import read_split
from spur_mpm.losses.reconstruction import ReconstructionLoss
from spur_mpm.metrics.image_metrics import psnr, ssim
from spur_mpm.models.registry import build_model, count_parameters, get_arm
from spur_mpm.ssl.holdout import HoldoutSpec
from spur_mpm.ssl.mms import MMSLoss
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.physics.acquisition import NoiseConfig

__all__ = ["main", "run_training"]

DEFAULTS = Path(__file__).parent / "configs" / "base.yaml"


def build_components(cfg: dict, arm_name: str, seed: int):
    """Datasets, loaders and the item config implied by an arm + config."""
    arm = get_arm(arm_name)
    cache = VolumeCache.load(cfg["data"]["index"],
                              fold=int(cfg["data"].get("fold", 0)))
    split = read_split(cfg["data"]["split"])
    folds = {int(f["fold"]): f for f in split["folds"]}
    fold = folds[int(cfg["data"]["fold"])]

    sampling = SamplingSpec.from_dict(cfg["sampling"])
    psf = PSFConfig(**cfg["physics"]["psf"])
    noise = NoiseConfig(**cfg.get("noise", {"poisson_photons": 0.0,
                                            "gaussian_sigma": 0.0, "seed": 0}))
    # Who loses half of the measured rows, and why:
    #   self          the objective needs withheld *measurements* as targets
    #   dense + holdout (E011) a control arm: same input as the self-supervised
    #                 one, dense target -- isolates input quantity from labels
    #   dual (E012)   both branches lose half, but disjoint halves, so the pair
    #                 covers the whole acquisition
    withhold = (arm.supervision == "self" or arm.dual or
                bool(cfg["selfsupervision"].get("split_supervised", False)))
    holdout = HoldoutSpec(ratio=float(cfg["selfsupervision"]["ratio"]),
                          geometry=str(cfg["selfsupervision"]["geometry"]),
                          seed=seed) if withhold else None

    item_cfg = ItemConfig(
        sampling=sampling, psf=psf, noise=noise,
        warm_start_kind=arm.warm_start, lam=float(cfg["physics"]["lam"]),
        features=arm.features, supervision=arm.supervision, holdout=holdout,
        dual=arm.dual,
        u_kind=cfg["physics"].get("u_kind", "geo"),
        geo_mode=cfg["physics"].get("geo_mode", "per_axis"),
    )

    geom0 = cache.geometry(fold["train"][0])
    context = int(cfg["data"].get("context_slices", 1))
    if cfg["data"].get("context_z_um"):
        context = geom0.context_depth(context_z_um=float(cfg["data"]["context_z_um"]))

    patch_hw = tuple(cfg["data"]["patch_hw"])
    train_ds = SparseVolumeDataset(
        cache, fold["train"], item_cfg, patch_hw=patch_hw, context_depth=context,
        patches_per_epoch=int(cfg["train"]["patches_per_epoch"]), seed=seed)
    val_ds = VolumeWindowDataset(
        cache, fold["val"], item_cfg, patch_hw=patch_hw, context_depth=context,
        z_stride=int(cfg["eval"]["z_stride"]))

    nw = int(cfg["train"]["num_workers"])
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch"]),
                              shuffle=True, num_workers=nw, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker,
                              generator=g)
    val_loader = DataLoader(val_ds, batch_size=max(1, int(cfg["train"]["batch"]) // 2),
                            num_workers=nw, pin_memory=True)
    return arm, train_ds, train_loader, val_loader, context, sampling, psf




@torch.no_grad()
def validate(model, loader, device, *, supervision: str, rho: str,
             amp: bool, max_batches: int | None = None,
             dual: bool = False) -> dict:
    """Validation on a held-out group.

    Supervised arms see dense-reference PSNR/SSIM and select on it -- they are
    the labelled upper bound by design.

    Self-supervised arms select on the held-out measurement loss **and do not
    compute the dense metrics at all**. Not "compute but ignore": if the number
    exists in the log, someone will eventually pick a checkpoint with their eyes.
    The dense reference for an SS arm is produced by ``evaluate.py`` after the
    weights are frozen (plan §2.3, strict self-supervised accounting).
    """
    model.eval()
    crit = MMSLoss(rho=rho)
    use_dense = supervision != "self"
    ps, ss, ms = [], [], []
    n_batches = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        n_batches += 1
        x = batch["inputs"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            if dual:
                out_a, out_b = model.forward_both(x)
            else:
                out = model(x)
        if dual:
            pred = 0.5 * (_select_centre(out_a.float(), batch["center_index"]) +
                          _select_centre(out_b.float(), batch["center_index"]))
        else:
            pred = _select_centre(out.float(), batch["center_index"])
        if use_dense:
            target = batch["target"].to(device)
            ps.append(float(psnr(pred, target)))
            ss.append(float(ssim(pred[:, None], target[:, None])))
        if dual:
            # The selection signal is each branch's own held-out error, kept
            # separate: averaging the predictions first would score a model that
            # is not the one that gets deployed and would flatter a branch that
            # simply copies its neighbour.
            y_hold = batch["hold_target"].to(device)
            m_a = batch["hold_mask_a"].to(device)
            m_b = batch["hold_mask_b"].to(device)
            la = crit(_select_centre(out_a.float(), batch["center_index"]), y_hold, m_a)
            lb = crit(_select_centre(out_b.float(), batch["center_index"]), y_hold, m_b)
            ms.append(0.5 * (float(la["mms"]) + float(lb["mms"])))
        elif batch["hold_mask"].sum() > 0:
            res = crit(pred, batch["hold_target"].to(device),
                       batch["hold_mask"].to(device))
            ms.append(float(res["mms"]))
    model.train()
    out = {"n_val": n_batches}
    if use_dense:
        out["dense_psnr"] = _mean(ps)
        out["dense_ssim"] = _mean(ss)
    if ms:
        out["holdout_loss"] = _mean(ms)
    # The selection metric, stated explicitly so the run log records *what*
    # chose the checkpoint.
    if supervision == "self":
        if out.get("holdout_loss") is None:
            # A self-supervised arm with nothing withheld has no selection
            # signal; picking the first checkpoint would be arbitrary.
            raise RuntimeError(
                "self-supervised selection requires a held-out measurement loss "
                "on the validation group, but no holdout voxel was present")
        out["select_metric"] = "holdout_loss"
        out["select_value"] = out["holdout_loss"]
        out["select_maximise"] = False
    else:
        out["select_metric"] = "dense_psnr"
        out["select_value"] = out.get("dense_psnr")
        out["select_maximise"] = True
    return out


def _mean(v: list[float]) -> float | None:
    return sum(v) / len(v) if v else None


def run_training(cfg: dict, arm_name: str, out_root: Path, name: str | None,
                 resume: bool = False) -> dict:
    device = torch.device(cfg["run"].get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    seed = int(cfg["run"]["seed"])
    set_seed(seed)

    arm, train_ds, train_loader, val_loader, context, sampling, psf = \
        build_components(cfg, arm_name, seed)

    model = build_model(arm, context, base=int(cfg["model"]["base"]),
                        window=int(cfg["model"]["window"]),
                        depths=tuple(cfg["model"]["depths"]),
                        heads=tuple(cfg["model"]["heads"])).to(device)

    run_name = name or f"{cfg['experiment']}__{arm.name}__{sampling.tag}__s{seed}"
    run = RunDir(out_root, run_name, resume=resume)
    cfg = dict(cfg)
    cfg["model"].update({"arm": arm.name, "features": list(arm.features),
                         "context_depth": context,
                         "n_parameters": count_parameters(model)})
    run.write_config(cfg)
    run.write_provenance(collect_provenance(
        config=cfg,
        artifacts={"split": cfg["data"]["split"], "index": cfg["data"]["index"]},
    ))
    print(f"[{run_name}] arm={arm.name} params={count_parameters(model):,} "
          f"context={context} sampling={sampling.tag} device={device}", flush=True)

    if arm.supervision == "self":
        train_crit = MMSLoss(rho=cfg["loss"]["rho"],
                             dc_weight=float(cfg["loss"].get("dc_weight", 0.0)))
    else:
        train_crit = ReconstructionLoss(w_l1=float(cfg["loss"]["l1"]),
                                        w_ms=float(cfg["loss"]["ms_ssim"]))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    agree = float(cfg["loss"].get("dual_agree", 0.0))
    steps = int(cfg["train"]["steps"])
    warmup = int(cfg["train"].get("warmup_steps", 0))
    accum = max(1, int(cfg["train"].get("accum", 1)))
    amp = bool(cfg["train"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best = float("inf") if arm.supervision == "self" else float("-inf")
    start = 0
    ckpt_path = run.path / "resume.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["step"], ck["best"]
        print(f"[{run_name}] resumed at step {start} (best={best:.4f})", flush=True)

    def batches():
        while True:
            for b in train_loader:
                yield b

    step = start
    model.train()
    for batch in batches():
        if step >= steps:
            break
        x = batch["inputs"].to(device, non_blocking=True)
        if arm.dual:
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                out_a, out_b = model.forward_both(x)
            ci = batch["center_index"]
            pred_a = _select_centre(out_a.float(), ci)
            pred_b = _select_centre(out_b.float(), ci)
            y_hold = batch["hold_target"].to(device)
            ra = train_crit(pred_a, y_hold, batch["hold_mask_a"].to(device))
            rb = train_crit(pred_b, y_hold, batch["hold_mask_b"].to(device))
            loss = ra["loss"] + rb["loss"]
            res = {"loss": loss, "mms_a": ra["mms"], "mms_b": rb["mms"]}
            if agree > 0:
                # Optional extra: the branches should agree where *neither* was
                # given anything. Detached on one side so the term cannot be
                # minimised by both branches drifting together for free.
                dis = 0.5 * (train_crit._rho(pred_a - pred_b.detach()).mean() +
                             train_crit._rho(pred_b - pred_a.detach()).mean())
                loss = loss + agree * dis
                res["loss"] = loss
                res["agree"] = dis.detach()
            pred = 0.5 * (pred_a + pred_b)
        else:
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                out = model(x)
            out = out.float()
            pred = _select_centre(out, batch["center_index"])

            if arm.supervision == "self":
                res = train_crit(pred, batch["hold_target"].to(device),
                                 batch["hold_mask"].to(device),
                                 y_in=batch["y_in"].to(device),
                                 m_in=batch["in_mask"].to(device))
                loss = res["loss"]
            else:
                res = train_crit(pred, batch["target"].to(device))
                loss = res["loss"]

        if step % accum == 0:
            opt.zero_grad(set_to_none=True)
        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

        if step % int(cfg["train"]["log_every"]) == 0:
            row = {"step": step, "loss": float(loss),
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(run.elapsed, 1)}
            for k, v in res.items():
                if k != "loss" and hasattr(v, "item"):
                    row[k] = float(v)
            run.log_step(row)

        if (step + 1) % int(cfg["train"]["val_every"]) == 0 or step == steps - 1:
            vm = validate(model, val_loader, device, supervision=arm.supervision,
                          rho=cfg["loss"]["rho"], amp=amp, dual=arm.dual)
            vm.update({"step": step + 1, "arm": arm.name})
            run.log_metrics(vm)
            sel = vm["select_value"]
            better = (sel < best - 1e-9) if arm.supervision == "self" \
                else (sel > best + 1e-9)
            if better:
                best = sel
                torch.save({"model": model.state_dict(), "config": cfg,
                            "val": vm}, run.path / "best.pt")
            dense = vm.get("dense_psnr")
            extra = ("" if dense is None
                     else f" densePSNR={round(dense, 2)}")
            print(f"[{run_name}] step {step+1}: {vm['select_metric']}="
                  f"{sel:.4f}{extra}", flush=True)

        if (step + 1) % int(cfg["train"].get("ckpt_every", 500)) == 0:
            snap = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step + 1, "best": best}
            torch.save(snap, run.path / "resume_tmp.pt")
            os.replace(run.path / "resume_tmp.pt", ckpt_path)
        step += 1

    torch.save({"model": model.state_dict(), "config": cfg}, run.path / "last.pt")
    final = validate(model, val_loader, device, supervision=arm.supervision,
                     rho=cfg["loss"]["rho"], amp=amp, dual=arm.dual)
    (run.path / "final_val.json").write_text(json.dumps(final, indent=2))
    problems = run.validate(expect_weights=True)
    run.close()

    summary = {"run": str(run.path), "arm": arm.name, "best": best,
               "final": final, "problems": problems, "config_hash": config_hash(cfg)}
    if problems:
        print(f"[{run_name}] run directory problems: {problems}", file=sys.stderr)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="experiment YAML")
    p.add_argument("--base", type=Path, default=DEFAULTS, help="base config to merge")
    p.add_argument("--arm", default=None, help="override model.arm")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="dotted override, e.g. --set sampling.ky=4")
    p.add_argument("--name", default=None)
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.base)
    if args.config:
        cfg = _merge(cfg, load_config(args.config))
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    arm = args.arm or cfg["model"]["arm"]
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed

    # run_training always returns a (truthy) result dict; the run is only
    # successful if it left no problems behind (empty/missing artefacts, etc.).
    result = run_training(cfg, arm, args.out_root, args.name, resume=args.resume)
    return 0 if not result.get("problems") else 1


def _merge(base: dict, patch: dict) -> dict:
    from spur_mpm.common.config import deep_merge
    return deep_merge(base, patch)


if __name__ == "__main__":
    raise SystemExit(main())

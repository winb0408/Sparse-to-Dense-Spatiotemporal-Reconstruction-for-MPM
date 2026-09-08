"""Generate reproducible command lines for the mask + Wiener prior-fusion study.

This script does not launch training. It prints shell commands so experiments can
be scheduled manually on the available GPU resources.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    model: str
    sparse_mode: str
    dataset_protocol: str
    extra: tuple[str, ...] = ()


def _base_train(exp: Experiment, data_root: str, cidc25_root: str | None) -> List[str]:
    cmd = [
        "python", "train.py",
        "--task", exp.task,
        "--model", exp.model,
        "--data_root", data_root,
        "--dataset_protocol", exp.dataset_protocol,
        "--sparse_mode", exp.sparse_mode,
        "--run_name", exp.name,
    ]
    if cidc25_root and exp.dataset_protocol == "cidc25_3d":
        cmd.extend(["--cidc25_root", cidc25_root])
    cmd.extend(exp.extra)
    return cmd


def _ablation_suffixes() -> Iterable[tuple[str, tuple[str, ...]]]:
    yield "bilinear_nomask", ("--no_mask_ch", "--prior_fusion", "none")
    yield "bilinear_mask", ("--prior_fusion", "none")
    yield "wiener_nomask", ("--wiener_init", "--no_mask_ch", "--prior_fusion", "none")
    yield "wiener_mask", ("--wiener_init", "--prior_fusion", "none")
    yield "wiener_mask_gated", ("--wiener_init", "--prior_fusion", "gated")


def build_experiments() -> List[Experiment]:
    experiments: List[Experiment] = []

    for mode in ("x", "y", "xy"):
        for suffix, flags in _ablation_suffixes():
            experiments.append(Experiment(
                name=f"prior2d_{mode}_{suffix}",
                task="spatial_2d",
                model="swin_unet_2d",
                sparse_mode=mode,
                dataset_protocol="generic_2d",
                extra=flags,
            ))

    for suffix, flags in _ablation_suffixes():
        experiments.append(Experiment(
            name=f"prior2p5d_z_{suffix}",
            task="spatial_2p5d",
            model="swin_unet_2d",
            sparse_mode="z",
            dataset_protocol="cidc25_3d",
            extra=("--temporal_window", "3", *flags),
        ))

    for suffix, flags in _ablation_suffixes():
        experiments.append(Experiment(
            name=f"prior3d_xyz_{suffix}",
            task="spatiotemporal_3d",
            model="video_swin_unet",
            sparse_mode="xyz",
            dataset_protocol="cidc25_3d",
            extra=("--volume_patch_t", "16", *flags),
        ))

    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Print prior-fusion experiment commands.")
    parser.add_argument("--data_root", default="/data2/wjb/threePM/ThreePM_openDatasets/Denoising_Dataset")
    parser.add_argument("--cidc25_root", default=None)
    parser.add_argument("--only", choices=["2d", "2p5d", "3d", "all"], default="all")
    args = parser.parse_args()

    for exp in build_experiments():
        if args.only != "all":
            key = {"spatial_2d": "2d", "spatial_2p5d": "2p5d", "spatiotemporal_3d": "3d"}[exp.task]
            if key != args.only:
                continue
        cmd = _base_train(exp, args.data_root, args.cidc25_root)
        print(" ".join(cmd))


if __name__ == "__main__":
    main()
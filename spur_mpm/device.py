"""Compute-device resolution that refuses to fall back to the CPU in silence.

Every entry point used to spell this out for itself::

    device = torch.device(cfg["run"].get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")

The fallback is correct on a laptop and wrong here.  On 2026-11-14 the E026
``real`` group was launched with ``CUDA_VISIBLE_DEVICES=3``; the card was
listed by the driver when the launch gate ran, and the floor evaluation
immediately before it finished normally on that card.  By the time the training
arm started, ``torch.cuda.is_available()`` was ``False`` -- so ``P2_ResidualI``
began training on the CPU at roughly 22 s/step, i.e. ~18 hours for a 3000-step
arm that takes ~45 minutes on a GPU, and its ``provenance.json`` recorded
``"gpus": []``.  Nothing failed, nothing warned; the run simply got 24 times
slower and stopped being comparable with the runs around it.

So the fallback now has to be *asked for*:

* ``run.device`` starting with ``cuda`` and CUDA unavailable is an error, with
  the reason and the override printed;
* ``SPUR_ALLOW_CPU=1`` is the visible override, for the cases where a CPU run
  is genuinely wanted (unit tests, a deliberate comparison of the two paths).

A deliberate ``run.device: cpu`` is untouched: that is a decision in the
config, not an accident of the environment.
"""

from __future__ import annotations

import os

import torch

__all__ = ["resolve_device"]

OVERRIDE_ENV = "SPUR_ALLOW_CPU"


def resolve_device(cfg: dict, *, purpose: str = "run") -> torch.device:
    """Resolve the device for ``purpose``, refusing a silent CPU fallback.

    ``purpose`` only appears in the error message; it exists so a failure says
    *what* was about to run on the CPU.
    """
    requested = str(cfg.get("run", {}).get("device", "cuda"))
    wants_cuda = requested.split(":")[0] == "cuda"
    if wants_cuda and not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        if os.environ.get(OVERRIDE_ENV) != "1":
            raise SystemExit(
                f"{purpose}: run.device={requested!r} but torch reports no "
                f"usable CUDA device (CUDA_VISIBLE_DEVICES={visible}).\n"
                "Refusing to fall back to the CPU: this is how a 45-minute arm "
                "became an 18-hour one on 2026-11-14 and recorded itself as "
                "GPU-free (docs/evidence/gpu2_falloff_20261113/).\n"
                f"Check the card with `nvidia-smi -L`, or set {OVERRIDE_ENV}=1 "
                "if a CPU run is what you actually want."
            )
        print(f"{purpose}: {OVERRIDE_ENV}=1, running on the CPU on purpose "
              f"(run.device={requested!r}, CUDA_VISIBLE_DEVICES={visible})",
              flush=True)
        return torch.device("cpu")
    return torch.device(requested)

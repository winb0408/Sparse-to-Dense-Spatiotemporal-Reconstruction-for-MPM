"""GT-independent intensity normalization (PLAN.md P0-3 / M4).

Rule: normalization constants come from TRAINING-SET calibration only,
are stored once, and are applied identically at train / val / test /
inference time.  No code path may ever read the min/max (or any
statistic) of the sample currently being processed.

CIDC25 stacks are int16 with a negative camera offset (about -250 to
-300 DN), so a fixed affine calibration (offset, scale) fitted on the
training stacks is used:

    x_norm = clip((x - offset) / scale, 0, 1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["IntensityCalibration"]


@dataclass(frozen=True)
class IntensityCalibration:
    offset: float
    scale: float

    def apply(self, x: torch.Tensor | np.ndarray):
        """Affine normalization using stored constants only."""
        if isinstance(x, np.ndarray):
            return np.clip((x - self.offset) / self.scale, 0.0, 1.0)
        return ((x - self.offset) / self.scale).clamp(0.0, 1.0)

    def to_dict(self) -> dict:
        return {"offset": self.offset, "scale": self.scale}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "IntensityCalibration":
        d = json.loads(Path(path).read_text())
        return cls(offset=d["offset"], scale=d["scale"])

    @classmethod
    def fit(cls, volumes: list[np.ndarray], low_pct: float = 0.5,
            high_pct: float = 99.9, max_samples: int = 20_000_000) -> "IntensityCalibration":
        """Fit (offset, scale) from TRAINING stacks only.

        offset : low percentile of training voxels (camera offset floor).
        scale  : high percentile of training voxels (dynamic-range ceiling).
        Sub-samples each volume for speed; deterministic (fixed rng).
        """
        rng = np.random.default_rng(0)
        lows, highs = [], []
        for v in volumes:
            flat = v.ravel()
            if flat.size > max_samples:
                flat = flat[rng.choice(flat.size, size=max_samples, replace=False)]
            lows.append(np.percentile(flat, low_pct))
            highs.append(np.percentile(flat, high_pct))
        offset = float(np.mean(lows))
        scale = float(np.mean(highs)) - offset
        if scale <= 0:
            raise ValueError("degenerate calibration: scale <= 0")
        return cls(offset=offset, scale=scale)

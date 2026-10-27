"""GT-independent intensity calibration.

Rule (plan §2.2): normalization constants come from the **training groups
only**, are stored once, and are applied identically everywhere. No code path
may read a statistic of the volume it is currently reconstructing -- that would
leak dense-reference information into a self-supervised run and make the
"few/zero labels" claim false.

    x_norm = clip((x - offset) / scale, 0, 1)

The clip is part of the definition, not an accident: it puts every volume on the
same [0, 1] image-domain scale regardless of laser power drift. The consequence
is that the inverse map ``x = x_norm * scale + offset`` is exact only for values
inside ``[offset, offset + scale]``; anything brighter saturates irreversibly.
Report that saturated fraction (:meth:`IntensityCalibration.saturated_fraction`)
rather than pretending the round trip is lossless -- a volume that clips badly
needs its percentiles revisited, not a quieter docstring.
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

    def apply(self, x):
        if isinstance(x, torch.Tensor):
            return ((x - self.offset) / self.scale).clamp(0.0, 1.0)
        return np.clip((np.asarray(x, dtype=np.float32) - self.offset) / self.scale,
                       0.0, 1.0)

    def inverse(self, x):
        """Undo :meth:`apply` exactly *within* the calibrated range."""
        if isinstance(x, torch.Tensor):
            return x * self.scale + self.offset
        return np.asarray(x) * self.scale + self.offset

    def saturated_fraction(self, x) -> float:
        """Fraction of voxels that :meth:`apply` clips (input not invertible)."""
        a = np.asarray(x, dtype=np.float64)
        if a.size == 0:
            return 0.0
        bad = (a < self.offset) | (a > self.offset + self.scale)
        return float(bad.mean())

    def to_dict(self) -> dict:
        return {"offset": self.offset, "scale": self.scale}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "IntensityCalibration":
        d = json.loads(Path(path).read_text())
        return cls(offset=float(d["offset"]), scale=float(d["scale"]))

    @classmethod
    def fit(cls, volumes: list[np.ndarray], low_pct: float = 0.5,
            high_pct: float = 99.9, max_samples: int = 20_000_000,
            seed: int = 0) -> "IntensityCalibration":
        """Fit ``(offset, scale)`` from **training** volumes only.

        ``offset`` is the low percentile (detector offset floor), ``scale`` the
        dynamic-range ceiling above it. Sub-sampling is deterministic so the
        calibration is reproducible bit for bit.
        """
        if not volumes:
            raise ValueError("calibration needs at least one training volume")
        rng = np.random.default_rng(seed)
        lows, highs = [], []
        for v in volumes:
            flat = np.asarray(v).ravel()
            if flat.size > max_samples:
                flat = flat[rng.choice(flat.size, size=max_samples, replace=False)]
            lows.append(float(np.percentile(flat, low_pct)))
            highs.append(float(np.percentile(flat, high_pct)))
        offset = float(np.mean(lows))
        scale = float(np.mean(highs)) - offset
        if scale <= 0:
            raise ValueError(f"degenerate calibration: scale={scale} <= 0")
        return cls(offset=offset, scale=scale)

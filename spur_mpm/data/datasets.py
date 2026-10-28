"""ZYX volume datasets: runtime virtual-sparse acquisition on real stacks.

Two datasets, one item builder:

* :class:`SparseVolumeDataset` -- training. Random patches, random sampling
  lattice phase anchored to global stack coordinates, optional holdout split.
* :class:`VolumeWindowDataset` -- validation / test / inference. Deterministic
  window grid over a whole volume so predictions can be stacked back into a
  full ``ZYX`` reconstruction.

Everything the network sees is produced by :func:`build_model_inputs`, so there
is one code path for the observation model, and the holdout split happens
*before* that call (plan §5.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spur_mpm.data.geometry import VolumeGeometry, stack_patch
from spur_mpm.data.normalize import IntensityCalibration
from spur_mpm.data.observation import build_model_inputs
from spur_mpm.physics.acquisition import AcquisitionConfig, NoiseConfig, observe
from spur_mpm.physics.psf import PSFConfig
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.ssl.holdout import HoldoutSpec, split_holdout
from spur_mpm.ssl.leakage import (
    assert_holdout_is_measured,
    assert_inputs_blind,
    assert_split_disjoint,
)

__all__ = ["CachedVolume", "VolumeCache", "SparseVolumeDataset", "VolumeWindowDataset"]


# ---------------------------------------------------------------------------
# cache index
# ---------------------------------------------------------------------------

@dataclass
class CachedVolume:
    """One prepared volume: a memory-mappable ``.npy`` plus its geometry."""

    sample_id: str
    path: str
    geometry: VolumeGeometry

    def to_dict(self) -> dict:
        return {"sample_id": self.sample_id, "path": self.path,
                "geometry": self.geometry.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "CachedVolume":
        return cls(d["sample_id"], d["path"], VolumeGeometry.from_dict(d["geometry"]))


class VolumeCache:
    """Lookup from ``sample_id`` to cached arrays and frozen calibrations.

    Calibrations are stored **per fold**, because the calibration of a fold may
    only be fitted on that fold's training animals. A single global calibration
    would be reproducible and convenient -- and would quietly leak every test
    animal's intensity statistics into the normalisation of its own evaluation.
    """

    def __init__(self, volumes: dict[str, CachedVolume],
                 calibration: IntensityCalibration,
                 calibrations: dict[int, IntensityCalibration] | None = None):
        self.volumes = volumes
        self.calibration = calibration
        self.calibrations = dict(calibrations or {})
        self._arrays: dict[str, np.ndarray] = {}

    def calibration_for(self, fold: int | None) -> IntensityCalibration:
        """Calibration to use for a given fold (fold 0 when unspecified)."""
        if fold is None:
            return self.calibration
        if not self.calibrations:
            return self.calibration
        if int(fold) not in self.calibrations:
            raise KeyError(
                f"no calibration for fold {fold}; the cache index has "
                f"{sorted(self.calibrations)}. Re-run prepare_data so every fold "
                "gets a calibration fitted on its own training animals.")
        return self.calibrations[int(fold)]

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.volumes

    def __len__(self) -> int:
        return len(self.volumes)

    def geometry(self, sample_id: str) -> VolumeGeometry:
        return self.volumes[sample_id].geometry

    def array(self, sample_id: str) -> np.ndarray:
        """Memory-mapped array; opened once per worker process."""
        arr = self._arrays.get(sample_id)
        if arr is None:
            vol = self.volumes[sample_id]
            arr = np.load(vol.path, mmap_mode="r")
            expected = vol.geometry.shape
            if expected and tuple(arr.shape) != tuple(expected):
                raise ValueError(f"{sample_id}: cache shape {arr.shape} != manifest "
                                 f"shape {expected}; re-run data preparation")
            self._arrays[sample_id] = arr
        return arr

    @classmethod
    def load(cls, index_path: str | Path, fold: int | None = None) -> "VolumeCache":
        """Load the cache; pass ``fold`` to select that fold's calibration.

        Every consumer must pass the fold it is working on. Leaving it ``None``
        falls back to the index's default calibration, which is only correct for
        inspection tools -- never for a run that reports a number.
        """
        payload = json.loads(Path(index_path).read_text())
        volumes = {v["sample_id"]: CachedVolume.from_dict(v)
                   for v in payload["volumes"]}
        calib = IntensityCalibration.load(payload["calibration"])
        calibrations = {int(k): IntensityCalibration.load(v)
                        for k, v in (payload.get("calibrations") or {}).items()}
        cache = cls(volumes, calib, calibrations)
        if fold is not None:
            cache.calibration = cache.calibration_for(fold)
        return cache


def write_cache_index(volumes: list[CachedVolume], calibration_path: str | Path,
                      out_path: str | Path,
                      calibrations: dict[int, str | Path] | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"volumes": [v.to_dict() for v in volumes],
               "calibration": str(calibration_path),
               "calibrations": {str(k): str(v) for k, v in (calibrations or {}).items()}}
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# item construction (shared by both datasets)
# ---------------------------------------------------------------------------

def required_context_depth(kz: int) -> int:
    """Window depth that still contains a measured plane whatever the phase.

    A k_z lattice leaves ``kz - 1`` consecutive unmeasured planes between two
    measured ones. The query plane can sit ``kz - 1`` planes away from the
    nearest measured plane, so a window needs that radius on both sides.
    """
    return 2 * (int(kz) - 1) + 1


def validate_context_depth(spec, context_depth: int) -> None:
    """Fail loudly instead of training on an input of zeros.

    With plane skipping and a single-plane window, every second query plane has
    no measured data at all: the network would be asked to invent a plane from
    an all-zero input, and a poor result would be blamed on the method rather
    than on the window. This is a configuration error, not a finding.
    """
    kz = int(getattr(spec, "kz", 1))
    need = required_context_depth(kz)
    if kz > 1 and int(context_depth) < need:
        raise ValueError(
            f"sampling {spec.tag} skips planes (kz={kz}) but context_depth="
            f"{context_depth}; at least {need} planes are needed so the window "
            "always contains a measured plane. Increase data.context_slices or "
            "data.context_z_um.")


@dataclass
class ItemConfig:
    """Everything needed to turn a dense patch into one training item."""

    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    psf: PSFConfig = field(default_factory=PSFConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    warm_start_kind: str = "wiener"
    lam: float = 0.01
    features: tuple[str, ...] = ("y", "mask", "x0")
    supervision: str = "dense"            # dense | self
    holdout: HoldoutSpec | None = None
    u_kind: str = "geo"
    geo_mode: str = "per_axis"
    check_leakage: bool = True

    def to_dict(self) -> dict:
        return {"sampling": self.sampling.to_dict(), "psf": self.psf.to_dict(),
                "noise": self.noise.to_dict(), "warm_start": self.warm_start_kind,
                "lam": self.lam, "features": list(self.features),
                "supervision": self.supervision,
                "holdout": self.holdout.to_dict() if self.holdout else None,
                "u_kind": self.u_kind, "geo_mode": self.geo_mode}


def _make_item(dense: np.ndarray, geometry: VolumeGeometry, cfg: ItemConfig,
               z0: int, y0: int, x0: int, holdout_seed: int,
               center: int | None = None) -> dict:
    """Build one (inputs, centre-plane targets) item from a dense patch.

    ``center`` is the index of the query plane inside the window; it defaults to
    the middle plane but is set explicitly when a border window is clamped.
    """
    depth = dense.shape[0]
    if center is None:
        center = depth // 2
    if not 0 <= center < depth:
        raise ValueError(f"centre plane {center} outside window of depth {depth}")

    spec = cfg.sampling
    # Anchor the lattice to the global stack so patches share one lattice.
    phase = (z0 % max(spec.kz, 1),
             y0 % max(spec.ky, 1),
             x0 % max(spec.ky, 1))
    acq = observe(
        torch.from_numpy(dense),
        AcquisitionConfig(sampling=spec, psf=cfg.psf, noise=cfg.noise, phase=phase),
    )

    hold_mask = torch.zeros_like(acq.mask)
    if cfg.supervision == "self":
        if cfg.holdout is None:
            raise ValueError("self-supervised supervision requires a HoldoutSpec")
        spec_h = HoldoutSpec(ratio=cfg.holdout.ratio, geometry=cfg.holdout.geometry,
                             seed=holdout_seed)
        m_in, hold_mask = split_holdout(acq.mask, spec_h)
        y_in = torch.where(m_in, acq.y, torch.zeros_like(acq.y))
    else:
        m_in, y_in = acq.mask, acq.y

    mi = build_model_inputs(
        y_in, m_in, psf=cfg.psf, lam=cfg.lam,
        warm_start_kind=cfg.warm_start_kind,
        u_kind=cfg.u_kind, geo_mode=cfg.geo_mode,
        ky=spec.ky, kz=spec.kz,
        voxel_xy_um=geometry.voxel_xy_um, voxel_z_um=geometry.z_step_um,
    )

    if cfg.check_leakage and cfg.supervision == "self":
        assert_split_disjoint(m_in, hold_mask)
        assert_holdout_is_measured(hold_mask, acq.mask)
        assert_inputs_blind(mi.y, hold_mask)

    target = torch.from_numpy(np.ascontiguousarray(dense[center]))
    return {
        "inputs": mi.stack(cfg.features),
        "target": target,                                   # dense reference
        "hold_target": acq.y[center],                       # real measurement
        "hold_mask": hold_mask[center].float(),              # centre plane only
        "acq_mask": acq.mask[center].float(),
        "phase": torch.tensor(phase, dtype=torch.long),
        "z0": torch.tensor(z0, dtype=torch.long),
        "center_index": torch.tensor(center, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# training dataset
# ---------------------------------------------------------------------------

class SparseVolumeDataset(Dataset):
    """Random-patch dataset with runtime virtual-sparse acquisition.

    Parameters
    ----------
    sample_ids     : volumes this split may draw from (already split-safe).
    patch_hw       : (H, W) patch size.
    context_depth  : odd number of Z planes stacked as input channels. The loss
                     is always taken on the centre plane (plan §3.3).
    patches_per_epoch : dataset length; patches are redrawn per epoch by seed.
    """

    def __init__(self, cache: VolumeCache, sample_ids: list[str], cfg: ItemConfig,
                 patch_hw: tuple[int, int] = (256, 256), context_depth: int = 1,
                 patches_per_epoch: int = 1000, seed: int = 0):
        if not sample_ids:
            raise ValueError("SparseVolumeDataset needs at least one sample")
        validate_context_depth(cfg.sampling, context_depth)
        missing = [s for s in sample_ids if s not in cache]
        if missing:
            raise KeyError(f"samples not in cache: {missing}")
        self.cache = cache
        self.sample_ids = list(sample_ids)
        self.cfg = cfg
        self.patch_hw = tuple(int(v) for v in patch_hw)
        self.depth = int(context_depth)
        if self.depth % 2 != 1:
            raise ValueError(f"context_depth must be odd, got {self.depth}")
        self.patches_per_epoch = int(patches_per_epoch)
        self.seed = int(seed)
        self._check_fits()

    def _check_fits(self) -> None:
        for sid in self.sample_ids:
            geom = self.cache.geometry(sid)
            arr = self.cache.array(sid)
            if self.depth > arr.shape[0]:
                raise ValueError(f"{sid}: context_depth {self.depth} exceeds Z "
                                 f"depth {arr.shape[0]}")
            if self.patch_hw[0] > arr.shape[1] or self.patch_hw[1] > arr.shape[2]:
                raise ValueError(f"{sid}: patch {self.patch_hw} larger than lateral "
                                 f"extent {(arr.shape[1], arr.shape[2])}")
            if not geom.shape:
                geom.shape = tuple(arr.shape)

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, index]).generate_state(1)[0])
        si = int(rng.integers(len(self.sample_ids)))
        sid = self.sample_ids[si]
        geom = self.cache.geometry(sid)
        arr = self.cache.array(sid)

        d, h, w = self.depth, self.patch_hw[0], self.patch_hw[1]
        z0 = int(rng.integers(0, arr.shape[0] - d + 1))
        y0 = int(rng.integers(0, arr.shape[1] - h + 1))
        x0 = int(rng.integers(0, arr.shape[2] - w + 1))

        raw = stack_patch(arr, z0, y0, x0, d, h, w)
        dense = self.cache.calibration.apply(raw).astype(np.float32)
        holdout_seed = int(np.random.SeedSequence(
            [self.seed, index, 77]).generate_state(1)[0])

        item = _make_item(dense, geom, self.cfg, z0, y0, x0, holdout_seed)
        item["sample_id"] = sid
        return item


# ---------------------------------------------------------------------------
# deterministic window dataset (val / test / inference)
# ---------------------------------------------------------------------------

class VolumeWindowDataset(Dataset):
    """Every Z plane of every volume, once, with a fixed acquisition seed.

    The window is centred on the query plane and clamped at the volume borders,
    so centre-plane predictions stack back into a full ``ZYX`` reconstruction.
    """

    def __init__(self, cache: VolumeCache, sample_ids: list[str], cfg: ItemConfig,
                 patch_hw: tuple[int, int] | None = None, context_depth: int = 1,
                 z_stride: int = 1):
        if not sample_ids:
            raise ValueError("VolumeWindowDataset needs at least one sample")
        validate_context_depth(cfg.sampling, context_depth)
        self.cache = cache
        self.sample_ids = list(sample_ids)
        self.cfg = cfg
        self.depth = int(context_depth)
        if self.depth % 2 != 1:
            raise ValueError(f"context_depth must be odd, got {self.depth}")
        self.z_stride = int(z_stride)

        self.windows: list[tuple[str, int, int, int]] = []
        for sid in self.sample_ids:
            arr = self.cache.array(sid)
            h, w = arr.shape[1], arr.shape[2]
            ph, pw = patch_hw if patch_hw else (h, w)
            y0 = max(0, (h - ph) // 2)
            x0 = max(0, (w - pw) // 2)
            for z in range(0, arr.shape[0], self.z_stride):
                self.windows.append((sid, z, y0, x0))
        if not self.windows:
            raise ValueError("no windows produced")
        self.patch_hw = patch_hw

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        sid, zq, y0, x0 = self.windows[index]
        geom = self.cache.geometry(sid)
        arr = self.cache.array(sid)
        half = self.depth // 2
        z0 = min(max(0, zq - half), max(0, arr.shape[0] - self.depth))
        w = self.patch_hw[1] if self.patch_hw else arr.shape[2]
        h = self.patch_hw[0] if self.patch_hw else arr.shape[1]

        raw = stack_patch(arr, z0, y0, x0, self.depth, h, w)
        dense = self.cache.calibration.apply(raw).astype(np.float32)

        # Deterministic holdout seed; inference uses no holdout at all.
        seed = int(np.random.SeedSequence([zq, y0, x0]).generate_state(1)[0])
        cfg = self.cfg
        if cfg.supervision == "self" and cfg.holdout is None:
            raise ValueError("validation of a self-supervised arm must define a "
                             "HoldoutSpec to score held-out measurements")
        centre = zq - z0
        item = _make_item(dense, geom, cfg, z0, y0, x0, seed, center=centre)
        item["sample_id"] = sid
        item["query_z"] = torch.tensor(zq, dtype=torch.long)
        item["window_z0"] = torch.tensor(z0, dtype=torch.long)
        item["center_index"] = torch.tensor(centre, dtype=torch.long)
        return item

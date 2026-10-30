"""Sampling masks: which positions were physically measured (support M).

``M(z, y, x) = 1`` means "the scanner actually acquired this voxel". M is a
measurement support, **not** an uncertainty field (plan §4.1).

Supported protocols (plan §3.2), in the order the project will validate them:

``y``    every ``ky``-th full row is kept, X stays dense. Matches line-skipping
         point scanning: acquisition time scales ~ ``ky`` and in-line pixels are
         nearly free, so this is the physically meaningful first protocol.
``z``    every ``kz``-th plane is kept; other planes are entirely missing.
``yz``   independent row spacing ``ky`` and plane spacing ``kz``
         (time ~ ``ky * kz``).
``xy``   lateral lattice inside every plane (stress test, not a scanning scheme).
``full`` everything measured (dense reference).

Two facts that must never be conflated (plan §3.2):

* the *ideal voxel fraction* a mask leaves is ``1/ky``, ``1/kz`` or
  ``1/(ky*kz)`` (borders aside). It is **not** a speed-up or dose factor;
* a lattice phase keeps the sampled pattern anchored to the whole stack, so
  patches cropped from one volume stay on a single consistent lattice instead
  of re-anchoring at every window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["SamplingSpec", "make_mask", "kept_rows", "kept_planes"]

_MODES = ("y", "z", "yz", "xy", "full")


@dataclass(frozen=True)
class SamplingSpec:
    """Declarative description of a virtual sampling protocol.

    Parameters
    ----------
    mode    : one of ``y | z | yz | xy | full``.
    ky      : row spacing (slow lateral axis). 1 = full rows.
    kz      : plane spacing (depth axis). 1 = full depth.
    pattern : ``xy`` mode only -- ``grid`` (row-cross lattice), ``rows`` or
              ``random`` (scattered pixels).
    seed    : seed for the ``random`` pattern.
    """

    mode: str = "y"
    ky: int = 1
    kz: int = 1
    pattern: str = "grid"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"unknown sampling mode {self.mode!r}; expected {_MODES}")
        if self.ky < 1 or self.kz < 1:
            raise ValueError(f"ky/kz must be >= 1, got ky={self.ky} kz={self.kz}")

    # -- geometry -----------------------------------------------------------
    def ideal_fraction(self) -> float:
        """Analytic measured-voxel fraction, ignoring border effects."""
        fy = 1.0 / self.ky
        fz = 1.0 / self.kz
        if self.mode == "y":
            return fy
        if self.mode == "z":
            return fz
        if self.mode == "yz":
            return fy * fz
        if self.mode == "xy":
            return (2 * self.ky - 1) / (self.ky ** 2) if self.pattern == "grid" \
                else 1.0 / (self.ky ** 2)
        return 1.0

    def mask(self, shape: tuple[int, ...],
             phase: tuple[int, ...] = (0, 0, 0)) -> torch.Tensor:
        return make_mask(self, shape, phase)

    @property
    def tag(self) -> str:
        """Short stable label, e.g. ``y_k2`` / ``yz_k2x4`` / ``full``."""
        if self.mode == "full":
            return "full"
        if self.mode == "y":
            return f"y_k{self.ky}"
        if self.mode == "z":
            return f"z_k{self.kz}"
        if self.mode == "yz":
            return f"yz_k{self.ky}x{self.kz}"
        return f"xy_{self.pattern}_k{self.ky}"

    def to_dict(self) -> dict:
        return {"mode": self.mode, "ky": self.ky, "kz": self.kz,
                "pattern": self.pattern, "seed": self.seed,
                "ideal_fraction": self.ideal_fraction()}

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingSpec":
        return cls(mode=d.get("mode", "y"), ky=int(d.get("ky", 1)),
                   kz=int(d.get("kz", 1)), pattern=d.get("pattern", "grid"),
                   seed=int(d.get("seed", 0)))


def _lattice(n: int, k: int, phase: int) -> torch.Tensor:
    """Boolean ``[n]``: global index g is measured iff ``(g - phase) % k == 0``."""
    if k == 1:
        return torch.ones(n, dtype=torch.bool)
    return (torch.arange(n) + phase) % k == 0


def _xy_mask(h: int, w: int, k: int, pattern: str, seed: int,
             phase_y: int, phase_x: int) -> torch.Tensor:
    rows = _lattice(h, k, phase_y)
    cols = _lattice(w, k, phase_x)
    if pattern == "grid":
        return rows.view(h, 1) | cols.view(1, w)
    if pattern == "rows":
        return rows.view(h, 1).expand(h, w)
    if pattern == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.rand(h, w, generator=g) < (1.0 / (k * k))
    raise ValueError(f"unknown xy pattern {pattern!r}")


def _make_mask_2d(spec: SamplingSpec, shape: tuple[int, int],
                  phase: tuple[int, ...] = (0, 0)) -> torch.Tensor:
    """Support for one ``[H, W]`` plane. Depth-axis modes are not meaningful."""
    if spec.mode not in ("y", "xy", "full"):
        raise ValueError(
            f"sampling mode {spec.mode!r} needs a depth axis and cannot be applied "
            f"to a 2-D [H, W] image; use 'y', 'xy' or 'full'")
    if int(spec.kz) != 1:
        raise ValueError(f"2-D sampling cannot skip planes (kz={spec.kz}); set kz=1")
    h, w = shape
    _, py, px = _as_phase(phase, 2)
    if spec.mode == "full":
        return torch.ones(shape, dtype=torch.bool)
    if spec.mode == "y":
        return _lattice(h, spec.ky, py).view(h, 1).expand(h, w).clone()
    return _xy_mask(h, w, spec.ky, spec.pattern, spec.seed, py, px)


def _as_phase(phase: tuple[int, ...], ndim: int) -> tuple[int, int, int]:
    """Normalise ``(py, px)`` / ``(pz, py, px)`` to a 3-tuple."""
    phase = tuple(int(p) for p in phase)
    if len(phase) == ndim:
        return (0,) * (3 - len(phase)) + phase if len(phase) < 3 else phase
    if len(phase) == 3:
        return phase
    raise ValueError(f"phase must have {ndim} or 3 entries, got {phase}")


def kept_planes(spec: SamplingSpec, depth: int,
                phase: int = 0) -> torch.Tensor:
    """Global Z indices the ``z`` lattice measures (all of them when kz = 1)."""
    return _lattice(int(depth), int(spec.kz), int(phase))


def kept_rows(spec: SamplingSpec, height: int, phase: int = 0) -> torch.Tensor:
    """Global Y indices the row lattice measures (all of them when ky = 1)."""
    return _lattice(int(height), int(spec.ky), int(phase))


def make_mask(spec: SamplingSpec, shape: tuple[int, ...],
              phase: tuple[int, ...] = (0, 0, 0)) -> torch.Tensor:
    """Build the measurement support for a ``[Z, H, W]`` volume or an ``[H, W]`` plane.

    The 2-D form exists because part of the evidence base is 2-D movies with no
    depth axis (MPM-3P-SQH) and because a single optical plane exported from a
    stack is still one acquisition: it must obey the *same* Y-row lattice, not a
    re-invented one. In 2-D only ``y | xy | full`` make sense -- ``z`` and ``yz``
    are rejected rather than silently degraded to something else.

    phase : ``(pz, py, px)`` (:``(py, px)`` in 2-D) lattice phase in global stack
            coordinates; patch windows cropped at ``(z0, y0, x0)`` should pass
            ``(z0 % kz, y0 % ky, x0 % ky)`` so the lattice is stack-aligned.

    Returns a bool tensor of ``shape``; True = measured.
    """
    if len(shape) == 2:
        return _make_mask_2d(spec, tuple(shape), phase)
    if len(shape) != 3:
        raise ValueError(f"expected [Z, H, W] or [H, W], got {tuple(shape)}")
    d, h, w = shape
    pz, py, px = phase

    if spec.mode == "full":
        return torch.ones(shape, dtype=torch.bool)

    planes = _lattice(d, spec.kz, pz).view(d, 1, 1)

    if spec.mode == "z":
        return planes.expand(d, h, w).clone()

    # ``y`` and ``yz`` keep complete rows (X stays dense, that is what makes
    # line skipping cheap); only the ``xy`` stress test uses the in-plane lattice.
    if spec.mode in ("y", "yz"):
        lateral = _lattice(h, spec.ky, py).view(h, 1).expand(h, w)
    else:
        lateral = _xy_mask(h, w, spec.ky, spec.pattern, spec.seed, py, px)

    if spec.mode in ("y", "xy"):
        return lateral[None].expand(d, h, w).clone()

    # yz
    return (planes & lateral[None]).expand(d, h, w).clone()

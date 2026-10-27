"""Volume geometry and axis provenance.

Every array entering this project carries a :class:`VolumeGeometry` that says
what each axis *is* and what one voxel is worth in micrometres. Nothing is
inferred from array shape: a ``[1340, 1340, 60]`` array is equally consistent
with ``ZYX``, ``TYX`` or a colour axis, and guessing wrong is the single
mistake this project is built to prevent (plan §2.2).

Rules enforced here
-------------------
* ``axes`` must be a permutation of ``ZYX`` (optionally prefixed by ``C``).
  A ``T`` axis disqualifies a volume from the 3-D pipeline: time is not depth.
* ``axes`` must match the array dimensionality, and the ``Z`` length must match
  the declared plane count when one is known.
* Physical voxel sizes must be positive; ``z_step_um`` is recorded separately
  from ``voxel_xy_um`` because MPM stacks are strongly anisotropic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["VolumeGeometry", "GeometryError", "infer_axes_from_ndim"]


class GeometryError(ValueError):
    """Raised when a volume's declared axes or physical scale are unusable."""


@dataclass
class VolumeGeometry:
    """Identity, axes and physical scale of one microscope volume."""

    sample_id: str
    mouse_id: str
    session_id: str
    fov_id: str
    channel: str = "vessels"
    axes: str = "ZYX"
    shape: tuple[int, ...] = ()
    dtype: str = ""
    voxel_xy_um: float = 1.0
    z_step_um: float = 1.0
    excitation_nm: int | None = None
    power_mw: float | None = None
    dwell_time_us: float | None = None
    depth_origin_um: float | None = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    # -- axis semantics -----------------------------------------------------
    @property
    def has_time_axis(self) -> bool:
        return "T" in self.axes.upper()

    @property
    def z_index(self) -> int:
        """Position of the depth axis inside ``axes``."""
        try:
            return self.axes.upper().index("Z")
        except ValueError as exc:  # pragma: no cover - guarded by validate()
            raise GeometryError(f"no Z axis in axes={self.axes!r}") from exc

    @property
    def spatial_axes(self) -> str:
        """``axes`` with any leading channel axis removed."""
        a = self.axes.upper()
        return a[1:] if a.startswith("C") else a

    @property
    def depth(self) -> int:
        return int(self.shape[self.z_index]) if self.shape else 0

    @property
    def depth_span_um(self) -> float:
        """Physical extent covered by the stack (0 for a single plane)."""
        return max(0, self.depth - 1) * self.z_step_um

    def context_depth(self, context_z_um: float | None = None,
                      context_slices: int | None = None) -> int:
        """Number of Z slices to stack so the window spans a fixed *physical* range.

        Plan §3.3 requires comparing acquisition settings at equal usable
        physical context; expressing the window in micrometres rather than
        slices is what keeps that comparison honest when ``z_step_um`` changes.
        Always odd, so the query plane sits in the middle.
        """
        if context_slices is not None:
            n = int(context_slices)
        elif context_z_um is not None:
            n = int(round(context_z_um / self.z_step_um)) + 1
        else:
            n = 1
        n = max(1, n)
        return n if n % 2 == 1 else n + 1

    # -- validation ---------------------------------------------------------
    def validate(self, actual_shape: tuple[int, ...] | None = None,
                 declared_planes: int | None = None) -> list[str]:
        """Return a list of problems; raise on anything disqualifying.

        Non-fatal oddities come back as warnings so the audit can log them
        instead of silently accepting a suspicious volume.
        """
        warnings: list[str] = []
        axes = self.axes.upper()

        if self.has_time_axis:
            raise GeometryError(
                f"{self.sample_id}: axes={self.axes!r} contains a time axis. "
                "3-D structural data must be ZYX; TYX belongs to the calcium "
                "pipeline (Phase 6) and must not be loaded as depth.")
        if sorted(axes.replace("C", "")) != ["X", "Y", "Z"]:
            raise GeometryError(f"{self.sample_id}: axes={self.axes!r} is not a "
                                "permutation of ZYX (with optional leading C)")
        if axes.startswith("C") and len(axes) == 4:
            warnings.append("channel axis present; the loader expects a single "
                            "channel and will fail unless one is selected")

        if self.voxel_xy_um <= 0 or self.z_step_um <= 0:
            raise GeometryError(f"{self.sample_id}: voxel sizes must be > 0, got "
                                f"xy={self.voxel_xy_um} z={self.z_step_um}")

        shape = tuple(actual_shape or self.shape)
        if shape:
            if len(shape) != len(axes):
                raise GeometryError(f"{self.sample_id}: axes={self.axes!r} "
                                    f"({len(axes)} dims) vs array shape {shape}")
            if self.shape and tuple(self.shape) != shape:
                raise GeometryError(f"{self.sample_id}: declared shape "
                                    f"{tuple(self.shape)} != actual {shape}")
            if declared_planes is not None and int(shape[self.z_index]) != declared_planes:
                raise GeometryError(
                    f"{self.sample_id}: declared {declared_planes} planes but Z axis "
                    f"has {shape[self.z_index]} -- axis order or file is wrong")
            if int(shape[self.z_index]) < 2:
                raise GeometryError(f"{self.sample_id}: Z depth "
                                    f"{shape[self.z_index]} is not a 3-D stack")

        if not self.sample_id:
            raise GeometryError("sample_id is required")
        for name in ("mouse_id", "session_id", "fov_id"):
            if not getattr(self, name):
                warnings.append(f"{name} missing: grouped splits and longitudinal "
                                "leakage checks cannot be guaranteed")
        # MPM stacks are always anisotropic (typically ~5x); warn only when the
        # ratio is far outside the usual range, so a routine warning never
        # trains the reader to ignore all warnings.
        if self.z_step_um > 10 * self.voxel_xy_um:
            warnings.append(f"extreme anisotropy (z_step {self.z_step_um} um vs xy "
                            f"{self.voxel_xy_um} um): axial context must span enough "
                            "micrometres for any depth comparison to be fair")
        return warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        d["depth_span_um"] = self.depth_span_um
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VolumeGeometry":
        kw = {k: v for k, v in d.items() if k in AUDIT_FIELDS}
        if "shape" in kw:
            kw["shape"] = tuple(int(v) for v in kw["shape"])
        # ``extra`` is an audit field, so it is already in ``kw``; a CSV round
        # trip turns it into a string, which must not reach the dataclass.
        extra = kw.pop("extra", None)
        if not isinstance(extra, dict):
            extra = {}
        return cls(**kw, extra=extra)


#: Fields the audit checklist (plan §2.2) requires for every volume.
AUDIT_FIELDS = tuple(VolumeGeometry.__dataclass_fields__)


def infer_axes_from_ndim(ndim: int) -> str:
    """Best-effort *label only* for an undeclared array.

    This is deliberately not a guess that can reach a training run: the audit
    records the result as ``axes_unverified`` and refuses to promote it.
    """
    if ndim == 3:
        return "ZYX"
    if ndim == 4:
        return "CZYX"
    if ndim == 2:
        return "YX"
    return "?" * ndim


def check_axis_consistency(shape: tuple[int, ...], axes: str,
                           xy_px: tuple[int, int] | None = None) -> list[str]:
    """Heuristic warnings for a declared axis order.

    A declared ``ZYX`` volume whose "X" axis is implausibly short compared with
    its "Y" axis is usually transposed. Heuristics only warn; they never change
    the declared order.
    """
    out: list[str] = []
    a = axes.upper()
    if len(shape) != len(a):
        return out
    dims = dict(zip(a, shape))
    if "Y" in dims and "X" in dims and min(dims["Y"], dims["X"]) > 0:
        ratio = max(dims["Y"], dims["X"]) / min(dims["Y"], dims["X"])
        if ratio > 2.0:
            out.append(f"axis lengths Y={dims['Y']} X={dims['X']} differ by "
                       f"{ratio:.1f}x -- suspect transposed/tiled acquisition")
    if "Z" in dims and dims["Z"] > max(dims.get("Y", 0), dims.get("X", 0)):
        out.append(f"Z depth ({dims['Z']}) exceeds a lateral extent -- verify "
                   "that the first axis is really depth and not time")
    return out


def stack_patch(arr: np.ndarray, z0: int, y0: int, x0: int,
                depth: int, height: int, width: int) -> np.ndarray:
    """Read a ``[depth, height, width]`` sub-volume from a ZYX array."""
    return np.asarray(arr[z0:z0 + depth, y0:y0 + height, x0:x0 + width],
                      dtype=np.float32)

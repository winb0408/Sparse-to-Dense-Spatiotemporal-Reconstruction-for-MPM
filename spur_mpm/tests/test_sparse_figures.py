"""The sparse figures must be drawn on a plane that was actually measured.

A Z-sparse panel rendered on a skipped plane is entirely cyan: it reads as
"this protocol measures nothing here", which is a display artefact, not a
property of the protocol.  These tests pin the plane choice and the refusal to
guess when the levels share no plane.
"""

from __future__ import annotations

import numpy as np
import pytest

from spur_mpm.data.sparse import measured_mask
from spur_mpm.physics.sampling import SamplingSpec
from spur_mpm.scripts.sparse_figures import _common_plane


def _entry(tmp_path, level, spec, shape, phase=(0, 0, 0)):
    m = measured_mask(shape, spec, phase=phase)
    p = tmp_path / f"{level}.npz"
    np.savez_compressed(p, packed=np.packbits(m.reshape(-1)),
                        shape=np.array(shape, dtype=np.int64))
    return {"level": level, "sample_id": "s", "spec": spec.to_dict(),
            "phase": list(phase), "mask_path": str(p)}


def _spec(mode, ky=1, kz=1):
    return SamplingSpec(mode=mode, ky=ky, kz=kz, pattern="grid", seed=0)


def test_common_plane_is_measured_by_every_level(tmp_path):
    shape = (9, 8, 8)
    ent = {"z_k2": _entry(tmp_path, "z_k2", _spec("z", kz=2), shape),
           "z_k4": _entry(tmp_path, "z_k4", _spec("z", kz=4), shape)}
    z = _common_plane(ent, shape)
    assert z == 4                                   # centre, and 4 % 4 == 0
    for e in ent.values():
        m = measured_mask(shape, SamplingSpec.from_dict(e["spec"]),
                          phase=tuple(e["phase"]))
        assert m[z].any()


def test_common_plane_moves_off_the_centre_when_needed(tmp_path):
    shape = (10, 8, 8)                              # centre 5 is skipped by k=2
    ent = {"z_k2": _entry(tmp_path, "z_k2", _spec("z", kz=2), shape),
           "z_k4": _entry(tmp_path, "z_k4", _spec("z", kz=4), shape)}
    z = _common_plane(ent, shape)
    assert z != shape[0] // 2
    assert z % 4 == 0                              # measured by the coarsest level
    assert z == 4                                  # ... and the nearest such plane


def test_common_plane_refuses_when_levels_share_nothing(tmp_path):
    shape = (8, 8, 8)
    ent = {"z_k2_p0": _entry(tmp_path, "z_k2_p0", _spec("z", kz=2), shape, (0, 0, 0)),
           "z_k2_p1": _entry(tmp_path, "z_k2_p1", _spec("z", kz=2), shape, (1, 0, 0))}
    with pytest.raises(RuntimeError, match="no single plane"):
        _common_plane(ent, shape)


def test_y_only_levels_keep_every_plane(tmp_path):
    shape = (7, 8, 8)
    ent = {"y_k4": _entry(tmp_path, "y_k4", _spec("y", ky=4), shape)}
    assert _common_plane(ent, shape) == shape[0] // 2

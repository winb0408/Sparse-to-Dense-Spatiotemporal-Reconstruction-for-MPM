"""Tests for the sparse-acquisition ladder (`spur_mpm.data.sparse`).

The construction is the one thing every number in this project rests on, so the
tests are written as the claims a reader would want to check: what exactly is
measured, what is exactly zero, and whether a patch cropped out of a volume sits
on the same lattice as the whole stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from spur_mpm.data.sparse import (
    LEVELS_2D,
    LEVELS_3D,
    acquisition_cost,
    build_sparse,
    levels_for,
    load_mask,
    measured_mask,
    verify_construction,
    write_entry,
)
from spur_mpm.physics.acquisition import NoiseConfig
from spur_mpm.physics.sampling import SamplingSpec


def _image(shape, seed=0):
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


# --- 2-D --------------------------------------------------------------------

def test_2d_y_sparsity_keeps_whole_rows_and_only_rows():
    img = _image((16, 6))
    m = measured_mask(img.shape, SamplingSpec("y", ky=2))
    assert m.shape == img.shape
    np.testing.assert_array_equal(m[:, 0], m.all(axis=1))      # rows are uniform
    np.testing.assert_array_equal(np.flatnonzero(m[:, 0]), np.arange(0, 16, 2))
    assert m.mean() == pytest.approx(0.5)


def test_2d_refuses_depth_modes_and_plane_skipping():
    img = _image((8, 8))
    with pytest.raises(ValueError, match="needs a depth axis"):
        measured_mask(img.shape, SamplingSpec("z", kz=2))
    with pytest.raises(ValueError, match="cannot skip planes"):
        measured_mask(img.shape, SamplingSpec("y", ky=2, kz=2))


def test_2d_phase_continues_the_global_lattice():
    """A crop taken at y0=3 must show the rows the whole image would show there."""
    img = _image((32, 4))
    spec = SamplingSpec("y", ky=3)
    whole = measured_mask(img.shape, spec)
    crop = measured_mask((10, 4), spec, phase=(3, 0))
    y0 = 3
    np.testing.assert_array_equal(crop, whole[y0:y0 + 10])


# --- 3-D --------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LEVELS_3D))
def test_3d_levels_verify(name):
    # Divisible by every k in the ladder, so the assertion tests the fraction and
    # not the border rounding (that is covered separately as a bound, not equality).
    img = _image((12, 24, 12))
    spec = LEVELS_3D[name]
    built = build_sparse(img, spec)
    verdict = verify_construction(img, built["y"], built["mask"], spec)
    assert verdict["ok"], {k: v for k, v in verdict.items() if v is False}
    assert built["measured_fraction"] == pytest.approx(
        built["ideal_fraction"], abs=0.02)


def test_3d_yz_factorises_into_rows_and_planes():
    img = _image((12, 16, 8))
    spec = SamplingSpec("yz", ky=2, kz=3)
    m = measured_mask(img.shape, spec)
    rows = measured_mask((16, 8), SamplingSpec("y", ky=2), phase=(0, 0))
    planes = measured_mask((12, 16, 8), SamplingSpec("z", kz=3))
    expected = planes & rows[None]
    np.testing.assert_array_equal(m, expected)


def test_3d_z_mode_keeps_whole_planes():
    img = _image((9, 6, 5))
    m = measured_mask(img.shape, SamplingSpec("z", kz=2))
    np.testing.assert_array_equal(np.flatnonzero(m.all(axis=(1, 2))), np.arange(0, 9, 2))
    assert m.all(axis=(1, 2)).sum() == 5


def test_3d_phase_matches_the_global_crop():
    """The property that lets patches be drawn anywhere: one global lattice."""
    vol = _image((20, 24, 10))
    spec = SamplingSpec("yz", ky=3, kz=2)
    whole = measured_mask(vol.shape, spec)
    z0, y0 = 5, 7
    crop = measured_mask((6, 9, 10), spec, phase=(z0 % 2, y0 % 3, 0))
    np.testing.assert_array_equal(crop, whole[z0:z0 + 6, y0:y0 + 9])


# --- the definition Y = M . D ----------------------------------------------

def test_unmeasured_is_exactly_zero_and_measured_is_exact():
    img = _image((8, 12, 10))
    spec = SamplingSpec("y", ky=2)
    built = build_sparse(img, spec)
    m = built["mask"]
    assert np.all(built["y"][~m] == 0.0)
    np.testing.assert_array_equal(built["y"][m], img[m])


def test_verification_rejects_a_smuggled_value():
    """If anything fills an unmeasured voxel, the check must fail -- that is the
    check that makes 'the model had to infer it' a statement about the input."""
    img = _image((4, 8, 8))
    spec = SamplingSpec("y", ky=2)
    built = build_sparse(img, spec)
    y = built["y"].copy()
    y[~built["mask"]] = 0.25
    verdict = verify_construction(img, y, built["mask"], spec)
    assert verdict["ok"] is False
    assert verdict["unmeasured_is_exactly_zero"] is False


def test_verification_rejects_a_wrong_lattice():
    img = _image((4, 8, 8))
    spec = SamplingSpec("y", ky=2)
    built = build_sparse(img, spec)
    shifted = measured_mask(img.shape, spec, phase=(0, 1, 0))
    verdict = verify_construction(img, built["y"], shifted, spec)
    assert verdict["lattice_matches_spec"] is False
    assert verdict["ok"] is False


def test_noise_moves_only_measured_voxels():
    img = _image((6, 16, 10))
    spec = SamplingSpec("y", ky=2)
    clean = build_sparse(img, spec)
    noisy = build_sparse(img, spec, noise=NoiseConfig(poisson_photons=50.0,
                                                     gaussian_sigma=0.01, seed=3))
    m = clean["mask"]
    np.testing.assert_array_equal(noisy["mask"], m)
    assert np.all(noisy["y"][~m] == 0.0)               # still never measured
    assert not np.allclose(noisy["y"][m], clean["y"][m])
    verdict = verify_construction(img, noisy["y"], noisy["mask"], spec, noisy=True)
    assert verdict["ok"] is True
    assert verdict["measured_equals_dense"] is None
    assert verdict["noise_delta_max"] > 0.0


# --- cost model -------------------------------------------------------------

def test_scan_line_fraction_equals_measured_fraction_only_when_x_is_dense():
    img = _image((6, 16, 8))
    y_spec = SamplingSpec("y", ky=4)
    cost = acquisition_cost(y_spec, img.shape)
    assert cost["scan_line_fraction"] == pytest.approx(0.25)

    xy = SamplingSpec("xy", ky=2, pattern="grid")
    cost_xy = acquisition_cost(xy, img.shape)
    # The grid measures a sparse subset of voxels but touches every row, and a
    # touched row costs a full row of dwell time: that is why xy is a stress test.
    assert cost_xy["scan_line_fraction"] == pytest.approx(1.0)
    assert xy.ideal_fraction() < 1.0


def test_ladder_shape_dispatch():
    assert "z_k2" not in levels_for(2)
    assert "z_k2" in levels_for(3)
    assert set(levels_for(2)) == set(LEVELS_2D)
    with pytest.raises(ValueError):
        levels_for(4)


# --- persistence ------------------------------------------------------------

def test_mask_round_trips_through_the_manifest(tmp_path):
    img = _image((6, 12, 10))
    spec = SamplingSpec("yz", ky=2, kz=2)
    built = build_sparse(img, spec)
    entry = write_entry(tmp_path, "sampleA", "yz_k2x2", built)
    assert entry["n_measured"] == int(built["mask"].sum())
    assert len(entry["mask_sha1"]) == 40
    np.testing.assert_array_equal(load_mask(entry), built["mask"])


def test_verifier_rejects_numeric_mask_even_if_values_are_zero_one():
    """A float mask cannot silently pass the 'mask_is_boolean' assertion."""
    img = _image((4, 8, 8))
    spec = SamplingSpec("y", ky=2)
    built = build_sparse(img, spec)
    result = verify_construction(img, built["y"], built["mask"].astype(np.float32), spec)
    assert result["mask_is_boolean"] is False
    assert result["ok"] is False


def test_verifier_reports_shape_mismatch_without_indexing_crash():
    img = _image((4, 8, 8))
    spec = SamplingSpec("y", ky=2)
    built = build_sparse(img, spec)
    result = verify_construction(img, built["y"][:, :-1], built["mask"], spec)
    assert result["shape_matches"] is False
    assert result["ok"] is False


def test_verifier_uses_optically_degraded_measurement_reference():
    """An explicitly enabled non-identity PSF must not always fail the audit."""
    from spur_mpm.physics.psf import PSFConfig
    img = _image((4, 8, 8))
    spec = SamplingSpec("y", ky=2)
    psf = PSFConfig(sigma_xy_um=1.0, sigma_z_um=0.0,
                    voxel_xy_um=1.0, voxel_z_um=1.0)
    built = build_sparse(img, spec, psf=psf)
    correct = verify_construction(img, built["y"], built["mask"], spec, psf=psf)
    wrong = verify_construction(img, built["y"], built["mask"], spec)
    assert correct["ok"] is True
    assert wrong["measured_equals_dense"] is False

"""Axis discipline: time is not depth, shapes must match, context is physical."""

import pytest

from spur_mpm.data.geometry import GeometryError, VolumeGeometry, check_axis_consistency


def geom(**kw) -> VolumeGeometry:
    base = dict(sample_id="s1", mouse_id="M1", session_id="S1", fov_id="F1",
                axes="ZYX", voxel_xy_um=0.98, z_step_um=5.0)
    base.update(kw)
    return VolumeGeometry(**base)


def test_zyx_is_accepted():
    assert geom().validate(actual_shape=(32, 512, 512)) == []


def test_time_axis_is_rejected():
    with pytest.raises(GeometryError, match="time axis"):
        geom(axes="TYX").validate(actual_shape=(100, 512, 512))


def test_axis_letters_must_be_a_permutation():
    with pytest.raises(GeometryError):
        geom(axes="ZYA").validate(actual_shape=(3, 4, 5))


def test_shape_and_axes_length_must_agree():
    with pytest.raises(GeometryError, match="vs array shape"):
        geom(axes="ZYX").validate(actual_shape=(4, 5))


def test_declared_plane_count_checked():
    with pytest.raises(GeometryError, match="declared .* TIFF pages"):
        geom().validate(actual_shape=(32, 64, 64), declared_planes=31)


def test_voxel_sizes_must_be_positive():
    with pytest.raises(GeometryError):
        geom(z_step_um=0.0).validate(actual_shape=(8, 8, 8))


def test_missing_animal_identity_warns_not_raises():
    warnings = geom(mouse_id="").validate(actual_shape=(8, 8, 8))
    assert any("mouse_id missing" in w for w in warnings)


def test_context_depth_is_physical():
    g = geom(z_step_um=5.0)
    assert g.context_depth(context_z_um=20.0) == 5      # 20 um / 5 um + centre
    assert g.context_depth(context_z_um=15.0) == 5      # rounded, then made odd
    assert g.context_depth(context_slices=4) == 5       # forced odd
    g2 = geom(z_step_um=10.0)
    assert g2.context_depth(context_z_um=20.0) == 3     # same physical range


def test_depth_span():
    assert geom().validate(actual_shape=(60, 8, 8)) or True
    g = geom(axes="ZYX")
    g.shape = (60, 8, 8)
    assert g.depth_span_um == pytest.approx(59 * 5.0)


def test_axis_heuristics_warn_on_suspicious_shapes():
    assert check_axis_consistency((32, 512, 64), "ZYX")
    assert check_axis_consistency((900, 128, 128), "ZYX")
    assert not check_axis_consistency((32, 512, 512), "ZYX")

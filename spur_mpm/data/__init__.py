"""Data layer: geometry audit, manifest, calibration, grouped splits, loaders."""

from spur_mpm.data.geometry import GeometryError, VolumeGeometry
from spur_mpm.data.normalize import IntensityCalibration
from spur_mpm.data.observation import FEATURES, ModelInputs, build_model_inputs
from spur_mpm.data.split import SplitConfig, SplitError, make_folds, write_split

__all__ = [
    "VolumeGeometry", "GeometryError",
    "IntensityCalibration",
    "FEATURES", "ModelInputs", "build_model_inputs",
    "SplitConfig", "SplitError", "make_folds", "write_split",
]

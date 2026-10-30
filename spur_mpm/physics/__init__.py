"""Forward model: optics, sampling, acquisition and the physics warm-start."""

from spur_mpm.physics.acquisition import (
    AcquisitionConfig,
    NoiseConfig,
    Observation,
    apply_otf,
    observe,
)
from spur_mpm.physics.psf import PSFConfig, gaussian_otf_2d, gaussian_otf_3d
from spur_mpm.physics.sampling import SamplingSpec, kept_planes, kept_rows, make_mask
from spur_mpm.physics.wiener import WARM_START_MODES, warm_start, wiener_deconv_3d

__all__ = [
    "AcquisitionConfig", "NoiseConfig", "Observation", "apply_otf", "observe",
    "PSFConfig", "gaussian_otf_2d", "gaussian_otf_3d",
    "SamplingSpec", "make_mask", "kept_rows", "kept_planes",
    "WARM_START_MODES", "warm_start", "wiener_deconv_3d",
]

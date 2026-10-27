"""SPUR-MPM: self-supervised physics-guided sparse reconstruction for MPM.

Package layout
--------------
common/    config loading, seeding, run logging, provenance hashes
physics/   forward model: PSF/OTF, sampling masks, acquisition, Wiener warm-start
data/      axis/geometry audit, manifest, normalization, grouped split, datasets
ssl/       measurement holdout split, MMS loss, leakage guards
models/    lightweight Swin-UNet, prior fusion stem, uncertainty fields
losses/    reconstruction + data-consistency objectives
metrics/   image, vessel-structure and grouped statistics
scripts/   data audit / split / synthetic-smoke entry points
train.py evaluate.py inference.py

Axis discipline: volumes are ``ZYX`` 3-D spatial stacks throughout. ``Z`` is
physical depth, never time. Time-series (``TYX``) calcium data belongs to
Phase 6 and must not be loaded by this package's 3-D loader.
"""

__version__ = "0.1.0"

#: Reference label used by the old PSRAS codebase this package derives from.
ATTRIBUTION = (
    "Portions adapted from winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-"
    "for-MPM, branch review-v4-20261019 (PSRAS/). Adaptations: ZYX axis semantics, "
    "physical-unit PSF, holdout self-supervision, new data/SSL layers."
)

# Effective PSF estimate (no calibration exists for this deposit)

- fold 0, sampling `y` k=2x1, 6 patches of 32x256x256 per volume, lambda 0.01
- training volumes only: M1_FOVII_S1_vessels, M1_FOVII_S2_vessels, M3_FOVI_S1_vessels, M3_FOVI_S2_vessels

## 1. Arithmetic from assumed optics

The deposit ships no NA (README: "imaging parameters are described in the manuscript Methods"). Every row below is therefore an **assumption**, printed so that none of them can be quietly picked.

| NA | sigma_xy (um) | sigma_z (um) | sigma_xy (voxels) | sigma_z (voxels) |
|---|---|---|---|---|
| 0.80 | 0.209 | 1.081 | 0.214 | 0.216 |
| 1.00 | 0.168 | 0.638 | 0.171 | 0.128 |
| 1.05 | 0.160 | 0.563 | 0.163 | 0.113 |

A PSF narrower than one voxel cannot be recovered from sampled data, and on this grid both columns land below 1 -- the sampling, not the optics, is what limits these stacks.

## 2. Width that best predicts the withheld rows

`delta` is held-out PSNR minus the interpolation baseline (sigma = 0).
`changed` is the fraction of voxels the deconvolution actually moved: a
row with `changed ~ 0` was skipped, not tested.

### lateral (sigma_z = 0)

| sigma_xy (um) | px | PSNR mean | delta PSNR | changed |
|---|---|---|---|---|
| 0.00 | 0.000 | 26.0015 | +0.0000 | 0.0000 |
| 0.20 | 0.204 | 25.8353 | -0.1662 | 0.9997 |
| 0.40 | 0.408 | 25.0176 | -0.9839 | 0.9999 |
| 0.60 | 0.612 | 22.5190 | -3.4825 | 1.0000 |
| 0.80 | 0.816 | 20.2771 | -5.7245 | 1.0000 |
| 1.00 | 1.020 | 20.3270 | -5.6745 | 1.0000 |
| 1.50 | 1.531 | 20.3939 | -5.6076 | 1.0000 |
| 2.00 | 2.041 | 21.1921 | -4.8095 | 1.0000 |
| 3.00 | 3.061 | 22.1936 | -3.8079 | 1.0000 |

### axial (sigma_xy = 0)

| sigma_z (um) | px | PSNR mean | delta PSNR | changed |
|---|---|---|---|---|
| 0.00 | 0.000 | 26.0015 | +0.0000 | 0.0000 |
| 1.00 | 0.200 | 25.8289 | -0.1726 | 0.9997 |
| 2.00 | 0.400 | 24.9383 | -1.0632 | 0.9999 |
| 3.00 | 0.600 | 22.4205 | -3.5810 | 1.0000 |
| 5.00 | 1.000 | 20.4884 | -5.5131 | 1.0000 |
| 8.00 | 1.600 | 21.4448 | -4.5568 | 1.0000 |

## Reading

- best lateral width: sigma_xy = 0.00 um (+0.0000 dB vs interpolation)
- best axial width: sigma_z = 0.00 um (+0.0000 dB vs interpolation)
- A best width at 0.00 means the withheld rows are predicted best by *not* deconvolving. That is an estimate of identity, not a failure of the search.
- Nothing here is a calibration. Any run using these numbers must set `physics.psf_source=estimated_from_data`; refusing to do so is the point.

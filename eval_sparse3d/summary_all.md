# Sparse 3D Reconstruction — Unified Evaluation

**GT**: `Validation/F0.tif` | max_frames=200

| # | Run | Algo | Sparse | Extras | Frames | PSNR (dB) | SSIM | Time |
|---|-----|------|--------|--------|--------|-----------|------|------|
| 1 | baseline_input_z | input | z |  | 200 | 74.14±8.29 | 1.0000±0.0000 | 0.0s |
| 2 | sparse3d_2p5d_z_K4 | 2p5d | z |  | 200 | 41.47±1.29 | 0.9739±0.0120 | 22.6s |
| 3 | sparse3d_3d_xy_K4_wiener | 3d | xy | wiener | 200 | 38.50±2.49 | 0.9169±0.0309 | 10.4s |
| 4 | baseline_input_xy | input | xy |  | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 5 | baseline_input_xyz | input | xyz |  | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 6 | sparse3d_2d_z_K4 | 2d | z |  | 200 | 24.30±1.71 | 0.6287±0.0924 | 21.1s |
| 7 | sparse3d_2d_xy_K4 | 2d | xy |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 25.4s |
| 8 | sparse3d_2d_xyz_K4 | 2d | xyz |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 20.8s |
| 9 | sparse3d_3d_xy_K4 | 3d | xy |  | 200 | 21.19±1.50 | 0.5251±0.0986 | 9.2s |
| 10 | sparse3d_3d_xy_K4_psf_loss | 3d | xy | psf_loss | 200 | 21.18±1.50 | 0.5250±0.0986 | 9.2s |
| 11 | sparse3d_3d_xyz_K4 | 3d | xyz |  | 200 | 21.18±1.50 | 0.5250±0.0986 | 9.5s |
| 12 | sparse3d_3d_z_K4 | 3d | z |  | 200 | 21.18±1.50 | 0.5249±0.0986 | 8.1s |
| 13 | sparse3d_3d_xy_K4_psf_ch | 3d | xy | psf_ch | 200 | 21.00±1.49 | 0.5199±0.0987 | 8.9s |
| 14 | sparse3d_3d_xy_K4_psf_full | 3d | xy | psf_full | 200 | 21.00±1.49 | 0.5197±0.0987 | 7.3s |
| 15 | sparse3d_2p5d_xyz_K4 | 2p5d | xyz |  | 200 | 19.55±0.91 | 0.5528±0.0634 | 24.4s |
| 16 | sparse3d_2p5d_xy_K4 | 2p5d | xy |  | 200 | 18.99±0.45 | 0.5454±0.0509 | 22.5s |

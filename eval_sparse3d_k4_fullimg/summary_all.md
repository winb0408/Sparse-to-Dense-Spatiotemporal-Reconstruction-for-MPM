# Sparse 3D Reconstruction — Unified Evaluation

**GT**: `Validation/F0.tif` | max_frames=200

| # | Run | Algo | Sparse | Extras | Frames | PSNR (dB) | SSIM | Time |
|---|-----|------|--------|--------|--------|-----------|------|------|
| 1 | baseline_input_z_K4 | input | z | K4 | 200 | 74.14±8.29 | 1.0000±0.0000 | 0.0s |
| 2 | sparse3d_2p5d_z_K4 | 2p5d | z |  | 200 | 50.10±2.46 | 0.9912±0.0075 | 19.7s |
| 3 | sparse3d_3d_xy_K4_wiener | 3d | xy | wiener | 200 | 38.50±2.49 | 0.9169±0.0309 | 66.6s |
| 4 | sparse3d_2p5d_xyz_K4 | 2p5d | xyz |  | 200 | 31.29±1.72 | 0.8095±0.0324 | 16.1s |
| 5 | sparse3d_2p5d_xy_K4 | 2p5d | xy |  | 200 | 26.24±1.75 | 0.7160±0.0264 | 16.9s |
| 6 | baseline_input_xy_K4 | input | xy | K4 | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 7 | baseline_input_xyz_K4 | input | xyz | K4 | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 8 | sparse3d_2d_z_K4 | 2d | z |  | 200 | 24.30±1.71 | 0.6287±0.0924 | 12.7s |
| 9 | sparse3d_2d_xy_K4 | 2d | xy |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 13.5s |
| 10 | sparse3d_2d_xyz_K4 | 2d | xyz |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 12.8s |
| 11 | sparse3d_3d_xy_K4 | 3d | xy |  | 200 | 21.19±1.50 | 0.5251±0.0986 | 20.8s |
| 12 | sparse3d_3d_xyz_K4 | 3d | xyz |  | 200 | 21.18±1.50 | 0.5250±0.0986 | 9.9s |
| 13 | sparse3d_3d_z_K4 | 3d | z |  | 200 | 21.18±1.50 | 0.5249±0.0986 | 11.4s |

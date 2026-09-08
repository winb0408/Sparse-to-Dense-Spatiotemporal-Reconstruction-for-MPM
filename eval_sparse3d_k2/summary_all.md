# Sparse 3D Reconstruction — Unified Evaluation

**GT**: `Validation/F0.tif` | max_frames=200

| # | Run | Algo | Sparse | Extras | Frames | PSNR (dB) | SSIM | Time |
|---|-----|------|--------|--------|--------|-----------|------|------|
| 1 | baseline_input_z_K4 | input | z | K4 | 200 | 74.14±8.29 | 1.0000±0.0000 | 0.0s |
| 2 | baseline_input_z_K2 | input | z | K2 | 200 | 73.86±7.71 | 1.0000±0.0000 | 0.0s |
| 3 | sparse3d_3d_z_K2 | 3d | z |  | 200 | 44.43±2.77 | 0.9763±0.0127 | 14.3s |
| 4 | sparse3d_2p5d_z_K2 | 2p5d | z |  | 198 | 43.96±0.70 | 0.9841±0.0020 | 20.6s |
| 5 | sparse3d_3d_z_K2_wiener | 3d | z | wiener | 200 | 42.26±4.66 | 0.9476±0.0570 | 58.6s |
| 6 | sparse3d_2p5d_z_K4 | 2p5d | z |  | 200 | 41.47±1.29 | 0.9739±0.0120 | 20.8s |
| 7 | sparse3d_3d_xy_K2 | 3d | xy |  | 200 | 40.64±1.65 | 0.9459±0.0112 | 13.0s |
| 8 | sparse3d_3d_xy_K2_wiener | 3d | xy | wiener | 200 | 40.61±2.37 | 0.9525±0.0143 | 21.5s |
| 9 | sparse3d_3d_xyz_K2_wiener | 3d | xyz | wiener | 200 | 39.06±2.87 | 0.9249±0.0388 | 21.0s |
| 10 | sparse3d_3d_xy_K4_wiener | 3d | xy | wiener | 200 | 38.50±2.49 | 0.9169±0.0309 | 21.9s |
| 11 | sparse3d_3d_xyz_K2 | 3d | xyz |  | 200 | 36.18±1.19 | 0.8778±0.0189 | 13.0s |
| 12 | sparse3d_2d_z_K2 | 2d | z |  | 200 | 32.41±0.38 | 0.9359±0.0073 | 21.3s |
| 13 | sparse3d_2p5d_xy_K2_mask | 2p5d | xy | mask | 198 | 29.41±0.26 | 0.8720±0.0140 | 25.0s |
| 14 | sparse3d_2p5d_xy_K2_mask_wiener | 2p5d | xy | mask_wiener | 198 | 29.01±0.15 | 0.8688±0.0103 | 32.0s |
| 15 | baseline_input_xy_K2 | input | xy | K2 | 200 | 26.10±3.48 | 0.4210±0.0339 | 0.0s |
| 16 | baseline_input_xyz_K2 | input | xyz | K2 | 200 | 26.10±3.48 | 0.4209±0.0339 | 0.0s |
| 17 | sparse3d_2p5d_xyz_K2 | 2p5d | xyz |  | 198 | 25.22±0.74 | 0.7385±0.0498 | 19.9s |
| 18 | baseline_input_xy_K4 | input | xy | K4 | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 19 | baseline_input_xyz_K4 | input | xyz | K4 | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 20 | sparse3d_2d_z_K4 | 2d | z |  | 200 | 24.30±1.71 | 0.6287±0.0924 | 19.3s |
| 21 | sparse3d_2p5d_xy_K2 | 2p5d | xy |  | 198 | 23.29±1.82 | 0.7195±0.0246 | 23.6s |
| 22 | sparse3d_2d_xy_K4 | 2d | xy |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 20.0s |
| 23 | sparse3d_2d_xyz_K4 | 2d | xyz |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 19.4s |
| 24 | sparse3d_3d_xy_K4 | 3d | xy |  | 200 | 21.19±1.50 | 0.5251±0.0986 | 12.8s |
| 25 | sparse3d_3d_xyz_K4 | 3d | xyz |  | 200 | 21.18±1.50 | 0.5250±0.0986 | 12.9s |
| 26 | sparse3d_3d_z_K4 | 3d | z |  | 200 | 21.18±1.50 | 0.5249±0.0986 | 13.3s |
| 27 | sparse3d_2d_xy_K2_wiener | 2d | xy | wiener | 200 | 21.02±1.96 | 0.6683±0.0081 | 25.8s |
| 28 | sparse3d_2p5d_xyz_K4 | 2p5d | xyz |  | 200 | 19.55±0.91 | 0.5528±0.0634 | 20.8s |
| 29 | sparse3d_2p5d_xy_K4 | 2p5d | xy |  | 200 | 18.99±0.45 | 0.5454±0.0509 | 20.4s |
| 30 | sparse3d_2d_xyz_K2 | 2d | xyz |  | 200 | 18.42±0.69 | 0.6194±0.0259 | 19.2s |
| 31 | sparse3d_2d_xy_K2 | 2d | xy |  | 200 | 16.85±0.90 | 0.5083±0.0287 | 21.1s |

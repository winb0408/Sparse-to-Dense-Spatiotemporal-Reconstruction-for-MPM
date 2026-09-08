# Sparse 3D Reconstruction — Unified Evaluation

**GT**: `Validation/F0.tif` | max_frames=200

| # | Run | Algo | Sparse | Extras | Frames | PSNR (dB) | SSIM | Time |
|---|-----|------|--------|--------|--------|-----------|------|------|
| 1 | baseline_input_z_K4 | input | z | K4 | 200 | 74.14±8.29 | 1.0000±0.0000 | 0.0s |
| 2 | baseline_input_z_K2 | input | z | K2 | 200 | 73.86±7.71 | 1.0000±0.0000 | 0.0s |
| 3 | sparse3d_2p5d_z_K2 | 2p5d | z |  | 198 | 51.72±1.18 | 0.9942±0.0019 | 9.5s |
| 4 | sparse3d_2p5d_z_K4 | 2p5d | z |  | 200 | 50.08±2.44 | 0.9912±0.0074 | 8.1s |
| 5 | sparse3d_3d_z_K2 | 3d | z |  | 200 | 44.43±2.77 | 0.9763±0.0127 | 5.9s |
| 6 | sparse3d_2d_z_K2 | 2d | z |  | 200 | 42.59±1.61 | 0.9621±0.0080 | 8.4s |
| 7 | sparse3d_3d_z_K2_wiener | 3d | z | wiener | 200 | 42.26±4.66 | 0.9476±0.0570 | 14.0s |
| 8 | sparse3d_3d_xy_K2 | 3d | xy |  | 200 | 40.64±1.65 | 0.9459±0.0112 | 8.6s |
| 9 | sparse3d_3d_xy_K2_wiener | 3d | xy | wiener | 200 | 40.61±2.37 | 0.9525±0.0143 | 16.0s |
| 10 | sparse3d_3d_xyz_K2_wiener | 3d | xyz | wiener | 200 | 39.06±2.87 | 0.9249±0.0388 | 14.4s |
| 11 | sparse3d_3d_xy_K4_wiener | 3d | xy | wiener | 200 | 38.50±2.49 | 0.9169±0.0309 | 15.0s |
| 12 | sparse3d_2p5d_xy_K2_mask_wiener | 2p5d | xy | mask_wiener | 198 | 37.82±1.53 | 0.9229±0.0123 | 15.7s |
| 13 | sparse3d_2p5d_xy_K2_mask | 2p5d | xy | mask | 198 | 37.80±1.35 | 0.9152±0.0139 | 10.8s |
| 14 | sparse3d_3d_xyz_K2 | 3d | xyz |  | 200 | 36.18±1.19 | 0.8778±0.0189 | 5.8s |
| 15 | sparse3d_2p5d_xyz_K4 | 2p5d | xyz |  | 200 | 31.29±1.77 | 0.8101±0.0328 | 7.9s |
| 16 | sparse3d_2d_xyz_K2 | 2d | xyz |  | 200 | 30.72±3.38 | 0.8463±0.0366 | 7.3s |
| 17 | sparse3d_2p5d_xyz_K2 | 2p5d | xyz |  | 198 | 30.30±2.72 | 0.8574±0.0235 | 8.1s |
| 18 | sparse3d_2p5d_xy_K2 | 2p5d | xy |  | 198 | 27.96±3.14 | 0.8267±0.0270 | 7.8s |
| 19 | sparse3d_2d_xy_K2 | 2d | xy |  | 200 | 27.90±3.41 | 0.8176±0.0412 | 15.4s |
| 20 | sparse3d_2p5d_xy_K4 | 2p5d | xy |  | 200 | 26.32±1.88 | 0.7180±0.0288 | 12.6s |
| 21 | baseline_input_xy_K2 | input | xy | K2 | 200 | 26.10±3.48 | 0.4210±0.0339 | 0.0s |
| 22 | baseline_input_xyz_K2 | input | xyz | K2 | 200 | 26.10±3.48 | 0.4209±0.0339 | 0.0s |
| 23 | sparse3d_2d_xy_K2_wiener | 2d | xy | wiener | 200 | 25.92±2.62 | 0.8064±0.0219 | 11.2s |
| 24 | baseline_input_xy_K4 | input | xy | K4 | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 25 | baseline_input_xyz_K4 | input | xyz | K4 | 200 | 25.16±3.48 | 0.1898±0.0451 | 0.0s |
| 26 | sparse3d_2d_z_K4 | 2d | z |  | 200 | 24.30±1.71 | 0.6287±0.0924 | 9.8s |
| 27 | sparse3d_2d_xy_K4 | 2d | xy |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 7.5s |
| 28 | sparse3d_2d_xyz_K4 | 2d | xyz |  | 200 | 22.18±1.74 | 0.5721±0.0994 | 6.8s |
| 29 | sparse3d_3d_xy_K4 | 3d | xy |  | 200 | 21.19±1.50 | 0.5251±0.0986 | 8.2s |
| 30 | sparse3d_3d_xyz_K4 | 3d | xyz |  | 200 | 21.18±1.50 | 0.5250±0.0986 | 5.8s |
| 31 | sparse3d_3d_z_K4 | 3d | z |  | 200 | 21.18±1.50 | 0.5249±0.0986 | 5.8s |

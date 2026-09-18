# EXP-020 quantitative comparison — 2p5d_xyz

reval (per-item PSNR/SSIM over 8 val items, best.pt). gain = b3 − b1 (Wiener x0 vs no warm start).
Single-seed (seed 0) protocol throughout, matching the external baselines; EXP-017 severity-matrix two-seed means (e.g. ours z b3 = 44.73) differ slightly from these seed-0 values.

| Arch | b1 (y+mask) | b3 (Wiener x0) | gain (b3−b1) | b4 (fusion) |
|---|---|---|---|---|
| Ours swin2d | 31.02 / 0.8763 | 31.60 / 0.8862 | **+0.58** | 31.88 / 0.8860 |
| Video Swin (official) | 29.46 / 0.8400 | 24.41 / 0.7963 | **-5.05** | — |
| SwinUNETR (official) | 30.83 / 0.8715 | 32.03 / 0.8876 | **+1.20** | — |
| RVRT (official) | 30.69 / 0.8623 | 14.67 / 0.1888 | collapse (excluded) | — |

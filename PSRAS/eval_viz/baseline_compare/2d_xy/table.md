# EXP-020 quantitative comparison — 2d_xy

reval (per-item PSNR/SSIM over 8 val items, best.pt). gain = b3 − b1 (Wiener x0 vs no warm start).
Single-seed (seed 0) protocol throughout, matching the external baselines; EXP-017 severity-matrix two-seed means (e.g. ours z b3 = 44.73) differ slightly from these seed-0 values.

| Arch | b1 (y+mask) | b3 (Wiener x0) | gain (b3−b1) | b4 (fusion) |
|---|---|---|---|---|
| Ours swin2d | 31.98 / 0.8995 | 31.69 / 0.8748 | **-0.29** | 32.00 / 0.8880 |
| Video Swin (official) | 30.94 / 0.8639 | 31.43 / 0.8797 | **+0.49** | — |
| SwinUNETR (official) | 30.93 / 0.8505 | 30.83 / 0.8427 | **-0.10** | — |
| RVRT (official) | 31.33 / 0.8754 | 6.30 / 0.1195 | collapse (excluded) | — |

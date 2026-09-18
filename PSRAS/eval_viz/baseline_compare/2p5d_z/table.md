# EXP-020 quantitative comparison — 2p5d_z

reval (per-item PSNR/SSIM over 8 val items, best.pt). gain = b3 − b1 (Wiener x0 vs no warm start).
Single-seed (seed 0) protocol throughout, matching the external baselines; EXP-017 severity-matrix two-seed means (e.g. ours z b3 = 44.73) differ slightly from these seed-0 values.

| Arch | b1 (y+mask) | b3 (Wiener x0) | gain (b3−b1) | b4 (fusion) |
|---|---|---|---|---|
| Ours swin2d | 38.91 / 0.9764 | 44.24 / 0.9881 | **+5.33** | 46.28 / 0.9929 |
| Video Swin (official) | 29.28 / 0.8231 | 34.70 / 0.9163 | **+5.42** | — |
| SwinUNETR (official) | 34.13 / 0.9666 | 40.52 / 0.9879 | **+6.39** | — |
| RVRT (official) | 34.08 / 0.9233 | 41.36 / 0.9848 | **+7.28** | — |

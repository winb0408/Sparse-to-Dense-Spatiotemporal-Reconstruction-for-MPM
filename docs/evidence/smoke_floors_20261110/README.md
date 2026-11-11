# 地板冒烟证据（2026-11-10，CPU，`patch_hw=[32,32]`，fold 0 test / M2）

这三个文件**不是头条结果**。它们只回答一个问题：新物理链路在真实数据上
是否按方法规格书说的那样行为。正式地板数必须用默认 256×256 重跑（NOT_RUN）。

| 臂 | psf_source | PSNR | SSIM | MAE |
|---|---|---|---|---|
| `P2_I0` | `identity` | 28.025295 | 0.750378 | 0.022582 |
| `P2_W0 (identity)` | `identity` | 28.025295 | 0.750378 | 0.022582 |
| `P2_W0 (synthetic_extra_degradation)` | `synthetic_extra_degradation` | 27.554492 | 0.628905 | 0.033829 |

两点可读的结论：

1. **identity 下 `P2_W0` 与 `P2_I0` 逐位相同**（两份 summary 除 `"arm"` 一行外完全一致），
   在真实数据上确认了 method spec §2 的退化，而不只是合成单测里成立。
2. **人为额外加模糊再反卷积会让结果变差**（PSNR 28.03 → 27.55，SSIM 0.750 → 0.629）。
   这不能读作"Wiener 有害"：真实 stack 已经带有该系统的光学，再按一个未标定的
   合成 PSF 去卷积，等于去掉了本来不存在的模糊。它能读作的是：把
   `synthetic_extra_degradation` 当成能刷分的旋钮是不成立的，这组数字就是反例。
   任何"调 σ 直到 Wiener 有增益"的做法都违反 method spec §2 的分组纪律。

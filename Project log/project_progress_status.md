# SPUR-MPM 项目进度状态

**截至 2026-10-28**：仓库底座（代码结构 + 可跑通的合成数据竖切）已完成；
Dryad 真实数据已到位并完成审计（SPUR-E000）、划分、预备与无学习基线（SPUR-E001）；
**尚无任何学习方法上的 SPUR 结论**（E002 正在跑）。
逐条实验记录见 [`Project_log.md`](Project_log.md)，研究计划见 [`plan.md`](plan.md)。

## 阶段状态

| 阶段 | 内容 | 状态 | Gate |
|---|---|---|---|
| Phase 0 | 数据审计（SPUR-E000） | ✅ 完成 | 29 文件 → 10 纳入 / 19 排除 / 3 重复 / **0 unresolved issue** |
| Phase 1 | 基线 A0/A1（SPUR-E001） | ✅ 完成 | 三折可复现（fingerprint `e473498d…`）；A0≡A1 已定性为恒等 PSF 的必然结果 |
| Phase 2 | 有监督上界 S0/S1（SPUR-E002） | 🔄 运行中 | 两臂同预算、同折，稠密参考验证集选模 |
| Phase 3 | MMS 自监督 SS0/SS1（SPUR-E003） | ⏳ 待 Phase 2 | holdout 仅用真实测量、选模只看 holdout loss |
| Phase 4 | 机制消融（SS2g / S1g，SPUR-E004/E005） | ⏳ 待数据 | 与 SS2 同预算同 fold |
| Phase 5 | 不确定性场（SPUR-E006） | ⏳ 待数据 | U 与误差的相关性为实验结论，非前提 |
| Phase 6 | 其他（DINOv3 / TYX 钙成像） | ⛔ 未启动 | 主线证明后才考虑 |

## 已完成（可复现）

- 单元测试 **75 项全绿**，含**决定性泄漏测试**：扰动 `y_hold` 后五个输入字段逐位不变；
  以及 `test_the_two_baseline_arms_are_the_same_method_under_an_identity_psf`。
- 合成数据全链路：生成 → 审计 → LOMO 划分 → prepare → 训练（S1、SS1）→ 评估（A1）→ 整卷推理。
  记录于 `SPUR-E000a`；**合成数据不作为任何证据**。
- 真实数据审计与隔离（SPUR-E000）：六条泄漏轴（动物 / session / FOV / 通道 / 重复文件 / 时间轴）各有代码断言；
  3 折留一鼠划分 + **逐折标定**（fold 各自 offset/scale，避免测试动物统计量回流）。
- 无学习基线（SPUR-E001）：A0 = A1 = PSNR 29.06 / Dice 0.869 / clDice 0.834（三折均值）。

## 立即待办（按序）

1. 完成 E002（S0/S1），再做 E003（SS0/SS1）与 E004（S1 vs SS1 同折配对）。
2. 深度分层结论一律以血管 Dice/clDice 为准，PSNR 不作深度指标（SPUR-E001 已说明原因）。
3. README 第 4 节列出的 `Vessels plus plaques*.tif` 不在上传包内，需向提供方确认。
4. 跨动物方差约 2.8 dB PSNR / 0.08 Dice，**任何小于 1 dB / 0.03 Dice 的差异不得作为结论**。

## 已知限制（写在这里，避免日后被当成结论）

- PSF 默认恒等：真实数据已含真实光学，**不做**合成高斯再模糊；因此 Wiener 分支退化为插值，
  这是可用结论，不是 bug。
- `virtual_sparse` 只讨论虚拟稀疏；**任何加速/降剂量声明只能在 `hardware_sparse` 下提出**。
- 仅 1 个测试组时 bootstrap 区间退化，已用 `ci_estimable: false` 显式标注，不得当作精度。

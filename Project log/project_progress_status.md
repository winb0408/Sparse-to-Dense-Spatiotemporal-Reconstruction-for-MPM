# SPUR-MPM 项目进度状态

**截至 2026-10-27**：仓库底座（代码结构 + 可跑通的合成数据竖切）已完成；
Dryad 真实数据仍在下载，**尚未产生任何 SPUR 科学结论**。
逐条实验记录见 [`Project_log.md`](Project_log.md)，研究计划见 [`plan.md`](plan.md)。

## 阶段状态

| 阶段 | 内容 | 状态 | Gate |
|---|---|---|---|
| Phase 0 | 数据审计（SPUR-E000） | ⏳ 待数据 | 轴/体素/动物身份全部可解析，0 unresolved issue |
| Phase 1 | 基线 A0/A1（SPUR-E001） | ⏳ 待数据 | 同一采样实例、同一 seed 下可复现 |
| Phase 2 | 有监督上界 S0-S2（SPUR-E002） | ⏳ 待数据 | 三臂同预算，稠密参考验证集选模 |
| Phase 3 | MMS 自监督 SS0-SS2（SPUR-E003） | ⏳ 待数据 | holdout 仅用真实测量、选模只看 holdout loss |
| Phase 4 | 机制消融（SS2g / S1g，SPUR-E004/E005） | ⏳ 待数据 | 与 SS2 同预算同 fold |
| Phase 5 | 不确定性场（SPUR-E006） | ⏳ 待数据 | U 与误差的相关性为实验结论，非前提 |
| Phase 6 | 其他（DINOv3 / TYX 钙成像） | ⛔ 未启动 | 主线证明后才考虑 |

## 已完成（可复现）

- 单元测试 68 项全绿，含**决定性泄漏测试**：扰动 `y_hold` 后五个输入字段逐位不变。
- 合成数据全链路：生成 → 审计 → LOMO 划分 → prepare → 训练（S1、SS1）→ 评估（A1）→ 整卷推理。
  记录于 `SPUR-E000a`；**合成数据不作为任何证据**。

## 立即待办（数据到位后，按序）

1. 解包 Dryad，跑 `audit_data`，人工核对 `audit_report.md`（轴顺序、z_step、各向异性、动物/FOV 身份）。
2. 把 `spur_mpm/configs/data/dryad_3p_vessels.yaml` 的 glob / regex / 逐文件元数据对齐真实目录结构。
3. `make_split` → `prepare_data`，确认标定只在训练组拟合。
4. 真实 TIFF 上先跑单臂单 fold（A0、A1、S1），确认管线不崩，再铺全臂全 fold。

## 已知限制（写在这里，避免日后被当成结论）

- PSF 默认恒等：真实数据已含真实光学，**不做**合成高斯再模糊；因此 Wiener 分支退化为插值，
  这是可用结论，不是 bug。
- `virtual_sparse` 只讨论虚拟稀疏；**任何加速/降剂量声明只能在 `hardware_sparse` 下提出**。
- 仅 1 个测试组时 bootstrap 区间退化，已用 `ci_estimable: false` 显式标注，不得当作精度。

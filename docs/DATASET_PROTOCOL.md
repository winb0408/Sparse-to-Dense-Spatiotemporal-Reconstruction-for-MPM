# 数据集与任务协议

本文件冻结当前项目的实验边界。后续训练、评估、论文表格都应先判断数据集身份，再选择模型维度；不要把不同维度、不同目标的数据混入一个总排名。

当前阶段的创新主线见 `docs/PRIOR_FUSION_EXPERIMENT_PROTOCOL.md`：以 `mask` 通道先验 + Wiener 物理初始化为核心，并进一步探索轻量 prior-aware fusion，使网络显式利用采样置信度和物理初值。

---

## 1. 数据集身份

| 数据集 | 当前身份 | 可作为主实验的数据形态 | 不作为主实验的用法 |
|---|---|---|---|
| CIDC25 | 3D stack / volume 数据集 | `[Z,H,W]` 或 `[T,H,W]` 体数据 | 只拿逐帧结果宣称 3D 重建 |
| MDC25-2D Nuclei | 2D 数据集 | 2D paired image，XY sparse / denoising | Z/XYZ sparse、3D volume benchmark |
| MDC25-3D Tribolium | 3D stack 数据集 | 3D stack；可派生 2D slice baseline 与 2.5D center-slice | 与 MDC25-2D 混成一个单一 MDC25 排名 |
| pollen | 2D real sparse 外部样本 | 当前由 stack mean projection 得到的 2D 图 | 严格 3D benchmark |

---

## 2. 模型维度定义

| 模型族 | 输入 | 输出 | 主实验含义 |
|---|---|---|---|
| 2D | 单张图或单切面 `[B,C,H,W]` | 单张图 `[B,1,H,W]` | 2D 重建；或 3D 数据的逐切面 baseline |
| 2.5D | 邻近切面作为 channel `[B,C×Wz,H,W]` | 中心切面 `[B,1,H,W]` | 利用 Z 邻域恢复中心切面，不是体输出 |
| 3D | 体块 `[B,C,Z,H,W]` | 体块 `[B,1,Z,H,W]` | 真正体重建 |

---

## 3. 主实验合法矩阵

| 数据协议 | 2D 模型 | 2.5D 模型 | 3D 模型 |
|---|---|---|---|
| CIDC25-3D | 合法，逐切面 baseline | 合法，中心切面重建 | 合法，体重建 |
| MDC25-2D | 合法，仅 XY | 不作为主实验 | 不作为主实验 |
| MDC25-3D | 合法，逐切面 baseline | 合法，中心切面重建 | 合法，体重建 |
| pollen-2D | 合法，外部 2D real sparse | 仅迁移探索 | 仅迁移探索 |

---

## 4. 稀疏模式解释

- `x`：2D/切面内单轴列向稀疏。用于 2D 主实验或 3D 单切面 baseline。
- `y`：2D/切面内单轴行向稀疏。用于方向敏感性对照。
- `xy`：帧内二维空间稀疏。2D 与 3D 数据都可解释。
- `z`：轴向/切面稀疏。只对 3D stack 有严格物理意义。
- `xyz`：轴向与帧内联合稀疏。只对 3D stack 有严格物理意义。

因此，2D 数据集上的 `z/xyz` 结果只能作为错误配置或迁移探索，不进入主实验。当前主线建议：2D 场景报告 `x/y/xy`，2.5D 场景报告 `z`，3D 场景报告 `xyz`。

---

## 5. 结果使用规则

1. 主表只收录同一数据协议、同一评估集合、同一稀疏定义下的结果。
2. 跨数据集、跨维度、不同帧数的结果不得做总排名。
3. 2.5D 结果表述为“中心切面重建”，不要表述为“3D 体重建”。
4. pollen 当前是 2D 外部样本；3D/2.5D 权重跑 pollen 时必须标为 transfer probe。
5. MDC25 必须拆成 MDC25-2D 和 MDC25-3D 两个协议讨论。
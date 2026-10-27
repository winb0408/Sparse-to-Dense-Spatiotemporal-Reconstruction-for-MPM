# 版本演进（v1 → v4）与目录对应

这个仓库里同时存在两代东西：仓库根目录的 **legacy threePM**（v1/v2）和 `PSRAS/`
下的 **协议重构**（v3/v4）。先看这张对应表，再往下读：

| 版本 | 时间 | 位置 | 分支 | 状态 |
|---|---|---|---|---|
| v1/v2 legacy threePM | ≤ 2026-09 | 仓库根目录（`models/`、`train.py`、`experiments/`、`report/`、`overleaf_*`） | `main`、`dev` | 冻结，只作引用，不作证据 |
| v3 PSRAS 协议重构 | 2026-09-09 ~ 09-21 | `PSRAS/`（当前代码结构的起点） | `PSRAS` 历史提交 | 关闭，部分结论被 v4 修正 |
| v4 PSRAS 协议重置 | 2026-09-22 ~ 至今 | `PSRAS/` | `PSRAS`，快照 `review-v4-20261019` | **当前主线** |

v3 的计划与进度表归档在 `PSRAS/Historical_Plan/`。

## v1/v2：legacy threePM

**做了什么**：Swin-UNet 2D / Video Swin-UNet 3D，在 CIDC25 上做 2D / 2.5D / 3D 三
条线的稀疏→稠密重建，损失是 MSE + MS-SSIM + DINOv2 特征。核心结论是"Wiener 初始
化是跨 2D/2.5D/3D 最稳定的收益来源；mask 通道有帮助但 2.5D 原始拼接会产生负迁移"。

**为什么被替代**：

1. 稀疏掩码是**离线预生成**的，与采集几何解耦，无法表达"行稀疏 vs 平面稀疏"的区
   别，也无法做 fixed-budget 分配研究；
2. 指标口径不统一（SSIM 窗口、val/test 划分、mask 语义多次变更），历史结果之间不
   可比；
3. 缺 PSF 的显式前向模型，"物理先验"实际是经验性的 Wiener 初始化，无法解释增益
   来源。

v3 的起点就是把这三件事制度化：统一前向模型 `Y = M⊙(H*X+n)`、运行时生成观测、显
式 PSF 与 mask。

## v3：PSRAS 协议重构（2026-09-09 ~ 09-21）

**目标**：把"训练一个去噪网络"改写成"回答采样协议问题"。新增 `mode ∈ {xy, y, z,
yz, xyz}` 的独立稀疏轴、runtime 采集仿真、物理先验家族（bilinear / Wiener /
fill-first）、门控融合、以及 severity 曲线与 fixed-budget 实验框架。

**产出**（EXP-001 ~ EXP-021，详见 `03_EXPERIMENTS.md`）：severity 全矩阵（K2/K4/
K8/K16 × B0–B4）、T-scaling 曲线、光子预算剂量-响应、外部仓库基线矩阵（vswin/
sunetr/rvrt）、no-PSF 消融。

**v3 内部被证伪的三条架构线**（v4 不再重复投入）：

- **residual 学习** `X̂ = X0 + ΔX`：在 K16 等易点跨种子反转，收益不稳定；
- **统一 3D 网络 U3D**：EXP-012/013 两轮试点均未通过退出条件；EXP-014 发现两个不
  同配置的试点逐位相同，定位到轴向分支实际未参与 → DEC-019 关闭；
- **video3d（完整 3D 移窗注意力）**：EXP-018 复验仍不通过，架构级证伪升级。

**v3 后期暴露的三个协议问题**（v4 用 DEC-025 一次性修掉）：

1. split 不诚实（patch 级重叠 → stack 级重划，split v2）；
2. SSIM/误差口径不统一（改用标准 2D 高斯窗 + 显式 mask 计算 e_measured/e_missing）；
3. 冻结 λ 的 Wiener 先验跨情境迁移失败（Y regime 下 −11 ~ −23 dB，Gibbs 崩坏）。

## v4：协议重置（2026-09-22 ~ 至今）

**变化**：DEC-025 把协议冻结成 7 条（split v2、指标口径、采样定义、归一化与 PSF、
Protocol A/B 语义、arm 命名、legacy 归档），并把阶段重排为 P0–P6。此后的所有数字都
是 v4 口径，与 v3 / legacy **不做同表对比**。

**阶段与结论**：见 `01_PROGRESS.md` 与 `02_CLAIMS_EVIDENCE.md`。一句话概括：

- P1 Y baseline：Bilinear 先验全 K 有正增益；冻结 λ Wiener 在 Y 上崩坏，但网络能
  吃下垃圾先验不崩（这个观察后来长成 C3）。
- P2 先验独立评估：IterDC 与 FillFirst-Wiener 入选，P3 自适应 λ 证伪。
- P3 YZ 主线：分配律 + 门控融合增益 + "先验好≠网络好"三条主张定型。
- P4 Unified：A_Z 适配器无稳健增量 → 定型 UnifiedV4-noAZ；跨协议代价 ≤0.74 dB。
- P5 XYZ / noise_stress / 加权：Extension gate 未达（−0.6 ~ −1.2 dB），且 EXP-028
  证明不是采样份额问题。

**v4 内部被证伪/未达标的方向**（写论文时的边界，详见 `05_NEGATIVE_RESULTS.md`）：

- 学习式轴向适配器 A_Z（EXP-025，5/6 种子为负）；
- 统一模型追平专训模型（EXP-026/028，3/3 seed 未达 0.5 dB 容差）；
- 采样权重（share）是稀释主因（EXP-028，直接证伪）；
- zero-shot 跨 regime 迁移（EXP-026，崩溃 7.8–15.1 dB）；
- IterDC 先验对 Bilinear 的普遍优势（EXP-024 结果 2，未达 3/3）。

## 命名与缩写对照（审查时容易混淆）

| 缩写 | 含义 |
|---|---|
| b1 / Direct | 只输入稀疏观测 Y 与掩码 M |
| b2 / Bilinear | 加 bilinear 几何先验通道 x0 |
| b3 / Wiener | 加冻结 λ 的 Wiener 先验（v3 主线，v4 仅作反面对照） |
| b4i / IterDC | 加迭代 data-consistent 反演先验 + 门控融合 |
| b4f / FillFirst | 加 fill-first Wiener 先验 + 门控融合（v4 最优） |
| SY / SYZ / SXYZ | 专训于 Y / YZ / XYZ 的统一架构模型 |
| U / U_noAZ / U15 / U15W | 9 点统一 / 无轴向先验 / 15 点统一 / 1:1:2 加权的 15 点统一 |
| y / yz2 / xyz2 | 只行稀疏 / 行+平面稀疏 / 横向 cross 网格+平面稀疏 |
| (ky,kz) | 行周期与平面周期，B=ky·kz 为预算代理 |

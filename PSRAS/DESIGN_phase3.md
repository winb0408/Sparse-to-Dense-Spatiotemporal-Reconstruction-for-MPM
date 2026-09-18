# Phase 3 设计提案：统一 3D 网络（解决"先验兑现率"瓶颈）

日期：2026-09-10 状态：待用户评审
前置：EXP-010（三）判定表闭合（Phase 2 收官）

## 1. 动机（来自 Phase 2 的量化瓶颈）

- 网络 X0 兑现率：K4≈50%，K8≈30%（B3/B4 @λ3e6 vs X0-only 58.50/54.80）
- 根因假设：2.5D UNet 把 T 个切片当作**独立通道**做 2D conv——轴向关系
  只能隐式学习；K8 时轴向空隙 7 切，隐式学习不足以跟踪高质量 X0
- Phase 2 已证明 X0 质量是最大杠杆（+6.5 dB），当前架构吃不满

## 2. 架构：axial-3D 统一网络（models/unified_3d.py）

```
输入 [B, F, T, H, W]（注意：保留 f-major 布局语义，DEC-013）
  stem:  PriorFusionStem-3D（e_y/e_w/e_s 沿 T 共享，输出 [B, base, T, H, W]）
  encoder: L × { spatial Swin window attn (H,W) + axial attn (T) }
           —— 空间窗口下采样、z 轴不下采样（anisotropy-aware，M6）
  decoder: 对称上采样 + axial attn
  head:  [B, 1, T, H, W]（每切片一输出通道）
```

关键点：
- **axial attention 在 T=1 时自然退化为恒等**（self-attn over 1 token），
  无需特判 → 同一权重覆盖 2D（T=1）与 2.5D（T=2K−1）
- axial attn 每层带 spatial 位置编码外的 z 位置编码（相位感知，DEC-012）
- 参数量对齐 B4（±10%），公平对比

## 3. 训练协议（DEC-016 提案）

- warm_start：`wiener`（λ=3e-6，DEC-014 修订为 z 线默认；横向线自动
  等价旧序——z 无横向空穴）；对照线补 λ=0.01
- T 采样：每 batch 从 {1, 2K−1} 中采样（T=1 保证 2D 兼容 + 权重共享验证）
- Q regime：query one-hot（现行）；验证阶段测 Q=[0,1,0] / 全 1 三种
- 损失：L1+MS-SSIM（现行）+ B6 的 DC 项作为消融开关
- 步数：8k 起步，K8 加到 12k（兑现率瓶颈可能部分是欠训练）

## 4. 预注册判定（Phase 3 出口条件）

| 判据 | 阈值 |
|---|---|
| U3D vs B4（z-K8, λ3e6） | ≥ +1.0 dB（兑现率 30%→50%+ 的直接检验） |
| U3D vs B4（z-K4, λ3e6） | ≥ 0（不伤易协议） |
| 同权重 T=1 vs 2D B4 基线 | ≥ −0.5 dB（退化容忍） |
| PSNR vs T 曲线 | 单调非降（context scaling 成立） |
| 多种子 | 每格 ≥2 种子，种子 σ 报告 |

## 5. 实施顺序

1. `models/unified_3d.py` + 单元测试（T=1 恒等退化、相位编码、参数量）
2. X0-only 基线全部复用（无新计算）
3. 训练 z-K4/K8 × λ3e6 × B4 等效配置（fusion on）
4. T-scaling 曲线（T ∈ {1, 3, 7, 15}，K=8）
5. 判定表更新 → EXP-012

## 6. 风险与回退

- axial attention 显存：T=15, H=W=256 → 15×15 attn 微小；spatial window 8 照旧
- 若 U3D 在 z-K8 不达标：先加步数到 16k（欠训练假设），再查 stem
- 若 T=1 退化不成立：给 axial 分支加零初始化残差缩放

# Project Progress Status — 对照 PLAN.md（v4）

> 用途：对照 [PLAN.md](PLAN.md)（Protocol v4，2026-09-24 重写）的阶段计划，验收当前进度。
> 实验证据树见 [PROJECT_LOG.md](PROJECT_LOG.md)（append-only）。
> v3 时代的进度表已归档：[Historical_Plan/Project_progress_status_v3_20260924.md](Historical_Plan/Project_progress_status_v3_20260924.md)。
> 更新至：2026-09-24
>
> 图例：✅ 完成且达标 ｜ 🟡 部分完成/有修订 ｜ ❌ 未通过/关闭 ｜ 🔵 进行中 ｜ ⬜ 未开始

---

## 0. 当前一句话状态

**Phase 0–4 全部关闭（Protocol Cleanup → Y baseline → YZ physics prior → YZ Mainline → Unified Framework）。
核心科学问题"Y/Z 稀疏如何分配最有效"已回答：预算省在 z（(2,8) 最优，3/3 种子），最优先验配方为 FillFirst-Wiener (b4f)。
Phase 4 收口（EXP-025）：统一架构定型 **UnifiedV4-noAZ**（A_Z 无稳健增量，轴向耦合由物理先验通道承载）；
unified 跨 regime 代价 ≤0.74 dB vs specialist 迁移代价 1.4~2.6 dB（Y 套件 gate 临界未达，诚实披露）。
下一步 = Phase 5 XYZ/XYT Extension + noise_stress（待预注册）。**

## 1. v4 阶段验收表

| Phase | 内容 | 状态 | 证据 | Gate 判定 |
|---|---|---|---|---|
| P0 | Protocol Cleanup（split v2 / SSIM 2D 窗 / 显式 mask / 冻结） | ✅ 2026-09-22 | DEC-025 | 7/7 P0 断言修复；修正指标下主结论存活 ✅ |
| P1 | Corrected Y baseline（EXP-022，36 run） | ✅ 2026-09-23 | EXP-022 结果1-4 | Bilinear>Direct 全 K 3/3 同号；冻结λ Wiener 先验崩坏（负增益）但网络鲁棒 ✅ |
| P2 | Genuine YZ physics prior（EXP-023，1008 行先验扫描） | ✅ 2026-09-23 | EXP-023 结果1 | P4 IterDC 5/6 最优、P2 FillFirst-Wiener kz 大时领先；P1/P3 淘汰；先验侧 (2,8) 最优 ✅ |
| P3 | YZ Mainline（EXP-024，40 run 双口径） | ✅ 2026-09-24 | EXP-024 结果1-4 | **(2,8)>(4,4) 3/3 种子 (+2.0~+4.2 dB)、(8,2) 垫底——"预算省在 z"成立**；b4f ≥ b4i 全 6 组合；先验+融合 vs Direct 全组合 +0.5~+3.4 dB ✅ |
| P4 | Unified Framework（shared encoder + fusion + Z adapter + specialist 对比） | ✅ 收口（部分达标） | EXP-025（2026-09-26） | 15/15 run；YZ 套件 gate 达标、Y 套件临界未达（−0.54~−0.74）；A_Z 无稳健增量→UnifiedV4-noAZ 定型；跨 regime 代价 1.4~2.6→≤0.74 dB |
| P5 | XYZ / XYT Extension + noise_stress | ⬜ | — | XYT 数据依赖待确认 |
| P6 | Protocol A / Real Acquisition | ⬜ | — | 未落地，不做主张 |

## 2. 已确立的论文级主张（v4 口径）

| # | 主张 | 支撑 | 强度 |
|---|---|---|---|
| 1 | **Fixed-budget Y/Z 分配律**：B=16 下行密采+轴疏采 (ky,kz)=(2,8) 最优，(4,4) 次之，(8,2) 最差 | EXP-024 结果4（4 arm 排序一致 + (2,8)>(4,4) 3/3 种子；先验侧 EXP-023 预演一致） | 核心创新 |
| 2 | **Wiener 家族物理先验 + 门控融合**在 yz 双轴稀疏下对 Direct 网络全组合稳定增益 | EXP-024 结果1（6/6 组合 +0.5~+3.4 dB，三元组 3/3 同号） | 强 |
| 3 | **先验好 ≠ 网络好**：融合机制偏好"有结构偏差但可学习"的先验（FillFirst-Wiener ≥ IterDC 全组合，与先验侧排序相反） | EXP-024 结果3 | 机制性发现 |
| 4 | 冻结 λ Wiener 先验跨情境迁移失败（Gibbs 崩坏），网络对劣质先验鲁棒 | EXP-022 Question C | 反面对照 |
| 5 | Bilinear 几何先验 > Direct（Y 稀疏全 K） | EXP-022 Question B | 基线级 |
| 6 | （v3 期遗留，机理仍成立）3PM 省时采样信息瓶颈在轴向；横向协议先验天花板=插值水平 | EXP-016/020/021（勘误后归因：侧向 PSF 退卷积） | 论证性消融 |

**不可主张**（预注册否证，如实报告）：IterDC-prior 对 Bilinear-prior 的普遍优势（Question C(yz) 未达 3/3）；residual 设计（B5）；K16 fusion 增益（v3 期跨种子反转）；2D/横向协议 Wiener 增益（≈0 或负）。

## 3. 关键读数存档（EXP-024 主矩阵，test PSNR/SSIM）

| (ky,kz) | Direct (b1) | Bilinear-prior (b2) | IterDC-prior (b4i) | FillFirst-Wiener (b4f) |
|---|---|---|---|---|
| (2,2) | 37.70/0.9603 | 38.78/0.9641 | 40.91/0.9785 | **41.06/0.9799** |
| (2,4) | 37.54/0.9604 | 39.07/0.9714 | 39.04/0.9705 | **40.45/0.9787** |
| (2,8) n=3 | 35.25/0.9358 | 36.86/0.9572 | 37.22/0.9615 | **37.96/0.9659** |
| (4,2) | 33.07/0.8992 | 34.14/0.9154 | **34.47/0.9225** | 34.40/0.9206 |
| (4,4) n=3 | 32.94/0.8970 | 33.78/0.9108 | 33.91/0.9130 | **34.17/0.9174** |
| (8,2) | 29.23/0.8034 | 29.24/0.8059 | 29.81/0.8291 | **29.82/0.8297** |

（val 口径同序；(2,8)/(4,4) 为 3 种子均值，(8,2) 单种子如实标注；原始 json 在 runs_v3/YZ_*/）

## 4. 下一步（待用户批准后预注册）

1. ✅ **Phase 4 Unified Framework 已收口（EXP-025，2026-09-26）**：15/15 run + 270 个 reval json。判读：① Unified gate Y 套件临界未达（U−SY 三种子 −0.54/−0.62/−0.74，均超 −0.5 线；Y 套件 yK2 最易点稀释 −1.2~−1.6 dB 为主因）、YZ 套件达标（−0.38/+0.09/+0.23）；② A_Z 无稳健增量（两系列 5/6 种子为负，U_noAZ 为 YZ 套件最佳 arm）→ 回退叙事成立：轴向耦合由物理先验通道承载即可，最终架构 **UnifiedV4-noAZ**；③ specialist 跨 regime 迁移代价 1.4~2.6 dB，unified 压缩至 ≤0.74 dB（核心卖点成立）。详见 PROJECT_LOG.md EXP-025（结果1-5）。
2. 候选并行：Phase 5 noise_stress（F1/F2/F3 配对 GT）、稳健性扫描（λ 敏感性 / PSF mismatch）。
3. 远期：Phase 5 XYZ/XYT、Phase 6 Protocol A（pollen 真实采集）。

## 5. 资源状态

- GPU1 他人进程（EikonalViT，~13GB）；GPU2/3 空闲（EXP-025 全部链 2026-09-26 11:59 DONE）；GPU0 他人进程不碰。根分区 / 接近满（~900MB 空闲，/tmp 31GB 垃圾），建议清理；本项目长链已统一 TMPDIR=/data2/wjb/tmp。
- runs_v3/：EXP-022 36 run + EXP-023 先验扫描 + EXP-024 40 run + EXP-025 15 run（UNIV4_*），全部含完整 config 与双口径指标。

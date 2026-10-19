# Project Progress Status — 对照 PLAN.md（v4）

> 用途：对照 [PLAN.md](PLAN.md)（Protocol v4，2026-09-24 重写）的阶段计划，验收当前进度。
> 实验证据树见 [PROJECT_LOG.md](PROJECT_LOG.md)（append-only）。
> v3 时代的进度表已归档：[Historical_Plan/Project_progress_status_v3_20260924.md](Historical_Plan/Project_progress_status_v3_20260924.md)。
> 更新至：2026-10-07
>
> 图例：✅ 完成且达标 ｜ 🟡 部分完成/有修订 ｜ ❌ 未通过/关闭 ｜ 🔵 进行中 ｜ ⬜ 未开始

---

## 0. 当前一句话状态

**Phase 0–4 全部关闭（Protocol Cleanup → Y baseline → YZ physics prior → YZ Mainline → Unified Framework）。
核心科学问题"Y/Z 稀疏如何分配最有效"已回答：预算省在 z（(2,8) 最优，3/3 种子），最优先验配方为 FillFirst-Wiener (b4f)。
Phase 4 收口（EXP-025）：统一架构定型 **UnifiedV4-noAZ**（A_Z 无稳健增量，轴向耦合由物理先验通道承载）；
unified 跨 regime 代价 ≤0.74 dB vs specialist 迁移代价 1.4~2.6 dB（Y 套件 gate 临界未达，诚实披露）。
Phase 5 XYZ 已收口（EXP-026）：15 点统一模型 U15 三 regime 全覆盖（XYZ 34.5~35.0 dB），Extension gate 未达（−0.6~−1.2 dB 混合稀释，诚实披露）、Regression 达标（≤0.6 dB）、zero-shot 崩溃确认新 regime 必须进训练集。noise_stress 已收口（EXP-027）：退化随噪声单调（F1 全点收敛 26~28 dB 噪声天花板）、XYZ 稀释不损噪声稳健性、退化主因是网络放大噪声而非先验坍塌（指向噪声感知训练方向）。regime 加权已收口（EXP-028）：Extension gate 仍 3/3 未达且加权零 XYZ 增益——**share 主导假设证伪，残余差距归因容量与任务间干扰**；稀释定律修正为『采样权重只能移动回退分配，无法缩小统一-专训差距』。下一步候选 = 两阶段训练 / 噪声感知训练 / Phase 6 Protocol A。**

## 1. v4 阶段验收表

| Phase | 内容 | 状态 | 证据 | Gate 判定 |
|---|---|---|---|---|
| P0 | Protocol Cleanup（split v2 / SSIM 2D 窗 / 显式 mask / 冻结） | ✅ 2026-09-22 | DEC-025 | 7/7 P0 断言修复；修正指标下主结论存活 ✅ |
| P1 | Corrected Y baseline（EXP-022，36 run） | ✅ 2026-09-23 | EXP-022 结果1-4 | Bilinear>Direct 全 K 3/3 同号；冻结λ Wiener 先验崩坏（负增益）但网络鲁棒 ✅ |
| P2 | Genuine YZ physics prior（EXP-023，1008 行先验扫描） | ✅ 2026-09-23 | EXP-023 结果1 | P4 IterDC 5/6 最优、P2 FillFirst-Wiener kz 大时领先；P1/P3 淘汰；先验侧 (2,8) 最优 ✅ |
| P3 | YZ Mainline（EXP-024，40 run 双口径） | ✅ 2026-09-24 | EXP-024 结果1-4 | **(2,8)>(4,4) 3/3 种子 (+2.0~+4.2 dB)、(8,2) 垫底——"预算省在 z"成立**；b4f ≥ b4i 全 6 组合；先验+融合 vs Direct 全组合 +0.5~+3.4 dB ✅ |
| P4 | Unified Framework（shared encoder + fusion + Z adapter + specialist 对比） | ✅ 收口（部分达标） | EXP-025（2026-09-26） | 15/15 run；YZ 套件 gate 达标、Y 套件临界未达（−0.54~−0.74）；A_Z 无稳健增量→UnifiedV4-noAZ 定型；跨 regime 代价 1.4~2.6→≤0.74 dB |
| P5 | XYZ / XYT Extension + noise_stress + regime 加权 | ✅ 主体关闭（EXP-026/027/028） | EXP-026（10-03）+ EXP-027（10-06）+ EXP-028（10-07） | Extension gate 未达且 share 假设证伪（加权 1:1:2 零 XYZ 增益，差距 −0.7~−1.0 纹丝不动→容量/干扰归因）；Regression gate 达标；zero-shot 崩溃 → 新 regime 必须进训练集；noise_stress：F1 天花板 26~28 dB、退化 = 网络噪声放大非先验坍塌；XYT 维持延后 |
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

## 4. 经验与不足（Phase 4+5 复盘，详见 PROJECT_LOG.md DEC-027）

**方法论经验**：① 零初始化恒等构造 + 预注册消融使架构问题低成本裁决（A_Z 两轮证伪成完整证据链）；② 逐 seed 配对 + val/test 双口径是 0.5 dB 判读最低门槛；③ 通道化 conditioning 的 regime 扩展成本极低（xyz2 仅 ~30 行 mask 代码，先验管线零改动）；④ 零样本基线先行锁定结论边界；⑤ 预注册回退分支让 gate 未达实验仍有完整产出。

**不足**：① 均匀点采样稀释定律——gate 0.5 dB 容差在 ≥12 点混合下结构性无法满足（数据占比决定回退幅度，EXP-026 XYZ 仅 6/15 → −0.6~−1.2 dB）；② 无 checkpoint-resume（崩溃即 7h 级重跑，两次事故损失 ~20 GPU·h）；③ bash 链 exit-0 误报（OOM/ENOSPC 崩溃被记为成功，须以 final_val.json 存在性为准）；④ 共享 GPU 脆弱（他人进程 5 run 崩溃，wait_idle 为事后补丁）；⑤ 单数据集依赖（全部结论基于 CIDC25 仿真，外部效度检验未开始）。

## 5. 下一步（决策框架见 DEC-027，待用户批准后预注册）

| 候选 | 投入 | 产出 | 状态 |
|---|---|---|---|
| B. noise_stress（F1/F2/F3 配对 GT，复用 U15/U_noAZ） | ~3h 评估 | 稳健性章节 + 外部效度首检 | ✅ 已收口（EXP-027，2026-10-06，90 reval 零训练） |
| A. regime 加权采样（U15W 重训 3 seed） | 3 run ~21h | Extension gate 能否达标的直接回答 | ✅ 已收口（EXP-028，2026-10-07）——gate 未达，share 假设证伪（加权零 XYZ 增益 +0.0 均值，付 Y −0.6 代价）；候选转向两阶段训练/容量扩展 |
| C. checkpoint-resume 机制 | ~2h 实现 | 消除长链重跑风险（工程债） | ✅ 已落地（train.py --ckpt_every 500 原子快照 + --resume；冒烟通过：model/opt/sched/scaler/step/best_psnr/mix_rng 恢复，log.csv resumed_from 标记；dataloader 批次序列不恢复为已声明限制） |
| D. Phase 6 Protocol A 预调研（pollen 数据可行性） | 0.5 天调研 | 论文最终 gate 可行性判定 | ⬜ 用户指示暂不做 |
| E. 噪声感知训练（EXP-027 结果4 指向） | 未预注册 | 收回 F1-F3 下 2~5 dB 网络噪声放大损失 | 新增候选，待讨论 |

论文素材现状：Fig1-6 齐备（EXP-022~026），统一图集见 `eval_viz/paper_figures/`，Fig7（XYT）数据依赖缺，Fig8（真实采集/噪声）noise_stress 侧已齐（EXP-027，图在 `eval_viz/exp027_noise_stress/`），真实采集侧待 D。

## 6. 资源状态

- GPU1/2/3 空闲（EXP-028 链 2026-10-07 10:34 DONE + reval 10:40 DONE）；GPU0 他人进程不碰。根分区 / 接近满（~900MB 空闲，/tmp 31GB 垃圾），建议清理；本项目长链已统一 TMPDIR=/data2/wjb/tmp。
- runs_v3/：EXP-022 36 run + EXP-023 先验扫描 + EXP-024 40 run + EXP-025 15 run + EXP-026 6 run + EXP-027 90 noise_stress reval + EXP-028 3 run（UNIV4_U15W_*）× 45 reval，全部含完整 config 与双口径指标。
- 可视化产物：`eval_viz/exp027_noise_stress/`（noise_curve + 5 panel，PNG+PDF）；`eval_viz/exp028_regime_weight/`（exp028_suite_curves 三套件三线对比）；x0 先验数值 `runs_v3/exp027_x0_prior_noise.json`。
- 论文级图集：`eval_viz/paper_figures/`（交付 2026-10-19，脚本 `scripts/paper_figs.py`）——创新点 C1~C4 各有定量图 + 定性重建图，索引见 `eval_viz/paper_figures/INDEX.md`；数值备份 `key_numbers.csv`，定性面板原始数组 `*_data.npz`；未新增实验，全部由已提交 ckpt 复现。

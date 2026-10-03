# PSRAS 项目计划（Protocol v4）

> 版本：v4（2026-09-24 重写）。依据《20260921PSRAS_NextStage_Project_Revision_Plan.md》（外部审查 v4.0-draft）
> + DEC-025 的 5 项本地修正（本地无独立测试栈 / 重评估代替全量重训 / repo 不重构 / XYT 数据依赖 / 矩阵分级）。
> v3 计划全文归档于 [Historical_Plan/PLAN_v3_20260909.md](Historical_Plan/PLAN_v3_20260909.md)；
> v3 阶段进度表归档于 [Historical_Plan/Project_progress_status_v3_20260924.md](Historical_Plan/Project_progress_status_v3_20260924.md)。
> 实验证据树：[PROJECT_LOG.md](PROJECT_LOG.md)（append-only，DEC/EXP 编号持续）。
> 当前状态速览：[Project_progress_status.md](Project_progress_status.md)。

---

## 0. 项目一句话定位

三光子显微镜（3PM）稀疏采样→稠密重建：在统一 forward model `Y = M⊙(H*X + n)` 下，
**acquisition regime 一级分类 = Y / YZ / XYZ / XYT**（2D/2.5D/3D 降为网络上下文描述），
核心创新押注：
1. **Fixed-budget Y/Z allocation**——同样采样预算下 Y 与 Z 如何分配最有效；
2. **Sampling-conditioned physics prior**——先填洞后 Wiener 退卷积（FillFirst-Wiener）等物理先验家族 + 门控融合；
3. **Unified 框架**（shared spatial backbone + axis adapter）。

PSRAS 缩写语义 v4 起待终定（Residual 分支已在 v3 期证伪淘汰，见 Historical_Plan 归档）。

**一句话项目规则**：先预注册再跑；结论必须附数据支撑；推翻旧结论不删除旧记录，追加 `supersedes`；
legacy（threePM 旧分支）结果只引用不作证据；实验记录统一"全称+缩写"双写法说明每个 arm 配置。

## 1. 已冻结的协议约定（Phase 0 产物，DEC-025）

| 项 | 约定 |
|---|---|
| Split v2 | train={A1,B1,C2,D2}，val=F0[0:750]，test=F0[750:1500]（内容独立、选择接触声明）；F1/F2/F3 为 F0 同内容噪声变体（corr .71/.36/.18）→ noise_stress，不作独立栈；calibration 只用训练栈 |
| 指标（v4 口径） | SSIM 用标准 2D 可分离高斯窗（训练损失与评估共用）；e_measured/e_missing 由显式 mask M 计算；val/test 双口径；多种子 mean±std |
| 采样 | make_mask_yz2(ky,kz)：行稀疏 ⊗ 平面稀疏，K=ky、K2=kz、T=2·kz−1；z 主线"Wiener"= 侧向 2D Wiener + 轴向插值（命名勘误 DEC-025） |
| 归一化/PSF | 固定标定常数（训练集统计），无 GT 依赖；PSF σxy=1.0 px / σz=1.5 px；noiseless 主线 |
| Protocol A | 未落地（真实采集 pollen 线），Phase 6 前不做主张 |
| Arm 命名 | 全称+缩写双写法：b1=Direct、b2=Bilinear-prior、b4i=IterDC-prior fusion、b4f=FillFirst-Wiener fusion（公式与 λ 见 EXP-024 预注册配置表） |

## 2. 分阶段计划与当前状态（v4 阶段编号）

> 注：2026-09-22~24 的日志中 EXP-023/024 曾按旧叫法称 "Phase 2 先导/Phase 2 YZ 主线"，
> 按 v4 方案编号对应关系为：EXP-023=Phase 2，EXP-024=Phase 3。

### Phase 0 — Protocol Cleanup ✅ 2026-09-22 关闭（DEC-025）
- [x] stack-level split 修正（split v2 诚实语义）
- [x] reval 显式 mask + --split 分文件
- [x] SSIM 标准 2D 高斯窗
- [x] 归一化 / PSF / 种子冻结
- [x] Protocol A/B 语义文档化；legacy 定量结果全部归档
- Gate：无未解决 P0 协议 bug ✅（7/7 断言核验修复；修正指标下主结论存活核验通过）

### Phase 1 — Corrected Y baseline ✅ 2026-09-23 关闭（EXP-022）
- [x] Y K∈{2,4,8,16}，Direct / Bilinear / 冻结λ Wiener，3 种子 × 双口径 = 36 run
- Gate：区分 "Wiener vs Bilinear" 与 "Wiener prior vs Direct Network" ✅
  - Question B（Bilinear-prior vs Direct）：全 K 3/3 同号正（K2 +1.5 → K16 +0.16）
  - Question C（冻结 λ=3e-6 Wiener）：全 K 3/3 同号**负**——先验自身 Gibbs 崩坏（−23~−11 dB），网络鲁棒（b3≈b1/b2）
  - 指示：主线先验必须 noise-adaptive / 迭代反演，不复用冻结 λ

### Phase 2 — Genuine YZ physics prior ✅ 2026-09-23 关闭（EXP-023）
- [x] 先验配方独立评估（无网络，Question A 口径）：P0 bilinear / P1 冻结λ / P2 fill-first / P3 密度自适应λ / P4 迭代 DC 反演，6 (ky,kz) 组合 × 1008 行扫描
- Gate：prior 本身经独立验证，不由下游网络定义 ✅
  - P4（IterDC）5/6 组合最优；P2（FillFirst-Wiener）在 kz≥4/ky=2 大幅领先；P1 崩坏证实、P3 证伪
  - fixed-budget B=16 先验侧预演：(2,8) 最优（"预算省在 z"）

### Phase 3 — YZ Mainline ✅ 2026-09-24 关闭（EXP-024）
- [x] KY×KZ 全矩阵（(2,2)/(2,4)/(2,8)/(4,2)/(4,4)/(8,2)）× 4 arm，40 run 双口径 reval
- [x] asymmetric sparsity + fixed-budget allocation（B=16 三元组 3 种子）
- [x] acquisition-efficiency 分析（4 arm 排序一致性）
- Gate：回答"Y/Z 稀疏如何分配最有效" ✅
  - Question B(yz)：先验+融合对 Direct 全 6 组合 +0.5~+3.4 dB，三元组 3/3 同号
  - 分配律：**(2,8) > (4,4) 3/3 种子（+2.0~+4.2 dB）、(8,2) 垫底（单种子）——"预算应省在 z"（行密采+轴疏采）成立**
  - 最优先验配方：**FillFirst-Wiener (b4f)** ≥ IterDC (b4i) 全 6 组合（先验好 ≠ 网络好：融合偏好"有结构偏差但可学习"的先验）
  - Question C(yz)：IterDC vs Bilinear 增益不均匀（部分否证，如实报告）

### Phase 4 — Unified Framework ✅（EXP-025，2026-09-26 收口）
- [x] shared spatial encoder（2D 主干，v3 期已证 axial 3D 网络三变体不过线，DEC-019）
- [x] sampling-conditioned fusion（b4f 先验默认输入；UnifiedV4 逐 slice 共享融合 stem，gate 零初始化）
- [x] Z adapter（A_Z 零初始化轴向适配器）：**消融无稳健增量**（两系列 5/6 种子为负）→ 采纳回退：轴向耦合由物理先验通道承载，最终架构 UnifiedV4-noAZ
- [x] specialist vs unified 对比（EXP-025，5 arm × 3 seed，matched budget）
- [x] 主验证采样点：全 9 点双口径（y-K{2,4,8} + yz2-6 组合）
- Gate：**部分通过**——YZ 套件达标（−0.38/+0.09/+0.23）；Y 套件临界未达（−0.54/−0.62/−0.74，超线 0.04~0.24 dB，归因混合稀释，yK2 最易点 −1.2~−1.6 dB 为主因）；specialist 跨 regime 迁移代价 1.4~2.6 dB vs unified ≤0.74 dB（核心卖点成立）

### Phase 5 — XYZ / XYT Extension 🔵（XYZ 已收口 EXP-026，2026-10-03；XYT/noise_stress 待推进）
- [x] XYZ：full spatial undersampling 鲁棒性（xyz2 regime：cross 网格 + 平面跳采，6 点；SXYZ vs U15 15 点统一）
  - Extension gate **未达**：U15−SXYZ = −1.15/−0.61/−0.83（3/3 超线，混合稀释归因；U15 仍达 XYZ 34.5~35.0 dB 三 regime 全覆盖）
  - Regression gate **达标**：vs EXP-025 U_noAZ，Y −0.53/−0.39/−0.57、YZ −0.24/−0.95/−0.09
  - Zero-shot 崩溃（7.8~15.1 dB vs 专训 35.4）→ 新 regime 必须进训练集；SXYZ 跨 regime 掉 11+ dB（特化最重）
- [ ] XYT：temporal generalization + temporal metrics（数据依赖待确认，DEC-025 修正项，维持延后）
- [ ] noise_stress（F1/F2/F3 配对干净 GT）——需预注册，复用 U15/U_noAZ checkpoint
- [ ] 候选：regime 加权采样修复 Extension gate（EXP-026 结果6）

### Phase 6 — Protocol A / Real Acquisition ⬜
- [ ] hardware-realistic sparse acquisition（pollen 等）
- [ ] measured acquisition time / exposure / dose proxy
- [ ] quality–speed trade-off 曲线
- Final gate：算法指标 + 实际 acquisition efficiency 完整闭环

### 计划内稳健性补充（并入 Phase 4/5 执行，v4 方案 §11-13）
- [ ] noise / photon budget（CIDC25 自带噪声 + F1-F3 noise_stress）
- [ ] Wiener λ 敏感性（v4 先验为 fill-first 家族，λ 语义不同于 v3 冻结λ）
- [ ] no-PSF 消融线沿用（v3 EXP-021 结论：Wiener 增益主体来自 PSF 退卷积物理）
- [ ] PSF mismatch（σ 标定误差）

## 3. 交付物与论文图（v4 方案 §18-19）

主图规划：Fig1 总概念 / Fig2 physics prior / Fig3 Y-sparse / Fig4 YZ-sparse /
**Fig5 fixed-budget sampling allocation（核心创新）** / Fig6 unified model / Fig7 XYT 扩展 / Fig8 真实采集效率。
已有素材：EXP-022/024 全部 run 的 config + 双口径指标 json（runs_v3/）；
v3 期可视化协议（DEC-020/021，visualize_v3.py）可复用于 v4 结果制图。

## 4. 风险与注意（沿用 v3 §6，v4 修订）

1. 每次实验先预注册（EXP 条目）再启动；判据写在预注册里，收口时如实报告（含否证）。
2. 3D Wiener 内存：解析可分离 OTF + FFT 逐 batch，未成瓶颈。
3. 与 legacy 结果不做同表对比；PSRAS 内部对比只用 v4 口径。
4. 每次改动在 PROJECT_LOG.md 追加记录（原理→改动→结论）。
5. 单种子结果如实标注（如 (8,2) 主矩阵仅 seed0）；主张级结论须 3/3 种子同号。

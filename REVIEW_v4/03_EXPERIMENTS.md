# 实验索引（EXP-001 ~ EXP-028）

编号是**追加**的，不是时间序（例：EXP-021/020 的条目写在 EXP-028 之后）。完整条目
（原理、设计、结果、判读、执行记录）在 `reports/PROJECT_LOG.md`，本表只做导航。

阶段标注：v3 = 协议 v3（2026-09-09 ~ 09-21）；v4 = 协议 v4（2026-09-22 之后）。

| EXP | 阶段 | 内容 | 规模 | 结论 | 状态 |
|---|---|---|---|---|---|
| 001 | v3 | 协议 v3 审阅核实 | — | 立项 | 关闭 |
| 002 | v3 | physics/data/losses/attention 模块 + 单测 | 45 tests | 载体就绪 | 关闭 |
| 003 | v3 | CIDC25 数据准备 + 标定裁剪审计 | 8 栈 | split v1 建立 | 关闭 |
| 004 | v3 | dataset/Swin-UNet/prior fusion/train.py + smoke | 2 run | B0/B3 首批 | 关闭 |
| 005 | v3 | B0 vs B3 首批（XY K=4） | 消融矩阵 | 首批正增益 | 关闭 |
| 006 | v3 | val 协议 leak 修复后重评测 | — | val 协议校正（DEC-011） | 关闭 |
| 007 | v3 | xy-grid K=4 消融矩阵判定 | 7 arm | xy 增益结构厘清 | 关闭 |
| 008 | v3 | y-K8 三点对照 | 3 arm | 横向先验增益≈0 | 关闭 |
| 009 | v3 | 2.5D z 模式物理先验 | — | B3 +1.77 dB，轴向才是关键 | 关闭 |
| 010 | v3 | Phase 2 三协议最终判定表 | 8k 预算 | z 情境成立、xy 不成立 | 关闭 |
| 011 | v3 | λ Wiener 扫描 + X0 基线缺失 | λ 网格 | 发现 λ 跨情境迁移失败；X0 顺序机制 | 关闭 |
| 012 | v3 | U3D 统一 3D 网络（z-K4 / z-K8） | 4 run | 退出条件未通过 | 关闭（证伪） |
| 013 | v3 | 回退试点 K4（DEC-017） | 2 run | 未通过 → 多头输出诊断 | 关闭（证伪） |
| 014 | v3 | 两个回退试点逐位相同 | 2 run | 轴向线证伪（DEC-019） | 关闭（证伪） |
| 015 | v3 | Phase 4 severity 曲线（K2/K4/K8/K16 × B0-B4） | 全矩阵多 seed | fusion 在难端反转、易端冗余 | 关闭 |
| 016 | v3 | xy severity K∈{2,8} 补全 | 补点 | 横向瓶颈结论 | 关闭 |
| 017 | v3 | 轴向上下文缩放曲线 T-scaling | T∈{3,7,9,15} | T 越大越好，存在收益递减 | 关闭 |
| 018 | v3 | video3d 完整 3D 移窗复验 | 复验 | 条件①未通过，架构级证伪升级 | 关闭（证伪） |
| 019 | v3 | 光子预算剂量-响应（ph5–ph50） | 5 预算 × 2 arm | b3 崩溃、b4 鲁棒；崩溃主因裁定 | 关闭 |
| 020 | v3 | 外部原始仓库 × 两臂 × 三情境 | 22 run | vswin +0.50（2D）/ sunetr −0.10 / rvrt 坍塌 | 关闭 |
| 021 | v3 | M1 no-PSF 消融 + 2D b1 补跑 | 3 run | Wiener 增益主体来自 PSF 退卷积 | 关闭 |
| 022 | v4 | Phase 1 Y baseline（b1/b2/b3 × K{2,4,8,16} × 3 seed） | 36 run | Bilinear>Direct 全 K 同号；冻结 λ Wiener 崩坏 | 关闭 |
| 023 | v4 | Phase 2 YZ 先验独立评估（7 配方 × 6 点） | 1008 行 | P4 IterDC 与 P2 FillFirst 入选；P1/P3 淘汰 | 关闭 |
| 024 | v4 | Phase 3 YZ 主线（4 arm × 6 点 + fixed budget） | 40 run | 分配律成立；b4f ≥ b4i 全组合；+0.5~+3.4 dB | 关闭 |
| 025 | v4 | Phase 4 Unified（5 arm × 3 seed，9 点） | 15 run | A_Z 无增量 → UnifiedV4-noAZ；跨协议代价 ≤0.74 dB | 关闭（部分达标） |
| 026 | v4 | Phase 5 XYZ 扩展（15 点） | 6 run + 90 零样本 | Extension gate 未达；zero-shot 崩溃；跨迁移 11–24 dB | 关闭（gate 未达） |
| 027 | v4 | noise_stress 配对评估（F1/F2/F3，零训练） | 90 reval | 单调退化 + F1 天花板 26–28 dB + 网络噪声放大 | 关闭 |
| 028 | v4 | regime 加权重训（XYZ 1:1:2） | 3 run + 45 reval | Extension gate 仍 3/3 未达，share 假设证伪 | 关闭（假设证伪） |

## v4 主实验的 run 目录对应

| 实验 | 目录 | run 数 |
|---|---|---|
| EXP-022 | `runs_v3/Y_K{2,4,8,16}_{B1,B2,B3}_s{0,1,2}` | 36 |
| EXP-023 | `runs_v3/exp023_prior_sweep/`（`summary.csv` 1008 行） | 无训练 |
| EXP-024 | `runs_v3/YZ_{2x2,2x4,2x8,4x2,4x4,8x2}_{b1,b2,b4i,b4f}[_s1,_s2]` | 40 |
| EXP-025/026/028 | `runs_v3/UNIV4_{SY,SYZ,SYZ_noAZ,SXYZ,U,U_noAZ}_s{0,1,2}`、`UNIV4_U15_s{0,1,2}`、`UNIV4_U15W_s{0,1,2}` | 15 + 6 + 3 |
| EXP-027 | 上述 ckpt 上的 `metrics_noise_stress_F{1,2,3}_*.json` | 90 reval |

逐 run 指标见 `numbers/v4_runs.csv`（543 行，含 regime/mode/K/K2/preset/warm_start/
特征数/test 与 val PSNR、SSIM）。

## 关键决策（DEC，详见 PROJECT_LOG.md §4）

| DEC | 内容 |
|---|---|
| 007 | 稀疏观测在 dataset 运行时生成，不预生成 |
| 012 | 采样栅格锚定 stack 全局坐标；z 模式 T=2K−1 |
| 015 | `wiener_fill_first` 先插值再反卷积 |
| 019 | 轴向 3D 网络线关闭，回退 swin2d |
| 020/021 | 可视化协议；查询平面语义（XZ 截面必须是滑动查询推理） |
| 023 | 外部原始仓库基线对比 |
| 025 | **协议 v4 重置**：split v2、指标口径、legacy 归档 |
| 026 | 文档体系重构 |
| 027 | Phase 4+5 复盘 + 下一步决策框架（含 checkpoint-resume） |

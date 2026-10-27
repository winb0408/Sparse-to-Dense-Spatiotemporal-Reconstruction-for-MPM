# PSRAS v4 审查快照（2026-10-19）

本目录是给**外部评审**（GPT / 合作者 / 审稿人）准备的自包含快照，对应分支
`review-v4-20261019`，基线是 `PSRAS` 分支的 `4a70d9e`（2026-10-19）。

## 这是什么项目

三光子显微镜（3PM）的 **稀疏采样 → 稠密重建**。统一前向模型：

```
Y = M ⊙ (H * X + n)
```

`X` 是真值体数据，`H` 是 PSF 模糊，`M` 是采样掩码，`n` 是噪声。项目的核心不是
"再训一个去噪网络"，而是回答一个采样协议问题：**在给定采集预算下，稀疏度应该
放在哪个轴上、物理先验该怎么用、一个网络能否覆盖多种采样协议。**

## 当前状态（一句话）

Protocol v4 的 Phase 0–5 全部关闭，Phase 6（真实采集验证）尚未开始。四条核心主张
有 3-seed、val/test 双口径证据；**统一网络追平专训网络**这一条**未达标且已如实
披露**（差 0.6–1.2 dB，容差 0.5 dB），并已通过 EXP-028 排除了"采样权重不足"这一
解释。

## 怎么读

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `01_PROGRESS.md` | 进展评估：阶段完成度、里程碑、剩余工作、风险 |
| 2 | `02_CLAIMS_EVIDENCE.md` | **核心**：5 条创新点 × 定量证据 × 定性图 × 局限 |
| 3 | `03_EXPERIMENTS.md` | EXP-001~028 全表：规模、结论、证据路径 |
| 4 | `04_VERSION_HISTORY.md` | v1→v4 迭代史、分支/目录对应、已证伪方向 |
| 5 | `05_NEGATIVE_RESULTS.md` | 否证清单（写论文时的"不可主张"边界） |
| 6 | `06_REPRODUCIBILITY.md` | 数据、口径、命令、代码地图、已知缺陷 |
| 7 | `07_REVIEW_QUESTIONS.md` | **给评审者的具体问题清单**（最需要被挑刺的地方） |
| 附 | `figures/` | 全部论文级图（PNG 300dpi + PDF），含 EXP-027/028 |
| 附 | `numbers/` | `v4_runs.csv`（543 行逐 run 指标）、`key_numbers.csv` |
| 附 | `reports/` | PLAN、PROJECT_LOG（1942 行完整证据树）、进度表、外部审查原文 |

## 三张最能说明问题的图

- `figures/fig1_alloc_law.png` — 同样采样预算下，`(k_y,k_z)=(2,8)` 比 `(4,4)` 高
  3.8 dB、比 `(8,2)` 高 8.1 dB（4 个 arm 排序一致）
- `figures/fig4_unified_cross_transfer.png` — 专训模型跨协议崩溃（13–24 dB），统一
  模型三协议稳定在 33.4–34.6 dB；关掉轴向先验后 XYZ 掉到 9.9 dB
- `figures/qual_yz_2x8.png` / `figures/qual_yz_2x8_err.png` — 同一查询平面，Direct /
  Bilinear / IterDC / FillFirst-Wiener 的重建与绝对误差

## 代码在哪

代码没有复制进本目录，仍在 `PSRAS/`（同一分支内可读）：

```
PSRAS/physics/      采集仿真、PSF、Wiener 先验家族（fill-first / iterdc / adaptive）
PSRAS/data/         数据集、观测构造、掩码
PSRAS/models/       Swin-UNet 2D 主干、UnifiedV4、门控融合
PSRAS/scripts/      train.py / reval.py / 各类可视化与队列脚本
PSRAS/runs_v3/      243 个 run 目录（config + 指标 json；权重 *.pt 未入库，17 GB 在本地）
PSRAS/eval_viz/     全部图；paper_figures/ 是本次交付的论文级图集
```

## 复现入口

```bash
cd PSRAS
export TMPDIR=/data2/wjb/tmp
python scripts/train.py   --preset b4f --regime 2.5d --mode yz2 --K 2 --kz 8 --steps 8000
python scripts/reval.py   --run runs_v3/YZ_2x8_b4f --split test
python scripts/paper_figs.py quant        # 定量图（无需 GPU）
python scripts/paper_figs.py qual_yz      # 定性图（GPU）
```

细节见 `06_REPRODUCIBILITY.md`。

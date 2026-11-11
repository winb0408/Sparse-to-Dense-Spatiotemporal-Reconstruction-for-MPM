# 本分支的结果快照（E020 队列 / E021 阶梯 / E023 手册执行）

这个仓库的正常规则是**结果不入库**（`.gitignore` 里 `runs/` 整目录忽略）。这一批是
强行加入的只读快照，为了在没有数据、没有 GPU 的机器上也能审阅：所有数字都能在
`runs/E021_ladder/` 里回查到产生它的 `summary.json`，不用相信任何一处抄写。

`docs/PHASE1_SUPERVISED_AUDIT_AND_RUNBOOK.md` 的 §4.1–§4.5 已全部执行完毕；执行记录、
踩到的坑和被推翻的旧数字见 [`Project log/Project_log.md`](../Project%20log/Project_log.md)
的 SPUR-E023 一节。

## 该看什么

| 想看什么 | 文件 |
|---|---|
| 主表：52 个 cell × 10 个指标 + 归因 + 预算曲线 | [`runs/E021_ladder/ladder.md`](../runs/E021_ladder/ladder.md) |
| 同一批数字的机器可读版 | `runs/E021_ladder/ladder.csv`、`ladder.json` |
| **等上下文 25 % 换轴表**（`y_k4_c7 / z_k4 / yz_k2x2_c7`，七臂齐全） | `ladder.md` 末尾 "Axis at a matched 25 % budget, one seven-plane context" |
| 参数量 / MACs / 延迟（含 S1g、S2g；A0/A1 的算子是实测值而非 `n/a`） | [`runs/E021_ladder/complexity.md`](../runs/E021_ladder/complexity.md)、`benchmark_supervised.json` |
| **整卷端到端真实耗时**（203×512×512，A0 2.85 s / S1 41.6 s） | `runs/E023_phase1/e2e/benchmark_end_to_end.json` |
| **整卷 3D 血管结构**（Dice / clDice / skelF1 + 计分层占比） | `runs/E021_ladder/E021_S1_y_k2/structure/structure_summary.md`；用 `--from-summary` 可从 JSON 重排版 |
| 固定视图的三协议对照（§4.4.1 D） | `runs/E021_ladder/figures_fixed/M2_FOVII_S1_vessels/compare_y_z_yz_25pct.png` |
| 同一批图的**逐像素一致性报告** | 同级 `consistency_report.json`（`target_diff = fixed_view_diff = input_diff = 0`） |
| 冻结的视图定义（切面、物理间距、统一窗宽、数据/配置指纹） | 同级 `view_spec.json` |
| 单协议定性图（输入 / A0 / S1 / 密集真值 + 误差 + 支撑 + 深度剖面） | `figures_fixed/M2_FOVII_S1_vessels/qualitative_*_{y_k4,z_k4,yz_k2x2}.png` |
| 五臂同输入对比（A0 / S0 / S1 / S2 / S2g，`y_k2`），**留出动物 M2 的全部四个体**（含最差卷） | `runs/E021_ladder/figures/qual_y_k2/` |
| 稀疏采集构建自检图（掩膜、2D 阶梯、覆盖率） | `spur_mpm/figures/sparse/` |
| **手册 §7 Fig. 2 网络管线 + 消融矩阵**（每一格都从 `models/registry.ARMS` 生成，每格旁边写明取自哪个字段；`n/a` 只出现在真为空的字段） | `spur_mpm/figures/framework/fig2_pipeline.png`、`fig2_ablation_matrix.png`；逐格来源审计在 `fig2_provenance.json`；`python -m spur_mpm.scripts.framework_figure` 重绘 |
| **手册 §7 Fig. 5 3D 掩膜/骨架 + 局部分叉 + 断裂/伪连接标记**（四卷 × 两臂，含跨卷汇总；冻结裁剪框存 JSON，重绘不跳位） | `runs/E021_ladder/figures/structure/structure_{S1,A1}_M2_*_vessels.png`、`structure_cross_volume_{S1,A1}.png`；`python -m spur_mpm.scripts.structure_figure` 重绘 |
| **手册 §7 Fig. 4 性能–稀疏曲线**（Y/Z/YZ 三族 Dice★ vs 实测比例，虚线为无学习下限；同 25 % 预算三轴对照，含等上下文 `_c7` 版） | `runs/E021_ladder/figures/curves/curve_{y,z,yz}.png`、`curve_matched_25pct{,_c7}.png`；`python -m spur_mpm.scripts.ladder_curves` 从 `ladder.csv` 重绘 |
| 每张定性图的逐臂数字（PSNR / Dice / Dice★）与几何 | 同级 `*.json` |
| 每个 cell 的完整评估（逐窗 `samples.csv` + 汇总 `summary.json`） | `runs/E021_ladder/E021_*/eval_test/` |
| 每个 cell 的实际配置（稀疏参数、上下文层数、训练超参） | `runs/E021_ladder/E021_*/config.yaml` |
| 代码地图与三个硬性图形约束 | [`docs/CODE_MAP.md`](CODE_MAP.md) |
| **手册验收记录**（改了什么 / 怎么复现 / 测试输出 / 图在哪 / 前后差别） | [`docs/PHASE1_ACCEPTANCE_REPORT.md`](PHASE1_ACCEPTANCE_REPORT.md) |
| **PR #1 审查意见逐条回复**（每条给状态 + 证据路径；未完成的写清代价） | [`docs/PHASE1_AUDIT_RESPONSE.md`](PHASE1_AUDIT_RESPONSE.md) |
| **骨架容差敏感性**（同一对掩膜、物理口径重算；解释"1 体素"在 Z 方向等于 5 µm） | `runs/E021_ladder/tolerance_sweep/tolerance_sweep.{md,json}`；`python -m spur_mpm.scripts.tolerance_sweep` 重算 |

## 一句话结论

- **可主张**：Y 方向行稀疏在 25 % / 12.5 % 预算上，监督重建的血管结构优于双线性
  （等面积 Dice★ +0.040 / +0.059，clDice★ +0.067 / +0.097）；灰度 PSNRmiss 全部九级涨 1.0–1.5 dB。
- **可主张**：插值的质量依赖稀疏轴（A0 三轴极差 0.0405 Dice），监督臂把三轴拉平到
  0.015–0.020——收益是"消除轴间差异"，不是"某个轴更好"。
- **可主张**：新增的门控融合臂 S2g（X0 + U + Gate）**没有**可测量增益（vs S1g +0.08 dB /
  +0.0001 Dice），却把 MACs 抬 17 %、延迟抬 6 %。这是本轮唯一一条"成本涨、收益没到"的净结论。
- **不可主张**：Z 轴与 Y+Z 轴的结构增益；50 % / 33 % 预算上的结构增益；三个创新点
  （X0 / U / 门控融合）的任何独立贡献；任何"某个稀疏轴更好"。
- **不可主张（容易误主张）**：**整卷 3D 结构上 S1 优于插值地板**。A1（零参数，单位 PSF
  下等于 bilinear）与 S1 的整卷差是 Dice +0.024 / clDice +0.030 / skelF1 +0.021，四卷符号
  一致但**都压在地板上**。过门槛的是等面积 tile 的 ★ 指标（+0.040 / +0.059），两个层次
  不能互相背书。
- **可主张（量级明显，Dice 看不见）**：拓扑分量的方向相反——插值把血管打碎到参考的
  1.1–1.7 倍，S1 合并到 0.53–0.71 倍。两种错误性质不同，别用 Dice 排序。
- **骨架分数的容差依赖**（`runs/E021_ladder/tolerance_sweep/`）：表的
  `tolerance_vox=1` 是各向同性的，而体素是 Z 5 µm / XY 0.98 µm，所以它实际宽容了纵深
  5 µm 的错位；换成物理口径后同一个"1"变成 0.98 µm，S1 的骨架 F1 从 0.7373 掉到 0.6951。
  S1 相对地板的优势也随容差放宽塌缩：0 µm +0.066 → 1 µm +0.022 → 10 µm +0.004。**引用
  骨架 F1 时必须写明容差及其口径**，否则同一份数据可以读出"明显更准"或"基本一样"。
- 门槛：跨动物方差约 2.8 dB / 0.08 Dice，所以 < 1 dB 与 < 0.03 Dice 的差异不写结论。

## 引用整卷结构分数时必须一起引用的东西

§4.3 的血管门控会剔掉参考前景占比 > 15 % 的层，实测占比：

| volume | 被门控层 / 总层 | 参与评分 |
|---|---|---|
| M2_FOVII_S1 | 57/203 | 72 % |
| M2_FOVII_S2 | 68/202 | 66 % |
| M2_FOVI_S1 | 82/201 | 59 % |
| M2_FOVI_S2 | 101/152 | **34 %** |

均值 Dice 0.8634 / clDice 0.7725 / skelF1 0.7373 是 **S1 在"幸存层上的分数"**，不是整卷分数；
四卷占比差一倍，跨卷比较也因此受限。`structure_summary.md` 现在逐卷打印
`gated/total (scored %)`。被门控清空的 slab 原实现记 `dice = 1.0`（两个空掩膜"完全一致"），
已改为 NaN，否则对 slab 求平均等于给最难深度发奖励。两张已存档的 summary 的
`depth_profile` 是在规则改变前算的，已用 `runs/E023_phase1/rescore_depth_profile.py`
从**导出的同一份 TIFF 掩膜**重算（不重跑推理），并重写了 `structure_summary.md`。

## 没有入库的东西

| 没有 | 大小 | 怎么重建 |
|---|---|---|
| 检查点 `best.pt` / `last.pt` / `resume.pt` | 每 cell ≈ 10 / 30 / 30 MB | `python -m spur_mpm.scripts.run_ladder --stage …`（见 CODE_MAP §6） |
| 定性图的 float32 `.npz` 体数据转储 | 单目录 ≈ 370 MB | `python -m spur_mpm.scripts.qualitative_figures …` |
| Fig. 5 的骨架缓存 `figures/structure/cache/*.npz` | ≈ 2 MB | 被 `*.npz` 规则排除；`python -m spur_mpm.scripts.structure_figure` 从导出的 TIFF 掩膜重算（图与 `crop_*.json` 已入库） |
| `figures_fixed/M2_FOVII_S1_vessels_z101/`（z=101 旧图，该层在 2/3 协议里被跳过） | ≈ 370 MB | 保留在本地做对照，未推送 |
| Dryad 3PM 原始数据与缓存 | 数十 GB | 见 `spur_mpm/scripts/prepare_data.py` 与日志 E000 |

## 环境

conda env `threePM`，python 3.10，PyTorch。测试：`python -m pytest spur_mpm/tests -o addopts="" -q`
（**390 项，全绿**；日志 `docs/evidence/pytest_20261111_T0_T2_guard.log`。`-o addopts=""` 是必需的：
`pytest.ini` 已经带 `-q`，命令行再加一个 `-q` 会变成 `-qq`、把 `390 passed` 那行一起吞掉，
只留下点阵。上一轮为 245 项，
本轮新增 `test_physics_prior_contract.py` 30 项、`test_dual_branch_contract.py` 38 项、
`test_archive_safety.py` 21 项、`test_psf_estimate.py` 16 项、
`test_run_e024_guard.py` 12 项，其余为 `test_framework_figure.py` 等既有文件的扩充）。

## PR2 物理先验 / 双分支：本文件不含新结果

本文档描述的是 **E021/E023 那一代（`A*/S*/SS*`）** 的数字。PR2 新方法代（`P2_*`）
**没有任何真实数据结果**：四张 GPU 全被其他用户占用，训练 `NOT_RUN`。

已有的是协议层证据，不是方法层结论：

| 项 | 结果 | 证据 |
|---|---|---|
| identity PSF 下 `P2_W0` 与 `P2_I0` | 真实测试卷上**逐位相同**（只有 `"arm"` 字段不同） | `docs/evidence/smoke_floors_20261110/` |
| 非恒等 H（人为额外加 blur）再反卷积 | PSNR 28.03 → 27.55，SSIM 0.750 → 0.629，**变差** | 同上 |
| 源数据是否含 PSF 标定 | **不含**：README 把成像参数指向正文 Methods，TIFF 头只有 `spacing=5.0`，spec 无 NA | `docs/PHASE1_PR2_PSF_ESTIMATE.md` §1 |
| 衍射极限算术（1340 nm 三光子） | σ_xy 0.16–0.21 体素、σ_z 0.11–0.22 体素——**比体素还细** | 同上 §2 |
| 留出真实测量行的最优反卷积宽度 | **两轴都是 σ=0**，9 档横向 / 6 档轴向全部更差（0.2 µm 即 −0.17 dB） | `docs/evidence/psf_estimate_20261110/` |

第二、第四条**不能**读作"Wiener 有害"：真实 stack 已带自身光学，再按一个不存在的
模糊去卷积，等于去掉不存在的模糊。它们能读作的是——本队列**没有可去除的光学模糊**，
所以 identity 是测出来的结论而非默认假设；而"Wiener 在本队列有独立增益"这句话，
现在有可复现的否定证据。方法口径见 [`docs/PHASE1_PR2_METHOD_SPEC.md`](PHASE1_PR2_METHOD_SPEC.md) §2，
估计过程见 [`docs/PHASE1_PR2_PSF_ESTIMATE.md`](PHASE1_PR2_PSF_ESTIMATE.md)，
逐条状态见 [`docs/PHASE1_PR2_ACCEPTANCE.md`](PHASE1_PR2_ACCEPTANCE.md)。

**新旧数字不可同列排名**：`A1` 是 `wiener_fill_first`（Z 未填洞），`P2_W0` 是
`wiener_fill_all`；`S1` 的 `X0` 只是拼接通道，`P2_ResidualI` 是残差锚点。

## 两个必须知道的口径陷阱

1. **`Dice` 与 `Dice★` 方向相反。** 参考阈值拟合出的 `Dice` 与等面积重取阈的 `Dice★` 在
   同一批预测上方向相反；差别全部来自预测的对比度漂移，不是结构差异。看表以 ★ 列为准，
   但 ★ 必须与不带星的列并读（`metrics/vessel.py::match_foreground` 的 docstring 记了它
   自己的局限，以及一个随机重叠基线 ≈ 0.19 作为量级自检）。
2. **cell 名会随上下文变。** `cell_name(arm, level, context)` 在"强制上下文等于层级自身
   上下文"时返回**不带** `_cN` 的裸名。对自身上下文 ≠ 7 的层级加 `--context-slices 7`
   重打分，会去找一个从未训练过的 `_c7` 兄弟目录、报 `no_checkpoint` 并把**修复前的旧
   分数**留在表里。全量重打分不要带 `--context-slices`（见
   `runs/E023_phase1/rescore_own_context.sh` 头部的说明）。

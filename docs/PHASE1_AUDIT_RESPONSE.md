# PR #1 审查意见逐条回复

对应 PR：[#1 \[Review\] Phase 1 supervised pipeline fixes + fixed-GT Y/Z/YZ visualization validation](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/pull/1)（head `review/phase1-supervised-fixes-20261103` → base `PUR-MPM`，已合并）。
本文件是逐条答复，不是新的审查意见。凡写"已完成"的，后面都给了可复现的产物路径或测试名；
凡写"未完成"的，都写明原因与所需代价，不做模糊处理。

状态口径：**已完成** = 代码 + 测试 + 真实数据产物都在仓里可查；**部分完成** = 代码/测试在，真实数据只覆盖一部分；
**未完成** = 没做。

---

## A. PR 正文"修复与新增"七项

| # | 审查项 | 状态 | 证据 |
|---|---|---|---|
| A1 | Z 边界窗口落在未训练输出通道 | 已完成 | `data/datasets.py` 改为中心对齐 + 虚拟 Z padding，越界层标为未测量；`tests/test_pipeline_smoke.py::test_z_boundary_windows_always_query_trained_centre[y-2-1|z-1-2|yz-2-2]` 三种协议各一条 |
| A2 | Direct S0 缺 `x0` 特征报错 | 已完成 | Dataset 显式返回 `warmstart_center`，`evaluate.py` / `inference.py` / `qualitative_figures.py` 共用；`test_pipeline_smoke.py::test_direct_arm_item_exposes_warmstart_without_x0_feature`；真实数据上 A0 的整卷 4 卷结构报告已跑通（见 U6） |
| A3 | Mask dtype / 形状 / PSF 三处严谨性 | 已完成 | `test_sparse_build.py::test_verifier_rejects_numeric_mask_even_if_values_are_zero_one`、`::test_verifier_reports_shape_mismatch_without_indexing_crash`、`::test_verifier_uses_optically_degraded_measurement_reference` |
| A4 | 缺完整 U+Gate 臂（S2g） | 已完成 | `models/registry.py` 注册 S2g，`test_models_metrics.py::test_model_forward_shape[S2g]`；真实训练与评估见 U5；Fig. 2 的消融矩阵逐格取自 registry（`figures/framework/fig2_provenance.json`） |
| A5 | 微米半径用几何平均像素间距 | 已完成 | `metrics/skeleton.py` 改用 `distance_transform_edt(sampling=(z_um, xy_um, xy_um))`；`test_skeleton.py::test_mean_radius_um_uses_anisotropic_distance_not_voxel_geometric_mean` |
| A6 | 无空闲 GPU 时静默抢 GPU0 | **部分完成** | `scripts/run_ladder.py::_free_gpus()` 已实现并作为 `--devices` 的默认来源，无空闲卡时 `SystemExit`；**但没有单测**——该函数直接调用 `nvidia-smi`，在无 GPU 的 CI 上无法稳定断言。要么给 `subprocess.run` 注入点，要么保持现状并在手册里注明"这一条只有人工核对" |
| A7 | 执行手册 | 已完成 | `docs/PHASE1_SUPERVISED_AUDIT_AND_RUNBOOK.md`；其开头的"尚未在训练机器运行"声明已在本次更新中改为执行后的实际状态 |

## B. PR 正文"未完成（在服务器执行）"六项

| # | 审查项 | 状态 | 证据 |
|---|---|---|---|
| B1 | `pytest spur_mpm/tests -q` | **已完成** | 245 项全绿（`python -m pytest -q`，最后一条 `[100%]` 无 F/E）。当年 PR 写的时候是 207 项 |
| B2 | 90 项 3D 采样构建 | **已完成** | `data_local/sparse/cohort3d/manifest.json` 与 `verification.json` 均为 90 条（10 卷 × 9 个等级）；`runs/E023_phase1/build3d.out` 结尾："all constructions verified: Y = M . D exactly on the support, 0.0 off it, kept rows full width, lattice as declared" |
| B3 | 按协议交叉核验 Dataset 窗 Mask 与构建器 Mask | **已完成** | `data_local/sparse/mask_crosscheck.json`；`runs/E023_phase1/crosscheck.out`：`y_k2/y_k4/z_k3/yz_k2x2` 各 270 个窗口 `ok`，"the training input equals the verified construction at every window checked" |
| B4 | 重评已训练权重（代码修复之后） | **部分完成** | 主表整体换成 E021 重训 + 重评：`runs/E023_phase1/rescore_all.out` 记录 38 个 cell 重打分（`stages=['ablation','sweep','matched','floors','control']`）。**但"拿 PR 之前的 E002/E009 旧 checkpoint 在新代码上再跑一遍"没有做**——旧数字已从 `docs/RESULTS.md`、`docs/PHASE1_ACCEPTANCE_REPORT.md` 中撤下，不再作为任何结论来源（两文件里已无 E002/E009 引用）。若审阅方坚持要看"只改边界、不改结论"的对照，这件事只需 eval 不需重训，可以补 |
| B5 | 固定队列上的有监督消融 | **已完成** | `runs/E021_ladder/`：ablation / matched（7 层 Z 上下文）/ sweep / floors / control 五组，38 个 cell，主表 `runs/E021_ladder/ladder_report.md` + `.csv`；采样密度曲线 `runs/E021_ladder/figures/curves/curve_{y,z,yz,matched_25pct{c7}}.png` |
| B6 | 整卷 ImageJ 结构报告 | **已完成（人工核验除外）** | `runs/E021_ladder/E021_{S1,A1,A0}_y_k2/structure/`：四卷 × `structure_summary.{json,md}` + `*_pred_mask.tif` / `*_ref_mask.tif`（就是算指标的同一对掩膜）；图 `runs/E021_ladder/figures/structure/`（8 张单卷 + 2 张跨卷）。**ImageJ 人工抽检仍待人工完成**，见 D3 |

## C. §4.4.1 P1 定性图五项（"必须完成"）

| # | 审查项 | 状态 | 证据 |
|---|---|---|---|
| C1 | 单一 `view_spec.json` + CLI 复用，`_view_spec` 不受 `M` 影响 | 已完成 | `scripts/viewspec.py`；产物 `runs/E021_ladder/figures_fixed/M2_FOVII_S1_vessels/view_spec.json`（含 `fingerprint`：数据文件名 + 形状 + 体素 + 窗口 + ROI）。测试 `test_qualitative_figures.py::test_a_reference_that_depends_on_the_protocol_is_a_hard_failure`、`::test_a_saved_spec_round_trips_and_is_validated` |
| C2 | 标签按实际 Mask 生成（Y 只报行、Z 只报层、YZ 两者），"XY 25/50/75%"是深度百分位 | 已完成 | `support_*.png` 的脚注改成 `z: every plane / N/M planes kept`（原"z never subsampled"的错误标签已删）；测试 `::test_z_only_protocol_reports_planes_and_never_rows`、`::test_y_only_protocol_reports_rows_and_never_planes`、`::test_yz_protocol_labels_both_axes_with_the_mask`、`::test_the_report_reads_the_protocol_spec_not_the_volume_tag` |
| C3 | 同列跨协议 GT 完全一致；统一归一化；禁止逐模型调对比度 | 已完成 | `consistency_report.json`：`target_diff = 0.0`、`view_diff = 0.0`、`input_diff = 0.0`，三协议 `shape_mismatch=false / view_mismatch=false`；显示窗取自参考而非预测，测试 `::test_display_windows_come_from_the_reference_not_from_the_prediction`、`::test_frozen_windows_are_the_ones_this_reference_derives` |
| C4 | `tests/test_qualitative_figures.py` + `figures_fixed/` + `consistency_report.json` + 失败非零退出 | 已完成 | 该测试文件 18 条；产物 `figures_fixed/M2_FOVII_S1_vessels/compare_y_z_yz_25pct.png`（25 % 同预算 y_k4 / z_k4 / yz_k2x2 三联）；不一致时抛 `RuntimeError` → 非零退出 |
| C5 | 复用 Z 边界修复与同输入路径；不得把中心 patch 当整卷；检查 `sparse_figures.py` 与 support 图 | **部分完成** | 复用已在（同一 `inference.py` / `dataset` 路径）；`sparse_figures.py` 已按等级名而非百分比命名。**但整卷代表性只做了 1 卷**：`figures_fixed/` 只有 `M2_FOVII_S1_vessels` 的两个 z 位置（z100 / z101），另外三卷（`M2_FOVI_S1`、`M2_FOVII_S2`、`M2_FOVI_S2`）没有按同一冻结 spec 出图。§4.4 要求"正式结果同时提供所有可用测试卷与失败案例"，这条没满足 |

## D. runbook §6"尚未关闭的风险"的现状

| # | 风险 | 现状 |
|---|---|---|
| D1 | 真实服务器必跑（pytest / build / crosscheck / 各臂正向推理） | 已跑完，见 B1–B3、B5 |
| D2 | 主文口径：整卷 3D 结构 + 固定参考分割 + 门控剔除层 + 物理尺寸 | 已落实：结构表用整卷 3D、阈值只在参考上拟合、逐卷报剔除层数（本队列 28 % / 34 % / 41 % / 66 %）、半径用各向异性 EDT。**骨架容差的物理敏感性已补做**：`runs/E021_ladder/tolerance_sweep/`。结论是"1 vox"比"1 µm"白送 0.042 分（S1 0.7373 vs 0.6951），差额来自 Z 方向 5 µm；S1 对地板的优势从 0 µm 的 +0.066 塌缩到 10 µm 的 +0.004 |
| D3 | ImageJ/Fiji 人工检查连续段、断裂、伪连通、深度变化 | **未完成（人工）**。TIFF 已备好：`runs/E021_ladder/E021_{S1,A1,A0}_y_k2/structure/A0|S1|A1_<volume>_{pred,ref}_mask.tif` |
| D4 | 缺人工解剖 GT | 未解决，且不打算用自动参考冒充解剖真值；论文口径需限定为"同一自动分割规则下的一致性" |
| D5 | 只有一只留出动物 M2，且被先前折结果影响过 | 未解决。取数纪律已写入：跨鼠方差 ≈2.8 dB PSNR / 0.08 Dice，**小于 1 dB / 0.03 Dice 的差异不声称** |
| D6 | 恒等 PSF 下 X0 应叫 interpolation-derived prior | 已执行：`Results`、`Project log`、Fig. 2 的 warm-start 文本均按 registry 的 `warm_start` 字段生成，不再出现"Wiener 引导"式表述；A1 ≡ A0 已作为数学恒等式写入并有测试 `test_physics.py::test_the_two_baseline_arms_are_the_same_method_under_an_identity_psf` |

## E. §7 预期图件的完成度

| 图 | 状态 | 产物 |
|---|---|---|
| Fig. 1 几何/动物划分 + 稀疏 Mask/预算 | 素材齐、**未合成单张** | `spur_mpm/figures/sparse/sparse_{1_construction,2_ladder_2d,3_coverage}.png`、`qualitative/sampling_diagram.png` |
| Fig. 2 流程图 + 消融矩阵 | 已完成 | `spur_mpm/figures/framework/fig2_{pipeline,ablation_matrix}.png` + `fig2_provenance.json`（逐格记录取自 registry 哪个字段） |
| Fig. 3 Y 稀疏下各消融的 XY/XZ/YZ | **部分** | `runs/E021_ladder/figures/qual_y_k2/`：四卷 × A0/S0/S1/S2/S2g。统一色标已做；但只有 `y_k2` 一档，且是窗口而非整卷 |
| Fig. 4 性能–稀疏程度曲线 | 已完成 | `runs/E021_ladder/figures/curves/curve_{y,z,yz,matched_25pct,matched_25pct_c7}.png` |
| Fig. 5 3D 掩膜/骨架、分叉、断裂/伪连接 | 已完成 | `runs/E021_ladder/figures/structure/`（8 张单卷 + 2 张跨卷）+ 冻结裁剪 `crop_*.json` |
| Fig. S1 参数量/MACs/耗时、背景误差、门控审计、全卷结果 | **部分** | `runs/E021_ladder/complexity.json`、`runs/E023_phase1/benchmark*.out`、门控层数在各 `structure_summary.json`；**未合成单张补充图** |

---

## 还差什么（按优先级）

1. **D3 ImageJ 人工抽检**——只能人工做，脚本侧已把 TIFF 与统计备齐。
2. **C5 另外三卷的冻结视角对比图**——同一 `view_spec` 复用即可，代价是 3 卷 × 3 等级的推理；这是 §4.4.1 明文要求却没满足的一条。
3. **B4 旧 checkpoint 重评对照**——脚本已写好并在跑（`spur_mpm/experiments/reval_pre_fix_checkpoints.sh`）：E002 S0/S1 × 3 折 + E009 A0/S1 × 3 折，结果写 `runs/E023_phase1/reval/`、不覆盖历史目录。已出的第一条 `E002_S0/fold0` 与 10-28 修复前的存档**逐位相同**（PSNR 30.400656756232767）。
4. ~~D2 骨架容差的物理敏感性~~ **已关闭**：`runs/E021_ladder/tolerance_sweep/`，脚本 `scripts/tolerance_sweep.py`，13 条测试。
5. ~~A6 GPU 选择函数的单测~~ **已关闭**：`tests/test_run_ladder_gpu.py` 9 条；同时修掉"uuid 映射不到时不排除任何卡"的漏洞。
6. **Fig. 1 / Fig. S1 的合成**、Fig. 3 扩到四卷与更多稀疏档。

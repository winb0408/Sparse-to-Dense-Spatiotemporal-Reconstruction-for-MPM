# Phase 1 PR2 验收清单（第一轮：T0–T2）

> 对应 `docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md` §7。状态取值：
> `TODO` / `CODE_DONE` / `UNIT_TESTED` / `GPU_VALIDATED` / `ACCEPTED` / `BLOCKED`。
> **凡是没实际运行的，状态一律不是 `ACCEPTED`。**
> 本轮日期：2026-11-10。分支 `feature/phase1-dual-prior-20261110`，基线 `ed060f0`。

## 一句话状态

代码与单测已完成并通过（390 项全绿，`docs/evidence/pytest_20261111_T0_T2_guard.log`）；
**全部 GPU 训练与真实数据头条结果 `NOT_RUN`**——四张 A6000 都被其他用户的作业占用，
带别人作业的卡不是基准设备。零参数地板已在真实数据上做过 CPU 冒烟（见 §3）。

---

## 1. T0 — 冻结历史、安全归档

| 项 | 状态 | 证据 |
|---|---|---|
| 从 `PUR-MPM` HEAD（`ed060f0`）建独立分支 | `ACCEPTED` | `git log`：`ed060f0` → `367e21c` |
| HEAD SHA / 数据 / 配置 / 权重索引快照 | `ACCEPTED` | `docs/archive/ARCHIVE_MANIFEST_20261110.md` |
| 旧 A0/A1/S0/S1/S2/S1g/S2g 与 E021/E023 只读保留 | `ACCEPTED` | 同上，§A 类 2 项 KEEP |
| 归档工具默认 dry-run，`--apply` 需 reviewed allowlist | `UNIT_TESTED` | `spur_mpm/scripts/audit_runs.py`、`tests/test_archive_safety.py`（21） |
| 本轮未删除任何文件 | `ACCEPTED` | `docs/archive/CLEANUP_REPORT_20261110.md`「本轮没有删除任何东西」 |
| 旧单测可重跑 | `ACCEPTED` | pytest 全绿 |

## 2. T1 — 物理初始化语义

| 项 | 状态 | 证据 |
|---|---|---|
| `I0 = Interp(Y,M)` 独立成公开算子 | `UNIT_TESTED` | `physics/wiener.py: interp_linear` |
| `W0 = Wiener(I0;H,λ)` 先填**所有**缺失轴再 FFT | `UNIT_TESTED` | `wiener_fill_all`；`test_axial_holes_are_filled_before_deconvolution` |
| identity ⇒ `W0 == I0` 逐位相等，不跑 FFT | `UNIT_TESTED` + 真实数据冒烟 | `test_identity_psf_gives_w0_equals_i0_exactly`；`docs/evidence/smoke_floors_20261110/` |
| `phys_delta = W0 - I0`，identity 下恒零 | `UNIT_TESTED` | 同上 |
| 2D-Y / 3D-Y / 3D-Z / 3D-YZ 四类掩膜 | `UNIT_TESTED` | `test_axial_holes_are_filled_before_deconvolution`（参数化） |
| 非恒等 H 必须标 `psf_source`，否则拒绝 | `UNIT_TESTED` | `test_an_unlabelled_non_identity_psf_is_refused` |
| 标签能穿过 config → dataset → prior | `UNIT_TESTED` | `test_the_item_config_carries_the_psf_source_to_the_prior` |
| 非法 λ / σ / 非有限值拒绝 | `UNIT_TESTED` | 4 条拒绝测试 |
| 旧 `wiener_fill_first` 语义不动 | `UNIT_TESTED` | `test_the_old_fill_first_mode_still_only_fills_lateral_holes` |
| 新 `mask_aware_inverse` 以 I0 为正则中心 | `UNIT_TESTED` | `test_mask_aware_inverse_holds_unmeasured_voxels_at_i0` |
| 旧 `iterative_dc_inverse` 不被暗改（仍向 0 收缩） | `UNIT_TESTED` | `test_the_old_iterative_inverse_is_unchanged_and_still_shrinks_toward_zero` |
| 真实数据上非恒等 H 数值有限 | 冒烟通过 | `P2_W0_blur_synthetic_summary_32px_cpu.json` |
| **源数据是否含 PSF 标定** | `ACCEPTED`（查证结论：**不含**） | `docs/PHASE1_PR2_PSF_ESTIMATE.md` §1：README 指向正文 Methods、TIFF 头只有 `spacing`、spec 无 NA |
| **自估 PSF：由假定 NA 反推衍射极限** | `UNIT_TESTED` + 真实数据 | `physics/psf_estimate.py::diffraction_limited_sigma`；σ 仅 0.16–0.21 / 0.11–0.22 体素 |
| **自估 PSF：留出测量预测扫描** | `UNIT_TESTED` + 真实数据 | `docs/evidence/psf_estimate_20261110/`；**两轴最优 σ=0**，非零宽度全部更差 |
| `estimated_from_data` 第三种标签存在且与 `calibrated_optical` 分开 | `UNIT_TESTED` | `test_an_estimated_psf_has_its_own_label_and_is_not_the_calibrated_one` |
| **Gate 1 的 phantom 数值稳定性（λ 扫描）** | `UNIT_TESTED` | 已知宽度 phantom 上扫 λ=1e-5…1e-1：有限且峰值有界；有模糊时优于插值；1 % 噪声下 λ=1e-4 的误差是插值基线的 7.7 倍。4 条断言 + 数字表见 `docs/PHASE1_PR2_PSF_ESTIMATE.md` §6 |

## 3. T2 — 双分支残差网络与训练器

| 项 | 状态 | 证据 |
|---|---|---|
| `DualPriorReconstructor` 新建，不改 `SwinUNet2D` | `UNIT_TESTED` | `models/dual_prior_reconstructor.py`；`test_the_old_backbone_is_still_the_old_class` |
| `P2_*` 注册，旧臂逐字段不变 | `UNIT_TESTED` | `test_the_old_arms_are_untouched`、`test_pr2_names_do_not_collide_with_the_old_ones` |
| 残差锚在 `I0`，初始化 `R ≡ 0` | `UNIT_TESTED` | `test_the_residual_starts_at_exactly_zero` |
| 两个分支都有梯度（第 0 步为零是刻意的） | `UNIT_TESTED` | `test_the_physics_branch_is_connected_and_learns_after_one_step` |
| hard consistency 在模型内部，train/eval 一致 | `UNIT_TESTED` | `test_hard_consistency_returns_the_measurement_exactly_where_it_exists`、`test_train_and_eval_mode_predict_the_same_thing` |
| 分支特征必须在特征列表中；非法参数拒绝 | `UNIT_TESTED` | 构造器校验 + 3 条拒绝测试 |
| 参数匹配对照 `{none,zeros,shuffle}` 参数量相等 | `UNIT_TESTED` | `test_the_physics_controls_keep_the_parameter_count` |
| 对照确实改变输出 | `UNIT_TESTED` | `test_the_physics_controls_actually_change_the_output` |
| 对照写进 `config.yaml` 与 `config_hash` | `UNIT_TESTED` | `test_the_control_and_consistency_reach_the_model_and_the_fingerprint` |
| `P2_DualI` 与 `P2_DualW` 参数量差 <5% | `UNIT_TESTED` | `test_duali_is_the_capacity_control_for_dualw` |
| 门控范围 [0,1]、起始 0.5、可关 | `UNIT_TESTED` | `test_the_gate_can_be_turned_off_and_is_reported_when_on` |
| 输出字典契约 + `as_reconstruction` 单点适配 | `UNIT_TESTED` | `test_forward_returns_the_documented_dict`、`test_as_reconstruction_accepts_both_output_kinds` |
| 训练/评估/推理/出图共用同一适配器 | `CODE_DONE` | `train.py`、`evaluate.py`、`inference.py`、`scripts/qualitative_figures.py` |
| Mask 洞值探针（未测 GT 改动不影响任何输入） | `UNIT_TESTED` | `test_w0_never_reads_the_dense_reference`、`test_the_item_builder_never_sees_the_dense_target` |
| 每个 epoch 重新抽 patch（含 holdout） | `UNIT_TESTED` | `test_each_epoch_draws_different_patches`（参数化两种） |
| 真实窗口端到端前向 + 反向 | `UNIT_TESTED` | `test_a_pr2_arm_runs_on_real_windows_and_backpropagates` |
| 2.5D Z/YZ 输入输出尺寸、`valid_z` 不参与堆叠 | `UNIT_TESTED` | 参数化 `t∈{1,7}`；`test_valid_z_is_not_a_stackable_channel` |
| **在各稀疏协议下真实训练** | `RUNNING`（y_k2 / fold 0；其余协议 `NOT_RUN`） | `runs/E024_dual_prior/`、`docs/PHASE1_PR2_E024_RESULTS.md` |
| **Gate 2 的"`P2_DualW` 在 identity H 下结果归因不虚报"** | `UNIT_TESTED`（结构层） | identity 退化已由测试与冒烟确认；**归因结论要等训练结果**，`NOT_RUN` |

## 4. T3–T6（本轮明确未做）

| 项 | 状态 | 说明 |
|---|---|---|
| T3 图像域 `L_rec`（Charbonnier + 缺失区加权） | `TODO` | 当前用既有 `ReconstructionLoss`（L1 + MS-SSIM），与旧臂一致以保公平 |
| T3 结构/拓扑损失 | `TODO` | 计划中排在 Gate 2 通过之后 |
| T4 E024 真实实验与报告（fold 0） | `RUNNING`：residual 已落地，dual/controls 在跑 | `docs/PHASE1_PR2_E024_RESULTS.md`、`runs/E024_dual_prior/` |
| T4 GPU 2 占用率闸门（<50 % 才可用，每次启动现量） | `UNIT_TESTED` | `run_e024.sh::gpu_mem`；`test_run_e024_guard.py` 7 项（注入假 `nvidia-smi`）；日志见 `Project log/Project_log.md` 2026-11-11 |
| T4 每次训练记录实测显存（`gpu_peak_{allocated,reserved}_mb`） | `CODE_DONE` | `train.py` 写进 `final_val.json`；GPU 上的 run 落地后核对 |
| T4 每卡一个作业的并发上限 | `UNIT_TESTED` | `run_e024.sh::make_room`；`test_run_e024_guard.py` 2 项（假 `python` 量真实重叠） |
| T4 断点续跑 `SKIP_DONE=1`（checkpoint + 测试汇总同时存在才算完成） | `UNIT_TESTED` | `run_e024.sh::is_done`；`test_run_e024_guard.py` 2 项 |
| T4 `run_ladder.py` 的 `pr2` stage | `TODO` | 本轮用独立脚本驱动；并入 ladder 之前需先有 E024 结果 |
| T5 `U_sampling` / `U_physics` 特征 | `CODE_DONE` + `UNIT_TESTED` | 已作为输入特征存在；**不是校准概率** |
| T5 `u_model`（集成/MC dropout） | `TODO` | — |
| T5 `tasks/vessel/` 拓扑置信适配器 | `TODO` | — |
| T6 安全清理实际执行 | `TODO` | 只出了 dry-run 清单；本轮不删任何东西 |

---

## 5. 测试与证据

| 项 | 值 |
|---|---|
| 全量回归 | 390 项，`EXIT=0`，`docs/evidence/pytest_20261111_T0_T2_guard.log`（日志自带 `390 passed` 汇总行，不需要数点） |
| 新增 `test_physics_prior_contract.py` | 30 |
| 新增 `test_dual_branch_contract.py` | 38 |
| 新增 `test_archive_safety.py`（T0 提交） | 21 |
| 新增 `test_psf_estimate.py` | 16（含 4 条 λ 稳定性） |
| 新增 `test_run_e024_guard.py` | 12（GPU 2 占用率闸门 ×7、无 `GPUS=` 拒绝、`SKIP_DONE` 语义 ×2、每卡一个作业 ×2） |
| 真实数据地板冒烟 | `docs/evidence/smoke_floors_20261110/`（3 份 summary + README） |
| 真实数据 PSF 估计 | `docs/evidence/psf_estimate_20261110/`（`psf_estimate.{json,md}`） |
| Fig. 2 重绘 | `spur_mpm/figures/framework/fig2_{pipeline,ablation_matrix}.png` + `fig2_provenance.json` |

复现命令：

```bash
# `-o addopts=""` overwrites pytest.ini's own `-q`; passing `-q` on top of it
# makes `-qq`, which drops the "N passed" line and leaves "count the dots" as
# the only evidence.
python -m pytest spur_mpm/tests -o addopts="" -q; echo "EXIT=$?"
python -m spur_mpm.scripts.framework_figure \
  --benchmark runs/E021_ladder/benchmark_supervised.json \
  --complexity runs/E021_ladder/complexity.json
```

---

## 6. 需要验收人裁决的三件事

1. **identity PSF 下能声称什么。** 本轮把它写死成"只能声称缺口填充"，
   并让 `P2_W0` 与 `P2_I0` 在真实数据上逐位相等。**这一条现在有了独立证据**：
   源数据不含标定（查证见 `docs/PHASE1_PR2_PSF_ESTIMATE.md` §1），两个自估器都
   指向 identity（算术上 PSF 比体素还细；经验上留出预测的最优宽度为 σ=0）。
   所以"先做 PSF 标定再进消融"这条待决项，实际上被降级为：**需要一个真实的
   标定体（beads 或仪器日志）才能改变结论**，而不是"再调一调参数"。
   若验收人有该数据，`P2_W0` 可以带标定 H 跑，但必须落在独立分组、独立目录
   （RUNBOOK §1.2）。
2. **是否允许"未经标定的 PSF 调参"进入主表。** 本轮的态度是不允许，
   并给出了两份反例：额外加 blur 再反卷积把 PSNR 从 28.03 压到 27.55；
   在真实数据上按 9 档 σ 做留出预测，每一档都比不做更差。现在新增了
   `estimated_from_data` 标签——**如果**验收人仍希望把自估宽度作为探索分支，
   该标签强制它在 summary 里区别于 `calibrated_optical`，可与主表并存而不混淆。
3. **E024 编号。** 目前 `runs/` 下没有 `E024_*`，脚本用 `runs/E024_dual_prior/`。
   若验收人要求顺延编号，脚本里 `OUT=` 一处改即可，但映射要记进本文档。

---

## 7. 本轮未运行的项，逐条列出（不得读作完成）

- `P2_Direct`、`P2_ResidualI`：已跑完（fold 0），见 `docs/PHASE1_PR2_E024_RESULTS.md`
- `P2_DualI`、`P2_DualW`、两个 `phys_control` 对照：`RUNNING`（2026-11-11 夜间落地）
- 默认 256×256 的地板正式数：`NOT_RUN`（只有 32×32 CPU 冒烟）
- 三折 × 多种子：`NOT_RUN`
- `z_k4` / `yz_k2x2` 等成本协议：`NOT_RUN`
- 整卷 3D 骨架报告、定性图、成本表：`NOT_RUN`
- `P2_DualWU` / `P2_DualWUT` / `P2_TopoConf`：`TODO`（前者需标定 H，后两者属 T3/T5）

Fig. 2 的 `planned` 栏、以及本文件的每一处 `NOT_RUN`，在真实测量落到
`runs/E024_dual_prior/` 之前都必须保持原样。

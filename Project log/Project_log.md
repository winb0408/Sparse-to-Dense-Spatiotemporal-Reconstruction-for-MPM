# SPUR-MPM 实验日志

只追加、不回改：每次实验（含脚手架级改动）追加一条，编号沿用 `plan.md` 的 `SPUR-E###`。
科学结论以本条目的「判断」为准，不以文字描述的印象为准。

条目字段：日期 / 目的 / 改动 / 配置 / 结果 / 结论 / 后续动作。

---

## SPUR-E000a — 仓库底座搭建（无数据）

- **日期**：2026-10-27
- **Git**：`research/spur-mpm-v1`（仓库初始化后的首个提交）
- **阶段**：Phase 0 前置（底座），**不是**数据审计，不产生任何科学结论
- **目的**：在 Dryad 数据下载完成前，把「一个输入 → 一次重建 → 一次评估」的完整竖切跑通，
  并把计划里的硬约束（ZYX 轴纪律、holdout 先拆分后建先验、分组划分、有/无监督模型选择口径）
  写成**结构性约束**而不是口头约定。

### 改动（新建模块）

| 层 | 文件 | 约束落点 |
|---|---|---|
| 物理 | `physics/psf.py` `sampling.py` `acquisition.py` `wiener.py` | 恒等 PSF 为默认（真实数据自带光学）；`M_acq` 定义为全局晶格 `(g-phase)%k==0` |
| 数据 | `data/geometry.py` `manifest.py` `normalize.py` `split.py` `observation.py` `datasets.py` | 出现 T 轴即报错；标定只在训练组拟合；<3 组拒绝划分 |
| 输入 | `data/observation.py::build_model_inputs` | **唯一**入口：`y_in = y * M_in`，先验只能由可见位置生成 |
| 模型 | `models/{attention,swin_unet_2d,prior_fusion,uncertainty,registry}.py` | A0/A1/S0-S2/SS0-SS2/SS2g/S1g 由同一 backbone + 配置臂区分 |
| 损失 | `losses/{reconstruction,data_consistency}.py` `ssl/mms.py` | MMS 目标是**真实含噪测量**，不是稠密参考、不是零 |
| 评估 | `metrics/{image_metrics,vessel,statistics}.py` | PSNR/SSIM/MAE + E_measured/E_missing + Dice/clDice + 分组 bootstrap |
| 脚本 | `scripts/{make_synthetic,audit_data,make_split,prepare_data}.py` `train.py` `evaluate.py` `inference.py` | 合成数据全程标注 `SYNTHETIC_SMOKE`，不作为任何证据 |

### 配置

`spur_mpm/configs/base.yaml`：恒等 PSF（`sigma_*_um: 0.0`，`voxel_xy_um: 0.98`，`voxel_z_um: 5.0`）、
`u_kind: geo`、`geo_mode: per_axis`、MMS `ratio 0.5 / geometry row`；
实验臂 `E001`（A0/A1）、`E002`（S0/S1/S2）、`E003`（SS0/SS1/SS2）。

### 结果

1. **单元测试全绿**：`python -m pytest spur_mpm/tests -q` → 68 passed。
   含泄漏决定性测试：扰动 `y_hold` 后 `y/mask/x0/confidence/uncertainty` 五个输入字段逐位不变。
2. **合成数据竖切跑通**（3 只鼠 × 2 FOV，12×96×96，全程 CPU）：
   生成 → 审计（0 issues）→ LOMO 划分（3 folds，fingerprint `2f41252d…`）→ prepare（6 卷，标定 offset 0.0 / scale 585.5）
   → S1 训练 20 步（dense_psnr 5.87 → 13.25）→ SS1 训练 20 步（holdout_loss 0.422 → 0.202）
   → A1 基线评估（clDice 0.955）→ 整卷推理（输出 `.npy` + `.mask.npy` + `.warmstart.npy` + 元数据 JSON）。
3. 竖切过程中暴露并修复的真实缺陷（均已进测试）：
   - `_distance_to_measured` 的 `cummax` 哨兵值取 `+n`：第一个测量之后的索引全部退化为距离 `n`，轴向间隙一律饱和到 1；
   - `yz` 采样误用了面内 `grid` 晶格（`rows | cols`），实际体素比例 0.375 ≠ 1/(ky·kz)=0.25；
   - `build_model_inputs` 依赖调用方「已把非测量位置清零」，改为在此处强制 `y * M_in`；
   - manifest 的 `path` 相对**数据根**而非 manifest 所在目录，`prepare` 原来按 manifest 目录解析，导致全部跳过；
   - 校准 `apply` 含 `clip(0,1)`，`inverse` 却宣称精确可逆：改为显式声明「在标定区间内可逆」，并新增 `saturated_fraction()` 报告饱和比例。

### 结论

底座可用，但**没有任何 SPUR 科学结论**。上述所有数字来自合成数据与 20 步训练，仅证明管线连通性。

### 后续动作

1. 等 Dryad 数据到位后：先 `audit_data`（SPUR-E000 真审计），逐条核对 axes / z_step / 体素各向异性，
   再 `make_split` + `prepare_data`。
2. 把 `spur_mpm/configs/data/dryad_3p_vessels.yaml`（当前为模板）的 glob、`filename_regex`、per-file 元数据与实际解包结构对齐。
3. 在真实数据上先跑 A0/A1 与 S1（单臂、单 fold）确认管线在真实 TIFF 上不崩，再铺全臂全 fold。

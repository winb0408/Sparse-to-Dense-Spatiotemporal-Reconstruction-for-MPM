# SPUR-MPM

**S**elf-supervised **P**hysics-guided **U**ncertainty-aware **R**econstruction for **M**ulti**P**hoton **M**icroscopy.

从稀疏采集的 3P 血管体数据重建稠密体数据，并把「采样几何」「物理可恢复性」「不确定性」
三件事分清楚：哪些是测量、哪些是插值、哪些只是先验。

> 当前状态：**代码底座已完成，真实数据未到**。
> 研究计划见 [`Project log/plan.md`](Project%20log/plan.md)，
> 实验记录见 [`Project log/Project_log.md`](Project%20log/Project_log.md)，
> 进度见 [`Project log/project_progress_status.md`](Project%20log/project_progress_status.md)。
> 仓库里目前所有数字都来自**合成数据**，不作为科学证据。

---

## 三条不可协商的纪律

1. **轴纪律**：体数据一律 `ZYX`，`Z` 永远是物理深度，**不是时间**。
   任何 `TYX`（钙成像时间序列）不得进入 3-D 加载路径 —— `VolumeGeometry.validate()` 见到 `T` 轴直接抛
   `GeometryError`。钙成像只在 Phase 6 出现。
2. **泄漏纪律**：自监督的 holdout 必须**先拆分、后建先验**。
   网络输入全部由 `build_model_inputs(y_in, M_in, …)` 生成，函数内部强制 `y_in = y * M_in`；
   holdout 目标是**真实含噪测量**，不是稠密参考，也不是零。`tests/test_ssl_leakage.py` 用
   「扰动 `y_hold` → 五个输入字段逐位不变」来钉死这一点。
3. **划分纪律**：按动物/独立组划分（leave-one-mouse-out 或 holdout），少于 3 组直接拒绝；
   patch 级随机划分不是替代方案。标定常数只在**训练组**上拟合。

## 目录结构

```
spur_mpm/
  physics/      正演与采样：psf（默认恒等）、sampling（y/z/yz/xy 晶格）、acquisition、wiener
  data/         geometry（轴/体素/审计字段）、manifest、normalize（标定）、split（分组划分）、
                observation（唯一输入构造入口）、datasets（patch / 逐平面窗口）
  models/       swin_unet_2d + attention、prior_fusion（先验注入）、uncertainty（C_s/U_geo/U_phys）、registry（实验臂）
  losses/       重建损失、数据一致性
  ssl/          holdout（M_in / M_hold 拆分）、mms（掩码测量自监督）、leakage（断言）
  metrics/      image_metrics、vessel（Dice / clDice）、statistics（分组 bootstrap）
  common/       config（严格点号覆盖）、provenance、runlog（运行目录契约）、seeding、tensors
  scripts/      make_synthetic、audit_data、make_split、prepare_data
  configs/      base.yaml、data/（数据集 spec）、experiments/（E001–E003 实验臂）
  tests/        68 项单元测试，含泄漏决定性测试与端到端竖切
  train.py      训练入口        evaluate.py  评估入口        inference.py  整卷推理
```

## 快速开始（无需真实数据）

```bash
pip install -r requirements.txt

# 1) 合成一个可审计的小数据集（全程标注 synthetic，不作证据）
python -m spur_mpm.scripts.make_synthetic --out data_local/synthetic \
    --spec spur_mpm/configs/data/synthetic_smoke.yaml --mice 3 --fovs 2 --depth 12 --height 96 --width 96

# 2) 审计 → 分组划分 → 缓存与标定
python -m spur_mpm.scripts.audit_data --spec spur_mpm/configs/data/synthetic_smoke.yaml --out runs/smoke/audit
python -m spur_mpm.scripts.make_split --manifest runs/smoke/audit/manifest.json --out runs/smoke/audit/split.json
python -m spur_mpm.scripts.prepare_data --manifest runs/smoke/audit/manifest.json \
    --split runs/smoke/audit/split.json --cache data_local/cache --index data_local/cache/index.json

# 3) 训练一个臂（SS1 = 自监督，只看 holdout loss 选模）
python -m spur_mpm.train --config spur_mpm/configs/experiments/E003_selfsup_y_k2.yaml --arm SS1 --fold 2 \
    --set data.index=data_local/cache/index.json --set data.split=runs/smoke/audit/split.json \
    --set data.patch_hw='[64,64]' --set train.steps=20 --set train.num_workers=0

# 4) 评估 / 整卷推理
python -m spur_mpm.evaluate --arm A1 --fold 2 --split val \
    --set data.index=data_local/cache/index.json --set data.split=runs/smoke/audit/split.json
python -m spur_mpm.inference --arm SS1 --ckpt runs/smoke/<run>/best.pt --sample SYN_M2_FOV1 --out runs/smoke/infer \
    --set data.index=data_local/cache/index.json --set data.split=runs/smoke/audit/split.json
```

`--set a.b=c` 是严格点号覆盖：键不存在会报错，不会静默新建配置项。
每个 run 目录必须能通过 `RunDir.validate()`（`config.yaml` / `provenance.json` / `log.csv` / `metrics.jsonl` / `best.pt`）。

## 真实数据（Dryad 3PM，已接入）

数据：`/ssd1/wjb/Data/Dryad_3PM`（DOI 10.5061/dryad.wh70rxx2j），规格文件
`spur_mpm/configs/data/dryad_3p_vessels.yaml`（glob / regex / 逐文件元数据 / z-step 决策均已与真实目录对齐）。
29 个 TIFF → 10 个可用血管卷（3 只鼠、2 个 session、最多 2 个 FOV），19 个显式排除，3 组像素级重复。

```bash
# 1) 审计（必须 0 unresolved issue 才能继续）
python -m spur_mpm.scripts.audit_data \
    --spec spur_mpm/configs/data/dryad_3p_vessels.yaml \
    --out spur_mpm/experiments/E000_data_audit

# 2) 留一鼠三折划分（验证集 = 剩余动物中一个完整 FOV，含两个 session）
python -m spur_mpm.scripts.make_split \
    --manifest spur_mpm/experiments/E000_data_audit/manifest.json \
    --group mouse_id --inner fov_id --mode lomo \
    --out spur_mpm/experiments/E000_data_audit/split.json

# 3) 载入缓存 + 逐折标定（标定只在训练动物上拟合）
python -m spur_mpm.scripts.prepare_data \
    --manifest spur_mpm/experiments/E000_data_audit/manifest.json \
    --cache data_local/dryad_cache --index data_local/dryad_cache/index.json \
    --split spur_mpm/experiments/E000_data_audit/split.json

# 4) 训练 / 评估（一折一 GPU）
python -m spur_mpm.train --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
    --arm S1 --fold 0 --set data.index=data_local/dryad_cache/index.json \
    --set data.split=spur_mpm/experiments/E000_data_audit/split.json \
    --set data.patch_hw='[256,256]'

# 5) 汇总（论文里的数字只能来自这里）
python -m spur_mpm.scripts.collect_results --pattern 'runs/E00*_*' \
    --index data_local/dryad_cache/index.json --out spur_mpm/experiments/summary.md
```

### 六条泄漏轴（不是口头约定，是代码断言）

| 轴 | 断言位置 |
|---|---|
| 动物 | `split.py::assert_no_leakage`（留一鼠，测试动物不得出现在 train/val） |
| session | `inner_key=(mouse, fov)`，同一 FOV 的两个 session 永远同侧 |
| FOV | `(mouse, fov)` 无条件不跨 train∩val |
| 通道 | 斑块通道全排除（与血管同一次扫描，跨通道泄漏组织身份） |
| 重复文件 | `manifest.py::content_fingerprint`（像素级，不用文件哈希） |
| 时间轴 | `geometry.py::VolumeGeometry.validate` 见 T 轴即 `GeometryError` |

**标定必须逐折**：`VolumeCache.load(fold=f)` 选该 fold 的 `calibration_fold{f}.json`；
单折标定会让别的折的测试动物用自己的统计量归一化。

## 实验臂

| 臂 | 含义 | 选模指标 |
|---|---|---|
| A0 / A1 | 双线性插值 / FillFirst-Wiener 基线（不训练） | — |
| S0 / S1 / S2 | 有监督（可用稠密参考），定义天花板 | 稠密参考验证集 PSNR |
| SS0 / SS1 / SS2 | MMS 自监督 | **仅 holdout 测量损失** |
| SS2g / S1g | 门控融合变体 | 同上，与 S2/SS2 同预算 |

自监督臂在**冻结权重之后**才允许看稠密参考；评估脚本与训练脚本的选模口径不同，这一点写在
`train.py::validate` 里并由 run 目录的 `select_metric` 字段留痕。

## 记录约定

每次实验在 `Project log/Project_log.md` 追加一条（日期 / 目的 / 改动 / 配置 / 结果 / 结论 / 后续动作），
编号沿用 `SPUR-E###`。「结论」只写数据支持得住的话；失败与不符预期的结果同样要写。

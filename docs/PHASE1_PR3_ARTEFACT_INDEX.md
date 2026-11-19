# PR3 阶段产物索引（E026 / E025）

> 本页只回答"东西在哪"。所有路径都相对仓库根 `/ssd1/wjb/SPUR-MPM`。
> `runs/` 被 `.gitignore` 排除，因此下面这些路径**在本地磁盘上存在、在 git 里不存在**；
> 文档、脚本、测试在 git 里，图与 npz 不在。

## 1. 结果报告（先读这三份）

| 文档 | 内容 |
|---|---|
| [`docs/PHASE1_PR3_E026_RESULTS.md`](PHASE1_PR3_E026_RESULTS.md) | E026 两组（`matched_synthetic` 已知核 + `real`）六臂：窗口口径、整卷结构口径、两套口径符号冲突的原因、深度分层、定性图裁剪指标、不主张什么 |
| [`docs/PHASE1_PR3_E025_RESULTS.md`](PHASE1_PR3_E025_RESULTS.md) | E025 单轴 `ky=4` 负对照：同样六臂，验证"效应与采样几何无关、冲突与几何无关" |
| [`docs/PHASE1_PR3_E026_OPERATOR_ORDER.md`](PHASE1_PR3_E026_OPERATOR_ORDER.md) | 为什么 `W0_legacy` 在 T=3 不是第二个地板（含 T-sweep 表与复现命令） |

配套：`docs/evidence/gpu2_falloff_20261113/INCIDENT.md`（掉卡事故与 CPU 回退事故）、
`Project log/Project_log.md`（逐日中文记录）、`Project log/project_progress_status.md`（11-18 状态）。

## 2. 定性对比图

**入库供审阅的版本**在 [`docs/figures/pr3/`](figures/pr3/README.md)（5 张，JPEG，合计约 12 MB；
原图与 `.npz` 不入库）。下面列出本地全分辨率路径。

每组两轮，**共用同一套冻结视角**（样本 `M2_FOVII_S1_vessels`，z = 101，y = x = 128，
ROI [160, 0, 96]，zoom 96 µm，全 z MIP）。两轮的原因是 `phys_control` 属于模型属性：
`P3_wiener` 与 `P3_interp` 不能共用一张图而不被错标。

| 组 | 图（第一轮：无控制臂） | 图（第二轮：`interp_only` 控制臂） |
|---|---|---|
| E026 `real` | `runs/E026_psras_condition/figures/E026_real_yz_k2_kz2_fold0_s0/qualitative_M2_FOVII_S1_vessels.png` | `.../qualitative_M2_FOVII_S1_vessels_interp.png` |
| E026 `matched_synthetic` | `runs/E026_psras_condition/figures/E026_matched_synthetic_yz_k2_kz2_fold0_s0/qualitative_M2_FOVII_S1_vessels.png` | `..._interp.png` |
| E025 `y_k4` | `runs/E025_adaptive_dual_prior/figures/E025_y_k4_fold0_s0/qualitative_M2_FOVII_S1_vessels.png` | `..._interp.png` |

每个目录里还有同一次渲染的配套面板与数据：

| 文件 | 读法 |
|---|---|
| `qualitative_*.png` | 主图：每臂一行，列依次为 XY / XZ / YZ / 放大 / z-MIP / 测量面 / 跳过面 / 测量行 / 跳过行（Y-only 组 7 列）；格内标注该裁剪区的 PSNR / Dice / clDice |
| `error_*.png` | 误差图：缺失区与已测量区分开着色（`--error-on-missing`） |
| `profile_*.png` | 沿 y 的深度/强度剖面，可看插值条带与网络平滑的差别 |
| `support_*.png` | 可见的采样支撑（哪些行/面是实测的） |
| `qualitative_*.json` | 每臂在该裁剪区的指标、视角指纹（index/split 的 bytes+mtime）、`measured_fraction` |
| `qualitative_*.npz` | 渲染用的整块数组（几十~一百多 MB，供重绘） |
| `viewspec_*.json` | 冻结视角本身；重绘时用 `--view-spec` 复用它，保证两张图看的是同一窗口 |

## 3. 定量分析

| 内容 | 位置 |
|---|---|
| 主表（六臂 × 指标 × 深度分层 + 运行清单） | `runs/E026_psras_condition/table_E026_real_yz_k2_kz2_fold0_s0.md`、`..._matched_synthetic_...md`、`runs/E025_adaptive_dual_prior/table_E025_y_k4_fold0_s0.md` |
| 每臂窗口口径明细 | `runs/E026_psras_condition/E026_<group>_yz_k2_kz2_fold0_s0_<arm>_test/summary.json`（同名 `.jsonl` 逐窗口） |
| 整卷结构报告（Dice/clDice/skelF1/精确率/召回率/半径/连通块/判分平面占比） | `runs/E026_psras_condition/structure_<TAG>_<arm>/structure_summary.{json,md}` |
| 配对比较（Δ 逐卷 + 均值 + 判分平面占比说明） | `runs/E026_psras_condition/structure_compare_<TAG>_<A>_vs_<B>.md` |
| 冻结的训练体选核（全部候选与误差） | `runs/E026_psras_condition/fold0_yz_k2_kz2_selection.json`、`runs/E025_adaptive_dual_prior/fold0_y_k4_prior_selection.json` |
| 队列与逐阶段日志 | `runs/PR3_queue_*.log`、`runs/E026_psras_condition/logs/`、`runs/E025_adaptive_dual_prior/logs/` |

**读这些表时的三条硬规则**（写在报告里，这里重复一次）：跨动物方差约 2.8 dB / 0.08 Dice，
小于 1 dB / 0.03 Dice 的差值不得作为结论；深度分层只用血管 Dice/clDice，不用 PSNR；
结构指标必须和"判分平面占比"一起看（同一只动物内判分比例 34 %–72 %，clDice 0.31–0.92）。

## 4. 复现命令

```bash
# 测试
python -m pytest spur_mpm/tests -o addopts="" -q

# E026（SOURCE_KIND=real 或 matched_synthetic；STAGE=select|floors|train|analysis|all）
DRY=1 STAGE=all SOURCE_KIND=real bash spur_mpm/experiments/run_e026_historical.sh
DRY=0 STAGE=all SOURCE_KIND=real GPU=0 bash spur_mpm/experiments/run_e026_historical.sh

# E025
DRY=0 STAGE=all FOLD=0 KY=4 GPU=0 bash spur_mpm/experiments/run_e025.sh
```

启动守卫在 `spur_mpm/experiments/_gpu_gate.sh`（设备可见 + 占用 < 60 %），
CPU 回退守卫在 `spur_mpm/device.py`（`SPUR_ALLOW_CPU=1` 才允许）。

## 5. 本轮没有的东西（避免被误读成"漏了"）

- 多 seed / 多 fold：`NOT_RUN`，因为初测无可信增益，加种子只会让零值更精确。
- 窗口口径的逐面阈值 + 门控重跑：`NOT_RUN`（会改动 E024 等已引用的指标口径，
  属于项目负责人决策，不是 pilot 能自行改的）。
- 任何延迟/吞吐/峰值显存作为 benchmark：全程共享卡，不作基准数。
- `runs/_invalid/20261114_cpu_fallback/`：一次被 CPU 回退污染的运行，保留作证据，不是结果。

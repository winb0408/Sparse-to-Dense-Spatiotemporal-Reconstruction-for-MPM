# 代码地图（有监督主线）

这份文档回答一个问题：*"我想看/改/复用某一部分，应该打开哪个文件？"*
每一节只列**职责单一**的文件，避免"一个脚本什么都干"。

主线是**有监督稀疏重建**。自监督路线的代码仍在仓库里，但被标注为冻结、
只用于复现已归档的结论，见 [archive/selfsup_v1/README.md](../archive/selfsup_v1/README.md)。

---

## 0. 数据这一层（从下载的原始数据到网络输入）

| 环节 | 文件 | 说明 |
|---|---|---|
| 原始数据清点 | `spur_mpm/scripts/audit_data.py` | 读 Dryad 目录，核对 shape / dtype，产出 manifest |
| 缓存 + 标定 | `spur_mpm/scripts/prepare_data.py` | 原始 `.npy` 缓存（不归一化），标定**只用该折训练集**拟合 |
| 队列划分 | `spur_mpm/scripts/make_split.py` + `spur_mpm/data/split.py` | 单折 holdout 或留一动物；`assert_no_leakage` 是硬闸门 |
| **稀疏采集构建** | **`spur_mpm/data/sparse.py`** | 本次交付的核心：稀疏梯度 + 真值核验 + 掩膜落盘 |
| 稀疏构建 CLI | `spur_mpm/scripts/build_sparse.py` | 一条命令构建全部稀疏等级并**逐条核验** |
| 稀疏构建出图 | `spur_mpm/scripts/sparse_figures.py` | 掩膜覆盖图 / 稀疏输入示意图（给人眼核验） |
| 采样算子 | `spur_mpm/physics/sampling.py` | `SamplingSpec`、`make_mask`、`kept_rows/kept_planes` |
| 前向观测模型 | `spur_mpm/physics/acquisition.py` | `Y = M · (H * R) + noise`，顺序固定：光学退化 → 采样 → 噪声 |
| PSF | `spur_mpm/physics/psf.py` | 高斯 OTF；默认恒等（无 PSF 标定前不引入额外自由度） |
| 网络输入构造 | `spur_mpm/data/observation.py` | 决定 Y / M / X0 物理热启动 / C_s / U 各是什么 |
| Dataset | `spur_mpm/data/datasets.py` | 窗口切片，相位锚定在全局索引上（切片不改变采集图样） |

**稀疏等级（`sparse.py` 里的 `LEVELS_2D` / `LEVELS_3D`）**

| 家族 | 2-D | 3-D | 说明 |
|---|---|---|---|
| Y 稀疏 | `y_k2/k3/k4/k8` | 同左 | 只保留 `(y-p) mod ky == 0` 的整行，X 方向全采 |
| Z 稀疏 | — | `z_k2/k3/k4` | 只保留整层，层内全采 |
| Y+Z 联合 | — | `yz_k2x2`（25%）, `yz_k2x4`（12.5%） | 行 ⊗ 层 的笛卡尔积 |

2-D 只有 Y 方向稀疏，因为 2-D 平面没有深度轴；`make_mask` 会显式拒绝
`z`/`yz` 模式而不是静默降级。三族在 25% 预算上可以对齐比较（`y_k4` /
`z_k4` / `yz_k2x2`）。

---

## 1. 网络结构

| 文件 | 内容 |
|---|---|
| `spur_mpm/models/swin_unet_2d.py` | 主干：Swin-UNet 2-D（逐切片重建，所以体数据的成本 = 切片数 × 单切片成本，没有跨切片项） |
| `spur_mpm/models/attention.py` | 主干内部的注意力块（与创新点无关，属结构本体） |
| `spur_mpm/models/registry.py` | **臂注册表**：`A0/A1/S0/S1/S2/S1g` 每个臂是一行声明式记录 |
| `spur_mpm/models/__init__.py` | 导出 |

臂 = 输入特征 × 物理热启动 × 融合方式的组合，**换臂只改配置，不改数据管线**。
这是"对比公平"的机制保证：同一份数据、同一个采样实例、同一个主干、同一预算。

---

## 2. 各创新点（消融即在这里）

三个创新点，每个一个独立文件，消融就是开/关它：

| 创新点 | 文件 | 臂间对比 | 关掉它 |
|---|---|---|---|
| 物理热启动 X0 | `spur_mpm/physics/wiener.py`（`wiener` / `wiener_fill_first` / `iterdc`） | S0 → S1 | `warm_start: bilinear` |
| 不确定度头 U | `spur_mpm/models/uncertainty.py` | S1 → S2 | 从 `features` 里去掉 `uncertainty` |
| 门控先验融合 | `spur_mpm/models/prior_fusion.py` | S1 → S1g | `fusion: false`（退化为 concat） |
| 无学习下限（对照） | `registry.py` 的 A0/A1 | — | `needs_training=False` |

数据一致性项与重建损失分开：`spur_mpm/losses/reconstruction.py`（主损失）、
`spur_mpm/losses/data_consistency.py`（只在测量到的位置上约束）。

**冻结（不属于有监督主线）**：`spur_mpm/models/complement.py`（双分支）、
`spur_mpm/ssl/*`。它们保留是为了让归档的 E011/E012 结论可复算，**新实验不要用**。

---

## 3. 训练 / 推理

| 文件 | 内容 |
|---|---|
| `spur_mpm/train.py` | 训练入口；从配置读取 `data.index / data.split / data.fold` |
| `spur_mpm/inference.py` | 滑动窗口推理，拼回整卷 |
| `spur_mpm/common/*` | 配置加载、随机种子、日志、provenance（每个 run 记录 git 状态） |
| `spur_mpm/configs/base.yaml` | 全局预算（步数、batch、patch、主干宽度），**任何表里的成本数字都从这里取** |
| `spur_mpm/configs/experiments/*.yaml` | 每个实验一个文件；换臂/换稀疏等级 = 换 YAML |
| `spur_mpm/scripts/run_ladder.py` | 在固定队列上跑"臂 × 稀疏等级"整个阶梯，阶段 = 问题（ablation / sweep / matched），逐 cell 写进度；`--eval-only` 只重打分不重训（换了指标/分割规则后用，检查点与这些无关） | 消融与稀疏曲线的主入口 |

---

## 4. 定量评估

| 文件 | 指标 | 什么时候用它 |
|---|---|---|
| `spur_mpm/metrics/image_metrics.py` | MAE / PSNR / SSIM，`split_errors` 把**测到的行**与**缺失的行**分开 | 报告强度吻合度。**不能单独用来声称结构恢复** |
| `spur_mpm/metrics/vessel.py` | `vesselness`（Sato 管状响应）、`segment_pair`（预测/参考**同一套**分割，阈值只在参考上拟合）、`mask_vessel_metrics`、`dice`、`cl_dice`、连通域数 | 结构重合。默认分割是管状响应而不是原始灰度 Otsu：实测原始 Otsu 会把单帧 37–55% 判成血管，骨架化后 95% 体素是节点（那是"面"不是"线"）；响应阈值下前景≈3–4%，与血管体积分数同量级。整卷另有一道**参考侧门控**：参考掩膜占该层比例超过 15% 的层面（离焦雾、致密亮区）在两侧同时清空并计数上报，否则一片 170 µm 的层能贡献 3437 mm"血管"（同卷干净层只有 20–40 mm） |
| `spur_mpm/metrics/skeleton.py` | 骨架化后的链长、分支、节点、端点、半径、按深度分层；`skeleton_overlap` 是容差内的骨架吻合；`export_imagej_mask` 导出 ImageJ 可开的 TIFF | **对应"ImageJ 提骨架后对比"**；回答"网络是否被打断/粘连" |
| `spur_mpm/metrics/complexity.py` | 参数量、MACs、实测延迟、整卷外推 | 和精度并列报告；延迟必带 device |
| `spur_mpm/metrics/statistics.py` | bootstrap CI、配对差、分组汇总 | 跨动物方差约 2.8 dB / 0.08 Dice，<1 dB / 0.03 Dice 的差异不构成结论 |
| `spur_mpm/evaluate.py` | 逐样本评估，写 `samples.csv` + `summary.json` | 所有表格的数据源 |
| `spur_mpm/scripts/collect_results.py` | 汇总多个 run 成一张表 | 出论文表格 |
| `spur_mpm/scripts/ladder_report.py` | 从各 run 的 `summary.json` 生成阶梯表 + 创新点归因，并把无学习下限摆在训练臂下面 | 出表；小于 1 dB / 0.03 Dice 的差异自动标为"不可主张" |
| `spur_mpm/scripts/structure_report.py` | **整卷** `analyse_volume`（含按深度分层）+ 导出 ImageJ 可读 TIFF；`--segment` 选分割方式、`--max-planes` 调试裁切 | 骨架层面的结构结论、人工核验。`analyse_volume(return_masks=True)` 让导出的 TIFF 就是算数字用的那两张掩膜 |
| `spur_mpm/scripts/verify_dataset_masks.py` | 训练代码路径重建的掩膜 vs 已核验 manifest 掩膜，逐点比对 | 证明"训练用的就是核验过的那份稀疏数据" |

为什么不用 PSNR 下按深度分层的结论：PSNR 受噪声主导，深层的信号弱、噪声占比高，
PSNR 会系统性偏低而与结构好坏无关。分层结论一律用 `dice`/`cl_dice`/骨架指标。

---

## 5. 可视化

| 文件 | 产物 |
|---|---|
| `spur_mpm/scripts/sparse_figures.py` | 稀疏构建示意、掩膜覆盖图、采集时间占比 |
| `spur_mpm/scripts/qualitative_figures.py` | 定性对比图（稀疏输入 / 各臂重建 / 密集真值），指标标在子图右下角 |
| `spur_mpm/scripts/sampling_diagram.py` | 采样图样与相位的示意图 |
| `spur_mpm/scripts/profile_depth.py` | 按深度的指标曲线（PSNR 与 Dice 分列） |

所有定性图都遵循同一约定：**未测量体素用青色 `#22d3ee` 画出**，灰度窗统一由密集
参考的 1%–99.5% 分位决定，所以"暗"是测量暗，不是窗宽造成的错觉。

---

## 6. 一键复现

```bash
# 1) 稀疏数据（可核验）
python -m spur_mpm.scripts.build_sparse \
    --index data_local/dryad_cache/index_cohort.json \
    --split spur_mpm/experiments/E020_cohort/split.json \
    --out data_local/sparse/cohort3d
python -m spur_mpm.scripts.build_sparse --as-2d ... --out data_local/sparse/cohort2d

# 2) 出图
python -m spur_mpm.scripts.sparse_figures --root data_local/sparse/cohort3d \
    --out spur_mpm/figures/sparse

# 3) 训练一个臂
python -m spur_mpm.train --config spur_mpm/configs/experiments/<exp>.yaml

# 4) 评估 / 汇总 / 成本
python -m spur_mpm.evaluate ...
python -m spur_mpm.scripts.collect_results ...
python -m spur_mpm.scripts.benchmark --markdown --arms A0 A1 S0 S1 S2 S1g

# 5) 消融阶梯（固定队列，逐 cell 落盘；--eval-only 只重打分不重训）
python -m spur_mpm.scripts.run_ladder --stage ablation --devices 2
python -m spur_mpm.scripts.run_ladder --stage ablation --eval-only --force
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder --out runs/E021_ladder/ladder.md

# 6) 结构评估（整卷骨架 + ImageJ 可读掩膜 + 定性图）
python -m spur_mpm.scripts.structure_report --arm S1 --ckpt runs/E021_ladder/E021_S1_y_k2/best.pt \
    --out runs/E021_ladder/E021_S1_y_k2/structure --device 2
python -m spur_mpm.scripts.qualitative_figures --arm S1=runs/E021_ladder/E021_S1_y_k2/best.pt ...
```

---

## 7. 测试

`python -m pytest spur_mpm/tests -q`

| 文件 | 保证什么 |
|---|---|
| `test_sparse_build.py` | 稀疏构建的**定义**：`Y = M·D` 逐点成立、未测量位置严格为 0、保留行整行全宽、相位可裁剪、噪声只落在测到的位置 |
| `test_physics.py` | 前向模型阶段顺序、2-D/3-D 分派 |
| `test_sampling.py` | 行/层格点、相位 |
| `test_split.py` | 划分无泄漏 |
| `test_skeleton.py` | 骨架指标的退化用例 |
| `test_complexity.py` | MACs/参数量在已知结构上的正确性 |
| `test_models_metrics.py` | 各臂前向形状、Dice/clDice 边界 |

冻结的自监督测试移到了 `archive/selfsup_v1/tests/`，不在主测试集里跑。

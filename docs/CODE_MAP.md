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
| 采样算子 | `spur_mpm/physics/sampling.py` | `SamplingSpec`、`MODES`（公开的格点全集 `y/z/yz/xy/full`，图件直接引用它，不再另抄一份）、`make_mask`、`kept_rows/kept_planes` |
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
| `spur_mpm/models/registry.py` | **臂注册表**：`A0/A1/S0/S1/S2/S1g/S2g` 与 PR2 的 `P2_*` 每个臂是一行声明式记录（S2g = X0+U+门控） |
| `spur_mpm/models/dual_prior_reconstructor.py` | **PR2 双分支模型**：观察编码器 + 物理编码器 → 逐尺度零初始化融合 → 残差解码器（`D_raw = I0 + R`）；hard consistency 在模型内部；`phys_control` 提供参数匹配对照。不改 `SwinUNet2D`，旧 checkpoint 照旧加载 |
| `spur_mpm/physics/prior.py` | **物理先验包**：`I0/W0/phys_delta/u_physics` + PSF 来源标注；`mask_aware_inverse`（以 `I0` 为正则中心，与旧 `iterative_dc_inverse` 并存而非取代） |
| `spur_mpm/physics/psf_estimate.py` | **无标定时的 PSF 估计**：`diffraction_limited_sigma`（由假定 NA/λ/n 反推衍射极限，并把结果同时给成体素单位）与 `holdout_width_scan`（用被欠采样的**真实测量行**做留出预测，扫宽度）。两者都不得自称标定；最优值落在 `σ=0` 就是"别反卷积"的估计 |
| `spur_mpm/scripts/estimate_psf.py` | 在真实训练队列上跑上面两个估计器，输出 `psf_estimate.{json,md}`；拒绝使用 test 组、不修改任何配置 |
| `spur_mpm/common/tensors.py` | `as_reconstruction()`：唯一知道"模型返回张量还是字典"的地方，训练/评估/推理/出图共用，防止三处对同一 checkpoint 做不同后处理 |
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
| `spur_mpm/scripts/run_ladder.py` | 在固定队列上跑"臂 × 稀疏等级"整个阶梯，阶段 = 问题（ablation / sweep / matched / floors / control），逐 cell 写进度；`--eval-only` 只重打分不重训（换了指标/分割规则后用，检查点与这些无关）；`--context-slices` 强制统一 Z 上下文（plan §3.3 的同上下文对照，cell 名加 `_c<N>`）；`--only` 按最终 cell 名筛子集；`--rebuild-state` 从 run 目录重建 `ladder.json` | 消融与稀疏曲线的主入口 |

**进度文件（`ladder.json`）以 cell 目录为键**，不是 `(arm, level)`：强制上下文的 `_c7` 与最小上下文的同级 cell 是两个实验，按 `(arm, level)` 去重会让前者**顶掉**后者的记录，之后 `done()` 认不出它已完成、下次会重训。真值仍在各 cell 目录的 `config.yaml` + `eval_test/summary.json` 里，所以任何时候都能 `--rebuild-state` 无损重建。

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
| `spur_mpm/metrics/vessel.py` `match_foreground` | 同一预测在**等面积工作点**再切一次（按参考前景面积取分位），只用于出 `dice_area` / `cl_dice_area` | 分离"形状对不对"与"对比度标定漂了"：参考阈值定的 Dice 下降可能只是预测响应变弱 |
| `spur_mpm/scripts/collect_results.py` | 汇总多个 run 成一张表 | 出论文表格 |
| `spur_mpm/scripts/ladder_report.py` | 从各 run 的 `summary.json` 生成阶梯表 + 创新点归因 + **预算曲线**（单个臂按测量比例从密到疏，看从哪里饱和）+ **同预算换轴对比**（`y_k4/z_k4/yz_k2x2` 都是 25%，看稀疏轴本身是否构成差异），并把无学习下限摆在训练臂下面 | 出表；小于 1 dB / 0.03 Dice 的差异自动标为"不可主张"；缺测的指标打印为 `–` 而不是 `nan` |
| `spur_mpm/scripts/structure_report.py` | **整卷** `analyse_volume`（含按深度分层）+ 导出 ImageJ 可读 TIFF；`--segment` 选分割方式、`--max-planes` 调试裁切 | 骨架层面的结构结论、人工核验。`analyse_volume(return_masks=True)` 让导出的 TIFF 就是算数字用的那两张掩膜 |
| `spur_mpm/scripts/verify_dataset_masks.py` | 训练代码路径重建的掩膜 vs 已核验 manifest 掩膜，逐点比对 | 证明"训练用的就是核验过的那份稀疏数据" |

为什么不用 PSNR 下按深度分层的结论：PSNR 受噪声主导，深层的信号弱、噪声占比高，
PSNR 会系统性偏低而与结构好坏无关。分层结论一律用 `dice`/`cl_dice`/骨架指标。

---

## 5. 可视化

| 文件 | 产物 |
|---|---|
| `spur_mpm/scripts/sparse_figures.py` | 稀疏构建示意、掩膜覆盖图、采集时间占比。XY 面板固定画在**所有层都测到的那一层**上（`_common_plane`），否则 Z 稀疏面板会因"画在没扫的层上"而整幅空白；每幅图都标注坐标方向（y 向下、x/y 向右、z 向下） |
| `spur_mpm/scripts/viewspec.py` | **固定视图层**（§4.4.1）：`ViewSpec` 只从密集参考里取一次，`protocol_columns` 按协议给列名、`take` 按列取面板、`ref_windows` 给统一窗、`check_group`/`check_protocols` 断言不同协议看到的是同一份参考与同一块面板 |
| `spur_mpm/scripts/qualitative_figures.py` | 定性对比图（稀疏输入 / 各臂重建 / 密集真值），指标标在子图右下角。面板、窗宽、ROI 全部来自 `viewspec`；`--compare` 跑多协议一致性自检（`target_diff`/`fixed_view_diff` 必须为 0），不一致时非零退出。`consistency_fields` 负责报告里三项**重算**出来的校验：窗宽重派生（`window_diff`）、掩膜是否真是它挂名的协议（`mask_semantics`）、形状/体素间距对得上冻结 spec（`geometry`） |
| `spur_mpm/scripts/sampling_diagram.py` | 采样图样与相位的示意图 |
| `spur_mpm/scripts/profile_depth.py` | 按深度的指标曲线（PSNR 与 Dice 分列） |
| `spur_mpm/scripts/framework_figure.py` | **Fig. 2**：网络管线示意 + 消融矩阵。矩阵**不手写**——每一格由 `models/registry.ARMS` 生成，`fig2_provenance.json` 逐格记录它取自 `ARMS['S1'].features` 之类的哪个字段；臂缺测量值直接 `KeyError`、冻结臂带数字直接 `RuntimeError`，空值写 `n/a` 而不四舍五入成 0。管线框里的块名、warm-start 模式串、格点模式（`physics/sampling.MODES`）和通道数都是**导入或数出来的**，不是散文里抄的 |
| `spur_mpm/scripts/structure_figure.py` | **Fig. 5**：3D 掩膜/骨架 MIP、局部分叉放大、断裂（红）/伪连接（蓝）标记图、跨卷汇总。断点/伪连接只在容差 1 voxel 之外非空；空配对记 NaN 不记 1.0；分叉裁剪框冻结进 `crop_*.json`（删掉才会重选），保证同一体积重绘不跳位 |

所有定性图都遵循同一约定：**未测量体素用青色 `#22d3ee` 画出**，灰度窗统一由密集
参考的 1%–99.5% 分位决定，所以"暗"是测量暗，不是窗宽造成的错觉。§4.4.1 之后还多三条
硬约束，由 `viewspec.py` 强制、由测试守着：

1. **固定面板**：XY/YZ/XZ 的位置、缩放、ROI 只从密集参考取一次并存进 `view_spec.json`，
   所有协议、所有臂共用同一份；
2. **方向与轴**：每个面板标注它到底是哪个轴（"XZ 实测行 / XZ 跳过的行"），Z-only 协议
   只报层、Y-only 协议只报行，不再出现"XZ 给定/缺失"这种把 Z 稀疏说成 X 稀疏的措辞；
3. **一致性自检**：`--compare` 跑完写 `consistency_report.json`，参考与固定面板在不同
   协议间必须逐像素相同（`target_diff == 0`），否则非零退出而不是出一张看不出问题的图。

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
python -m spur_mpm.scripts.benchmark --markdown --arms A0 A1 S0 S1 S2 S1g S2g
#    A0/A1 没有参数也没有 MACs，但 latency 列是**实测**的 warm-start 算子耗时
#    （同为 [Z,H,W] 体），不是 n/a——否则"零参数"会被读成"零成本"
python -m spur_mpm.inference --config <exp>.yaml --arm S1 --ckpt <best.pt> \
    --sample M2_FOVII_S1_vessels --out /tmp/recon.npy
#    整卷端到端真实耗时写在输出 JSON 的 timing.wall_seconds（含缓存加载、逐层
#    前向、堆叠、np.save），单位 patch 延迟不能当部署成本

# 5) 消融阶梯（固定队列，逐 cell 落盘；--eval-only 只重打分不重训）
python -m spur_mpm.scripts.run_ladder --stage ablation --devices 2
python -m spur_mpm.scripts.run_ladder --stage ablation --eval-only --force
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder --out runs/E021_ladder/ladder.md

# 6) 结构评估（整卷骨架 + ImageJ 可读掩膜 + 定性图）
python -m spur_mpm.scripts.structure_report --arm S1 --ckpt runs/E021_ladder/E021_S1_y_k2/best.pt \
    --out runs/E021_ladder/E021_S1_y_k2/structure --device 2
python -m spur_mpm.scripts.qualitative_figures --manifest runs/E021_ladder/manifest_y_k2.json \
    --level y_k2 --save-view-spec runs/E021_ladder/view_spec.json --out spur_mpm/figures/qualitative_y_k2
python -m spur_mpm.scripts.qualitative_figures --compare runs/compare_y_z_yz.json --out spur_mpm/figures/fixed

# 7) PR2 双先验阶梯（默认 dry-run，必须显式给 GPUS；顺序不可跳）
DRY=1 STAGE=all bash spur_mpm/experiments/run_e024.sh
GPUS=0,1,2 DRY=0 STAGE=floors bash spur_mpm/experiments/run_e024.sh
GPUS=0,1,2 DRY=0 STAGE=residual bash spur_mpm/experiments/run_e024.sh
GPUS=0,1,2 DRY=0 STAGE=dual bash spur_mpm/experiments/run_e024.sh
GPUS=0,1,2 DRY=0 STAGE=controls bash spur_mpm/experiments/run_e024.sh
#    protocol 由 MODE/KY/KZ/CONTEXT 决定，run 目录标签随之生成（不会出现
#    目录叫 y_k2 而实际跑的是 yz_k2x2）；完整说明见 docs/PHASE1_PR2_RUNBOOK.md
```

---

## 6b. 验收记录

`docs/PHASE1_ACCEPTANCE_REPORT.md`：手册 §2–§4.5 的逐条验收结果（改动的文件、可复现命令、
测试输出、§4.4.1 D 门槛对照表、修复前后差别、未闭合风险）。

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
| `test_benchmark.py` | 成本表的**诚实性**：无参数臂也给出实测延迟（不是 `n/a`）、缺 PSF 配置时退化为说明而不是崩溃、算子跑不动时进备注而不是抛栈 |
| `test_models_metrics.py` | 各臂前向形状、Dice/clDice 边界 |
| `test_qualitative_figures.py` | §4.4.1 的固定视图：参考不随协议变、目标不一致必须硬失败、面板与窗宽来自参考、Z-only 只报层、Y-only 只报行、同输入重复会被拒；`projection` 说明随面板变化；掩膜挂错协议/窗宽不是本参考的/报告缺协议 spec 都必须被抓 |
| `test_sparse_figures.py` | 稀疏图必须画在所有层都测到的平面上（`_common_plane`），没有共同平面时拒绝出图 |
| `test_run_ladder_state.py` | 阶梯状态按 cell 目录名去重：`_c7` 单元格不能顶掉同臂同层的兄弟 |
| `test_ladder_report.py` | 出表的**口径**：差异小于 1 dB / 0.03 Dice 必须打印"不可主张"、同预算换轴不足 0.03 Dice 不得写成"轴有影响"、缺测指标打印为 `–` 而不是 `nan` |
| `test_framework_figure.py` | Fig. 2 的矩阵必须来自 `ARMS`：每格带出处、冻结臂带数字要抛错、缺测量值要抛错、空值只写 `n/a`；管线框里的模式串与格点列表必须跟着代码走（把 fact 改掉、渲染结果跟着变），不能在散文里硬编码 |
| `test_structure_figure.py` | Fig. 5 里会撒谎的那部分：断点/伪连接图仅在容差外非空、空配对是 NaN 不是 1.0、分叉裁剪框在同一体积两次渲染之间不移动 |
| `test_structure_report.py` | 整卷表的表头必须说清它引用的是**骨架**分量数（不是掩膜连通块数，两者在同一份 JSON 里差几个量级）；`--from-summary` 重排版不许再索要 `--arm/--out` |

| `test_psf_estimate.py` | PSF 估计的契约：算术上 NA≥n 必须被拒、宽度必须以体素报告；留出扫描能找回它看得见的宽度、无模糊时估计为 identity、"拿测量当真值"与"拿锐利物体当真值"是两个不同问题；`estimated_from_data` 与 `calibrated_optical` 不得混名 |
| `test_physics_prior_contract.py` | PR2 的物理口径：`I0` 只来自测量、`H=I ⇒ W0==I0` 逐位、非恒等 H 必须先填满所有轴再 FFT、未标 `psf_source` 的 blur 必须被拒、新 `mask_aware_inverse` 把未测点按在 `I0` 而旧 `iterative_dc_inverse` 仍向 0 收缩（旧语义被锁住） |
| `test_dual_branch_contract.py` | PR2 的模型契约：初始化 `R≡0`、两个分支第 1 步都有梯度（第 0 步为零是刻意的，被断言而非隐藏）、hard consistency 在测量点精确且 train/eval 一致、对照参数量相等且确实改变输出、`P2_DualI` 是 `P2_DualW` 的容量对照、identity 下两算子相等、每个 epoch 重新抽 patch |
| `test_archive_safety.py` | 归档工具的**拒绝面**：默认 dry-run、受保护路径、路径越界、未授权通配、缺 allowlist 一律拒绝 |

冻结的自监督测试移到了 `archive/selfsup_v1/tests/`，不在主测试集里跑。

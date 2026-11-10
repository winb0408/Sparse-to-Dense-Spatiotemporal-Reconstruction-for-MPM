# PUR-MPM 第一层：有监督微血管稀疏重建——代码审查、执行规范与验收清单

> 状态：审查修复分支 `review/phase1-supervised-fixes-20261103`。基线 `PUR-MPM`；不覆盖历史实验。2026-11-03。  
> **执行声明（已更新）：** 源码审阅、修订与回归测试已在 GitHub 完成；本版本**已在持有 `/ssd1/wjb/Data/Dryad_3PM` 的训练机器上执行**：全量 pytest（245 项）、90 项 3D 采样构建与 `verify_dataset_masks` 交叉核验、固定队列监督消融（E021，38 cells）、整卷 3D 结构报告。逐条执行结果见 `docs/PHASE1_AUDIT_RESPONSE.md`。**尚未执行**的是 ImageJ/Fiji 人工抽检、其余三卷的冻结视角对比图、旧 E002/E009 checkpoint 的 eval-only 复评。  
> **研究范围：** 现阶段只做有监督；自监督、神经元分割、钙活动保真保持后续独立阶段，不作为本阶段前置条件。

## 0. 科学目标和证据边界

给定真实三光子高密度血管图像 D，依据点扫描的可解释采样几何生成虚拟稀疏观测 `Y=M·D`，以原始高密度图像作为 **有噪参考（dense reference）而非无噪声 GT**，比较插值、Wiener/物理初值辅助、显式 mask、不确定性通道和门控融合的有监督重建。核心结论以**完整三维血管骨架结构的保真**为准，PSNR、SSIM、CNR 只作补充。

与真实硬件稀疏采集不同，本阶段为 `virtual_sparse`；没有实测成像加速、光剂量节省数据，不声称实际时间/光功率收益。

### 历史背景与不应越界的结论

- `E001/E002` 的历史三折 `y_k2`：A0 插值 PSNR 29.06 / Dice .869 / clDice .834；S0 30.43/.884/.858；S1 30.46/.885/.859。它们**不是**新 E020 固定队列运行的结果。
- 主线 `physics.psf` 默认恒等。A0= A1（插值与所谓 Wiener），S0≈S1；在没有可测 PSF 下，不可写“Wiener 反卷积提供独立增益”，应改称 *warm-start interpolation/prior*。只有单独标定物理 PSF 且消融有效之后，才恢复更强的物理主张。
- 同一个高密度原始参考包含噪声和光学模糊，不能重复合成 PSF 并把结果当成默认主线。
- 用户希望固定一鼠测试的设计已实现，但测试鼠 M2 是参考先前 LOMO 验证时的较佳 Dice 选出的；这个测试集受既有结果影响。固定不再重选仍不能完全消除选择偏差。后续独立测试集可补强外部证据。

## 1. 固定队列（严禁改变后选择最佳结果）

- 分组文件：`spur_mpm/experiments/E020_cohort/split.json`。
- 训练：M1 FOVII S1/S2 + M3 FOVI S1/S2（4 卷，2 鼠）。
- 验证：M1 FOVI S1/S2（2 卷，与训练同鼠但不同 FOV）。
- 测试：**M2 FOVI/FOVII S1/S2（4 卷，完整 1 鼠）**。
- 所有臂、稀疏等级、重复种子使用相同的测试文件与参考处理，不能根据最终 M2 指标再选模型、阈值或样本。
- `spur_mpm/data/geometry.py` 检查 `ZYX` 语义；`spur_mpm/configs/data/dryad_3p_vessels.yaml` 审计 10 个血管栈，记录 Z step 元数据冲突；`prepare_data.py` 逐折/队列只用训练卷拟合归一化。
- 参考完整性：同鼠不同会话是相关样本，不可视为四只独立动物。M2 的不确定性区间只能描述这只动物内部的视野/切片变化，不是跨动物泛化 CI。

## 2. 显式稀疏数据构建（先于训练，单独交付核验）

### 2.1 一览

核心代码：`spur_mpm/physics/sampling.py`（严格周期晶格）、`spur_mpm/physics/acquisition.py`（前向模型）、`spur_mpm/data/sparse.py`（级别与构建验证）、`spur_mpm/scripts/build_sparse.py`（CLI）。

- 2D（`[H,W]`）仅 Y: `y_k2,y_k3,y_k4,y_k8`；用原始 3D 体数据中明确 Z 索引导出 2D 时必须记录 `#z<N>`。
- 3D（`[Z,H,W]`）Y: `y_k2,k3,k4,k8`；Z: `z_k2,k3,k4`；联合: `yz_k2x2,yz_k2x4`。
- 行保留：`(y+phase_y)%ky==0`，被保留的整行 X 全采；层保留 `(z+phase_z)%kz==0`；联合为行与层的交集。
- 可比较的 **理论测量份额 25%**：`y_k4`、`z_k4`、`yz_k2x2`。端部非整除时使用实际 `measured_fraction` 而不是只填理想值。
- 默认 `H=I` 且不再加合成噪声，`Y[M]=D[M]`、`Y[~M]=0`。`Mask` 是实测位置，不是可靠度概率。

### 2.2 可运行命令（在仓库根目录执行，路径应在原服务器存在）

```bash
# 复核数据审计与当前固定队列：不要重新划分测试动物
python -m spur_mpm.scripts.audit_data \
  --spec spur_mpm/configs/data/dryad_3p_vessels.yaml \
  --out spur_mpm/experiments/E000_data_audit

# 缓存准备请优先复用已存在、已在 E020 验证的 index_cohort.json，
# 若不存在再按 README 中 prepare_data 的固定 E020 split 命令生成。
test -f data_local/dryad_cache/index_cohort.json
test -f spur_mpm/experiments/E020_cohort/split.json

# A. 3D: 全部 Y / Z / YZ 密度，显式保存测量 Y 用于肉眼独立验收
python -m spur_mpm.scripts.build_sparse \
  --index data_local/dryad_cache/index_cohort.json \
  --split spur_mpm/experiments/E020_cohort/split.json \
  --fold 0 --out data_local/sparse/cohort3d --dump-y

# B. 2D: 所有体数据的 Z 中间层，仅作 2D Y 采样示例；不是所有切面
python -m spur_mpm.scripts.build_sparse \
  --index data_local/dryad_cache/index_cohort.json \
  --split spur_mpm/experiments/E020_cohort/split.json \
  --fold 0 --as-2d --out data_local/sparse/cohort2d --dump-y

# C. 独立核对：正式训练时按 Dataset 重新生成的窗口 Mask
# 必须和保存的 3D 构建器 Mask 逐点一致
python -m spur_mpm.scripts.verify_dataset_masks \
  --index data_local/dryad_cache/index_cohort.json \
  --split spur_mpm/experiments/E020_cohort/split.json \
  --manifest data_local/sparse/cohort3d/manifest.json \
  --levels y_k2 y_k4 z_k3 yz_k2x2 \
  --depth 7 --patch 64 \
  --out data_local/sparse/mask_crosscheck.json

# D. 采样图
python -m spur_mpm.scripts.sparse_figures \
  --index data_local/dryad_cache/index_cohort.json \
  --manifest3d data_local/sparse/cohort3d/manifest.json \
  --manifest2d data_local/sparse/cohort2d/manifest.json \
  --sample M2_FOVI_S1_vessels --out spur_mpm/figures/sparse
```

**核对文件：** `cohort3d/manifest.json`, `verification.json`, `masks/<level>/<sample>.npz`（bit-packed mask；`y` 为可选保存）、`cohort2d/...`, `mask_crosscheck.json`, `spur_mpm/figures/sparse`。确认每个 `verification.json` 的 `ok=true` 且总数和 10 卷×9 个等级匹配；校验真实 Y、Z、YZ 掩膜和中间采样比例。

**本修复分支：** `verify_construction` 不再把非布尔 Mask 静默强制转 bool 后判合格；形状不匹配安全报告失败；选用非恒等 PSF 的单独压力测试对照 `H(D)` 而不是原始 `D`。物理主线仍须 `H=I`。

## 3. 统一网络与消融（有监督）

| 臂 | 实测观测 Y + M | 额外初值 X0 | 几何不确定性 U | Gate | 用途 |
|---|---|---|---|---|---|
| A0 | 是 | 单独输出插值 | 否 | 否 | 零学习下限 |
| A1 | 是 | FillFirst-Wiener | 否 | 否 | 恒等 PSF 时与 A0 重合 |
| S0 | 是 | 否 | 否 | 否 | 基础有监督重建 |
| S1 | 是 | 是 | 否 | 否 | 增量：初值通道 |
| S2 | 是 | 是 | 是 | 否 | 增量：U 通道 |
| S1g | 是 | 是 | 否 | 是 | 增量：Gate |
| **S2g（本分支新增）** | 是 | 是 | 是 | 是 | 完整方法 |

`spur_mpm/models/registry.py` 是唯一臂注册表；`models/swin_unet_2d.py` 是共享主干；`physics/wiener.py` 初值；`models/uncertainty.py` 为几何/物理 U；`models/prior_fusion.py` 为门控 stem；`losses/reconstruction.py` 为有监督损失；`train.py`、`evaluate.py`、`inference.py` 是独立入口。**此时 U 是构造的可靠性代理，而不是已标定的预测误差，不宜宣称不确定性估计准确。**

注意 S1、S2 的输入通道数与 S0 不同，S1g 的额外 stem 也有参数，不能因“主干一样”就称整个模型等参数。需要在定稿消融报告中并列列出参数/MACs；若 U 增益很小或有疑义，再补等通道零 U/打乱 U 的容量控制。

## 4. 训练→评估→整卷结构：逐项核验

### 4.1 回归测试、配置与模型代码

```bash
python -m pytest spur_mpm/tests -q
python -m spur_mpm.scripts.run_ladder --stage ablation --dry-run
python -m spur_mpm.scripts.run_ladder --stage matched --context-slices 7 --dry-run
```

本分支修复训练评估一项严重不一致：旧 `VolumeWindowDataset` 在首尾层夹紧窗口，将查询放在非中心通道；而有监督训练损失仅监督中心输出。现在验证/整卷推理始终让查询在中心，越界层填充为零且标为未测量。**因此所有受影响的旧 Z 边界预测、整卷骨架指标必须重新生成**；没有必要为这一修复重新训练模型（前提是验证集选择和训练权重未另受影响，使用旧 best.pt 仍须审视原验证时的选模偏差）。

旧 `inference.py` 在 S0 Direct 没有 x0 输入时仍强行读取 x0，整卷推理报错；现用 Dataset 单独返回的 `warmstart_center` 导出诊断初值。逐层验证和可视化也改为使用该字段。

### 4.2 跑最小的有监督消融

```bash
# 先确认物理 GPU 2 空闲；--devices 是物理 GPU index 列表
python -m spur_mpm.scripts.run_ladder \
  --stage ablation --devices 2 --jobs 1 --out-root runs/E021_ladder

# 消融统一设定的 25% 预算；强制相同的 7 层 Z 上下文
python -m spur_mpm.scripts.run_ladder \
  --stage matched --context-slices 7 --devices 2 --jobs 1 \
  --out-root runs/E021_ladder

# 采样密度曲线与零学习下限
python -m spur_mpm.scripts.run_ladder \
  --stage sweep --stage floors --devices 2 --jobs 1 \
  --out-root runs/E021_ladder

# 重新评估已训练的权重（代码修复之后尤其必要）
python -m spur_mpm.scripts.run_ladder \
  --stage ablation --eval-only --force --devices 2 --jobs 1 \
  --out-root runs/E021_ladder

# 生成主定量表
python -m spur_mpm.scripts.ladder_report \
  --root runs/E021_ladder \
  --out runs/E021_ladder/ladder_report.md \
  --csv runs/E021_ladder/ladder_report.csv
```

`run_ladder.py` 从固定 E020 index/split 读取输入，按每个 cell 产出 `config.yaml`、`best.pt`、`eval_test/summary.json` 及训练/评估日志。**第一次用 `--steps 20` 的 smoke 与正式 3000-step 结果不可混表**。没有空闲 GPU 时，本分支拒绝静默回退占用 GPU0。

### 4.3 完整 3D 微血管骨架报告（主要生物学评价）

```bash
# 例如使用上面生成的 S1/y_k2 权重
python -m spur_mpm.scripts.structure_report \
  --arm S1 --ckpt runs/E021_ladder/E021_S1_y_k2/best.pt \
  --out runs/E021_ladder/E021_S1_y_k2/structure \
  --device cuda:2 \
  --set data.index=data_local/dryad_cache/index_cohort.json \
  --set data.split=spur_mpm/experiments/E020_cohort/split.json \
  --set data.fold=0 \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1
```

**主要指标：** 完整 3D 的 skeleton precision/recall/F1、clDice、分叉/端点/连通域、血管总长度、长度密度、半径误差、按深度分层；PSNR/SSIM/E_missing 作为辅助。 `structure_report.py` 导出 `*_pred_mask.tif`、`*_ref_mask.tif`，二者就是计算指标的同一对 Mask，可直接在 ImageJ/Fiji 中打开并使用 Analyze Skeleton 手动核对。原始高密度参考是带噪的，不是人工血管标注，所以这些结构分数反映相同自动分割规则下的一致性，不能直接等同于真实生理解剖真值。

**重要协议区别：** `evaluate.py` 默认按中心 `256x256` 的 2D 窗口逐层得分；`structure_report.py` 用 `inference.py` 做完整宽度的 3D 整卷预测，再在 3D 体上分割、骨架化。阈值/响应维度不同，不能把这两类 Dice/clDice 放进同一数值排名。应优先把整卷 3D 结果作为论文结构主指标。对于参考掩膜占比 >15% 而被自动剔除的层，必须逐卷报告剔除比例，并在 Fiji 抽查，防止容易出错的层被大量排除导致分数虚高。

`metrics/skeleton.py` 本分支修复了物理半径估计：基于 `distance_transform_edt(mask, sampling=(z_step_um, xy_um, xy_um))`，而不是原先先算 isotropic radius 再乘像素间距几何均值。历史半径微米数需要重算。骨架默认“容差 1 voxel”不是“容差 1 µm”（Z=5 µm 与 XY≈0.98 µm），应在定稿时加上物理距离容差敏感性分析。

### 4.4 定性图和 ImageJ 人工核验

```bash
python -m spur_mpm.scripts.qualitative_figures \
  --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --index data_local/dryad_cache/index_cohort.json --fold 0 \
  --sample M2_FOVI_S1_vessels \
  --arm A0 \
  --arm S0=runs/E021_ladder/E021_S0_y_k2/best.pt \
  --arm S1=runs/E021_ladder/E021_S1_y_k2/best.pt \
  --arm S2=runs/E021_ladder/E021_S2_y_k2/best.pt \
  --arm S2g=runs/E021_ladder/E021_S2g_y_k2/best.pt \
  --out runs/E021_ladder/figures/qual_y_k2 \
  --set data.split=spur_mpm/experiments/E020_cohort/split.json \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1
```

该脚本生成 `qualitative_*.png`、`error_*.png`、`support_*.png`、强度 profile、原数组 `*.npz`；不要把来自不同 Mask/归一化/评估域的图片拼成同一个方法比较。先挑 M2 的同一视野做视觉诊断，正式结果同时提供所有可用测试卷与失败案例。对于 Z/YZ，应使用同协议配置并完整记录 `context_slices`；推荐同时画 XY、XZ、YZ 以及断裂/伪连接局部放大。

### 4.4.1 【新增待 DeepSeek 实施】跨稀疏程度定性图的 GT / 视野 / Mask 一致性修复（P1，正式出图前必须通过）

**发现位置与实例**：`spur_mpm/scripts/qualitative_figures.py` 的 `_view_spec()` 用当前 `in_mask` 选择 `y_meas` 和 `y_gap`；`_views()` 的 `XY 25%/50%/75%` 是 Z 深度的相对位置，并非采样密度；`render()` 的 support 图底部固定写着“x and z are never subsampled”，在 `z_k4` 与 `yz_*` 下不成立。样例：`runs/E021_ladder/figures/qualitative_M2_FOVII_S1_vessels_z_k4.png` 及相应 `_y_k4.png`。当前单张图第 2 行的多个 GT 小图是 **同一体数据的不同空间切面/投影，互不相同属正常**；真正需要修复的是跨采样协议时部分剖面坐标随 Mask 改变，以及 Z-only 下“XZ given/gap”语义错误，不能误判为原密集参考已经被采样改变。

#### A. 数据真值与可视化范围的强制不变量

1. **同一 `sample_id`、同一原始密集体、同一缓存版本、同一空间域和归一化配置**，所有 `y_k2/y_k4/y_k8/z_k2/z_k4/yz_k2x2/...` 的参考 `target` 必须数值一致；GT 取自训练集标定过的密集缓存，不能从 `Y`、`M` 或重建结果生成，也不能根据 `sampling.mode` 填洞、滤波或调整它。
2. 对一组横向比较图必须**冻结完整的 ViewSpec**（至少 `sample_id, dataset/index fingerprint, z_index, y_index, x_index, ROI[y0:y1,x0:x1], XY/XZ/YZ 切面坐标、MIP 的 Z 范围、各显示色标的上下限、输入的采样协议及网络的上下文深度`），并对所有实验传入同一个 ViewSpec，而不是各图运行 `_view_spec(ref, mask)` 自行重选坐标。ViewSpec 应从参考体和显式人工/配置坐标生成，可以保存 JSON 并由 `--view-spec` 等参数重复读取；先验证坐标与参考体尺寸合法，再出图。
3. 固定的视图（例如 `XY @ z=80`, `XZ @ y=120`, `YZ @ x=120`, `Zoom @ z=80, y=... , x=...`）在各稀疏等级下必须像素级相同；但**一张多列图内的 XY、XZ、YZ、MIP 本身是不同视图，不要求相同**。各方法在同列必须对齐相同物理 ROI。保存像素间距（Z/XY）与方向信息，勿把 Z=5 µm 的轴以 XY=0.98 µm 等比例误画；标签清楚标注位置/µm。
4. 显示统一：每个比较组对相应视图从同一个**密集参考**提取强度窗（例如固定分位数），并对所有稀疏输入、基线、重建、GT 重用；误差图用同一误差尺度，禁止逐模型自动增强对比度制造视觉优势。不同深度本来强度会变；可逐视图窗，但同一视图在全部协议/方法之间必须一致。必要时另外保存固定全局窗对照。
5. 同一参考体、同一视图的稀疏输入只通过 `Y=M·D` 与合法 Mask 改变（可选实验明确标注额外噪声/PSF）；模型结果随对应的输入和权重改变。GT 层独立于所有采样版本。
6. 保持定量分析的原始 `float32` 数组，不在图中重新归一化结果或改写用于评价的数据。画图、检查和运行来源要分离：历史 PNG 不覆盖；新图放 `runs/E021_ladder/figures_fixed/` 或独立带版本的目录，记录修复前后。

#### B. 按采样方向定义有物理意义的视图（不要用同一模板硬套）

| 模式 | **输入**应突出 | GT / 重建（同一固定 ROI） | 不应使用的标签 |
|---|---|---|---|
| **Y-only**（2D-Y / 3D-Y） | `XY @ fixed z` 的保留 Y 行与缺失 Y 行；`XZ @ measured y`、`XZ @ missing y` 可以额外作为诊断，但所选两条 Y 坐标需在同一比较组固定记录；可显示相应 `M` | 固定 XY 和固定 ROI、同截面 YZ/XZ | 勿将 “XY 25%/50%/75%” 写得像采样比例；明确 `XY at 25% Z depth` 或改成绝对 z index/depth |
| **Z-only**（`z_k2/z_k4`） | **Measured XY**（整张真实采到的层）与 **Missing XY**（被跳过、要求网络预测的层），两者的 z 坐标必须明确；固定 XZ/YZ 跨层示意和 Z 向采样 mask | 对应各个同坐标 XY、XZ、YZ，保留真正的完整 GT | 删掉/改掉 `XZ given/gap` 这种按 Y 行区分采样的标签；不得称 “z is never subsampled” |
| **YZ-joint**（`yz_k*`） | 分开显示 **Z 已采层中的 Y 保留/缺失行** 与 **Z 缺失整层**，以及联合 3D Mask；各类像素是否测量由全局 `M(z,y,x)` 判定 | 同一固定 XY/XZ/YZ + 缺失层的放大，标出 `z/y` 与 Mask | 勿把某行/某层笼统叫作 given，除非该视图内对应采样位置确实有测量；应显示真正 Mask/采样百分比 |

**跨 `y_k4`, `z_k4`, `yz_k2x2` 的 25% 预算比较**：必选一组共享的 `sample_id`、固定缺失/已测 `z` 坐标（同时明确某 `z` 在不同模式可能采/不采）、固定 ROI 和统一的 GT。允许按模式增设诊断列，但跨模式主对照列必须完全一致。若固定的 `z` 刚好在某个模式被采集，不要把该列错误标为 `Missing XY`；标签从实际 mask 判断。

#### C. 需要 DeepSeek 修改的代码与最小测试

- 优先修改 `spur_mpm/scripts/qualitative_figures.py` 的 `_view_spec()` / `_views()` / `render()` / CLI，增加可保存/复用的固定 ViewSpec，解耦参照图选取与当前 Mask；同样复查 `spur_mpm/scripts/sparse_figures.py` 的采样展示位置与 `support_` 图的方向说明。
- 如果单体绘图仍保留自动选代表 ROI 的便利模式，必须基于 dense reference 或用户指定的 ROI，仅执行**一次**并导出 ViewSpec；跨等级拼图必须读取该 ViewSpec，禁止再运行 mask-dependent `_view_spec`。独立的“单个采样模式诊断视图”如确实需要保留 mask-dependent 行，必须标注为 diagnostic、不能与跨等级严格对照图混用。
- `z_k4` 以前的图在第 2 行显示多个不同 GT 小图，不要采取“强行把所有子图复制同一个 XY GT”的错误修复。要把深度/视角写清，并验证**相同列跨模型及跨等级**的一致性。
- 新增/扩充 `spur_mpm/tests/test_qualitative_figures.py`（或等价测试）：**(i)** 从同一 dense reference 构建 Y/Z/YZ 多种 M，断言数值 GT 数组及固定 views 的对应裁剪像素完全一致；**(ii)** 将同一 ViewSpec 作用于各采样协议时，XY/XZ/YZ 坐标一致、标签正确；**(iii)** `z_k4` 的 measured/missing Z 区别与 `mask[z].all()/mask[z].any()` 一致，不把 Y 行错说成 given/gap；**(iv)** `yz_k2x2` 测量行、跳行和跳层的展示/标签都正确；**(v)** 方法间共享强度/误差色标；**(vi)** `npz` / JSON sidecar 包含固定 ViewSpec、实际 sampling fraction、归一化/数据及模型来源；**(vii)** 相同协议下所有臂输入和参考相同（沿用已有 assert）。
- 注意：当前 `qualitative_figures.py` 主要做中心 `patch_hw=[256,256]` 的逐层可视化，不等于 `inference.py` 的整卷重建。若最终论文使用 full-volume 结构图，需从完整体积推理统一生成相同坐标、相同 ROI 的图，不能将 patch 与 full-volume 结果误拼到一张对照图里。
- **边界窗口修复已在本 PR 中**，DeepSeek 应在该分支上接续开发，确保新可视化沿用同一采样/推理构建链，不复制一份可能旧版的 Z 边界读取逻辑。

#### D. 必需产物 / 验收门槛（最终运行时填写真实结果）

- `figures_fixed/<sample>/view_spec.json`：固定坐标、物理 voxel spacing、统一窗和数据/配置指纹、MIP 范围。
- `figures_fixed/<sample>/compare_y_z_yz_25pct.png`：`y_k4/z_k4/yz_k2x2` 同视野同 GT 的主对照图（同一固定 ROI；显示真实采样 Mask 与完整 GT）。
- `figures_fixed/<sample>/qualitative_*_<level>.png` 和 `support_*_<level>.png`：各稀疏级别正确标签与真实测量支持图。
- `figures_fixed/<sample>/consistency_report.json`：包含同一参考的 `max_abs_diff_gt`（期望精确 0）、各对应截面的 `max_abs_diff_view`（0）、ViewSpec/物理坐标一致性、实际 measured fraction、`mask` 采样语义校验、统一 colorbar / intensity range、逐臂输入一致性；失败应让检查命令退出非零，不能仅 `print warning`。
- 各主图 `PNG + JSON + NPZ`：留原始 `float32` 源数据或以可复核格式保存，`npz` 中未压缩的计算真值与重建数组不要因保存为 float16 破坏像素级严格比较。
- `pytest` + 用 **M2_FOVII_S1_vessels** 输出 `y_k4 / z_k4 / yz_k2x2` 的同坐标比较图：目测对应 GT、缺失层/缺失行、色标与标注无误；保存通过的日志及版本信息。
- **禁止仅改 label 不修坐标/输入逻辑**；不允许为了让输出图片好看修改 GT、调换 reference、事后改变 split/ROI 挑选更好的方法比较效果。

验收完成后，在 PR 中标明具体修改文件、可重现命令、测试输出、图路径及修复前后差别。若发现不同协议下完整 `target` 数组真的不同，视为 P0 数据路径问题，停止论文出图并向上追查 Dataset/normalization/cache，而非仅修图。


### 4.5 参数量 / 计算复杂度 / 延迟

```bash
python -m spur_mpm.scripts.benchmark --markdown \
  --arms S0 S1 S2 S1g S2g \
  --patch 256 --n-slices 200 \
  --device cuda:2 --warmup 10 --iters 50 \
  --out runs/E021_ladder/benchmark_supervised.json
```

指标须分别报告 `params`、`MACs`（或明确 FLOPs≈2×MACs 的计算约定）、单位切片延迟 p10/p50/p90、峰值显存、整卷预测时间和所用 GPU。当前 `benchmark.py` 的参数/MACs 仅包括网络，不包括 Wiener/插值构建、CPU数据加载、全卷重建、结构分割和 TIFF 写入。论文中应同时测**实际整卷端到端 wall-time**，不得把单位 patch 估算时间当成真实部署用时。A0/A1 不能因零参数就把插值/反卷积执行耗时写为零。

## 5. 已修复错误（提交于独立 review 分支）

| 问题 | 类型 | 改动文件 | 影响 |
|---|---|---|---|
| Z 边界窗口使用未训练输出通道 | 实现逻辑错误 | `data/datasets.py` | 全卷重建、整卷骨架与边界评价 |
| S0 推理缺 x0 特征报错 | 运行时错误 | `data/datasets.py`, `inference.py` | Direct 不能正常导出完整体数据 |
| 基线评估/可视化硬编码中心初值 | 路径不一致 | `evaluate.py`, `scripts/qualitative_figures.py` | 边界值和体推理不一致 |
| Mask dtype 校验在 bool 转换后进行 | 虚假通过的验证缺陷 | `data/sparse.py` | 非布尔 Mask 被误判为合法 |
| 开启合成 PSF 时仍与未模糊 D 比较 | 断言错误 | `data/sparse.py`, `scripts/build_sparse.py` | 显式 PSF 压力测试错误失败 |
| 缺完整有监督 U + Gate 臂 | 消融不完整 | `models/registry.py`, `scripts/run_ladder.py` | 无法完整隔离新增模块贡献 |
| 微米血管半径用几何平均像素间距换算 | 物理单位估计错误 | `metrics/skeleton.py` | 半径 µm 的物理解释错误 |
| 无空闲 GPU 时默认抢 GPU0 | 共享服务器运行风险 | `scripts/run_ladder.py` | 可能占用其他用户显卡 |
| 以上核心路径回归测试 | 覆盖不足 | `tests/test_pipeline_smoke.py`, `tests/test_sparse_build.py` | 提交前必须执行 |

## 6. 尚未关闭的风险 / 下一轮验收判据

1. **真实服务器必跑：** 全部 pytest 通过；`build_sparse` 的验证全通过；`verify_dataset_masks` 全通过；S0/S1/S2/S1g/S2g 在同样输入上完成正向推理，checkpoint 能加载，测试样本支持不变。此次 GitHub 修改没有执行 GPU 训练或对用户私有数据进行实际复评。
2. **评价结论：** 主文报告整卷 3D 结构；固定参考分割与规则、门控剔除层和物理尺寸；基于 ImageJ/Fiji 人工检查至少一些连续段、断裂、伪连通与深度变化。
3. **性能验证：** 所有消融与采样强度用同一固定队列与参考；对于 Z 跳采、YZ 跳采，应强制 Z 上下文一致后比较；同时给测量区域和缺失区域的误差。
4. **缺少人工解剖 GT：** Sato vesselness + 参考侧 Otsu 的“血管真值”只是从原密集图自动提取的参考，不能证明所有血管连接真实。若最终论文主打拓扑恢复，建议人工标注少量可追踪微血管/分叉作为小规模核验集。
5. **泛化问题：** M2 只有一个留出动物，不能用对该动物切片 bootstrap 的极窄 CI 声称跨动物稳定。M2 被先前折结果影响过，应在文章说明并寻求独立测试数据。
6. **方法命名：** 恒等 PSF 时，把 `X0` 实质称为 *interpolation-derived prior*；若要主张 Wiener 引导，应先补经实验标定的 PSF 与独立消融。没有必要为争取创新描述而向原真实图像叠加一个未标定的高斯模糊。

## 7. 预期第一层论文图

- Fig. 1：真实 Dryad 数据几何/动物划分 + 2D/3D Y/Z/YZ 稀疏 Mask、测量预算示意。
- Fig. 2：稀疏观测、插值/物理先验、Mask/U/Gate 和共享 Swin-UNet；消融矩阵。
- Fig. 3：Y 稀疏下所有消融的全卷典型区域 XY/XZ/YZ，统一色标和误差图。
- Fig. 4：Z/YZ 稀疏的性能–稀疏程度曲线，同 25% 预算、固定 Z 上下文对照。
- Fig. 5：微血管参考与重建的 3D 掩膜/骨架、局部分叉、断裂/伪连接标记、clDice、skeleton F1 与长度/分支统计。
- Fig. S1：参数/MACs/推理耗时、背景强度误差、门控删除层审计、测试动物所有视野完整结果。

> **阶段完成判据：** 不是只“运行完全部脚本”，而是报告中每一个主图指标都可追溯到：固定划分指纹 → 单一稀疏协议 → config/训练权重 → 同输入测试 → 完整体预测 → 同一分割/骨架规则 → ImageJ 可复核的 TIFF 与原数组。

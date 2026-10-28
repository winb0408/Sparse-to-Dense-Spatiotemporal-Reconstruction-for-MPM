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

---

## SPUR-E000 — Dryad 3PM 真实数据审计与泄漏隔离

- **日期**：2026-10-28
- **Git**：`research/spur-mpm-v1`
- **阶段**：Phase 0
- **数据**：`/ssd1/wjb/Data/Dryad_3PM`（DOI 10.5061/dryad.wh70rxx2j）
- **产物**：`spur_mpm/experiments/E000_data_audit/{manifest.csv,manifest.json,audit_report.md,split.json}`、
  `data_local/dryad_cache/`（10 卷 `.npy` + `index.json` + `calibration_fold{0,1,2}.json`）
- **目的**：先读数据自带说明，再回答两个问题——**(a) 这批数据能怎么用；(b) 泄漏边界在哪、如何用代码锁死**。

### 数据本身（README + 逐文件核对）

APP NL-G-F 小鼠，Texas Red 标血管 + methoxy-XO4 标斑块，同一只鼠**相隔 4 周**成像两次；
XY 0.98 µm，Z 步长 5 µm（血管/斑块）与 10 µm（神经元），成像深度约 0–900 µm，激发 1340 nm。
另有 GFP 神经元片层与 6 张新生鼠肺图像（功率稳定性标定用）。

29 个 TIFF → **10 纳入 / 19 排除 / 3 组像素级重复 / 0 未解决 issue**。

### 泄漏边界与处置（本次审计的核心）

| 泄漏轴 | 具体情况 | 处置 |
|---|---|---|
| **动物** | 只有 3 只鼠（M1/M2/M3），M3 仅 1 个 FOV | 测试集按**留一鼠**，不做单文件随机划分 |
| **纵向 session** | 同一 FOV 的 Session 1/2 是同一批血管，相隔 4 周 | S1/S2 视为同一 `inner_key=(mouse, fov)`，**不跨 train/val** |
| **FOV** | 每鼠 1–2 个视野 | 验证集 = 剩余动物中的**一个完整 FOV（含两个 session）** |
| **通道** | 血管与斑块来自**同一次扫描** | 斑块通道全部排除：跨通道复用会泄漏组织身份 |
| **重复文件** | `Fig 3/3a Vessels.tif` 与 `M2_FOVII_S1_vessels` 像素完全一致；`Fig 3/3a Plaques.tif` 同理；`M3_FOVI_S2_plaques` 与 `M2_FOVI_S2_plaques` 像素完全一致（**跨动物**） | 像素级指纹 `content_fingerprint()`（sha256 over shape+dtype+bytes）自动识别；重复项一律 `use=excluded` |
| **时间轴** | 6 张新生鼠肺是 `TYX` 时间序列 | 明确拒绝进入 3D 管线（`GeometryError`），留给 Phase 6 钙成像 |
| **图版文件** | `Fig 3/*`、`3b - GFP neurons` 是从数据集中派生的插图 | 全部排除，且它们本身缺少 mouse/session/fov 标签 |

**为什么必须做像素指纹而不是文件哈希**：`Fig 3/3a Vessels.tif` 与原始卷的 TIFF 头（ImageJ 元数据）不同，
字节哈希不同，但像素完全相同。只比文件大小或 md5 会漏掉这类泄漏。

**标定必须是 per-fold**：`offset/scale` 用训练动物的分位数拟合。若用单一全局标定，
fold 1/2 的测试动物会以**自身统计量**归一化，测试集信息通过归一化常数回流。
现在 `VolumeCache.load(fold=f)` 在载入时选择该 fold 的标定，未登记则 `KeyError` 而非静默回退。

### 划分结果（`split.json`，fingerprint `e473498d…`）

| fold | test（留出动物） | val（留出 FOV，含两 session） | train |
|---|---|---|---|
| 0 | M1（4 卷） | M2 FOV I | M2 FOV II + M3（4 卷） |
| 1 | M2（4 卷） | M1 FOV II | M1 FOV I + M3（4 卷） |
| 2 | M3（2 卷） | M1 FOV II | M1 + M2（6 卷） |

标定（仅训练动物拟合）：fold0 offset 32727.5 / scale 13631.5（4 卷）；fold1 32037.75 / 13511.25（4 卷）；
fold2 32396.5 / 14603.33（6 卷）。

### Z 步长决策

ImageJ 头里的 spacing 在 5.0 / 7.0 / 15.0 之间自相矛盾，甚至同一 session 的血管与斑块通道给出不同值。
**以 README 的 5 µm 为准**（血管/斑块）、10 µm（神经元），所有不一致写进 `audit_report.md` 的警告区，
**不改数据、不静默取平均**——Z 步长直接决定深度分层的物理尺度。

### 审计中暴露并修复的缺陷

1. `content_fingerprint` 起初按文件字节算，漏掉「元数据不同、像素相同」的重复 → 改为像素指纹。
2. 重复组的 canonical 按路径字典序选，把 `3a Plaques` 选成"正本" → 改为按 identity score（是否带 mouse/fov/session 标签）排序。
3. 审计在肺数据上因 `no Z axis in axes='TYX'` 直接崩溃 → 拒绝是对的，但崩溃是错的：`z_span_um` 改为 try/except 记 0。
4. `3b - GFP neurons` 被误判为 262 页超栈 → 声明页数应与 `prod(shape[:-2])` 比较，而非只比 Z 轴。
5. manifest 存的是**相对数据根**的路径，`prepare` 却按 manifest 目录解析，导致全部跳过 → 补 `dataset_root` 与 `resolve_row_path`。

### 结论

数据可用但很小（10 卷、3 只鼠），**统计功效受限是结构性的**，bootstrap CI 只有 3 组时不可估（已用 `ci_estimable` 标注）。
泄漏边界已全部落成代码与测试：动物 / session / FOV / 通道 / 重复文件 / 时间轴六条轴各自有断言。
**本条目不含任何重建性能结论。**

### 后续动作

1. README 第 4 节列出的 `Vessels plus plaques.tif` / `Vessels plus plaques 2.tif` **不在上传包内**，需向数据提供方确认（不阻塞当前计划）。
2. 单鼠 3D 血管分割评测的 GT 来自 vesselness 阈值化（无人工标注），需在论文口径中如实说明。
3. 继续 Phase 1：SPUR-E001 无学习基线。

---

## SPUR-E001 — 无学习基线（A0 双线性 / A1 填零后 Wiener）

- **日期**：2026-10-28
- **阶段**：Phase 1（plan §6 参照基线）
- **目的**：在真实数据的三个留一鼠 fold 上建立下游一切学习方法的参照线，
  并**先证明「Wiener 不比插值好」这件事在本数据上的原因**，避免后面把插值器的上限当成方法的功劳。
- **配置**：`configs/base.yaml`，Y 方向行采样 `k=2`（测量比例 0.5），恒等 PSF，`context_depth=1`，
  `patch 512×512`（整幅），每折测试动物全部平面，共 782 个窗口/折。
- **命令**：`for f in 0 1 2; do for a in A0 A1; do python -m spur_mpm.evaluate --arm $a --fold $f --split test ...; done; done`
- **产物**：`runs/E001_{A0,A1}_fold{0,1,2}_test/`、`spur_mpm/experiments/E001_baselines/E001_table.md`

### 结果（测试动物，三折均值，括号内为跨动物 min–max）

| 指标 | A0 双线性 | A1 填零+Wiener |
|---|---|---|
| PSNR | 29.0616（27.60–30.42） | 29.0616（27.60–30.42） |
| SSIM | 0.7132（0.687–0.729） | 0.7132（0.687–0.729） |
| MAE | 0.0200（0.0172–0.0231） | 0.0200（0.0172–0.0231） |
| E_measured | 0.0000 | 0.0000 |
| E_missing | 0.0401（0.0343–0.0461） | 0.0401（0.0343–0.0461） |
| 血管 Dice | 0.8692（0.825–0.907） | 0.8692（0.825–0.907） |
| clDice | 0.8339（0.790–0.863） | 0.8339（0.790–0.863） |

**A0 与 A1 在全部三折上逐位相同（PSNR 到小数点后四位一致），这不是 bug，而是恒等 PSF 的必然结果。**
A1 的正则化解是 `argmin ||M_acq ⊙ (H x) − y||² + λ||x − x0||²`；当 `H = I` 时它在测量位置就是 `y` 本身，
在未测量位置退化为 `x0`（双线性插值）。也就是说**真实数据自带的点扩散函数让「反卷积」这一支失去意义**——
在没有额外 PSF 信息的前提下，Wiener 与插值是同一个算子。
已加测试 `test_the_two_baseline_arms_are_the_same_method_under_an_identity_psf` 锁住这一点，
并同时验证在非恒等 PSF（`sigma_xy_px=1.0, sigma_z_px=1.5`）下两臂确实分叉，说明代码路径本身是通的。
plan §6 第 405 行预警过这个局面（"Wiener 不比简单插值好 → 检查像域目标、正则化／边界／物理 PSF"），
现在的处置是：**保留插值作为先验，不伪造 PSF 优势**。

### 按深度分层（A0，band 200 µm，三折合计）

| 深度 (µm) | n | PSNR | Dice | clDice |
|---|---|---|---|---|
| 0–200 | 400 | 26.61 | 0.9085 | 0.8564 |
| 200–400 | 400 | 27.93 | 0.8625 | 0.8173 |
| 400–600 | 400 | 27.92 | 0.8551 | 0.8120 |
| 600–800 | 390 | 30.09 | 0.8661 | 0.8434 |
| 800–1000 | 266 | 32.38 | 0.8299 | 0.8156 |

**PSNR 随深度上升而血管 Dice 下降**：深层信号本身衰减，插值误差的绝对幅度随之变小（PSNR 好看），
但血管对比度不足让分割更容易断线（clDice 变差）。**结论：在本数据上 PSNR 不是深度层面的有效指标，
后续所有深度结论以血管 Dice/clDice 为准**，PSNR 只作为整体一致性检查。

### 需要注意的三件事

1. **E_measured = 0 是结构恒等式，不是性能**：插值器在测量位置原样返回 `y`。
   它的价值是回归测试——任何学习臂若 E_measured > 0，说明它把已经测到的体素改坏了。
2. **跨动物方差远大于方法差异的量级**：同一方法在三只鼠上的 PSNR 极差 2.8 dB、Dice 极差 0.08。
   只有 3 只动物，**任何小于约 1 dB / 0.03 Dice 的差异都不应被当作结论**；跨折只用范围表达，不报置信区间
   （`ci_estimable=false`，脚本已强制标注）。要主张方法优势，必须靠同一折内的配对比较。
3. 折间差异本身就是数据异质性：M2 作为测试动物时 PSNR 最低（27.60），与 M2 FOV I Session 2 只有 152 层有关。

### 结论

无学习基线在本数据上完全确定：**插值已是恒等 PSF 下的最优先验**，Dice 0.869 / clDice 0.834。
任何学习臂必须在**同一折内**显著超过这条线，否则只能说明它学到了插值。
A1 作为独立方法在本数据上不复存在，后续表格中与 A0 并列仅用于展示恒等性。

### 后续动作

1. SPUR-E002：有监督 S0/S1/S2（稠密参考作监督，用稠密验证集选模）。
2. SPUR-E003：自监督 MMS SS0/SS1/SS2（holdout 只用真实测量，选模只看 held-out 测量损失）。
3. 若要恢复 Wiener 的反卷积意义，需从数据估计 PSF（血管边缘的横向剖面）；已记录待决，不阻塞 E002/E003。

---

## SPUR-E003a — 自监督泄漏闸门（真实数据）

- **日期**：2026-10-28
- **阶段**：Phase 3 前置（plan §2.3「严格自监督口径」、SPUR-E003「防伪自监督」）
- **目的**：「零稠密标签」是一个需要被证明的断言，不是一句声明。在任何 SS 臂训练之前，
  先在**真实数据**上把这个断言变成可以失败的检查。
- **产物**：`spur_mpm/scripts/check_selfsup_leakage.py`、
  `spur_mpm/experiments/E003_selfsup/leakage_check.json`
- **覆盖面**：3 个 fold × (train/val/test 三种角色) = **30 卷**，每卷 3 个平面 × 3 个 holdout seed = **270 次探测**

### 三项检查

1. **盲性（主检查）**：定位 holdout 体素，把这些体素的**稠密像素替换为垃圾值**（`(x+0.37) mod 1`，保证与原值不同），
   其余一切不变，重建 item，要求网络输入**逐位相同**。臂能读到的所有信息都在 `inputs` 张量里
   （`y/mask/x0`，`U` 由 `y`、`mask` 推出），因此输入不变 = holdout 无路可达模型。
2. **正对照（让检查有可能失败）**：把**测量位置**（`M_in \ M_hold`）的体素同样替换为垃圾值，要求输入**必须改变**。
   没有这一项，第 1 项即使写成恒真也会"通过"。
3. **目标必须是测量**：holdout 目标必须等于该位置的实际采集值；不能是稠密参考（那等于有监督），
   不能是全零（那等于训练网络输出全黑）。同时校验 `M_in`、`M_hold` 恰好划分 `M_acq`。

### 结果

```
volumes checked: 30  planes/volume: 3
holdout share of M_acq: 0.500-0.500 (requested 0.5)
OK: holdout is blind on real data
```

270 次探测全部通过：输入对 holdout 完全不敏感、对测量位置敏感、目标全部是真实测量、划分比例精确 0.500。

### 闸门自身的测试（`spur_mpm/tests/test_leakage_harness.py`）

只证明「检查通过」不够，还要证明「检查会失败」。新增 3 个测试：

- 诚实构造 → 通过；
- 构造器把 `hold_target` 偷偷拼进 `inputs`（模拟真实泄漏）→ 检查**必须**报 `inputs changed`；
- 构造器把 holdout 目标置零 → 检查**必须**报 `all zero` / `not the acquired`。

### 结论

自监督路径在真实数据上的泄漏边界成立，且验证它的工具本身是可证伪的。
这不是性能结论，是口径结论：后续 SS 臂的「无标签」主张有这个工件支撑。

### 后续动作

1. E003b：SS0/SS1 训练，选模**只用** held-out 测量损失（验证 FOV 的 holdout，不读稠密参考）。
2. 冻结权重之后才允许 `evaluate.py` 读稠密参考，并在 run 目录留下 `select_metric=holdout_loss` 痕迹。

---

## SPUR-E000b — 逐层深度剖面（真实数据的可用性约束）

- **日期**：2026-10-28
- **阶段**：Phase 0 补全（plan §2.2 要求记录「Z 各层信号强度、背景、饱和像素、漂移」）
- **产物**：`spur_mpm/scripts/profile_depth.py`、
  `spur_mpm/experiments/E000_data_audit/depth_profile.{csv,json,md}`
- **方法**：对 10 卷全部平面（共 1866 层）逐层统计 mean / median / p99 / std / 精确零值比例 / 满量程比例，
  深度用 `z_step_um` 换算为物理 µm。

### 结果 1：没有数据缺陷

**全部 10 卷、1866 层的满量程（65535）像素比例为 0.00%**，也没有空白层。
这一条很关键：16-bit 上限若被触顶，标定 `offset+scale` 会把一段真实取值压到 1.0，
之后所有指标测的都是截断而不是方法。本数据不存在这个问题。

### 结果 2：每一卷都存在 >14% 的深度衰减（分析约束，不是缺陷）

| 卷 | 层数 | 首层 p99 → 末层 p99 | 比值 | 首层 mean → 末层 mean | 饱和 |
|---|---|---|---|---|---|
| M1_FOVI_S1 | 202 | 48303 → 34005 | 0.70 | 43411 → 32958 | 0.00% |
| M1_FOVI_S2 | 188 | 44563 → 32896 | 0.74 | 39701 → 31496 | 0.00% |
| M1_FOVII_S1 | 202 | 48878 → 35531 | 0.73 | 43211 → 33763 | 0.00% |
| M1_FOVII_S2 | 190 | 42714 → 35766 | 0.84 | 37602 → 33965 | 0.00% |
| M2_FOVI_S1 | 201 | 40349 → 34602 | 0.86 | 36175 → 33372 | 0.00% |
| M2_FOVI_S2 | 152 | 48004 → 37071 | 0.77 | 41416 → 34276 | 0.00% |
| M2_FOVII_S1 | 203 | 47390 → 37033 | 0.78 | 43113 → 34435 | 0.00% |
| M2_FOVII_S2 | 202 | 48153 → 34887 | 0.72 | 42925 → 33463 | 0.00% |
| M3_FOVI_S1 | 158 | 42332 → 34594 | 0.82 | 36658 → 33355 | 0.00% |
| M3_FOVI_S2 | 168 | 46457 → 34118 | 0.73 | 41899 → 33174 | 0.00% |

（深度跨度 755–1010 µm。注意背景/均值都在 ~33000 附近，说明这批数据有较大的直流偏置，
这正是标定用分位数 offset 而不是 min–max 的原因。）

### 结论（对后续实验的硬约束）

1. **深度分层必须报告，不能把整卷平均当作结论**：同一卷内 p99 相差 14–30%，
   SNR 区间被混合，整卷平均会把"浅层好、深层差"平均成"都还行"。
   这直接解释了 SPUR-E001 中 PSNR 随深度上升、血管 Dice 随深度下降的现象。
2. **不存在饱和**，所以标定与指标不受截断污染；`saturated_fraction()` 在真实数据上恒为 0。
3. 直流偏置较大（背景 ~33000/65535），标定必须以分位数为准；用 min–max 会把噪声放大成"对比度"。

### 后续动作

1. 后续所有学习臂的深度分层表沿用 `collect_results.py --band-um 200`，与本节口径一致。
2. 若要讨论"哪一层最难"，以最深一档（600–1000 µm）的 Dice/clDice 为准。

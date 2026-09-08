# ThreePM真实数据采集最终版SOP（双光子先行，兼容三光子扩展）

## 1. 文档目的与适用范围

本SOP用于指导`ThreePM_RealData_v2`真实采集数据构建，当前优先覆盖双光子（2P）花粉与beads样本，后续可无缝扩展到三光子（3P）与其他生物样本。

适用对象：
- 显微成像实验执行人员
- 数据标注/整理人员
- 算法训练与评估人员

核心目标：
- 构建可用于稀疏重建、低SNR去噪、联合退化重建的高质量配对数据
- 保障数据的可重复、可追溯、可用于统计检验
- 降低光漂白、漂移、配准误差对结论的干扰

---

## 2. 实验总目标与阶段拆分

### 2.1 总体实验目标
- 目标A：评估不同稀疏采样模式对重建质量的影响（含各向异性）
- 目标B：构建可控SNR梯度数据，评估低信噪场景鲁棒性
- 目标C：构建“稀疏+低SNR”联合退化数据，模拟真实快速低剂量成像场景

### 2.2 阶段规划
- Phase 1（Pilot，约1周）：验证采集流程、阈值、配准与质控体系
- Phase 2（核心数据集，2-3周）：构建训练/验证/测试主数据
- Phase 3（扩展，1-2周）：联合退化扩展与3D体数据补充

---

## 3. 角色分工与交付物

- 实验负责人（PI/高级研究员）
  - 确认目标、样本范围、功率/停留时间安全上限
  - 批准最终参数和质控阈值
- 采集执行者
  - 按SOP采集并记录日志
  - 发现异常立即标注并中止视野
- 数据管理员
  - 执行命名规范、目录整理、校验脚本
  - 产出metadata与QC报告
- 算法工程师
  - 进行加载、训练、评估闭环验证
  - 反馈需要补采或重采的数据类型

主要交付物：
- 原始图像（tif）
- 采样mask（npy）
- 每视野metadata（json）
- 质控报告（csv/pdf）

---

## 4. 采集前准备（必须完成）

### 4.1 仪器与环境
- 显微系统预热完成，激光稳定
- 物镜、光路、PMT状态检查通过
- 舞台漂移检测通过（空载或标准样本）
- 当前实验参数写入当日日志（日期、操作者、仪器编号）

### 4.2 标定采集
- PSF标定（beads 100nm z-stack）
- 暗场/噪声标定（固定PMT增益下）
- 扫描模式mask文件生成并锁定版本（含seed）

### 4.3 样本准备
- 样本ID编码（不可重名）
- 样本制备记录完整（介质、荧光类型、时间）
- 样本安装后完成焦平面和视野稳定检查

---

## 5. 命名规范与目录规范

## 5.1 根目录结构

```text
ThreePM_RealData_v2/
├── 1_Calibration/
├── 2_SparseReconstruction/
├── 3_LowSNR_Denoising/
├── 4_Joint_SparseLowSNR/
└── 5_3D_Volumetric/
```

### 5.2 文件命名格式（统一）

```text
{sampleType}_{sampleID}_{fovID}_{mode}_{param}_{rep}.tif
```

示例：
- `pollen_P003_F012_dense_power15mw_r01.tif`
- `beads_B007_F004_sparse2x_checkerboard_power8mw_r01.tif`
- `pollen_P003_F012_dense_power15mw_post.tif`

约束：
- `sampleType`：`pollen`/`beads`（后续可拓展）
- `fovID`：固定长度编号，如`F001`
- `rep`：重复采集编号，建议`r01/r02`
- 所有文件名仅使用小写英文、数字、下划线

---

## 6. 核心采集流程（逐步执行）

## 6.1 每日启动流程（Day Start Checklist）
1. 创建当日采集日志：`acq_log_YYYYMMDD.md`
2. 写入仪器状态、操作者、样本批次
3. 采集1个标准视野做快速验证（SNR、漂移、曝光）
4. 若验证通过，进入正式采集；否则排查后重测

### 6.2 每个视野（FOV）标准流程（强制）
1. 选择并锁定FOV，记录`sampleID + fovID`
2. 采集`Dense-High-Pre`（作为前基准）
3. 执行该FOV对应变体序列（见6.3/6.4/6.5）
4. 采集`Dense-High-Post`（作为后基准）
5. 立即运行快速QC：
   - 配准偏移
   - 强度漂移（Pre vs Post）
   - 文件完整性
6. QC失败则标记`REJECTED`并决定重采

### 6.3 稀疏模式实验（目标A）
最小可执行集合（Pilot推荐）：
- `GT_dense_power15mw`
- `Sparse_2x_rowskip_power15mw`
- `Sparse_2x_colskip_power15mw`
- `Sparse_2x_checkerboard_power15mw`

顺序建议（同FOV）：
1. `dense high pre`
2. `2x rowskip`
3. `2x colskip`
4. `2x checkerboard`
5. `dense high post`

说明：
- 2x模式只改采样模式，其他参数固定
- 重复次数至少1次，关键样本建议2次（r01/r02）

### 6.4 低SNR梯度实验（目标B）
推荐采用激光功率梯度（固定dwell time、PMT）：
- 功率：`1/2/4/8/15 mW`
- 建议顺序：低到高，最后补`Dense-High-Post`

顺序建议：
1. `dense high pre`
2. `dense power1mw`
3. `dense power2mw`
4. `dense power4mw`
5. `dense power8mw`
6. `dense power15mw`
7. `dense high post`

说明：
- 若出现明显漂白，立即中断并标记该FOV

### 6.5 联合退化实验（目标C）
Pilot最小组合（建议先做）：
- 稀疏：`2x checkerboard`
- SNR：`2/4/8 mW`
- 加入对应dense对照

顺序建议（光剂量从低到高）：
1. `dense high pre`
2. `sparse2x_power2mw`
3. `sparse2x_power4mw`
4. `sparse2x_power8mw`
5. `dense_power2mw`
6. `dense_power4mw`
7. `dense_power8mw`
8. `dense_power15mw`
9. `dense high post`

---

## 7. 参数推荐与边界

默认推荐（2P阶段）：
- 分辨率：`512 x 512`
- bit depth：`16-bit`
- dwell time：`10 us/pixel`
- PMT gain：`750`
- 激光波长：按样本与系统建议值（记录在metadata）
- frame averaging：建议固定（如3）

边界条件：
- 若样本光漂白快，优先降低重复次数，不盲目提高功率
- 若SNR过低不可用，记录“不可用阈值”而不是强行采满

---

## 8. 质控（QC）标准

### 8.1 必过项（Hard Rules）
- 文件命名与目录结构合规
- 同一FOV变体齐全（缺失即失败）
- GT配对可定位且可配准
- metadata字段完整且可解析

### 8.2 数值阈值（建议初版，可在Pilot后固化）
- 漂移阈值：`<= 1 pixel`（或`<= 0.1 um`，按像素尺寸换算）
- 漂白阈值：`Dense-High-Pre`与`Dense-High-Post`平均强度变化`<= 10%`
- SNR梯度单调性：至少满足总体趋势下降（允许局部波动）

### 8.3 异常分级
- `A`：可直接用于训练/测试
- `B`：可用于训练，不用于关键评估
- `C`：仅归档，不进入建模

---

## 9. metadata记录规范

每个FOV至少一个`metadata.json`，包含：
- 仪器信息（型号、物镜、激光、探测器）
- 样本信息（类型、制备、介质）
- 成像参数（分辨率、像素尺寸、bit depth、gain、dwell）
- 变体列表（scan_mode、sparsity_factor、laser_power、mask路径）
- 质量指标（漂白率、漂移量、稳定性）

建议新增字段：
- `qc_grade`：A/B/C
- `qc_notes`：异常说明
- `operator`：采集人员
- `session_id`：当天批次编号

---

## 10. 数据集切分与防泄漏规则

- 按FOV切分，不按图像切分，防止同FOV泄漏到train/val/test
- 建议先固定测试集（不变），其余再分train/val
- 样本分布保持平衡：pollen与beads在各split比例尽量一致

推荐比例：
- Train/Val/Test = `70/15/15`（可按样本量微调）

---

## 11. 执行清单（可直接勾选）

### 11.1 Day级清单
- [ ] 仪器预热与标定完成
- [ ] 当日日志创建
- [ ] 标准样本试采通过
- [ ] 参数版本锁定（功率、dwell、gain、mask）

### 11.2 FOV级清单
- [ ] Dense-High-Pre已采集
- [ ] 稀疏模式序列完成
- [ ] SNR序列完成
- [ ] 联合退化序列完成（若当日计划包含）
- [ ] Dense-High-Post已采集
- [ ] 快速QC通过/失败已标记
- [ ] metadata已写入

### 11.3 Day End清单
- [ ] 文件完整性校验通过
- [ ] 重名/漏采检查通过
- [ ] QC汇总表导出
- [ ] 备份完成（本地+服务器）

---

## 12. 风险与应急策略

- 风险1：光漂白快速累积
  - 策略：减少重复、降低功率、缩短单FOV序列
- 风险2：机械漂移导致配准失败
  - 策略：缩短序列时长、增加中间参考帧、稳定样本固定
- 风险3：参数漂移（多人操作）
  - 策略：每日参数锁定，禁止临时改参数不记录
- 风险4：命名混乱导致配对失败
  - 策略：采集后立即自动校验脚本检查

---

## 13. 与训练代码对接建议（当前仓库）

当前代码默认`Training/high`和`Training/low`配对结构。要对接本SOP数据，建议后续实现：
- 统一索引器：根据`metadata.json`建立`GT -> degraded`配对映射
- 新参数：
  - `--dataset_mode {sparse,lowsnr,joint}`
  - `--pattern {rowskip,colskip,checkerboard,grid,random}`
  - `--snr_level {power1mw,power2mw,power4mw,power8mw,power15mw}`

---

## 14. 版本管理

- 文档版本：`v1.0-final`
- 生成日期：`2026-04-16`
- 适用阶段：2P先行采集与建库
- 维护建议：Pilot结束后冻结`v1.1`（更新阈值和参数）


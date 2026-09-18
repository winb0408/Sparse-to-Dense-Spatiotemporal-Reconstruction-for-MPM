# PSRAS 修改计划（Protocol v3）

> PSRAS = **P**hysics-**S**ampling-**R**esidual **A**ware **S**ynthesis
> 基于对 threePM（`dev` 分支）的 ChatGPT 审阅结论 + 独立代码核实 + 经验修正整理而成。
> 本文是执行依据；实验过程与结论记录见 [PROJECT_LOG.md](PROJECT_LOG.md)。
>
> 原则：**不修改 threePM 根目录任何旧代码**。所有新代码、新实验、新结果均在 `PSRAS/` 下；
> 旧结果一律视为 `legacy / exploratory`，不进入新论文正文。

---

## 1. 审阅结论核实（2026-09-09，针对 dev 分支源码）

| # | 论断 | 证据位置 | 核实结果 |
|---|---|---|---|
| C1 | DINOv2 perceptual loss 无梯度反传 | `loss.py:198` `_extract` 整体 `@torch.no_grad()`，pred/target 特征都在 no_grad 下提取 | **属实**，loss 数值只是记录，从未优化重建网络 |
| C2 | `wiener_init_3d` 是逐帧 2D Wiener | `dataset.py:453-465`，`psf_sigma_axial` 参数被忽略 | **属实** |
| C3 | 2D/3D Swin 均无 shifted-window attention mask | `models/swin_unet_2d.py:174`、`models/video_swin_unet.py:210`，只有 `roll` 无区域 mask | **属实**，非标准 SW-MSA |
| C4 | normalization 依赖 GT 统计量 | `dataset.py:1485,1638,1759`（GT/patch min-max 决定输入尺度） | **属实**，属 target-dependent preprocessing |
| C5 | 预插值数据被当作 raw sparse | `scripts/prepare_cidc25_sparse3d.py`：Z 模式保存 `zoom` 插值后体数据；XY/XYZ 保存 warm-start 后结果 | **属实**，三个 regime 的 inverse problem 定义不一致 |
| C6 | `baseline_input_z_K4` 74 dB / SSIM 1.0 可疑 | 来自 legacy eval 输出 | **不重查旧账**，直接在新 protocol 下重建 baseline，该问题自然消失 |

C1–C5 全部独立核实为真，构成 P0 修复项的事实基础。

---

## 2. 对 GPT 方案的经验修正与补充

GPT 的总体方向（物理一致 forward model → anisotropic 3D Wiener → sampling-aware prior → residual 重建 → 统一 3D 网络）正确，以下是需要修正/收紧的点：

### M1. Wiener 与 "无 PSF 的纯 masking" 不匹配 —— 必须先定死 forward model
GPT 的 Protocol B 写作 `Y = M ⊙ (H*X + n)`（含 PSF），但 CIDC25 历史 sparse 只是纯遮蔽（无 blur）。
**纯遮蔽下做 Wiener 反卷积在物理上是错配的**——缺失像素不是光学模糊造成的。
决定：
- **新协议的 acquisition simulation 必须包含 PSF（H）与噪声（n）**，即统一使用 `Y = M ⊙ (H*X + n)`。这既更接近 3PM 真实成像，也让 Wiener 有明确物理角色。
- 同时保留一条 **no-PSF 消融线**（`H = δ`，即纯 inpainting），作为 ablation 而非主线。
- 若 `H = δ`，warm-start 退化为 zero-fill + 插值，此时不得声称 "deconvolution"。

### M2. Z-sparse 下 3D Wiener 的作用要讲清楚
轴向缺失切片是纯 inpainting 问题，OTF 沿 z 的反卷积**不解决**缺层；
3D anisotropic Wiener 的收益主要体现在 **XY / XYZ sparse**（横向去模糊 + 利用轴向邻域）。
- Z 模式的 warm-start：对 measured slices 做横向 Wiener，缺失切片用轴向插值（在 Wiener 之后插值，而非之前）。
- 论文叙事中不要声称 "3D Wiener 解决 Z sparse"，而是 "anisotropic inversion for lateral degradation + interpolation for axial gaps"。

### M3. Data Consistency loss 的适用范围
`L_DC = ‖M ⊙ (H*X̂ − Y)‖` 在**含 PSF 的 forward model**下才有意义（去模糊后重新解释观测）。
纯 masking 且 `H = δ` 时，measured 像素上的 DC 退化为恒等约束，几乎无信息。
→ DC loss 只在含 PSF / Protocol A 数据上启用；λ_DC 从 {0, 0.01, 0.05, 0.1} 消融。

### M4. normalization 的具体方案（比 GPT 更可执行）
固定标定常数，而不是逐样本统计：
- uint16 物理量程 → `float32 [0,1]`（除以 65535）；
- 若各 stack 增益差异大，用 **训练集全局标定常数**（如训练集 99.9 分位数），存入 metadata，train/val/test/inference 共用同一常数；
- 任何阶段禁止读取当前样本 GT 的 min/max。

### M5. residual 学习的适用性判断
z-K2 场景 X0 已达 ~51 dB（legacy 证据），residual 信号极小，网络可能学成恒等——这是**预期行为**而非失败；
XY/K8 等重度退化下 residual 项才是主要贡献。按 sampling severity 分层报告 residual 幅度 ‖ΔX‖/‖X0‖，本身就是一个分析维度。

### M6. 统一 3D 网络采用轴向分离注意力（GPT 方案 A），分阶段引入
- Phase 1–2 先用**修复后的 2D Swin-UNet** 打通 protocol（它是 B0–B6 消融的载体）；
- Phase 3 再实现 unified axial-3D net（spatial window attention + axial/z attention，T∈{1,3,5,9,16}，query mask Q 控制输出目标）；
- T=1 时 axial attention 自然退化为无操作，不需要特判。
- 保留 spatial-only downsampling（不对 z 做 /2），包装为 anisotropy-aware 设计。

### M7. stack-level split 提前到 Phase 1
GPT 放在 P1，但这直接决定所有数字的可信度且改造成本低，**提前**：
CIDC25 按 stack 划分 train/val/test，禁止同一 stack 的 patch 跨集合。

### M8. 评估
- 采纳 measured/missing split（`E_measured`、`E_missing`）为核心新增指标，成本极低且直击方法故事；
- paired 统计按 sample_id merge 后再 Wilcoxon；
- 删除 FwPSNR；MS-SSIM 换用验证过的标准实现；
- foreground 定义改用 GT 形态学阈值（Otsu on GT, train 集固定参数），不用逐样本 75 分位。

---

## 3. PSRAS 目录结构（目标态）

```
PSRAS/
├── PLAN.md                    # 本文件
├── PROJECT_LOG.md             # 项目思路 + 每次实验改动的原理与结论（持续追加）
├── physics/
│   ├── psf.py                 # anisotropic 3D Gaussian PSF（物理单位 σ_xy_um / σ_z_um）
│   ├── wiener.py              # build_gaussian_psf_3d / build_otf_3d / wiener_deconv_3d
│   ├── sampling.py            # XY/Z/XYZ mask 生成（Protocol A/B 统一）
│   └── acquisition.py         # simulate_acquisition: H, S, n, resampling 统一算子
├── data/
│   ├── observation.py         # build_observation -> (Y_grid, M, X0, C_s, U_p, metadata)
│   ├── cidc25.py              # stack-level split datasets
│   └── transforms.py          # dihedral 等（无 GT 统计依赖）
├── models/
│   ├── attention.py           # 带正确 shift mask 的 window attention（2D/3D 共用）
│   ├── swin_unet_2d.py        # 修复版 2D Swin-UNet（Phase 1/2 载体）
│   ├── unified_3d.py          # axial-3D 统一网络（Phase 3）
│   └── prior_fusion.py        # sampling-aware prior encoding + gated fusion
├── losses/
│   ├── reconstruction.py      # L1 + MS-SSIM
│   └── data_consistency.py    # masked measurement-domain DC（含 PSF 时启用）
├── metrics/
│   ├── image_metrics.py       # PSNR/SSIM/MS-SSIM/MAE
│   ├── structure_metrics.py   # measured/missing split, FG metrics, profile corr
│   └── statistics.py          # 按 sample_id 配对 Wilcoxon / bootstrap CI
├── scripts/
│   ├── prepare_sparse_v3.py   # 只保存 GT / raw Y / M / metadata
│   ├── train.py
│   ├── evaluate_v3.py
│   └── sanity_checks.py       # 单元/合成 sanity test 入口
├── tests/
│   ├── test_wiener.py
│   ├── test_acquisition.py
│   └── test_normalization.py
└── runs_v3/                   # 所有新实验输出（含完整 config + seed）
```

---

## 4. 分阶段执行计划

### Phase 0 — 冻结与基建（0.5 天）
- [ ] threePM 旧结果不删除、不改动；在 PROJECT_LOG.md 记录 legacy 结论清单
- [ ] 搭建 `PSRAS/` 骨架；`tests/` 就位
- [ ] 固定随机种子与 config 保存规范（每次 run 落盘完整 yaml/json config）

### Phase 1 — 科学正确性修复（P0，2–4 天，不含大模型改动）✅ 2026-09-09 完成
- [x] `physics/acquisition.py`：统一 `Y = M⊙(H*X+n)`，严格分离 ①光学退化 ②采样 ③噪声 ④几何重采样 ⑤warm-start
- [x] `scripts/prepare_sparse_v3.py`：训练集标定 + 归一化转换 + stack-level split；
      sparse 观测改为 dataset runtime 生成（DEC-007，supersedes 原离线保存方案）
- [x] 去除一切 GT 依赖 normalization（M4 方案，IntensityCalibration）
- [x] `physics/wiener.py`：真 3D anisotropic Wiener（解析 Gaussian OTF，物理单位 σ）；
      z 模式 warm-start 按 DEC-003（先反卷积 measured 切片、再轴向插值）
- [x] `losses/data_consistency.py`：masked measurement-domain DC（M3 适用范围，identity 算子拒绝使用）
- [x] 删除 DINOv2 loss（新代码中不再实现）
- [x] `models/attention.py`：补齐 shifted-window attention mask（2D/3D 共用实现 + 相对位置偏置）
- [x] **单元测试**：26 用例全部通过（acquisition 恒等性 / mask 几何与确定性 / Wiener 频域恢复 /
      z warm-start 精确插值 / SW-MSA 跨区域阻断 / DC measured-域一致性 / 标定无泄漏 /
      C_s-U_p 语义与单调性）
- [x] **stack-level split** 落地（train={A1,B1,C2,D2}，val=F0[0:750)，test=F0[750:1500)）
- [x] 数据就绪：`runs_v3/data/cidc25_v3/`（标定裁剪审计：train/val 均 ~0.03–0.04%，可忽略）

### Phase 2 — 新 baseline 与消融矩阵（核心实验）
在修复后的 2D Swin-UNet（XY）/ 2.5D（Z）上重建证据链：

```
B0  raw sparse → network（无 mask）
B1  + sampling mask M
B2  + bilinear warm-start X0
B3  + true 3D Wiener X0（替换 bilinear）
B4  + sampling-aware prior fusion（C_s + U_p，gated）
B5  + residual 学习 X̂ = X0 + ΔX
B6  + data consistency loss（仅含 PSF 线）
```

- 采样：K ∈ {2,4,8}，方向 ∈ {XY, Z, XYZ}（主表只取 2D-XY / 2.5D-Z / 3D-XYZ 三条代表线，其余进附录）
- 指标：PSNR / SSIM / MS-SSIM / MAE / E_measured / E_missing / FG-PSNR
- 每个 B 步骤单独成 run，全部带完整 config
- **判定标准（预注册）**：B3>B2 且 B5>B3 且 B4>B1 才能写进论文主张；否则回退修改方法而不是改叙事

### Phase 3 — 统一 3D 网络
- [ ] `models/unified_3d.py`：spatial window attention + axial attention，T ∈ {1,3,5,9,16}，query mask Q
- [ ] 验证同一权重/架构下 Q=[1] / [0,1,0] / 全 1 三种 regime
- [ ] "PSNR vs axial context T" 曲线（context scaling 分析，GPT §27）

### Phase 4 — 稳健性实验
- [ ] sampling severity 曲线：PSNR vs r ∈ {50%, 25%, 12.5%, 6.25%}
- [ ] photon budget：λ_photon ∈ {5,10,20,50} + read noise 扫描
- [ ] Wiener λ 敏感性
- [ ] no-PSF（纯 inpainting）消融线（M1）

### Phase 5 — 泛化
- [ ] Same-domain：CIDC25 → CIDC25 val（主结论）
- [ ] Zero-shot：CIDC25 → MDC25
- [ ] External acquisition transfer：CIDC25 → pollen（标注 acquisition-protocol difference）

---

## 5. Benchmark 与数据协议（Protocol v3 定义）

### 统一 forward model
```
Protocol A（真实采集，pollen 等）:
  X → H*X → S_K 采样 + 噪声 → Y_low → 几何重采样 R → Y_grid   （M 用 support 映射，非 bilinear）

Protocol B（dense-grid 模拟，CIDC25/MDC25）:
  Y = M ⊙ (H*X + n)     （含 PSF；no-PSF 作为消融线）

两者统一网络输入: (Y_grid, M, X0, C_s, U_p)
```

### 概念命名（严格区分，写入所有代码与文档）
| 符号 | 名称 | 含义 |
|---|---|---|
| M | measurement support / sampling mask | 该位置是否有真实测量（**不是** uncertainty） |
| C_s | sampling confidence | G_σ * M，距真实采样点的远近 |
| U_p | physics uncertainty | 由 OTF 衰减 / Wiener 放大导出，K/(|H|²+K) |
| X0 | physics warm-start | W(Y; H, M)，Wiener 或插值 |
| Q | query/target mask | 2D=[1]，2.5D=[0,1,0]，3D=全 1 |

### Loss
```
L = L1 + 0.5·L_MS-SSIM + λ_DC·L_DC        （λ_DC ∈ {0, 0.01, 0.05, 0.1} 消融，仅含 PSF 线）
```

---

## 6. 风险与注意

1. **不要在 Phase 1 完成前跑任何"正式"实验**——protocol 未冻结前产出的数字都会成为新的 legacy。
2. 3D Wiener 内存开销：OTF 为实可分离 Gaussian 时用 FFT 逐 batch 做；体积大时先在 patch 上验证。
3. B5（residual）在 K=2-Z 下 ΔX 很小是预期，勿据此否定 residual 设计（见 M5）。
4. 与 legacy 结果**不做任何同表对比**（数据协议不同，数值不可比）；对比只在 PSRAS 内部进行。
5. 每次实验改动必须在 PROJECT_LOG.md 追加一条记录（原理 → 改动 → 结论），保持证据树可追溯。

---

## 7. 执行优先级（供 codex / 后续会话使用）

```
[P0] 01 acquisition 统一算子        02 prepare_sparse_v3        03 去 GT normalization
     04 anisotropic 3D Wiener       05 masked DC loss           06 删 DINO（不带入新代码）
     07 shift attention mask        08 stack-level split        09 sanity/单元测试
[P1] 10 B0–B6 消融矩阵              11 sampling confidence C_s  12 physics uncertainty U_p
     13 prior fusion 升级           14 residual 学习            15 统一 3D 网络 + Q mask
[P2] 16 K/噪声/λ 扫描               17 MDC25 zero-shot          18 pollen transfer
     19 论文图表
```

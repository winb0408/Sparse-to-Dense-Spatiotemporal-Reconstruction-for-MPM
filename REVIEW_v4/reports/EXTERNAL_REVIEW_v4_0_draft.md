# 🧬 PSRAS 下一阶段项目修正与重构方案

> **Version:** v4.0-draft  
> **Date:** 2026-09-21  
> **Scope:** `PSRAS/` branch and all subsequent sparse-to-dense reconstruction experiments  
> **Status:** 🔴 Protocol reset → 🟡 corrected baseline → 🟢 main experiments → 🔵 unified framework → 🟣 real-acquisition validation

---

## 0. 📌 Executive Summary

本阶段不再对现有 PSRAS 代码和结果做局部补丁，而是进行一次完整的 **实验协议重置（protocol reset）+ 方法主线重构（method re-design）+ 数据划分修复（split correction）**。

项目的核心科学问题由原来的：

> **Sparse observation → deep reconstruction → dense image**

升级为：

> **在 3PM 成像时间、曝光和光剂量受限的情况下，如何设计跨采集维度的稀疏采样策略，并利用采样结构与物理可恢复性进行高保真 sparse-to-dense reconstruction？**

新的总体定位建议为：

> **Sampling–Physics–Reconstruction Co-design for Fast Sparse-to-Dense 3PM Imaging**

核心思想是将：

- **采集维度（acquisition axis）**
- **稀疏预算（sampling budget）**
- **物理前向模型（PSF / acquisition operator）**
- **physics-based prior**
- **sampling-aware conditioning**
- **shared reconstruction backbone**
- **实际 acquisition time / photodose**

统一到一个可验证的研究框架中。

### 🔑 本阶段最重要的五条原则

1. **Wiener vs. Bilinear** 与 **Wiener prior vs. Direct Network** 必须作为两个独立科学问题重新验证。
2. 当前 z 主线的 `B3` 虽然调用了 3D Wiener 函数，但实际 warm-start 路径并不是严格意义上的 full 3D Wiener；因此后续不得继续直接声称“z reconstruction 使用真正 3D Wiener”。
3. **Y / YZ / XYZ 是 acquisition regimes；2D / 2.5D / 3D 更适合描述网络上下文或计算形式，不再作为项目主分类。**
4. **XYZ（空间体数据）与 XYT（时序数据）不可简单混为一类。** 二者可共享 spatial backbone，但 physics prior 与 missing-axis interaction 应分别建模。
5. 当前 CICD25 的 F0 temporal split 可用于 temporal hold-out，但 **不能继续表述为严格 independent stack-level test**；最终主结果需要真正独立的 stack / session / sample-level test。

---

# 1. 🎯 项目总体目标重定义

## 1.1 科学目标

项目最终不只是训练一个 reconstruction network，而是研究：

> **给定有限 acquisition budget，如何在不同采集方向上分配稀疏度，并利用 physics- and sampling-aware reconstruction 恢复高质量图像。**

可以将总体优化目标写成：

\[
\max_{\mathcal S}\; \frac{Q(\hat X_{\mathcal S})}{T_{\mathrm{acq}}(\mathcal S)}
\]

并进一步考虑光剂量：

\[
\max_{\mathcal S}\; Q(\hat X_{\mathcal S}),
\qquad
\min D_{\mathrm{laser}}(\mathcal S)
\]

其中：

- \(\mathcal S\)：采样策略；
- \(Q\)：重建质量；
- \(T_{\mathrm{acq}}\)：实际采集时间；
- \(D_{\mathrm{laser}}\)：曝光/光剂量 proxy。

因此论文的最终核心不应只围绕 **PSNR improvement**，而应围绕：

> **quality–efficiency trade-off**

以及更进一步的：

> **sampling-budget allocation**。

---

# 2. 🧭 新的项目科学主线

```text
Acquisition constraint
        ↓
Sparse sampling design
        ↓
Measurement mask M
        ↓
Physics-aware inverse / prior X0
        ↓
Sampling-conditioned fusion
        ↓
Shared spatial reconstruction backbone
        ↓
Axis-specific interaction
        ↓
Dense reconstruction
        ↓
Quality ↔ acquisition time ↔ photodose
        ↓
Sampling budget optimization
```

最终形成：

> **sampling–physics–reconstruction co-design**

这一定位比“更换一个 Transformer backbone”更适合作为论文主创新框架。

---

# 3. 🔴 P0：现有项目必须先修正的问题

> **规则：P0 问题未闭环前，不新增复杂模型，不扩展大量主实验。**

## 3.1 数据划分问题：CICD25 F0

### 当前情况

当前数据结构为：

```text
Train: A1 / B1 / C2 / D2
Validation: F0 temporal subset
Test: F0 another temporal subset
```

这意味着 val 和 test 虽然时间区间不重叠，但仍然来自同一个 F0 stack。

### 需要修正

当前 F0 split 可以保留为：

> **temporal hold-out evaluation**

但不能继续定义为：

> independent stack-level test

### 最终推荐 split

```text
Training stacks
A1 / B1 / C2 / D2 / ...
        ↓
Validation stack/session
Held-out acquisition session
        ↓
Independent test stack/session
Held-out stack or animal/session
```

### 最低验收标准

- [ ] train / val / test 明确区分 stack-level independence
- [ ] test 不与 train 共享同一 acquisition stack
- [ ] test 不参与 normalization calibration
- [ ] test 不参与模型、λ、采样率或超参数选择

---

## 3.2 🔴 `reval.py` 的 measurement mask 问题

当前 `reval.py` 使用 observation value 是否非零来推断 measurement support：

```python
x.view(...)[0][q] != 0
```

该做法不成立，因为：

\[
Y_i = 0 \not\Rightarrow M_i = 0
\]

实际 measured position 完全可能为零或接近零。

### 必须改为

直接使用真实 mask：

```python
M
```

并严格计算：

\[
E_{\mathrm{measured}}
=
\frac{\|M\odot(\hat X-X)\|}{|M|}
\]

\[
E_{\mathrm{missing}}
=
\frac{\|(1-M)\odot(\hat X-X)\|}{|1-M|}
\]

### 验收标准

- [ ] all split-region metrics use explicit `M`
- [ ] no code path infers support from `Y != 0`
- [ ] re-evaluation scripts and main evaluation scripts use identical mask semantics

---

## 3.3 🔴 SSIM 实现问题

当前 Gaussian window 实现实际是：

```text
1 × 1 × 1 × K
```

再用于 `conv2d`，因此只沿一个空间方向进行 smoothing，并非标准二维 Gaussian window。

### 应修正为

真正二维 Gaussian kernel：

\[
G(x,y)=G_x(x)G_y(y)
\]

对应：

```text
1 × 1 × K × K
```

### 重要性

SSIM 修正后，旧实验 SSIM 与新实验 SSIM 不应直接混合统计。

因此：

> **SSIM implementation freeze 后，重新跑所有最终 quantitative experiments。**

---

## 3.4 🔴 当前 z-mainline 的“真 3D Wiener”命名问题

代码中确实存在 mathematically genuine 3D FFT Wiener：

\[
X_W
=
\mathcal F^{-1}
\left[
\frac{H^*}{|H|^2+\lambda}
\mathcal F(Y)
\right]
\]

但当前 z-sparse `warm_start()` 的主路径实际上是：

```text
measured slices
    ↓
lateral 2D Wiener
    ↓
axial interpolation
```

而不是：

```text
sparse z measurement
    ↓
full 3D PSF-aware Wiener inversion
```

### 因此

后续论文与代码中必须区分：

- `3D Wiener function exists` ✅
- `current z warm-start is a full 3D Wiener prior` ❌

### 新原则

真正的：

> **3D physics prior**

必须在 **YZ / XYZ sparse** 主线中重新设计并正式验证。

---

## 3.5 🟡 Protocol A 尚未真正实现

当前主实现主要是：

\[
X\rightarrow H*X\rightarrow M\rightarrow noise\rightarrow Y
\]

属于 controlled synthetic sparse measurement。

而目标的 acquisition-realistic Protocol A 是：

\[
X
\rightarrow H*X
\rightarrow S_K
\rightarrow noise
\rightarrow Y_{low}
\rightarrow R
\rightarrow Y_{grid}
\]

目前 `acquisition.py` 中 Protocol A 还主要停留在说明/注释层面。

### 后续定位

**Protocol B**：机制研究 / controlled simulation  
**Protocol A**：实际采集几何 / acquisition-realistic validation

最终论文应形成：

> simulation establishes mechanism → acquisition-realistic experiments establish applicability

---

# 4. 🟠 采集维度重新定义：Y / YZ / XYZ / XYT

## 4.1 为什么不再使用“2D / 2.5D / 3D”作为一级分类

“2D / 2.5D / 3D”混淆了：

1. acquisition dimensionality
2. reconstruction dimensionality
3. network context dimensionality

因此新的一级分类采用 acquisition geometry。

---

## 4.2 🟢 Regime A：Y-sparse 2D

定义：

\[
X:\;dense
\]
\[
Y:\;sparse
\]

即单个 XY plane 中，X 保持 dense，Y 进行 undersampling。

### 科学意义

对于 point-scanning 3PM，这一模式更直接对应：

> **减少 scan-line / per-frame imaging time**

适合回答：

> 单帧 acquisition speed 在不明显损失重建质量的情况下可以提高多少？

---

## 4.3 🟢 Regime B：YZ-sparse 3D（核心主线）

定义：

\[
X:\;dense
\]
\[
Y,Z:\;sparse
\]

即：

> **X-dense, YZ-sparse volumetric imaging**

这是新的核心 3D 主线。

### 原因

在点扫描体成像中，Y 和 Z 更直接决定：

- scan lines 数量
- optical section 数量
- volume acquisition burden

因此这是最符合项目最终目标的 3D sparse acquisition regime。

在理想化、忽略固定 overhead 的情况下，可以写：

\[
T_{\mathrm{acq}}\propto N_YN_Z
\]

对应 nominal acceleration：

\[
A_{\mathrm{nominal}}\approx K_YK_Z
\]

> ⚠️ 只有在真实硬件中测得实际 scan time 后，才允许将其描述为 **measured speedup**。

---

## 4.4 🟡 Regime C：XYZ-sparse 3D

定义：

\[
X,Y,Z:\;sparse
\]

它用于研究：

> **extreme spatial undersampling / volumetric compression**

而不是默认把：

\[
K_XK_YK_Z
\]

直接当成实际 acquisition acceleration。

### 必须区分

\[
\text{sampling compression}
\neq
\text{hardware speedup}
\]

因为 X sparsity 是否真正减少 acquisition time 取决于具体扫描硬件与扫描模式。

因此 XYZ 在论文中建议定位为：

> **stress-test / extreme spatial undersampling extension**

---

## 4.5 🔵 Regime D：XYT-sparse temporal imaging

定义：

\[
X,Y:\;spatial
\]
\[
T:\;temporal
\]

它描述的是：

> **spatiotemporal reconstruction**

而不是 spatial volumetric reconstruction。

### 核心区别

XYZ 中：

\[
D=Z
\]

Z 是空间维度。

XYT 中：

\[
D=T
\]

T 是时间维度。

因此二者虽然可以使用相似的 tensor operation，却具有不同的物理先验：

| Aspect | XYZ | XYT |
|---|---|---|
| Missing axis | Z | T |
| Physical meaning | depth | time |
| Main prior | spatial continuity / PSF | temporal continuity / dynamics |
| Motion | usually not central | potentially critical |
| PSF role | central | not equivalent to Z |
| Main error type | spatial artifact | temporal inconsistency / flicker |
| Main extension | volumetric imaging | dynamic imaging |

### 论文原则

不能把 XYZ 与 XYT 简化写成：

> “both are 3D tensors”

应明确：

> **They share a computational representation but differ in physical semantics and missing-axis priors.**

---

# 5. 🧠 Physics Prior 重构方案

## 5.1 从“Wiener filter”升级为“sampling-conditioned physics inversion”

对于一般稀疏采样，建议统一定义：

\[
X_0
=
\arg\min_X
\left\|
W\odot(\mathcal A(X)-Y)
\right\|_2^2
+
\lambda R(X)
\]

其中：

- \(\mathcal A\)：forward operator
- \(W\)：measurement weighting
- \(Y\)：measurement
- \(R(X)\)：物理/平滑 regularizer

Protocol B：

\[
\mathcal A=S\circ H
\]

Protocol A：

\[
\mathcal A=R\circ S\circ H
\]

其中：

- \(H\)：PSF / blur operator
- \(S\)：sampling operator
- \(R\)：resampling / acquisition geometry operator

这样可以统一处理：

- Y
- YZ
- XYZ

而不需要为每一种 sparsity mode 手写一个完全不同的 heuristic interpolation pipeline。

---

## 5.2 Y-sparse prior

至少重新验证四种 baseline：

```text
1. Direct observation / no prior
2. Bilinear prior
3. Lateral Wiener prior
4. Iterative / optimization-based physics prior（候选）
```

---

## 5.3 YZ-sparse prior

核心应建立真正的 3D physics-aware initialization：

\[
X_0^{YZ}
=
P_{YZ}(Y,M,H)
\]

至少比较：

- trilinear / separable interpolation
- genuine 3D Wiener
- iterative data-consistent inversion

重点回答：

> **When both Y and Z are undersampled, does an explicit 3D physical inverse provide information that interpolation cannot provide?**

---

## 5.4 XYZ prior

进一步：

\[
X_0^{XYZ}
=
P_{XYZ}(Y,M,H)
\]

重点研究：

- extreme sampling ratio
- PSF mismatch
- photon noise
- prior reliability
- data consistency

---

## 5.5 XYT prior

XYT 不建议强行套用 spatial Wiener。

更合理的 prior 是：

\[
X_0^{XYT}=P_T(Y,M)
\]

主要利用：

- temporal interpolation
- forward/backward temporal context
- motion-aware propagation
- temporal consistency

后续如加入 physics model，也应明确其与 XYZ 的物理含义不同。

---

# 6. 🧩 Sampling-aware Information Representation

统一输入建议保持：

\[
(Y_{grid},M,X_0,C_s,U_p)
\]

其中：

### `M`

Measurement support：

> 哪里真正测到了数据。

### `C_s`

Sampling confidence：

> 从 sampling geometry 推导出的局部支持连续性。

### `U_p`

Physics-derived recoverability / difficulty：

> 不是 calibrated posterior uncertainty。

后续论文不得把 `U_p` 直接表述为：

> “uncertainty prediction”

更稳妥的表述是：

> **physics-derived recoverability / reconstruction difficulty field**

---

# 7. 🏗️ 新的 Unified Reconstruction Network

## 7.1 总体原则

不再追求一个巨型 full-3D Transformer。

原因：此前已有：

- factorized 3D U-Net variants
- bottleneck variants
- full 3D window-attention
- other 3D architectures

其表现均未稳定超过轻量 2D / 2.5D 主线，因此没有理由继续扩大 full-3D Transformer 的复杂度。

---

## 7.2 推荐架构

> **Shared Spatial Backbone + Axis-specific Interaction Adapter**

输入：

\[
[B,T,F,H,W]
\]

其中：

\[
F=5
\]

代表：

```text
Y
M
X0
Cs
Up
```

### Step 1：shared spatial encoder

逐 slice / frame 使用共享 encoder：

\[
E_s:[B,T,5,H,W]\rightarrow[B,T,C,H,W]
\]

不要采用：

```text
Conv2D(5T → C)
```

因为该方式依赖固定 T，无法自然统一 Y / YZ / XYZ / XYT。

---

## 7.3 Step 2：sampling / prior encoder

分别编码：

\[
F_M=E_M(M,C_s,U_p)
\]

\[
F_P=E_P(X_0)
\]

---

## 7.4 Step 3：sampling-conditioned fusion

建议：

\[
G=
\sigma
(\mathrm{MLP}[F_M,F_P])
\]

\[
F=F_s+G\odot F_P
\]

这样网络能够根据：

- sampling density
- measurement support
- physics recoverability

动态决定 prior 应该参与多少。

---

## 7.5 Step 4：axis-specific interaction

### XYZ / YZ

\[
F_Z=A_Z(F)
\]

学习 spatial depth interaction。

### XYT

\[
F_T=A_T(F)
\]

学习 temporal interaction。

### Y-only

可以直接使用 spatial backbone 或极轻量 interaction。

---

## 7.6 Step 5：shared decoder

\[
\hat X=D(F)
\]

最终实现：

```text
Y sparse
      ↓
shared spatial encoder
      ↓
sampling-conditioned prior fusion
      ↓
axis adapter (optional)
      ↓
shared decoder
      ↓
dense image
```

---

# 8. ⚖️ Unified vs. Specialist：不要预设答案

不建议在“一个网络”和“多个网络”之间直接凭经验二选一。

应通过实验回答。

## 8.1 Specialist models

### Specialist-Y

专门训练 Y-sparse。

### Specialist-YZ

专门训练 YZ-sparse。

---

## 8.2 Unified model

一个模型共同训练：

```text
Y
YZ
```

后续再扩展：

```text
XYZ
XYT
```

---

## 8.3 公平比较条件

必须控制：

- parameter count
- total optimizer steps
- training sample exposure
- normalization
- augmentation
- test set

### Scientific question

> **Can a sampling-conditioned unified reconstruction model approach specialist performance across acquisition regimes without sacrificing robustness?**

如果答案成立，它会成为一个独立的方法学贡献。

---

# 9. 🧪 Main Experiment Plan

# 9.1 🟢 Main Experiment 1 — Y-sparse 2D

## Goal

重新回答横向 sparse acquisition 下：

> Wiener 到底有没有价值？

尤其区分：

### Question A

\[
Q(X_{W})-Q(X_{B})
\]

即 Wiener vs. Bilinear。

### Question B

\[
Q(f(Y,X_W))-Q(f(Y))
\]

即 Wiener prior vs. direct network。

### Question C

\[
Q(f(Y,X_W))-Q(f(Y,X_B))
\]

即 Wiener prior vs. Bilinear prior。

这三个问题必须分别报告。

---

## Proposed matrix

\[
K_Y=\{2,4,8,16\}
\]

每个 K 运行：

```text
Direct network
Bilinear-only
Wiener-only
Direct + Bilinear prior
Direct + Wiener prior
```

### Multi-seed

至少 3 seeds；最终主结果建议使用 mean ± 95% CI，并进行 paired statistical analysis。

---

# 9.2 🟢 Main Experiment 2 — YZ-sparse 3D

这是新的核心实验。

## Goal

验证：

> X-dense, YZ-sparse volumetric acquisition 是否能够实现明显 acquisition-time reduction，同时维持高保真体重建。

## Sampling matrix

至少包含：

```text
(KY, KZ)
(2,2)
(2,4)
(4,2)
(4,4)
(4,8)
(8,4)
(8,8)
```

这样可以区分：

- Y difficulty
- Z difficulty
- joint difficulty

而不仅仅研究对称稀疏。

---

# 9.3 ⭐ Main Experiment 3 — Fixed-budget Y/Z Allocation

这是本项目强烈建议重点发展的 innovation。

固定：

\[
K_YK_Z=B
\]

例如：

```text
(2,8)
(4,4)
(8,2)
```

在相同 acquisition budget 下比较重建质量。

### 优化问题

\[
(K_Y^*,K_Z^*)
=
\arg\max_{K_Y,K_Z}
Q(\hat X)
\]

subject to：

\[
C(K_Y,K_Z)\leq B
\]

### 科学价值

回答：

> **When acquisition budget is fixed, how should sparsity be allocated across Y and Z?**

这会把项目从：

> reconstruction network

进一步提升到：

> **acquisition strategy optimization / sampling–reconstruction co-design**

---

# 9.4 🟡 Main Experiment 4 — XYZ extreme spatial sparsity

目标：

> 验证 framework 在 full spatial undersampling 下的 robustness。

重点报告：

- sampling ratio
- missing-region recovery
- PSF sensitivity
- photon budget robustness
- prior robustness

同时严格区分：

```text
nominal sampling reduction
vs.
actual hardware acceleration
```

XYZ 不作为项目唯一核心创新点，因为已有 2026 年工作已经针对 sparse X-Y-Z 3PM reconstruction 展示了显著 acceleration；因此你的差异化应落在 **physics-aware prior、sampling-axis conditioning 和 acquisition-budget co-design** 上，而不是“XYZ + Transformer”本身。[1]

---

# 9.5 🔵 Main Experiment 5 — XYT temporal extension

## Goal

验证同一 reconstruction framework 是否能够扩展到动态成像。

推荐：

\[
K_T=\{2,4,8\}
\]

同时比较：

```text
temporal interpolation
specialist temporal model
unified model
axis-specific temporal adapter
```

### Additional metrics

除 PSNR / SSIM / MAE 外，还应增加：

- temporal consistency
- temporal flicker
- motion consistency
- biological event preservation

### 定位

XYT 建议作为：

> **spatiotemporal generalization**

而不是与 XYZ 并列的“第三种 3D imaging”。

---

# 10. 📊 Evaluation Framework

最终评价必须从单纯 reconstruction quality 扩展为三层。

## 10.1 Image quality

\[
PSNR
\]

\[
SSIM
\]

\[
MS\text{-}SSIM
\]

\[
MAE
\]

---

## 10.2 Missing-region recovery

重点指标：

\[
E_{missing}
\]

不要只看 full-image PSNR，因为 measured region 本身提供了真实信息。

建议同时报告：

```text
measured-region error
missing-region error
full-image error
```

全部基于真实 `M` 计算。

---

## 10.3 Acquisition efficiency

定义实际：

\[
A_T=
\frac{T_{dense}}{T_{sparse}}
\]

以及：

\[
A_D=
\frac{D_{dense}}{D_{sparse}}
\]

其中：

- \(A_T\)：measured acquisition-time acceleration
- \(A_D\)：dose reduction ratio

### 推荐主图

> **Reconstruction quality vs. acquisition acceleration**

而不是只做：

> PSNR vs. K

---

# 11. 🔬 Ablation Plan

## 11.1 Prior ablation

```text
No prior
Bilinear prior
Wiener prior
Iterative physics prior
```

---

## 11.2 Sampling information ablation

```text
Y only
Y + M
Y + M + Cs
Y + M + Cs + Up
```

---

## 11.3 Fusion ablation

```text
concatenation
fixed-weight fusion
adaptive gating
```

---

## 11.4 Axis conditioning ablation

```text
shared interaction
Z-specific adapter
T-specific adapter
shared + axis-specific
```

---

## 11.5 Unified training ablation

```text
Specialist-Y
Specialist-YZ
Unified-Y+YZ
Unified-Y+YZ+XYZ
```

---

# 12. 🧪 Robustness Experiments

保留之前已有的几个非常有价值的变量，但重新放入正确的位置。

## 12.1 λ sweep

继续保留：

\[
\lambda
\]

因为此前实验已经表明 prior quality 对 λ 高度敏感。

### 原则

λ 必须：

- 仅使用 training / validation 选择
- test 阶段冻结
- 不得针对 test result 回调

---

## 12.2 Photon budget

建议保留：

\[
P=\{50,20,10,5\}
\]

并注意此前观察到：固定 λ 的 Wiener collapse 主要与 λ mismatch 有关，因此后续应将其作为：

> **prior robustness / hyperparameter sensitivity**

而不是简单解释为：

> “physics prior fails at low photons”。

---

## 12.3 PSF mismatch

至少比较：

```text
Matched PSF
Mild mismatch
Strong mismatch
```

验证 physics prior 是否对 model mismatch 稳健。

---

# 13. 🧠 T-scaling 的重新定位

此前 z-K8 的结果显示：

```text
T=1  → low
T=3  → substantial gain
T=7  → large jump
T=15 → saturation
T=31 → marginal / saturated
```

这说明性能并非简单随 T 单调提升，而更可能受：

> **effective axial context / reaching a sampled plane**

控制。

因此后续 T scaling 不再作为“2.5D superiority experiment”，而作为：

> **context sufficiency analysis**

用于回答：

> **How much neighboring context is sufficient for different acquisition regimes?**

---

# 14. 🧱 External Architecture Validation

此前 EXP-020 的多 backbone 实验仍然有价值，但应降级为：

> **architecture-independence validation**

而不是继续扩充大量 backbone。

建议保留 2–3 个代表性外部模型：

```text
CNN / UNet-like model
Video/temporal transformer
3D transformer
```

重点检查：

\[
\Delta Q
=
Q(\text{with physics prior})
-
Q(\text{without physics prior})
\]

是否具有 architecture-independent tendency。

旧 EXP-020 对这一点已经提供了初步证据，但由于其 single-seed / batch setting 并非最终冻结协议，因此后续只作为 preliminary evidence，不直接承担最终主结论。[Internal legacy evidence]

---

# 15. 🧹 Project Naming 修正

当前：

> **PSRAS = Physics-Sampling-Residual Aware Synthesis**

但 B5 residual 已被实验淘汰，因此项目命名与最终方法容易出现语义不一致。

### 后续建议

不要为了保留缩写而硬保留 `Residual`。

建议逐步迁移到类似：

> **Physics- and Sampling-aware Sparse-to-Dense Reconstruction**

或：

> **Sampling-Conditioned Physics-Guided Sparse-to-Dense Reconstruction**

最终名称应由最终方法结构决定，而不是由历史缩写决定。

---

# 16. 🗂️ Recommended Repository Structure

建议将 `PSRAS/` 重构为：

```text
PSRAS/
│
├── README.md
├── PLAN.md
├── PROJECT_LOG.md
├── Project_progress_status.md
│
├── LEGACY_RESULTS/
│   ├── legacy_z_sparse/
│   ├── legacy_xy_wiener/
│   ├── legacy_3d_transformer/
│   └── legacy_metrics/
│
├── CORRECTED_PROTOCOL/
│   ├── split/
│   ├── acquisition/
│   ├── normalization/
│   ├── metrics/
│   └── validation/
│
├── physics/
│   ├── acquisition.py
│   ├── wiener.py
│   ├── physics_inverse.py
│   └── sampling.py
│
├── data/
│   ├── observation.py
│   ├── datasets.py
│   └── normalization.py
│
├── models/
│   ├── spatial_encoder.py
│   ├── sampling_encoder.py
│   ├── prior_encoder.py
│   ├── axis_adapter_z.py
│   ├── axis_adapter_t.py
│   ├── fusion.py
│   └── unified_reconstructor.py
│
├── losses/
│   ├── reconstruction.py
│   └── data_consistency.py
│
├── metrics/
│   └── image_metrics.py
│
├── scripts/
│   ├── prepare_sparse_v4.py
│   ├── train_y.py
│   ├── train_yz.py
│   ├── train_unified.py
│   └── reval.py
│
└── tests/
    ├── test_acquisition.py
    ├── test_wiener.py
    ├── test_physics_inverse.py
    ├── test_metrics.py
    └── test_unified.py
```

---

# 17. 🚦 Project Phase & Acceptance Criteria

## 🔴 Phase 0 — Protocol Cleanup

### Must complete

- [ ] Correct stack-level split
- [ ] Explicit measurement mask everywhere
- [ ] Correct 2D SSIM implementation
- [ ] Freeze normalization
- [ ] Freeze PSF / voxel-spacing convention
- [ ] Freeze random seeds
- [ ] Document Protocol A/B semantics
- [ ] Archive all legacy quantitative results

### Gate

> **No unresolved P0 protocol bug.**

---

## 🟠 Phase 1 — Corrected Y baseline

### Must complete

- [ ] Y K=2/4/8/16
- [ ] Direct
- [ ] Bilinear
- [ ] Lateral Wiener
- [ ] Wiener prior network
- [ ] Bilinear prior network
- [ ] Multi-seed
- [ ] 95% CI

### Gate

必须能够明确区分：

\[
Wiener\ vs.\ Bilinear
\]

与：

\[
Wiener\ prior\ vs.\ Direct\ Network
\]

---

## 🟡 Phase 2 — Genuine YZ physics prior

### Must complete

- [ ] 3D PSF model
- [ ] true YZ inverse prior
- [ ] genuine 3D Wiener baseline
- [ ] iterative physics inverse candidate
- [ ] data-consistency validation

### Gate

> Prior 本身必须经过独立验证，不能仅由 downstream network 结果定义其有效性。

---

## 🟢 Phase 3 — YZ Mainline

### Must complete

- [ ] full KY × KZ matrix
- [ ] asymmetric sparsity
- [ ] fixed-budget allocation
- [ ] acquisition-efficiency analysis

### Gate

> 能够回答“Y/Z 稀疏如何分配最有效”。

---

## 🔵 Phase 4 — Unified Framework

### Must complete

- [ ] shared spatial encoder
- [ ] sampling-conditioned fusion
- [ ] Z adapter
- [ ] specialist vs unified comparison
- [ ] matched parameter / training budget

### Gate

> Unified model 不应显著牺牲 specialist performance。

---

## 🟣 Phase 5 — XYZ / XYT Extension

### XYZ

验证 full spatial undersampling robustness。

### XYT

验证 temporal generalization，并加入 temporal metrics。

---

## ⚫ Phase 6 — Protocol A / Real Acquisition

### Must complete

- [ ] hardware-realistic sparse acquisition
- [ ] measured acquisition time
- [ ] exposure / dose measurement or proxy
- [ ] reconstruction quality
- [ ] quality–speed trade-off curve

### Final gate

> 必须形成“算法指标 + 实际 acquisition efficiency”的完整闭环。

---

# 18. 📈 Recommended Main Figures for the Future Paper

## Figure 1 — Overall concept

```text
Acquisition constraint
      ↓
Y / YZ sparse sampling
      ↓
Physics prior
      ↓
Sampling-conditioned reconstruction
      ↓
Fast dense imaging
```

---

## Figure 2 — Physics prior

展示：

- measurement
- mask
- PSF
- physics inverse
- prior reliability

重点突出：

> physical prior ≠ interpolation

---

## Figure 3 — Y-sparse 2D

核心图：

> **Quality vs. Y sampling density / frame acceleration**

同时展示：

- Direct
- Bilinear
- Wiener
- Proposed

---

## Figure 4 — YZ-sparse 3D

核心图：

> **Volumetric reconstruction quality vs. nominal/measured acceleration**

---

## Figure 5 — Fixed-budget sampling allocation ⭐

例如：

```text
(2,8)
(4,4)
(8,2)
```

回答：

> **Where should the sparse budget be spent?**

---

## Figure 6 — Unified model

展示：

```text
Y
YZ
XYZ
```

以及 shared backbone + axis adapter。

---

## Figure 7 — XYT extension

展示：

```text
spatial branch
vs.
temporal branch
```

强调“共享计算框架，但不同物理语义”。

---

## Figure 8 — Real acquisition efficiency

最终展示：

\[
Quality
\;vs.\;
Acquisition\ Time
\]

以及：

\[
Quality
\;vs.\;
Photodose
\]

---

# 19. ✍️ Recommended Paper Storyline

最终论文建议遵循：

```text
1. 3PM is limited by acquisition speed / photon budget
            ↓
2. Sparse acquisition can reduce burden
            ↓
3. But Y and Z have different information difficulty
            ↓
4. Therefore sparse budget should not be allocated blindly
            ↓
5. We formulate sampling-conditioned physics-aware reconstruction
            ↓
6. Physics prior provides a recoverability-aware initialization
            ↓
7. Shared spatial backbone + axis-specific interaction handles
   different acquisition semantics
            ↓
8. Fixed-budget experiments identify effective Y/Z allocation
            ↓
9. YZ achieves efficient volumetric reconstruction
            ↓
10. XYZ demonstrates extreme spatial undersampling robustness
            ↓
11. XYT demonstrates temporal generalization
            ↓
12. Real acquisition validates actual time/dose reduction
```

### 最重要的论文主旨

不要把论文核心写成：

> “We propose another 3D sparse reconstruction network.”

而应逐步靠近：

> **We develop a physics- and sampling-conditioned sparse-to-dense reconstruction framework that jointly considers acquisition geometry, recoverability, and sparse-budget allocation for accelerated microscopy.**

---

# 20. ⚠️ Current Conclusions That Must NOT Be Reused Directly

以下旧结论必须标记为 preliminary / legacy：

### 20.1 “Lateral Wiener gain ≈ 0”

只能解释为：

> lateral Wiener ≈ Bilinear in the previously tested setting

不能解释为：

> lateral Wiener provides no benefit to the new reconstruction network.

---

### 20.2 “z-mainline uses true 3D Wiener”

不能继续使用。

当前主 warm-start 不是严格 full 3D Wiener pipeline。

---

### 20.3 “F0 has independent validation and test stacks”

不能使用。

应描述为：

> temporally separated hold-out subsets within F0

直到真正独立 stack/session test 建立。

---

### 20.4 “Full 3D transformer is ineffective”

可作为当前探索阶段的 empirical observation，但不能升级为普适性理论结论。

未来模型结构发生变化后，应重新验证。

---

### 20.5 “B4 is always better under low photons”

不能这样表述。

此前结果说明：

> fixed λ mismatch can severely degrade Wiener prior, whereas B4 showed stronger robustness in that tested condition.

这属于 robustness evidence，不是 universal superiority。

---

# 21. 🧪 Statistical and Reproducibility Rules

最终实验统一：

### Randomness

- fixed seeds
- multi-seed runs
- deterministic or documented nondeterminism

### Confidence interval

主表推荐：

\[
mean\pm95\%CI
\]

### Paired analysis

同一 test sample 的不同 model 应优先采用 paired statistical analysis。

### Test isolation

test 数据不得用于：

- λ selection
- augmentation selection
- architecture selection
- threshold selection
- sampling allocation selection

### Versioning

每个主实验记录：

```text
commit hash
config
seed
split version
normalization version
PSF version
metric version
```

---

# 22. 🧩 Recommended Experiment Naming Convention

建议从下一阶段开始使用统一命名：

```text
Y_K2
Y_K4
Y_K8
Y_K16

YZ_2x2
YZ_2x4
YZ_4x2
YZ_4x4
YZ_4x8
YZ_8x4
YZ_8x8

XYZ_K...

XYT_K2
XYT_K4
XYT_K8
```

模型则使用：

```text
DIRECT
BILINEAR
WIENER
PHYSINV
UNIFIED
SPECIALIST
```

例如：

```text
YZ_4x4_PHYSINV_UNIFIED_seed03
```

这样后续 log、checkpoint、figure、table 可以直接自动解析。

---

# 23. 📋 Final Priority Checklist

## 🔴 Must Fix Before Any New Main Claim

- [ ] F0 split semantics
- [ ] explicit measurement mask
- [ ] SSIM implementation
- [ ] current z B3 naming
- [ ] normalization freeze
- [ ] PSF / voxel-spacing freeze
- [ ] Protocol A/B documentation

## 🟠 Must Run Again

- [ ] Y Direct vs Bilinear vs Wiener
- [ ] Wiener-only vs network
- [ ] Wiener-prior vs Bilinear-prior
- [ ] corrected multi-seed statistics

## 🟡 Must Build

- [ ] genuine YZ physics prior
- [ ] fixed-budget Y/Z allocation
- [ ] acquisition-time accounting

## 🟢 Core Innovation

- [ ] shared spatial encoder
- [ ] sampling-conditioned prior fusion
- [ ] Z-specific adapter
- [ ] Unified vs Specialist

## 🔵 Extensions

- [ ] XYZ stress test
- [ ] XYT temporal branch

## 🟣 Final Validation

- [ ] Protocol A
- [ ] real acquisition time
- [ ] dose / exposure proxy
- [ ] external independent test stack

---

# 24. 🏁 Final Project Definition

### Recommended project title

> **Physics- and Sampling-Conditioned Sparse-to-Dense Reconstruction for Accelerated Multimodal Microscopy**

### Core scientific problem

> **How can sparse acquisition be designed and reconstructed in a physics-aware manner so that imaging time and photodose are reduced without unacceptable loss of structural or temporal fidelity?**

### Main methodological contribution

> **A sampling-conditioned reconstruction framework that combines a physics-derived prior, explicit measurement support, recoverability-aware conditioning, a shared spatial backbone, and axis-specific interaction modules.**

### Main experimental contribution

> **A systematic study of sparse acquisition across Y, YZ, XYZ, and XYT regimes, with particular emphasis on fixed-budget Y/Z allocation for accelerated volumetric imaging.**

### Final application contribution

> **Demonstration that sparse acquisition can translate into measurable acquisition-time and/or exposure reduction under acquisition-realistic conditions.**

---

# 25. 📚 References / Literature Context

**[1]** Y. Li et al., “Hundred-Nanosecond Equivalent Pixel Dwell Time for Deep-Tissue 3D Three-Photon Fluorescence Microscopy via Sparse X-Y-Z Reconstruction,” *Small Methods*, 2026. PubMed: <https://pubmed.ncbi.nlm.nih.gov/41546413/>

**[2]** Compressed sensing / sparse acquisition for fluorescence microscopy, with reported reductions in acquisition time and/or light exposure. See: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5617510/>

**[3]** CACS / computational acceleration of 3D fluorescence microscopy. See: <https://www.nature.com/articles/s41467-020-18030-0>

> 注：以上文献仅用于项目定位与相关工作背景。最终论文引用格式、文献条目与具体结论应在正式写稿阶段重新核对原文。

---

# 26. 📝 Change Log

## v4.0-draft — 2026-09-21

本版本首次将下一阶段项目正式重构为：

- acquisition-first
- Y / YZ / XYZ / XYT axis-aware
- genuine physics-prior
- fixed-budget sampling allocation
- unified spatial backbone + axis-specific adapters
- specialist vs unified experimental validation
- acquisition efficiency as a primary endpoint
- independent test split requirement

同时将以下历史问题明确降级为 legacy evidence：

- 横向 Wiener ≈ Bilinear
- z-mainline pseudo-3D Wiener warm-start
- F0 same-stack val/test
- old SSIM implementation
- `Y != 0` inferred mask
- previously rejected residual branch
- previously tested full-3D transformer line

---

# 🔑 One-Sentence Project Rule

> **先把“数据是否独立、mask 是否正确、physics prior 到底是什么”彻底做对，再讨论“网络是否更强”；先证明 Y/YZ 稀疏采集真正节省 acquisition burden，再扩展到 XYZ 与 XYT。**

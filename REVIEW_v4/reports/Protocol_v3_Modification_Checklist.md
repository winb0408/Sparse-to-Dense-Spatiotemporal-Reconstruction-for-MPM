# Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM
## Protocol v3：代码审计结论、修改清单与实施方案

> 目标：将当前项目从“多个 2D/2.5D/3D 实验分支”重构为统一、物理一致、可复现实验协议，并为后续统一 3D 条件重建网络与 physics-aware residual reconstruction 做准备。
>
> 本文针对 `dev` 分支当前代码与结果进行规划。历史结果不要直接删除，统一标记为 `legacy / exploratory`，待 Protocol v3 重跑后再决定是否进入正文。

---

## 1. 首先修正对 P0-1 的理解：真实采集与 CIDC25 模拟 sparse 不是同一种 forward model

### 1.1 真实 Y 方向稀疏采集

以 pollen 的 Y=2 稀疏采集为例，如果硬件只采集每隔一个 Y 位置，那么原始 acquisition 在 Y 方向的采样数直接减半，得到的是较低空间尺寸/采样密度的数据。

更合理的物理模型是：

\[
X \xrightarrow{H} H*X \xrightarrow{S_K} Y_{low} + n
\]

其中：

- `X`：目标 dense object/image
- `H`：光学 PSF / forward operator
- `S_K`：真实采样/降采样算子，例如 Y 方向 K=2
- `n`：shot/readout noise

为了让网络在统一的 H×W 网格上处理，可以再进行确定性的几何重采样：

\[
Y_{up}=R_K(Y_{low})
\]

这里的 `R_K` 只是将低分辨率 acquisition 映射回 dense 网格，不应被称为“恢复了缺失信息”。真正的信息恢复由网络完成。

因此：

- `Y_low`：真实观测域
- `Y_up`：统一网络输入网格上的 observation representation
- `M`：原始采样支持位置在目标网格上的表示
- 插值：确定性的输入预处理，而不是模型 reconstruction

### 1.2 CIDC25 当前的 sparse 构造

CIDC25 的 `prepare_cidc25_sparse3d.py` 是在已有 dense stack 上直接进行像素/切片遮蔽，因此本质上更接近：

\[
Y=M\odot X
\]

或者更完整地写为：

\[
Y=M\odot(H*X+n)
\]

但它当前又把 `warmstart`/Z 线性插值后的结果保存成 `low`，导致后续 dataset/evaluation 很容易再次插值或再次 masking。代码中：

- XY：`sparsify_xy()` 后调用 `warmstart_xy()` 并保存；
- Z：`sparsify_z()` 本身直接线性插值；
- XYZ：Z 插值后再做 XY sparse，再 warm-start。

因此真正的问题不是“interpolation 不能存在”，而是：

> **必须明确区分 acquisition simulation 中的 deterministic resampling，与 reconstruction pipeline 中的 warm-start prior；二者不能混用，也不能让一个已经插值过的 `low` 再被当作 raw sparse observation。**

### 1.3 Protocol v3 的统一原则

建立两个明确的 observation protocol：

#### Protocol A：Acquisition-faithful sampling

用于 pollen / 真实采样模拟等：

\[
X \rightarrow H*X \rightarrow S(X) \rightarrow Y_{low}
\rightarrow R(Y_{low})=Y_{grid}
\]

同时保留高分辨率支持图：

\[
M_{grid}=R_M(M_{raw})
\]

其中 `R_M` 使用 nearest-neighbor / binary support mapping，而不是 bilinear。

#### Protocol B：Dense-grid masking

用于 CIDC25 等已有 dense 图模拟稀疏采样：

\[
Y=M\odot(H*X+n)
\]

不改变 H×W 网格；warm-start 只在 dataset runtime 构建：

\[
X_0=W(Y;H,M)
\]

### 1.4 两种 protocol 的共同网络输入

最终都转成统一形式：

\[
(Y_{grid}, M, X_0, U)
\]

其中：

- `Y_grid`：统一空间网格上的观测表示
- `M`：真实 measurement support
- `X_0`：physics / interpolation warm-start
- `U`：reconstruction uncertainty / difficulty field，可选但建议最终加入

这样就不再把“插值”与“重建”混成一个步骤。

---

# 2. P0：必须先修复的代码/实验问题

## P0-1. 重写 sparse data preparation

### 当前问题

`scripts/prepare_cidc25_sparse3d.py` 把 warm-start 后的结果保存为 sparse `low`；Z 模式直接保存 interpolation 后的体数据。

### 修改方案

新建/重写成：

```text
scripts/prepare_sparse_protocol_v3.py
```

只负责保存：

```text
GT X
RAW sparse observation Y
sampling mask M
acquisition metadata
```

不要在离线 preparation 阶段保存 `warmstart` 作为 `low`。

建议目录：

```text
ProtocolV3/
  CIDC25/
    K2_xy/
      Training/
        gt/
        sparse/
        mask/
      Validation/
        gt/
        sparse/
        mask/
    K2_z/
    K2_xyz/
  pollen/
    yK2/
    xK2/
```

或者对大数据直接保存 `.npy/.npz`，避免大量小 TIFF。

---

## P0-2. 统一 observation model

建立单一接口，例如：

```python
observation = simulate_acquisition(
    x_gt,
    psf=psf_model,
    sampling=sampling_operator,
    noise=noise_model,
)
```

输出：

```python
observation.raw
observation.grid
observation.mask
observation.metadata
```

严格区分：

1. optical degradation
2. sampling
3. noise
4. geometric resampling
5. reconstruction warm-start

不要再让 dataset、prepare script、inference.py 各自重复实现一套 sparse logic。

---

## P0-3. 禁止 target-dependent normalization

当前 `normalise_pair()` 与 3D dataset 中存在使用 GT 统计量决定 normalization 的情况。

例如：

```python
vmin = patch_np.min()
vmax = patch_np.max()
```

或者：

```python
vmin = min(gt_np.min(), sp_np.min())
vmax = max(gt_np.max(), sp_np.max())
```

这在真实 inference 中不可获得，也会使 evaluation protocol 不严谨。

### Protocol v3 推荐

优先固定到：

```text
uint16 physical range -> float32 [0,1]
```

如果需要 acquisition gain correction：

\[
Y'=aY+b
\]

其中 `a,b` 来自 training-set calibration / acquisition metadata，而不是 GT。

如果为了研究强度归一化，单独做 normalization ablation，不与主 protocol 混合。

---

## P0-4. 重新实现真正的 anisotropic 3D Wiener

当前 `wiener_init_3d()` 实际上逐 frame 调用 `wiener_init_2d()`，`psf_sigma_axial` 没有进入 3D Wiener inverse。

需要实现：

\[
X_W=\mathcal F^{-1}\left[
\frac{H^*}{|H|^2+K}
\mathcal F(Y)
\right]
\]

其中：

\[
H=\mathcal F(h_{xyz})
\]

且 `h_xyz` 是 anisotropic 3D PSF。

### 推荐实现

新建：

```text
physics/wiener.py
```

提供：

```python
build_gaussian_psf_3d(...)
build_otf_3d(...)
wiener_deconv_3d(...)
wiener_deconv_mask_aware_3d(...)
```

### 第一版

先使用空间可分离 Gaussian：

\[
h(x,y,z)=G_{\sigma_{xy}}(x,y)G_{\sigma_z}(z)
\]

### 第二版

考虑真实 voxel spacing：

\[
\sigma_{x,pixel}=\frac{\sigma_x}{\Delta x},
\quad
\sigma_{z,pixel}=\frac{\sigma_z}{\Delta z}
\]

即 PSF 参数应优先定义在物理单位，而不是直接写成“pixel sigma”。

### Wiener regularization

至少测试：

```text
lambda = 0.001 / 0.005 / 0.01 / 0.05 / 0.1
```

最终可考虑 frequency-dependent regularization：

\[
K(f)=\lambda\frac{S_n(f)}{S_x(f)+\epsilon}
\]

---

## P0-5. PSF consistency loss 改成 masked data consistency

当前 PSF loss 实际是 2D blur，不应继续作为“3D physics consistency”。

Protocol v3 定义：

\[
\hat Y=\mathcal H(\hat X)
\]

\[
L_{DC}=\frac{1}{|M|}
\|M\odot(\hat Y-Y)\|_1
\]

如果观测域是低分辨率 acquisition：

\[
L_{DC}=\|S(\mathcal H(\hat X))-Y_{low}\|_1
\]

这里必须在**真实 measurement domain**比较，而不是和插值后的 `Y_grid` 比较。

这是统一真实采集与 CIDC25 synthetic mask 的关键。

---

## P0-6. DINOv2 直接从主 loss 删除

删除：

```text
DINOv2 perceptual loss
```

不需要现在引入 3D DINO。

主损失改为：

\[
L=L_{recon}+\lambda_{dc}L_{DC}
\]

其中：

\[
L_{recon}=L_1+\lambda_{SSIM}L_{MS\text{-}SSIM}
\]

后续如果发现 residual reconstruction 对高频结构存在系统性损失，再单独研究 microscopy-domain perceptual loss。

---

## P0-7. 修复 2D/3D Swin shifted-window attention mask

当前代码有 cyclic shift，但没有标准 shifted-window attention 的区域 mask。

需要补齐：

```text
window_partition
shifted_window_attention_mask
window_reverse
```

2D 和 3D 都一起修。

如果最终准备换成成熟实现，可直接复用验证过的 Swin implementation，而不是维护自定义 attention。

---

## P0-8. 重写 inference.py

当前 `inference.py` 与 `dataset.py` API 已不同步，例如 warm-start 调用参数不一致。

Protocol v3 下 inference 必须复用统一 observation builder：

```python
prepare_observation(...)
```

而不是自行再次实现：

```text
PSF
noise
mask
warmstart
```

避免 training / evaluation / deployment 三套逻辑继续漂移。

---

## P0-9. 重新核查异常 baseline 结果

重点核查：

```text
baseline_input_z_K4
PSNR ~74 dB
SSIM ~1.0
```

需要确认：

- baseline 是否真正使用 raw sparse observation；
- 是否与 GT 在同一 sampling/reconstruction domain；
- 是否误用了预插值数据；
- 是否 evaluation 仅在 sampled positions 计算；
- normalization 是否使用 GT 信息；
- frame indexing 是否严格对应。

在核查完成前不要把这一结果作为正文证据。

---

# 3. P1：建议进行的 dataset.py 重构

## 3.1 建议拆成四个层次

```text
physics/
  acquisition.py
  sampling.py
  psf.py
  wiener.py

training/
  datasets.py
  transforms.py

models/
  unified_3d_reconstruction.py
  prior_fusion.py

metrics/
  reconstruction.py
  structural.py
  statistical.py
```

`dataset.py` 不再承担全部 physics logic。

---

## 3.2 新的 Dataset 输出

统一输出：

```python
{
    "observation": Y,
    "mask": M,
    "warm_start": X0,
    "target": X,
    "target_mask": Q,
    "metadata": {...},
}
```

其中：

- `Y`：观测
- `M`：sampling support
- `X0`：Wiener/bilinear prior
- `X`：GT
- `Q`：target/output mask

---

# 4. Sampling mask 与 uncertainty 的最终设计

## 4.1 Mask 不等于 uncertainty

保留：

\[
M(x)\in\{0,1\}
\]

其定义是：

> 该位置/采样单元是否有真实 measurement support。

因此不建议命名为 `uncertainty_mask`。

建议命名：

```text
sampling_mask
measurement_mask
measurement_support
```

---

## 4.2 增加 sampling confidence

可以从 sampling geometry 构造：

\[
C_s(x)=G_\sigma*M(x)
\]

例如离真实采样点越近，`C_s` 越高。

它表达：

> 该位置附近有多少真实 observation support。

这是 sampling prior，而不是 learned uncertainty。

---

## 4.3 增加 physics/reconstruction uncertainty

建议定义：

\[
U_p=f(H,M,K)
\]

可由：

- PSF transfer attenuation；
- Wiener inverse amplification；
- distance to measured samples；
- local photon/SNR；

构成。

一种可解释的基础版本：

\[
U_f=\frac{K}{|H(f)|^2+K}
\]

再转换到 spatial domain 得到 coarse uncertainty field。

注意：这只是第一版 physics-derived uncertainty，不应直接声称为 calibrated posterior uncertainty。

---

# 5. Sampling-aware Prior Encoding 推荐方案

最终输入信息：

\[
(Y,M,X_0,C_s,U_p)
\]

但是不要全部直接 concat。

推荐：

\[
F_Y=E_Y(Y)
\]

\[
F_W=E_W(X_0)
\]

\[
F_S=E_S([M,C_s,U_p])
\]

然后：

\[
G=\sigma(Conv([F_S,F_W]))
\]

\[
F_{prior}=G\odot F_W
\]

\[
F_{in}=F_Y+F_{prior}
\]

最终 residual decoder：

\[
\Delta X=R_\theta(F_{in})
\]

\[
\hat X=X_0+\Delta X
\]

这个设计保留现有 prior-fusion 思想，但把它从“mask channel modulation”升级为 sampling/physics-conditioned prior fusion。

---

# 6. Residual reconstruction：正式纳入方法

不要让主网络直接从零预测：

\[
\hat X=f_\theta(Y)
\]

建议改为：

\[
X_0=W(Y;H,M)
\]

\[
\Delta X=f_\theta(Y,M,X_0,U)
\]

\[
\boxed{\hat X=X_0+\Delta X}
\]

优点：

1. 网络只需学习 physics prior 与真实图像之间的 residual；
2. PSF/Wiener 真正成为 reconstruction pipeline 的一部分；
3. 可以直接分析 `X0` 与 `ΔX`；
4. 极端 sparse 下更容易解释 failure mode。

---

# 7. Unified 3D Network：统一 2D / 2.5D / 3D

## 7.1 目标

一个统一的 3D network 支持：

```text
T = 1       -> 2D
T = 3/5     -> 2.5D
T = 8/16    -> 3D
```

统一输入：

\[
[B,C,T,H,W]
\]

---

## 7.2 Target mask 机制

定义：

\[
Q_t\in\{0,1\}
\]

例如：

### 2D

```text
Q = [1]
```

### 2.5D

```text
Q = [0,1,0]
```

### 3D

```text
Q = [1,1,...,1]
```

模型始终输出：

\[
\hat X\in\mathbb R^{B\times1\times T\times H\times W}
\]

loss 只在 `Q=1` 的 target slices 上计算。

这样同一个网络可以同时支持：

- full 3D reconstruction
- center-slice reconstruction
- single-slice reconstruction

---

## 7.3 Sparse mask 完全统一

\[
M\in\mathbb R^{B\times1\times T\times H\times W}
\]

自动表示：

- XY sparse：每层均有 spatial missing
- Z sparse：部分 T 层全采样、其他层缺失
- XYZ sparse：两者同时存在

网络本身不需要 hard-code：

```text
if 2d:
if 2.5d:
if 3d:
```

只根据 `T`、`M` 和 `Q` 工作。

---

## 7.4 不建议直接沿用当前固定 `(2,8,8)` window

因为统一网络至少要处理：

```text
T=1, 3, 5, 8, 16
```

建议：

### 方案 A：2D spatial window + axial/Z mixing

```text
Spatial Swin block
       ↓
Axial/Z interaction
       ↓
Spatial Swin block
```

优点：

- 对 anisotropic volume 更自然；
- T 可以是可变长度；
- 计算量比完整 3D attention 更稳定。

### 方案 B：可变 T 的 3D window attention

实现动态 temporal padding/window mask。

优先级低于方案 A。

---

# 8. Loss 设计

最终推荐：

\[
L=\lambda_1L_1
+\lambda_2L_{MS\text{-}SSIM}
+\lambda_3L_{DC}
+\lambda_4L_{structure}
\]

第一版可以只使用：

\[
L=L_1+0.5L_{MS\text{-}SSIM}+\lambda_{DC}L_{DC}
\]

其中 `lambda_DC` 建议单独做：

```text
0 / 0.01 / 0.05 / 0.1 / 0.5
```

不要一次加入过多新 loss。

---

# 9. Evaluation Protocol v3

## 9.1 基本指标

正文建议只保留：

- PSNR
- SSIM
- MS-SSIM
- MAE
- foreground PSNR / SSIM（如有可靠结构 mask）
- profile correlation / peak shift
- data consistency error

不建议为了数量保留当前不够标准的 `FwPSNR`。

---

## 9.2 强制增加 measured/missing split

定义：

\[
E_{measured}=\frac{\|M\odot(\hat X-X)\|}{|M|}
\]

\[
E_{missing}=\frac{\|(1-M)\odot(\hat X-X)\|}{|1-M|}
\]

如果输入在低分辨率 acquisition domain，则 measured error 应在 acquisition domain 计算。

这样可以明确回答：

> 模型是在真实采样点保持一致，还是主要依赖 hallucination？

---

## 9.3 统计

所有 paired test 必须按 sample/frame ID 对齐：

```python
merge(runA, runB, on=["sample_id"])
```

之后再做 Wilcoxon signed-rank。

不要继续直接使用：

```python
a[:n]
b[:n]
```

bootstrap CI 要明确是：

- per-image mean 的 CI；还是
- per-frame mean 的 CI。

对于有 stack/frame 层级的数据，正文建议优先以 **stack/sample 为统计单位**，避免同一 stack 的数百 frame 被视为独立实验单位。

---

# 10. Benchmark 重建

## 主 benchmark：CIDC25

建议主表：

| Regime | Sampling | K | Output |
|---|---|---:|---|
| 2D | XY | 2/4/8 | center/single slice |
| 2.5D | Z | 2/4/8 | center slice |
| 3D | XYZ | 2/4/8 | full volume |

但论文不做“全部组合大排名”。

核心 ablation：

```text
B0   raw sparse / no mask prior
B1   + sampling mask
B2   + bilinear warm start
B3   + true 3D Wiener
B4   + sampling-aware prior fusion
B5   + residual reconstruction
B6   + data consistency
```

---

# 11. 跨数据测试重新定义

## Same-domain

```text
Train: CIDC25
Test : CIDC25 Validation
```

用于主论文结论。

## Zero-shot cross-domain

```text
Train: CIDC25
Test : MDC25
```

回答 acquisition/domain variation 下的 transfer 能力。

## External real sparse transfer

```text
Train: CIDC25
Test : pollen
```

如果 pollen 的采样方式和 CIDC25 不同，应标注为：

> external acquisition-protocol transfer

不要与 same-domain benchmark 混排。

---

# 12. 训练与验证数据划分

当前 CIDC25 3D dataset 直接从某个 GT stack 随机切 patch。这个设置可以用于 prototype，但正式论文最好按 **stack-level split**，不能只按 patch 随机分。

推荐：

```text
Training stacks -> train patches
Validation stacks -> validation patches
Test/Validation stacks -> final test
```

绝对避免同一 stack 的相邻 patch 同时出现在 train/test 中。

因为 3D microscopy 的相邻 patches 高度相关，patch-level random split 会明显高估泛化。

---

# 13. 推荐代码重构后的最小目录

```text
project/
├── physics/
│   ├── acquisition.py
│   ├── sampling.py
│   ├── psf.py
│   └── wiener.py
│
├── datasets/
│   ├── cidc25.py
│   ├── pollen.py
│   └── common.py
│
├── models/
│   ├── unified_3d_reconstruction.py
│   ├── prior_fusion.py
│   └── attention.py
│
├── losses/
│   ├── reconstruction.py
│   └── data_consistency.py
│
├── metrics/
│   ├── image_metrics.py
│   ├── structure_metrics.py
│   └── statistics.py
│
├── scripts/
│   ├── prepare_sparse_v3.py
│   ├── evaluate_v3.py
│   └── visualize_v3.py
│
├── train.py
├── inference.py
└── options.py
```

当前 `dataset.py`、`eval.py` 可以先保留为 `legacy_*`，待 Protocol v3 验证后删除。

---

# 14. 实施顺序

## Phase 0：冻结旧结果

不要删除：

```text
runs/
eval_* /
archive/
```

新增：

```text
results_legacy/
```

所有现有结果标记：

```text
LEGACY / NOT FINAL
```

---

## Phase 1：Protocol v3

优先完成：

1. raw sparse observation
2. explicit sampling mask
3. acquisition-domain representation
4. no-GT normalization
5. unified physics operator
6. real 3D Wiener
7. masked data consistency

这一阶段不改大模型。

---

## Phase 2：Unified reconstruction model

首先实现：

\[
X_0=W(Y)
\]

\[
\hat X=X_0+\Delta X
\]

输入：

```text
Y + M + X0
```

然后加入：

```text
C_s + U_p
```

得到完整 prior-aware version。

---

## Phase 3：统一 2D/2.5D/3D

先验证：

```text
T=1
T=3
T=5
T=16
```

共享同一个 model checkpoint architecture。

分别测试：

```text
Q=[1]
Q=[0,1,0]
Q=[1,...,1]
```

---

## Phase 4：核心实验

只优先跑：

```text
K=2,4,8
XY
Z
XYZ
```

加：

```text
bilinear
Wiener
mask
prior fusion
residual
DC loss
```

形成 factorial ablation，但正文只展示最关键的链条。

---

## Phase 5：generalization

最后再跑：

```text
CIDC25 -> CIDC25
CIDC25 -> MDC25
CIDC25 -> pollen
```

并明确 acquisition-protocol difference。

---

# 15. 最终方法的推荐数学定义

统一 observation model：

\[
Y_{obs}=S\left(H*X\right)+n
\]

或者 dense-grid masking：

\[
Y_{grid}=M\odot(H*X+n)
\]

统一表示：

\[
(Y,M,H)
\]

physics initialization：

\[
X_0=\mathcal W(Y;H,M)
\]

sampling confidence：

\[
C_s=g(M)
\]

physics uncertainty：

\[
U_p=h(H,M,K)
\]

prior-aware encoder：

\[
F=\Phi(Y,M,X_0,C_s,U_p)
\]

residual reconstruction：

\[
\Delta X=R_\theta(F)
\]

final output：

\[
\boxed{\hat X=X_0+\Delta X}
\]

measurement-domain consistency：

\[
L_{DC}=\|S(H*\hat X)-Y_{obs}\|_1
\]

最终 loss：

\[
\boxed{
L=L_{rec}+\lambda_{DC}L_{DC}
}
\]

统一支持：

\[
T=1\quad(2D),
\qquad
T=3/5\quad(2.5D),
\qquad
T=8/16\quad(3D)
\]

---

# 16. 对当前项目最重要的判断

当前最值得保留的核心不是“Video Swin + 很多指标”，而是：

> **Sparse microscopy reconstruction as a physics-constrained inverse problem with explicit sampling support.**

最有潜力的贡献链条：

```text
anisotropic acquisition
        ↓
sparse observation + measurement support
        ↓
physics-based Wiener initialization
        ↓
sampling/physics-aware prior encoding
        ↓
residual reconstruction
        ↓
measurement-domain data consistency
        ↓
one unified 3D model for 2D/2.5D/3D
```

这条线能够把目前仓库中的主要组件统一起来，同时避免继续堆叠互相独立的 engineering modules。

---

# 17. Codex 执行时的工作原则

1. **不要直接覆盖旧实验结果。**
2. 新代码优先通过新文件/新模块实现，旧模块标记 legacy。
3. 每次修改后运行 shape check 与 synthetic sanity check。
4. 对 physics operator 做单元测试：
   - no blur/noise/no sampling -> identity
   - mask=1 -> reconstruction 不应劣于对应 no-mask case 的输入定义
   - known Gaussian volume -> Wiener 输出应改善频域恢复
5. 对 normalization 做 test：evaluation 过程不能读取 GT statistics。
6. 对 measured/missing consistency 做 test。
7. 所有新实验保存完整 config，包括：
   - dataset
   - sampling
   - PSF
   - noise
   - Wiener lambda
   - model
   - seed
   - split
8. 新结果默认标注为 `protocol_v3`，不得与 legacy result 自动混表。

---

# 18. 第一阶段 Codex 实际执行优先级

```text
[P0]
01  修复/重写 sparse preparation
02  分离 raw observation 与 warm-start
03  去除 GT-dependent normalization
04  实现 anisotropic 3D Wiener
05  实现 measurement-domain data consistency
06  删除 DINO loss
07  修复 shifted-window attention mask
08  重写 inference API
09  重核 baseline 74 dB / SSIM=1.0
10  重写 evaluation pairing/statistics

[P1]
11  residual reconstruction
12  sampling confidence
13  physics-derived uncertainty
14  prior-aware encoder
15  unified 3D model
16  T=1/3/5/16 validation

[P2]
17  extensive K=2/4/8 experiments
18  MDC25 zero-shot transfer
19  pollen external acquisition transfer
20  final paper figures/tables
```

---

## 结论

### 关于当前 P0-1 的修正

不是简单地说“CIDC25 预插值是错误的”。更准确的结论是：

> **真实采集中的 interpolation/resampling 是 acquisition representation 的一部分；CIDC25 中的 warm-start 是 reconstruction prior。两者必须严格分层。**

因此 Protocol v3 要求：

\[
\text{acquisition sampling/resampling} \neq \text{reconstruction warm-start}
\]

最终无论是 pollen 真实 Y 稀疏采样，还是 CIDC25 dense-grid masking，都进入同一个：

\[
(Y,M,X_0,U)
\]

框架。

这将是后续统一 2D/2.5D/3D、真实采样建模以及 physics-aware residual reconstruction 的基础。

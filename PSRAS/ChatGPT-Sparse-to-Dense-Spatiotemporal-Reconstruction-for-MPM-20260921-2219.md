# Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM

**User:** Anonymous  
**Created:** 9/8/2026 22:23:51  
**Updated:** 9/21/2026 22:06:09  
**Exported:** 9/21/2026 22:19:39  
**Link:** [https://chatgpt.com/c/6aa01a75-f754-83ec-87ea-964861042edf](https://chatgpt.com/c/6aa01a75-f754-83ec-87ea-964861042edf)  

## Response:
2026/9/21 15:15:10

我会直接审阅 `PSRAS` 分支下的 `PSRAS/` 目录，重点把“实验记录 → 实际代码 → 当前结果”三者对起来，再逐项对照我们之前的 Protocol v3。

我已经把当前实验日志、状态表、核心 physics/data/model/loss 代码以及代表性 run 的 `config/metric` 对过一遍。整体并没有“乱跑”，但发现了几处比日志里更关键的科学性缺口，尤其是 **“真 3D Wiener”命名、CIDC25 的 GT/噪声语义、验证集独立性和 MS-SSIM 实现**。

> **明确协议重点**
>
> 我正在整理统一协议：区分真实低分辨率采集与合成掩膜，保留测量支持、物理初值和不确定性，并采用残差重建、数据一致性及统一二维至三维模型。真实花粉采集在 K=2 时确实会降低观测维度，因此需明确插值回目标网格。
>
> **梳理实验进展**
>
> 我正在汇总项目日志：第一阶段已完成，主比较暂采用 B0–B4；采样模式转向更贴近三光子成像时间的 Y、Z、YZ 方向，同时记录数据划分、归一化、噪声、PSF 与后续统一三维网络计划。
>
> **明确协议重点并梳理实验进展**
>
> 
>
> **检查并评估PSRAS项目代码、实验日志、结果及研究进展**
>
> 
>
> **检查PSRAS模型、数据处理及重建实现一致性**
>
> 
>
> 思考了 3m 44s

我把 `PSRAS` 分支下的实验日志、状态报告、核心 physics/data/model/loss 代码，以及代表性实验的 `config.json` 和 reval 结果都交叉审了一遍。整体判断非常明确：

> **这次推进没有出现无方向的“方案漂移”，反而形成了一条比较完整的“提出假设 → 预注册判据 → 实验验证/证伪 → 收缩主张 → 再设计”的研究链条。**
>
> 但与此同时，当前版本已经从我们最初的“统一 2D/2.5D/3D、真实采集与 synthetic sampling 统一、physics + uncertainty + residual”的方案，明显收缩成了一个更聚焦的 **“z 轴稀疏采样 + 物理反卷积先验 + sampling-aware fusion + 2.5D 主干”** 项目。
>
> 这个收缩本身是合理的；真正需要现在警惕的是：**有几处代码实现已经和现在论文想讲的故事不完全一致，而且其中至少两处会直接影响论文主张的准确性。**

下面我按“研究目标 → 当前路线 → 与初始方案的偏移 → 修改合理性 → 当前真正不足 → 下一步创新”完整拆开。

---

# 一、先重新定义：这个项目现在到底在研究什么

目前 `PSRAS` 最准确的研究目标，已经不是原先比较宽泛的：

> sparse-to-dense microscopy reconstruction with a unified 3D model

而应该重新表述为：

> **研究在三光子显微镜（3PM）稀疏采样条件下，如何利用已知成像物理、采样支持以及轴向上下文，对稀疏观测进行高保真稠密重建，并分析不同采样方向、采样严重程度、先验质量和噪声水平对重建性能的影响。**

其核心问题是：

$$
Y=M\odot(H*X+n)
$$

其中：

- $X$：dense target volume
- $H$：optical PSF
- $M$：measurement support
- $n$：measurement noise
- $Y$：sparse observation

然后构造 physics-based prior：

$$
X_0=W(Y;H,M,\lambda)
$$

再交给 reconstruction network：

$$
\hat X=f_\theta(Y,M,X_0,C_s,U_p)
$$

当前实际主线的网络不是一个真正的全 3D reconstruction transformer，而是：

> **2.5D input + 2D Swin-UNet backbone + sampling-aware prior fusion**

其中邻近的 $T$ 个 z slices 被当作 channel 输入网络。

这一点现在已经非常明确。

[PSRAS 项目状态报告](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/Project_progress_status.md?utm_source=chatgpt.com)

---

# 二、当前 PSRAS 的实际研究思路

现在整个项目实际上形成了五层。

## 第一层：采样问题建模

区分不同稀疏方式：

- `xy`
- `y`
- `z`
- `yz`
- `xyz`

但实验逐渐发现：

> **z-axis sparsity 是目前最有研究价值的主问题。**

尤其是：

$$
z\text{-K2},z\text{-K4},z\text{-K8},z\text{-K16}
$$

形成了 severity curve。

---

## 第二层：物理先验

当前主要比较：

```text
B0: Y
B1: Y + M
B2: Y + M + bilinear X0
B3: Y + M + Wiener X0
B4: Y + M + Wiener X0 + Cs + Up + gated fusion
```

核心实验结果非常漂亮的一点是：

### z-K 全部成立

你现在的双种子矩阵：

| K | B0 | B2 | B3 | B4 | B3-B2 | B4-B3 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 43.62 | 43.14 | 50.04 | 50.88 | **+6.90** | +0.85 |
| 4 | 41.08 | 42.99 | 49.84 | 50.48 | **+6.85** | +0.64 |
| 8 | 39.17 | 40.61 | 44.73 | 46.48 | **+4.12** | +1.75 |
| 16 | 37.24 | 39.49 | 44.56 | 41.85 | **+5.07** | **-2.70** |

也就是说：

$$
B3>B2
$$

在 K=2/4/8/16 都成立，而且跨 seed。

这是当前最扎实的一条结果。

[PROJECT_LOG.md 实验记录](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/PROJECT_LOG.md?utm_source=chatgpt.com)

---

# 三、当前项目实际上已经得到三个非常重要的科学发现

## 发现 1：物理先验确实有价值，但价值高度依赖采样方向

这是现在最明确的结果。

z：

$$
+5.33\sim +7.28\,dB
$$

而 2D xy：

$$
\approx0
$$

甚至出现负值。

例如你已经实测：

$$
B3-B2=
-4.61\,dB
$$

在 xy-K2。

所以现在不能再写：

> Wiener prior is generally beneficial for sparse reconstruction.

应该写成：

> **The benefit of physics-based Wiener initialization is strongly sampling-regime dependent, with substantial gains under axial sparsity and little or negative gains under lateral grid sparsity.**

这实际上比原始假设更有意思。

---

# 四、发现 2：真正决定先验价值的是三个因素

现在日志提出：

$$
\text{Prior Gain}
\approx
\text{Coupling Strength}
\times
\text{Prior Quality}
\times
\text{Network Utilization}
$$

这条思路目前是成立的，但我建议注意一个措辞：

> **它更适合称为“分析框架”或者“interaction framework”，还不应称为严格的乘法定律。**

因为目前并没有真正证明：

$$
\Delta P =
C\cdot Q\cdot R
$$

只是在实验上观察到三个因素共同控制先验收益。

这个差别论文里很重要。

---

# 五、发现 3：当前任务的关键瓶颈不是简单“网络不够大”，而是采样和上下文的可恢复性

T-scaling 是这一阶段很有价值的一组实验。

现在：

| T | B3 |
|---|---:|
| 1 | 25.82 |
| 3 | 31.82 |
| 7 | 45.12 |
| 15 | 44.73 |
| 31 | 45.99 |

表现为：

```text
T≤3      平台
          ↓
T=3→7    突然跃升
          ↓
T≥7      基本饱和
```

所以现在比较合理的物理解释不是：

> “更多 temporal context continuously improves reconstruction.”

而是：

> **the critical factor is whether the receptive field reaches a measured axial plane.**

也就是：

> **“够不够到采样锚点”比“上下文有多长”更重要。**

这个发现其实很适合成为一个独立的小节。

---

# 六、现在项目距离初始 Protocol v3 到底有没有偏移？

答案是：

> **有明显偏移，但大多数偏移属于实验驱动的合理收缩，而不是错误偏离。**

我建议把它分成三类。

---

# 七、第一类：我认为修改非常合理，而且应该保留

## ① 把 sparse observation 从离线生成改成 runtime generation

初始问题：

```text
prepare sparse
↓
save interpolated/warm-start image
↓
dataset
↓
network
```

现在改成：

```text
GT
↓
acquisition operator
↓
Y + M
↓
runtime warm-start
↓
network
```

这是正确的。

因为这样才能保证：

$$
\text{sampling}
\rightarrow
\text{physics}
\rightarrow
\text{warm start}
$$

是一条透明数据链。

`prepare_sparse_v3.py` 明确只保存 calibration、stack 和 split，不预生成 sparse/warm-start，这个修改是对原始方案的重要修正。 

[prepare_sparse_v3.py](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/scripts/prepare_sparse_v3.py?utm_source=chatgpt.com)

---

## ② 去掉 DINOv2

正确。

既然它之前没有真正参与梯度优化，那么继续保留只会增加方法复杂度。

现在的新代码已经把它拿掉，这个决策是正确的。

---

## ③ 先做 λ sweep，再决定 Wiener

这是非常好的调整。

原先：

$$
\lambda=0.01
$$

其实是一个很差的选择。

现在发现：

$$
\lambda=3\times10^{-6}
$$

在当前无额外噪声协议下效果非常好。

这一步甚至比原计划更重要，因为它揭示：

> **network architecture improvement can be completely masked by an improperly calibrated physics prior.**

这是非常有价值的实验观察。

---

# 八、第二类：修改合理，但现在需要收紧表述

## ① 从“统一 3D 网络”退回 2D/2.5D 主干

这一点我认为是正确的。

你已经做了：

- factorized axial
- bottleneck axial
- center-only output
- full 3D shifted-window

结果都没有超过 2.5D Swin 基线。

最新 full 3D：

$$
45.82
$$

而主线：

$$
46.28
$$

并且未达到预注册的：

$$
47.28
$$

所以继续强行追求“统一 3D 网络”没有必要。

更重要的是：

> 你不是“3D 模型没调好”，而是已经做了一个比较有纪律的 negative-result closure。

这个在科研上完全成立。

但论文里千万不能写：

> “3D Transformer is inferior to 2D.”

只能说：

> **Under the present sampling protocol, parameter budget, training schedule, and reconstruction formulation, the evaluated explicit 3D transformer variants did not outperform the channelized 2.5D Swin-UNet baseline.**

因为你现在的 2.5D 其实有一个很强的 inductive bias：

```text
T slices
↓
flatten as channels
↓
every layer mixes them
```

而你的 3D 网络却是：

```text
XY features
+
explicit Z interaction
```

两者不是“只有 2D vs 3D”这一单变量。

这一点 PROJECT_LOG 自己已经意识到了，处理是正确的。

---

# 九、② 放弃 residual learning

这个调整我基本同意。

但这里有一个非常值得注意的问题：

### 你的项目名字仍然叫

> PSRAS = Physics-Sampling-Residual Aware Synthesis

但是最终 residual 已经被你自己判定：

> 不进入论文。

所以现在：

> **项目名称与最终方法已经不一致。**

开发分支叫 PSRAS 没问题。

但真正准备论文的时候，建议重新命名，至少不要再把 “Residual” 放到核心 acronym 里。

否则审稿人看到：

> “Residual Aware”

然后全文没有 residual branch，会很奇怪。

---

# 十、③ 把 lateral Wiener 主张放弃，转成“轴向瓶颈”论证

这个修改我也认为是正确的。

目前：

```text
xy:
B3 ≈ B2 or worse

z:
B3 >> B2
```

而且已经有：

- xy-K2
- xy-K8
- y-K8
- z-K2/4/8/16

多个方向和 severity 实验。

所以现在“横向 Wiener 无收益”反而变成非常有用的 negative control。

不过这里存在一个更深层的问题，后面会讲：

> 现在的“轴向瓶颈”还不能完全归因于 sampling direction，因为你同时人为设置了 anisotropic PSF。

---

# 十一、现在真正需要重点修正的第一个问题：你现在所谓“真 3D Wiener”，实际上没有用于主线 z reconstruction

这是我审查当前代码后认为**最重要的问题**。

当前：

```text
wiener_deconv_3d()
```

确实是一个真正的：

$$
3D\ FFT
$$

Wiener：

$$
X_W=
\mathcal F^{-1}
\left[
\frac{H^*}
{|H|^2+\lambda}
\mathcal F(Y)
\right]
$$

这一点没问题。

但问题在：

```text
warm_start()
```

对于 z-sparse：

```python
for t in range(T):
    if mask_z[t]:
        out[t] = wiener_deconv_3d(
            out[t][None],
            lat_psf,
            lam
        )[0]
out = _axial_interp(out, mask_z)
```

这里：

- 每一个 measured plane 单独做 Wiener
- `T=1`
- `sigma_z=0`
- 然后沿 z 做 interpolation

所以主线 z-K4/K8/K16 的 B3 实际流程是：

```text
Sparse z stack
↓
measured plane
↓
2D lateral Wiener
↓
axial interpolation
```

而不是：

```text
3D anisotropic Wiener
↓
利用 H(x,y,z)
↓
3D inverse
```

这两者不是一回事。

[physics/wiener.py](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/physics/wiener.py?utm_source=chatgpt.com)

### 所以现在论文不能写：

> “We introduce a true 3D anisotropic Wiener prior for z-sparse reconstruction.”

准确说法应该更接近：

> **“We use anisotropic Wiener deconvolution on measured planes followed by axial interpolation to construct a physics-informed warm start under axial sparsity.”**

或者：

> **“an anisotropic physics-based warm start combining lateral Wiener deconvolution and axial interpolation.”**

这是一个非常重要的概念修正。

---

# 十二、而且这个问题还揭示了一个更深的问题

当前 `wiener.py` 的 docstring 写：

> xyz：先 axial zero-fill/interp，再 full anisotropic 3D Wiener

但实际代码逻辑也不是严格意义上的这个流程。

所以现在：

> **“true 3D Wiener”属于底层函数能力，而不是目前主实验 pipeline 的真实 prior。**

这是一定要在下一阶段纠正的。

---

# 十三、第二个重大问题：Protocol A 现在实际上还没有真正落地

这是我们最初方案中的另一个核心。

原先明确区分：

### Protocol A：真实低分辨率采集

$$
X\rightarrow H*X\rightarrow S\rightarrow Y_{low}
\rightarrow R\rightarrow Y_{grid}
$$

例如你说的 pollen：

> Y 方向 K=2 后，真实图像高度减少一半。

### Protocol B：dense-grid synthetic mask

$$
Y=M\odot(H*X+n)
$$

现在 `acquisition.py` 的实际实现还是：

```text
X
↓
blur
↓
mask
↓
noise
↓
zero-filled Y
```

并没有：

```text
low-resolution geometry
↓
resampling
↓
grid observation
```

虽然代码注释写了 Protocol A，但函数：

```python
simulate_acquisition()
```

本身只接受已经和 network grid 对齐的：

$$
[T,H,W]
$$

没有真正的 `Y_low -> R -> Y_grid` 数据流。

[physics/acquisition.py](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/physics/acquisition.py?utm_source=chatgpt.com)

这意味着：

> **我们之前提出的“真实采集 vs synthetic masking”统一协议，现在还停留在设计层，没有真正完成。**

这不是小问题。

因为你未来如果要做 pollen transfer：

```text
CIDC25 synthetic
↓
pollen real acquisition
```

现在还没有真正统一的 measurement-domain representation。

---

# 十四、第三个重大问题：CIDC25 的 GT / noise 语义目前存在明显混杂

这是我现在最担心的实验协议问题之一。

当前 `prepare_sparse_v3.py` 将：

```text
A1
B1
C2
D2
```

作为训练 stacks。

同时：

```text
F0
```

作为 validation/test。

而项目日志明确说明：

> CIDC25 training stacks 本身带 simulated noise，而 F0 是 clean stack。

这意味着现在实际上可能是：

```text
train target = noisy dense stack
val/test target = clean dense stack
```

然后又在 acquisition 里继续：

```text
GT
↓
PSF
↓
sampling
```

也就是说：

> **训练目标和验证目标并不是完全同分布的 target definition。**

更严重的是，当前项目一直把 training stack 当作：

> “GT”

但如果它本身已经是 noisy acquisition，则严格说不能再叫 clean GT。

这个问题比模型结构重要得多。

---

# 十五、第四个重大问题：所谓“stack-level split”其实没有完全做到 stack-independent

当前：

```text
train = A1 B1 C2 D2
val   = F0[0:750]
test  = F0[750:1500]
```

这只是：

> training stacks 与 F0 分开

但是：

> validation 和 test 仍来自同一个 F0 stack。

因此不是严格的：

```text
train stack
vs
validation stack
vs
test stack
```

而是：

```text
Train = other stacks

Validation = first half of F0

Test = second half of F0
```

对于高度相关的 microscopy volume，这仍然可能存在：

- temporal correlation
- spatial correlation
- structural recurrence

所以最终论文不能直接说：

> “stack-level independent train/val/test split”

更准确应该说：

> **training stacks are separated from the held-out F0 stack, while validation and test are temporally disjoint subsets of F0.**

[datasets.py](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/data/datasets.py?utm_source=chatgpt.com)

---

# 十六、第五个重大问题：当前 B4 的“gate 创新”还没有真正完成因果消融

目前：

$$
B3
\rightarrow
B4
$$

变化的不只是：

> gate

而是同时增加：

- mask
- confidence
- uncertainty
- separate feature encoder
- gated prior fusion

因此：

$$
B4>B3
$$

并不能严格说明：

> gate 本身造成了提升。

例如：

```text
B3
↓
simple concatenation of X0 + M + Cs + Up
```

如果也提升到相同程度，那么真正有效的是：

> “增加条件信息”

而不是：

> “gated prior fusion”。

目前 `PriorFusionStem` 是：

$$
G=\sigma(
Conv([F_S,F_W])
)
$$

$$
F=F_Y+G\odot F_W
$$

这本身是合理设计。[prior_fusion.py](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/models/prior_fusion.py?utm_source=chatgpt.com)

但还差一组非常关键的控制：

### B4 ablation

```text
B3
B4-concat
B4-gate
B4-gate-no-C
B4-gate-no-U
B4-random-U
```

否则最终“uncertainty-aware gating”这个说法还不够牢。

---

# 十七、第六个问题：你现在的 `U_p` 还不够资格叫 uncertainty

当前：

$$
U_p=f(M,H,\lambda)
$$

本质上由：

- mask
- PSF
- λ

计算出来。

它没有直接使用：

- measurement SNR
- photon count
- local residual
- model ensemble variance
- posterior variance

所以它严格来说更像：

> **physics-derived recoverability / difficulty field**

而不是：

> calibrated uncertainty。

这一点你们其实已经比较诚实，DEC-024 也没有把它直接解释成 error predictor。

但是下一版论文里，我建议统一命名：

$$
U_p
\rightarrow
R_p \text{ or } D_p
$$

例如：

> Physics Recoverability Map

比：

> Physics Uncertainty Map

更稳妥。

---

# 十八、DEC-024 的 gate-uncertainty 相关性很有意思，但现在不能过度解读

现在：

$$
corr(g,U_p)=-0.698
$$

z 情境下确实很漂亮。

但它证明的是：

> gate tendency is negatively associated with the prescribed physics difficulty map.

不能证明：

> gate predicts reconstruction uncertainty.

因为：

$$
corr(g,|error|)
$$

只有：

$$
-0.131
$$

所以这个结果应该写成：

> **the learned gate responds to the physics-derived recoverability field**

而不是：

> **the gate estimates reconstruction uncertainty**。

这是两个层次。

---

# 十九、第七个问题：MS-SSIM 的“standard implementation”实际上还存在实现问题

这一点是代码层面一个比较明确的 bug。

`reconstruction.py`：

```python
_gaussian_window()
```

返回的是：

```text
1 × 1 × 1 × K
```

然后用：

```python
F.conv2d()
```

进行卷积。

这只在 **一个空间方向**进行了 Gaussian smoothing。

严格的标准 SSIM / MS-SSIM 应该使用二维 Gaussian window，或者横纵两个方向的 separable filtering。

所以现在代码注释说：

> “standard MS-SSIM”

并不完全准确。

[reconstruction.py](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/losses/reconstruction.py?utm_source=chatgpt.com)

这会直接影响：

- SSIM 数值
- MS-SSIM 数值
- 论文表格

所以这一项应该在下一阶段优先修。

---

# 二十、第八个问题：`reval.py` 的 measured/missing 指标有一个实际实现错误

`metrics/image_metrics.py` 的：

```python
split_errors(pred,target,mask)
```

本身是正确的。

但是 `reval.py` 传进去的不是显式 measurement mask，而是：

```python
x.view(...)[0][q] != 0
```

这里 `[0]` 是：

> Y

所以现在 measured/missing mask 是：

$$
Y\neq0
$$

而不是：

$$
M
$$

这在理论上是不严谨的，因为：

> **真实测量值可以合法地等于 0。**

当前 observation 本来已经有明确：

```text
M
```

就应该直接使用。

[reval.py](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/scripts/reval.py?utm_source=chatgpt.com)

这一项建议立即修。

---

# 二十一、第九个问题：当前最漂亮的 46-50 dB 结果都还不是最终 test evidence

这是现在实验体系最容易被忽略的一点。

你当前最主要的数据，例如：

$$
B4=46.28
$$

来自：

```text
val
n_items = 8
ckpt = best
```

也就是说：

> **这是 validation evidence，不是最终 held-out test evidence。**

现在：

- severity matrix
- T-scaling
- photon budget
- external architecture

大量结果都是：

```text
reval best
n=8
F0 validation subset
```

真正的：

$$
\text{final test}
$$

还没有做完整收口。

所以现阶段我会把这些结果定位成：

> **method development evidence**

而不是：

> final paper performance.

---

# 二十二、外部架构实验值得保留，但不要给它过高权重

EXP-020 做得比之前版本规范很多。

你现在有：

- Video Swin
- SwinUNETR
- RVRT
- own Swin2D

并且有：

```text
2D
2.5D-z
3D-xyz
```

这是很好的补充实验。

而且最有价值的结果其实不是“我方模型赢了”，而是：

$$
\text{Wiener gain in z}
$$

在多个架构都为正：

- Video Swin +5.42
- RVRT +7.28
- SwinUNETR +6.39
- own +5.82

这说明：

> z-oriented physics prior 并不是只对你自己的网络有效。

这一点非常值得保留。

但 RVRT：

```text
2D / xyz
```

发生 collapse，所以这些负结果不能直接解释成：

> “RVRT is unsuitable.”

更准确是：

> **the tested adapter/training configuration was unstable for these regimes.**

因此外部 baseline 最适合放：

> Supplementary / architecture robustness study

而不是主表核心结论。

---

# 二十三、还有一个非常重要的“隐藏偏移”：你现在所谓的“轴向瓶颈”还存在 PSF confounding

这是我建议下一步一定做的实验。

现在主要 PSF 是：

$$
\sigma_{xy}=1.0
$$

$$
\sigma_z=1.5
$$

所以天然有：

$$
\sigma_z>\sigma_{xy}
$$

同时又比较：

```text
lateral sparse
vs
axial sparse
```

最后得到：

> axial is the bottleneck.

但这里至少有两个变量同时变了：

### 变量 A

sampling pattern

$$
M_{xy}
\neq
M_z
$$

### 变量 B

PSF information loss

$$
\sigma_{xy}\neq\sigma_z
$$

所以目前严格来说不能说：

> “the bottleneck is inherently axial.”

更准确的是：

> **Under the current anisotropic optical model and point-scanning sampling protocol, axial sparsification causes a substantially larger reconstruction degradation than lateral sparsification.**

这是科学上更稳的说法。

---

# 二十四、下一阶段我认为最值得做的，不是再追一个更大的 3D Transformer

这一点我非常明确。

现在已经有足够证据说明：

```text
generic 3D Transformer
        ↓
没有带来稳定增益
```

继续：

```text
SwinUNETR
UNETR++
3D ViT
3D Mamba
Video Swin V2
```

很容易重新进入：

> “换 backbone 刷 benchmark”

这会把项目重新带偏。

真正值得深挖的是：

---

# 二十五、创新方向 1：从“固定 λ Wiener”升级为 Noise-Adaptive Wiener Prior

我认为这是目前**最值得做的主创新升级**。

你现在已经发现：

$$
\lambda=3\times10^{-6}
$$

在无附加噪声下很好。

但 photon experiment 已经证明：

$$
P=50
$$

时：

$$
\lambda=3\times10^{-6}
$$

明显失配。

而：

$$
3\times10^{-5},10^{-4}
$$

可以把 b3：

$$
16.64
$$

恢复到：

$$
34.50,\ 35.29.
$$

这实际上告诉你：

> 当前真正需要解决的不是“更复杂网络”，而是 **physics prior 的 reliability adaptation**。

因此下一版可以做：

$$
\lambda=\lambda(Y,M)
$$

甚至：

$$
\lambda(x)=f_\theta(Y,M)
$$

即：

> **Adaptive Wiener Prior**

例如：

```text
Y
↓
noise estimator
↓
estimated σ_n / local SNR
↓
adaptive λ
↓
Wiener X0
↓
quality-aware fusion
```

这样就能把现在：

> “B4 对 λ mismatch 有免疫能力”

升级成一个真正的方法创新：

> **noise-adaptive physics prior generation and fusion**

这比目前 heuristic gate 更有论文价值。

---

# 二十六、创新方向 2：把现在的 `U_p` 从“几何 heuristic”变成真正的 physics recoverability map

现在：

$$
U_p=f(M,H,\lambda)
$$

下一步可以加入：

$$
SNR(x)
$$

或者：

$$
\hat\sigma_n(x)
$$

形成：

$$
R_p(x)=
g\left(
H,M,\lambda,\mathrm{SNR}
\right)
$$

甚至定义：

$$
R(f)=
\frac{|H(f)|^2}
{|H(f)|^2+\lambda(f)}
$$

再结合：

$$
M
$$

构造 spatial recoverability。

这样：

```text
sampling support
+
optical transfer
+
noise level
```

三者共同决定：

> “这里到底有多少 information 可以可靠恢复”。

这个定义会比当前的 `U_p = 1-C_p` 强很多。

---

# 二十七、创新方向 3：做“Prior Quality Calibration”，而不是简单 gate

当前 B4：

$$
F=F_Y+G\odot F_W
$$

它只有：

$$
0\leq G\leq1
$$

所以只能：

> suppress prior

不能：

> amplify prior。

而你的实验实际上已经发现：

> 不同 K / photon budget 下，prior quality 变化非常大。

因此可以改成：

$$
F=
F_Y+
\alpha(x)F_W
$$

其中：

$$
\alpha(x)\in[0,\alpha_{\max}]
$$

甚至：

$$
\alpha>1
$$

允许网络在高度可靠区域放大 physics prior。

更重要的是：

$$
\alpha(x)=f(M,C_s,R_p,\mathrm{SNR},X_0)
$$

这就从：

> gated fusion

进一步升级成：

> **prior reliability calibration / adaptive prior weighting**

这是当前 B4 最自然的第二代版本。

---

# 二十八、创新方向 4：真正把 Data Consistency 做起来

现在 B6 基本没有完成。

但是：

$$
L_{DC}
=
\frac1{|M|}
\|
M\odot(H*\hat X-Y)
\|_1
$$

其实很值得做。

因为这是目前整个项目里最缺的：

> **physics → prediction → physics consistency**

闭环。

现在的结构是：

```text
physics
↓
X0
↓
network
↓
Xhat
```

加入 DC 后：

```text
physics
↓
X0
↓
network
↓
Xhat
↓
forward model
↓
Yhat
↓
compare Y
```

这样你的网络就不是：

> physics-inspired

而会更接近：

> **physics-consistent reconstruction**

我认为这比再做一个 3D attention 更值得投入。

---

# 二十九、但 DC 不能只是简单加一个 loss

我建议最终升级成：

$$
X^{k+1}
=
D_\theta(X^k,Y,M)
-
\eta
A^T(A(X^k)-Y)
$$

也就是：

> learned reconstruction + explicit data-consistency refinement

甚至可以做：

```text
Stage 0: Wiener X0
Stage 1: neural refinement
Stage 2: measurement consistency
Stage 3: neural refinement
```

形成轻量 iterative reconstruction。

这会比现在单次：

$$
X_0\rightarrow Network\rightarrow\hat X
$$

在方法上更完整。

---

# 三十、创新方向 5：真正实现 Protocol A，然后做 Synthetic → Real transfer

我认为这是目前整个项目**最重要的应用级升级**。

把：

```text
CIDC25
```

和：

```text
Pollen
```

真正统一。

---

## Synthetic

$$
Y=M\odot(HX+n)
$$

---

## Real

$$
Y_{low}=S(HX+n)
$$

然后：

$$
Y_{grid}=R(Y_{low})
$$

统一成：

$$
(Y_{grid},M,X_0,R_p)
$$

这样你才能真正测试：

> **synthetic sparse training → real sparse acquisition inference**

这是整个项目从：

> “synthetic reconstruction benchmark”

走向：

> **real microscopy acquisition reconstruction**

的关键一步。

---

# 三十一、创新方向 6：不要放弃 3D，而是放弃“重型 3D Transformer”

这点需要特别区分。

我不建议：

> 回去继续做 full 3D attention。

但是可以做：

> **2.5D channel-mixing backbone + lightweight explicit axial interaction**

因为实验已经告诉你：

> channelized 2.5D 实际是一个非常强的 inductive bias。

所以可以保留：

```text
T slices → channels
```

然后在 bottleneck 只加入一个非常轻的：

```text
depthwise 1D Conv over T
```

或者：

```text
low-rank axial mixing
```

甚至：

$$
F_z = A_zF
$$

其中 $A_z$ 是一个很小的：

$$
T\times T
$$

动态 mixing matrix。

这样：

```text
不是：
2D → full 3D Transformer

而是：
2D/2.5D → explicit low-cost axial mixing
```

这更符合当前实验给出的真实结构规律。

---

# 三十二、创新方向 7：进一步可以走到“uncertainty-guided adaptive acquisition”

这是我认为有潜力成为下一篇更强工作的方向。

当前：

$$
U_p(x)
$$

表示哪里难恢复。

那么自然可以进一步：

$$
U_p
\rightarrow
\text{next sampling decision}
$$

例如：

```text
Initial sparse acquisition
        ↓
Reconstruction
        ↓
Recoverability / uncertainty map
        ↓
select additional z plane / y line
        ↓
new acquisition
        ↓
reconstruction
```

目标：

$$
\max
\frac{\Delta Q}{\Delta T_{acq}}
$$

即：

> **单位额外采集时间带来的最大质量提升。**

这样整个项目就从：

> sparse reconstruction

变成：

> **adaptive microscopy acquisition + reconstruction co-design**

这是更大的研究方向。

---

# 三十三、我建议现在不要继续增加实验，而是先完成一次“科学协议清算”

当前实验已经很多了，再继续跑会越来越乱。

现在最优先应该整理成一个新的：

# Phase 5A - Protocol Closure

只做这些事情：

### P5A-1

修：

```text
reval mask ≠ 0
```

改成显式 `M`。

### P5A-2

修标准 SSIM/MS-SSIM。

### P5A-3

重新定义：

```text
GT
source noise
measurement noise
```

彻底弄清 CIDC25 A1/B1/C2/D2/F0 的物理含义。

### P5A-4

正式区分：

```text
training target
validation target
test target
```

### P5A-5

把：

```text
true 3D Wiener
```

和：

```text
z warm start = lateral Wiener + axial interpolation
```

重新命名、重新描述。

### P5A-6

做一次真正独立的 test evaluation。

---

# 三十四、然后 Phase 5B：只做三个实验

我建议下一阶段不要再铺十几个实验。

只做：

## 实验 A：PSF-controlled axial-vs-lateral

固定：

$$
\sigma_x=\sigma_y=\sigma_z
$$

再做：

```text
xy
z
```

然后与当前：

$$
\sigma_z>\sigma_{xy}
$$

结果比较。

这样才能回答：

> axial bottleneck 到底是 sampling effect，还是 PSF anisotropy effect，还是两者共同作用。

---

## 实验 B：B4 mechanism decomposition

至少：

```text
B3
B4-concat
B4-gate
B4-no-U
B4-random-U
```

回答：

> 到底是 X0 有效，还是 uncertainty-aware gating 有效？

---

## 实验 C：Adaptive λ

至少：

```text
fixed λ
oracle tuned λ
estimated λ
adaptive λ
```

在：

```text
P = ∞,50,20,10,5
```

测试。

这个实验一旦做漂亮，当前 photon budget 那一条线就从“稳健性实验”升级成：

> **核心方法贡献。**

---

# 三十五、我现在对当前项目的创新层级判断

如果按照现阶段证据，我会把贡献强度分成：

### 已经比较扎实

**① Sampling-regime-dependent physics prior**

z 有稳定收益，xy 几乎无收益。

---

**② Anisotropic optical prior 的贡献经过 no-PSF 消融确认**

当前：

$$
+5.33\,dB
$$

去掉 PSF 后：

$$
+0.81\,dB
$$

这是相当有价值的 mechanistic evidence。

[EXP-021 / no-PSF 实验结果](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PSRAS/PROJECT_LOG.md?utm_source=chatgpt.com)

---

**③ Severity/context scaling**

K 曲线 + T 曲线已经形成比较完整的 sampling difficulty characterization。

---

### 目前有条件成立

**④ Sampling-aware fusion**

K≤8 成立，但 K16 反转。

所以不能写成 unconditional gain。

---

### 目前不应作为贡献

**⑤ Residual reconstruction**

已经被实验证伪。

---

**⑥ Unified full 3D Transformer**

已经关闭。

---

**⑦ 横向 Wiener prior**

目前是 negative control，而不是 contribution。

---

# 三十六、目前最值得担心的不是“创新不够”，而是“论文故事和实际代码不同步”

目前至少有五个地方必须同步：

| 论文可能想写的 | 当前代码真实情况 | 风险 |
|---|---|---|
| true 3D Wiener prior | z 主线实际为 2D measured-plane Wiener + axial interpolation | **高** |
| real + synthetic unified protocol | Protocol A 尚未真正实现 | **高** |
| clean GT reconstruction | train stacks 本身可能带噪，F0 clean | **高** |
| stack-level independent split | val/test 同属 F0 | **高** |
| uncertainty-aware fusion | $U_p$ 是 physics recoverability heuristic，且 B4 尚缺 mechanism ablation | 中高 |
| standard MS-SSIM | 当前 Gaussian filter 实际只沿一个方向 | **高** |
| measured/missing metrics | reval 用 $Y\neq0$，不是 M | 中高 |
| residual-aware method | residual 已删除 | 中 |

这些问题解决后，项目的科学完整性会比继续增加 backbone 提升得更多。

---

# 三十七、还有一个文档层面的明显问题

`Project_progress_status.md` 顶部写：

> 更新至：2026-09-12

但里面已经包含：

- EXP-020
- EXP-021
- 2026-09-18

甚至 `PROJECT_LOG` 已经推进到 9 月 18 日。

另外 `DESIGN_phase3.md` 仍然写着：

> 状态：待用户评审

但现在 Phase 3 实际已经关闭。

因此现在建议建立：

> **single source of truth**

也就是：

```text
PLAN.md
    ↓
PROJECT_LOG.md   ← 唯一实验事实记录
    ↓
Project_progress_status.md ← 自动/人工同步摘要
```

不要三个文件分别维护“现在进行到哪里”。

---

# 三十八、我对当前项目的最终判断

我现在会把它定位成：

> **一个已经完成“问题重新建模和物理先验验证”的研究项目，但还没有完成“最终方法闭环与真实数据验证”的项目。**

它现在最有价值的主线其实已经逐渐清楚：

$$
\boxed{
\text{Axial Sparse Acquisition}
\rightarrow
\text{Anisotropic Physics Prior}
\rightarrow
\text{Prior Reliability Assessment}
\rightarrow
\text{Sampling-aware Reconstruction}
}
$$

而不是：

$$
\boxed{
\text{3D Transformer}
}
$$

也不是：

$$
\boxed{
\text{Residual Learning}
}
$$

---

# 三十九、所以我建议把下一版总体方案进一步改成这样

```text
                    Sparse Acquisition
                          │
              ┌───────────┴───────────┐
              │                       │
        Sampling Support M       Optical PSF H
              │                       │
              └───────────┬───────────┘
                          ↓
                  Observation Y
                          ↓
                 Noise / SNR estimation
                          ↓
               Adaptive Wiener Prior
                          ↓
                         X0
                          │
            ┌─────────────┴─────────────┐
            ↓                           ↓
      Sampling Confidence       Physics Recoverability
             Cs                          Rp
            └─────────────┬─────────────┘
                          ↓
                 Prior Reliability
                          ↓
                 Sampling-aware Fusion
                          ↓
             2.5D Swin Reconstruction
                          ↓
                         X̂
                          ↓
                 Forward Physics A
                          ↓
                 Measurement DC
                          ↓
                  Final Reconstruction
```

这实际上是对当前所有实验结果最自然的归宿。

---

# 四十、最后给你一个非常具体的优先级

我不建议下一步立即开始新的大型模型。

应该按照：

**第一阶段：修协议**

> GT/noise semantics → independent test → Protocol A → Wiener 命名 → metric bug → MS-SSIM

**第二阶段：修方法**

> adaptive λ → reliability-aware fusion → DC

**第三阶段：做真正外部验证**

> CIDC25 → MDC25 → pollen real acquisition

**第四阶段：才考虑更高层创新**

> uncertainty-guided adaptive sampling

这样项目会从：

> “一系列做得比较漂亮的 ablation”

真正变成：

> **一个完整的 physics-aware microscopy reconstruction framework。**

目前最需要做的并不是继续堆实验，而是把上述几个科学协议问题清干净；尤其是 **“主线 z-B3 到底是不是 3D Wiener”“CIDC25 到底什么才是 GT”“真实低分辨率 Protocol A 还没落地”** 这三个问题，它们会直接决定后面论文方法部分能写到什么程度。

## Prompt:
2026/9/21 17:21:43

你的建议都非常有价值。但我想补充一点疑问：我的目标是通过稀疏采集节约成像速度（或者说节约单帧成像速度可以延长观测时间减少光毒性），那么这种情况下，2D（XY稀疏）和3D（XYZ稀疏）就显得尤为重要，尤其是在采集Y方向和Z方向时是能够真正提速的（X稀疏采集提速微乎其微）。但是当前大部分实验结果都是围绕2D（yz）或2.5D，真正有意义的是不是更应聚焦于3D（yz稀疏x稀疏）/3D（xyz稀疏）/2D(y稀疏x不稀疏)？这样的话是统一一个网络能够处理这几种情形最好，还是要继续分维度训练不同backbone？

## Response:
2026/9/21 17:22:01

> **已搜索 7 个网站**
>
> 
>
> 思考了 11s

你的这个补充非常关键。我重新结合当前 PSRAS 的实验结果和你真正的显微成像目标来看，**我认为确实应该把研究主线重新从“2.5D 重建”上移到“真正的提速型稀疏采集”**。

而且这会带来一个很重要的变化：

> **研究对象不应该再按“2D / 2.5D / 3D 网络”划分，而应该按“采集维度是否真正节省 acquisition time”来划分。**

已有工作也正在朝这个方向发展：2026 年的 DeepR-SXYZ 就直接把 3PFM 的 sparse X-Y-Z acquisition 与成像加速、降低 dwell time 和光毒性联系起来，而不是把问题仅视为图像插值。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

---

# 一、先把你的真正物理目标重新写清楚

对于点扫描式三光子显微镜，真正值得优化的是：

$$
\boxed{
\text{Acquisition Time}
\downarrow
\quad
\text{while}
\quad
\text{Reconstructed Quality}
\approx
\text{Dense Acquisition}
}
$$

进一步，如果把生物应用考虑进去：

$$
\boxed{
\text{same observation duration}
\rightarrow
\text{lower photon exposure / lower phototoxicity}
}
$$

或者：

$$
\boxed{
\text{same phototoxicity budget}
\rightarrow
\text{longer observation}
}
$$

这与压缩感知显微成像的一般目标也是一致的：减少实际采集量可以同时降低 acquisition time 和 light exposure，从而提高时间分辨率或降低光损伤。([PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5499637/?utm_source=chatgpt.com))

所以你的论文核心问题应该变成：

> **How much imaging speed can be saved by sparsifying the physically time-consuming acquisition dimensions while preserving volumetric structural fidelity through physics-aware reconstruction?**

这个问题比：

> “Swin-UNet 能不能把 sparse image reconstruct 好？”

高一个层次。

---

# 二、按照这个目标，你现在的三种实验实际上应该重新定义

我建议不要再叫：

- 2D
- 2.5D
- 3D

作为主线分类。

而是定义：

## Regime A：2D lateral sparse acquisition

$$
X(x,y)
\rightarrow
S_y
$$

即：

> **X fully sampled，Y sparse**

也就是你现在代码中的：

```text
y mode
```

这一类是非常有意义的，因为：

$$
T_{\rm acq}\propto N_y
$$

所以 Y 每隔 $K$ 行采一次：

$$
r_y=\frac1K
$$

理论 acquisition-time reduction 约：

$$
Speedup\approx K
$$

而你之前大量 `xy` grid 实验实际上包含了 X 稀疏。

这与真实点扫描系统并不完全一致。

所以：

> **未来主线应该把 `y` 作为 2D 稀疏主线，而把 `xy` 降为非物理/应力测试。**

这一点我现在非常赞同你。

---

# 三、Regime B：3D YZ sparse acquisition，X dense

这是我认为你下一阶段**最值得重点做的核心实验**。

形式：

$$
X(x,y,z)
\rightarrow
S_yS_zX
$$

也就是：

```text
X：dense
Y：1/K
Z：1/K
```

采样率：

$$
r=\frac1{K^2}
$$

而如果 acquisition time：

$$
T\propto N_yN_z
$$

那么理论上：

$$
\boxed{
Speedup\approx K^2
}
$$

例如：

| K | Y保留率 | Z保留率 | 3D数据保留率 | 理论采集时间 |
|---:|---:|---:|---:|---:|
| 2 | 50% | 50% | 25% | 4× |
| 4 | 25% | 25% | 6.25% | 16× |
| 8 | 12.5% | 12.5% | 1.56% | 64× |

这才真正体现：

> **少采样 → 少扫描 → 快很多。**

因此我认为它应该成为你论文里的**核心 3D acquisition regime**。

---

# 四、Regime C：XYZ sparse

再进一步：

$$
X\rightarrow S_xS_yS_zX
$$

但是这里必须非常谨慎地处理论文叙事。

因为你自己已经指出：

> X 稀疏几乎不节省 point-scanning acquisition time。

因此：

$$
S_x
$$

主要增加的是：

- information compression
- detector/data throughput reduction
- computational burden reduction

而不是主要的：

$$
T_{acq}\downarrow
$$

所以我不建议把：

> XYZ sparse

直接定义为“速度主线”。

更准确的定位是：

> **extreme sparse volumetric reconstruction / stress test**

而真正的 acquisition-speed 主线是：

$$
\boxed{Y}
$$

以及：

$$
\boxed{YZ}
$$

---

# 五、这样一重新定义之后，你现在很多实验的优先级会发生变化

当前：

```text
xy
z
xyz
```

其实和真实 acquisition relevance 并不完全对应。

我建议未来改成：

```text
                 Physical acquisition dimensions
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          X              Y              Z
       ~no speed      speed saving    speed saving
          │              │              │
          ↓              ↓              ↓
     stress test        2D-Y           3D-YZ
                          │              │
                          └──────┬───────┘
                                 ↓
                         main acquisition line
                                 │
                                 ↓
                         3D-XYZ stress test
```

这样整篇论文的物理逻辑会立刻清晰很多。

---

# 六、那么第二个问题：到底应该“统一一个网络”，还是“不同维度不同 backbone”？

我的判断是：

> **最终应该统一一个网络，但不是把现在的 full-3D Transformer 硬改成 universal 3D network。**

这一点非常重要。

我不建议：

```text
2D → SwinUNet2D
YZ → SwinUNet3D
XYZ → Video Swin
```

三个 backbone 各训一个。

因为这样虽然容易得到每个 regime 的局部最优结果，但论文科学问题会变成：

> “不同模型分别处理不同稀疏模式。”

而你的真正问题其实是：

> **同一个 sparse acquisition reconstruction framework 能否根据 sampling pattern 自适应恢复不同 acquisition regime？**

后一种问题明显更完整。

---

# 七、但“统一网络”应该是统一什么？

不是统一成：

$$
3D\ Transformer
$$

而应该统一成：

$$
\boxed{
\text{Shared Spatial Reconstruction Backbone}
+
\text{Lightweight Sampling-conditioned Axial Module}
}
$$

也就是说：

### 共享的空间 backbone

始终是：

```text
2D Swin encoder
      ↓
spatial feature extraction
      ↓
spatial decoder
```

这部分负责：

$$
X-Y
$$

结构恢复。

---

### 轻量 axial module

只在存在多个 z slice 时工作：

$$
F_{z}=A(F_{t})
$$

例如：

- 1D depthwise convolution
- lightweight axial mixer
- low-rank temporal mixing
- gated cross-slice aggregation

而不是大规模：

$$
T\times H\times W
$$

联合 attention。

---

# 八、其实你现在已有的失败实验已经给出了这个方向的依据

之前 full-3D Transformer：

$$
45.82
$$

反而低于：

$$
46.28
$$

而 factorized 3D 版本也明显下降。

这说明：

> **在你的任务里，“显式 3D Transformer”并不是提升性能的必要条件。**

原因之一是：

你的 2.5D Swin 实际上已经把：

```text
T slices
```

全部作为 channel：

```text
[T × features]
```

然后每个 2D convolution / feature mixing layer 都能混合 z information。

这其实是一个很强的低成本 axial interaction。

所以我认为现在不应该推翻这个结构，而应该把它**正式抽象出来**。

---

# 九、我建议新的统一网络设计成“slice-shared + axial-mixer”

可以变成：

$$
Y,M,X_0,C_s,U_p
$$

输入维度：

$$
[B,F,T,H,W]
$$

然后：

### Step 1：每一层的空间编码共享权重

$$
F_t=E([Y_t,M_t,X_{0,t},C_t,U_t])
$$

所有 $t$ 使用同一个 encoder。

所以：

$$
T=1
$$

自然就是：

> Y-sparse 2D

而：

$$
T>1
$$

自然就是：

> YZ / XYZ volumetric reconstruction

---

### Step 2：轻量轴向聚合

$$
F'_t=
F_t+
G_t\odot A_z(F_1,\cdots,F_T)
$$

其中：

$$
G_t=f(M,C_s,U_p)
$$

所以只有当：

> 邻层真的有可靠信息

时才加强 z interaction。

---

### Step 3：共享空间 decoder

$$
\hat X_t=D(F'_t)
$$

因此一个网络自然支持：

```text
T=1
↓
Y-sparse 2D

T=3/5/7...
↓
局部 YZ / XYZ

T=15
↓
完整局部 3D reconstruction
```

---

# 十、这里有一个关键点：不要再让 T 决定网络结构

当前旧方案有一个隐患：

```text
T = 1
→ model A

T = 15
→ model B
```

而真正应该做到：

$$
\boxed{
T\text{ 是输入条件，而不是 architecture condition}
}
$$

也就是说模型应该接受：

$$
T\in[1,T_{max}]
$$

通过：

- padding
- query mask $Q$
- axial attention/mixer mask
- validity mask

处理不同的 T。

这样：

$$
T=1
$$

不是一个特殊的 2D model。

它只是：

> **universal model 在单层输入上的退化情况。**

这才真正有“统一模型”的价值。

---

# 十一、但我要特别提醒：现在不能简单把“当前 Swin2D 改成 T-variable”

因为当前：

```python
in_channels = F × T
```

这意味着：

$$
T=1\Rightarrow5 channels
$$

而：

$$
T=15\Rightarrow75 channels
$$

因此不能直接让第一层 convolution 接收任意 T。

所以新模型最好不要：

```text
Conv2D(5T → 48)
```

而改成：

$$
\boxed{
\text{per-slice shared encoder}
}
$$

即：

```text
[B,T,5,H,W]
        ↓
shared spatial encoder
        ↓
[B,T,C,H,W]
```

之后再：

```text
axial mixer
        ↓
[B,T,C,H,W]
```

这会从根本上解决：

> variable T + unified model

的问题。

---

# 十二、这其实还能解决你现在另一个关键问题：YZ 与 XYZ

因为对每一个 slice：

```text
Y_t
M_t
X0_t
Cs_t
Up_t
```

都是统一的。

差别只存在于：

$$
M
$$

本身。

例如：

### Y-sparse

```text
M = full X × sparse Y
```

### YZ-sparse

```text
M = sparse Y × sparse Z
```

### XYZ-sparse

```text
M = sparse X × sparse Y × sparse Z
```

所以网络根本不需要：

```text
if mode == y:
    model_y
elif mode == yz:
    model_yz
elif mode == xyz:
    model_xyz
```

而是：

$$
\boxed{
\text{same network}
+
M
+
C_s
+
R_p
}
$$

自己知道现在的 acquisition geometry。

这恰好也与你现在最初提出的：

> sampling-aware reconstruction

高度一致。

---

# 十三、这样一来，你原来的“uncertainty-aware fusion”反而有了真正的作用

因为现在不同 sampling regime：

$$
Y
$$

与：

$$
YZ
$$

与：

$$
XYZ
$$

的：

$$
M
$$

完全不同。

于是：

$$
C_s
$$

告诉模型：

> “这里离真实采样点有多近。”

而：

$$
U_p
$$

告诉模型：

> “这里从光学上有多少信息是可恢复的。”

那么：

$$
G=f(M,C_s,U_p)
$$

就不是一个单纯为 z-K8 调出来的 trick。

而变成真正的：

> **sampling-conditioned prior modulation**

这会明显提升论文方法的完整性。

---

# 十四、所以现在我建议把三个核心实验改成这样

## Main Experiment 1：Y-sparse 2D

$$
X \rightarrow Y\text{-sparse}
$$

例如：

$$
K_y=2,4,8,16
$$

评价：

- quality
- acquisition speed
- speed-quality trade-off

这里不需要重点讨论 Wiener，因为现在实验已经说明：

> 横向 Wiener 增益接近 0。

所以：

$$
X_0
$$

可以主要作为：

> interpolation / spatial prior

而不是核心物理反卷积贡献。

---

# 十五、Main Experiment 2：YZ-sparse 3D

这是我认为应该变成论文主表的实验。

例如：

$$
K_y=K_z=2,4,8
$$

然后：

$$
r=\frac1{K_yK_z}
$$

报告：

$$
\text{Acquisition reduction}
$$

以及：

$$
PSNR,\ SSIM,\ MAE
$$

最重要的是：

$$
\boxed{
\text{Quality vs Speed}
}
$$

例如：

```text
Dense:        1× speed
YZ-K2:        4×
YZ-K4:       16×
YZ-K8:       64×
```

然后展示 reconstruction quality 曲线。

这样论文会突然从：

> image reconstruction paper

变成：

> **computational microscopy acceleration paper**

这是非常重要的升级。

---

# 十六、Main/Stress Experiment 3：XYZ-sparse

例如：

$$
K_x,K_y,K_z
$$

但这里不要把 X sparsity 解释为：

> acceleration factor.

而应该拆成：

$$
\text{sampling compression}
$$

和：

$$
\text{acquisition acceleration}
$$

两个指标。

例如：

$$
\text{data compression}
=
\frac1{K_xK_yK_z}
$$

但是：

$$
\text{time reduction}
\approx
\frac1{K_yK_z}
$$

这会是一个很漂亮的物理定义。

---

# 十七、所以我现在反而不建议继续单独研究“2.5D”

2.5D 可以继续存在，但它应该被重新定义为：

> **network context mechanism**

而不是一个 scientific acquisition regime。

也就是说：

```text
2.5D
```

回答：

> 网络看多大 z-context？

而：

```text
Y / YZ / XYZ
```

回答：

> 实际采集省多少时间？

这两个维度必须分开。

这一点非常重要。

---

# 十八、你现在的 T-scaling 实验也因此有了新的定位

目前：

$$
T=1,3,7,15,31
$$

主要回答：

> how much axial context is needed?

它不应该作为“2.5D 主结果”。

而应该成为：

### Context sufficiency experiment

证明：

$$
T\ge 7
$$

以后性能趋于饱和。

那么统一网络就可以把：

$$
T=7
$$

左右作为默认 context。

这样计算量可以大幅降低。

---

# 十九、这会直接带来一个很有价值的研究问题：最小采样上下文

你现在已经观察到：

$$
T=3\rightarrow7
$$

出现约：

$$
13\,dB
$$

跃升。

这说明：

> **不是“看得越多越好”，而是“至少看到一个可靠的 axial anchor”。**

所以后面可以把它正式定义成：

$$
T_{\min}(K)
$$

例如：

$$
T_{\min}\approx 2K-1
$$

或者你现在观察到：

> 实际有效上下文大概只需要达到最近 measured plane。

这可以进一步研究：

$$
T_{\min}
=
f(K,\mathrm{PSF},SNR)
$$

这其实非常有意思。

---

# 二十、因此新的统一模型，我推荐这个结构，而不是 full 3D Transformer

我会建议最终架构：

```text
                 Sparse volume
                     │
          ┌──────────┴──────────┐
          │                     │
     Observation Y          Sampling M
          │                     │
          └──────────┬──────────┘
                     ↓
              Physics Prior
             ┌───────┴────────┐
             │                │
         Wiener X0       Recoverability Rp
             │                │
             └───────┬────────┘
                     ↓
          Slice-wise Shared Encoder
                     ↓
                F[t,H,W,C]
                     ↓
          Sampling-conditioned
             Axial Mixer
                     ↓
                F'[t,H,W,C]
                     ↓
             Shared Decoder
                     ↓
             Dense Volume X̂
                     ↓
           Forward Physics / DC
                     ↓
               Final output
```

这一个网络：

$$
\boxed{
T=1
}
$$

可以做 Y-sparse 2D。

$$
\boxed{
T>1,\ M_z\neq1
}
$$

可以做 YZ 3D。

$$
\boxed{
M_x,M_y,M_z
}
$$

全部 sparse 时可以做 XYZ。

---

# 二十一、我甚至建议把“统一网络”作为方法贡献，而不是“3D Transformer”

这是两种完全不同的故事。

### 旧故事

> We develop a unified 3D transformer for sparse-to-dense reconstruction.

目前证据不支持。

### 新故事

> **We develop a sampling-conditioned unified reconstruction framework that uses a shared spatial reconstruction backbone and lightweight axial interaction to handle heterogeneous sparse acquisition patterns.**

这个与现有实验结果高度吻合，而且：

- 不需要证明 full 3D attention 更强；
- 不需要三个 backbone；
- 不需要为了 3D 强行增加计算量；
- 可以真正连接 Y / YZ / XYZ 三种 acquisition。

---

# 二十二、还有一点需要根据最近的文献现实来调整预期

2026 年已经有工作直接做：

> sparse X-Y-Z three-photon fluorescence microscopy

并且明确把 sparse volumetric acquisition 与速度、dwell time、光毒性联系起来；其方法甚至报告了 XY 平面加速和 Z 层恢复。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

所以你的工作如果最终只做到：

> “CIDC25 synthetic sparse → 2.5D reconstruction PSNR 提高若干 dB”

竞争力会比较弱。

反过来，如果做成：

$$
\boxed{
\text{Y / YZ acquisition}
+
\text{physics-aware reconstruction}
+
\text{speed-quality tradeoff}
+
\text{real acquisition validation}
}
$$

就会明显更贴近当前三光子显微成像的发展方向。已有压缩感知工作也说明，减少采集量、降低光照和加速 3D/4D 成像本身就是一个重要目标。([PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5499637/?utm_source=chatgpt.com))

---

# 二十三、因此，我会正式修改我们之前的项目主线

从原来的：

> **Sparse-to-Dense Reconstruction with Physics-aware 3D Network**

改成：

> **Physics-aware reconstruction for time-saving sparse volumetric three-photon microscopy**

核心研究变量：

$$
\boxed{
\text{Sampling pattern}
\times
\text{Sampling severity}
\times
\text{Optical prior quality}
\times
\text{Context}
}
$$

核心目标：

$$
\boxed{
\max\ \frac{\text{reconstruction fidelity}}
{\text{acquisition time}}
}
$$

而不是：

$$
\max\text{PSNR}
$$

---

# 二十四、最终我对“统一网络 vs 分 backbone”的明确建议

我的建议是：

> **最终采用统一网络。**

但是统一的是：

$$
\boxed{
\text{Shared spatial backbone}
+
\text{sampling-conditioned lightweight axial module}
}
$$

而不是：

$$
\boxed{
\text{one giant 3D Transformer}
}
$$

具体地：

| 任务 | 是否独立 backbone | 推荐 |
|---|---|---|
| Y-sparse 2D | 否 | unified model, T=1 |
| YZ-sparse 3D | 否 | unified model, T>1 |
| XYZ-sparse 3D | 否 | unified model + sampling mask |
| 不同 K | 否 | K 作为 sampling condition |
| 不同 T | 否 | variable T + query/valid mask |
| 极端 XYZ | 否 | 同模型，作为 stress regime |
| 外部真实 acquisition | 否 | 同模型，换 acquisition operator |

但我建议：

> **先不要马上把现有代码直接改成这个 unified network。**

因为你现在最合理的顺序应该是：

### Step 1
先把主线实验从：

```text
xy / z / xyz
```

改成：

```text
Y / YZ / XYZ
```

并把真实 acquisition-time model 固定下来。

### Step 2
在现有 `Swin2D` 上分别跑：

```text
Y-K2/4/8
YZ-K2/4/8
XYZ-K2/4/8
```

建立真正的 speed-quality baseline。

### Step 3
再实现：

$$
\text{Shared slice encoder + axial mixer}
$$

做：

```text
separate models
vs
unified model
```

这个 comparison 本身就是非常重要的实验。

### Step 4
最后再把：

$$
X_0+C_s+U_p+\text{adaptive λ}+DC
$$

逐渐加进去。

---

**这次你的这个补充实际上让我建议对整个项目做一次“研究目标层面的再定位”：主线应该从“不同维度图像重建”转向“物理上真正节省采集时间的 Y / YZ 稀疏体成像”。** 现有 PSRAS 的大量 z-方向实验并不是浪费，它们恰好给了新的 YZ 主线非常重要的 physics-prior 证据；只是下一步应该把 `z` 从“2.5D 实验对象”提升为“YZ volumetric acquisition 的一个维度”。 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

---

**Sources:**

- [Hundred-Nanosecond Equivalent Pixel Dwell Time for Deep-Tissue 3D Three-Photon Fluorescence Microscopy via Sparse X-Y-Z Reconstruction - PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com)
- [Faster and less phototoxic 3D fluorescence microscopy using a versatile compressed sensing scheme - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5499637/?utm_source=chatgpt.com)

## Prompt:
2026/9/21 18:26:48

我还需要补充两点：①你在Main Experiment 1：Y-sparse 2D 提到”不需要重点讨论 Wiener，因为现在实验已经说明：横向 Wiener 增益接近 0“，但先前在XY情境下的实验，一方面只是得出横向 Wiener 增益相较于Bilinear接近 0，而不是横向 Wiener相较于直接网络恢复增益为0；另一方面先前的网络与当下思路不一致，可能实验结论不一定会完全保持，还是需要补充对比的。②在3D情境下设计方法时，还需要区分考虑XYZ（体成像）和XYT（时间帧序列）的区别，虽然在重建算法上可以有共同之处，但不能简单混为一谈，尤其是在写出论文表述时看哪种更有利于提高创新价值。依据以上信息（包括先前对话中分析到的“真 3D Wiener”，实际上没有用于主线 z reconstruction、CICD25 stack-level split时val和test都用到同一笔F0等问题），帮我将下一阶段需要修改完善的地方形成一个完整的项目修正方案。

## Response:
2026/9/21 18:27:06

我会把下一阶段修正分成“实验可信性修复、方法学重构、XYZ/XYT分流、统一网络设计、主实验矩阵、论文叙事”六部分，并把当前已暴露的问题逐一闭环。

先纠正一个关键判断：旧的“横向 Wiener 增益接近 0”只回答了 **Wiener→Bilinear** 的问题，没有回答 **Wiener-prior→直接网络** 的问题；因此这条结论不能直接删掉。

> **制定项目修正方案**
>
> 我会把修正分成数据隔离、Wiener实现、网络公平对照、损失设计、训练验证和结果分析几条主线，先修正可影响结论可信度的问题，再优化模型指标。
>
> **制定项目修正方案**
>
> 
>
> **已搜索 8 个网站**
>
> 
>
> 思考了 26s

可以。结合你补充的两个关键点，以及前面已经审计出来的 **真 3D Wiener、F0 split、SSIM、mask、Protocol A、旧网络结论不可直接继承** 等问题，我建议不要再做局部补丁，而是把 PSRAS 下一阶段定义为一次 **“实验协议重置 + 方法主线重构 + 数据划分重建”**。

尤其有一个现在必须纳入论文定位的现实因素：2026 年已经有工作专门针对 3PM 做 sparse X-Y-Z reconstruction，并把 CNN 与 Transformer 结合用于体数据重建，因此后续不能再把“XYZ 稀疏 + Transformer”本身作为主要创新点。真正需要建立的是 **采集维度-物理先验-稀疏预算-统一重建网络-实际成像时间/光剂量之间的系统关系**。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

下面我给出一个可以直接作为项目下一阶段执行依据的完整修正方案。

---

# 一、首先重新定义项目科学问题

原来的问题实际上偏向：

> sparse input → reconstruction network → dense image

现在应该升级为：

> **在 3PM 实际采集时间和光剂量受限的情况下，如何设计跨采集维度的稀疏采样策略，并利用物理可恢复性与采样结构进行高保真 sparse-to-dense reconstruction？**

因此项目不再单纯追求：

$$
\max Q(\hat X)
$$

而是：

$$
\max_{\mathcal S}
\frac{Q(\hat X_{\mathcal S})}
{T_{\mathrm{acq}}(\mathcal S)}
$$

或者进一步考虑剂量：

$$
\max_{\mathcal S}
\frac{Q(\hat X_{\mathcal S})}
{T_{\mathrm{acq}}(\mathcal S)}
,\qquad
\min D_{\mathrm{laser}}(\mathcal S)
$$

其中 $\mathcal S$ 是采样策略。

这会让项目从“做一个 reconstruction network”变成：

> **sampling-physics-reconstruction co-design**

这个定位明显比单纯换网络更有研究价值。

---

# 二、第一优先级：先把旧实验中的“不可继承结论”全部封存

建议在当前 `PSRAS/` 下明确建立一个：

```text
PSRAS/
├── LEGACY_RESULTS/
├── CORRECTED_PROTOCOL/
└── MAIN_EXPERIMENTS/
```

旧的 z-only、旧 2D、旧 full-3D transformer 等结果都保留，但论文主结论不能直接继承。

尤其把以下三条明确标记为：

> **Legacy evidence - not directly used as final quantitative evidence**

### 1. 横向 Wiener ≈ Bilinear 的结论不能外推为“Wiener 没用”

你指出的问题完全正确。

旧实验实际回答的是：

$$
Q(X_{0,\mathrm{Wiener}})
-
Q(X_{0,\mathrm{Bilinear}})
\approx 0
$$

但你真正关心的另外一个问题是：

$$
Q(f(Y))
$$

vs.

$$
Q(f(Y,X_{0,\mathrm{Wiener}}))
$$

即：

> **横向 Wiener 作为 network prior，是否能帮助新的 reconstruction network？**

这两个实验问题完全不同。

因此新的 Y-sparse 实验必须至少包含：

| Condition | 输入 |
|---|---|
| Direct network | $Y,M$ |
| + Bilinear | $Y,M,X_{0,B}$ |
| + Lateral Wiener | $Y,M,X_{0,W}$ |
| Wiener only | $X_{0,W}$ |
| Bilinear only | $X_{0,B}$ |

而且必须保证：

- 完全相同 backbone
- 完全相同训练集
- 完全相同随机种子
- 完全相同 optimizer
- 完全相同训练 epochs
- 完全相同 normalization
- 唯一变量就是 $X_0$

这样才可以回答：

$$
\Delta_{\mathrm{Wiener|Bilinear}}
=
Q(f(Y,X_W))-Q(f(Y,X_B))
$$

和

$$
\Delta_{\mathrm{Wiener|Direct}}
=
Q(f(Y,X_W))-Q(f(Y))
$$

两个不同问题。

所以我建议：

> **Lateral Wiener 不能删掉，必须重新实验，但它从“预设有效/无效的 prior”改成正式的 ablation variable。**

---

# 三、第二优先级：彻底修正“真 3D Wiener”问题

这是目前理论和代码之间最值得优先修正的问题之一。

现在 `PSRAS/physics/wiener.py` 中：

```python
wiener_deconv_3d(...)
```

数学上确实是 3D FFT Wiener：

$$
X_W =
\mathcal F^{-1}
\left[
\frac{H^*}
{|H|^2+\lambda}
\mathcal F(Y)
\right]
$$

但是当前主线 z reconstruction 的 `warm_start()` 并不是：

> sparse Z measurement → genuine 3D Wiener

而是近似：

> measured slices → lateral 2D Wiener → axial interpolation

所以论文里现在绝对不要再写：

> “3D Wiener prior is used for z-sparse reconstruction.”

这会被代码直接反驳。

应该拆成三个明确的 prior。

### Y-sparse

$$
X_0^{Y}=P_Y(Y)
$$

可以分别测试：

$$
P_Y^{Bilinear}
$$

与

$$
P_Y^{Wiener-XY}
$$

### YZ-sparse

需要真正定义：

$$
X_0^{YZ}=P_{YZ}(Y,M,H)
$$

它必须同时面对：

- Y 方向缺失
- Z 方向缺失
- 3D PSF

因此才有资格称为：

> **3D physics-based prior**

### XYZ-sparse

进一步：

$$
X_0^{XYZ}=P_{XYZ}(Y,M,H)
$$

这里才是真正完整的三维稀疏逆问题。

---

# 四、甚至建议不要把“标准 Wiener”作为最终 physics prior

这是下一阶段很值得升级的地方。

对于 YZ / XYZ sparse sampling，简单：

$$
X_0 =
\mathcal F^{-1}
\left[
\frac{H^*}{|H|^2+\lambda}
\mathcal F(Y)
\right]
$$

在 zero-filled sparse data 上并不一定是最合理的物理逆。

更合理的统一形式是：

$$
X_0=
\arg\min_X
\left\|
M\odot(\mathcal H(X)-Y)
\right\|_2^2
+
\lambda R(X)
$$

其中：

$$
\mathcal H = S\circ H
$$

或者真实 acquisition Protocol A 中：

$$
\mathcal H=R\circ S\circ H
$$

这样：

- $H$：PSF
- $S$：sparse sampling
- $R$：实际采集分辨率/几何重采样
- $M$：measurement support

全部统一进 forward model。

这比简单讲：

> “we use a 3D Wiener filter as prior”

明显更强。

甚至可以进一步做成：

> **Sampling-conditioned physics inversion**

然后网络学习的是：

$$
\hat X = f_\theta(Y,M,X_0,C_s,U_p)
$$

而不是直接从 sparse image 猜 dense image。

这应该成为下一阶段最值得投入的方法创新方向。

---

# 五、第三优先级：彻底重构 acquisition regime

这一点我建议现在正式定下来。

不要再用：

> 2D / 2.5D / 3D

作为核心实验分类。

因为这三个词把三个完全不同的概念混在了一起：

1. acquisition dimensionality
2. reconstruction dimensionality
3. network context

应该改成：

## A. Spatial acquisition sparsity

### Regime Y

$$
X:\text{dense}
\qquad
Y:\text{sparse}
$$

即：

> **Y-sparse 2D imaging**

单个 XY plane。

---

### Regime YZ

$$
X:\text{dense}
\qquad
Y,Z:\text{sparse}
$$

即：

> **YZ-sparse volumetric imaging**

这是我建议新的核心 3D 主线。

因为对于 point-scanning 3PM：

- X 是单条 scan line 内部的采样
- Y 决定 scan line 数量
- Z 决定 optical section 数量

因此真正直接减少 volume acquisition burden 的就是 Y 和 Z。

理想情况下：

$$
T_{\mathrm{acq}}
\propto N_YN_Z
$$

因此：

$$
A_{\mathrm{ideal}}
\approx K_YK_Z
$$

但论文中必须叫：

> **idealized/nominal acceleration**

而不能直接写成实际硬件 speedup，除非后面实测。

---

### Regime XYZ

$$
X,Y,Z:\text{sparse}
$$

这个作为：

> **extreme spatial undersampling / compression regime**

而不是你的核心 acquisition-speed regime。

因为 X sparsity 对 point scanning 的实际时间优势取决于具体硬件和扫描实现。不能在没有硬件测量的情况下把 X sparse 的采样比例直接等价为 acquisition acceleration。

---

# 六、非常建议新增一个核心实验：Fixed-budget sampling allocation

这实际上可能成为项目里很漂亮的创新点。

例如保持：

$$
K_YK_Z=16
$$

比较：

$$
(K_Y,K_Z)
=
(2,8),(4,4),(8,2)
$$

或者：

$$
K_YK_Z=32
$$

比较不同 allocation。

这样回答的不再只是：

> “我能不能从 sparse data reconstruct dense volume？”

而是：

> **在相同 acquisition budget 下，Y 和 Z 应该分别稀疏多少？**

即：

$$
\max_{K_Y,K_Z}
Q(\hat X)
$$

subject to

$$
C(K_Y,K_Z)\leq B
$$

这非常契合你的真实应用目标。

因为 Y 和 Z 的：

- PSF
- sampling difficulty
- anatomical redundancy
- interpolation difficulty
- physics recoverability

都不一样。

所以：

$$
K_Y=K_Z
$$

本身并不是合理的默认最优设计。

这个实验有潜力把项目从：

> reconstruction

提升到：

> **acquisition strategy optimization**

---

# 七、XYZ 与 XYT 必须正式分成两个 scientific branch

你第二点也非常重要。

虽然数据张量形式都可能写成：

$$
X\times Y\times D
$$

但是 $D$ 的物理意义完全不同。

---

## Branch A：XYZ = volumetric reconstruction

这里：

$$
D=Z
$$

Z 是空间坐标。

目标是：

$$
I(X,Y,Z)
$$

因此三维相关性主要是：

> spatial structural continuity

而且存在真实的 optical PSF：

$$
H(x,y,z)
$$

所以：

- 3D Wiener
- PSF
- sampling mask
- spatial recoverability
- volumetric consistency

都具有明确物理意义。

---

# 八、Branch B：XYT = temporal reconstruction

这里：

$$
D=T
$$

T 不是 spatial axis。

目标是：

$$
I(X,Y,t)
$$

关键 prior 变成：

> temporal continuity / motion consistency / dynamic evolution

而不是：

> axial PSF consistency

这是必须在方法和论文写作上分开的。

尤其：

### XYZ

缺失的是：

$$
Z
$$

可以通过空间邻层推断。

### XYT

缺失的是：

$$
T
$$

但相邻时间帧之间可能存在：

- motion
- deformation
- appearance change
- intensity dynamics
- cell displacement

因此：

$$
I_t \not\approx I_{t+1}
$$

不能简单把 T 当成 Z。

---

# 九、XYZ 和 XYT 可以共享网络，但不能共享“先验定义”

我推荐的最终架构不是：

> 一个完全统一的 3D Transformer

也不是：

> 两套完全独立 backbone

而是第三种：

# Shared Backbone + Axis-specific Physics/Context Adapter

可以理解为：

```text
                 sparse observation
                        │
             ┌──────────┴──────────┐
             │                     │
        Spatial Encoder      Sampling Encoder
             │                     │
             └──────────┬──────────┘
                        │
               Shared Feature Space
                        │
            ┌───────────┴───────────┐
            │                       │
      Spatial/ZY Adapter       Temporal Adapter
            │                       │
        XYZ / YZ / Y             XYT
            │                       │
            └───────────┬───────────┘
                        │
                  Shared Decoder
                        │
                 Dense reconstruction
```

这样真正实现：

> **same reconstruction backbone, different physical meaning**

---

# 十、统一网络不要再采用当前的 `Conv2D(5T → base)` 思路

这个旧设计对于固定 T 可以工作，但对于你现在要统一 Y / YZ / XYZ / XYT 会产生结构性问题。

应该改成：

$$
[B,T,5,H,W]
$$

先对每个 slice / frame 使用共享 encoder：

$$
E_s:
[B,T,5,H,W]
\rightarrow
[B,T,C,H,W]
$$

也就是：

```text
slice 1 ──┐
slice 2 ──┤
slice 3 ──┤ → shared spatial encoder
...       │
slice T ──┘
```

之后才进行 axial interaction：

$$
F' = A(F)
$$

这里的 $A$ 可以根据任务不同：

### XYZ / YZ

$$
A=A_Z
$$

学习 volumetric axial interaction。

### XYT

$$
A=A_T
$$

学习 temporal interaction。

这样：

$$
T=1
$$

自然就是 Y-sparse 2D。

$$
T>1
$$

自然支持 YZ / XYZ / XYT。

这比人为把 2D、2.5D、3D 定义成不同网络更干净。

---

# 十一、但“一个网络”不等于“所有东西共享”

这是我建议最终采用的核心设计原则：

## Shared

共享：

- spatial encoder
- low-level feature extraction
- reconstruction decoder
- sampling feature encoder
- basic fusion mechanism

## Axis-specific

不共享：

- physics prior generation
- axial Z module
- temporal T module
- possibly Z/T attention adapter

这样能同时回答两个审稿人问题：

### 审稿人 A

> Why not train separate networks?

因为不同 acquisition regimes 共享低层 spatial representation。

### 审稿人 B

> How can Z and T be treated as equivalent?

它们并没有被等价处理。

而是：

$$
A_d(F),\qquad d\in\{Z,T\}
$$

采用 dimension-specific adapter。

这会比简单“统一网络”更有说服力。

---

# 十二、XYZ 与 XYT 在论文里的地位，我建议不要平权

从你目前整个项目的科学目标出发，我建议最终论文 hierarchy 明确设置成：

## Main storyline

### ① Y-sparse 2D

证明：

> sparse Y acquisition 可以降低单平面 acquisition burden，同时 reconstruction 保持高质量。

### ② YZ-sparse 3D

这是核心。

证明：

> simultaneous spatial sparsity in Y and Z allows accelerated volumetric acquisition.

### ③ Fixed-budget Y/Z allocation

这是创新增强点。

证明：

> acquisition budget 应该根据 reconstruction difficulty 在 Y/Z 之间自适应分配。

### ④ XYZ

作为 extreme stress-test：

> further extending the framework to full spatial undersampling.

这里需要格外注意已有 2026 工作已经专门做 sparse XYZ 3PFM，因此你不能仅靠“XYZ + Transformer”构成 novelty。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

### ⑤ XYT

作为 extension：

> temporal generalization to dynamic imaging.

而不是主创新。

这样论文叙事会非常清晰：

> **从二维采集加速 → 三维体采集加速 → 最优采样预算 → 极端空间稀疏 → 动态时间维度扩展**

而不是一上来：

> “我们同时解决 2D、3D、video。”

后者非常容易让论文看起来像 generic reconstruction framework。

---

# 十三、XYT 最好不要作为“第三个 3D”来写

论文里最好明确：

$$
XYZ = \text{3D spatial volume}
$$

$$
XYT = \text{spatiotemporal sequence}
$$

可以明确写：

> Although volumetric and temporal reconstruction share a common computational backbone, they represent fundamentally different missing dimensions: Z corresponds to spatial depth, whereas T represents temporal evolution.

然后：

$$
X_0^{XYZ}
=
P_{\mathrm{physics}}(Y,M,H)
$$

但：

$$
X_0^{XYT}
=
P_{\mathrm{temporal}}(Y,M)
$$

两者不要强行共用 Wiener。

这一点反而可以成为论文里体现“physics-aware”的重要地方：

> **the reconstruction prior is conditioned not only on sampling density but also on the physical semantics of the missing axis.**

---

# 十四、现有项目的数据划分必须优先修

这是当前实验可信度的 P0 问题。

现在 CICD25 结构里：

- train：A1/B1/C2/D2
- val/test：都来自 F0 的不同 temporal range

这并不能严格称为：

> independent stack-level test

因为 val 和 test 都来自同一个 F0 stack。

虽然如果时间区间完全不重叠，它未必属于传统意义上的“target leakage”，但它仍然存在：

> **same-stack correlation / limited independence / pseudoreplication risk**

因此 final paper 中不能把它包装成完全独立 test set。

---

# 十五、建议重新设计 split

理想情况：

```text
Training stacks
A1
B1
C2
D2
        ↓
Validation stack
F0
        ↓
Independent Test stack
NEW STACK
```

最佳情况则进一步做到：

```text
train:
multiple animals / experiments

validation:
held-out acquisition session

test:
held-out animal / acquisition session
```

如果当前数据确实只有 F0 可作为 held-out：

那么：

### 当前 F0 temporal split 可以保留

但只能称为：

> temporal hold-out evaluation

不能称：

> independent stack-level test

最终必须补：

> independent stack / sample / acquisition session

作为真正 test。

这是比换一个网络更加重要的问题。

---

# 十六、同时修正一个当前 re-evaluation 的统计问题

`PSRAS/scripts/reval.py` 现在：

```python
x.view(...)[0][q] != 0
```

来推断 measurement mask。

这个方法不可靠，因为：

$$
Y_i=0
$$

并不等于：

$$
M_i=0
$$

真实 measured pixel 完全可能出现零值或接近零值。

应该直接使用：

```python
M
```

计算：

$$
E_{\mathrm{measured}}
=
\frac{
\|M\odot(\hat X-X)\|
}{
|M|
}
$$

以及：

$$
E_{\mathrm{missing}}
=
\frac{
\|(1-M)\odot(\hat X-X)\|
}{
|1-M|
}
$$

这个属于必须修的 evaluation bug。

---

# 十七、SSIM 实现也应该在下一阶段一起冻结

目前：

```python
_gaussian_window()
```

实际上是：

```text
1 × 1 × 1 × K
```

再进入 `conv2d`。

因此它不是标准意义上的二维 Gaussian window。

应该改为真正：

$$
G(x,y)
=
G_x(x)G_y(y)
$$

即：

```text
1 × 1 × K × K
```

然后重新跑所有最终 quantitative experiments。

否则：

> 老 SSIM 与新 SSIM

无法直接比较。

因此：

> **PSNR / SSIM implementation freeze**

应该在重新跑 Main Experiment 之前完成。

---

# 十八、Protocol B 和 Protocol A 也要正式分离

目前 `acquisition.py` 的主实现本质上还是：

$$
X
\rightarrow
H*X
\rightarrow
M
\rightarrow
noise
\rightarrow
Y
$$

即：

> synthetic sparse measurement

而 Protocol A：

$$
X
\rightarrow
H*X
\rightarrow
S_K
\rightarrow
noise
\rightarrow
Y_{low}
\rightarrow
R
\rightarrow
Y_{grid}
$$

目前还没有真正实现。

因此下一阶段应该：

### Protocol B

作为：

> controlled simulation / mechanism study

用于：

- PSF
- λ
- sampling density
- photon noise
- mask study

### Protocol A

作为：

> acquisition-realistic validation

用于：

- actual scanner geometry
- real low-resolution sampling
- actual exposure
- actual timing
- actual noise

最终论文的逻辑应该是：

> simulation establishes mechanism → acquisition-realistic experiments establish applicability.

---

# 十九、重新定义 Main Experiment Matrix

我建议不要立即进入复杂网络。

先做一个 **Corrected Baseline Matrix**。

## Experiment 1 - Y-sparse 2D

固定：

$$
X:\text{dense}
$$

扫描：

$$
K_Y=\{2,4,8,16\}
$$

每个 K：

```text
Direct network
Bilinear prior
Lateral Wiener prior
Bilinear-only
Wiener-only
```

核心回答：

1. Y sparsity 本身有多难？
2. Bilinear prior 有多少价值？
3. Wiener 相比 Bilinear 本身多少价值？
4. Wiener 作为 network prior 是否额外有价值？
5. 新网络是否真的优于 direct network？

这一步必须重新做。

---

# 二十、Experiment 2 - YZ-sparse 3D

这是新的主实验。

建议：

$$
K_Y,K_Z
\in
\{2,4,8\}
$$

至少包括：

```text
(2,2)
(2,4)
(4,2)
(4,4)
(4,8)
(8,4)
(8,8)
```

这里不要只比较：

$$
K_Y=K_Z
$$

因为真正有价值的是知道：

> Y 和 Z 哪个方向更难恢复？

---

# 二十一、Experiment 3 - fixed acquisition budget

例如固定：

$$
K_YK_Z=B
$$

然后：

```text
(2,8)
(4,4)
(8,2)
```

比较：

$$
Q(\hat X)
$$

以及：

$$
T_{\mathrm{acq}}
$$

这一个实验甚至应该进入 main figure，而不是 supplementary。

因为它直接对应：

> **How should the sparse acquisition budget be allocated across Y and Z?**

---

# 二十二、Experiment 4 - XYZ extreme sparsity

例如：

$$
K_X>1,\quad K_Y>1,\quad K_Z>1
$$

这里重点不再是：

> actual acceleration = $K_XK_YK_Z$

而是区分：

### nominal sampling reduction

和

### actual acquisition acceleration

这两个指标。

尤其需要明确：

$$
\mathrm{Compression}
\neq
\mathrm{Speedup}
$$

如果 X sparsity 没有减少实际扫描时间，那么它只是减少 acquired samples，而不是直接增加 hardware speed。

---

# 二十三、Experiment 5 - XYT temporal reconstruction

重新建立一个独立的数据定义：

$$
X,Y:\text{spatial}
$$

$$
T:\text{temporal}
$$

例如：

$$
K_T=2,4,8
$$

然后比较：

```text
frame interpolation
temporal network
unified model
axis-specific temporal adapter
```

关键指标除了：

- PSNR
- SSIM
- MAE

还应该考虑：

- temporal consistency
- motion consistency
- temporal flicker
- biological event preservation

因为 XYT 的错误不是简单的空间 MAE。

已有时空稀疏成像研究也表明，时间维度的稀疏重建通常需要利用前后帧的信息传播，而不能简单等同于 Z-axis interpolation；例如 SparseFLIM 就直接以 XYT 数据为基础构建双向时序信息传播。([PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11494201/?utm_source=chatgpt.com))

---

# 二十四、Unified network 的正式结构，我建议这样定

最终模型可以写成：

$$
Y,M,C_s,U_p,X_0
$$

↓

### Shared Spatial Encoder

$$
F_s=E_s(Y)
$$

↓

### Sampling Encoder

$$
F_m=E_m([M,C_s,U_p])
$$

↓

### Prior Encoder

$$
F_p=E_p(X_0)
$$

↓

### Sampling-conditioned fusion

$$
G=
\sigma
(
\mathrm{MLP}
[
F_m,F_p
]
)
$$

$$
F=
F_s+G\odot F_p
$$

↓

### Axis-specific interaction

$$
F_Z=A_Z(F)
$$

or

$$
F_T=A_T(F)
$$

↓

### Shared Decoder

$$
\hat X=D(F)
$$

---

# 二十五、这样就能统一四种 acquisition，但不强迫它们完全一样

| Regime | Shared spatial | Prior | Axis module |
|---|---:|---:|---:|
| Y | ✓ | XY prior | none / lightweight |
| YZ | ✓ | 3D physics prior | $A_Z$ |
| XYZ | ✓ | 3D physics prior | $A_Z$ |
| XYT | ✓ | temporal prior | $A_T$ |

这是我认为目前最合理的折中：

> **one shared reconstruction framework, but not one blindly shared physical model.**

---

# 二十六、Separate backbone 还是 unified network？

现在不建议二选一。

应该设计成三组实验：

### Specialist-Y

专门训练 Y-sparse。

### Specialist-YZ

专门训练 YZ-sparse。

### Unified

一个模型同时训练 Y + YZ。

然后比较：

$$
Q_{\mathrm{Unified}}
$$

vs

$$
Q_{\mathrm{Specialist}}
$$

但必须控制：

- parameter count
- training samples
- optimizer steps
- total training budget

否则 unified model 训练量天然更大，比较不公平。

这样你就可以科学地回答：

> **Can one sampling-conditioned model generalize across acquisition geometries without sacrificing specialist performance?**

这个问题本身就是一个方法学贡献。

---

# 二十七、训练时必须避免“不同 regime 相互伤害”

这是 unified model 最大的潜在问题。

Y-sparse 和 YZ-sparse 的信息难度完全不同。

如果：

```text
random Y batch
random YZ batch
random XYZ batch
```

直接混合，很可能出现：

> easy regime dominates optimization

或者：

> extreme regime destabilizes learning.

因此应该使用：

### regime-balanced sampler

例如每个 batch 固定：

```text
1/3 Y
1/3 YZ
1/3 XYZ
```

然后加入：

$$
e_{\mathrm{axis}}
$$

或：

$$
e_{\mathrm{sampling}}
$$

作为 condition embedding。

虽然 mask 已经包含部分 geometry 信息，但显式 conditioning 可以让网络更容易区分：

```text
Y sparse
YZ sparse
XYZ sparse
XYT
```

---

# 二十八、现在的“Residual”应该重新处理

当前：

> PSRAS = Physics-Sampling-Residual Aware Synthesis

但是 B5 residual 已经实验上被否定。

那么继续把 Residual 放在项目名字里会造成论文方法和实验结果的不一致。

我建议最终方法名重新考虑成：

> **Physics-Sampling-Prior Aware Reconstruction**

或者类似：

> **Physics- and Sampling-aware Sparse-to-Dense Reconstruction**

不要为了保留 PSRAS 缩写而强行保留已经放弃的技术点。

论文中最忌讳：

> method name 暗示一个核心 module，但实验里已经没有这个 module。

---

# 二十九、当前 EXP-020 的 architecture matrix 也要重新定位

旧结果：

```text
Video Swin
RVRT
SwinUNETR
Ours
```

证明了：

> z-axis Wiener/prior 在多个 architecture 上有帮助。

这是非常有价值的机制证据。

但是下一阶段不要继续无限增加 backbone。

它应该变成：

> **Architecture-independence validation**

而不是：

> backbone zoo。

也就是说：

### Main model

只保留一个主 backbone。

### External models

选 2-3 个代表：

- CNN/UNet
- video transformer
- 3D transformer

证明：

$$
\text{physics prior gain}
$$

不是因为某一个特殊 backbone 才出现。

做到这一点后就足够了。

---

# 三十、重新定义评价体系

不要只报告：

> PSNR / SSIM

最终应该分三层。

## Reconstruction quality

$$
PSNR,\ SSIM,\ MS\text{-}SSIM,\ MAE
$$

## Missing-region recovery

$$
E_{\mathrm{missing}}
$$

这是最重要的，因为 measured region 很容易。

## Acquisition efficiency

$$
A_T=
\frac{T_{\mathrm{dense}}}{T_{\mathrm{sparse}}}
$$

以及：

$$
A_D=
\frac{D_{\mathrm{dense}}}{D_{\mathrm{sparse}}}
$$

其中 $D$ 是实际激光曝光/剂量 proxy。

最终画：

$$
Q
\quad\text{vs.}\quad
A_T
$$

而不是只画：

$$
Q
\quad\text{vs.}\quad
K.
$$

这是从 algorithm paper 向 biomedical imaging paper 转型的关键。

3PM 本身确实存在成像速度、空间分辨率、光子数和曝光之间的现实权衡；近期 sparse X-Y-Z 3PFM 工作也明确把 acceleration 与低 phototoxicity 作为目标。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

---

# 三十一、建议形成新的项目阶段划分

## Phase 0 - Protocol cleanup

必须完成：

```text
[ ] 修正 split
[ ] 修正 mask evaluation
[ ] 修正 SSIM
[ ] 修正 naming
[ ] freeze normalization
[ ] freeze PSF units
[ ] freeze random seeds
[ ] legacy results archive
```

这是 P0。

---

## Phase 1 - Corrected 2D-Y baseline

完成：

```text
Y K=2/4/8/16
Direct
Bilinear
Wiener
Bilinear prior network
Wiener prior network
```

这一阶段最终回答：

> **横向 Wiener 到底有没有作为 prior 帮助新网络？**

不要提前假设答案。

---

## Phase 2 - Genuine YZ physics prior

实现真正：

$$
P_{YZ}(Y,M,H)
$$

并验证：

```text
Bilinear
Wiener
iterative physics prior
```

这里是真正的新 physics branch。

---

## Phase 3 - YZ main experiment

完成：

```text
Ky × Kz matrix
```

同时分析：

$$
K_Y
$$

与

$$
K_Z
$$

的独立影响。

---

## Phase 4 - Budget-aware sampling allocation

固定：

$$
K_YK_Z
$$

比较不同：

$$
(K_Y,K_Z)
$$

这是建议重点发展成论文核心 innovation 的阶段。

---

## Phase 5 - Unified model

比较：

```text
Specialist Y
Specialist YZ
Unified Y+YZ
```

控制模型容量和训练预算。

---

## Phase 6 - XYZ

作为 extreme spatial sparsity。

重点：

- robustness
- sampling ratio
- compression
- prior recovery

同时明确与 2026 DeepR-SXYZ 类工作的差异。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

---

## Phase 7 - XYT

再加入：

$$
A_T
$$

重点验证：

> temporal generalization

而不是把 XYT 和 XYZ 混成一种“3D reconstruction”。

---

## Phase 8 - Protocol A / real acquisition

最终闭环：

```text
real sparse acquisition
        ↓
physics model
        ↓
reconstruction
        ↓
actual acquisition time
        ↓
dose / phototoxicity proxy
```

这一步决定项目最终能否从：

> simulation-based reconstruction

提升到：

> **real fast-imaging methodology**

---

# 三十二、最终项目的数据流建议彻底改成下面这个形式

## Y-sparse

```text
Dense XY
   │
   ↓
Sparse Y acquisition
   │
   ├── M
   ├── Cs
   ├── Physics prior XY
   ↓
Shared Spatial Encoder
   ↓
Sampling-conditioned Fusion
   ↓
Decoder
   ↓
Dense XY
```

## YZ-sparse

```text
Dense XYZ
   │
   ↓
Sparse YZ acquisition
   │
   ├── M
   ├── Cs
   ├── 3D Physics Prior
   ├── Up / recoverability
   ↓
Shared Spatial Encoder
   ↓
Z-axis Adapter
   ↓
Decoder
   ↓
Dense XYZ
```

## XYZ

```text
Dense XYZ
   │
   ↓
Sparse XYZ
   │
   ├── M
   ├── Cs
   ├── 3D Physics Prior
   ↓
Shared Spatial Encoder
   ↓
Z-axis / spatial Adapter
   ↓
Decoder
```

## XYT

```text
Dense XYT
   │
   ↓
Sparse temporal acquisition
   │
   ├── M
   ├── Cs
   ├── Temporal prior
   ↓
Shared Spatial Encoder
   ↓
Temporal Adapter
   ↓
Decoder
   ↓
Dense XYT
```

这四个数据流看起来属于同一个 framework，但物理含义完全不同。

---

# 三十三、下一版论文最值得形成的核心方法概念

综合目前所有实验，我认为最值得发展的不是：

> Swin + Wiener

而是：

# **Sampling- and Axis-Conditioned Physics-Guided Sparse-to-Dense Reconstruction**

核心思想是：

$$
\boxed{
\hat X
=
f_\theta
\left(
Y,\,
M,\,
C_s,\,
U_p,\,
X_0,\,
d
\right)
}
$$

其中 $d$ 是 missing/acquisition dimension：

$$
d\in\{Y,YZ,XYZ,XYT\}
$$

同时：

$$
X_0=P_d(Y,M,H)
$$

而不是所有维度都使用相同 prior。

进一步，在固定 acquisition budget 下：

$$
(K_Y^*,K_Z^*)
=
\arg\max_{K_Y,K_Z}
Q\left(
f_\theta(P_{YZ}(Y))
\right)
$$

这就把：

> sparse sampling  
> physics prior  
> reconstruction  
> acquisition design

真正连起来了。

---

# 三十四、我建议把当前项目“修改优先级”定成这样

| 优先级 | 修改 | 必要性 |
|---|---|---|
| **P0** | F0 val/test split 重构 | 必须 |
| **P0** | `reval.py` 改为显式 M | 必须 |
| **P0** | SSIM 改为标准 2D Gaussian | 必须 |
| **P0** | 重新定义当前 z-B3，不再称 full 3D Wiener | 必须 |
| **P0** | 冻结 normalization / PSF / evaluation protocol | 必须 |
| **P1** | 重做 Y-sparse Wiener vs Bilinear vs Direct | 必须 |
| **P1** | 实现真正 YZ physics prior | 必须 |
| **P1** | 建立 YZ 主实验 | 必须 |
| **P1** | 加 fixed-budget $K_Y/K_Z$ allocation | 强烈建议 |
| **P2** | unified shared backbone | 建议 |
| **P2** | axis-specific Z/T adapters | 强烈建议 |
| **P2** | unified vs specialist | 必须作为验证 |
| **P2** | XYZ | 建议 |
| **P3** | XYT | 作为 extension |
| **P3** | Protocol A | 最终应用验证 |
| **P3** | actual acquisition time/dose | 最终闭环 |

---

# 三十五、最关键的几个结论先固定下来

经过这次修正，我建议以后整个项目都遵循下面五条原则：

**第一，不能再把“Wiener 比 Bilinear 好不好”与“Wiener 作为 network prior 有没有用”混为一个问题。**

**第二，当前 z-mainline 不应再声称用了真正的 3D Wiener；真正的 3D physics prior 要在 YZ/XYZ 重新实现。**

**第三，Y、YZ、XYZ 是 acquisition regimes；2D/2.5D/3D 更适合描述网络上下文，不再作为项目主分类。**

**第四，XYZ 与 XYT 共享 reconstruction machinery 是可以的，但 physics prior 和 axis interaction 必须分开；主论文优先讲 XYZ/YZ acquisition acceleration，XYT 作为 temporal generalization。**

**第五，当前 F0 的 val/test 虽然可以作为 temporal hold-out，但不能继续作为严格意义上的 independent stack-level test；最终实验必须增加真正独立的 stack/session/sample-level test。**

另外，考虑到 2026 年已经出现专门的 sparse X-Y-Z 3PFM reconstruction 工作，后续真正值得押注的创新点应从“又一个 XYZ reconstruction network”转向 **“physics-aware prior + sampling-axis conditioning + fixed-budget acquisition allocation + real acquisition efficiency”**。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com))

这也意味着，**下一阶段不应该先写新网络代码**。最合理的执行顺序是先把 `PSRAS` 的 protocol、split、metric、Wiener 定义修干净，然后做 **Y corrected baseline → YZ corrected baseline → fixed-budget allocation → unified model**。这样后面的每一层创新都有可靠的实验地基。

---

**Sources:**

- [Hundred-Nanosecond Equivalent Pixel Dwell Time for Deep-Tissue 3D Three-Photon Fluorescence Microscopy via Sparse X-Y-Z Reconstruction - PubMed](https://pubmed.ncbi.nlm.nih.gov/41546413/?utm_source=chatgpt.com)
- [Overcoming photon and spatiotemporal sparsity in fluorescence lifetime imaging with SparseFLIM - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11494201/?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
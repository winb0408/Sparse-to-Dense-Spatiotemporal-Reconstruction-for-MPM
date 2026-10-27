# 审阅项目与创新建议

**User:** Anonymous  
**Created:** 9/8/2026 22:23:51  
**Updated:** 9/9/2026 20:12:29  
**Exported:** 9/9/2026 20:41:42  
**Link:** [https://chatgpt.com/c/6aa01a75-f754-83ec-87ea-964861042edf](https://chatgpt.com/c/6aa01a75-f754-83ec-87ea-964861042edf)  

## Response:
2026/9/8 22:23:53

我会先从仓库的 README、目录结构、核心训练/重建代码、已有实验结果和报告入手，分别判断“作者原始意图”和“代码实际实现”是否一致，再据此评估不足与可发表的创新空间。

> **审查训练与评估**
>
> 我在检查 `dev` 分支实现时发现若干关键问题：DINO特征提取被停用梯度，且缺少位置编码与令牌处理；评估归一化使用目标图像，可能造成数据泄漏。
>
> **审查训练与评估**
>
> 
>
> **检查 normalise_pair 调用及代码问题**
>
> 
>
> 思考了 53s

我已经按你给出的 `dev` 分支对这个项目做了一轮“研究型代码审阅”，重点核对了：

- 项目 README 与实验协议
- 数据集/任务维度定义
- 2D / 2.5D / 3D 模型组织
- sparse sampling、PSF、Wiener warm-start、mask 的实现
- loss 与训练流程
- prior-aware fusion 当前实现
- CIDC25、MDC25、pollen 已有结果
- 当前 reorganized report 的研究叙事

先给你一个总体判断：

> **这个项目不是“没有东西”，恰恰相反，已经有一个相当完整的研究雏形；真正的问题是当前科学问题还没有被压缩成一个足够强、足够干净的主线。**
>
> 目前最有价值的方向并不是继续堆 2D/2.5D/3D 网络，而是把问题重新定义为 **“physics-aware sparse reconstruction under anisotropic 3D sampling”**，并让 **sampling mask + physical inverse initialization + learned residual reconstruction** 成为统一方法框架。

而且我发现了几处**会直接影响论文可信度的实现/评估问题**，其中有一处尤其重要：目前报告中把 DINOv2 写成训练损失，但从代码来看，它实际上很可能**没有向重建网络提供梯度**。这类问题比“模型不够创新”更需要优先处理。

---

# 1. 我理解到的项目原始目标

目前 README 的定义是一个 **3PM（Three-Photon Microscopy）Sparse-to-Dense Reconstruction Pipeline**，目标是在三光子显微成像中减少实际采样量，再利用神经网络把稀疏观测恢复成 dense image / volume。项目覆盖：

- 2D sparse reconstruction
- 2.5D center-slice reconstruction
- 3D volumetric reconstruction

模型主要使用：

- Swin-UNet 2D
- Video Swin-UNet 3D
- MSE + MS-SSIM + DINOv2 perceptual loss
- PSF simulation
- Wiener initialization
- sampling mask

README 当前已经明确把问题写成：

$$
\hat X=f_\theta(Y,M)\approx X
$$

其中：

- $Y$：稀疏观测
- $M$：sampling mask
- $X$：dense ground truth
- $\hat X$：reconstructed image/volume

这一步其实是正确的，因为它把问题从单纯的 image-to-image translation 转成了一个 **inverse reconstruction problem**。

---

# 2. 项目现在实际上有三层研究问题

把仓库中已有内容抽象以后，我认为项目实际上包含三个层次。

## 第一层：Sparse sampling reconstruction

给定稀疏采样：

$$
Y=M\odot X
$$

或者更真实一点：

$$
Y=M\odot \mathcal H(X)+n
$$

恢复 dense $X$。

这里研究的是：

> **采样越少，网络还能恢复多少信息？**

这是最基础的问题。

---

## 第二层：不同空间维度的信息利用

你目前实际上研究了三个 inference regime：

### 2D

$$
Y_t \rightarrow \hat X_t
$$

只使用当前切面。

### 2.5D

$$
(Y_{t-k},...,Y_t,...,Y_{t+k})
\rightarrow \hat X_t
$$

利用相邻 Z-slices 重建中心 slice。

### 3D

$$
Y_{t-k:t+k}
\rightarrow
\hat X_{t-k:t+k}
$$

直接进行 volumetric reconstruction。

你的数据协议已经开始正确地区分这三者，并明确指出：

- 2D 是单切面；
- 2.5D 输出是 center slice；
- 3D 才是 volume；
- 不能把三者混成一个总 benchmark。

这是目前项目结构里我认为**非常正确的一次重整**。

---

# 3. 第三层其实才是目前最有潜力的科学问题

你现在新加入的：

> **Mask + Wiener + prior-aware fusion**

其实比单纯的 Swin-UNet / Video Swin 更有研究价值。

你目前的设计是：

$$
Y,M
\rightarrow
W(Y)
\rightarrow
\text{prior-aware fusion}
\rightarrow
\hat X
$$

其中：

- Wiener：提供物理逆问题的 initial estimate
- mask：提供“哪些位置是真实采样、哪些位置是推断出来的”这一 sampling confidence
- gated fusion：让网络显式利用这些 prior

项目文档对这个逻辑已经描述得比较清楚。

这其实已经接近一个很不错的研究故事：

> **Sparse microscopy reconstruction should not treat all input pixels equally. The reconstruction network should distinguish measured evidence from inferred content and should initialize reconstruction from a physics-based inverse solution.**

这个方向比“用了 Swin-UNet，所以效果更好”强很多。

---

# 4. 目前已经得到什么结果？

这里开始出现一个非常有意思的现象。

## 4.1 Z-sparse：2.5D 非常强

CIDC25/F0 的已有结果显示：

| 方法 | Sparse | PSNR | SSIM |
|---|---|---:|---:|
| 2.5D | Z, K=2 | **51.72** | **0.9942** |
| 2.5D | Z, K=4 | **50.08** | **0.9912** |
| 3D | Z, K=2 | 44.43 | 0.9763 |
| 2D | Z, K=2 | 42.59 | 0.9621 |

这些结果说明：

> **对于纯 Z-axis sparse，邻层信息非常有价值，而 2.5D 可能已经足以获得大部分 3D context benefit。**

这是一个非常值得深入研究的结果，而不是一个简单的 baseline。

---

# 5. XY sparse：Wiener 的价值开始出现

K=4、XY sparse：

普通 3D：

$$
21.19\,dB,\quad SSIM=0.5251
$$

加入 Wiener：

$$
38.50\,dB,\quad SSIM=0.9169
$$

这是一个非常大的改善。

从研究角度看，这个现象非常重要：

> **当 sparse degradation 足够严重时，仅靠 learned prior 容易崩，而 physics-based initialization 可以显著改善 optimization/reconstruction stability。**

这个结果本身就是一个论文故事的种子。

---

# 6. 但现阶段“最明显的问题”不是模型，而是 benchmark

这是我认为项目现在最大的结构性问题。

你的旧实验确实存在：

> 2D / 2.5D / 3D  
> × CIDC25 / MDC25 / pollen  
> × x / y / xy / z / xyz  
> × K=2 / K=4  
> × Wiener / mask / no mask

组合爆炸。

最终非常容易产生一个问题：

> **哪个结果到底证明了什么？**

你现在的 `RESULT_TAXONOMY.md` 已经开始解决这个问题，把：

- CIDC25 → 主 benchmark
- MDC25 → stress test
- pollen → external transfer

进行了区分。

我非常赞同这个方向。

但我进一步建议：

### 最终论文甚至不应该把“2D/2.5D/3D谁最好”作为主问题。

因为这个问题天然不公平：

- 输入信息量不同；
- 输出目标不同；
- 参数量不同；
- receptive field 不同；
- computational cost 不同。

更好的问题是：

> **在相同采样预算下，如何利用不同维度的空间冗余与物理先验进行稳定重建？**

这样 2D/2.5D/3D 都变成实验条件，而不是贡献本身。

---

# 7. 我发现的一个非常严重的代码问题：DINOv2 loss 当前很可能没有真正参与训练

这个问题需要特别重视。

你的 `DINOv2PerceptualLoss` 中：

```python
@torch.no_grad()
def _extract(...)
```

随后：

```python
pred_feats   = self._extract(pred)
target_feats = self._extract(target)
loss = sum(F.mse_loss(pf, tf) ...)
```

也就是说，**prediction 和 target 的 DINO features 都是在 `torch.no_grad()` 下提取的**。因此 `loss_dino` 本身不会建立从 `pred` 回到 reconstruction network 的梯度链。

换句话说：

> 你现在可以记录一个 DINO loss 数值，但它不等于 DINO loss 正在优化 reconstruction network。

而训练脚本确实直接把 `criterion` 的总 loss 拿来 backward。 

### 正确做法应该是：

冻结 **DINO 参数**，但不能冻结 **输入 feature extraction 的 autograd**。

也就是说：

```python
for p in dino.parameters():
    p.requires_grad_(False)
```

可以保留；

但是：

```python
_extract()
```

不能整体 `@torch.no_grad()`。

因为你需要：

$$
\frac{\partial L_{DINO}}{\partial \hat X}
$$

而不是：

$$
\frac{\partial L_{DINO}}{\partial \theta_{DINO}}
$$

这两件事情完全不同。

这是我建议你**第一优先级修复**的问题。

---

# 8. 更深一层：你现在实现的 DINO feature 也不完全是标准 DINOv2 feature

你的代码直接：

```python
x = self.patch_embed(x)

for i, blk in enumerate(self.blocks):
    x = blk(x)
```

但标准 ViT / DINOv2 的 token preparation 并不是简单：

$$
patch\ embedding \rightarrow blocks
$$

还涉及 positional embedding、special tokens 等。

因此当前这段代码即便打开梯度，严格来说也不应该直接描述成：

> standard DINOv2 intermediate representation

更准确的做法应该是调用 DINOv2 自身提供的 token extraction API，或者完整复现其 forward preparation。

所以这其实是第二个需要修的问题：

### 当前 DINO 部分：

**概念上先进 → 实现上不严谨。**

---

# 9. 第二个非常重要的问题：Wiener 所谓的“3D physics prior”实际上还是 2D Wiener

这个问题更加值得注意。

你在 `wiener_init_3d()` 中：

```python
for t in range(T):
    out[t] = wiener_init_2d(...)
```

也就是说：

> 所谓 3D Wiener initialization，目前实际上是逐 frame 的 2D Wiener deconvolution。

而且：

```python
psf_sigma_axial
```

虽然作为参数存在，但 `wiener_init_3d()` 并没有真正建立 3D OTF。

这意味着当前：

$$
W_{3D}(Y)
$$

实际上更接近：

$$
W_{2D}(Y_1),W_{2D}(Y_2),...,W_{2D}(Y_T)
$$

而不是：

$$
\mathcal F^{-1}
\left[
\frac{H^*_{3D}}
{|H_{3D}|^2+\lambda}
\mathcal F(Y)
\right]
$$

---

# 10. 这个反而给你提供了一个非常强的创新机会

你现在有机会把项目真正升级成：

## Anisotropic Physics-Guided Sparse Reconstruction

三光子显微镜最大的物理特点之一就是：

$$
\sigma_z \gg \sigma_{xy}
$$

你的项目已经在 PSF simulation 里明确用了：

- lateral sigma
- axial sigma

而且 3D blur 确实实现成 anisotropic 3D Gaussian。

但是 Wiener 反演还没有跟上。

因此可以构造：

$$
H(f_x,f_y,f_z)
$$

真正的 anisotropic 3D OTF，然后：

$$
X_{W}
=
\mathcal F^{-1}
\left(
\frac{H^{*}}
{|H|^2+\lambda(f)}
\mathcal F(Y)
\right)
$$

这会比现在的“逐帧 Wiener”明显更有科学意义。

---

# 11. 当前 prior fusion 也还比较弱

你现在的 gated fusion 基本是：

$$
G=\sigma(Conv(M))
$$

然后：

$$
X'=X\odot
\left[1+\alpha(G-0.5)\right]
$$

其中 $\alpha$ 初始化为 0。代码实现确实如此。

优点：

- 简洁
- 稳定
- 参数少
- 从 identity 初始化

但是从论文角度看：

> 目前还更像一个“合理 engineering module”，而不是一个很强的方法创新。

---

# 12. 我更建议把 mask 从 binary mask 升级为“uncertainty / confidence field”

现在：

$$
M(x)=
\begin{cases}
1,& measured\\
0,& missing
\end{cases}
$$

其实还不够。

因为真实情况中：

> “测到了” ≠ “这个 pixel 一定可靠”。

可以构造：

$$
C(x)
=
M(x)
\cdot
q(Y(x))
$$

或者：

$$
C(x)=f(M,Y,\mathrm{PSF},SNR)
$$

形成一个 **sampling confidence map**。

然后网络做：

$$
F_{prior}
=
\phi(X_W,C)
$$

再：

$$
F_{rec}
=
F_{image}+G(C)\odot F_{prior}
$$

这样你的 mask 就不再只是：

> “哪里有数据”

而变成：

> “哪里值得相信”

这个概念要强很多。

---

# 13. 更进一步：应该让网络预测 residual，而不是从零生成 X

我非常建议你把最终框架改成：

$$
X_W = W(Y)
$$

然后网络学习：

$$
R_\theta
=
f_\theta(X_W,Y,M)
$$

最后：

$$
\hat X=X_W+R_\theta
$$

也就是：

# Physics initialization + learned residual correction

这比：

$$
\hat X=f_\theta(Y,M)
$$

有几个明显优势：

### 第一

网络任务难度下降。

它不再需要从 sparse image 完全“想出” dense image。

### 第二

物理先验真正成为 reconstruction backbone。

### 第三

可以直接解释 residual：

$$
R=X-X_W
$$

这个 residual 本身就是：

> **physics model cannot explain 的 component**

非常适合论文分析。

---

# 14. 你目前 PSF loss 也存在“名义 3D、实际 2D”的问题

`PSFConsistencyLoss` 的 `_blur_2d()` 实际上是 2D Gaussian blur。

因此：

```python
sigma_axial
```

没有真正用于 reconstruction consistency。

代码注释写的是：

> PSF * pred should approximate noisy input

但现在具体实现其实只有：

$$
PSF_{xy} * \hat X
$$

而不是：

$$
PSF_{xyz} * \hat X
$$

这会让“physics-aware”这条故事出现理论和实现之间的不一致。

---

# 15. 当前最值得警惕的是训练/评估规范问题

你现在代码中的 normalization：

```python
normalise_pair(low, high)
```

对于 `minmax`：

$$
g_{min}
=
\min(low,high)
$$

$$
g_{max}
=
\max(low,high)
$$

然后同时归一化 low 和 high。

这个做训练 pair normalization 本身可以理解，但：

> **如果 validation/test 阶段也通过 low+GT 一起确定 normalization 参数，就会把 GT 信息引入输入预处理。**

那就是 evaluation leakage。

我建议最终必须严格变成：

### Training

可以依据训练样本的 acquisition statistics。

### Validation/Test

normalization parameters **只能来自输入 / training statistics**，绝对不能看 GT。

否则 PSNR / SSIM 会受到污染。

---

# 16. 目前一些结果本身就已经暴露出 benchmark 可能存在问题

最明显的是：

```text
baseline_input_z_K4
PSNR = 74.14 dB
SSIM = 1.000
```

而模型：

```text
2.5D z K4
PSNR = 50.08 dB
SSIM = 0.9912
```

这当然可以是因为 baseline input 的定义存在特殊情况，但从实验逻辑上：

> **一个 sparse input 在 z-K4 下得到 74 dB / 1.0 SSIM，非常值得重新检查。**

特别需要确认：

1. baseline input 到底是不是 sparse frame；
2. 是否实际上读取了 GT frame；
3. Z mask 和 evaluation target 是否对齐；
4. frame indexing 有没有 offset；
5. GT normalization 是否进入 input；
6. evaluation 是否只在 sampled frames 上算；
7. 为什么 sparse baseline 能得到近似完美结果。

这不是普通的小误差，我会把它标成：

### **P0 - 必须人工核查的结果**

---

# 17. MDC25 的结果已经告诉你一个非常重要的事实

MDC25 当前结果非常差：

例如：

- 2.5D z：15.57 dB / 0.1872
- 3D xy：14.67 dB / 0.1291
- 2D xy：7.13 dB / 0.0557

这其实不一定是坏消息。

反而说明：

> **模型存在明显 domain shift。**

也就是说你现在的模型可能在：

$$
P_{train}(X,Y)
$$

下有效，

但到了：

$$
P_{MDC25}(X,Y)
$$

就崩了。

如果进一步证明：

- synthetic sparse → strong
- same-system sparse → strong
- external acquisition → degraded

那么你可以把论文进一步升级到：

> **physics-aware reconstruction improves robustness under acquisition-domain variation**

这比单纯追求某个 PSNR 高几个 dB 更有价值。

---

# 18. pollen 的结果也说明：必须区分“看起来好”和“真正泛化”

pollen 中不同方法差异非常大。

比如某些 setting 下：

- 3D + Wiener 可以达到约 25-26 dB；
- 另一些模型只有 5-15 dB。

但现在 pollen 是通过 mean projection 得到 2D external sample，因此你目前已经正确地把它降级为 transfer probe。

这个决策应该保持。

---

# 19. 我认为现在项目真正的“研究核心”应该重新定义

我建议不要再把论文题目理解为：

> Sparse-to-Dense Spatiotemporal Reconstruction for 3PM

这个太宽，而且容易落入“又一个 Transformer reconstruction”。

我更建议把科学问题压缩成：

# Physics-Guided Sparse Reconstruction under Anisotropic Sampling

核心 formulation：

$$
Y
=
M\odot
\left(
H*X+n
\right)
$$

其中：

- $H$：anisotropic optical PSF
- $M$：sampling pattern
- $n$：photon/readout noise

然后：

$$
X_0
=
\mathcal W(Y;H)
$$

即 physics-based inverse initialization。

再：

$$
C
=
g(Y,M,H)
$$

得到 sampling confidence。

最后：

$$
R_\theta
=
f_\theta(X_0,Y,C)
$$

$$
\boxed{
\hat X=X_0+R_\theta
}
$$

这就是我认为最值得你继续发展的主干。

---

# 20. 由此可以形成一个非常漂亮的三模块架构

## Module 1 - Physics Initialization

### Anisotropic Wiener Reconstruction

利用：

$$
H_{3D}
$$

构造：

$$
X_W
$$

---

## Module 2 - Sampling-aware Prior Encoding

输入：

$$
Y,\;M,\;X_W
$$

得到：

$$
F_Y,\;F_M,\;F_W
$$

其中 mask 不再只是普通 channel。

可以使用：

- gated fusion
- FiLM
- confidence attention
- mask-aware normalization

---

## Module 3 - Residual Reconstruction

学习：

$$
R_\theta
=
F(F_Y,F_W,F_M)
$$

最终：

$$
\hat X=X_W+R_\theta
$$

这样网络本身就天然具备：

> **physics coarse solution → uncertainty-aware correction**

的逻辑。

---

# 21. 我甚至认为你可以进一步把 2D/2.5D/3D 统一起来

这是我比较推荐的创新方向。

当前：

```text
2D → Swin UNet
2.5D → Swin UNet
3D → Video Swin UNet
```

看起来像三个模型。

可以统一改成：

$$
\hat X =
X_W+
\mathcal R_\theta(Y,M,X_W,D)
$$

其中：

$$
D\in\{2D,2.5D,3D\}
$$

相当于把：

> dimension

看作 reconstruction regime，而不是完全不同的 algorithm。

最终可以研究：

### Dimension-aware reconstruction

让网络根据：

- axial sampling density
- lateral sampling density
- PSF anisotropy
- local confidence

动态决定需要使用多少 spatial / axial context。

这是比单纯比较 2D/2.5D/3D 更值得做的。

---

# 22. 最值得补的实验，不是继续跑很多模型

我建议接下来优先做下面这组实验。

## Experiment A：Physics prior ablation

必须形成：

| Model | Bilinear | Wiener |
|---|---:|---:|
| no mask | A0 | A2 |
| mask | A1 | A3 |
| gated mask | - | A4 |

这个实验应该成为论文核心消融。

你项目已经定义了 A0-A4。

但是一定要保证：

$$
A3\rightarrow A4
$$

真正证明的是：

> **structured fusion**

而不是：

> 更多输入 channel。

---

# 23. Experiment B：Physics correctness

这一组其实比继续换 Transformer 更重要：

### 2D Wiener

vs

### slice-wise Wiener

vs

### true 3D anisotropic Wiener

然后：

$$
PSF_{xy}
$$

vs

$$
PSF_{xyz}
$$

如果结果支持真正 3D Wiener：

这就是非常好的方法贡献。

---

# 24. Experiment C：sampling severity

不要只做 K=2 / K=4。

建议至少：

$$
K=2,4,6,8
$$

或者按照实际 sampling ratio：

$$
r=50\%,25\%,12.5\%,6.25\%
$$

绘制：

$$
PSNR \ vs.\ sampling\ ratio
$$

这个图会比一个表格强得多。

最终论文可以回答：

> **我们的模型到底能把采样降到什么程度？**

这才真正对应 3PM acceleration 的科学价值。

---

# 25. Experiment D：realistic acquisition noise

现在已经支持：

- Poisson
- Gaussian readout noise
- PSF

这是非常好的基础。

下一步应该系统研究：

$$
\lambda_{photon}
$$

例如：

$$
5,\ 10,\ 20,\ 50
$$

同时：

$$
\sigma_{read}
$$

变化。

最终形成：

> reconstruction performance under photon budget

这会比纯 synthetic sparsity 更符合三光子成像实际。

---

# 26. Experiment E：结构保持，而不是只看 PSNR

你当前已经意识到这一点，并建立了：

- foreground metrics
- background metrics
- local profile
- FWHM

这个方向是对的。

但我建议进一步引入更具显微成像意义的指标：

### Structure preservation

例如：

$$
NCC_{FG}
$$

$$
edge\ preservation
$$

$$
profile\ correlation
$$

$$
peak\ localization\ error
$$

以及：

### hallucination rate

也就是：

> GT 没有结构的地方，模型生成了多少结构。

这会非常适合你的 mask-aware 方法。

---

# 27. 一个尤其值得做的指标：Measured / Reconstructed consistency

既然你的方法最大的特点是：

> 我知道哪些位置是真实采样的。

那么可以直接定义：

$$
E_{measured}
=
\|M\odot(\hat X-Y)\|
$$

以及：

$$
E_{missing}
=
\|(1-M)\odot(\hat X-X)\|
$$

这样就可以分析：

> measured region 保真度

和：

> hallucinated/missing region reconstruction capability

这是当前普通 PSNR 无法表达的。

我认为这会非常适合成为你的核心 evaluation。

---

# 28. 当前最值得做的一件事情：重新建立“证据树”

我建议最终论文不再是：

```text
2D result
2.5D result
3D result
MDC result
pollen result
...
```

而是：

```text
Problem
  ↓
Sparse sampling destroys structure
  ↓
Physics initialization helps
  ↓
Sampling mask contains missing-information geometry
  ↓
Physics + mask structured fusion helps
  ↓
Residual learning recovers fine structures
  ↓
Works across 2D / 2.5D / 3D
  ↓
Robustness under stronger sparsity / lower photon budget
```

这会非常连贯。

---

# 29. 我给当前项目的阶段性评价

| 维度 | 当前状态 | 评价 |
|---|---|---|
| 问题价值 | ★★★★★ | 很好 |
| 数据基础 | ★★★★☆ | 足够做论文 |
| 2D/2.5D/3D探索 | ★★★★☆ | 已经比较充分 |
| Physics prior | ★★★★☆ | 很有潜力 |
| Sampling mask | ★★★★☆ | 值得深化 |
| Prior fusion | ★★★☆☆ | 目前偏工程 |
| 3D physics correctness | ★★☆☆☆ | 需要加强 |
| Loss implementation | ★★☆☆☆ | 存在关键问题 |
| Benchmark consistency | ★★☆☆☆ | 需要系统重构 |
| External generalization | ★★★☆☆ | 已发现明显 domain shift |
| 当前论文创新性 | ★★★☆☆ | 经过重构后可明显提高 |
| 潜在论文创新性 | ★★★★☆-★★★★★ | 我认为值得继续做 |

---

# 30. 我会把当前问题按照优先级这样处理

## P0：先不要继续堆实验

必须检查：

**① DINO loss 是否真正反传**

这是代码级 bug。

**② DINO feature extraction 是否符合 DINOv2 标准 forward**

这是实现严谨性问题。

**③ Wiener 是否真的为 3D Wiener**

目前不是。

**④ PSF consistency 是否真的 3D**

目前也不是。

**⑤ z baseline 74 dB / SSIM 1.0 是否合理**

这个结果非常可疑，需要重算。

**⑥ test normalization 是否读取 GT**

必须排除 leakage。

---

## P1：重新冻结 benchmark

只保留：

### Main benchmark

**CIDC25**

- 2D：XY
- 2.5D：Z
- 3D：XYZ

分别：

$$
K=2,4,8
$$

### External validation

**MDC25-3D**

### External transfer

**pollen**

这样整个论文就会干净很多。

---

# 31. P2：正式升级方法

我建议最终核心方法变成：

# **Physics-Initialized Sampling-Aware Residual Reconstruction**

大致流程：

$$
Y,M
$$

↓

### Anisotropic PSF inversion

$$
X_W=\mathcal W(Y;H)
$$

↓

### Sampling confidence encoder

$$
C=g(Y,M,H)
$$

↓

### Prior-aware fusion

$$
F_0=\Phi(Y,X_W,C)
$$

↓

### Residual Swin/Video Swin

$$
R=f_\theta(F_0)
$$

↓

$$
\boxed{\hat X=X_W+R}
$$

↓

### Data consistency

$$
\mathcal L
=
\mathcal L_{recon}
+
\lambda_{dc}
\|\mathcal H(\hat X)-Y\|
$$

这样你整个项目就从：

> “用了几个 Transformer 做 sparse-to-dense”

升级为：

> **“利用显微成像物理模型与采样几何显式约束深度重建。”**

这两者在论文层面的说服力差别很大。

---

# 32. 最后，我对这个仓库目前的真实判断

我不会把它评价为“半成品杂乱代码”。

实际上，你已经完成了几个很重要的阶段：

1. 已经有真实的 3PM sparse reconstruction problem；
2. 已经覆盖 2D / 2.5D / 3D；
3. 已经有 PSF / Poisson / readout noise 等物理退化；
4. 已经发现了 2.5D Z-sparse 的明显优势；
5. 已经发现 Wiener 对困难 XY-sparse 场景有巨大帮助；
6. 已经开始把 mask 从普通输入提升到 prior；
7. 已经意识到 MDC25 / pollen 与主数据存在 domain shift；
8. 最近的实验协议已经明显比早期版本严谨。 

**真正的问题是，现在这些东西还是“多个好点子的集合”，还没有完全收敛成一个不可替代的方法。**

我目前最看好的收敛方向是：

> **Anisotropic physics initialization + sampling-aware confidence + residual reconstruction + explicit data consistency**

而不是继续增加 Transformer 的复杂度。

这个方向同时能把你现在的 **2D / 2.5D / 3D、PSF、Wiener、mask、sampling、noise、structure preservation** 全部统一进一个理论框架中。

还有一点我建议特别强调：当前 report 已经把“Mask + Wiener”定位成主线，但报告里的若干结论仍然建立在尚未彻底核验的历史结果上。 因此**现在最合理的下一步不是继续写论文，而是先做一次“代码—数据—指标—结果”的 forensic audit，把可信结果筛出来。**

如果按照科研推进效率，我下一轮最建议直接做的是：**逐文件审计 `dataset.py → models/swin_unet_2d.py → models/video_swin_unet.py → loss.py → eval.py`，把所有可能影响实验有效性的代码问题逐项定位，并最终给你整理成一份“P0/P1/P2 修复清单 + 新论文方法框架 + 必做实验矩阵”。**

## Prompt:
2026/9/9 16:48:28

大致同意你的建议，你可以先帮我逐文件审计一下。我补充回复以下几点：①现在DINOv2 loss似乎确实我没有让它参与训练，可以直接先去掉，或者考虑直接把backbone替换为3D DINO（但会改动太大）。②建立真正的3D Wiener确实很有必要，这个需要重点帮我审查并设计一下。③你提出把 mask 从 binary mask 升级为“uncertainty / confidence field”  ，但是mask本身不是我已知的真实采样点吗？未被采样的地方通过Wiener初始化，这些地方建立不确定场似乎才更合适？这里我没有太理解，需要你再解释一下。但Sampling-aware Prior Encoding 的设计确实可以考虑。④让网络预测 residual 这个也可以考虑进正式的修改方案。⑤训练/评估规范问题 需要重点审查。⑥你提到的domain shift 问题可能源自于在另一笔数据测试上时我没有重新训练，而是直接使用已训练好的权重，但数据源不同。⑦你提到的进一步把 2D/2.5D/3D 统一起来  这件事，帮我评估设计一下吧，如果一个3D网络可以同时做到，我觉得确实更有意义

## Response:
2026/9/9 16:48:50

你的补充判断很关键，尤其是“mask=真实采样位置、缺失区域才应承载不确定性”这一点。我会据此调整设计：不把 mask 本身误称为 uncertainty，而是区分 **sampling certainty** 与 **reconstruction uncertainty**。接下来重点做代码级审计，并把 2D/2.5D/3D 统一方案一起验证。

我已经把数据管线和评估管线追到比较深的位置。现在有几个比之前更重要的发现：**当前 3D Wiener 确实只是逐层 2D Wiener；3D 数据训练时又是从单个“GT stack”在线合成 sparse，而不是利用真实 paired acquisition；另外部分评估指标/统计实现也需要调整。** 接下来我会把这些与模型结构逐项对齐。

> **获取并检查仓库及视频Swin与CIDC数据集代码**
>
> 
>
> **克隆仓库以审计数据集与评估代码**
>
>
> **克隆仓库并审计数据集与评估代码**
>
> 
>
> **评估统一架构**
>
> 我正在检查视频模型的统一设计。当前网络能处理可变时间长度，但时间维度未下采样；二维半输入还需中心切片输出或掩码损失，并妥善处理单帧输入与时间窗口不匹配的问题。
>
> **评估了统一架构 решений**
>
> 
>
> **审计数据集构建、三维重建与评估流程**
>
> 
>
> **审计CIDC25数据划分评估基线与实验协议差异**
>
> 
>
> 思考了 1m 19s

可以。结合你刚补充的几点，我把审计结论进一步收紧了。尤其是你对 mask 的理解是正确的：**binary mask 本身应该表示“measurement certainty / acquisition support”，而不是 uncertainty；真正的不确定性应该定义在 $1-M$ 的缺失区域，并且最好与 Wiener 逆问题的条件性结合起来。**

下面这轮我按照“**文件 → 实际行为 → 科学含义 → 风险 → 修改建议**”来审计。目前我认为已经发现了几项需要在重新实验之前解决的 **P0 级问题**。

---

# 一、先给出目前最重要的结论

经过逐文件追踪，目前整个项目实际上存在一个比“模型创新不够”更核心的问题：

> **现在仓库中的 2D、2.5D、3D 三条实验线，在数据退化模型、normalization、sparse representation 和 evaluation protocol 上并没有真正统一。**

因此，当前报告里的很多数值可以作为“探索性结果”保存，但**暂时不能直接作为最终论文的定量证据**。

尤其需要注意下面五个问题：

### P0-1：CIDC25 的 2D/2.5D sparse 数据已经被预先插值，然后评估又进行不同程度的再次处理

`prepare_cidc25_sparse3d.py` 中：

- XY：先生成 sparse，再执行 `warmstart_xy()`，最后把 **warm-start 后的结果**保存成 `low`
- Z：直接 `sparsify_z()`，而 `sparsify_z()` 本身就已经做了线性插值
- XYZ：Z 插值后再做 XY sparse，再 warm-start。

但后面的 unified evaluation 又：

- 2D：对这个已经 warm-start 的 low 再套 `make_mask_2d()` + `warm_start/Wiener`
- 2.5D：直接读取预插值的 low frames，不再重新产生 raw sparse mask
- 3D：反而完全不使用 prepared sparse volume，而是重新从 GT volume 生成 sparse。

所以现在实际上是：

```text
2D     : sparse → interpolation → save → 再mask → 再interpolation → network

2.5D   : sparse → interpolation → save → network

3D     : GT → sparse → interpolation/Wiener → network
```

这三个任务并不是同一个 inverse problem。

这也是我现在对 `74.14 dB / SSIM=1.0` 这类结果最警惕的原因之一：当前 baseline 并不是在严格意义上评价“raw sparse input → GT”。

---

# 二、`dataset.py` 审计

这是目前整个项目最需要重构的文件。

---

## 2.1 TIFF 读取基本没有问题

`load_tiff_float32()` 对 uint16 / uint8 / float 数据进行了处理，并且默认对 stack 使用 `error`，这一点很好，可以避免静默把 3D 数据压成 2D。

这个设计建议保留。

---

# 三、最大的 dataset 问题：GT leakage through normalization

这是我认为目前最重要的实验有效性问题之一。

例如：

```python
vmin = patch_np.min()
vmax = patch_np.max()
patch_np = (patch_np - vmin) / (vmax - vmin)
```

这里的 `patch_np` 在 `CIDC25Volume3DDataset` 中就是 **GT volume patch**。

也就是说：

$$
X \rightarrow \mathrm{minmax}(X)
$$

然后才：

$$
Y=M\odot X
$$

这意味着 sparse observation 的数值尺度，是通过 GT 的 min/max 决定的。

更进一步，在 `SparseStack3D` 中：

```python
vmin = min(gt_np.min(), sp_np.min())
vmax = max(gt_np.max(), sp_np.max())
```

也把 GT 用于确定输入 normalization。

这严格来说属于：

# target-dependent preprocessing

它未必算传统意义上的“标签泄漏”，但在真实 inference 场景下是不成立的，因为实际部署时你不知道 GT 的：

$$
\min(X),\max(X)
$$

---

## 3.1 正确的方案

应该采用：

$$
Y_{norm}=N(Y;\theta_{train})
$$

其中 normalization 参数：

$$
\theta_{train}
$$

只能来自：

- acquisition calibration；
- training-set statistics；
- 或 observed $Y$ 本身。

**不能由 target $X$ 决定。**

我建议最终 3PM pipeline 直接固定 intensity convention，例如：

$$
Y\in[0,1]
$$

然后所有模型都使用同一个 acquisition normalization。

这样 PSNR 才真正可比较。

---

# 四、2.5D 数据目前不是严格意义上的“Z sparse reconstruction”

这是一个非常重要的发现。

`prepare_cidc25_sparse3d.py` 中：

```python
kept = vol[kept_indices]
interpolated = zoom(kept, ...)
```

所以 Z-sparse 以后，保存下来的是：

> **已经经过线性 Z interpolation 的 dense volume**

而不是：

$$
M_z\odot X
$$

形式的 sparse observation。

因此目前的 2.5D 模型实际上更接近：

$$
\text{interpolated neighboring slices}
\rightarrow
\text{center slice refinement}
$$

而不是：

$$
\text{raw sparse neighboring slices}
\rightarrow
\text{center slice reconstruction}
$$

这件事情特别重要。

---

# 五、我建议重新定义 Z-sparse 的 observation model

真正的 Z sparse 应该是：

$$
Y_z=M_z\odot X
$$

例如 K=4：

```text
Slice 0   measured
Slice 1   missing
Slice 2   missing
Slice 3   missing
Slice 4   measured
...
```

然后：

### 2.5D

输入：

$$
[Y_{t-1},Y_t,Y_{t+1},M_{t-1},M_t,M_{t+1}]
$$

输出：

$$
\hat X_t
$$

这样才真正利用：

> adjacent measured slices + sampling geometry

而不是先由传统 interpolation 填掉，再让网络 refine。

这个修改我认为是**必须做的**。

---

# 六、XY sparse 也是类似问题

当前 `prepare_cidc25_sparse3d.py`：

$$
X
\rightarrow
M_{xy}X
\rightarrow
\text{warmstart}
$$

然后保存 warm-start。

如果之后再做 mask，就已经改变了物理过程。

所以最终应该：

### dataset 保存 raw sparse measurement

$$
Y=M\odot H(X)+n
$$

### warm-start 只在 dataset runtime 生成

$$
X_0=W(Y)
$$

这样才干净。

---

# 七、Wiener：现在确实是“假的 3D Wiener”

你补充说这个需要重点设计，我完全同意，而且这里非常值得升级。

当前：

```python
wiener_init_3d()
```

实际上：

```python
for t:
    out[t] = wiener_init_2d(...)
```

也就是：

$$
X_W(z,:,:)
=
W_{2D}(Y(z,:,:))
$$

而传入的：

```python
psf_sigma_axial
```

根本没有参与 Wiener inverse。

---

# 八、真正的 3D Wiener 应该怎么设计

你这个项目非常适合做真正的 anisotropic 3D Wiener。

三光子场景下，假设：

$$
H(x,y,z)
$$

是三维 PSF。

对应：

$$
\mathcal{H}(X)
=
H*X
$$

观测：

$$
Y=M\odot(H*X)+n
$$

那么 3D Wiener initial estimate：

$$
X_W
=
\mathcal F^{-1}
\left[
\frac{H^*}
{|H|^2+K}
\mathcal F(Y)
\right]
$$

其中：

$$
H=\mathcal F(h_{xyz})
$$

且：

$$
h_{xyz}
=
\mathcal G_{\sigma_{xy},\sigma_z}
$$

满足：

$$
\sigma_z>\sigma_{xy}
$$

---

# 九、但我不建议直接使用一个固定 scalar Wiener $K$

你现在：

$$
K=0.01
$$

实际上过于简单。

更合理的是：

$$
K(f_x,f_y,f_z)
=
\frac{S_n(f)}{S_x(f)}
$$

如果没有真实 PSD，可以从观测估计：

$$
K=\lambda\cdot \mathrm{NoiseLevel}
$$

然后至少做：

$$
\lambda\in\{0.001,0.005,0.01,0.05,0.1\}
$$

消融。

进一步可以实现：

### SNR-adaptive Wiener

$$
W(f)=
\frac{H^*(f)}
{|H(f)|^2+\lambda(f)}
$$

这会比“加入 Wiener”更像一个真正的 physics-guided method。

---

# 十、进一步升级：Wiener 不只给你一个初值，还能给 uncertainty

这里正好回应你刚才对 mask 的疑问。

你的理解是：

> mask 是已知真实采样位置，因此它本身不是 uncertainty。

**完全正确。**

所以我建议以后不要叫：

> uncertainty mask

而叫：

# Sampling Support / Measurement Mask

即：

$$
M(x)=1
$$

代表：

> 真正测到了。

$$
M(x)=0
$$

代表：

> 没测到。

---

# 十一、真正的 uncertainty 应该在哪里？

应该定义：

$$
U(x),\quad x\in\{M=0\}
$$

即：

> **模型对 missing region 的重建可信程度。**

而且这个 $U$ 最好不是纯神经网络 hallucination，而是具有物理依据。

例如可以构造：

$$
C(x)
=
G_\sigma*M(x)
$$

这里：

- 离真实采样点近 → confidence 高
- 离真实采样点远 → confidence 低

这其实是最简单的 sampling-support confidence。

---

# 十二、还可以进一步把 Wiener 的 conditioning 纳入 uncertainty

例如在频域里：

$$
U(f)
\propto
\frac{K}{|H(f)|^2+K}
$$

当：

$$
|H(f)|\rightarrow0
$$

说明光学系统根本没有保留该频率的信息，那么 Wiener 对该频率的恢复就是高度不确定的。

所以可以构造：

$$
U
=
g(M,H,K)
$$

最终形成：

```text
M             = measurement support
C             = sampling confidence
U             = reconstruction uncertainty
X_W           = physics initialization
```

这就非常漂亮了。

---

# 十三、因此你的 Sampling-aware Prior Encoding 可以真正重新定义

我现在更推荐：

$$
(Y,M,X_W,C,U)
$$

五类信息。

但不是把它们直接 concat。

可以：

$$
F_Y=E_Y(Y)
$$

$$
F_W=E_W(X_W)
$$

$$
F_C=E_C(C)
$$

然后：

$$
F_{prior}
=
\mathrm{GatedFusion}(F_W,F_C)
$$

再：

$$
F_{rec}
=
F_Y+
(1-U)\odot F_{prior}
$$

这个逻辑特别值得研究：

> **越不确定的区域，越避免过度相信 physics initial estimate；越有可靠采样支持的区域，越可以利用 prior。**

这比当前单纯：

$$
M\rightarrow gate
$$

强很多。

---

# 十四、当前 `prior_fusion.py` 本身没有严重 bug，但创新程度还比较有限

现在：

$$
G=\sigma(Conv(M))
$$

$$
X'=X(1+\alpha(G-0.5))
$$

而且：

$$
\alpha=0
$$

初始化。

工程上没问题。

但科学上，它现在还只能说：

> mask-guided feature modulation

距离一个完整的 physics-aware reconstruction mechanism 还有一段距离。

因此我建议不急着继续做 cross-attention。

先完成：

$$
Y+M+X_W+U
\rightarrow
\text{confidence-aware residual reconstruction}
$$

就足够形成漂亮的方法。

---

# 十五、`loss.py`：DINOv2 可以直接删掉

这一点按照你的意见，我完全赞成。

而且现在删掉反而更干净。

当前项目的主要问题是显微图像 reconstruction，并不是 semantic recognition。

DINOv2 在这里存在两个问题：

第一，当前实现确实使用了：

```python
@torch.no_grad()
```

整个 feature extraction 被包起来，因此你之前判断“它没有参与网络优化”是正确的。

第二，即使修复，也存在 domain mismatch：

$$
fluorescence\ microscopy
\neq
natural\ image
$$

所以我不建议现在为了“高级 loss”再引入 3D DINO。

### 当前版本：

$$
L=
\lambda_1L_{pixel}
+
\lambda_2L_{structure}
+
\lambda_3L_{data}
$$

就够了。

具体可以使用：

$$
L_{rec}=L_1+\lambda_{ssim}L_{MS-SSIM}
$$

以及真正的：

$$
L_{DC}
=
\|M\odot(H*\hat X-Y)\|_1
$$

最终：

$$
\boxed{
L=L_{rec}+\lambda_{DC}L_{DC}
}
$$

这比 DINO loss 更符合你的研究问题。

---

# 十六、当前 PSF consistency loss 也应该重写

现在 `PSFConsistencyLoss` 使用的是 `_blur_2d()`，所以：

$$
L_{PSF}
=
\|H_{xy}*\hat X-Y\|^2
$$

对于 3D，它没有真正使用：

$$
H_{xyz}
$$

虽然参数中存在 `sigma_axial`，但实际 blur 并没有使用 axial dimension。

因此以后应该统一成：

$$
L_{DC}
=
\|M\odot(\mathcal H(\hat X)-Y)\|^2
$$

注意这里一定要有：

$$
M
$$

因为 missing region 没有观测，不能拿 prediction 和不存在的 measurement 比。

这一点非常关键。

---

# 十七、`models/swin_unet_2d.py`：发现一个隐藏但很重要的问题

你的代码实现了：

```text
W-MSA
SW-MSA
```

但是在 `SwinBlock2D` 里：

```python
torch.roll(...)
window_partition(...)
self.attn(wins)
```

并没有构造标准 Swin Transformer 的 **shifted-window attention mask**。

也就是说：

> 进行了 cyclic shift，但没有阻止 roll 后不相邻区域之间进行非法 attention。

因此严格来说：

# 这不是完整标准意义上的 Swin Transformer SW-MSA。

这不是致命的运行 bug，但会影响：

- 局部边界建模；
- receptive field；
- 和标准 Swin 的可比性；
- 2D/3D window attention 的科学解释。

建议重写。

---

# 十八、3D Video Swin 有同样的问题

`VideoSwinBlock` 也是：

$$
\text{roll}
\rightarrow
\text{partition}
\rightarrow
\text{attention}
$$

但没有 shift attention mask。

因此 3D 版本也需要一起修。

---

# 十九、3D 网络目前其实有一个很有意思的结构特点

它的：

```python
PatchMerging3D
```

只在：

$$
H,W
$$

上 downsample，而：

$$
T
$$

完全保持。

这其实未必是缺点。

对于你的显微问题反而合理，因为：

$$
z
$$

方向的 sampling density 与：

$$
xy
$$

本身就不同。

如果对 z 也随意 /2，会混淆真实 axial spacing。

所以我建议：

> **保留 spatial-only downsampling，反而可以包装成 anisotropy-aware 3D architecture。**

但是需要让真实物理尺度显式进入网络。

---

# 二十、关于你提出的“一个 3D 网络统一 2D/2.5D/3D”

我认真评估后认为：

# **值得做，而且比现在三个模型完全独立更有研究意义。**

但是不能简单地说：

> “把 2D 输入也喂给 Video Swin。”

需要做成真正的：

# Unified 3D Conditional Reconstruction Network

---

# 二十一、我建议的统一方式

统一模型始终接受：

$$
X\in\mathbb R^{B\times C\times T\times H\times W}
$$

然后：

### 2D

$$
T=1
$$

输入：

$$
[Y_0,M_0,X_{W,0}]
$$

输出：

$$
\hat X_0
$$

---

### 2.5D

$$
T=3/5
$$

例如：

$$
[Y_{t-1},Y_t,Y_{t+1}]
$$

输出：

$$
\hat X_t
$$

只计算 center slice loss。

---

### 3D

$$
T=16
$$

输入 16-frame volume。

输出：

$$
\hat X_{1:16}
$$

计算全 volume loss。

---

# 二十二、统一网络其实可以用一个“target/query mask”解决

设计：

$$
Q\in\{0,1\}^{T}
$$

例如：

### 2D

$$
Q=[1]
$$

### 2.5D

$$
Q=[0,1,0]
$$

### 3D

$$
Q=[1,1,\cdots,1]
$$

网络输出整个：

$$
\hat X_{1:T}
$$

但 loss：

$$
L=
\frac{
\sum_t Q_t L_t
}{
\sum_t Q_t
}
$$

这样：

> **同一个 3D reconstruction model 可以同时执行 2D、2.5D、3D。**

这是我认为很漂亮的统一设计。

---

# 二十三、进一步把 sampling mode 也统一

使用：

$$
M\in\mathbb R^{1\times T\times H\times W}
$$

自动表示：

### XY sparse

$$
M(t,:,:)
$$

每个 slice 都 sparse。

### Z sparse

某些：

$$
M_t=1
$$

某些：

$$
M_t=0
$$

### XYZ sparse

两者同时存在。

于是模型根本不需要告诉自己：

> “现在我是 2D / 2.5D / 3D。”

它只需要看：

$$
Y,M
$$

以及：

$$
T
$$

---

# 二十四、这样模型就从“维度特定模型”升级成了“采样条件模型”

最终可以写成：

$$
\boxed{
\hat X=
F_\theta(Y,M,H)
}
$$

或者加入 initialization：

$$
\boxed{
\hat X=
X_W+
R_\theta(Y,M,X_W,U)
}
$$

这就是我认为最值得发展的版本。

---

# 二十五、但当前 Video Swin 不能直接做到这个统一

原因有两个。

### 问题 1：Temporal window

现在：

```python
window_size=(2,8,8)
```

要求 T 基本上能够被 2 整除。

而 2D：

$$
T=1
$$

天然不满足。

---

### 问题 2：relative positional bias 固定 window size

当前：

$$
(2T-1)(2H-1)(2W-1)
$$

relative bias table 是固定大小。

因此要真正统一：

$$
T=1,3,5,8,16
$$

最好不要继续依赖这种严格固定 window 的实现。

---

# 二十六、所以我更推荐最终重构为：

## Unified Axial-3D Reconstruction Network

不是简单的 Video Swin。

可以采用：

$$
Spatial\ Window\ Attention
+
Axial\ Z\ Attention
$$

例如：

```text
Input
 │
 ├── measurement feature
 ├── mask/support feature
 ├── Wiener feature
 └── uncertainty feature
 │
 ▼
Spatial encoder
 │
 ▼
Axial/Z interaction
 │
 ▼
Residual decoder
 │
 ▼
ΔX
 │
 ▼
XW + ΔX
```

这样：

- T=1 → 纯空间 reconstruction
- T=3 → local Z-context
- T=16 → volumetric reconstruction

完全可以共用参数。

---

# 二十七、这样做还有一个额外优势：可以研究“context scaling”

你甚至可以固定同一个模型：

$$
T=1,3,5,9,17
$$

做：

$$
PSNR \text{ vs. axial context}
$$

这样就能回答：

> 到底需要多少 Z-context 才能恢复目标结构？

这个问题本身很有研究价值。

---

# 二十八、关于你说的 domain shift，我修正之前的表述

你指出：

> MDC25 / pollen 很可能是直接用已训练好的权重，没有重新训练。

这个解释是非常合理的。

因此我现在不会把当前 MDC25 结果直接称作：

> “模型存在严重 domain shift”

更准确的说法应该是：

# zero-shot cross-dataset generalization

也就是说目前实验回答的是：

$$
P_{train}
\rightarrow
P_{test}
$$

发生变化时，模型还能不能工作。

这其实可以保留，而且很有价值。

但应该明确分成：

### Same-domain reconstruction

训练：

$$
CIDC25
$$

测试：

$$
CIDC25
$$

### Cross-domain transfer

训练：

$$
CIDC25
$$

测试：

$$
MDC25
$$

### External real sparse transfer

训练：

$$
CIDC25
$$

测试：

$$
pollen
$$

而不能混成一个 benchmark。

---

# 二十九、`eval.py` 的统计设计也需要修改

这里我发现几个值得修的地方。

## 1. Wilcoxon 当前 pairing 不严格

现在：

```python
a[:n]
b[:n]
```

然后直接做 paired Wilcoxon。

这只有在：

$$
a_i
\leftrightarrow
b_i
$$

是完全相同测试样本时才成立。

应该根据：

```text
image / frame ID
```

merge 后再配对。

否则：

> frame ordering 一旦不同，p-value 就没有统计意义。

---

## 2. `MS-SSIM` 实现不是严格标准实现

现在是对每个尺度计算：

$$
SSIM
$$

然后直接 weighted product。

但标准 MS-SSIM 的 luminance / contrast-structure components 组合并不完全等价于简单：

$$
\prod_i SSIM_i^{w_i}
$$

所以论文里最好统一使用经过验证的标准 implementation。

---

# 三十、`FwPSNR` 我建议删除

当前：

$$
wmap=
|\mathrm{IFFT}(|FFT(GT)|)|
$$

然后作为 spatial weight。

这不是一个非常标准、容易被 reviewer 接受的 frequency-weighted PSNR 定义。

所以如果不是你的核心贡献：

> **不要为了“指标很多”保留。**

论文中真正推荐：

- PSNR
- SSIM
- MS-SSIM
- MAE
- foreground PSNR/SSIM
- structural/profile metric
- data consistency

就足够。

---

# 三十一、foreground metric 也要谨慎

当前：

```python
threshold = percentile(GT, 75)
```

然后定义 foreground。

这可以作为 exploratory metric，但论文里如果强调：

> cellular structure preservation

最好不要完全依赖 intensity percentile。

更理想的是：

- segmentation mask；
- cell/nucleus annotation；
- 或基于 GT morphology 的稳定 ROI。

否则不同方法可能改变 intensity distribution，反过来影响 foreground mask。

---

# 三十二、`inference.py` 目前存在明显代码不同步

这个文件已经不是“科研设计问题”，而是实际运行问题。

例如它调用：

```python
warm_start_2d(sparse, mask, opts.sparsity_factor)
```

但当前 `dataset.py` 的定义是：

```python
warm_start_2d(sparse, factor)
```

也就是说当前 inference 代码与 dataset API 已经不一致。

3D 也是类似问题。

因此：

# inference.py 应该在 protocol 重构后全部重写

不要继续 patch 旧代码。

---

# 三十三、现在我对整个代码仓库的文件级评价

| 文件 | 当前评价 | 优先级 |
|---|---|---|
| `options.py` | 参数很多，但逐渐形成 protocol guard | P1 |
| `dataset.py` | 功能完整，但数据协议混杂、GT normalization leakage | **P0** |
| `models/swin_unet_2d.py` | 基本可用，但 SW-MSA 缺 attention mask | **P0** |
| `models/video_swin_unet.py` | 有真正 3D attention 雏形，但仍是固定 T-window | **P1** |
| `models/prior_fusion.py` | 简洁合理，但方法创新还比较弱 | P1 |
| `loss.py` | DINO 当前应删除；PSF loss 应改成 masked 3D DC | **P0** |
| `train.py` | 基本框架合理 | P1 |
| `eval.py` | 内容丰富，但统计/metric 有若干问题 | **P0/P1** |
| `inference.py` | 已与现有 API 不同步 | **P0** |
| `prepare_cidc25_sparse3d.py` | 是当前最大 protocol 问题来源之一 | **P0** |
| `eval_sparse3d_unified.py` | 三种 task 的评估路径不一致 | **P0** |
| `generate_prior_fusion_experiments.py` | 实验矩阵思路很好 | P1 |
| `docs/*` | 最近的协议重整方向正确 | 保留 |

---

# 三十四、因此我建议不要直接在当前代码上继续加功能

应该做一次：

# **Protocol v3 重构**

核心只留下一个 physics-consistent data model：

$$
\boxed{
Y=M\odot(H*X+n)
}
$$

然后 dataset 负责生成：

$$
(Y,M)
$$

network 前端生成：

$$
X_W=\mathcal W(Y;H)
$$

再：

$$
U=C(Y,M,H)
$$

最后：

$$
\hat X=X_W+R_\theta(Y,M,X_W,U)
$$

---

# 三十五、最终推荐的方法框架

我现在最推荐的正式版本可以叫：

## **Physics-Initialized Sampling-Aware Residual Reconstruction**

简称可以考虑：

**PI-SARR**

整体：

```text
                 Ground-truth X
                       │
                       │ synthetic / real acquisition
                       ▼
             Anisotropic PSF H + Noise
                       │
                       ▼
                  Sparse Y
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        Sampling Mask M      3D Wiener
             │                   │
             │                   ▼
             │                  XW
             │                   │
             └───────┬───────────┘
                     ▼
           Sampling/Physics Encoder
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
      Support Confidence C   Uncertainty U
             │                │
             └───────┬────────┘
                     ▼
             Prior-aware Fusion
                     │
                     ▼
             Residual Network
                     │
                     ▼
                  ΔX
                     │
                     ▼
              X̂ = XW + ΔX
                     │
                     ▼
             Data Consistency
       M ⊙ [H * X̂ - Y]
```

这个框架已经可以同时容纳：

- 2D
- 2.5D
- 3D
- XY sparse
- Z sparse
- XYZ sparse
- Wiener
- PSF
- mask
- uncertainty
- residual learning
- data consistency

---

# 三十六、接下来我建议的重构顺序

不是继续做实验，而是严格按下面顺序。

### 第一阶段：先修“科学正确性”

1. 删除 DINOv2
2. 重写 sparse generation
3. 禁止 GT-dependent normalization
4. raw sparse 与 warm-start 分离
5. 重写真正 3D anisotropic Wiener
6. 重写 3D masked data consistency
7. 修复 2D/3D shifted-window attention mask
8. 重写统一 evaluation

### 第二阶段：建立新 baseline

先只比较：

$$
B0:\quad Y\rightarrow Network
$$

$$
B1:\quad Y+M\rightarrow Network
$$

$$
B2:\quad X_W+M\rightarrow Network
$$

$$
B3:\quad X_W+M+Residual
$$

$$
B4:\quad X_W+M+U+Residual
$$

这个矩阵会非常干净。

### 第三阶段：统一 3D 网络

让同一个模型支持：

$$
T=1,\;3,\;5,\;16
$$

并通过 target/query mask 同时支持：

- 2D
- 2.5D
- 3D

### 第四阶段：最后再做跨数据泛化

这时：

$$
CIDC25\rightarrow CIDC25
$$

作为主实验；

$$
CIDC25\rightarrow MDC25
$$

作为 zero-shot transfer；

$$
CIDC25\rightarrow pollen
$$

作为 external real-sparse validation。

这样逻辑会非常完整。

---

# 三十七、我对你刚才七点补充意见的最终判断

### ① DINOv2

**直接去掉。**

目前没有必要改成 3D DINO，成本大、收益不明确，反而会稀释核心物理 reconstruction story。

### ② 真正 3D Wiener

**强烈建议做，而且应该成为 P0。**

它甚至可能成为整个论文最重要的物理模块之一。

### ③ Mask / uncertainty

你的理解正确：

$$
M=\text{measurement support}
$$

不是 uncertainty。

真正值得加入的是：

$$
C=\text{sampling confidence}
$$

和：

$$
U=\text{reconstruction uncertainty}
$$

其中 $U$ 可以由 PSF conditioning、Wiener amplification、采样距离等物理因素构造。

### ④ Residual

**建议正式加入。**

$$
\hat X=X_W+\Delta X
$$

这是非常自然的 physics-guided reconstruction formulation。

### ⑤ Train/eval protocol

**是当前最优先的问题。**

特别是 GT-dependent normalization、预插值 sparse 数据、2D/2.5D/3D evaluation 不一致。

### ⑥ Domain shift

你的解释成立。

当前更准确的称呼应该是：

> zero-shot cross-domain generalization

不应该直接把 MDC25 差结果当成“方法失败”。

### ⑦ 一个 3D 网络统一 2D/2.5D/3D

**我认为值得做，而且很可能比单纯再做一个更复杂的 Transformer 更有创新性。**

但需要做成真正的：

$$
\boxed{
\text{Unified 3D Conditional Residual Reconstruction}
}
$$

而不是简单把 2D image reshape 成 3D。

---

**最关键的一点是：目前仓库已经不是“继续调参就能得到最终论文结果”的阶段，而是应该先进行一次 protocol-level refactor。** 目前 `prepare_cidc25_sparse3d.py → dataset.py → eval_sparse3d_unified.py` 这条链必须先统一，否则继续跑 A4、Wiener、gated fusion 只会产生更多难以解释的历史结果。  

下一步最值得直接动手的是：**我可以继续逐文件把 `dataset.py / Wiener / unified 3D model / loss / eval` 设计成一套新的 Protocol v3，先给出完整的数据流、张量尺寸、数学定义、训练目标和代码修改清单，再开始实际改代码。**

## Prompt:
2026/9/9 20:03:29

你提到的“P0-1：CIDC25 的 2D/2.5D sparse 数据已经被预先插值，然后评估又进行不同程度的再次处理  ”问题。在实际采集时，比如pollen数据，如果对其进行Y方向K=2的采样采集，那么得到的图像在Y方向少了一半像素后会自动聚合缩减Y方向的尺寸，需要对其在Y方向插值才能得到于GT一样的图像尺寸（分辨率）。不过CIDC25的sparse数据构造是基于已有的给定dense数据进行mask遮蔽吗。这些问题差异可能需要考虑一下。然后你再帮我整理需要修改的清单，以及给出如何修改的方案吧，回复的同时也将清单及方案整理为一个md文档给我，便于我让codex执行

## Response:
2026/9/9 20:03:51

你这个补充区分非常重要：**真实扫描降采样**与“在已有 dense 图上做 mask 模拟 sparse”不是同一个 forward model。我会把两类协议明确拆开，再据此修正前面的 P0-1 判断。

> **厘清采样模型**
>
> 我正在区分两种退化：真实低分辨率采集应先经过光学模糊、降采样与噪声，再插值到统一尺寸；CIDC25 的掩膜缺失则保留原网格。插值只是预处理，不应被当作信息恢复。
>
> **梳理并审计稀疏到稠密时空重建代码设计**
>
> 
>
> **编写 Protocol v3 审计修改清单**
>
> 
>
> 思考了 57s

你这个补充后，我会修正之前对 **P0-1** 的表述：问题不是“CIDC25 有插值，所以错了”，而是必须把 **真实采集造成的降采样/重采样** 与 **模型用于提供初始解的 warm-start** 分开定义。

以你说的 pollen 为例：

$$
X \xrightarrow{\text{optics}} H*X
\xrightarrow{\text{Y sampling }K=2}Y_{\text{low}}
\xrightarrow{\text{resize/interpolation}}Y_{\text{grid}}
$$

这里为了让网络重新工作在 GT 的 H×W 网格上，**对 Y 方向插值是合理且必要的**；但它只是把 acquisition representation 映射回统一网格，并没有恢复丢失的信息。

而 CIDC25 是：

$$
Y=M\odot X
$$

这种已有 dense image 上的 sparse simulation。这里如果为了模拟“缺失像素”而直接保存一个已经 warm-start 的结果，再在 dataset/evaluation 阶段继续 mask 或 interpolation，就会改变实验定义。

所以后续 Protocol v3 我建议把这两个层次明确拆开：

```text
真实采集：
raw low-resolution observation
        ↓
geometric resampling
        ↓
network-grid observation

CIDC25 synthetic masking：
dense-grid observation
        ↓
masking
        ↓
network-grid observation
```

两者最终都进入：

$$
(Y_{\text{grid}},M,X_0,U)
$$

其中 $X_0$ 才是网络的 reconstruction initialization。

另外，你关于 mask 的理解也是正确的：**mask 是 measurement support，不是 uncertainty。** 真正值得研究的是 missing region 上的 reconstruction uncertainty，以及由 sampling geometry/PSF conditioning 导出的 confidence field。

我已经把完整的 **Protocol v3 修改清单、代码重构方案、3D Wiener、Sampling-aware Prior、Residual Reconstruction、统一 2D/2.5D/3D 网络、训练/评估规范和 Codex 执行优先级** 整理成 Markdown 文件：

[下载 Protocol v3 修改清单与实施方案](sandbox:/mnt/data/Protocol_v3_Modification_Checklist.md)

其中我特别把第一阶段设定为**先修 protocol 和 physics correctness，不急着增加模型复杂度**。这样后面 Codex 执行时，可以避免在旧实验协议上继续叠加新的模块。

## Prompt:
2026/9/9 20:11:16

回复似乎不完整，给我一个完整回复

## Response:
2026/9/9 20:11:37

当然。下面给你一个完整的、可以直接作为后续 Codex 开发依据的版本。我把前面的仓库审查结论、你后来补充的“真实低分辨率采集 vs. CIDC25 人工稀疏”的关键区别，以及统一 3D 模型、真正 3D Wiener、数据一致性、评估协议等全部重新整合进去。

---

# Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM  
## Protocol v3：完整技术审查、科学问题梳理与代码修改实施方案

仓库：

`winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM`

分支：

`dev`

---

# 一、首先明确这个项目真正要解决的问题

当前项目的核心任务不是简单的 image-to-image denoising，而是：

> **从不完整、稀疏、低分辨率或受采样约束的显微成像观测中，利用已知的采样物理过程和空间/时间上下文，重建高质量 dense microscopy image/volume。**

因此更准确的数学描述应该是：

$$
Y = \mathcal{A}(X) + n
$$

其中：

- $X$：真实 dense image / volume
- $Y$：实际获得的 sparse / low-resolution observation
- $\mathcal{A}$：成像系统与采样过程
- $n$：噪声

而整个网络应学习：

$$
\hat X=f_\theta(Y,\mathcal A,M)
$$

其中：

- $M$：已知测量支持 / sampling mask
- $\mathcal A$：PSF、采样方式、轴向/横向采样间距等物理信息

所以项目真正有价值的方向不是：

> “Swin-UNet + mask + Wiener”

而应该升级为：

> **physics-guided, sampling-aware, residual sparse-to-dense reconstruction**

也就是：

**采样模型 + 物理反卷积先验 + sampling-aware feature modulation + residual reconstruction + measurement-domain consistency**

---

# 二、当前代码最核心的问题

经过对目前 `dev` 分支的结构进行逐文件审查，问题并不是单个 bug，而是：

> **数据生成、预处理、Wiener 初始化、模型输入、loss、evaluation、inference 几部分实际上没有使用完全一致的 observation model。**

这才是当前项目最大的科学性问题。

目前存在以下几个明显不一致：

### 1. 2D、2.5D、3D 使用了不同的 sparse observation 定义

例如：

- 2D 使用预生成 sparse image
- 2.5D 使用预生成 sparse frames 再 stacking
- 3D 却重新从 GT 在线生成 mask

所以三个任务实际上不是严格意义上的同一个 reconstruction problem。

---

### 2. `prepare_cidc25_sparse3d.py` 已经把 sparse observation 做成了插值后的图像

尤其是 Z sparse：

```python
sparsify_z()
→ zoom()
→ save interpolated output
```

这意味着：

> 后面的 network 实际看到的已经不再是原始 sparse measurement。

而是一个经过 interpolation 的 representation。

这会导致你后面再引入 mask / Wiener / residual reconstruction 时，物理含义非常模糊。

---

### 3. 当前 `wiener_init_3d()` 并不是真正的 3D Wiener

现在实际上是：

```python
for t in range(T):
    out[t] = wiener_init_2d(...)
```

也就是说：

$$
W_{3D}(Y)
=
[W_{2D}(Y_1),W_{2D}(Y_2),...,W_{2D}(Y_T)]
$$

而不是：

$$
W_{3D}(Y)
=
\mathcal F_{3D}^{-1}
\left[
\frac{H^*}
{|H|^2+K}
\mathcal F_{3D}(Y)
\right]
$$

所以当前的 `psf_sigma_axial` 实际没有真正进入 Wiener inversion。这个问题应当作为 **P0 priority** 修复。 

---

### 4. 当前 mask 被错误地倾向于解释成“不确定性”

这个需要明确纠正。

已知的 sparse sampling mask：

$$
M(x)\in\{0,1\}
$$

表示：

> 这个位置有没有真实测量。

它不是：

> 这个位置有多么 uncertain。

两者完全不同。

例如：

```text
M = 1
```

表示：

> 有真实 measurement。

而：

```text
M = 0
```

只表示：

> 没有测量。

并不意味着：

> uncertainty = 1。

---

### 5. `dataset.py` 存在 target leakage / target-dependent normalization 风险

例如 CIDC25 dataset 中：

```python
patch_min = min(gt)
patch_max = max(gt)
```

这种 normalization 使用 GT 本身决定输入尺度。

测试阶段如果同样如此，就相当于使用：

> ground truth-derived statistics

来辅助 inference。

这会破坏严格的 reconstruction evaluation。

此外：

`SparseStack3DDataset`

使用：

```python
min(gt, sparse)
max(gt, sparse)
```

本质上也是 target-dependent preprocessing。 

---

### 6. 当前 Swin implementation 并不是标准 shifted-window attention

当前代码进行了：

```python
torch.roll()
```

然后 window partition。

但是并没有构建标准 Swin 中的 shifted-window attention mask。

因此目前更准确地说是：

> cyclic shifted local window attention

而不是严格意义上的：

> standard Swin Transformer block

2D 和 3D 都存在这个问题。  

---

### 7. 当前 Video Swin 对统一 2D/2.5D/3D 并不友好

目前 temporal window 默认类似：

```python
window_size = (2, 8, 8)
```

这种设计会导致：

- T=1 的 2D 情况很别扭
- T=3/5 的 2.5D 需要额外逻辑
- T=8/16 的 3D 才比较自然

而且当前 patch merging / expanding 只做空间降采样，不对 temporal/Z 做统一处理。

因此它更像：

> “3D video reconstruction model”

而不是：

> **unified 2D/2.5D/3D reconstruction model**

---

# 三、一个非常关键的修正：真实采集和 CIDC25 不能混为一谈

这是我们前面讨论中最重要的一点。

你的补充实际上改变了 protocol 的核心定义。

---

# 四、应该定义两种完全不同的 observation protocol

---

## Protocol A：真实低分辨率 / 稀疏采集

例如 pollen。

实际情况可能是：

原始 dense object：

$$
X
$$

经过成像：

$$
H*X
$$

然后真正进行低采样：

$$
Y_{low}=S(H*X)+n
$$

例如：

> Y 方向 K=2，只实际采集一半位置。

那么获得的图像尺寸可能真的从：

$$
H\times W
$$

变成：

$$
H/2\times W
$$

这时候恢复回 network grid：

$$
Y_{low}
\xrightarrow{R}
Y_{grid}
$$

这里的：

$$
R
$$

只是：

> geometric resampling / interpolation

而不是 reconstruction。

因此：

```text
真实低分辨率采集
        ↓
Y_low
        ↓
deterministic resampling
        ↓
Y_grid
        ↓
network
```

是完全合理的。

但是：

> resampling 只是把 data 映射回网络需要的坐标网格。

它并没有创造真实 information。

---

# 五、Protocol B：CIDC25 的 synthetic dense-grid masking

CIDC25 则不同。

它本身拥有 dense image / volume。

然后你人为构造：

$$
Y=M\odot(H*X+n)
$$

此时：

- 图像尺寸没有改变
- sparse observation 与 GT 在同一个 grid 上
- `M` 可以完全知道
- missing pixels 是人为 mask 出来的

因此这里根本不需要“真实 acquisition resampling”。

---

# 六、所以最终统一接口应该是

无论数据来源是什么，进入 reconstruction model 之前都统一成：

$$
(Y_{grid}, M, X_0, U)
$$

分别为：

### 1. `Y_grid`

network grid 上的 observation。

---

### 2. `M`

sampling support / measurement mask：

$$
M\in\{0,1\}
$$

---

### 3. `X0`

physics-based warm start：

例如：

$$
X_0=W(Y;H,M)
$$

可以是：

- bilinear interpolation
- nearest interpolation
- Wiener initialization

---

### 4. `U`

可选的 uncertainty / difficulty representation。

注意：

> `U` 和 `M` 必须是两个独立概念。

---

# 七、推荐的统一 observation builder

建议新建：

```text
data/
    observation.py
```

统一负责：

```python
build_observation(...)
```

输出：

```python
Observation(
    y_grid,
    mask,
    warm_start,
    confidence,
    uncertainty,
    raw_measurement,
    metadata,
)
```

这样训练、验证、测试、inference 全都调用同一个 observation builder。

这是整个项目应当优先完成的架构重构之一。

---

# 八、Sampling Mask 应该怎么使用

建议不要直接：

```text
mask = uncertainty
```

而应该分成三层。

---

## 1. Measurement support

$$
M(x)\in\{0,1\}
$$

这是硬 mask。

---

## 2. Sampling confidence

从 mask 生成：

$$
C_s = G_\sigma * M
$$

这样：

- measured location → confidence high
- near measured location → intermediate
- far from any sample → low

它表达的是：

> 当前空间位置距离真实测量有多远。

这比把 mask 直接塞进网络合理得多。

---

## 3. Physics-derived uncertainty / difficulty

例如 Wiener amplification 可以定义：

$$
U_f=
\frac{K}
{|H(f)|^2+K}
$$

或者：

$$
A_f=
\frac{|H(f)|^2}
{|H(f)|^2+K}
$$

然后转换成 spatial / feature-scale representation。

这里表达的是：

> 某些 frequency / spatial component 是否难以恢复。

它与 sampling support 是不同的。

---

# 九、Sampling-aware Prior Encoding

建议正式设计一个 sampling-aware prior module。

输入：

$$
Y,M,X_0,C_s,U
$$

先分别编码：

$$
F_Y=E_Y(Y)
$$

$$
F_W=E_W(X_0)
$$

$$
F_S=E_S([M,C_s,U])
$$

然后：

$$
G=\sigma(
Conv([F_S,F_W])
)
$$

得到 gating：

$$
F_{prior}=G\odot F_W
$$

最后：

$$
F_{in}=F_Y+F_{prior}
$$

这样网络不再简单 concat：

```text
image + mask + Wiener
```

而是：

> sampling condition 决定 physics prior 在什么位置、以什么强度参与 reconstruction。

这就能够形成比较完整的方法学创新。

---

# 十、Residual Reconstruction 是合理的

建议把：

```text
network → full X
```

改为：

```text
observation
      ↓
physics reconstruction
      ↓
X0
      ↓
network predicts residual
      ↓
ΔX
      ↓
X̂ = X0 + ΔX
```

数学形式：

$$
\hat X=X_0+\Delta X_\theta
$$

其中：

$$
X_0=W(Y;H,M)
$$

这样 Wiener 不再是一个普通输入，而是：

> physics-based reconstruction prior。

network 只需要学习：

$$
\Delta X=X-X_0
$$

这通常比直接学习 full reconstruction 更容易解释。

---

# 十一、真正的 3D Wiener 必须重新实现

建议新增：

```text
physics/
    __init__.py
    psf.py
    wiener.py
```

---

## 1. 3D anisotropic Gaussian PSF

因为 microscopy 中：

$$
\Delta x=\Delta y \neq \Delta z
$$

所以：

$$
h(x,y,z)
=
G_{\sigma_{xy}}(x,y)
G_{\sigma_z}(z)
$$

而不是 isotropic 3D Gaussian。

更进一步建议直接使用 physical-unit 参数：

```text
sigma_xy_um
sigma_z_um
voxel_size_xy_um
voxel_size_z_um
```

转换：

$$
\sigma_{pixel}
=
\frac{\sigma_{physical}}
{\Delta_{pixel}}
$$

这样跨 dataset 才有物理意义。

---

## 2. OTF

$$
H=\mathcal F(h)
$$

---

## 3. Wiener inversion

$$
X_W=
\mathcal F^{-1}
\left[
\frac{H^*}
{|H|^2+\lambda}
\mathcal F(Y)
\right]
$$

至少实现：

```python
build_gaussian_psf_3d()
build_otf_3d()
wiener_deconv_3d()
```

---

## 4. 初步 λ 搜索

建议：

```text
0.001
0.005
0.01
0.05
0.1
```

然后根据 validation set 选择。

---

# 十二、但是需要特别注意：稀疏采样下，普通 Wiener 不是最终答案

对于：

$$
Y=M\odot(H*X)
$$

直接：

```text
Y → FFT → Wiener
```

其实并不严格。

因为：

> missing samples 并不是 Gaussian blur 造成的 information loss。

因此第一版可以做：

$$
Y_{zf}
$$

zero-filled / resampled observation：

$$
X_0=W(Y_{zf})
$$

然后让 mask：

$$
M
$$

进入 network。

但是从长期方法创新角度，更进一步可以考虑：

> masked / iterative Wiener-proximal reconstruction。

也就是让 reconstruction periodically enforce data consistency，而不是仅仅使用 one-shot Wiener。

这一部分可以作为后续增强，不需要第一版就全部实现。

---

# 十三、Data Consistency Loss 应该加入主损失

这是非常值得增加的一个模块。

---

## CIDC25 dense-grid masking

$$
L_{DC}
=
\frac{1}{|M|}
\left\|
M\odot
(H*\hat X-Y)
\right\|_1
$$

也就是说：

> reconstructed image 经 forward physics 后，必须重新解释真实 measurement。

---

## 真实低分辨率 acquisition

不能拿 interpolation 后的 `Y_grid` 来做严格 measurement consistency。

应该：

$$
L_{DC}
=
\left\|
S(H*\hat X)-Y_{low}
\right\|_1
$$

即：

> 回到真实 measurement domain 中比较。

这一点对论文方法的严谨性非常重要。

---

# 十四、建议的最终 loss

建议删掉当前并未真正参与训练的 DINOv2 loss。

当前 DINO loss 不值得继续保留；同时没必要为了替换它而强行引入 3D DINO backbone。

建议：

$$
L=
L_{rec}
+
\lambda_{DC}L_{DC}
$$

其中：

$$
L_{rec}
=
L_1+
\lambda_{SSIM}L_{MS-SSIM}
$$

如果加入 residual regularization，可以后续测试：

$$
L_{res}
=
\|\Delta X\|_1
$$

但不建议第一版加入过多项。

---

# 十五、DINOv2 应从主训练流程删除

目前仓库中 DINO 相关配置、loss weight、feature extraction 等设计存在，但根据当前训练流程，它并没有形成有效的 optimization contribution。

所以建议：

### 删除：

```text
dino_loss
dino_weight
dino feature consistency
```

### 暂时不要：

> 为了保留 DINO 而强行迁移到 3D DINO。

因为这会显著扩大开发范围，而且与当前项目的核心科学问题并不直接相关。

---

# 十六、统一 2D / 2.5D / 3D 是值得做的

我非常建议这一点。

因为从科研角度：

```text
三个独立模型
```

的贡献明显弱于：

> **one unified sparse-to-dense reconstruction model**

---

# 十七、统一模型的输入格式

无论任务是什么：

```text
[B, C, T, H, W]
```

---

## 2D

$$
T=1
$$

例如：

```text
[B, C, 1, H, W]
```

---

## 2.5D

例如：

```text
T=3
```

输入：

```text
[z-1, z, z+1]
```

但只重建中心：

$$
Q=[0,1,0]
$$

---

## 3D

例如：

```text
T=8, 16, ...
```

所有 slice 都可以计算：

$$
Q=[1,1,1,\dots,1]
$$

---

# 十八、加入 Target Mask Q

这和 sampling mask 是两件完全不同的事情。

### Sampling mask

$$
M
$$

表达：

> 哪些位置被真实测量。

### Target mask

$$
Q
$$

表达：

> 当前输入中的哪些 slice 是 supervision target。

例如：

```text
2D:
Q = [1]

2.5D:
Q = [0,1,0]

3D:
Q = [1,1,1,1,...]
```

loss：

$$
L=
\frac{
\sum_tQ_tL_t
}{
\sum_tQ_t
}
$$

这样一个模型就可以真正统一：

```text
2D
2.5D
3D
```

---

# 十九、推荐的 unified 3D backbone 不应简单复制 Video Swin

目前 Video Swin：

```text
window = (2,8,8)
```

不适合所有 T。

更推荐：

> **spatial local attention + axial/temporal interaction**

也就是：

```text
Spatial Window Attention
        +
Z / Temporal Interaction
```

原因是：

$$
\Delta z \neq \Delta x
$$

显微镜的 axial spacing 通常和 lateral spacing 不一样。

因此可以：

### 在 XY 方向

使用 window attention：

```text
8 × 8
```

### 在 Z 方向

使用：

```text
axial attention
```

或轻量 temporal convolution。

这样：

- T=1
- T=3
- T=5
- T=8
- T=16

都可以统一处理。

---

# 二十、这是当前 Video Swin 最重要的 architectural limitation

目前：

```text
PatchEmbedding
PatchMerging
PatchExpanding
```

主要操作：

```text
H,W
```

但是：

```text
T
```

基本保留。

这其实不是坏事。

相反可以利用它重新定义模型为：

> anisotropy-aware 3D reconstruction network

这样就可以形成比较自然的 architecture story：

```text
XY:
local hierarchical feature extraction

Z:
cross-slice interaction

No isotropic assumption
```

这比单纯说 “3D Swin-UNet” 更有科研意义。

---

# 二十一、Shifted Window 必须修正

如果继续使用 Swin，必须补：

```python
attn_mask
```

标准流程：

```text
shift
↓
window partition
↓
attention mask
↓
attention
↓
reverse shift
```

否则跨 window 的 attention 连接方式并不严格符合标准 Swin。

---

# 二十二、Dataset 层必须统一重构

建议把 dataset 中：

```text
sampling
normalization
interpolation
warm start
mask
physics prior
```

全部分离。

例如：

```text
dataset.py

observation.py

normalization.py

sampling.py

physics/
    wiener.py
    psf.py
```

---

# 二十三、Normalization 必须严格重新设计

目前：

```python
minmax(gt)
```

或者：

```python
minmax(gt, sparse)
```

存在 target leakage 风险。 

推荐：

### 方案 A：dataset-level fixed statistics

训练集计算：

$$
\mu_{train},\sigma_{train}
$$

然后：

```text
train
val
test
```

全部使用相同统计量。

---

### 方案 B：measurement-only normalization

如果物理动态范围不同，可以从 observation：

$$
Y
$$

估计：

$$
\mu_Y,\sigma_Y
$$

而绝不能依赖：

$$
X_{GT}
$$

---

# 二十四、Pollen 必须保留真实 acquisition protocol

现有：

`scripts/build_pollen_sparse_2d_dataset.py`

已经能够从 pollen 数据中构建 2D GT 与 mask，这个思路可以保留。 

但建议后续不要把：

```text
mask
```

仅仅保存成：

```text
processed image
```

而应该保存：

```text
raw sparse acquisition
sampling mask
resampled observation
GT
metadata
```

至少：

```text
raw_sparse.tif
sampling_mask.tif
grid_sparse.tif
gt.tif
metadata.json
```

---

# 二十五、CIDC25 也不应该再提前保存 warm-start 作为 sparse data

目前：

`scripts/prepare_cidc25_sparse3d.py`

会把 sparse observation 处理成 interpolation / warm-start 后保存。 

建议修改为：

只保存：

```text
GT
raw_sparse
mask
sampling_config
```

真正的：

```text
interpolation
Wiener
confidence
uncertainty
```

应该在：

```text
dataset runtime
```

中动态产生。

这样：

> sampling protocol → observation builder → model

是一条完全透明的数据链。

---

# 二十六、现有 evaluation protocol 存在比较严重的问题

`eval.py` 目前功能非常丰富，但是太多 metric 并不一定提高论文质量。 

尤其需要注意：

---

## 1. MS-SSIM 当前实现并不是标准严格实现

当前更接近：

> 多尺度 scalar SSIM 加权组合。

不建议直接在论文里称为标准 MS-SSIM，除非实现完全按照标准算法。

---

## 2. FwPSNR 建议删除

当前：

```text
FFT(gt)
→ abs
→ IFFT
```

构建空间 weight 的方式并没有足够明确的理论依据。

如果论文里无法严格解释：

> 为什么这种权重代表 reconstruction quality，

建议直接删掉。

---

## 3. CNR 不应依赖固定阈值 0.5

因为不同 dataset：

```text
intensity range
contrast
normalization
```

不同。

固定：

```text
threshold = 0.5
```

可能导致 metric instability。

---

# 二十七、建议重新定义核心评估指标

论文真正需要的核心指标建议控制在：

### Image quality

- PSNR
- SSIM
- MS-SSIM
- MAE

### Structural fidelity

- profile correlation
- peak position error
- foreground MAE

### Physics consistency

新增：

$$
E_{DC}
$$

---

# 二十八、非常建议增加 measured vs missing error

这是这个研究特别值得做的 metric。

定义：

$$
E_{measured}
=
\frac{
\|M\odot(\hat X-X)\|_1
}{
|M|
}
$$

以及：

$$
E_{missing}
=
\frac{
\|(1-M)\odot(\hat X-X)\|_1
}{
|1-M|
}
$$

这样论文可以明确说明：

> 网络不是单纯把 measured pixels denoise 得更好，而是真正恢复了 missing regions。

这对 sparse-to-dense reconstruction 特别重要。

---

# 二十九、真实 acquisition 数据需要 measurement-domain evaluation

例如 pollen：

网络产生：

$$
\hat X
$$

再经过：

$$
H
$$

以及真实 sampling operator：

$$
S
$$

得到：

$$
\hat Y_{low}
=
S(H*\hat X)
$$

然后：

$$
E_{meas}
=
\|\hat Y_{low}-Y_{low}\|
$$

这比直接比较：

```text
interpolated observation
vs prediction
```

更加合理。

---

# 三十、Wilcoxon statistical test 当前有问题

目前类似：

```python
a[:n]
b[:n]
```

进行比较。

这是不安全的。

应该按照：

```text
sample_id
```

进行严格 pairing：

```text
sample_001 → modelA / modelB
sample_002 → modelA / modelB
...
```

而且：

> 如果一个 volume 有很多 correlated frames，不应该把所有 frame 当成独立统计样本。

最好以：

```text
stack / volume / specimen
```

作为 inferential unit。

---

# 三十一、训练 / validation / test protocol 必须重新定义

建议正式写成：

---

## Level 1：Same-domain

```text
Train CIDC25
↓
Validation CIDC25
↓
Test CIDC25
```

这是主要结果。

---

## Level 2：Zero-shot domain transfer

```text
Train CIDC25
↓
Test MDC25
```

不 retrain。

称：

> zero-shot cross-domain generalization

而不是“模型失败”。

---

## Level 3：External acquisition protocol

```text
Train CIDC25
↓
Test pollen
```

重点测试：

> synthetic sparse-grid masking → real sparse acquisition transfer

这个实验实际上非常有论文价值。

---

# 三十二、必须固定 sparse sampling seed

不同方法不能每次生成不同 mask。

必须：

```text
seed = fixed
```

并保证：

```text
GT
mask
noise
observation
```

完全一致。

否则 model comparison 不公平。

---

# 三十三、Inference.py 建议直接重构，而不是小修小补

目前 `inference.py` 里有独立实现的：

- PSF
- noise
- mask
- warmstart
- sliding window

而它与当前 `dataset.py` 已出现 API mismatch。

例如当前 inference 使用：

```python
warm_start_2d(sparse, mask, factor)
```

而 dataset 当前接口已经不同。

所以继续 patch 它很容易产生：

> training uses one physics model  
> inference uses another physics model

建议：

```text
inference.py
```

全面改成调用：

```text
observation.py
physics/wiener.py
model.py
```

而不保留第二套物理逻辑。

---

# 三十四、推荐的新项目结构

建议最终逐步整理成：

```text
project/
│
├── data/
│   ├── dataset.py
│   ├── observation.py
│   ├── sampling.py
│   └── normalization.py
│
├── physics/
│   ├── __init__.py
│   ├── psf.py
│   ├── wiener.py
│   ├── forward_model.py
│   └── data_consistency.py
│
├── models/
│   ├── unified_reconstruction.py
│   ├── sampling_prior.py
│   ├── residual_decoder.py
│   ├── swin_blocks.py
│   └── ...
│
├── losses/
│   ├── reconstruction.py
│   ├── data_consistency.py
│   └── ...
│
├── evaluation/
│   ├── metrics.py
│   ├── statistics.py
│   └── protocol.py
│
├── scripts/
│   ├── prepare_cidc25.py
│   ├── prepare_pollen.py
│   └── ...
│
└── inference.py
```

不要求一次完成，但是最终应该逐渐靠近这个结构。

---

# 三十五、建议的最终网络结构

可以把整个方法概括成：

```text
                 Sampling Physics
                       │
             ┌─────────┴─────────┐
             │                   │
         Sampling Mask       PSF / OTF
             │                   │
             ↓                   ↓
      Confidence / U      Wiener Initialization
             │                   │
             └─────────┬─────────┘
                       ↓
                Sampling Prior
                       │
                       ↓
        Sparse Observation Encoder
                       │
                       +
              Physics Prior Encoder
                       │
                       ↓
             Unified 3D Backbone
                       │
             Spatial + Axial blocks
                       │
                       ↓
                Residual Decoder
                       │
                       ↓
                     ΔX
                       │
                  X0 + ΔX
                       │
                       ↓
                     X̂
                       │
              ┌────────┴────────┐
              ↓                 ↓
       Reconstruction      Data Consistency
           Loss                 Loss
```

---

# 三十六、方法创新点可以最终归纳成四个

如果以后写论文，我建议把 contribution 控制在下面四点。

---

## Innovation 1 - Physics-consistent unified observation modeling

统一：

> real low-resolution acquisition  
> synthetic dense-grid sparse sampling

都映射到：

$$
(Y_{grid},M,X_0,U)
$$

这是整个框架的 foundation。

---

## Innovation 2 - Sampling-aware physics prior

不是直接 concat：

```text
sparse + mask + Wiener
```

而是：

$$
F_{prior}=G(M,C_s,U)\odot F_W
$$

实现：

> sampling-conditioned prior modulation。

---

## Innovation 3 - Anisotropic true 3D Wiener initialization

真正考虑：

$$
\sigma_{xy}\neq\sigma_z
$$

实现 physical PSF：

$$
H(f_x,f_y,f_z)
$$

并生成：

$$
X_0
$$

---

## Innovation 4 - Unified 2D/2.5D/3D residual reconstruction

同一个：

$$
f_\theta
$$

处理：

```text
T=1
T=3/5
T=8/16
```

并通过 target mask 实现：

> 统一 reconstruction formulation。

这比维护三个独立网络更有价值。

---

# 三十七、建议分成 P0 / P1 / P2

这是非常重要的，不要同时改所有东西。

---

# P0：必须先解决

### P0-1 Observation protocol

重新定义：

```text
real acquisition
vs
synthetic dense-grid masking
```

---

### P0-2 Unified observation builder

建立：

```python
build_observation()
```

---

### P0-3 真正 3D Wiener

删除目前：

```text
slice-wise Wiener
```

替换成：

```text
anisotropic 3D Wiener
```

---

### P0-4 删除 GT-dependent normalization

---

### P0-5 training / validation / test split 审计

确保：

```text
patient / specimen / stack
```

不会 leakage。

---

### P0-6 inference 与 dataset physics 统一

只保留一套 observation / physics implementation。

---

# P1：核心方法升级

### P1-1 Sampling-aware prior

```text
M
Cs
U
Wiener
```

共同生成 prior。

---

### P1-2 Residual reconstruction

$$
\hat X=X_0+\Delta X
$$

---

### P1-3 Data consistency loss

$$
L_{DC}
$$

---

### P1-4 Standard shifted-window mask

---

### P1-5 Unified 2D/2.5D/3D model

---

# P2：实验和论文增强

### P2-1 measured / missing metrics

### P2-2 zero-shot cross-domain

### P2-3 pollen acquisition transfer

### P2-4 statistical pairing correction

### P2-5 uncertainty / difficulty visualization

### P2-6 iterative physics refinement

---

# 三十八、建议的 ablation matrix

这个项目后续实验不要随意增加，而应围绕科学假设构建。

---

## Ablation A：Physics prior

```text
No prior
Bilinear
2D Wiener
3D Wiener
```

---

## Ablation B：Sampling awareness

```text
No mask
Mask
Mask + Confidence
Mask + Confidence + Uncertainty
```

---

## Ablation C：Reconstruction formulation

```text
Direct prediction
Residual prediction
```

---

## Ablation D：Data consistency

```text
No DC
LDC
```

---

## Ablation E：Dimensionality

```text
2D
2.5D
3D
Unified 2D/2.5D/3D
```

---

# 三十九、实验结果最重要的几个问题

最终不要只看：

```text
PSNR
SSIM
```

而应该回答：

### Q1

Wiener prior 是否真的提供帮助？

### Q2

mask information 是否帮助 missing-region reconstruction？

### Q3

uncertainty/confidence 是否进一步提高困难区域恢复能力？

### Q4

residual learning 是否比 direct reconstruction 更稳定？

### Q5

data consistency 是否提高 measurement fidelity？

### Q6

统一模型能否同时处理 2D / 2.5D / 3D？

### Q7

CIDC25 synthetic sparse training 是否能够迁移到真实 pollen acquisition？

这几个问题才构成完整故事。

---

# 四十、Codex 最推荐的实际实施顺序

不要让 Codex 一次性重构整个仓库。

建议按下面顺序执行。

---

## Phase 0：Audit only

Codex 只允许：

```text
read
trace
report
```

不修改代码。

输出：

```text
data flow graph
train flow
eval flow
inference flow
API mismatch list
```

---

## Phase 1：Physics layer

先完成：

```text
physics/psf.py
physics/wiener.py
physics/forward_model.py
```

然后写 unit tests：

```text
shape
FFT/IFT reconstruction
anisotropy
PSF response
numerical stability
```

---

## Phase 2：Observation protocol

建立：

```text
data/observation.py
```

先支持：

```text
CIDC25
Pollen
```

输出统一：

```text
y_grid
mask
warm_start
confidence
uncertainty
```

---

## Phase 3：Dataset refactor

所有 dataset：

```text
CIDC25
SparseStack
Pollen
```

全部调用：

```python
build_observation()
```

禁止 dataset 自己重新实现：

```text
mask
Wiener
interpolation
```

---

## Phase 4：Model

再实现：

```text
SamplingAwarePrior
ResidualDecoder
Unified3DBackbone
```

---

## Phase 5：Loss + Evaluation

加入：

```text
L_rec
L_DC
measured error
missing error
```

然后修复：

```text
paired statistics
stack-level inference
```

---

# 四十一、对 Codex 的硬性要求

建议在执行任务时明确规定：

### 1.

不要删除旧代码。

先移动或重构。

---

### 2.

每修改一个 module 都要保留 backward-compatible wrapper，直到新 pipeline 跑通。

---

### 3.

任何 physics function 都必须：

```text
unit test
```

---

### 4.

任何 dataset change 都必须输出：

```text
input shape
mask ratio
warm-start shape
GT shape
```

---

### 5.

第一阶段禁止同时改：

```text
dataset
model
loss
evaluation
```

否则很难定位 regression。

---

# 四十二、建议的最终 config

未来的 `options.py` 不应该再让用户选择大量彼此耦合的参数。

可以逐步整理成：

```yaml
observation:
  mode: synthetic_masking / real_acquisition
  sparse_axes: [z]
  sampling_factor: 2
  interpolation: bicubic

physics:
  psf:
    sigma_xy_um: ...
    sigma_z_um: ...
  wiener:
    enabled: true
    lambda: 0.01

prior:
  mask: true
  confidence: true
  uncertainty: true

model:
  architecture: unified_3d
  residual: true
  spatial_window: 8
  axial_mode: attention

loss:
  reconstruction:
    l1: 1.0
    ms_ssim: ...
  data_consistency:
    enabled: true
    weight: ...

task:
  target_mode: center / all
```

这样会比现在：

```text
task
model
sparse_mode
sparsity_factor
wiener
prior_fusion
dino
...
```

大量互相约束的 flags 更清晰。

---

# 四十三、最终应该形成一个非常清晰的数据流

## Real acquisition

```text
Dense object
     ↓
Microscope PSF
     ↓
Real sparse acquisition
     ↓
Y_low
     ↓
Geometric resampling
     ↓
Y_grid
     ↓
Mask + Confidence
     ↓
3D Wiener
     ↓
X0
     ↓
Unified Reconstruction Network
     ↓
ΔX
     ↓
X̂ = X0 + ΔX
     ↓
Forward acquisition model
     ↓
Ŷ_low
     ↓
Data consistency
```

---

## CIDC25

```text
Dense GT
     ↓
PSF + noise
     ↓
Synthetic sampling mask
     ↓
Y + M
     ↓
Interpolation / zero fill
     ↓
Y_grid
     ↓
3D Wiener
     ↓
X0
     ↓
Unified Reconstruction Network
     ↓
ΔX
     ↓
X̂
     ↓
Forward model
     ↓
Ŷ
     ↓
Data consistency
```

这两个 protocol 最终通过统一 interface 汇合。

---

# 四十四、当前仓库最值得保留的东西

并不是所有东西都需要推倒重来。

目前比较值得保留的包括：

- TIFF 16-bit loading
- CIDC25 volume patch extraction
- existing Swin-UNet encoder/decoder 基础代码
- existing evaluation infrastructure 的部分 metric
- pollen data preparation logic
- prior-fusion experiment script
- sliding-window inference 思路

例如：

`load_tiff_float32()` 当前对 16-bit TIFF 的处理是合理的，而且 stack TIFF 默认拒绝而不是错误地把 stack 当成单帧，是值得保留的。 

---

# 四十五、最终论文故事可以变成什么

如果按照这个方向完成，最终文章的方法故事就不再是：

> “We designed a Swin-Unet for sparse microscopy reconstruction.”

而会变成：

> **We propose a physics-guided sampling-aware reconstruction framework that unifies synthetic sparse-grid masking and real low-resolution acquisition within a common observation representation. An anisotropic 3-D Wiener initializer provides a physically informed reconstruction prior, while sampling confidence and physics-derived uncertainty modulate the prior features according to the local reliability of measurements. A unified anisotropy-aware 3-D residual network then reconstructs sparse observations across 2-D, 2.5-D, and 3-D settings, with measurement-domain consistency enforcing agreement with the actual acquisition process.**

这个故事的完整性明显高于当前版本。

---

# 四十六、我对项目优先级的最终判断

如果只允许你优先做几件事情，我建议严格按这个顺序：

**第一优先级**

> 修 observation protocol。

尤其明确：

```text
real acquisition + resampling
vs
synthetic dense-grid masking
```

---

**第二优先级**

> 真正实现 anisotropic 3D Wiener。

这是目前最明显、最容易验证、也最适合形成 physics contribution 的部分。当前代码实际上还是 slice-wise 2D Wiener。 

---

**第三优先级**

> 把 training / evaluation / inference 的 observation model 完全统一。

否则后面的任何性能提升都很难解释。

---

**第四优先级**

> Sampling-aware prior + residual reconstruction。

这是算法创新主体。

---

**第五优先级**

> Unified 2D/2.5D/3D model。

这是架构层面的提升。

---

**第六优先级**

> Data consistency + measured/missing metrics + rigorous statistical protocol。

这是把工程模型提升为完整科研方法的关键。

---

## 对应的 Codex 修改清单

我已经把上述方案整理成了一份可直接交给 Codex 使用的 Markdown 实施清单，其中把 **P0/P1/P2、目录重构、函数级任务、统一 observation protocol、3D Wiener、unified 3D model、loss、evaluation、inference 和阶段性实施顺序** 都列出来了：

[Protocol v3 修改清单与实施方案](sandbox:/mnt/data/Protocol_v3_Modification_Checklist.md)

这份清单建议作为 Codex 后续修改仓库时的主 specification，而不是让 Codex 一次性自由重构整个项目。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
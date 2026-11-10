# Phase 1 PR2 方法规格书：物理先验 + 双分支残差重建

> 对应任务书：`docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md` §2–§6。
> 代码：`spur_mpm/physics/prior.py`、`spur_mpm/models/dual_prior_reconstructor.py`、
> `spur_mpm/models/registry.py`（`P2_*`）。
> 测试：`spur_mpm/tests/test_physics_prior_contract.py`（29）、
> `spur_mpm/tests/test_dual_branch_contract.py`（37）。
> 状态：**代码与单测完成；真实数据实验 NOT_RUN**（见 `docs/PHASE1_PR2_ACCEPTANCE.md`）。

本文档写死的每条公式与限制，都至少对应一条自动测试。若某条在代码里不再成立，
那就是本文档错了，而不是"实现细节变了"。

---

## 0. 目标域：先声明预测的是什么，再谈物理

第一阶段预测的是显微镜**已经成像的密集观测图像域 `D`**，不是无光学模糊的潜在荧光 `X`。
理由是数据本身如此：真实 3PM 密集 Z-stack 已经包含该系统的光学 PSF 与噪声，
把它们当成 `X` 需要先有一个可信的 PSF 标定，而本轮没有。

因此本层任务的全部物理内容只有一项：**在跳行/跳层的测量几何下，如何把已知的观测
扩成一个合理的密集初值**。任何"恢复光学模糊之前的信息"的说法都超出本协议。

仿真稀疏协议：

```
D        密集真实观测（学到的映射的目标域）
M        采样 Mask，仅实测位置为 True
Y = M ⊙ D   盲观测（未测位置为 0，且构造时强制乘 M）
```

`Y` 与 `M` 是网络的唯一实测信息。干扰探针
（`test_the_item_builder_never_sees_the_dense_target`）把 `D` 在 `~M` 上全部改写、
保持 `M` 上不变，要求模型输入逐元素不变——包括归一化、统计量和任何软先验。

---

## 1. 四个算子，各自的定义与边界

| 记号 | 定义 | 代码 | 何时用 |
|---|---|---|---|
| `I0` | `Interp(Y, M)`：按 Mask 在所有缺失轴（先 Z、后横向）线性插值，**不跑 FFT** | `wiener.interp_linear` / `warm_start(mode="bilinear")` | 残差锚点、`P2_I0` 地板 |
| `W0` | `Wiener(I0; H, λ)`：**先填满所有缺口**，再对填好的体做正则化反卷积 | `wiener.wiener_fill_all` / `warm_start(mode="wiener_fill_all")` | `P2_W0`、`P2_DualW` 的物理候选 |
| `phys_delta` | `W0 - I0` | `prior.physics_prior` | 物理分支的显式差异通道 |
| `mask_aware_inverse` | `argmin_X ½‖M(HX−Y)‖² + λ/2‖X−I0‖²` | `prior.mask_aware_inverse` | **附加消融**，不取代 Wiener |

三条必须保持的区别（上一轮把它们混在一起了）：

1. **纯插值 ≠ 插值后 Wiener。** `I0` 不跑 FFT，`W0` 跑。两者在 identity PSF 下数值上
   完全相等（见 §2），所以在该情形下它们的差别为 0，不是"很小"。
2. **插值后 Wiener ≠ 残差重建。** 前两者零参数；残差网络是在 `I0` 上学习修正 `R`。
3. **残差重建 ≠ 双分支融合。** 单分支残差（`P2_ResidualI`）和双分支（`P2_DualI` /
   `P2_DualW`）容量相近但信息通路不同（§4、§6）。

### 1.1 为什么旧 `wiener_fill_first` 不够

旧的 `wiener_fill_first` 只把**横向**缺口填了再 FFT；Z 方向仍以零参与 FFT，
在 `k_z > 1` 的协议下等于把"未采样"当成"信号为 0"送进反卷积，产生振铃与暗带。
`wiener_fill_all` 先填**所有**缺失轴，再 FFT。旧函数语义保留不动，供旧 checkpoint
复算（`test_the_old_fill_first_mode_still_only_fills_lateral_holes`）。

### 1.2 为什么另写 `mask_aware_inverse` 而不是修旧函数

旧 `iterative_dc_inverse` 的正则项是 `+ λx`，即把 `x` 往 **0** 收缩；其 docstring 曾
写成"往插值先验收缩"。这是文档 bug，但数值后果真实：未测位置会被逐步压暗。
新算子把正则中心换成 `I0`，步长 `1/(1+λ)`，在 `H=I` 且无观测处不动点恰为 `I0`。
两条行为都有回归测试锁住（旧函数不许被"顺手修好"，否则旧权重语义漂移）。

---

## 2. PSF 来源：三种 H，三种许可

`PSFConfig` 保存 σ_xy、σ_z、体素间距、λ、边界处理和 padding。
`psf_source` 必须显式标注，非恒等 `H` 缺标签直接 `ValueError` 拒绝
（`test_an_unlabelled_non_identity_psf_is_refused`）。

| `psf_source` | 含义 | 允许的结论 | 禁止的说法 |
|---|---|---|---|
| `identity` | 数据已含该系统光学，不再额外汇卷 | 只能说"缺口填充"的收益 | 不能说 Wiener 有效/无效 |
| `calibrated_optical` | 经标定的真实光学 | 可讨论"在已标定光学下反卷积候选的贡献" | 不能与合成退化混表 |
| `synthetic_extra_degradation` | 额外人为加模糊做压力测试 | 只能作为独立分组，标为压力测试 | 不能进真实跳行主表 |
| `unlabelled_identity_assumed` | 记录口径：未标定、按恒等处理 | 同 `identity` | 同上 |

**本项目当前所有真实实验都在 `identity` 下**（真实 3PM 数据没有可信 PSF 标定），
因此：

- `W0 ≡ I0`，`phys_delta ≡ 0`，`u_physics ≡ 0`；
- `P2_W0` 与 `P2_I0` 数值上必然相等；
- `P2_DualW` 与 `P2_DualI` 的物理分支信息差为 0，二者只剩容量/结构的差别；
- **任何"我们已经验证 Wiener 反卷积带来增益"或"Wiener 无效"的表述都不成立**，
  两者都不是本队列能回答的问题。能回答的只有"残差参数化和第二编码器是否有用"。

这不是缺陷陈述，是口径陈述。要检验反卷积，必须先做 PSF 标定，那属于新的实验分组
（§2 表第三行），不能借主表结论。

---

## 3. 输入契约

`build_model_inputs(y, mask, psf, lam, warm_start_kind, ..., psf_source)` 是训练、
验证、测试、整卷推理**唯一**的输入构造入口，返回 `ModelInputs`：

```
y            [Z,H,W] float32   未采样位置强制为 0
mask         [Z,H,W] float32   仅实测为 1
interp       [Z,H,W] float32   I0（旧字段 x0 在 bilinear 下与之同源）
wiener       [Z,H,W] float32   W0
phys_delta   [Z,H,W] float32   W0 - I0
u_sampling   [Z,H,W] float32   采样几何不确定性
u_physics    [Z,H,W] float32   |W0-I0| 相对分歧，∈[0,1]，**不是误差概率**
valid_z      [Z] bool          False 处为物理体外的 padding，永不算测量
metadata     dict              psf_source / λ / 间距 / ky,kz / phys_delta_zero
```

`u_physics` 的语义必须与名字一致：它说"物理候选在这里把图像改动得多"，
是**去看一眼的理由**，不是**可以信任的理由**。领域里把它直接当方差用的做法，
本项目不采用。

`u_sampling` 与 `u_physics` 在 `identity` 下分别退化为纯几何量与恒零——
后者零是真实结论，不是 bug（`test_u_physics_is_zero_when_the_candidates_agree`）。

---

## 4. 双分支网络

`spur_mpm/models/dual_prior_reconstructor.py`。**不改 `SwinUNet2D`**，
旧 checkpoint 的权重映射不受影响。

```
obs   = stack(Y, M[, u_sampling])
phys  = stack(I0, W0, W0-I0[, M, u_physics])
f_i   = E_obs,i(obs) + g_i · P_i(E_phys,i(phys))        i = 尺度编号
R     = D(f_1..f_n)                                      decoder
D_raw = I0 + R                     （residual=True）
D_raw = R                          （residual=False，即 S0 语义）
D_hat = M·Y + (1-M)·D_raw          （consistency="hard"，可选）
```

六个设计决定，每条都有对应测试：

1. **残差锚在 `I0`，不是 `W0`。** `I0` 是永远有定义的算子；`identity` 下 `W0 ≡ I0`，
   锚在 `W0` 会让架构看起来依赖一个在本数据上不存在的反卷积。
2. **head 零初始化** ⇒ 第 0 步 `R ≡ 0`、`D_raw ≡ I0`，所有共享 decoder 的臂都从
   同一个"零学习地板等价物"出发（`test_the_residual_starts_at_exactly_zero`）。
   不是"接近 0"：随机 head 会让未训练的臂在某些窗口上凭运气赢过零参数地板，
   第一个验证数字就无法解释。
3. **融合投影零初始化 + 可选门控**（`sigmoid(0)=0.5`，与 `PriorFusionStem` 的约定一致）。
   第 0 步物理编码器只有投影拿到梯度，这是**刻意的**、并且被断言
   （`test_the_physics_branch_is_connected_and_learns_after_one_step`）：
   它把"布线错了"和"初始化故意这样"两件事分开——前者第 1 步仍然全零，后者不会。
4. **hard consistency 在模型内部**，不在三个调用点。`train.py` / `evaluate.py` /
   `inference.py` 因此不可能对同一个 checkpoint 做不同后处理
   （`test_train_and_eval_mode_predict_the_same_thing`，无 dropout、无 BN）。
5. **2.5D 是多 Z 层输入/输出，不是原生 3D 卷积。** 不许在文里写成 3D 网络。
6. **核心里没有任何血管成分。** 输出是一张图像；分割、骨架、拓扑置信度是
   `tasks/` 下的可插拔适配器。第二阶段神经元/钙信号不需要改这里。

---

## 5. 输出契约

```python
{
  "reconstruction_raw": Tensor,          # [B, t, H, W] 一致性处理前
  "reconstruction":     Tensor,          # 一致性处理后（off 时与 raw 相同）
  "reconstruction_uncertainty": None,    # 本轮未实现 u_model
  "aux": {"residual": Tensor, "interp": Tensor, ...}
}
```

`common/tensors.as_reconstruction(out)` 是唯一知道"张量还是字典"的地方，
训练/评估/推理/出图全部经由它。旧的 `SwinUNet2D` 继续返回裸张量，
两者都被同一个 helper 接受（`test_as_reconstruction_accepts_both_output_kinds`）。

`aux["gate"]` 只在 `collect_aux=True` 时出现，且已 detach。
默认关闭，因为把门控张量挂在每次前向的返回值上会让显存随 batch 累积。

---

## 6. 臂命名与对照

新旧名字必须可追溯（§7 表）。`P2_*` 与 `A0/A1/S*/SS*` **并行存在**，
旧臂的记录逐字段不变（`test_the_old_arms_are_untouched`）。

| 臂 | 结构 | 唯一目的 | 参数量 |
|---|---|---|---|
| `P2_I0` | 无网络，`I0` | 插值地板（`A0` 的复算） | 0 |
| `P2_W0` | 无网络，`W0` | 物理初值独立检验，需标定 `H` 才有意义 | 0 |
| `P2_Direct` | 单编码器，`D_raw=R` | `S0` 语义在新模型下 | 基准 |
| `P2_ResidualI` | 单编码器，`D_raw=I0+R`，看到 `I0` | 残差参数化本身买到什么 | 同基准+stem |
| `P2_DualI` | 双分支，物理分支只看到 `I0` | 容量/结构对照 | 与 DualW 差 <5% |
| `P2_DualW` | 双分支，物理分支看到 `I0,W0,Δ` | Wiener 候选是否贡献额外收益 | 同上 |
| `P2_DualWU` | 上式 + 门控（M, u_sampling, u_physics） | U 是否独立有效 | 条件启动 |

**控制变量（`phys_control`）**：`zeros` 保留架构与参数量、删掉信息；
`shuffle` 保留边缘分布、删掉空间对应。系数与参数量逐臂相等
（`test_the_physics_controls_keep_the_parameter_count`），且确实改变输出
（`test_the_physics_controls_actually_change_the_output`）。
"物理候选有用"必须在 `none > zeros` 且 `none > shuffle` 同时成立时才允许说；
只剩 `none > zeros` 时，能说的只是"第二编码器有用"。

`P2_DualI` 是 `P2_DualW` 的容量控制：两者只差物理 stem 的输入通道数
（1 vs 4），参数量差 <5%。若 `DualW ≈ DualI`，则 `W0` 没有贡献额外信息——
在 `identity` 下这是**预期结果**，不是失败。

---

## 7. 旧 → 新 对应表

| 旧臂 | 新臂 | 差别 | 旧结果可否直接比较 |
|---|---|---|---|
| `A0` | `P2_I0` | 纯插值，同一算子两个入口 | 可（应逐位相同） |
| `A1` | `P2_W0` | 旧为 `wiener_fill_first`（Z 未填），新为 `wiener_fill_all` | **不可**，Z 方向行为不同 |
| `S0` | `P2_Direct` | 同语义，新模型（残差框架下 residual=False） | 不可直接比较，架构不同 |
| `S1` | `P2_ResidualI` | 旧 `X0` 只是拼接通道；新为残差锚点 | 不可，参数化不同 |
| `S1/S2` 双分支 | `P2_DualI` / `P2_DualW` | 旧无物理分支 | 不可 |

"不可比较"的含义是：两张表的数字不能放进同一列排名，也不能用差值的正负
去推断新旧方法谁好。新旧可以在**同一张表里并列**，但必须分栏并注明代码代际。

---

## 8. 允许与禁止的结论表述

**允许**（有测试/或届时实验支撑时）：

- "在 identity PSF 下，`W0` 与 `I0` 数值上完全相等，因此本队列只能检验缺口填充。"
- "残差参数化（`P2_DualI` vs `P2_Direct`）在固定预算、固定 Mask 下改变了 X dB / Y Dice。"
- "物理候选在参数匹配对照下没有带来额外收益"——这是合法负结果，必须保留。
- "u_physics 在本数据上恒为零，门控臂因此退化为 `P2_DualW`。"

**禁止**：

- "Wiener 反卷积在本任务上有效/无效"——identity PSF 下无法辨识（§2）。
- "双分支带来了提升，所以物理先验有用"——先过 `zeros`/`shuffle` 对照（§6）。
- "我们的 3D 网络"——是 2.5D。
- "不确定性估计准确"——`u_*` 是相对特征，不是标定过的误差概率。
- 用任何 <1 dB PSNR 或 <0.03 Dice 的差值支撑结论（跨动物方差 ≈2.8 dB / 0.08 Dice）。
- 用 PSNR 支撑分层（深度）结论；分层只看 vessel Dice / clDice。

---

## 9. 本轮范围之外（明确不在本规格书里）

| 项目 | 计划任务 | 状态 |
|---|---|---|
| `P2_DualWUT`（拓扑/结构损失） | T3 | 未实现 |
| `P2_TopoConf`（`tasks/vessel/` 适配器） | T5 | 未实现 |
| `u_model`（深度集成 / MC dropout） | T5 | 未实现 |
| 任何 GPU 训练与真实数据结果 | T4 | NOT_RUN |

未运行的东西在任何图、表、PR 里都标 `NOT_RUN`，不标 `COMPLETE`、不标 `n/a`。
Fig. 2 的第四栏就是为此存在的（`spur_mpm/figures/framework/fig2_ablation_matrix.png`）。

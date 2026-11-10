# SPUR-MPM — Phase 1 Physics-prior Dual-branch Reconstruction Refactor

> 执行方案 v1.0 | 2026-11-10 | 面向 DeepSeek 实施 | 基线：`PUR-MPM` @ `ed060f0366c17e43f0329a913234d550b88b7682`
>
> **重要**：本文是 Draft PR 中的工程实施任务书，尚未实施所列模型代码或完成 GPU 实验验收。

## 0. 目标、范围与不可破坏的约束

**研究对象是通用多光子显微成像（MPM）稀疏重建，不是“血管专用重建网络”。** 同一核心模型未来应能迁移至神经元形态恢复、ROI 分割、钙活动趋势/事件的功能保真检验。第一阶段使用 Dryad 真实 3PM 脑微血管数据建立结构保真证据，血管骨架、分叉、连通性与拓扑可信度是**第一阶段评价或可选任务头**，不是核心重建网络的输入先验或不可替换输出。第二阶段引入新数据后，可更换任务评价器或下游头，而不重写稀疏采样、物理初始化、双分支融合主干。对 XYT、ZYX、TZYX 坚持按真实元数据显式区分；严禁将 T 当 Z。

研究主线保持**有监督**；本 PR 不启动自监督、DINOv3/LoRA，不改变现行 M2 固定测试动物队列，不修改旧实验的定义、配置和 checkpoint。所有新实验以新实验编号和新模型名运行。**负结果只要协议正确便是研究记录，必须保留。**

本轮先达到：
1. A0 为真正独立的线性插值基线，A1 为先在观测图像域填补缺失采样、再执行 PSF/噪声模型明确的 Wiener 运算，A0/A1 绝不暗中共用一个被改写的实现；
2. 真实观测 `Y` 与布尔 `M` 明确标记实测点；`I0=Interp(Y,M)` 与 `W0=Wiener(I0;H,λ)` 明确标记估计点；
3. 具备 `direct observation` 与 `physics-prior` 双分支、残差输出、可选数据一致性与可解释门控；
4. 拆解验证插值、Wiener、残差结构、双分支、不确定性、拓扑约束各自的增量贡献；
5. 先输出通用重建图和一般性不确定性，再在**第一阶段任务适配器**中输出“候选血管段/连接的置信度热力图”，不能只是背景上色的普通像素误差图；
6. 按安全清单归档、清理无用实验文件，**不删除执行正确但效果不佳的上一阶段结果**。

## 1. 现有仓库接口与待修复依据

已从 `PUR-MPM` 公共源码确认的现有入口：
- `spur_mpm/physics/wiener.py`：`warm_start()` 的 `bilinear`、`wiener`、`wiener_fill_first`、`iterdc`；当前 `wiener_deconv_3d` 自身要求未测位置先填充，但普通 `wiener` 非恒等 PSF 分支可先对零填充 Y 做 FFT。
- `spur_mpm/physics/psf.py`：默认 sigma 为零，恒等 PSF 下所谓 Wiener 与插值相同，不能称为反卷积增益。
- `spur_mpm/models/registry.py`：旧 A0/A1/S0/S1/S2/S1g/S2g 与 SS 系列必须保留原名及语义（旧 checkpoint 兼容），新增名称不得冲突。`SS1c` 是“互补观测划分双网络”，**不是**本方案的“观测分支+物理先验分支”，务必区分。
- `spur_mpm/data/observation.py`：训练/评估/推理共用的输入构建入口 `build_model_inputs()`；扩充此接口但保留向后兼容。
- `spur_mpm/data/datasets.py`：共享数据项构造、正确 Z 维度和固定中心窗口；核查 epoch 间 patch 种子是否更新，不能只靠猜测。
- `spur_mpm/models/swin_unet_2d.py`、`prior_fusion.py`：现有通道拼接和门控；需要增加独立架构而不改变旧模型权重结构。
- `spur_mpm/train.py`、`evaluate.py`、`inference.py`、`scripts/run_ladder.py`、`metrics/vessel.py`、`metrics/skeleton.py`、`scripts/structure_report.py`：数据路径、评估和复现入口。
- `runs/E021_ladder/`、`runs/E023_phase1/`、`docs/RESULTS.md`、`docs/PHASE1_AUDIT_RESPONSE.md`：现有正确执行但结果可能为负的历史基线或验收资料，禁止清除或覆盖。

需先检查 repo HEAD 是否与这些路径相符。如已变化，在 PR 中列出差异而非按过期假设生搬硬套。

## 2. 物理问题定义（先于改代码冻结）

先明确目标域：第一阶段默认预测显微镜**已成像的密集观测图像域** `D`，而不是凭空声称预测无光学模糊的潜在荧光 `X`。现有真实密集 Z-stack 本身包含光学 PSF 与噪声。仿真稀疏协议为 `Y=M⊙D`；A0 插值 `I0 = Interp(Y,M)`。不能为了制造 Wiener 收益，在这份已有光学模糊的 D 上再无依据地卷积一遍 Gaussian PSF，然后把结果混入原任务主表。

Wiener 候选 `W0=Wiener(I0;H,λ)` 的 H 来源必须记录：`identity`、`calibrated_optical` 或明确标为 `synthetic_extra_degradation`；后两者不允许混用结论。若预测目标为 D，则 W0 是用于恢复结构信息的**候选特征**，不应不经检验就作为 D 的绝对物理真值。若未来明确重建潜在 X，则单独定义 PSF 前向一致性 `M⊙H(X)` 并配套新评价协议。

针对真实跳行/跳层任务，至少实现并区分：
- `interp_linear`: 对已采样行/层进行线性插值，不执行 FFT、反卷积、神经网络。
- `wiener_fill_first`: 按当前真实 Mask **先填完所有稀疏轴的缺失位置**，然后在已声明的 H 和 λ 下执行正则化 Wiener；处理 2D-Y、3D-Y、3D-Z、3D-YZ，严禁零填充缺口直接参与 FFT。
- `mask_aware_inverse`（独立附加消融，不强行取代 Wiener）：求解 `min_X 0.5||M⊙(H X-Y)||² + λ/2||X-I0||²`，以插值先验为正则中心。若 H=I，测试它不会无条件把未观测像素向 0 拉；此算子对目标域的适配需明示。

`PSFConfig` 保存光学测量来源、XY/Z 尺度（µm）、λ、边界处理、FFT padding 及作用于图像的域。对于 2D Y 采用 2D 处理或清楚记录单层上下文；对于 Z/YZ 的 2.5D 处理保证只从真实稀疏窗口构造输入；检查边界层、`valid_z` 和 Mask 传播，绝不读取未采集的真实密集邻层作为网络输入。对于无校准 PSF 的真实主任务，既报告 `H=I` 的不可辨识结果，也可将有物理依据的 PSF 估计作为显式探索分支；不能把调 PSF 得到的最佳测试结果当作无偏结论。

## 3. 输入/输出契约

建议新增 `ReconstructionInputs`（或扩充 `ModelInputs`，旧字段仍可使用）：
```
y            : float32 [Z,H,W]   # 未采样位置为 0 的真实稀疏观测
mask         : bool [Z,H,W]      # 仅实测位置为 True（输入端可转 float）
interp       : float32 [Z,H,W]   # I0，仅来自 y、mask
wiener       : float32 [Z,H,W]   # W0，来自 I0、H、λ
phys_delta   : float32 [Z,H,W]   # W0-I0
u_sampling   : float32 [Z,H,W]   # 采样几何不确定性
u_physics    : float32 [Z,H,W]   # 物理候选分歧/可靠性特征，不能直接宣称为概率
valid_z      : bool [Z]         # 虚拟 padding 和有效深度
metadata     : dict             # H来源、λ、采样和体素间距、版本指纹
```
训练、验证、测试、整卷推理须共用一个输入构造函数；任何特征只能从 `Y,M` 及已冻结的标定参数计算。严格避免 dense target 信息泄漏（包括归一化、伪标签、阈值和特殊 sample ID 逻辑）。不需要在每次训练都提前导出巨大 X0 缓存；若启用缓存必须以数据/Mask/参数/代码哈希做 cache key。

通用输出接口：
```
{
  "reconstruction_raw": Tensor,    # 同输入空间的密集图像
  "reconstruction": Tensor,        # 可选观测一致性处理后
  "reconstruction_uncertainty": Tensor | None,
  "aux": {"gate_map": ..., "delta": ..., "physical_reliability": ...}
}
```
**血管掩膜、骨架、血管段/连接置信度**经 `tasks/vessel/` 或 `metrics/vessel/` 的可选任务适配器计算；不作为核心模型强制输出。后续 `tasks/neuron/` 与 `tasks/calcium/` 能在相同重建接口之上独立构建。旧网络仍可通过轻量适配器返回同类型字典，避免改变旧权重。

## 4. 新模型：观察分支 + 物理先验分支

建议新建 `spur_mpm/models/dual_prior_reconstructor.py`，不要修改旧 `SwinUNet2D` 以致旧 checkpoint 无法加载。

- **Observation encoder**：输入 `(Y,M)`，可选 `u_sampling`；强调实测位置和测量几何。
- **Physics encoder**：输入 `(I0,W0,W0-I0,M)`；学习来自物理估计的多尺度结构候选，而不是把每个 Wiener 增强结果视作真实。
- **Multi-scale fusion**：按特征层融合，支持无门控的简单加法/拼接控制；门控受 `M,u_sampling,u_physics` 调节，防止不可靠的 Wiener 伪影主导输出。提供 gate 可视化和数值范围测试；不要简单把高不确定性作为固定更低权重而导致困难区域完全不学。
- **Residual image decoder**：`D_raw=I0 + R_theta(F_obs,F_phys,M,U)`；明确残差相对 `I0` 而不是默认相对 `W0`。先跑无需 U 的网络，再分步加入 U。网络主头通用、不得卷入 vesselness/分割作为强制先验。
- **Observation consistency**：对当前以测量图像 D 为目标的模拟稀疏任务，提供 `hard` / `off` 统一配置，`D_hat=M*Y+(1-M)*D_raw`。评估与推理路径相同；所有方法报告一致的后处理策略。后续去噪或 latent-X 任务禁用盲目的硬替换，改为合适的噪声/光学一致性约束。
- **Parameter matching**：至少提供相同结构、相近参数量的物理候选置零/随机候选/仅插值的对照，不能把双分支更大模型的提升直接归因于 Wiener；记录 Params、MACs、纯推理时间和 CUDA 峰值显存。
- 2D-Y 与 3D-Y/Z/YZ 保留统一接口。2.5D 是多 Z 层输入/预测，而非原生 3D 卷积；不要夸大网络结构类型。

## 5. 不确定性与任务适配器：分两次实施

**第一步：通用重建不确定性（非血管专用）。**
- `u_sampling`：采样 Mask 几何/物理距离/局部密度，显式考虑 voxel Z/XY 间距；
- `u_physics`：`|W0-I0|`、PSF/λ 敏感性、候选不稳定性；它们是输入特征，**不是已经校准的真实误差概率**；
- 只有在对照成立后再实现 `u_model`（深度集成或 MC dropout，固定训练预算），评估不确定性与缺失区域误差关联、风险-覆盖率曲线，不能使用测试 GT 调节不确定性公式。

**第二步：第一阶段血管拓扑不确定性适配器（条件启动）。**
- `tasks/vessel/topology_confidence.py`（建议新增）仅对重建中感兴趣的候选血管区域输出结构可信度；原始全背景透明，禁止让广阔背景像素主导置信度评价。
- 使用阈值冻结的结构概率/骨架候选、多个输入候选或模型集成结果产生候选段；候选不能只来自最终预测，否则完全漏掉的血管无从告警。输出每条候选段、每个连接/分叉的结构一致性和不确定性；提供图结构 CSV/JSON 与 3D TIFF，以及 XY/XZ/YZ 冻结 ROI 的叠加热力图。
- 在参考结构有人工审阅的验证子集上校准，报告结构风险-覆盖率、Brier/ECE（若能形成可信二元标签），以及高风险连接的错误检出能力。没有校准证据时用“相对置信度/风险评分”，不要叫“真实存在概率”。
- 拟引入的 soft-clDice/结构损失必须有可追踪参考（人工标注或明确标注为自动生成的伪标签），严禁基于测试参考生成训练掩膜/阈值。
- 该任务适配器只在 Phase 1 血管数据启用。后续神经元 ROI、神经突起、钙活动保真应使用可替换任务评价器，不得让核心 reconstruction_head 输出 vessel mask。钙成像使用真实 T 轴数据，专测 ΔF/F、事件检出、时间相关性和动态保真，不把 2P/3P 结构 Z-stack 的层序当时间。

## 6. 实验矩阵与运行顺序

新名称建议 `P2_*`，与旧 A0/A1/S* **并行存在**；所有旧结果保持完全不变。

| 新臂 | 配置 | 唯一目的 | 首轮启用 |
|---|---|---|---|
| P2_I0 | Linear interpolation | 插值纯基线（复算验证） | 是 |
| P2_W0 | Fill-first Wiener | 物理初值独立检验，H 需要标记 | 是 |
| P2_Direct | 原 S0 语义，Y+M | 直接观测重建 | 是 |
| P2_ResidualI | I0 + residual network，Y/M/I0 | 残差参数化增益 | 是 |
| P2_DualI | 双分支，物理分支仅 I0、无 W 增强 | 网络容量/双分支结构对照 | 是 |
| P2_DualW | 双分支，I0+W0+物理差异 | Wiener 候选是否贡献额外收益 | 是（有有效 H 时） |
| P2_DualWU | P2_DualW + uncertainty-guided fusion | U 是否独立有效 | 条件启动 |
| P2_DualWUT | 上述 + task-aware topology objective | 结构损失能否改善真实微结构 | 后续条件启动 |
| P2_TopoConf | 阶段任务适配器，不是新图像重建模型 | 拓扑置信度可靠性 | 后续条件启动 |

若 PSF=identity，则 `P2_W0` 和 `P2_I0` 理应一致、`P2_DualW` 与无物理差异臂功能上退化；该情形只能验证架构或残差设计，**严禁写作“Wiener 已被验证无效/有效”。** 有校准 PSF 时，在真实光学图像域作为候选先验独立检验；合成额外退化时另立实验分组，不与真实跳行主表混合。

**实验一致性**：固定现有 M1/M3 训练、M1 验证、M2 单鼠测试，固定数据索引版本和归一化；相同协议、相同 mask 相位、物理像素间距、训练迭代预算、初始化随机种子、输入上下文（尤其 25% 的 Y/Z/YZ 同 7 层）、相同窗口 GT/ROI；比较不同网络容量时匹配参数量（建议偏差不超过 ±5%，超出需解释），至少 3 seeds 并报告均值与离散程度，逐卷指标也要列出而非只给 pooled 平均。

**先执行最小阶梯**：Y-only `y_k2/y_k4/y_k8` 检验物理和结构假设；以 `z_k4/yz_k2x2/y_k4_c7` 做 matched 25% 三轴实验；之后扩大完整 9 协议矩阵。失败也记录，绝不因为达不到理想结果而改变测试鼠或只出“最好一张图”。

**输出**：PSNRmiss/SSIM（含测量与非测量分开统计）、普通阈值 Dice/clDice、Dice★ 作为参考面积辅助指标但不能取代普通指标、严格物理距离骨架 F1、断裂/伪连接/分支检出、按血管直径分层、门控计分层比例和未剔除整卷指标。若评价算法会屏蔽“困难层”，必须明确屏蔽比例，并提供**不屏蔽主表**或独立可信人工审核，不得用选择性的层代表完整 3D。固定视角、GT/ROI/色标与支持率，图文结果逐项绑定 config/seed/checkpoint/commit SHA。

性能：分开报 `(a)` 网络前向纯 CUDA 耗时、`(b)` 插值+Wiener+模型+后处理算法计算耗时、`(c)` 峰值 GPU 显存、Params/MACs。**不含数据加载、图像写盘、可视化文件保存**。预热后 CUDA Event / 同步，统一 batch/shape/precision/device，多次重复提供中位数和 P95。当前将写盘前停止计算耗时**不视为错误**；只修正指标名称/计时边界使其符合目标。

## 7. 工程任务拆分与验收 Gate

### T0 — 创建安全开发分支并冻结历史（先做）
- 从 `PUR-MPM` 当前 HEAD 建新 `feature/phase1-dual-prior-20261110`；记录 commit SHA、环境锁、数据与 index hash；设置 PR 为 Draft，**不要直接提交 PUR-MPM**。
- 快照 `docs/RESULTS.md`、`docs/PHASE1_AUDIT_RESPONSE.md`、`runs/E021_ladder/`、`runs/E023_phase1/` 的现有索引与所有有效负结果。历史文件保持只读，旧分支 `PSRAS` 等不得改动。建立 `docs/archive/ARCHIVE_MANIFEST_20261110.md`。
- 列出仓库实际现存目录，不要假设所有路径都存在。仅编写清理清单，执行前发出 dry-run 报告。
**Gate 0**：能重跑旧单测；有 HEAD SHA 和归档清单；无删除正确实验。

### T1 — 统一物理初始化语义、修复非恒等 PSF 和 z/yz 路径
- 修改 `physics/wiener.py`、`physics/psf.py`（尽量新增函数，不改变旧模式含义）及 `data/observation.py`，为新实验显式导出 `I0/W0/phys_delta`；old `wiener`/`wiener_fill_first` 语义保留供旧权重复算。
- Fill-first 在 **所有缺失维度** 插值，再进行 FFT；必要时补齐 2D/3D 的非周期边界 padding、实测点一致性诊断，捕捉数值发散、非有限值、PSF 参数不合法、λ<=0。
- `iterative_dc_inverse()` 旧注释和 `+λx` 目标不匹配“向插值先验收缩”；新版本另写正则目标 `+λ(x-I0)`，不暗改旧函数。
- 区分噪声模型与目标域，测试物理 σ、Z/XY 间距、图像轴次；必须检查 z-only、yz-only、边界和无可用观测的窗口能安全失败，不可静默读 GT。
**Gate 1**：identity 下 `W0 == I0`；非零 H 下合成标准 phantom 反卷积数值有限且对合理噪声及 λ 稳定；填洞时不使用任何真实缺失值；相同输入、H/λ 各分支一致；PSF 未标定时实验声明与日志正确。

### T2 — 模型与训练器（最小残差→双分支）
- 新增 `models/dual_prior_reconstructor.py`；在 `models/registry.py` 注册 `P2_*`，**禁止修改原 A*/S* 对应类和旧配置含义**。
- 数据输入通过 `build_model_inputs` 构建，统一 train/eval/inference；`train.py` 不再隐含假设所有模型只返回 Tensor，适配结果字典或轻量 wrapper。
- 严格先跑 `P2_ResidualI` 再跑 `P2_DualI` 和 `P2_DualW`；门控与 U 延后。无 GT 的推理正常，hard consistency 按统一开关执行。保证推理和评价同配置指纹。
- 检查/修复 Dataset 按 `(seed,epoch,index,worker)` 生成训练 patch 的随机机制，确保多 worker 可复现、不同 epoch 覆盖不同 patch；旧 checkpoint 的输入协议保持不变。
**Gate 2**：2D-Y、2.5D-Z/YZ 输入/输出尺寸、Z 边界、Mask 和梯度传播；无 GT 泄漏；`P2_DualW` 在 identity H 下物理差异归零且结果归因不虚报。

### T3 — 训练目标和任务解耦
- 先用图像域 `L_rec`（建议 L1/Charbonnier+缺失区加权）并保持公平；可选 `L_obs` 但不能与 hard consistency 冗余；所有损失明确标记作用域和标签来源。
- 将结构损失和拓扑置信度放在 `tasks/vessel/` 或清晰可开关的结构插件中，不得在核心重建网络强制引用 vessel mask。只有训练数据上有合法参考/可信伪标签时才启用结构损失，并报告伪标签规则和人工核查范围。
**Gate 3**：关闭 task-specific loss 时图像重建模型仍正常；未来神经元/钙验证可无血管模块运行。

### T4 — 评价、可视化及消融报告
- 保留当前整卷、固定 ViewSpec 和一致性报告机制；新增分支对比固定 ROI/GT/色标、物理间距、掩膜覆盖率；不得只挑选表现好的单卷。
- 修正/补充被层门控掩盖的结构评价；提供 3D 骨架连通、宽/细血管分层、ImageJ/Fiji 人工抽检清单；保留所有真正执行正确的负结果。
- 所有评价 `summary.json` 包含源码 SHA、数据哈希、实验配置/seed、PSF 来源及 λ、checkpoint 路径及 hash、实际采样比例、GT/crop/view hash、损失与指标版本。
**Gate 4**：结果可从冻结 config、checkpoint、数据索引复算；出现 `no_checkpoint` 等错误则整体评估失败，不允许被静默标记 successful。

### T5 — 不确定性（仅在 T1–T4 有基线后）
- 增加 `U_sampling`, `U_physics` 可解释特征，比较“有/无 U”、物理候选替换和参数匹配对照；不以测试 GT 构造 U。
- 血管 topology confidence 是可选 `tasks/vessel` adapter；输出候选血管段、节点、置信度/相对风险以及透明背景热力图；不要只展示网络判为血管的正例而忽略假阴性风险。
**Gate 5**：风险分数与独立核验结构错误呈正相关；不能校准则明确标为相对风险，不称概率。不以单张漂亮热图替代定量评价。

## 8. 必须新增的自动化测试清单

建议新增 `spur_mpm/tests/test_physics_prior_contract.py`、`test_dual_branch_contract.py`、`test_topology_adapter.py`、`test_archive_safety.py`，加一个 `scripts/audit_runs.py`（均为**待新增**路径）。

1. `A0=Interp(Y,M)`；identity PSF 下 `W0=I0` 数值容差内完全一致，不允许偷偷从外部读 dense target；非零 PSF 在明确的合成 H 下不同、无 NaN/Inf。
2. 2D-Y、3D-Y、3D-Z、3D-YZ 的各稀疏点数/相位与物理位置正确；所有已测点保留、未测点输入 0；填洞算法不跨动物或 FOV、Z 边界与 `valid_z` 正确。
3. Mask 探针测试：任意改动未测的密集 GT 值，固定 `Y,M` 后，I0/W0/所有模型输入必须不变。
4. 稀疏采样 Z 模式下窗口不访问未采样的相邻密集层；context=1 不可运行 z_k4；与旧既有测试一同通过。
5. 新残差网络初始化可令 `R≈0`、输出≈I0；双分支反向传播到两个 encoder；U/Gate 可关且相应控制分支公平。
6. hard consistency `D_hat[M] == Y[M]` 且 eval/inference 一致；noisy mode 不能强制无条件硬覆盖。
7. `H=I` 情形的新物理差异为 0，PR 报告不得显示声称有“Wiener 反卷积增益”的自动图题。
8. 数据集同 seed 跨进程可复现；不同 epoch 实际 patch 改变且不与测试动物混合。
9. 评价回归：Dice★ 显式单列辅助；门控层覆盖率可计算；只有完整有效卷才可标为“full-volume”；不同采样协议固定参考/ROI/显示色标。
10. Topology confidence: 候选段需能覆盖集成中不稳定/缺失连接；背景透明；输出图结构与体素热图坐标对齐，校准只使用验证集。
11. 计算性能计时不含磁盘读写；CUDA 同步/预热一致，峰值显存测量重置；无 GPU 时报告 CPU 限制或跳过 GPU 基准而非伪造。
12. 清理工具默认 dry-run；不允许删除白名单内历史正确实验、数据和 checkpoint；拒绝 `../`、符号链接跳出工程根目录及未授权通配路径。

## 9. 归档和清理策略（严格执行）

### 9.1 分类与处置

| 类别 | 例子 | 处置 |
|---|---|---|
| A 正确执行、科学结果较好 | 通过全部协议验收的历史实验、可复现结论 | **保留**：完整配置/结果/引用位置；大 checkpoint 可在用户选定外部存储归档并保留可校验索引 |
| B 正确执行、科学结果不佳 | E021 有监督消融的真实负结果、已完成的正确 baseline | **必须保留**：它们是证伪和消融的重要证据，不能因失败删掉 |
| C 执行无效/协议错误 | 使用错误 checkpoint、mask 与 target 不匹配、Z 轴错误、数据泄漏、重评映射错误的中间文件 | **标记 INVALID 并归档少量诊断证据**：保留失败原因、日志摘要、hash、正确替代运行 ID，确认无引用后可删除大体积错误产物；不能当有效结果发表 |
| D smoke/临时产物 | smoke 测试的随机权重、一次性可再生图、临时缓存、`__pycache__`/`.pytest_cache`、残缺上传 | **可清理**，仅在路径白名单且先 dry-run 确认未被报告或代码引用后；测试脚本本身应保留 |
| E 原始科学数据、用户手工标注、有效权重 | Dryad 原始、索引/标定、手工 Fiji 标注、旧 PSRAS 分支、未备份 checkpoint | **严格保护，不纳入自动删除** |

### 9.2 安全执行步骤（命令范例是流程要求，非自动删除脚本）

1. **只盘点、不删除**：收集 `git status --short`、`git ls-files runs`、`du`、报告引用/脚本读写引用、每个 run 的配置、状态和 checkpoint hash。把当前树和历史结果生成 `docs/archive/ARCHIVE_MANIFEST_20261110.md` 与 `cleanup_candidates.json`（建议字段 `path,category,reason,run_id,referenced_by,sha256,bytes,action,archive_destination`）。
2. 建立 protectlist：`data_local/`（原始和可复现数据）、`runs/E021_ladder/`、`runs/E023_phase1/` 已有有效记录、`docs/RESULTS.md`、`docs/PHASE1_AUDIT_RESPONSE.md`、所有论文引用的图、旧实验 checkpoint、当前 git 中的受版本管理结果快照、标注与基线源码。一旦引用链指向候选文件，先修复索引而非删除。
3. **dry-run** 输出完整路径、大小、行动、删除理由、恢复方式，建议先单独审阅；清理工具 `--apply --allowlist <reviewed-manifest>` 才能执行。遇到路径越界、符号链接、受保护路径、无存储校验、归档失败立即停止。
4. 对 C 类（协议错误）先记录为何无效与正确替代 run，保留最小日志及配置，确保可审计后**仅删除冗余大文件**；对 D 类 smoke 删除临时产物但不删测试源码。受 git 跟踪的二进制/结果必须通过显式 `git rm` 的变更在 PR 中可审阅；未跟踪临时文件单独按 allowlist 删除。
5. 归档大文件不强行加入 git：如有需要，使用用户指定的长期存储（不是自行假定某个路径或账号），写存储位置、checksum、恢复命令。Git 提交小体积 Markdown/JSON manifest 和可重建脚本。未确认备份成功前一律不删除有效 checkpoint。
6. 将清理前后文件计数、磁盘体积、候选逐项决策、保留的负结果列表、恢复验证和 `git diff --stat` 放进 `docs/archive/CLEANUP_REPORT_20261110.md`。不允许 `rm -rf runs/`、`git clean -fdx`、批量通配删除历史文件、直接 force-push PUR-MPM。

### 9.3 针对上一轮具体记录

- `E021_ladder`：旧实验**执行正确**、一些消融科学上不支持假设；保留完整配置、评估报告、可视化以及可以定位的 checkpoint 或其归档 manifest。负结果应出现在新 PR 的对照表。
- `E023_phase1`：正确执行的审计与重评分日志保留。曾有 `no_checkpoint` 原因的错误中间评分输出，可以在 `INVALID_RUNS.md` 中注明被成功补跑所取代后清除冗余大产物；不能删除补跑成功的有效记录。
- smoke：只删除试验性 run/随机 checkpoint/临时图等产物，不删除 `test_*` 及未来回归测试脚本。
- `runs/` 有被 force-add 到 git 的只读结果快照；清理前用 `git ls-files` 区分 tracked 与 ignored，不能因 `.gitignore` 忽略而误以为一定不是已跟踪文件。

## 10. PR 交付物与 Review Gate

交付 PR 草稿标题：`[Draft] Phase 1: physics-prior dual-branch MPM reconstruction + safe experiment archival`。**Base: `PUR-MPM`，Head: `feature/phase1-dual-prior-20261110`**；PR 初始为 Draft，不要提前合并。

PR 必须包含：
- 代码：前述物理输入模块、新实验模型、兼容训练/推理接口、可选结构适配器、安全归档工具与自动测试；
- 文档：`docs/PHASE1_PR2_METHOD_SPEC.md`（公式、目标域与不同 H 的适用范围）、`docs/PHASE1_PR2_RUNBOOK.md`（可执行命令和实验矩阵）、`docs/PHASE1_PR2_ACCEPTANCE.md`（逐条测试/证据/未完成清单），以及归档和清理报告；
- 结果：新 `runs/E024_dual_prior/`（编号若被占用则选择下一个且记录映射）的机器可读 manifest、配置 hash、逐卷指标、参数/时间/显存、固定视角图与失败例；不得覆盖 E021/E023；
- 清晰的旧/新模型名映射；历史结果的完整保留表；真实实验未运行时状态 MUST 为 `NOT_RUN`，严禁把静态自测写成完成正式验收。

每一批提交按 `T0 freeze+audit` → `T1 physics` → `T2 residual/dual` → `T3 tests+metrics` → `T4 GPU experiments` → `T5 uncertainty/task adapters (optional)` → `T6 safe cleanup` 拆分，便于逐提交审核。科学验收只在代码核验、单测与真实固定队列实验全部可回查且关键问题关闭后才允许由 Draft 改为 Ready；任何假设不成立都要如实报告，不需要为了指标提升更改测试数据或清理负结果。

## 11. 给 DeepSeek 的明确执行指令

**立即实施 T0 → T1 → T2 的最小版本，并提交 Draft PR。** 在 PR 中按每个任务维护 `TODO / CODE_DONE / UNIT_TESTED / GPU_VALIDATED / ACCEPTED / BLOCKED` 状态、相关文件及精确日志/提交链接。不要一次性扩展自监督、神经元任务、完整拓扑置信度网络；后者应先建立可替换任务接口并只做小范围原型，待 Wiener 双分支在真实缺失区域和结构指标上验证后追加。必须先跑迁移后旧基线自检与非恒等 PSF phantom 数值测试，再开始新的 GPU 训练。清理先 dry-run 出清单，且**任何科学协议正确、但结果不支持假设的实验必须保留**。
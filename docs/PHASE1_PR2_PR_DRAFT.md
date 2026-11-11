# [Draft] Phase 1: Physics-prior dual-branch MPM reconstruction + safe experiment archival

**Base:** `PUR-MPM`  
**Head:** `feature/phase1-dual-prior-20261110`  
**状态：** Draft。T0/T1/T2 的代码与单测已完成（390 项全绿）；**真实数据实验进行中（首次）**
（四张 GPU 被其他用户占用）。未完成真实训练验收前不得合并。

> 本文件即 PR 描述正文；下文链接均为**仓库根目录相对路径**（GitHub 上可直接点开）。

| 任务 | 状态 | 证据 |
|---|---|---|
| T0 冻结 + 归档审计 | `ACCEPTED`（未删除任何文件） | `docs/archive/ARCHIVE_MANIFEST_20261110.md`、`CLEANUP_REPORT_20261110.md` |
| T1 物理语义分离 | `UNIT_TESTED` | `test_physics_prior_contract.py`（30）、`physics/prior.py` |
| T2 双分支残差模型 | `UNIT_TESTED` | `test_dual_branch_contract.py`（38）、`models/dual_prior_reconstructor.py` |
| T3 结构/拓扑损失 | `TODO` | — |
| T4 真实实验（E024） | `RUNNING`（首次真实训练；结果落地后回填） | 脚本 `spur_mpm/experiments/run_e024.sh`，说明 `docs/PHASE1_PR2_RUNBOOK.md` |
| T5 不确定性与任务适配器 | `TODO`（`U_sampling/U_physics` 特征已存在） | — |
| T6 清理执行 | `TODO`（只有 dry-run 清单） | — |

逐条清单：[`docs/PHASE1_PR2_ACCEPTANCE.md`](docs/PHASE1_PR2_ACCEPTANCE.md)；
方法口径：[`docs/PHASE1_PR2_METHOD_SPEC.md`](docs/PHASE1_PR2_METHOD_SPEC.md)；
可执行命令：[`docs/PHASE1_PR2_RUNBOOK.md`](docs/PHASE1_PR2_RUNBOOK.md)。

## 背景 / Research positioning

SPUR-MPM 的研究目标是**多光子显微成像的通用稀疏重建**，而非血管专用图像增强。第一阶段采用真实 3PM 微血管 ZYX 进行形态和拓扑保真验证；后续第二阶段将扩展到神经元结构、ROI 与钙功能信号保真。核心模型、稀疏 Mask、Wiener 初始化和不确定性均需具有跨生物结构和功能数据的可复用性。血管骨架与拓扑置信度属于第一阶段的可插拔任务头或评价器。

上一轮 E021/E023 已提供**执行正确但未能证明 Wiener/U/Gate 独立贡献**的有效结果；不能因结论不理想就删除。当前 identity PSF 使 Wiener 退化为插值，非恒等路径存在先对零填充观测 FFT 的风险；S1 的 X0 只是拼接通道，而非残差式初始化。因此需要建立新版本方法与独立消融，不覆盖旧实验语义。

## 变更范围 / Required implementation

- [x] **T0 冻结基线并审计产物**：从 `PUR-MPM` 建分支，记录 SHA、数据/配置/权重索引；旧 A0/A1/S0/S1/S2/S1g/S2g、旧 checkpoint、E021/E023 和报告只读保留。新增 archive manifest 和 cleanup dry-run 清单。
- [x] **T1 物理初始化**（单测级；真实数据只有地板冒烟）：明确 A0 纯线性插值 `I0` 与 A1 `W0=Wiener(Interp(Y,M);H,λ)`；补齐 Y/Z/YZ 的填洞顺序、PSF 来源、物理间距和数值稳定测试。identity 下必须检测到 W0=I0；非恒等 H 不允许对零填充孔直接 FFT。可选新 `mask_aware_inverse` 必须以 I0 为正则先验，不可暗改旧函数语义。
- [x] **T2 双分支残差网络**（单测级；训练 `NOT_RUN`）：Observation branch 接受 Y/M；Physics branch 接受 I0/W0/W0-I0/M；特征融合输出 `D_raw=I0+R`；可选 hard observed-point consistency 且 train/eval/inference 统一；新命名 `P2_*`，旧 registry 及 checkpoint 保持兼容。
- [ ] **T3 训练目标与任务解耦**——`TODO`：图像域 `L_rec`（Charbonnier + 缺失区加权）尚未启用，本轮沿用既有 `ReconstructionLoss`（L1 + MS-SSIM）以与旧臂同口径；结构/拓扑损失排在 Gate 2 通过之后，且不得让核心网络强制依赖 vessel mask。
- [ ] **T4 评价、可视化及消融（E024）**——臂已注册、脚本已备，**执行 `NOT_RUN`**：P2_I0、P2_W0、P2_Direct、P2_ResidualI、P2_DualI（容量控制）、P2_DualW；如果物理候选在非恒等 PSF 下有效，再测 P2_DualWU；不通过就如实记录，不启动更复杂网络。评价链路已适配新输出契约：固定 M2 测试动物、固定 GT/ROI/Mask/7-slice matched context、物理间距、训练预算；报告缺失像素 PSNR、普通/等面积结构指标、真实 3D 骨架及错误连接、不同血管尺度、有效层覆盖率；所有图有视角和 config/hash。性能统计网络纯推理和算法计算、CUDA 峰值显存、Params/MACs，不含磁盘读取与写盘。
- [ ] **T5 不确定性**——`U_sampling/U_physics` 已作为输入特征存在（**不是校准概率**），`u_model`、`tasks/vessel/` 拓扑置信适配器未实现：将候选血管段/分叉/连接转为拓扑置信热图（背景透明）并用独立核验校准。不得将 vessel mask 作为核心模型永久依赖。第二阶段神经元和钙信号有独立适配器，保持 T/Z 语义。
- [x] **T6 安全归档与清理**——只出清单，**本轮未删除任何东西**：先报告 `git ls-files`、路径引用、大小和 SHA256。正确执行但结果不佳的 E021/E023/旧基线**必须保留**。协议错误或 `no_checkpoint` 错误临时产物先留无效记录/替代 run 关联，确认可恢复后只删冗余大文件；smoke 只删临时 run/缓存/随机权重，不删测试代码。清理脚本默认 dry-run，执行必须 reviewed allowlist；不允许 `rm -rf runs/` 或 `git clean -fdx`。

## 测试与验收证据 / Proof required

- [x] `python -m pytest spur_mpm/tests -o addopts="" -q` 通过（390 项，`EXIT=0`），日志 `docs/evidence/pytest_20261111_T0_T2_guard.log`；新增 physics / dual-branch / archive safety / PSF estimate / run-guard 五组单测。
- [x] Mask 洞值探针：修改未测 GT 不改变任何模型输入；Z 边界不读取未采样密集邻层。
- [x] `H=I => W0 == I0`（逐位，不是近似；真实数据上一并验证）；非恒等 H 先填洞后反卷积、无 NaN/Inf。λ/边界稳定性做成 phantom 扫描的部分仍 `TODO`。
- [x] 残差模型初始化 `R ≡ 0`；两个分支第 1 步都有梯度；hard consistency 仅在含 `y/mask` 的臂上可用。
- [x] 旧 E021/E023 原样索引；新 run 编号/命名规则在 RUNBOOK §6 定义（尚无实际 run）。
- [x] 所有未跑项标记 `NOT_RUN`（含 Fig. 2 新增栏）；2D-Y/3D-Y/Z/YZ 指标待 T4 执行。
- [x] 已提交 `ARCHIVE_MANIFEST_20261110.md`、`CLEANUP_REPORT_20261110.md`，逐项列出决策与恢复方式。
- [x] **PSF 来源已查证并自行估计**：源数据**不含**标定（README 指向正文 Methods、TIFF 头只有 `spacing`、spec 无 NA）。两个独立估计器都指向 identity——算术上 1340 nm 三光子的 σ 只有 0.16–0.21 体素（横向）/ 0.11–0.22（轴向）；经验上用被抽掉的真实行做留出预测，两轴最优宽度均为 σ=0。新增 `estimated_from_data` 标签，与 `calibrated_optical` 刻意不同名。见 [`docs/PHASE1_PR2_PSF_ESTIMATE.md`](docs/PHASE1_PR2_PSF_ESTIMATE.md)、`docs/evidence/psf_estimate_20261110/`。
- [x] **归因链补齐**：`psf_source` 已贯穿 config→dataset→prior 并写进 `summary.json` 的 `physics` 块；`phys_control`/`consistency` 已进入 `build_model` 与 `config_hash`。此前三条都不通——带标签的非恒等 PSF 会在第一个 batch 被拒，`--set model.phys_control=zeros` 会用头条 run 的名字训对照组，且两次 run 哈希相同。三处均有单测。

## 本 PR 暂不做 / Out of scope

自监督/DINOv3、完整神经元/钙功能实验、未经数据校准的“PSF 提升”强结论、用测试 GT 优化拓扑置信度、删除有效负结果或直接覆写 `PUR-MPM`。Topology confidence 完整任务头和结构损失可在核心方法通过 Gate 后作为 PR 后续独立提交。

## 提交安排

建议分 commits：`baseline freeze & archive audit` → `physics prior refactor & tests` → `residual/dual-branch & tests` → `evaluations & benchmark` → `GPU results & method analysis` → `uncertainty/task adapter (conditional)` → `safe cleanup report`。

**验收人关注：**是否真正证明 Wiener 物理候选在控制网络容量后的独立贡献；是否在完整三维结构而非仅 PSNR/Dice★ 上改善；是否保留科学负结果且不对历史结果实施破坏性清理。

详细实施任务书：[`docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md`](docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md)；开发基线 `ed060f0366c17e43f0329a913234d550b88b7682`。

本分支 commit 序列（基线 `ed060f0`）：

| commit | 内容 |
|---|---|
| `367e21c` | T0：冻结基线、审计 `runs/`（未删除任何文件） |
| `31c8434` | T1：`I0` 与 `W0` 分离，FFT 前填满所有缺失轴 |
| `957a7df` | T2：双分支重建器、`P2_*` 臂族、输出契约 |
| `06f0662` | 归因链补齐 + 方法书/执行手册/验收清单 + 地板冒烟证据 + `run_e024.sh` |

本次提交包含 T0–T2 的代码、单测、归档报告与三份文档。GPU 实验尚未执行，所有真实数据结果为 `NOT_RUN`，理由与证据见 `docs/PHASE1_PR2_ACCEPTANCE.md`。
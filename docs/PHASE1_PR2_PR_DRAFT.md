# [Draft] Phase 1: Physics-prior dual-branch MPM reconstruction + safe experiment archival

**Base:** `PUR-MPM`  
**Head:** `feature/phase1-dual-prior-20261110`（执行时若命名冲突请记录实际分支）  
**状态：** Draft（任务文档已提交；实施代码、真实训练和验收未开始），未完成真实训练验收前不得合并。

## 背景 / Research positioning

SPUR-MPM 的研究目标是**多光子显微成像的通用稀疏重建**，而非血管专用图像增强。第一阶段采用真实 3PM 微血管 ZYX 进行形态和拓扑保真验证；后续第二阶段将扩展到神经元结构、ROI 与钙功能信号保真。核心模型、稀疏 Mask、Wiener 初始化和不确定性均需具有跨生物结构和功能数据的可复用性。血管骨架与拓扑置信度属于第一阶段的可插拔任务头或评价器。

上一轮 E021/E023 已提供**执行正确但未能证明 Wiener/U/Gate 独立贡献**的有效结果；不能因结论不理想就删除。当前 identity PSF 使 Wiener 退化为插值，非恒等路径存在先对零填充观测 FFT 的风险；S1 的 X0 只是拼接通道，而非残差式初始化。因此需要建立新版本方法与独立消融，不覆盖旧实验语义。

## 变更范围 / Required implementation

- [ ] **T0 冻结基线并审计产物**：从 `PUR-MPM` 建分支，记录 SHA、数据/配置/权重索引；旧 A0/A1/S0/S1/S2/S1g/S2g、旧 checkpoint、E021/E023 和报告只读保留。新增 archive manifest 和 cleanup dry-run 清单。
- [ ] **T1 物理初始化**：明确 A0 纯线性插值 `I0` 与 A1 `W0=Wiener(Interp(Y,M);H,λ)`；补齐 Y/Z/YZ 的填洞顺序、PSF 来源、物理间距和数值稳定测试。identity 下必须检测到 W0=I0；非恒等 H 不允许对零填充孔直接 FFT。可选新 `mask_aware_inverse` 必须以 I0 为正则先验，不可暗改旧函数语义。
- [ ] **T2 双分支残差网络**：Observation branch 接受 Y/M；Physics branch 接受 I0/W0/W0-I0/M；特征融合输出 `D_raw=I0+R`；可选 hard observed-point consistency 且 train/eval/inference 统一；新命名 `P2_*`，旧 registry 及 checkpoint 保持兼容。
- [ ] **T3 最小消融**：P2_I0、P2_W0、P2_Direct、P2_ResidualI、P2_DualI（容量控制）、P2_DualW；如果物理候选在非恒等 PSF 下有效，再测 P2_DualWU；不通过就如实记录，不启动更复杂网络。
- [ ] **T4 不确定性与结构任务解耦**：先实现通用 `U_sampling/U_physics` 与重建风险输出；后续第一阶段 `tasks/vessel/` 将候选血管段/分叉/连接转为拓扑置信热图（背景透明）并用独立核验校准。不得将 vessel mask 作为核心模型永久依赖。第二阶段神经元和钙信号有独立适配器，保持 T/Z 语义。
- [ ] **T5 评价与性能**：固定 M2 测试动物、固定 GT/ROI/Mask/7-slice matched context、物理间距、训练预算；报告缺失像素 PSNR、普通/等面积结构指标、真实 3D 骨架及错误连接、不同血管尺度、有效层覆盖率；所有图有视角和 config/hash。性能统计网络纯推理和算法计算、CUDA 峰值显存、Params/MACs，不含磁盘读取与写盘。
- [ ] **T6 安全归档与清理**：先报告 `git ls-files`、路径引用、大小和 SHA256。正确执行但结果不佳的 E021/E023/旧基线**必须保留**。协议错误或 `no_checkpoint` 错误临时产物先留无效记录/替代 run 关联，确认可恢复后只删冗余大文件；smoke 只删临时 run/缓存/随机权重，不删测试代码。清理脚本默认 dry-run，执行必须 reviewed allowlist；不允许 `rm -rf runs/` 或 `git clean -fdx`。

## 测试与验收证据 / Proof required

- [ ] `pytest spur_mpm/tests -q` 的回归测试通过并提交真实日志；新增 physics/dualbranch/archive safety 单测。
- [ ] Mask 洞值探针：修改未测 GT 不改变任何模型输入；Z 边界不读取未采样密集邻层。
- [ ] `H=I => W0≈I0`；非恒等 H 先填洞后反卷积、无 NaN/Inf、λ/边界稳定。
- [ ] 残差模型零修正输出≈I0，两个分支有梯度，hard consistency 只在适合的目标域启用。
- [ ] 旧 E021/E023 的有效结果与配置能原样索引；新 run 唯一编号、可追踪 commit/config/checkpoint/seed/PSF/GT/ViewSpec。
- [ ] 所有 2D-Y、3D-Y/Z/YZ 实验给出同一输入 Mask 的指标和可复现固定视图；真实未跑标记 NOT_RUN 而不是 COMPLETE。
- [ ] 单独提交 `ARCHIVE_MANIFEST_20261110.md`、`CLEANUP_REPORT_20261110.md`，逐项列出 KEEP/INVALID/DELETE 决策和恢复方式。

## 本 PR 暂不做 / Out of scope

自监督/DINOv3、完整神经元/钙功能实验、未经数据校准的“PSF 提升”强结论、用测试 GT 优化拓扑置信度、删除有效负结果或直接覆写 `PUR-MPM`。Topology confidence 完整任务头和结构损失可在核心方法通过 Gate 后作为 PR 后续独立提交。

## 提交安排

建议分 commits：`baseline freeze & archive audit` → `physics prior refactor & tests` → `residual/dual-branch & tests` → `evaluations & benchmark` → `GPU results & method analysis` → `uncertainty/task adapter (conditional)` → `safe cleanup report`。

**验收人关注：**是否真正证明 Wiener 物理候选在控制网络容量后的独立贡献；是否在完整三维结构而非仅 PSNR/Dice★ 上改善；是否保留科学负结果且不对历史结果实施破坏性清理。

详细实施任务书：[`docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md`](docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md)；开发基线 `ed060f0366c17e43f0329a913234d550b88b7682`。本次提交只包含 PR 任务文档，模型代码、GPU 结果与数据归档尚未实施；所有任务状态初始为 `NOT_STARTED`。
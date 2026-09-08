# 结果分层索引

本索引用于重新解释已有结果目录，不移动历史文件，避免破坏既有引用。

---

## 1. 正文主实验候选

### CIDC25 / F0 / 3D sparse reconstruction

| 目录 | 定位 | 使用方式 |
|---|---|---|
| `eval_sparse3d_k2_fullimg/` | CIDC25-like F0，K2/K4 混合汇总 | 可用于正文主结果，但需要按 K 和 sparse mode 分组，不做全局大排名 |
| `eval_sparse3d_k4_fullimg/` | CIDC25-like F0，K4 汇总 | 可用于 K4 主结果与消融 |
| `eval_sparse3d_k2/` | 早期 K2 patch/overview 结果 | 可作补充，优先使用 fullimg 重评结果 |
| `eval_sparse3d/` | 早期 K4 结果 | 可作历史对照，优先使用 fullimg 重评结果 |

推荐正文结论：

- mask 通道先验 + Wiener 初始化：当前阶段的核心创新主线。Wiener 提供物理逆问题初值，mask 提供采样置信度，二者应通过结构化融合而不仅是普通通道拼接来验证。
- 2D `x/y/xy` sparse：用于 MDC25-2D 与 3D 单切面 baseline，回答单轴/双轴空间稀疏恢复问题。
- 2.5D + `z` sparse：中心切面重建强，解释为利用邻切面冗余与轴向采样先验。
- 3D + `xyz` sparse：作为严格体重建主挑战，验证联合空间/轴向稀疏下的先验融合收益。
- 普通 PSF channel / PSF loss：当前收益不稳定，作为消融项，不作为最稳主贡献。

---

## 2. 独立验证 / 压力测试

### MDC25

| 目录 | 定位 | 使用方式 |
|---|---|---|
| `eval_mdc25_sparse/` | MDC25 K2 optimized validation | 当前帧数不齐，作为跨数据压力测试；若进入正文，需同帧集合重评 |
| `eval_newsample_mdc25_preds/` | 新样本推理与结构重评 | 作为真实样本/结构泛化探索，不与 CIDC25 主表混排 |

注意：MDC25 必须拆分为 MDC25-2D Nuclei 与 MDC25-3D Tribolium。不要把 2D Nuclei 的 XY 任务和 3D Tribolium 的 Z/XYZ 任务混成一个协议。

---

## 3. 外部 2D real sparse / transfer probe

### pollen

| 目录 | 定位 | 使用方式 |
|---|---|---|
| `eval_pollen_sparse/` | 2D pollen real sparse 上的跨模型评估 | 作为 transfer probe，不进入主 benchmark |
| `eval_pollen_sparse_structural/` | pollen 结构指标重评 | 作为结构感知附录/失败边界 |

当前 pollen 数据来自 stack mean projection，因此是 2D 外部样本。3D/2.5D 权重跑 pollen 的结果只能说明跨维度迁移行为，不能说明 3D 体重建能力。

---

## 4. 历史归档

| 目录 | 定位 |
|---|---|
| `archive/failed_experiments_20260413/` | 失败路径与旧协议归档 |
| `archive/legacy_stage2_stage3_20260409/` | 旧 2D/2.5D/3D 探索结果 |

历史归档只用于解释实验演变，不作为当前主结论来源。
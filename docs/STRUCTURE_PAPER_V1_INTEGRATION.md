# E027 M0 — 工作分支盘点与选择性整合（Issue #4）

> 结论先行：`PUR-MPM` 与 PR3 之间**不是"两份并列的代码"**——`ed060f0` 是 PR3 分支的
> 祖先（`git rev-list --count HEAD..origin/PUR-MPM` = 0），所以整合只能是一条快进路径；
> 真正需要"选择性"的部分不在搬运，而在**审完之后判什么能留、什么必须改**。四个已知 review P0
> 已逐条核验，其中三个在核验中变成了"必须改才能整合"，改动全部落在 `PUR-MPM` 上。

## 1. 分支与提交

| 分支 | HEAD（整合开始时） | 状态 |
|---|---|---|
| `PUR-MPM`（主干） | `ed060f0` Quantify what "one voxel" buys the skeleton score | 本地与远端一致 |
| PR #2 `feature/phase1-dual-prior-20261110` | `b637bde` | Draft，未合并；是 PR3 的祖先 |
| PR #3 `feature/phase1-adaptive-prior-pr3-20261112` | `d19c4ca` | Draft，未合并；**包含 `PUR-MPM` 的全部提交** |

因此 `git merge` 是一次 fast-forward，不会产生"未经审查的合并提交"。审查发生在合并前后：

| 提交 | 内容 |
|---|---|
| `d19c4ca`（快进） | PR2+PR3 共 87 个提交进入 `PUR-MPM`：136 个文件、约 28.8k 行 |
| `a6c0a08` | **P0-4 修复**：洪水门控在 2-D（单窗口）上静默失效 |
| `572dff2` | **P0-2 + P0-3 修复**：`interp_only` 的 `u_physics` 泄漏；2-D OTF 缺 import；`wiener_deconv_3d` 接受 2-D 输入 |
| `bb89198` | **P0-1 更正**：T=3 两算子等价的机制说明（原说明的频点写错）＋ 相位拆分表 ＋ 测试 |

## 2. 四个 review P0 的核验结果

| # | 项 | 核验前状态 | 现在 | 证据 |
|---|---|---|---|---|
| P0-1 | 合成窗口 FFT 一致性（两 Wiener 顺序在 T=3 等价） | PARTIAL：结论对，**解释错**，且没有测试 | **CLOSED** | 见 §2.1；`spur_mpm/tests/test_operator_order.py`（3 条测试） |
| P0-2 | `interp_only` 遗留 `u_physics` | **OPEN**：`u_physics=|W0−I0|/(…)` 由反卷积结果算出，非 identity 核时控制臂会拿到处理臂的输出 | **CLOSED** | `models/dual_prior_reconstructor.py`；测试 `test_interp_only_removes_every_channel_that_encodes_the_deconvolution` |
| P0-3 | 2D OTF import | PARTIAL：Wiener 路径确实全 3-D，但 `acquisition.py` 调用了未 import 的 `gaussian_otf_2d`；`wiener_deconv_3d` 对 2-D 输入不报错 | **CLOSED** | `physics/acquisition.py`、`physics/wiener.py` |
| P0-4 | 结构评价口径 | PARTIAL：`STAGE=vessel_ab` 的"门控"在窗口口径下**从未生效** | **CLOSED（实现）／A/B 重跑 DEFERRED** | `metrics/vessel.py`；`tests/test_vessel_protocol.py` 新增 4 条 |

### 2.1 P0-1：结论对，但原解释不成立

原文档写"T=3 时轴向 OTF 只采样到 DC 与 Nyquist 对"——长度 3 的 FFT 根本没有 Nyquist 频点，
频点是 `{0, ±1/3}`。正确机制：填充后窗口的 Z 剖面只有两种形态（中心为实测面 → 常数；
中心为缺失面 → 三点线性 `[A,(A+B)/2,B]`），其 `±1/3` 系数等大反号，而高斯 OTF 对 `f_z` 是偶函数，
因此两项在中心样本处相消，与 `σ_z` 无关 → 3-D 算子的中心面恰等于"先横向 Wiener 再轴向插值"。
这一条现在由测试固定，不再只是"我们测到了"。

同时把 T 扫描按**晶格相位**拆开：T=5 时实测中心面差 6.8e-3，缺失中心面仍是 3.6e-8；
原先的单均值把这件事盖住了。

### 2.2 P0-2：控制臂里到底该不该留 `u_physics`

原代码注释认为 `u_physics` 是 `(M, H, λ)` 的函数、与 W0 无关，所以控制臂应保留，否则控制臂
"知道得更少"。这个读法在 identity 核下看不出问题（此时 `u_physics ≡ 0`），但它与定义冲突：
`physics_prior` 里 `u_physics = |W0 − I0| / (|I0| + |W0 − I0| + 1e-6)`，**是反卷积结果算出来的**。
非 identity 核时它等于一张"先验把哪里改了"的地图，恰好是匹配控制最不该看到的。现改为同时置零，
`u_sampling`（只依赖掩膜）保留。E024–E026 全部在 identity 核下，结论不受影响。

## 3. 整合内容（按用途分组）

| 区域 | 文件数 | 内容 | 处置 |
|---|---|---|---|
| `spur_mpm/physics` | 4 | PSF/OTF、Wiener、prior、acquisition；前向额外模糊与先验 PSF 分离 | 已整合，含 2 处修复 |
| `spur_mpm/data` | 2 | 窗口构造（Z 边界、晶格相位、双重分支 holdout） | 已整合 |
| `spur_mpm/models` | 2 | 双分支重构器 + 注册表（`P3_DualAdaptive`、`phys_control`） | 已整合，含控制臂修复 |
| `spur_mpm/metrics` | 1 | 血管分割/骨架/门控 | 已整合，含 2-D 门控修复 |
| `spur_mpm/{train,evaluate,inference}.py`、`device.py`、`common` | 5 | 训练/评估/整卷推理/设备守卫 | 已整合 |
| `spur_mpm/scripts` | 12 | `structure_report`、`compare_structure`、`collect_results`、`qualitative_figures`、`viewspec`、`select_adaptive_prior`、`tolerance_sweep` 等 | 已整合；E027 将新增 `paper_figures/` |
| `spur_mpm/experiments` | 5 | `_gpu_gate.sh`、E024/E025/E026 运行脚本、`run_vessel_ab.sh` | 已整合（E027 不重跑这些组） |
| `spur_mpm/configs` | 4 | E024/E025/E026 + 新增 `E027_literature_psf_figures.yaml` | 已整合 + 新增 |
| `spur_mpm/tests` | 14 | 444 项测试 | 已整合 |
| `docs/`、`docs/evidence/`、`docs/figures/` | 44 | PR2/PR3 报告、事故证据、5 张定性图 | 已整合（作为证据，不作为 E027 的结论） |
| `runs/`（被 git 跟踪的摘要） | 37 | E024 结构摘要等小文本 | 已整合（Issue 要求保留正确实验） |

## 4. 延期与不做

| 项 | 原因 |
|---|---|
| 血管口径 A/B 全量重跑（17 次评测） | 门控修复后才有效；已完成的 2 次（I0、W0_full3d）在 old code 下门控失效，移入 `runs/_invalid/20261119_vessel_gate_noop/`。E027 的图表使用 `docs/PHASE1_PR3_E026_RESULTS.md` §4.3 的结论与 `E027` 配置里统一的门控口径，重跑**不在本 Issue 的验收范围内** |
| 多 seed / 多 fold | 初测无可信 Wiener 增益，加种子只会让零值更精确 |
| 合成已知核组（`matched_synthetic`） | 本 Issue 只做真实数据；synthetic 组排除在图表之外 |

## 5. E027 还需要新增的东西

1. `spur_mpm/scripts/paper_figures/`：确定性渲染器（伪彩 LUT、比例尺、panel letters、冻结 view spec）+ `validate_figures.py`。
2. `figures/paper_structure_v1/`：可控大小的 PNG 预览 + SVG/PDF。
3. `tables/paper_structure_v1/`：Table 1/2 的 CSV + markdown/tex。
4. `reports/paper_structure_v1/`：`summary.csv`、`per_volume.csv`、`ROI_manifest.json`、`method_manifest.json`、PSF provenance、测试日志。
5. Fig. 1–4 与 Supplement 的编排脚本；Table 2 的数从 CSV 生成。

## 6. 验证

```bash
python -m pytest spur_mpm/tests -o addopts="" -q    # 444 passed (2026-11-20)
```

- 旧 checkpoint `strict=True` 加载：E026/E025 的 `best.pt` 在本分支上可加载（`test_pr3_adaptive_prior.py` 覆盖 state_dict 键一致性）。
- 设备守卫：`CUDA_VISIBLE_DEVICES=9 python -m spur_mpm.train …` 立即退出而非静默用 CPU。
- E027 配置可加载、identity 覆盖可用（`physics.psf.sigma_*=0` → `is_identity=True`）。

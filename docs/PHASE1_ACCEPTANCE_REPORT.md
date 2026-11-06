# 第一层（有监督）验收报告

对照 `docs/PHASE1_SUPERVISED_AUDIT_AND_RUNBOOK.md` §2–§4.5 的执行记录。这份文件的用途是
贴在 PR 里：**改了什么文件、怎么复现、测试输出、图在哪、修复前后差在哪**。全部数字都能在
`runs/` 下回查到产生它的 `summary.json` / `*.json`；方法学叙述见
[`Project log/Project_log.md`](../Project%20log/Project_log.md) 的 SPUR-E023 一节。

---

## 1. 结论

| 项 | 状态 |
|---|---|
| §2 显式稀疏数据构建与独立核对 | **通过**（90/90 构建逐点校验） |
| §4.1 回归测试 / 配置 / 模型代码 | **通过**（207 条测试全绿） |
| §4.2 最小有监督消融 + 重打分 + 主表 | **通过**（52 个 cell，含 S2g 与等上下文对照） |
| §4.3 整卷 3D 血管结构 | **通过，带门控披露**（计分层占比 34–72 %） |
| §4.4 定性图（全部测试卷） | **通过** |
| §4.4.1 跨稀疏程度一致性修复（P1） | **通过**（一致性自检 0 差异，硬性非零退出） |
| §4.5 参数 / MACs / 延迟 / 整卷端到端 | **通过**（零参数臂也实测，整卷 41.6 s/卷） |

**没有一个协议下的完整 `target` 不同**（`target_diff = 0.0`），所以 §4.4.1 里"若发现不同
视为 P0 数据路径问题"的条件没有触发：问题确实只在绘图层的视图选择与标签上。

---

## 2. 修改的文件

### 2.1 审查 PR 带入的修复（已在本分支上验证）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `spur_mpm/data/datasets.py` | Z 边界窗口居中（`y0 = (h - ph)//2`），`warmstart_center`，mask dtype 前置校验 | 旧实现在首尾层夹紧窗口，查询层不在中心通道，而有监督损失只监督中心输出 → 跨层协议的边界层预测与整卷骨架指标全部失真 |
| `spur_mpm/data/sparse.py` | PSF 下的自检断言修正 | 开启合成 PSF 时曾拿模糊后的 `Y` 与未模糊的 `D` 比较 |
| `spur_mpm/models/registry.py` | 新增 S2g（X0 + U + 门控） | 消融缺"完整特征 + 门控"臂，无法隔离各模块贡献 |
| `spur_mpm/metrics/skeleton.py` | 微米半径用真实体素间距 | 几何平均间距把 Z=5 µm 与 XY=0.98 µm 混算 |
| `spur_mpm/scripts/run_ladder.py` | 无空闲 GPU 不抢 0 号卡；`--only`；状态按目录名去重 | 共享服务器上抢卡/误重训 |

### 2.2 本轮新增或修改

| 文件 | 改动 |
|---|---|
| `spur_mpm/scripts/viewspec.py` | **新文件**。冻结视图层：`build_view_spec`（只从密集参考取坐标）、`validate_view_spec`、`annotate`、`fingerprint`、`projection_note`（从实际面板派生）、`protocol_columns`（按协议给诊断面板）、`sampling_semantics`、`window_stability`、`check_group` / `check_protocols` |
| `spur_mpm/scripts/qualitative_figures.py` | `_view_spec()` 的 mask 依赖改为一次性冻结；`--view-spec` / `--save-view-spec` / `--figure-name` / `--z-view` / `--y-view` / `--x-view`；`--compare` 写 `consistency_report.json` 并在任一条不成立时 `return 2`；窗宽只从参考取 |
| `spur_mpm/scripts/sparse_figures.py` | `_common_plane()`：XY 面板画在**所有层级都测到**的那一层；2D 阶梯图用掩膜实际切片层；方向标注 y↓/x→/z↓ |
| `spur_mpm/scripts/ladder_report.py` | 新增等上下文 25 % 换轴表（plan §3.3），七臂齐全 |
| `spur_mpm/scripts/structure_report.py` | 逐卷打印 `gated/total (scored %)`；`--from-summary` 可从 JSON 重排版（不需重跑推理） |
| `spur_mpm/scripts/benchmark.py` | 零参数臂的 warm-start 算子**实测**（不再 `latency n/a`）；结果含 `p10/p50/p90`、峰值显存 |
| `spur_mpm/inference.py` | 整卷端到端 `timing.wall_seconds` 写入 meta |
| `spur_mpm/metrics/skeleton.py` | 被门控清空的 slab 记 `NaN`（原记 `dice = 1.0`，等于给最难深度发奖励） |
| `spur_mpm/tests/test_qualitative_figures.py` | 19 条：参考不随协议变、目标不一致硬失败、面板/窗宽来自参考、Z-only 只报层、Y-only 只报行、同输入重复被拒、指纹随配置变、投影说明与面板一致、掩膜必须是它声称的协议、冻结窗宽等于参考派生 |
| `spur_mpm/tests/test_sparse_figures.py` | 4 条：共同平面、无共同平面拒绝出图 |
| `spur_mpm/tests/test_benchmark.py` | 4 条：零参数臂被计时、缺 PSF 退化为说明、markdown 声明、算子失败进备注 |
| `spur_mpm/tests/test_ladder_report.py` / `test_skeleton.py` | 等上下文表必须含 S2g；空 slab 必须是 NaN |
| `docs/CODE_MAP.md`、`docs/RESULTS.md` | 补 `viewspec.py`、S2g、新 CLI、三个新测试文件与口径陷阱 |

---

## 3. 复现命令

```bash
cd /ssd1/wjb/SPUR-MPM && conda activate threePM

# §2 数据构建与独立核对（全部在 CPU 上，90/90 通过）
python -m spur_mpm.scripts.build_sparse --as-3d ...        # 见 runs/E023_phase1/build_data.sh
python -m spur_mpm.scripts.verify_dataset_masks ...
python -m spur_mpm.scripts.sparse_figures --root data_local/sparse/cohort3d \
    --out spur_mpm/figures/sparse

# §4.2 阶梯（GPU 2）：matched 上下文 + 全量重打分 + 主表
bash runs/E023_phase1/run_ladder_all.sh                    # → ALL_DONE
bash runs/E023_phase1/rescore_own_context.sh               # → 9/9 ok（注意：不要带 --context-slices）
python -m spur_mpm.scripts.ladder_report --root runs/E021_ladder \
    --out runs/E021_ladder/ladder.md

# §4.3 整卷结构
python -m spur_mpm.scripts.structure_report --arm S1 \
    --ckpt runs/E021_ladder/E021_S1_y_k2/best.pt \
    --out runs/E021_ladder/E021_S1_y_k2/structure --device 2

# §4.4.1 固定视图主对照图（同一 ViewSpec，三协议，z=100）
bash runs/E023_phase1/refreeze_figures.sh                  # → compare_y_z_yz_25pct.png + consistency_report.json

# §4.4 全部测试卷的五臂图
bash runs/E023_phase1/qual_one_volume.sh M2_FOVII_S1_vessels   # 其余三卷同理

# §4.5 成本（cuda:2 空闲时）
python -m spur_mpm.scripts.benchmark --markdown --arms A0 A1 S0 S1 S2 S1g S2g \
    --patch 256 --n-slices 200 --device cuda:2 --out runs/E021_ladder/benchmark_supervised.json
bash runs/E023_phase1/benchmark_e2e.sh                     # 整卷端到端
PYTHONPATH=. python runs/E023_phase1/probe_warmstart_cost.py
```

## 4. 测试输出

```
$ python -m pytest spur_mpm/tests -q
# 245 条全部通过（rc=0），整轮约 13 s
```

版本信息记录在 `runs/E021_ladder/figures_fixed/M2_FOVII_S1_vessels/run_log.json`：
python 3.10.19 / torch 2.6.0+cu124 / numpy 2.2.6 / RTX A6000 / 绘图代码 commit `037dadf`。

---

## 5. §4.4.1 D 验收门槛逐条对照

产物目录：`runs/E021_ladder/figures_fixed/M2_FOVII_S1_vessels/`

| 门槛（手册原文） | 实测 | 判定 |
|---|---|---|
| `view_spec.json`：固定坐标、物理 voxel spacing、统一窗、数据/配置指纹、MIP 范围 | `z_view 100`、`y_view 128`、`x_view 128`、`roi [160,0,96]`、`voxel_um [5.0, 0.98, 0.98]`、5 个面板窗、`config_sha256`、`mip_z [0,203]`、`projection` 由面板派生 | 通过 |
| `compare_y_z_yz_25pct.png`：`y_k4/z_k4/yz_k2x2` 同视野同 GT 主对照图 | 6.1 MB，三协议 × 两臂，同一固定 ROI，显示真实 Mask 与完整 GT | 通过 |
| `qualitative_*_<level>.png` + `support_*_<level>.png`（标签正确、真实测量支持） | 三协议各一套；支持图标注由实际 mask 判定（Z-only 报"实测层/跳过的层"，Y-only 报"实测行/跳过的行"） | 通过 |
| `max_abs_diff_gt`（期望精确 0） | `target_diff = 0.0` | 通过 |
| 各对应截面 `max_abs_diff_view`（0） | `fixed_view_diff = 0.0`，逐协议 `view_diff = 0.0` | 通过 |
| ViewSpec / 物理坐标一致性 | `geometry.<level>.matches_spec` 与 `voxel_ok` 全 True | 通过 |
| 实际 measured fraction | `y_k4 0.25`、`z_k4 0.2512`、`yz_k2x2 0.2512`；与各自协议的 `ideal_fraction` 之差在 `2/H + 2/Z` 内 | 通过 |
| `mask` 采样语义校验 | `mask_semantics.<level>.axes_ok` 全 True（z 是否稠密对得上 `kz`，y 是否稠密对得上 `ky`） | 通过 |
| 统一 colorbar / intensity range | `display_windows.window_diff = 0.0`（把窗宽从同一参考重新派生一次并要求完全一致） | 通过 |
| 逐臂输入一致性 | 每组 `input_diff = 0`（含 `in_mask` 逐点相同） | 通过 |
| 失败必须非零退出，不能只 warning | `return 2` + 打印整个 report | 通过 |
| PNG + JSON + NPZ，float32 未压缩 | `--dump f32`，`qualitative_*.npz` 为 float32 | 通过 |
| pytest + 保存通过的日志与版本信息 | 245 条全过（rc=0） + `run_log.json` | 通过 |
| 禁止仅改 label 不修坐标/输入逻辑 | 坐标从密集参考一次性冻结；`projection` 说明改由面板派生并配测试；输入由 `Y = M·D` 产生，`input_diff` 校验 | 通过 |

**这三项本身就是"断言"改"计算"的产物**：`display_windows` / `mask_semantics` / `geometry`
在手册里被要求写进 `consistency_report.json`，但第一版只是把冻结值抄进 JSON——即使掩膜
根本不是该协议、或窗宽来自别的体积，报告也照样写 `ok`。现在三者都是重算后比对：

- `display_windows.window_diff = 0.0`（把窗宽从同一参考再派生一次，要求与冻结值逐位相等）；
- `mask_semantics.<level>` 用 `measured_fraction` 对 `ideal_fraction`（容差 `2/H + 2/Z`）+
  `z_dense`/`y_dense` 对 `kz`/`ky` 双路校验，实测 `0.25 / 0.2512 / 0.2512`，全部 `axes_ok`；
- `geometry.<level>.matches_spec` + `voxel_ok`，实测 `[203,256,256]`、`[5.0,0.98,0.98]`。

这套校验第一次跑就崩了：报告把 volume sidecar 里的 `sampling` **标签字符串**（`y_k4`）
当成协议 spec 传给了校验函数。修法不是加 `try`——组里必须带 `SamplingSpec.to_dict()`
（`ky/kz/ideal_fraction`），缺了直接 `ValueError`，否则校验会"空过"。已加
`test_the_report_reads_the_protocol_spec_not_the_volume_tag` 钉住这个路径（含"标签贴错协议
必须被抓"与"缺 spec 必须报错"两个断言）。

**三协议上下文一致**：`context_depth = {y_k4: 7, z_k4: 7, yz_k2x2: 7}`（plan §3.3）。

### 5.1 手册 §7 图件清单（哪张已出、在哪、哪张还缺）

| 手册要求 | 状态 | 文件 |
|---|---|---|
| Fig. 1 几何/动物划分、Y/Z/YZ 稀疏 Mask 与预算示意 | 出图 | `spur_mpm/figures/sparse/sparse_1_construction.png`、`sparse_2_ladder_2d.png`、`sparse_3_coverage.png` |
| Fig. 2 网络/消融矩阵示意 | 出图（本轮补齐，**不是手绘**：矩阵每一格由 `models/registry.ARMS` 生成，`fig2_provenance.json` 逐格记录取自 `ARMS['S1'].features` 之类的哪个字段） | `spur_mpm/figures/framework/fig2_pipeline.png`、`fig2_ablation_matrix.png`、`fig2_provenance.json`（脚本 `spur_mpm/scripts/framework_figure.py`，19 条测试） |
| Fig. 3 Y 稀疏下各消融 XY/XZ/YZ + 统一色标 + 误差图 | 出图 | `runs/E021_ladder/figures_fixed/M2_FOVII_S1_vessels/`（三协议 × 两臂：`compare_y_z_yz_25pct.png`、`qualitative_*`、`error_*`、`support_*`、`profile_*`）；五臂全测试卷另在 `runs/E021_ladder/figures/qual_y_k2/` |
| Fig. 4 Z/YZ 性能–稀疏曲线、同 25 % 预算、固定 Z 上下文对照 | 出图（本轮补齐） | `runs/E021_ladder/figures/curves/curve_{y,z,yz}.png`、`curve_matched_25pct.png`、`curve_matched_25pct_c7.png`（脚本 `spur_mpm/scripts/ladder_curves.py`） |
| Fig. 5 3D 掩膜/骨架、局部分叉、断裂/伪连接标记 | 出图（本轮补齐）：四卷 × 两臂（S1 / A1 零参数地板）逐卷图 + 跨卷汇总图；断点/伪连接图只在容差 1 voxel 外非空，空的配对记 NaN 而不是 1.0；分叉裁剪框冻结在 JSON（删 JSON 才会重选） | `runs/E021_ladder/figures/structure/structure_{S1,A1}_M2_{FOVI,FOVII}_S{1,2}_vessels.png`、`structure_cross_volume_{S1,A1}.png`、`crop_*.json`、`cache/*.npz`（脚本 `spur_mpm/scripts/structure_figure.py`，14 条测试）；数值仍见 `runs/E021_ladder/E021_{S1,A1}_y_k2/structure/*_{pred,ref}_mask.tif`、`*_recon.json` |
| Fig. S1 参数量/MACs/耗时、背景强度误差、门控审计、全视野结果 | 出图 | `runs/E021_ladder/complexity.md`、`error_*`（背景强度）、`figures_fixed/.../qualitative_*`（含逐层门控占比说明） |

Fig. 4 的两点口径写进图注而不是画线下限：0.03 Dice 的"主张地板"约束的是**臂与臂之间的差**，
不是柱子的绝对高度，所以没有画成底部灰带（画成灰带会被读成"柱子低到这个值以下就无效"）。

---

## 6. 修复前后差别

| 项目 | 修复前 | 修复后 |
|---|---|---|
| 面板坐标 | 每次运行用**当前 mask** 选 `y_meas/y_gap`，各协议各选各的 | 一次性从密集参考冻结并落盘，跨协议/跨臂复用 |
| 固定 XY 面板 | `Z//2 = 101`，而 `z_k4`/`yz_k2x2` **正好跳过该层** → 共享列输入行全空 | 显式 `--z-view 100`（三协议都测到：25 %/100 %/50 %）；旧图另存 `..._z101/` 未覆盖 |
| Z-only 标签 | `XZ given/gap`（按 Y 行说事，语义错误） | "实测层/跳过的层" + 实测 `measured_fraction` |
| support 图脚注 | 固定写"x and z are never subsampled" | 按协议实际采样轴生成 |
| `projection` 说明 | 常量"none (…)"，但图里已有 MIP-z 列 | 由列表面派生，并加测试防再漂移 |
| 窗宽 | 每张图按自己自动拉伸 | 每面板一次从参考取，全部协议/方法复用 |
| 图的元数据 | 重绘后仍写着"none (…mip_z 留给将来的投影)"，而图里已有 MIP-z 列 | 说明由实际面板列派生；重绘后**PNG 字节数与重绘前完全一致**（`compare_y_z_yz_25pct.png` 6,080,722 B），只有 JSON 变了——像素没动，改的是那句话 |
| 不一致时的行为 | `print` 警告 | 非零退出 |
| 被门控清空的 slab | `dice = 1.0`（两空掩膜"完全一致"） | `NaN`，并在表里标出计分层占比 |

**跨层协议的旧数字已全部作废**：Z 边界窗口修复改变的是跨层读取顺序，因此 `z_*`、`yz_*`
的旧预测与整卷骨架指标不可再用；`y_*` 族逐位不变（见下）。

| cell | 修复前 Dice | 修复后 Dice |
|---|---|---|
| `S1 y_k2` | 0.7199 | **0.7199（逐位不变）** |
| `S1 z_k2` | 0.6992 | 0.7062 |
| `S1 z_k4` | 0.5204 | 0.5297 |
| `A0 z_k4` | 0.6048 | 0.6105 |

---

## 7. 尚未关闭的风险（照抄手册 §6，附本轮实测）

1. **门控准则过激**：参考前景 > 15 % 的层被剔出，逐卷剔除 57/203、68/202、82/201、101/152
   ——最差一卷只有 **34 % 的层参与评分**，且集中在表面与深层。整卷分数必须与这个占比一起引用。
2. **A1 在单位 PSF 下退化为 A0**（同一算子，1.6 ms/卷，见 `warmstart_cost.json`）。要主张
   Wiener 引导，先补经实验标定的 PSF，否则只是"换了名字的插值"。
3. **训练方差未知**：全部单种子、单 split；跨动物方差（≈2.8 dB / 0.08 Dice）大于本轮所有
   消融差异，所以 < 1 dB / < 0.03 Dice 一律不写结论。
4. **M2 只有一只留出动物**，不能用对切片 bootstrap 的窄 CI 声称跨动物稳定。
5. **人工解剖 GT 缺失**：Sato vesselness + 参考侧 Otsu 只是自动参考，不能证明连接真实；
   建议人工标注少量可追踪微血管/分叉做小规模核验。
6. **ImageJ 抽查待做**（由人完成）：`*_pred_mask.tif` / `*_ref_mask.tif` 已导出，可直接打开；
   重点看最深 slab 的被门控层与断裂/伪连通。

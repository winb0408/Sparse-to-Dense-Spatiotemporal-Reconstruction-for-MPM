# E027 论文图规范（STRUCTURE_PAPER_V1）

> 这份文件规定"图长什么样、为什么这样、怎么重画"。所有图由
> `spur_mpm/scripts/paper_figures/` 生成，数据来自 `reports/paper_structure_v1/`，
> **没有任何数字是手填的**。

## 1. 统一视觉规则（Issue #4 §3）

| 项 | 规则 | 实现位置 |
|---|---|---|
| 血管强度 | **magenta/pink** 顺序伪彩（黑 → 深洋红 → 亮粉 → 暖白），不用灰度、不用 rainbow | `style.VESSEL_CMAP` |
| 误差 | **magma** 连续色带；未计分体素为**中性灰**（不是白，白会被读成"无误差"） | `style.ERROR_CMAP` |
| 缺失区 | **青色** `#22d3ee` 半透明叠加；图例说明"cyan = never measured" | `style.MISSING` |
| 强度窗（LUT） | **每个协议一个窗口**，取自参考（Dense）的 p0.5–p99.5；同协议所有方法共用；禁止逐方法调对比度 | `_lut()` |
| 比例尺 | 物理 **µm**，画在每块首列 | `style.add_scalebar` |
| 视角 | XY 为主，配 **XZ**；XZ 按物理比例（z 步 5 µm vs 平面 0.98 µm ⇒ 纵横比 ≈ 5.1） | `_xz()` + `set_aspect` |
| panel letter | 小写 a–f，白色描黑边（压在显微图上也可读） | `style.panel_letter` |
| 尺寸 | 双栏 183 mm；字 7–9 pt | `style.mm(183)` |
| 输出 | PNG（预览，400 dpi）+ PDF（矢量文字）。SVG 由同一命令生成，因显微面板是位图、单文件约 5 MB，暂不入库 | `style.save_figure` |

## 2. Fig. 2 — 真实数据定性主图

三行 = 三种协议（Y-50 %、Y-75 %、YZ-75 %）；六列 =
`Dense reference | Sparse sampled | Interpolation | W0 literature PSF | ResidualI | Proposed`。

| 文件 | 内容 |
|---|---|
| `fig2a_protocols.{png,pdf}` | 每协议两行：XY（ROI 320×320 px ≈ 314 µm）与 XZ 断面。Sparse 列用青色标出未测量体素；RZ 列的青色条纹即 z 跳层 |
| `fig2b_error_missing.{png,pdf}` | 每协议一行：**只在从未测量的体素上**的 `|pred − dense|`，协议内共享色标，格内印该裁剪区 MAE |
| `fig2c_zoom.{png,pdf}` | YZ-75 % 的 96 µm 放大，同一冻结 LUT，用于看细血管与断裂 |

**NOT_RUN 的呈现**：Y-50 % 没有训练臂的 checkpoint，因此该格画成灰底 "NOT RUN"——
不从别的协议借、不用别的先验补。这是 Issue #4 的硬要求。

**W0 列为什么留着**：M1 的结论是文献 FWHM 先验在本网格上**变差**（缺失区 MAE +57 %~+80 %，
见 `docs/STRUCTURE_PAPER_V1_RESULTS.md`）。把它画出来是负结果的证据；删掉它才是"选数据"。

## 3. 复现

```bash
# 1) ROI 缓存（floor 在 CPU；训练臂需要一张卡，--reuse 可只重裁剪）
python -m spur_mpm.scripts.paper_figures.cache_rois \
    --out reports/paper_structure_v1/rois --arms gpu     # 首次
python -m spur_mpm.scripts.paper_figures.cache_rois \
    --out reports/paper_structure_v1/rois --arms gpu --reuse   # 换 z 范围

# 2) 出图（不需要 GPU）
python -m spur_mpm.scripts.paper_figures.fig2_qualitative \
    --rois reports/paper_structure_v1/rois --out figures/paper_structure_v1
```

## 4. 已知限制（写在图注里，不藏在正文）

- 结构参考是**自动 Sato 分割**（阈值只在参考上拟合；参考填充 >15 % 的平面被门控），
  不是人工标注的血管拓扑真值。图上不得称其为 manual GT。
- 本图是**一个测试动物、一个 FOV、一个 ROI**，用于看机制，不是统计证据；
  统计在图 3/表 2。
- `W0 literature PSF` 的条纹不是显示伪影，是 Wiener 在采样频率上的放大（M1 §机制）。

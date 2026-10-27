# 图目录

全部图 300 dpi PNG + 同版 PDF；定性面板为灰度 vmin=0/vmax=1、无坐标轴。图的原始
数字在 `../numbers/key_numbers.csv`，详细说明（含重画命令）在 `INDEX_original.md`。

## 定量

| 文件 | 内容 | 对应主张 |
|---|---|---|
| `fig1_alloc_law` | 6 个 YZ 采样点 × 4 个 arm 曲线（1a）；同预算 `ky·kz=16` 三点柱状（1b） | C1 |
| `fig2_prior_vs_network` | 先验侧 PSNR / 网络侧 PSNR / 两侧差值符号 | C3 |
| `fig3_gain_over_direct` | 各 arm 相对 Direct 的逐点增益 | C2 |
| `fig4_unified_cross_transfer` | 专训 vs 统一 × 三 regime（4a）；统一变体（4b） | C4 |
| `noise_curve` | 5 个评估点在 F0–F3 噪声下的退化曲线（含 x0 先验与原始噪声输入参考线） | C5 |
| `exp028_suite_curves` | U15W / U15 / SXYZ 在三套件上的曲线 | 否证 N2 |

## 定性（val split，固定 patch 网格，items 0–2）

列顺序统一为 GT | 稀疏 Y（2.5D 时取最近实测平面）| X0 先验 | 各模型输出，重建面板
下方标注该面板的 PSNR。

| 文件 | 内容 |
|---|---|
| `qual_yz_2x8` | EXP-024 最优点：GT vs b1/b2/b4i/b4f |
| `qual_yz_8x2` | 最差点，同样列——可直观看到分配律的差距 |
| `qual_yz_2x8_err` / `qual_yz_8x2_err` | 绝对误差图（统一色标 0–0.12）+ MAE |
| `xz_yz_2x8` / `xz_yz_8x2` | XZ 截面（最亮列），横向条纹 = 未采样平面 |
| `qual_univ_y_K4` / `qual_univ_yz2_K4x4` / `qual_univ_xyz2_K4x4` | 统一 U15 vs 对应专训模型 |
| `xz_univ_yz2_K4x4` | 统一模型在 YZ 上的 XZ 截面 |
| `panel_y_K2` / `panel_y_K4` / `panel_y_K8` / `panel_yz2_K4x4` / `panel_xyz2_K4x4` | EXP-027：行 = F1/F2/F3，列 = GT(干净 F0) / 噪声原始 / 稀疏 Y / X0 / U15 / U_noAZ |

## 看图的三个要点

1. `qual_yz_2x8` 与 `qual_yz_8x2` 必须并排看：同样的总预算，一个几乎贴合 GT，一个
   连结构都糊掉——这是 C1 的直观证据。
2. `qual_yz_*_err` 的色标是**统一**的（0–0.12），所以列间可直接比较误差强度。
3. `xz_yz_2x8` 里稀疏 Y 列的水平条纹就是 k_z=8 的未采样平面；看模型是否真的在条纹
   之间补出了结构（而不是复制相邻平面）。

## 未包含的图

- v3 时代的 severity / T-scaling / 光子预算 / 外部基线图在 `PSRAS/eval_viz/`，未复制
  进来（它们属于 v3 证据链）。
- XYT（时序）图：数据依赖，尚未开始。

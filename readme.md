## 当前推荐阅读路径

本项目已经按“数据维度先行”的实验主线重整。建议按以下顺序阅读：

1. `docs/DATASET_PROTOCOL.md`：冻结 CIDC25、MDC25、pollen 的数据身份，以及 2D / 2.5D / 3D 合法组合。
2. `docs/RESULT_TAXONOMY.md`：说明哪些结果属于正文主实验，哪些只能作为压力测试或迁移探索。
3. `report/SUCCESS_PATH_2D_2P5D_3D.md`：短版主线说明。
4. `report/reorganized_main.tex`：完整论文式 LaTeX 报告，覆盖实验目标、总体设计、具体方法、实验结果、分析、不足和后续工作；正文以文字和表格为主，文末附录集中放置关键图集。

核心原则：不要把 CIDC25、MDC25、pollen 的结果混成一个总排名；主结论只来自数据协议一致的实验。

---

## 项目概览：3PM 稀疏采样重建与可视化套件

本项目实现了 3PM（Three-Photon Microscopy）图像从 **稀疏采样（noisy / sparse）到高质量重建（dense / GT-like）** 的完整评估与论文级可视化流水线，包含：

- **模型评估 (`eval.py`)**：对整个测试集进行滑窗重建，计算完整指标（PSNR/SSIM/MS-SSIM/LPIPS/sSNR/FwPSNR 等），输出 `csv` 结果与 `npy/tif` 图像。
- **高阶可视化 (`plot.py` / `visualize.py`)**：生成 Nature 级别排版的复合大图（Fig.1–5），包括：
  - 定性大图 + ROI 放大 + 误差热力图 + 1D 截线轮廓
  - 频域分析（PSD、sSNR、FwPSNR）
  - 单模型统计图（箱线图/小提琴图）
  - 消融对比图、训练曲线
  - 稀疏度容忍度曲线（接口已预留）

你只需要按命令行调用，即可从 checkpoint → 指标 → 论文插图，一条龙自动生成。

---

## 1. 评估脚本：`eval.py`

### 1.1 单模型评估（推荐）

**功能**：对一个 checkpoint 在完整测试集上做滑窗重建，输出：

- 每张图的全套指标（`metrics_{run}.csv`）
- 指标汇总（`summary_{run}.csv`，包含 mean/std/95% CI）
- `images/{run}/*_pred.npy`、`*_gt.npy`、`*_noisy.npy`
- 可选的 `*_pred.tif`（需安装 `tifffile`）

**命令示例**：

```bash
python eval.py \
  --task baseline \
  --model swin_unet_2d \
  --data_root /path/to/Dataset \
  --checkpoint runs/full/best.pth \
  --output_dir eval_results \
  --patch_size 256
```

**关键参数（来自 `options.get_opts()`）**：

- `--data_root`：数据根目录，要求存在 `Test` 子目录（含 `low`/GT 通道）。
- `--checkpoint`：待评估的 `.pth` 权重文件。
- `--patch_size`：滑窗 patch 大小（与训练保持一致）。
- `--sparsity_factor`：稀疏度（2x/4x/8x/16x）。
- `--scan_pattern`：扫描模式（如 raster / spiral）。
- `--norm_mode`：归一化方式（如 `minmax`）。
- 其它如 `--no_mask_ch`、`--apply_psf`、`--psf_sigma_lateral` 等控制 PSF 与 mask。

python eval.py \
  --task baseline \
  --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/Denoising_Dataset \
  --checkpoint runs/run/best.pth \
  --output_dir eval_results \
  --no_mask_ch \
  --patch_size 256

**输出目录结构**（假设 `--output_dir eval_results`）：

- `eval_results/eval/metrics_full.csv`：每张图的所有指标
- `eval_results/eval/summary_full.csv`：指标汇总 + 95% CI
- `eval_results/eval/images/full/{stem}_pred.npy`
- `eval_results/eval/images/full/{stem}_gt.npy`
- `eval_results/eval/images/full/{stem}_noisy.npy`（若存在）
- 可选：`eval_results/eval/images/full/{stem}_pred.tif`

> `run` 名默认取自 checkpoint 所在文件夹名（例如 `runs/full/best.pth` → `run = full`）。

### 1.2 多模型消融评估（ablation）

**功能**：一次性评估多个 checkpoint，并给出：

- `ablation_all.csv`：所有 run 的 per-image 指标拼表（供可视化使用）
- `ablation_summary.csv`：每个 run 的 mean/std/CI
- `ablation_pvalues.csv`：各 run 两两间的 Wilcoxon 检验 p 值（Bonferroni 校正）

**命令示例**：

```bash
python eval.py \
  --task baseline \
  --model swin_unet_2d \
  --data_root /path/to/Dataset \
  --eval_checkpoints runs/full/best.pth runs/no_dino/best.pth \
  --eval_run_names   full              no_dino \
  --output_dir eval_results/ablation
```

**额外参数**：

- `--eval_checkpoints`：多个 checkpoint 路径。
- `--eval_run_names`：与 checkpoint 一一对应的 run 名（会写进 CSV）。
- `--cnr_signal_thresh`：CNR 前景阈值（默认 0.5）。
- `--n_bootstrap`：bootstrap 轮数（默认 2000，用于 CI）。
- `--no_save_images`：只算指标，不保存 `npy/tif` 图像。

---

## 2. 高级可视化：`plot.py`（原 `visualize.py`）

### 2.1 Figure 1：高级定性分析面板（全局 + ROI 放大）

**核心函数**：`figure_qualitative`（内部由 CLI 调用）

功能：

- 显示 **Input / Prediction / Ground Truth** 三张全局灰度图。
- 在每个全局图上画出 **黄色虚线 ROI 边框**，表示放大区域。
- 第二行展示对应的 ROI 放大图，并用连线与上方全局图连接。
- 支持像素物理尺寸（`pixel_size_um`）和标尺（scale bar）。
- 在 Pred 面板内叠加 PSNR/SSIM 数值，在 Input 面板显示其 PSNR。

**命令示例**（从 `eval.py` 输出的 `npy` 生成 Fig.1–3）：

```bash
python plot.py --mode single \
  --pred_npy  eval_results/eval/images/full/000001_pred.npy \
  --gt_npy    eval_results/eval/images/full/000001_gt.npy \
  --noisy_npy eval_results/eval/images/full/000001_noisy.npy \
  --roi 200 300 128 128 \
  --pixel_size_um 0.103 \
  --sample_name "HeLa nucleus" \
  --output_dir figures --fmt pdf
```

python plot.py --mode single \
  --pred_npy  '/data2/wjb/threePM/eval_results/eval/images/run/cell migration high and low - Position 0 - 1_XY1562738052_Z0_pred.npy' \
  --gt_npy    '/data2/wjb/threePM/eval_results/eval/images/run/cell migration high and low - Position 0 - 1_XY1562738052_Z0_gt.npy' \
  --noisy_npy '/data2/wjb/threePM/eval_results/eval/images/run/cell migration high and low - Position 0 - 1_XY1562738052_Z0_noisy.npy' \
  --roi 200 300 128 128 \
  --pixel_size_um 0.103 \
  --sample_name 'HeLa nucleus' \
  --output_dir figures --fmt pdf

生成的文件：

- `fig1_qualitative.pdf`：三列（Input/Pred/GT）×（可选 ROI 行），带色条、scale bar、指标注释。

### 2.2 Figure 2：误差热力图 + 1D 截线 + FWHM

**核心函数**：`figure_error_analysis`

功能：

- (a) 绝对误差图 `|GT - Pred|`，使用发散型或高亮色图，并配色条。
- (b) 符号误差 `Pred - GT`，使用 `RdBu_r` 等发散色图，高亮正负伪影。
- (c) 误差直方图（密度），右上角标注 MAE/RMSE 数值。
- (d) 指定行的 1D 强度轮廓线，**同时绘制 Input / Pred / GT 三条曲线**：
  - GT：绿色实线
  - Pred：蓝色实线
  - Input（如有）：灰色或虚线
  - 叠加 ±|err| 阴影带
  - **可选 FWHM 计算**：通过 `scipy.signal.find_peaks` + `peak_widths` 计算若干主峰的 FWHM（单位 µm），并在图上用彩色阴影框 + 文本标注 `FWHM GT ≈ xx µm / Pred ≈ yy µm`。

**命令行调用（随 `--mode single` 自动触发）**：

```bash
python plot.py --mode single \
  --pred_npy  ... \
  --gt_npy    ... \
  --noisy_npy ... \
  --profile_row 256 \
  --pixel_size_um 0.103 \
  --output_dir figures
```

内部等价调用：

```python
figure_error_analysis(
    pred, gt, noisy,
    profile_row=profile_row,
    pixel_size_um=pixel_size_um,
    sample_name=sample_name,
    output_path=Path("fig2_error.pdf"),
    compute_fwhm=True,  # CLI 默认开启 FWHM 计算
)
```

输出文件：

- `fig2_error.pdf`：四联图（误差地图 + 误差分布 + 1D 截线 + FWHM 标注）。

### 2.3 Figure 3：频域保真度分析

**核心函数**：`figure_frequency`

功能：

- (a–c) Input / Pred / GT 的 2D log-PSD（频谱强度图）。
- (d) 径向平均 PSD 曲线（log-log），对比噪声与重建的频率带保留情况。
- (e) `sSNR` 和 `FwPSNR` 条形图，定量展示频域指标（内部重用 `eval.py` 中的 `_spectral_snr` 和 `_fw_psnr`）。

通过 `--mode single` 自动生成 `fig3_frequency.pdf`。

---

## 3. Task 2：像素级保真度（1D 轮廓 + FWHM）

已经通过 `figure_error_analysis` 实现：

- **1D 空间强度轮廓**：`profile_row` 控制截线所在行，绘制 Input/Pred/GT 三条曲线，并做 0–1 归一化显示。
- **FWHM 计算**（可选，CLI 默认启用）：
  - 使用 `scipy.signal.find_peaks` 检测 GT 曲线中的若干主峰。
  - 使用 `scipy.signal.peak_widths(..., rel_height=0.5)` 计算半高宽（像素）。
  - 乘以 `pixel_size_um` 转为物理单位（µm），并在图中用阴影框和文本标注 GT / Pred 的平均 FWHM。

如需单独在代码中调用：

```python
from pathlib import Path
import numpy as np
from plot import figure_error_analysis

pred  = np.load("eval/images/full/000001_pred.npy")
gt    = np.load("eval/images/full/000001_gt.npy")
noisy = np.load("eval/images/full/000001_noisy.npy")

figure_error_analysis(
    pred, gt, noisy,
    profile_row=256,
    pixel_size_um=0.103,
    sample_name="HeLa nucleus",
    output_path=Path("fig2_error.pdf"),
    compute_fwhm=True,
)
```

---

## 4. Task 3：定量统计分析（PSNR/SSIM + Tolerance Curve）

### 4.1 单模型测试集统计图（箱线图 + 小提琴图）

**核心函数**：`figure_statistics_single_run`

输入：

- `metrics_csv`：由 `eval.py` 生成的 `metrics_{run}.csv`，每行对应一张测试图，至少包含 `psnr` 和 `ssim` 列。

输出：

- `fig3_stats.{fmt}`：两列子图
  - (a) PSNR 分布（小提琴 + box，中位数/四分位数 + 分布形状）
  - (b) SSIM 分布（同上）

**命令示例**：

```bash
python plot.py --mode stats \
  --metrics_csv eval_results/eval/metrics_full.csv \
  --output_dir figures --fmt pdf
```

或在代码中直接调用：

```python
from pathlib import Path
from plot import figure_statistics_single_run

figure_statistics_single_run(
    metrics_csv=Path("eval_results/eval/metrics_full.csv"),
    output_path=Path("fig3_stats.pdf"),
)
```

> 若已安装 `seaborn`，内部自动使用 `seaborn.violinplot` + `pandas`，否则回退到纯 `matplotlib` 实现。

### 4.2 稀疏度容忍度曲线（Tolerance Limit Curve）接口

**核心函数**：`figure_tolerance_curve`

你可以先在外部脚本中整理不同稀疏度下的指标均值和方差，然后调用该函数绘图。

接口定义：

```python
from pathlib import Path
from plot import figure_tolerance_curve

sparsity = [2, 4, 8, 16]
psnr_mean = [32.1, 31.4, 29.8, 27.5]
psnr_std  = [0.6, 0.7, 0.8, 1.0]
ssim_mean = [0.92, 0.90, 0.87, 0.82]
ssim_std  = [0.01, 0.015, 0.02, 0.025]

figure_tolerance_curve(
    sparsity_factors=sparsity,
    psnr_mean=psnr_mean,
    psnr_std=psnr_std,
    ssim_mean=ssim_mean,
    ssim_std=ssim_std,
    output_path=Path("fig3_tolerance.pdf"),
)
```

特性：

- x 轴为稀疏度因子（对数刻度、2 进制，标签显示为 `2×/4×/8×/...`）。
- 左 y 轴：PSNR（dB），折线 + 误差棒（std 或 CI）。
- 右 y 轴（可选）：SSIM，折线 + 误差棒。
- 适合作为 “性能随信噪比/稀疏度衰减” 的定量曲线。

> 后续如果需要，可以很容易扩展一个小辅助脚本，从多个 `summary_{run}.csv` 中自动解析 `sparsity_factor` 与 PSNR/SSIM 均值，再调用该函数画图。

---

## 5. 消融分析与训练曲线（已有）

这些功能已在 `plot.py` 中实现，与你当前需求密切相关但不需要额外改动：

- `figure_ablation`：读取 `eval/ablation_all.csv` 和 `ablation_pvalues.csv`，生成：
  - PSNR/SSIM 的箱线图 + 小提琴图（多 run 比较）
  - 多指标条形图（mean ± 95% CI），并可叠加显著性标记（p 值星号）。
- `figure_training`：读取 TensorBoard `event` 文件，画出：
  - Loss 曲线（train/val，标记 best-epoch）
  - PSNR 曲线（train/val）
  - 学习率调度曲线

命令示例可参考文件头部 docstring，也可以根据你后续需要再做定制。

---

## 5.1 混合训练（AI4Life-MDC25 large + small，测试集仍为 Denoising_Dataset）

训练时若使用 **AI4Life-MDC25「2D denoising (Nuclei)」** 的 large + small 混合数据，并**依旧用 Denoising_Dataset 的测试集**做验证与最终评估：

- **训练集**：`--train_data_root` 指向 `.../AI4Life-MDC25/2D denoising (Nuclei)/`  
  - 自动使用 `large_images/`（1024×1024，随机 256 crop）+ `small_images/`（256×256 成对，不裁剪），仅支持 `task=baseline`。
- **验证 / 测试集**：仍用 `--data_root` 指向 **Denoising_Dataset**（`Test/high`, `Test/low`）；`eval.py` 也始终用 `--data_root` 作为测试集路径。

**示例**：

```bash
python train.py --task baseline --model swin_unet_2d \
  --train_data_root "/data2/wjb/threePM/ThreePM_openDatasets/AI4Life-MDC25/2D denoising (Nuclei)" \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/Denoising_Dataset \
  --patch_size 256 --patches_per_image 50 \
  --run_name run_MDC25 --epochs 100
```

评估时照常指定 Denoising_Dataset 与对应 checkpoint：

```bash
python eval.py --task baseline --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/Denoising_Dataset \
  --checkpoint runs/run_MDC25/best.pth \
  --output_dir eval_results_MDC25 --patch_size 256
```

---

## 5.2 结果文件夹按数据集归档命名（便于区分）

训练/评估/可视化结果可按**数据集简写**归档，避免混淆：

| 简写   | 含义 |
|--------|------|
| **DD** | Denoising_Dataset（Zenodo Noisy Nuclei） |
| **MDC25** | AI4Life-MDC25 2D denoising (Nuclei)，混合 large+small 训练 |

**用法**：在项目根目录执行：

```bash
# 将当前 runs/eval_results/figures 归档为 *_DD（仅用 Denoising_Dataset 时）
bash scripts/archive_results_by_dataset.sh DD

# 混合训练后的结果归档为 *_MDC25
bash scripts/archive_results_by_dataset.sh MDC25
```

脚本会将 `runs/run` → `runs/run_<SUFFIX>`，`eval_results` → `eval_results_<SUFFIX>`，`figures` → `figures_<SUFFIX>`；若目标已存在则跳过。

---

## 5.3 验证集选项（独立 / 从训练划分 / 不验证）

训练时的验证逻辑由两个选项决定：

**① 是否有独立验证集**

- **`--val_root`**（可选）：若提供路径，且该目录下存在 **high/** 和 **low/**（与 Training 相同结构），则**直接使用该目录作为验证集**，不再从训练集划分。
- 例：`--val_root /path/to/Denoising_Dataset/Val`

**② 没有独立验证集时**

- **`--split_val_from_train`**（默认 **0**）：
  - **> 0**（如 **0.15**）：从训练“配对”中按该比例划分出一部分作为验证集（按对划分，用 `--seed` 固定），用于选 best 和监控；测试集仅在训练结束后用 `eval.py` 评估。
  - **= 0**：**不划分、不做监督验证**；全部训练数据用于训练，训练过程中不跑验证。训练结束后用 **测试集** 在 `eval.py` 中做一次测试即可。会保存 `last.pth` 和按训练 loss 的 `best.pth`。

总结：有独立验证集 → 用 `--val_root`；没有独立验证集 → 需要验证则 `--split_val_from_train 0.15`，不需要验证则保持默认 0，训练完再用测试集评估。

---

## 6. 如何串联整个 pipeline

1. **训练完模型**，得到若干 checkpoint（不同稀疏度 / 不同 ablation）。
2. 使用 **`eval.py`** 跑测试集评估，得到：
   - 每个 run 对应一份 `metrics_{run}.csv` + `summary_{run}.csv`。
   - 以及 `images/{run}/{stem}_*.npy`，供可视化使用。
3. 单张典型图像（如 best/median PSNR）：
   - 用 `plot.py --mode single` 生成：
     - `fig1_qualitative`（全局 + ROI）
     - `fig2_error`（误差图 + 1D 轮廓 + FWHM）
     - `fig3_frequency`（频域分析）
4. 整个测试集统计：
   - 用 `plot.py --mode stats --metrics_csv ...` 生成 `fig3_stats`（PSNR/SSIM 分布图）。
   - 如有多 run，对比使用 `plot.py --mode ablation --ablation_csv eval/ablation_all.csv`。
5. 若有多种稀疏度设置：
   - 从多个 `summary_{run}.csv` 中取出各稀疏度的 PSNR/SSIM 均值及误差，调用
     `figure_tolerance_curve` 画出 **极限容忍度曲线**。

这样，你就可以非常系统地、以论文要求的标准，展示：

- **定性**（Figure 1：全局+局部 Zoom-in）
- **像素级保真度**（Figure 2：误差地图 + 1D 截线 + FWHM）
- **统计显著性与稳健性**（Figure 3：PSNR/SSIM 分布 + Tolerance Curve）
- 以及可选的频域分析、消融、训练曲线等。

---

## 7. AI4Life-CIDC25 Training 数据三维显示

新增脚本：`scripts/visualize_cidc25_3d.py`

用途：

- 读取 CIDC25 的 `Training` 或 `Validation` 下单个 `*.tif` 体数据（`[F,H,W]`）
- 生成三视图（`XY / XZ / YZ`）图片
- 生成基于高强度体素的 3D 点云图（便于快速查看三维结构）

示例命令：

```bash
python scripts/visualize_cidc25_3d.py \
  --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --split Training \
  --stack A1 \
  --output_dir /data2/wjb/threePM/figures_cidc25_3d \
  --threshold 0.85 \
  --max_points 120000
```

主要参数：

- `--cidc_root`：`AI4Life-CIDC25` 根目录（应含 `Training/Validation`）
- `--split`：选择 `Training` 或 `Validation`
- `--stack`：栈文件名（不带 `.tif`），如 `A1`、`B1`、`C2`、`D2`、`F0`、`F1`
- `--threshold`：3D 点云阈值（归一化后），值越小点越多
- `--max_points`：点云最多采样点数，控制绘图速度

输出文件：

- `{split}_{stack}_orth_views.png`
- `{split}_{stack}_pointcloud3d.png`

若本机是无图形界面环境（如远程服务器，`DISPLAY` 为空），`--interactive` 不会弹窗。  
此时请使用网页交互模式：

```bash
python scripts/visualize_cidc25_3d.py \
  --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --split Training \
  --stack A1 \
  --output_dir /data2/wjb/threePM/figures_cidc25_3d \
  --threshold 0.85 \
  --web_interactive
```

将生成：

- `{split}_{stack}_interactive3d.html`（浏览器打开后可旋转/缩放/平移）

交互式浏览（推荐）：

```bash
python scripts/visualize_cidc25_3d.py \
  --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --split Training \
  --stack A1 \
  --output_dir /data2/wjb/threePM/figures_cidc25_3d \
  --interactive
```

交互操作：

- 在 `XY` 视图上滚动鼠标滚轮：快速切换 `z` 层
- 拖动 `z/y/x` 滑条：同步更新三视图切片
- 拖动 `thr` 滑条：实时调整 3D 点云阈值（高阈值更稀疏，低阈值更密）

---

## 8. CIDC25 三维稀疏重建 + PSF 物理先验

### 8.1 新增能力总览

| 功能 | 文件 | 说明 |
|------|------|------|
| 3D 稀疏数据生成 | `scripts/prepare_cidc25_sparse3d.py` | 从 CIDC25 原始栈生成 XY/Z/XYZ 三种稀疏模式 |
| 3D Volume Dataset | `dataset.py` → `CIDC25Volume3DDataset` | 直接加载栈、3D patch 裁剪、在线稀疏化 |
| Z 方向稀疏 warm-start | `dataset.py` → `warm_start_3d` | 支持独立 Z 轴 factor |
| 3D 稀疏 mask | `dataset.py` → `make_mask_3d` | 支持 xy/z/xyz 三种 sparse_mode |
| PSF 编码通道 | `dataset.py` + `models/__init__.py` | `--psf_input_channel` 在输入端加 PSF map |
| PSF 一致性损失 | `loss.py` → `PSFConsistencyLoss` | `--lambda_psf 0.1` 启用物理约束 |
| Wiener 预反卷积 | `dataset.py` → `wiener_init_2d/3d` | `--wiener_init` 替代 bilinear warm-start |
| 3D 训练路径 | `build_dataloaders` | `--cidc25_root` 接入 VideoSwinUNet 训练 |

### 8.2 3D 点云总览对比图（XY / Z / XYZ / GT）

与 `200um-f10-z5_sparse/comparison_3d_overview_pointcloud.png` 同风格（复用 `visualize_cidc25_3d.py` 点云渲染：`inferno`、阈值采样、`elev=28/azim=40`）。每种稀疏率 K 一张图，四列分别为 XY / Z / XYZ 稀疏体与 GT。

```bash
python scripts/plot_cidc25_sparse3d_pointcloud_overview.py \
  --sparse_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d \
  --gt_stem A1 --split Training --ks 2 3 4
```

输出：`CIDC25_sparse3d/comparison_3d_overview_pointcloud_K{2,3,4}.png`

### 8.3 3D 稀疏数据准备

```bash
python scripts/prepare_cidc25_sparse3d.py \
  --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --output_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d \
  --gt_stem A1 --split Training \
  --sparsity_factors 2 4 8 --modes xy z xyz
```

### 8.4 3D VideoSwinUNet 训练（XY 稀疏）

```bash
python train.py --task spatiotemporal_3d --model video_swin_unet \
  --cidc25_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --cidc25_gt_stem A1 \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d/K4_xy \
  --sparse_mode xy --sparsity_factor 4 \
  --volume_patch_t 16 --patch_size 256 --batch_size 2 \
  --n_frames 16 --epochs 50 --no_dino \
  --run_name sparse3d_3d_xy_K4
```

### 8.5 PSF 增强（物理先验）

```bash
# PSF 编码通道 + PSF 一致性损失
python train.py --task spatiotemporal_3d --model video_swin_unet \
  --cidc25_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --cidc25_gt_stem A1 \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d/K4_xy \
  --sparse_mode xy --sparsity_factor 4 \
  --psf_input_channel --lambda_psf 0.1 \
  --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
  --volume_patch_t 16 --patch_size 256 --batch_size 2 \
  --n_frames 16 --epochs 50 --no_dino \
  --run_name sparse3d_3d_xy_K4_psf_full
```

### 8.6 全量实验脚本

```bash
bash scripts/run_sparse3d_experiments.sh
```

包含 Phase 1（9 baseline runs）+ Phase 2（4 PSF 增强 runs）。

### 8.7 K2 全图评估：按稀疏模式的总览伪彩对比图

针对 `eval_sparse3d_k2_fullimg`，为 **xy / z / xyz** 各生成一张 2×5 总览图（全局 + ROI 放大），列顺序：

`稀疏输入 | 2D 重建 | 2.5D 重建 | 3D 重建 | GT`

重建列自动选用带 `--wiener_init`（及 2.5D 的 `mask_wiener`）的 checkpoint；配色与 `plot_pollen_structural_comparison.py` 一致（`magma` + 每列 `robust_range` 0.5–99.5% + colorbar）。

```bash
python scripts/plot_sparse3d_k2_overview_pseudocolor.py \
  --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --sparse_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d \
  --runs_dir runs \
  --output_dir eval_sparse3d_k2_fullimg/qualitative \
  --k 2 --frame 100 --modes xy z xyz
```

输出：

- **magma 版**（与 pollen 结构伪彩一致）：`eval_sparse3d_k2_fullimg/qualitative/overview_{mode}_K2_f{frame}_pseudocolor.png/pdf`
- **紫色版**（保留）：`eval_sparse3d_k2_fullimg/qualitative_purple/overview_{mode}_K2_f{frame}_pseudocolor_purple.png/pdf`

**K=4 全图评估目录** `eval_sparse3d_k4_fullimg/`：

```bash
# 仅评估 K=4 的 checkpoint
python scripts/eval_sparse3d_unified.py \
  --cidc_root .../AI4Life-CIDC25 \
  --sparse_root .../CIDC25_sparse3d \
  --gt_stem F0 --split Validation \
  --runs_dir runs --output_dir eval_sparse3d_k4_fullimg \
  --sparsity_k 4 --max_frames 200

# magma 对比图
python scripts/plot_sparse3d_k2_overview_pseudocolor.py \
  --k 4 --display-style magma \
  --output_dir eval_sparse3d_k4_fullimg/qualitative

# 紫色对比图
python scripts/plot_sparse3d_k2_overview_pseudocolor.py \
  --k 4 --display-style purple \
  --output_dir eval_sparse3d_k4_fullimg/qualitative_purple
```

---

## 9. 实验归档整理（2026-04-13）

为避免“探索失败/中间态实验”干扰最终论文主线，新增分层归档：

### 9.1 Stage2-3 未达预期实验归档

归档目录：

- `archive/failed_experiments_20260413/stage2_stage3_unsuccessful/`

已归档内容（示例）：

- `STAGE2_INTERIM_REPORT.md`
- `STAGE2_EXTENDED_REPORT.md`
- `STAGE3_REPORT.md`
- `STAGE3_REVIEW_GLM5.md`
- `stage2_pilot_data/`
- `stage3_logs/`

### 9.2 Stage4 PSF 未达预期变体归档

归档目录：

- `archive/failed_experiments_20260413/stage4_psf_unsuccessful/runs/`
- `archive/failed_experiments_20260413/stage4_psf_unsuccessful/eval/`

已归档变体：

- `sparse3d_3d_xy_K4_psf_ch`
- `sparse3d_3d_xy_K4_psf_loss`
- `sparse3d_3d_xy_K4_psf_full`

说明：

- 上述变体在当前预算下未优于 3D baseline，先归档保留，不纳入当前主线。
- `sparse3d_3d_xy_K4_wiener` 保留在主路径，作为有效物理先验方案。

### 9.3 归档索引

- `archive/failed_experiments_20260413/index/ARCHIVE_INDEX.md`

---

## 10. 成功探索主线文档（2D -> 2.5D -> 3D）

新增文档：

- `report/SUCCESS_PATH_2D_2P5D_3D.md`

该文档仅保留“有效/成功探索”，包含：

- 研究目标
- 总体架构
- 具体方法（含原理与公式）
- 关键实验结果与解释
- 主线收敛结论与后续建议

---

## 11. Phase 3A：K 值扫描与 XY 稀疏优化（2026-04-13）

本阶段目标：

1. 在 `K=2` 下补齐 `xy/z/xyz` 全模式实验（`run_sparse3d_k_sweep.sh` 已剔除 `K=3`，减少批量实验量）。
2. 补齐 2.5D 的显式 mask 先验，并加入 Wiener 初始化通道级对照。
3. 将 2D 路径扩展到 `--wiener_init`，统一与 3D 的预处理范式。

### 11.1 新增/更新能力

- `dataset.py`
  - 新增 `CIDC25Volume2p5DDataset`（直接从 CIDC25 体数据构造 2.5D 输入）：
    - 输入支持 `[T interp + T mask]`（默认）或仅 `[T interp]`
    - 支持 `sparse_mode=xy/z/xyz`
    - 支持 `wiener_init`
  - `NucleiDenoisingDataset` 增加 `wiener_init` 分支（2D warm-start / Wiener 二选一）
  - `build_dataloaders`：
    - `task=spatial_2p5d` 且传入 `--cidc25_root` 时，自动走 `CIDC25Volume2p5DDataset`
    - 保留原 `NucleiTemporal2p5DDataset` 路径以兼容旧数据组织
- `models/__init__.py`
  - `spatial_2p5d` 输入通道：
    - 旧路径（配对帧 `NucleiTemporal2p5DDataset`）：`in_ch = temporal_window`
    - 新路径（`--cidc25_root` + `CIDC25Volume2p5DDataset`）：默认 `in_ch = 2 * temporal_window`（interp + mask）
- `scripts/run_sparse3d_k_sweep.sh`
  - K 扫描脚本（当前策略）：
    - **仅 K=2**，mode=xy/z/xyz
    - **GPU 2 与 GPU 3 双 worker 并行**：对待跑任务按顺序轮询分配（2→3→2→3…），每个 GPU 上串行执行各自队列
    - baseline runs：`2d / 2p5d / 3d / 3d_wiener`
    - XY 增强 runs：`2d_wiener / 2p5d_mask / 2p5d_mask_wiener`
    - 数据集已齐备时跳过 `prepare`；已有 `runs/<name>/best.pth` 时跳过训练
- `scripts/eval_sparse3d_unified.py`
  - 支持 `2d + wiener`
  - 支持 `2p5d(mask/wiener)` 新路径评估
  - baseline 统计扩展为自动扫描所有可用 `K*_mode` 目录
- `scripts/plot_eval_results.py`
  - 新增 `fig6_k_sweep_psnr`
  - 新增 `fig7_optimization_gain`

### 11.2 推荐执行顺序

```bash
# 1) 训练 K 扫描与增强实验
bash scripts/run_sparse3d_k_sweep.sh

# 2) 统一评估（建议先用 Validation/F0）
python scripts/eval_sparse3d_unified.py \
  --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --sparse_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d \
  --gt_stem F0 --split Validation \
  --runs_dir runs --output_dir eval_sparse3d --max_frames 200

# 3) 绘图
python scripts/plot_eval_results.py \
  --eval_dir eval_sparse3d \
  --output_dir eval_sparse3d/figures
```

---

## 12. 花粉稀疏采样数据预处理（尺寸对齐 + Mask 生成）

新增脚本：`scripts/preprocess_pollen_sparse_restore.py`

用途：

- 将 `pollen_test/Sparse` 中被“压缩导出”的稀疏采样帧还原回 GT 尺寸（与 dense 一致）。
- 未扫描像素补 0。
- 同步生成二值 `mask`（1=实际扫描到，0=未扫描）。

支持的命名模式（从文件夹名自动解析）：

- `2xX` / `4xX`：仅 X 方向稀疏
- `2xY` / `4xY`：仅 Y 方向稀疏
- `2xXY` / `4xXY`：X、Y 同时稀疏

示例命令：

```bash
python scripts/preprocess_pollen_sparse_restore.py \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test \
  --output_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/preprocessed
```

输出结构（每个稀疏采集目录）：

- `{acq}/CellVideo1/CellVideo/CellVideo_restored.tif`：恢复到 GT 尺寸后的图像栈
- `{acq}/CellVideo1/CellVideo/CellVideo_mask.tif`：对应 mask 栈

说明：

- 脚本会自动识别并跳过 `dense` 采集目录。
- 若输出文件已存在，默认跳过；可加 `--overwrite` 强制覆盖。

### 12.1 构建二维汇总数据集（100 帧均值）

新增脚本：`scripts/build_pollen_sparse_2d_dataset.py`

用途：

- 将预处理后的每个采集栈（100 帧）沿时间维做均值，得到单张 2D 图像。
- 文件名自动去掉时间戳（例如 `pollen-fov1-2xX-32.5mW_2026-04-17_17-25-47` -> `pollen-fov1-2xX-32.5mW`）。
- 输出到 `Sparse_Datasets/`，并在 `Sparse_Datasets/mask/` 生成对应 2D mask。
- 自动纳入 GT（dense）二维均值图，构成可直接使用的 2D 数据集。

示例命令：

```bash
python scripts/build_pollen_sparse_2d_dataset.py \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test \
  --preprocessed_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/preprocessed \
  --output_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets
```

输出结构：

- `Sparse_Datasets/*.tif`：2D 稀疏/GT 图像
- `Sparse_Datasets/mask/*_mask.tif`：对应 2D mask

### 12.2 转换为标准测试结构（Test/low/high）

新增脚本：`scripts/build_pollen_test_low_high.py`

用途：

- 将 `Sparse_Datasets` 转为常用评估目录：
  - `Test/low`：稀疏 2D 图像
  - `Test/high`：对应 GT 2D 图像（同名配对）
- 额外输出：
  - `Test/mask/*_mask.tif`：low 的采样 mask
  - `Test/mask/*_gt_mask.tif`：对应 GT mask（全 1）

示例命令：

```bash
python scripts/build_pollen_test_low_high.py \
  --sparse_dataset_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets \
  --output_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets_LowHigh
```

输出目录：

- `Sparse_Datasets_LowHigh/Test/low/*.tif`
- `Sparse_Datasets_LowHigh/Test/high/*.tif`
- `Sparse_Datasets_LowHigh/Test/mask/*.tif`

### 12.3 花粉结构感知评估协议（推荐用于显微结构可信度）

旧版 `eval_pollen_sparse/summary_pollen.csv` 主要按全图 PSNR 排序，且 SSIM 为全图均值。对于显微图像，这容易被背景、亮度尺度和暗区域主导，可能出现“视觉结构恢复较好，但 SSIM 偏低”的冲突。

新增结构感知评估脚本：`scripts/eval_pollen_structural.py`

核心变化：

- 自动从 GT 生成前景结构 mask，只在花粉/细胞结构区域计算关键指标。
- 保留全图指标 `psnr_full` / `ssim_full` 作为参考。
- 新增结构指标：
  - `ssim_fg` / `ms_ssim_fg`
  - `ncc_fg`
  - `psnr_fg` / `mae_fg`
  - `*_cal`：在前景区域做线性强度匹配后再计算，强调结构一致性而非整体亮度尺度。
- 新增 `composite_score`：结构优先的综合分，用于选择 Best 2D / 2.5D / 3D。
- `SSIM >= 0.85` 只建议用于修正后的 `ssim_fg_cal` 或 `ms_ssim_fg_cal`，不建议直接套用旧版全图 SSIM。

运行结构评估：

```bash
python scripts/eval_pollen_structural.py \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets \
  --mask_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets/mask \
  --runs_dir /data2/wjb/threePM/runs \
  --output_dir /data2/wjb/threePM/eval_pollen_sparse_structural
```

生成结构排序定性图：

```bash
python scripts/plot_pollen_structural_comparison.py \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets \
  --mask_root /data2/wjb/threePM/ThreePM_openDatasets/pollen_test/Sparse_Datasets/mask \
  --eval_dir /data2/wjb/threePM/eval_pollen_sparse_structural
```

输出目录：

- `eval_pollen_sparse_structural/summary_structural.csv`
- `eval_pollen_sparse_structural/summary_structural.md`
- `eval_pollen_sparse_structural/predictions/`
- `eval_pollen_sparse_structural/qualitative_structural/`
- `eval_pollen_sparse_structural/qualitative_structural_pseudocolor/`


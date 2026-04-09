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


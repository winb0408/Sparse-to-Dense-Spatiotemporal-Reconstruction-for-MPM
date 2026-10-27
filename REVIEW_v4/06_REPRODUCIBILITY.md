# 复现说明

## 1. 环境

```
Python 3.10.19
PyTorch 2.6.0+cu124
GPU: NVIDIA RTX A6000（实验使用 GPU1/2/3；GPU0 有他人进程，从不使用）
解释器: /data2/wjb/anaconda3/envs/threePM/bin/python
临时目录: export TMPDIR=/data2/wjb/tmp   # 根分区 /tmp 曾被打满导致 dataloader 静默挂死
```

仓库根目录的 `main` / `dev` 分支是 legacy threePM，与本文档无关；一切命令都在
`PSRAS/` 下执行。

## 2. 数据

来源：AI4Life **CIDC25** challenge 训练集 TIFF（8 个栈，每栈 1500 × 490 × 490，
float16，归一化到 [0,1]，固定标定常数，无 GT 依赖）。预处理产物在
`PSRAS/runs_v3/data/cidc25_v3/`（`.npy`，未入库，`PSRAS/.gitignore` 已排除）。

split v2（DEC-025）：

| split | 内容 | 用途 |
|---|---|---|
| train | A1, B1, C2, D2（全帧） | 训练 |
| val | F0[0:750] | 模型选择 / 早期停止 |
| test | F0[750:1500] | 所有报告数字 |
| noise_stress | F1, F2, F3（与 F0 同内容，噪声递增；corr .71/.36/.18，残差 std .072/.180/.309） | EXP-027 |

val/test 同栈不同帧（内容独立），F1–F3 只作噪声变体、不算独立栈。所有数值结论都
在 test 上报告，val 口径作为一致性核对。

## 3. 指标口径（v4）

- PSNR：全图，逐 item 平均；
- SSIM：标准 2D 可分离高斯窗（11×11，σ=1.5），训练损失与评估共用同一实现；
- `e_measured` / `e_missing`：分别只在掩码 M=1 与 M=0 的像素上计算；
- 每个 run 有 `metrics_test*.json`（test）与 `val_metrics_corrected*.json`（val），
  统一网络按评估点分文件（`metrics_test_y_K4.json` 等）。

## 4. 代码地图

| 路径 | 职责 |
|---|---|
| `physics/acquisition.py` | 前向模型：PSF 模糊、采样掩码、泊松/高斯噪声、栅格相位 |
| `physics/psf.py` | PSF 配置与可分离 OTF |
| `physics/wiener.py` | 先验家族：`bilinear` / `wiener`（冻结 λ）/ `wiener_fill_first` / `iterdc:λ:steps` / `adaptive:λ0` |
| `data/datasets.py` | 运行时可复现的观测生成、确定性 val 网格、EXP-027 paired 模式 |
| `data/observation.py` | `(Y, M, X0, C_s, U_p)` 输入构造 |
| `models/` | `swin_unet_2d`（主线）、`unified_v4`（P4 统一架构）、`unified_3d` / `video3d`（已证伪，保留复现）、`ext`（外部基线） |
| `scripts/train.py` | 训练入口（含 `--regime_mix`、`--regime_weights`、`--resume`、`--ckpt_every`） |
| `scripts/reval.py` | 重评估（`--split`、`--protos`、`--noise_stack`、`--alias`） |
| `scripts/paper_figs.py` | 论文级图（quant / qual_yz / qual_unified） |
| `scripts/visualize_v3.py` | v3 可视化协议（panel / curve / pcolor / fusion） |
| `scripts/visualize_noise_stress.py` | EXP-027 曲线与面板 |
| `scripts/exp023_prior_sweep.py` | 先验独立评估（1008 行） |

## 5. 训练与评估命令

```bash
cd PSRAS
export TMPDIR=/data2/wjb/tmp

# EXP-024 最优点（b4f，YZ (2,8)）
python scripts/train.py --preset b4f --regime 2.5d --mode yz2 --K 2 --kz 8 \
    --steps 8000 --name YZ_2x8_b4f

# EXP-025/026 统一模型（15 点混合，T 无关架构）
python scripts/train.py --preset b4f --model univ4 --regime 2d \
    --regime_mix 'y:2,4,8;yz2:2x2,2x4,2x8,4x2,4x4,8x2;xyz2:2x2,2x4,2x8,4x2,4x4,8x2' \
    --steps 8000 --name UNIV4_U15_s0

# 重评估（双口径）
python scripts/reval.py --run runs_v3/YZ_2x8_b4f --split test
python scripts/reval.py --run runs_v3/YZ_2x8_b4f --split val

# EXP-027 噪声配对评估
python scripts/reval.py --run runs_v3/UNIV4_U15_s0 --noise_stack F1 --alias y_K4

# 图
python scripts/paper_figs.py quant
python scripts/paper_figs.py qual_yz
python scripts/paper_figs.py qual_unified
```

## 6. 产物组织

```
runs_v3/<run>/          config.json, log.csv, metrics_test*.json,
                        val_metrics_corrected*.json, best.pt（未入库）
runs_v3/exp023_prior_sweep/summary.csv     1008 行先验扫描
runs_v3/exp027_x0_prior_noise.json        噪声下 x0 先验 PSNR
eval_viz/paper_figures/                   论文级图 + INDEX.md + key_numbers.csv
eval_viz/exp027_noise_stress/             noise_curve + 5 panel
eval_viz/exp028_regime_weight/            三套件加权对比曲线
```

`runs_v3/` 本地 17 GB（权重），入库的只有 config 与指标（2022 个文件）。

## 7. 复现性注意事项

1. **config 记录瑕疵**：mixed（统一）run 的 `psf.sigma_z_px` 记为 0.0（顶层 regime
   为 `2d` 所致），训练时 2.5D 点实际用 `--psf_z 1.5`。影响已实证可忽略（EXP-027
   结果 5），但复现定性图时应显式用 1.5（`paper_figs.py` 已如此处理）。
2. **GPU 竞争**：共享卡，他人进程会挤爆 run。所有队列脚本有 `wait_idle` 保护，
   完成判据以 `final_val.json` 存在性为准（bash exit code 对 OOM 会误报 0）。
3. **seed**：seed 只影响训练随机性（权重初始化、batch 顺序）；val/test 的采集种子由
   `(split, stack, t0)` 派生，跨 run 完全一致，所以逐 seed 配对差是有效的。
4. **checkpoint-resume**：`--ckpt_every` 每 N 步原子快照 `resume.pt`，`--resume` 恢复
   model/opt/sched/scaler/step/best_psnr/mix_rng；dataloader 的批次序列**不**恢复
   （已声明的限制）。

# E024：双先验阶梯的第一批真实数字

> **状态（2026-11-11 16:40）：** residual 阶段完成；`dual` 与 `controls` 仍在跑。
> 本文件只写**已经落盘**的数字，未完成的臂在表里标 `RUNNING`，跑完后回填。
> 单折（fold 0）、单种子（s=0）、单一测试动物（M2）——每个数字都要带着这句话读。

## 1. 协议：所有臂共用的那一半

| 项 | 值 |
|---|---|
| 数据 | Dryad 三光子队列，fold 0 训练动物 M1/M3，测试动物 M2 |
| 稀疏观测 | `y_k2`：沿 y 每 2 行取 1 行，理想采样率 50 %（M 之外一条数据都不给） |
| 训练 | 3000 步、batch 8、AdamW lr 2e-4（100 步 warmup + cosine）、AMP、patch 256×256 |
| 损失 | L1 + 0.5 × MS-SSIM（**与旧臂 `A*/S*` 一致**，T3 的 Charbonnier/缺失区加权尚未启用） |
| 骨干 | `base=48`、`window=8`、`depths=(2,2,2)`、`heads=(4,8,16)`、`context=1` |
| 模型选择 | 验证集 `dense_psnr`（patch 级），取最优保存 `best.pt` |
| 评估 | test 划分 **758 个 512×512 窗口**，全部窗口都有缺口（`n_windows_fully_measured=0`） |
| 物理先验 | identity PSF（依据见 `docs/PHASE1_PR2_PSF_ESTIMATE.md`），`lam=0.01` |

**`psnr_measured` / `psnr_missing` 是这张表里最重要的一对列**，不是装饰：
前者只在被测量体素上算，后者只在缺口上算。总 PSNR 会把两者混在一起，而方法之间
真正的差别只在缺口上——测量值谁都能逐位复现（插值地板就是 120 dB 封顶）。

## 2. 结果（fold 0，758 窗口）

| 臂 | 参数量 | PSNR | PSNR(measured) | PSNR(missing) | SSIM | MAE | Dice | Dice★ | clDice | pred_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| `P2_I0` 纯插值地板 | 0 | 27.8118 | 120.0 | 24.8015 | 0.6891 | 0.02260 | 0.7440 | 0.7433 | 0.7378 | 0.1871 |
| `P2_W0` 插值+Wiener | 0 | 27.8118 | 120.0 | 24.8015 | 0.6891 | 0.02260 | 0.7440 | 0.7433 | 0.7378 | 0.1871 |
| `P2_Direct` 直接重建 | 1.71 M | 29.2342 | 48.90 | 26.2581 | 0.7572 | 0.02021 | 0.7347 | 0.7618 | 0.7282 | 0.1129 |
| `P2_ResidualI` 残差锚在 I0 | 1.71 M | 29.4132 | 70.94 | 26.4031 | 0.7629 | 0.01890 | 0.7366 | 0.7683 | 0.7317 | 0.1041 |
| `P2_DualI` 双分支（I0） | 2.97 M | `RUNNING` | | | | | | | | |
| `P2_DualW` 双分支（W0） | — | `RUNNING` | | | | | | | | |
| `P2_DualW` ctrl=zeros | — | `RUNNING` | | | | | | | | |
| `P2_DualW` ctrl=shuffle | — | `RUNNING` | | | | | | | | |

参考：`ref_fraction = 0.1676`，`vessel_threshold = 0.03455`（该阈值由参考掩膜拟合，
不是逐臂调的）。证据：`runs/E024_dual_prior/*_test/summary.json`；
`ci_estimable=false`——单动物，区间退化。

## 3. 这张表能读出什么

**能读（超过声称阈值的）**

- **缺口填充的增益是实的**：相对插值地板，缺口 PSNR `P2_Direct` +1.46 dB、
  `P2_ResidualI` +1.60 dB；总 PSNR +1.42 / +1.60 dB。都超过 1 dB 阈值。
  SSIM +0.068 / +0.074、MAE −0.0024 / −0.0037，方向一致。
- 这条增益**不来自 Wiener**：`P2_I0` 与 `P2_W0` 逐位相同，因为本队列测出来是
  identity PSF（见 PSF_ESTIMATE 文档）。所以增益来自**学习到的先验**，不是物理反卷积。

**不能读（低于阈值，一律不声称）**

- `P2_Direct` → `P2_ResidualI`：缺口 PSNR +0.15 dB、MAE −0.0013、SSIM +0.0057、
  Dice +0.0019。**全部低于 1 dB / 0.03 Dice 阈值**，因此这一轮只能说
  "在这个预算下未观测到残差锚点的独立贡献"，不能说"残差锚点无用"。
- Dice / clDice 相对地板的变化：`P2_Direct` −0.0093 / −0.0096、`P2_ResidualI`
  −0.0074 / −0.0061。方向是**变差**，但幅度低于阈值，两个方向都不能声称。

**尤其不能读**

- 不能读成"新方法不如插值"。强度的增益是明确的，结构指标的变化在噪声内。
- 不能读成跨动物结论。单折单种子，`ci_estimable=false`。

## 4. 一个必须先说明的诊断：模型在**欠分割**

`pred_fraction` 那一列：地板 0.1871（比参考 0.1676 略胖），`P2_Direct` 0.1129、
`P2_ResidualI` 0.1041——**训练后的臂把血管面积少判了 33 % / 38 %**。

这解释了一件否则会看起来自相矛盾的事：

| 臂 | Dice（不加星） | Dice★（等面积重取阈） |
|---|---|---|
| `P2_I0` / `P2_W0` | 0.7440 | 0.7433 |
| `P2_Direct` | 0.7347 ↓ | 0.7618 ↑ |
| `P2_ResidualI` | 0.7366 ↓ | 0.7683 ↑ |

**两列的符号是相反的**，而这正是 `docs/RESULTS.md` 里记着的口径陷阱：不加星的
Dice 用参考拟合出的固定阈值，预测的整体对比度一变，面积就跟着变；加星的 Dice★
先把面积对齐，量的是形状。这里模型的强度更准（PSNR 高）但整体偏低，于是固定
阈值下切出来的面积少，Dice 掉；一旦按面积对齐，Dice★ 反而升 0.019 / 0.025。

**两个方向的差值都低于 0.03，所以这一轮不声称结构改善**。但诊断本身是有信息量的：
增益集中在强度域，二值化后没有同步改善，且面积是系统性偏小而不是随机散布。
这给 T3 的面积/拓扑约束提供的是**实测理由**，不是计划书上的待办项。

## 5. 显存与占卡（容量数字，不是性能数字）

| 臂 | 实测占用（进程级，`nvidia-smi`） | 记录值 |
|---|---|---|
| `P2_Direct` / `P2_ResidualI` | 13 394 MiB | 这两次启动早于 `gpu_peak_*` 记录代码，故 `final_val.json` 里没有 |
| `P2_DualI` | **19 489 MiB**（45 176 − 别人 25 687） | `final_val.json::gpu_peak_reserved_mb` |

dual 臂比 residual 臂大 **45 %**，这是排期的硬约束：本机只有 GPU0 在残差臂收工后
能腾出 ~23 GB，GPU3 只有 15.9 GB、GPU1 4.6 GB、GPU2 占用 53.5 %（高于 50 % 不启用），
所以 dual 与 controls **只能串行**在 GPU0 上排队。

四张卡都有别人的作业且利用率 100 %，因此：
**精度结论有效；延迟、吞吐、峰值显存一律不作为性能数字报告。**
本轮**不运行** `scripts.benchmark`——它输出的正是这三样，共卡条件下这些数字既不可
比也不可复现，跑出来只会给表里添一行不能用的话。等有一张真正的独占卡再补，届时
`benchmark.json` 里的 `P2_*` 会自动进 Fig. 2 的实测栏。

## 6. 预登记：dual 臂必须满足的一个结构检查

模型契约里 hard consistency 是**在模型内部**做的（`test_hard_consistency_returns_the_measurement_exactly_where_it_exists`），
因此 `P2_DualI` / `P2_DualW` 的 `psnr_measured` 必须等于 **120.0**（`psnr()` 在
mse=0 时封顶 120 dB）。若跑出来不是 120，说明约束没生效，该 run 不能进表——
这是写在这里、事后再看结果的检查，不是事后解释。

## 7. 复现

```bash
# 地板（零参数，无需训练）
GPUS=0 DRY=0 STAGE=floors bash spur_mpm/experiments/run_e024.sh

# 训练阶梯；SKIP_DONE=1 表示只补缺的
SKIP_DONE=1 STAGE=residual GPUS=0,3 DRY=0 bash spur_mpm/experiments/run_e024.sh
SKIP_DONE=1 STAGE=dual     GPUS=0   DRY=0 bash spur_mpm/experiments/run_e024.sh
SKIP_DONE=1 STAGE=controls GPUS=0   DRY=0 bash spur_mpm/experiments/run_e024.sh
```

## 8. 本轮没有的

- 折 1–2、多种子：`NOT_RUN`（首轮只做 fold 0 以确认链路）
- `z_k4` / `yz_k2x2` 等成本协议：`NOT_RUN`
- 整卷 3D 骨架报告与定性图：`NOT_RUN`（等全部臂落地后统一出图，同一 ViewSpec、同一强度窗）
- `P2_DualWU` / `P2_DualWUT` / `P2_TopoConf`：`TODO`（分别需要标定 H 与 T3/T5）

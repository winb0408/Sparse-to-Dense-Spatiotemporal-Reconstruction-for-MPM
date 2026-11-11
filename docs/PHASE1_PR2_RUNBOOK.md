# Phase 1 PR2 运行手册：E024 双先验阶梯

> 方法依据：`docs/PHASE1_PR2_METHOD_SPEC.md`。任务依据：`docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md` §6、§7。
> **当前状态：** 地板与首批真实训练已执行（见 `docs/PHASE1_PR2_ACCEPTANCE.md` 的实际记录）；
> PSF 侧已有独立证据（`docs/PHASE1_PR2_PSF_ESTIMATE.md`）。本手册描述命令，执行结果不回填在这里。

执行前提：`conda activate threePM`（python 3.10.19 / torch 2.6.0+cu124），工作目录 `/ssd1/wjb/SPUR-MPM`。

---

## 0. 开跑前的四项检查（每次都要做）

```bash
# (1) 回归测试必须全绿。EXIT=0 是唯一信号：pytest.ini 里已经带 -q，命令行再加一个
#     -q 会变成 -qq 并把 "N passed" 汇总行吞掉，所以必须用 -o addopts="" 覆盖掉它。
python -m pytest spur_mpm/tests -o addopts="" -q; echo "EXIT=$?"

# (2) GPU 必须真的空闲。看**显存**判断能不能跑，看**利用率**判断能不能报性能：
#     与别人的作业共卡时，训练精度结论仍然有效，但延迟/吞吐/峰值显存一律不可报。
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
          --format=csv

# (3) 物理先验的来源必须写在命令里。identity 之外的 H 缺 psf_source 会被第一个 batch 拒掉；
#     本队列没有标定，自估值用 estimated_from_data，理由见 PSF_ESTIMATE 文档。
#     (3a) 复核估计器（CPU，约 8 分钟，只碰训练动物）
python -m spur_mpm.scripts.estimate_psf --fold 0 --n-patches 6 \
  --out runs/E024_dual_prior/psf_estimate

# (4) 明确要跑什么，先看 dry-run
DRY=1 STAGE=all bash spur_mpm/experiments/run_e024.sh
```

第 (4) 步的 dry-run 是默认行为：`run_e024.sh` 不给 `GPUS=` 就只打印、不执行，
也不存在"静默回退到 GPU 0"的分支。这条是 T0 归档纪律在训练侧的对应物。

`run_e024.sh` 的两级顺序：**跨阶段严格串行**（residual → dual → controls，前一阶段
全部成功才进下一阶段），**同阶段多臂并行**（同一阶段内各臂互相独立，按 `GPUS=`
轮转铺开）。前一阶段任一 run 失败，脚本以非零码退出而不是继续——"在缺失的地板上
继续往上堆"正是这个顺序要防的事。

---

## 1. 零参数地板（不需要训练，最先跑）

```bash
GPUS=0,1,2 DRY=0 STAGE=floors bash spur_mpm/experiments/run_e024.sh
```

跑 `P2_I0`（纯插值）与 `P2_W0`（`wiener_fill_all`）在 test 划分上的指标，
产物在 `runs/E024_dual_prior/E024_P2_{I0,W0}_y_k2_fold{0,1,2}_test/summary.json`。

### 1.1 预期的结果，以及它意味着什么

在 identity PSF 下 `W0 ≡ I0`（代码走 `i0.clone()` 快路径，不跑 FFT），
所以两臂的 summary 应当**只有 `arm` 字段不同**。这不是 bug，是本队列能回答什么
的边界：**它证明的是缺口填充，不是反卷积。**

如果两臂出现了不一致的数字，说明协议错了（PSF 被意外设成非恒等、λ 变化、
或 warm-start 模式没传对），应当停下查，而不是当作"Wiener 有增益"报出去。

### 1.2 非恒等 H 的分组（当前不存在，禁止混表）

要检验反卷积，必须先有 `calibrated_optical` 或明确标为
`synthetic_extra_degradation` 的 H：

```bash
# 压力测试分组，产物必须落在独立目录，不得进 E024 主表
OUT=runs/E025_psf_stress GPUS=0 STAGE=floors DRY=0 \
  bash spur_mpm/experiments/run_e024.sh \
  --set physics.psf.sigma_xy_um=1.5 --set physics.psf.sigma_z_um=4.0 \
  --set physics.psf_source=synthetic_extra_degradation
```

未标 `psf_source` 的非恒等 H 会在第一个 batch 抛 `ValueError`
（`test_the_item_config_carries_the_psf_source_to_the_prior`），这是设计如此。

### 1.3 已实际执行的冒烟（2026-11-10）

在真实 test 卷（fold 0，M2）上以 `patch_hw=[32,32]`、CPU 各跑一遍两臂：

| 臂 | PSNR | SSIM | MAE | 证据 |
|---|---|---|---|---|
| `P2_I0` | 28.02529527 | 0.7503781 | 0.02258195 | `docs/evidence/smoke_floors_20261110/P2_I0_summary_32px_cpu.json` |
| `P2_W0` | 28.02529527 | 0.7503781 | 0.02258195 | `docs/evidence/smoke_floors_20261110/P2_W0_summary_32px_cpu.json` |

`diff` 后仅 `"arm"` 一行不同。**这两个数字不是头条结果**（32×32 窗口、CPU、单折），
它们的作用只有一个：在真实数据上确认 identity 退化成立，而不是只在合成测试里成立。
正式地板数必须用默认 256×256 重跑。

---

## 2. 训练阶梯（按顺序，不可跳）

```bash
GPUS=0,1,2 DRY=0 STAGE=residual bash spur_mpm/experiments/run_e024.sh   # P2_Direct, P2_ResidualI
GPUS=0,1,2 DRY=0 STAGE=dual     bash spur_mpm/experiments/run_e024.sh   # P2_DualI,  P2_DualW
GPUS=0,1,2 DRY=0 STAGE=controls bash spur_mpm/experiments/run_e024.sh   # P2_DualW x {zeros, shuffle}
```

顺序的理由（每个"先"都对应一个必须先存在的对照）：

| 顺序 | 跑什么 | 必须先有的是 | 这一步能回答的问题 |
|---|---|---|---|
| 1 | `P2_I0` / `P2_W0` | — | 零学习地板在哪 |
| 2 | `P2_Direct` | 1 | 新模型在不加残差锚点时的水平 |
| 3 | `P2_ResidualI` | 2 | 残差锚在 `I0` 上本身买到什么 |
| 4 | `P2_DualI` | 3 | 第二个编码器 + 容量买到什么 |
| 5 | `P2_DualW` | 4 | `W0`、`phys_delta` 的**信息**买到什么 |
| 6 | controls | 5 | 第 5 步的增益里，多少只是"多了一个编码器" |

第 5、6 步的读法（写在文档里，避免事后解释）：

- `DualW > DualI` 且 `DualW > ctrlzeros` 且 `DualW > ctrlshuffle` → 可以说是**信息**的贡献；
- 只有 `DualW > ctrlzeros` 而 `shuffle` 持平 → 只能说边缘分布有用，空间对应没用；
- 三者都在阈值内 → 如实写"未观测到独立贡献"。按取数纪律，PSNR 差 <1 dB、
  Dice 差 <0.03 一律不声称。

第 6 步的 run 名带 `_ctrlzeros` / `_ctrlshuffle` 后缀，`model.phys_control` 也已写进
`config.yaml` 与 `config_hash`，所以控制组不会和头条组撞名或被误当重复实验。

### 2.1 本轮不启动的臂

| 臂 | 为什么现在不跑 |
|---|---|
| `P2_DualWU` | identity PSF 下 `u_physics ≡ 0`，门控退化为常数，等价于 `P2_DualW`；只在有标定 H 的分组里才有意义 |
| `P2_DualWUT` | 拓扑/结构损失，T3 未实现 |
| `P2_TopoConf` | `tasks/vessel/` 适配器，T5 未实现 |

---

## 3. 多种子与多协议

首轮默认 `SEEDS=0`、`FOLDS="0 1 2"`（三折各留一只测试鼠）。阶梯出结果后，
**只对进入下一轮的臂**补种子：

```bash
STAGE=dual GPUS=0,1,2 DRY=0 SEEDS="0 1 2" bash spur_mpm/experiments/run_e024.sh
```

三种 25% 预算的等成本对比（`y_k4` / `z_k4` / `yz_k2x2`，7 层上下文）用同一份脚本换协议参数：

```bash
OUT=runs/E024_matched SEEDS=0 GPUS=0,1,2 DRY=0 STAGE=dual \
  bash spur_mpm/experiments/run_e024.sh \
  --set sampling.mode=yz --set sampling.ky=2 --set sampling.kz=2 \
  --set data.context_slices=7
```

`z` / `yz` 协议下 `context_slices` 必须 ≥3，且窗口只能从真实稀疏层构造输入
（`valid_z` 会把 padding 标成非测量）。`context=1` 跑 z 跳层在本仓库是拒绝的。

---

## 4. 评估、结构与定性图

```bash
# 定量（脚本已在上一步自动执行；这里是重跑单个 run 的形状）
python -m spur_mpm.evaluate --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --arm P2_DualW --ckpt runs/E024_dual_prior/E024_P2_DualW_y_k2_fold0_s0/best.pt \
  --fold 0 --split test --out runs/E024_dual_prior/E024_P2_DualW_y_k2_fold0_s0_test \
  --set data.index=data_local/dryad_cache/index_cohort.json \
  --set data.split=spur_mpm/experiments/E020_cohort/split.json \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1

# 整卷 3D 骨架（主结构指标；与上面的 2D 窗口 Dice 不可同列比较）
python -m spur_mpm.scripts.structure_report --arm P2_DualW \
  --ckpt runs/E024_dual_prior/E024_P2_DualW_y_k2_fold0_s0/best.pt \
  --out runs/E024_dual_prior/E024_P2_DualW_y_k2_fold0_s0/structure \
  --device cuda:0 \
  --set data.index=data_local/dryad_cache/index_cohort.json \
  --set data.split=spur_mpm/experiments/E020_cohort/split.json --set data.fold=0 \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1

# 固定视角定性图：所有臂共用同一个 ViewSpec，同一个强度窗
python -m spur_mpm.scripts.qualitative_figures \
  --config spur_mpm/configs/experiments/E002_supervised_y_k2.yaml \
  --index data_local/dryad_cache/index_cohort.json --fold 0 \
  --sample M2_FOVI_S1_vessels \
  --arm A0 --arm P2_I0 \
  --arm P2_Direct=runs/E024_dual_prior/E024_P2_Direct_y_k2_fold0_s0/best.pt \
  --arm P2_ResidualI=runs/E024_dual_prior/E024_P2_ResidualI_y_k2_fold0_s0/best.pt \
  --arm P2_DualI=runs/E024_dual_prior/E024_P2_DualI_y_k2_fold0_s0/best.pt \
  --arm P2_DualW=runs/E024_dual_prior/E024_P2_DualW_y_k2_fold0_s0/best.pt \
  --out runs/E024_dual_prior/figures/qual_y_k2 \
  --set data.split=spur_mpm/experiments/E020_cohort/split.json \
  --set sampling.mode=y --set sampling.ky=2 --set sampling.kz=1
```

三条出图纪律（沿用上一轮修复后的口径）：

1. 同一比较组共用冻结的 ViewSpec 与强度窗；禁止逐模型自动增强对比度。
2. `A0` 与 `P2_I0` 是同一算子的两个入口，应当逐像素相同——若不同，说明其中一条路径
   的输入构造不一致，先查再出图。
3. 每根柱子、每张图都要能回指 `config.yaml` / `config_hash` / `best.pt` / seed。

---

## 5. 汇总与成本

```bash
python -m spur_mpm.scripts.ladder_report --root runs/E024_dual_prior \
  --out runs/E024_dual_prior/ladder_report.md \
  --csv runs/E024_dual_prior/ladder_report.csv

python -m spur_mpm.scripts.benchmark \
  --benchmark runs/E024_dual_prior/benchmark.json \
  --complexity runs/E024_dual_prior/complexity.json
```

计时口径：`(a)` 网络前向纯 CUDA 耗时、`(b)` 插值+Wiener+模型+后处理的算法耗时、
`(c)` 峰值显存，分开报；**不含磁盘读写**。整卷端到端 wall time 单独写在
`inference` 输出的 `timing.wall_seconds`，不能拿来当单位 patch 延迟。

同步重绘 Fig. 2（`P2_*` 那一栏会从 `NOT_RUN` 自动变成实测值，只要
`benchmark.json` 里出现对应臂名，无需手改名单）：

```bash
python -m spur_mpm.scripts.framework_figure \
  --benchmark runs/E024_dual_prior/benchmark.json \
  --complexity runs/E024_dual_prior/complexity.json
```

---

## 6. 记录规则（缺一项则该 run 不能进主表）

每个 run 目录必须有：`config.yaml`（含解析后的 `phys_control` / `consistency` /
`n_parameters`）、`provenance.json`（数据与 split 的 hash）、`log.jsonl`、
`best.pt`、`final_val.json`。评估目录必须有 `summary.json`，其中的 `physics`
块带 `psf_source`，`checkpoint` 字段非空（除零参数地板）。

新 run 不得覆盖 E021/E023；E024 的编号如被占用则顺延并在
`docs/PHASE1_PR2_ACCEPTANCE.md` 记录映射。

真实数据未跑的格子写 `NOT_RUN`，不写 `n/a`、不写 `COMPLETE`。

---

## 7. 失败时怎么办

| 现象 | 含义 | 处置 |
|---|---|---|
| 第一个 batch 抛 `psf_source` | 非恒等 H 未标来源 | 在设计该实验的配置里标，不要在调用点塞默认值 |
| `W0` 与 `I0` 地板数不一致（identity 下） | 输入构造路径分叉了 | 停下查 `build_model_inputs`，不要写成"反卷积增益" |
| 双分支梯度在第 0 步全零 | 预期行为（零初始化 head + 零初始化投影） | 不是 bug；看第 1 步后是否两边都有梯度 |
| `no_checkpoint` | 评估链路的硬错误 | 整个评估失败，不允许标 successful |
| 某折 `Dice` 掉但 `Dice★` 没掉 | 面积配准掩盖了中心线误差 | 以普通 Dice/clDice 为准，Dice★ 只作辅助单列 |

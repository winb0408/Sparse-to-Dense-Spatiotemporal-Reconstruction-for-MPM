# 本分支的结果快照（E020 队列 / E021 阶梯 / E022 稀疏轴）

这个仓库的正常规则是**结果不入库**（`.gitignore` 里 `runs/` 整目录忽略）。这一批是
强行加入的只读快照，为了在没有数据、没有 GPU 的机器上也能审阅：所有数字都能在
`runs/E021_ladder/` 里回查到产生它的 `summary.json`，不用相信任何一处抄写。

## 该看什么

| 想看什么 | 文件 |
|---|---|
| 主表：30 个 cell × 10 个指标 + 归因 + 预算曲线 + 换轴表 | [`runs/E021_ladder/ladder.md`](../runs/E021_ladder/ladder.md) |
| 同一批数字的机器可读版 | `runs/E021_ladder/ladder.csv`、`ladder.json` |
| 参数量 / MACs / 延迟 | `runs/E021_ladder/complexity.md` |
| 定性图（稀疏输入 / A0 / A1 / S1 / S2 / 密集真值 + 误差 + 支撑 + 深度剖面） | `runs/E021_ladder/figures/*.png` |
| 每张定性图的逐臂数字（PSNR / Dice / Dice★）与几何 | `runs/E021_ladder/figures/*.json` |
| 每个 cell 的完整评估（逐窗 `samples.csv` + 汇总 `summary.json`） | `runs/E021_ladder/E021_*/eval_test/` |
| 每个 cell 的实际配置（稀疏参数、上下文层数、训练超参） | `runs/E021_ladder/E021_*/config.yaml` |
| 方法学与结论（含每处失败模式的测量） | [`Project log/Project_log.md`](../Project%20log/Project_log.md) 的 E020 / E021 / E022 |
| 代码地图 | [`docs/CODE_MAP.md`](CODE_MAP.md) |

## 一句话结论

- **可主张**：Y 方向行稀疏在 25 % / 12.5 % 预算上，监督重建的血管结构优于双线性
  （Dice★ +0.040 / +0.059，clDice★ +0.067 / +0.097）；灰度 PSNRmiss 全部九级涨 1.0–1.5 dB。
- **不可主张**：Z 轴与 Y+Z 轴的结构增益；50 % / 33 % 预算上的结构增益；三个创新点
  （X0 / U / 门控融合）的任何独立贡献。
- 门槛：跨动物方差约 2.8 dB / 0.08 Dice，所以 < 1 dB 与 < 0.03 Dice 的差异不写结论。

## 没有入库的东西

| 没有 | 大小 | 怎么重建 |
|---|---|---|
| 检查点 `best.pt` / `last.pt` / `resume.pt` | 每 cell ≈ 10 / 30 / 30 MB | `python -m spur_mpm.scripts.run_ladder --stage …`（见 CODE_MAP §6） |
| 定性图的 `.npz` 体数据转储 | 单张 ≈ 350 MB | `python -m spur_mpm.scripts.qualitative_figures …` |
| Dryad 3PM 原始数据与缓存 | 数十 GB | 见 `spur_mpm/scripts/prepare_data.py` 与日志 E000 |

## 环境

conda env `threePM`，python 3.10，PyTorch。测试：`python -m pytest spur_mpm/tests -q`
（149 项，全绿；本轮新增的有 `test_run_ladder_state.py` 与 `test_ladder_report.py`）。

## 一个必须知道的口径陷阱

参考阈值拟合出的 `Dice` 与等面积重取阈的 `Dice★` 在**同一批预测**上方向相反。差别全部
来自预测的对比度漂移，不是结构差异。看表时以 ★ 列为准，但 ★ 必须与不带星的列并读
（`metrics/vessel.py::match_foreground` 的 docstring 记了它自己的局限，以及一个随机
重叠基线 ≈ 0.19 作为量级自检）。

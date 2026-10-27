# 数值索引

## `v4_runs.csv`（543 行）

每个 run × 每个评估点一行。列：

| 列 | 含义 |
|---|---|
| `run` | run 目录名（对应 `PSRAS/runs_v3/<run>/`） |
| `experiment` | EXP-022 / EXP-024 / EXP-025/026/028 / v3-legacy |
| `regime`,`mode`,`K`,`K2`,`kz`,`t` | 协议参数（K2 = 平面周期，仅 yz2/xyz2 有） |
| `preset`,`warm_start`,`lam` | arm 配置（warm_start 即 x0 的配方） |
| `n_features` | 输入通道组数（b1=2, b2=3, b4*=5） |
| `tag` | 评估点标签（`y_K4`、`yz2_K2x8`、`xyz2_K4x4` …）；YZ 主矩阵为空 |
| `test_psnr`,`test_ssim` | test split（F0[750:1500]） |
| `val_psnr` | val split（F0[0:750]） |

注意：`v3-legacy` 行属于协议 v3，口径不同（split v1、旧 SSIM 窗），**不可与 v4
行同表比较**，仅为完整性保留。

## `key_numbers.csv`（84 行）

只包含论文图表用到的数字，`kind` 列区分：

- `yz_arm`：EXP-024 主矩阵，6 点 × 4 arm，`n` 为 seed 数（(2,8)/(4,4) 为 3）；
- `prior`：EXP-023 先验扫描，7 个配方在 6 点上的均值；
- `unified_matrix`：6 个统一/专训模型 × 3 个套件（Y-K4 / YZ-4×4 / XYZ-4×4），
  3 seed 均值 ± sd。

这两份文件足以复算 `02_CLAIMS_EVIDENCE.md` 与 `figures/` 里的每一个数字，无需
GPU、无需读图片。

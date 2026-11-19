# PR3 定性对比图（入库版）

这些是从 `runs/` 里挑出来的**代表性**定性结果，压缩后入库供审阅。原始渲染图
（3265×3077 PNG，4–6 MB）和渲染数组（`.npz`，80–124 MB）**不入库**，留在本地：

```
runs/E026_psras_condition/figures/E026_real_yz_k2_kz2_fold0_s0/
runs/E026_psras_condition/figures/E026_matched_synthetic_yz_k2_kz2_fold0_s0/
runs/E025_adaptive_dual_prior/figures/E025_y_k4_fold0_s0/
```

入库版统一处理：LANCZOS 缩放到宽 2800 px、JPEG q93（渐进式）。文字标注在高 DPI 屏上可读，
需要逐像素核对时请回本地看原 PNG——**图中数字以本地 `qualitative_*.json` 为准**。

## 共同视角（三组共用一套冻结 spec）

样本 `M2_FOVII_S1_vessels`，fold 0 测试动物；z = 101 切片；y = x = 128 中心；ROI [160, 0, 96]；
zoom 96 µm；MIP 覆盖全部 203 层。spec 冻结在各自的 `viewspec_*.json` 里，重绘时用 `--view-spec`
复用，保证不同图看的是同一个窗口。

每个格子里标注该裁剪区的 PSNR / MAE / Dice / clDice；行是臂，列依次为：

1. **Input**：该臂实际拿到的稀疏输入（可见采样条带）
2. **GT**：参考（密集真值）
3–7. **XY / XZ / YZ / Zoom / MIP-z**：三个正交视图、放大块与全 z 投影
8–9. **测量面 / 跳过面**（与 **测量行 / 跳过行**）：诊断列，用来核对"缺失区"到底在哪

## 每张图在看什么

| 文件 | 臂 | 看什么 |
|---|---|---|
| [`e026_real_qualitative.jpg`](e026_real_qualitative.jpg) | `I0`、`W0full`、`W0leg`、`ResidI`、`P3_W` | **主图**。真实数据、恒等核选核结果。三行地板是**同一幅图**（恒等核下三个算子等价）；两行训练臂明显更"淡"——血管画得少而准，这是窗口口径 Dice 下降、整卷口径精确率上升的同一个现象 |
| [`e026_real_qualitative_control.jpg`](e026_real_qualitative_control.jpg) | `I0`、`ResidI`、`P3_int` | **控制臂那一轮**。`P3_int`（`phys_control=interp_only`）与上一张的 `P3_W` 是同一幅图——构造性退化 `P3_interp ≡ P3_wiener` 的视觉版本；两轮分开渲染是因为 `phys_control` 属于模型属性，不能共用一张图而不被错标 |
| [`e026_real_error_missing.jpg`](e026_real_error_missing.jpg) | 同上 | **误差图**，按"已测量 / 缺失"分区着色。真实数据上训练臂在缺失区的误差带更暗，对应 `psnr_missing` +1.4 dB |
| [`e026_matched_synthetic_qualitative.jpg`](e026_matched_synthetic_qualitative.jpg) | 同主图五臂 | **已知核合成条件**。这里 `W0full`/`W0leg` 与 `I0` **不再相同**——被告知的模糊确实被反卷积了（`W0 − I0` 整卷 +0.063 Dice），这是 E026 里物理通道唯一一次产生可测效果；`W0leg` 与 `W0full` 仍逐位相同（T=3 算子顺序退化） |
| [`e025_y_k4_qualitative.jpg`](e025_y_k4_qualitative.jpg) | 同主图五臂 | **单轴 75 % 缺失负对照**。列数为 7（Y 单轴没有 YZ/XZ 的测量面配对），条带方向与 YZ 组不同；结论与主图同构 |

## 读图规则（与报告一致）

- 跨动物方差约 2.8 dB PSNR / 0.08 Dice；**图上任何小于 1 dB / 0.03 Dice 的差别不要读**。
- 图中 Dice 是**窗口口径**（全局 Otsu、不门控），它的参考前景占 18.65 %，地板因为同样过检而"赢"。
  见 `docs/PHASE1_PR3_E026_RESULTS.md` §4.3：两套口径符号相反，图上看到的是其中一套。
- 这些图是**一张图、一个窗口、一个卷**，用于看机制（哪里淡、哪里断、误差在哪），不是统计证据。

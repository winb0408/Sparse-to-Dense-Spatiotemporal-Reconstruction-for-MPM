# Mask + Wiener 先验融合实验协议

本文件定义当前阶段的研究主线：以 `mask` 通道先验和 Wiener 物理初始化为核心创新点，并用轻量 prior-aware fusion 让网络显式利用二者，而不是只把它们作为普通输入通道。

---

## 1. 核心创新链路

```text
稀疏观测 Y 与采样 mask M
        ↓
Wiener 初始化 W(Y)：基于 PSF 的物理逆问题初值
        ↓
mask 通道 M：标注真实采样位置与估计填补位置
        ↓
prior-aware fusion：用 M 调制 W(Y) 的早期特征
        ↓
重建输出 X_hat
```

解释边界：

- Wiener 初始化负责提供物理可解释的初值。
- mask 通道负责提供采样置信度先验。
- prior-aware fusion 负责把二者从“输入拼接”提升为“结构化融合”。
- PSF channel / PSF loss 仍作为消融项，不作为当前最稳主贡献。

---

## 2. 三类主实验

| 场景 | 数据 | 稀疏模式 | 输出 | 主问题 |
|---|---|---|---|---|
| 2D | MDC25-2D Nuclei；CIDC25/MDC25-3D 单切面 | `x` / `y` / `xy` | 2D 图像 | 单轴与双轴空间稀疏下，mask + Wiener 是否提升结构恢复 |
| 2.5D | CIDC25-3D；MDC25-3D | `z` | 中心切面 | 轴向缺采样下，邻切面上下文与 mask/Wiener 是否提升中心帧重建 |
| 3D | CIDC25-3D；MDC25-3D | `xyz` | 3D 体块 | 联合空间/轴向稀疏下，物理初始化与采样先验是否稳定提升体重建 |

说明：

- `x`：只沿横向列方向稀疏采样，适合模拟单轴快速扫描缺采样。
- `y`：只沿纵向行方向稀疏采样，作为方向敏感性对照。
- `xy`：二维空间联合稀疏，是 2D 空间稀疏主协议。
- `z` / `xyz`：只在 3D stack 中具有严格物理意义。

---

## 3. 结构探索优先级

第一阶段只实现低侵入版本，保证可复现和可解释。

| 优先级 | 结构 | 作用 | 风险 |
|---|---|---|---|
| P0 | 普通通道拼接 | 现有 baseline | 低 |
| P1 | gated prior fusion | 用 mask 生成门控，调制 Wiener/warm-start 图像通道 | 低 |
| P2 | FiLM / adapter | 用 mask 或 mask density 调制中间 stage | 中 |
| P3 | mask-aware attention bias | 将 mask 下采样到 token/window，加入注意力 logits | 中高 |
| P4 | cross-attention | 图像 token 与 prior token 显式交互 | 高 |

当前建议先跑 P0/P1。P2/P3 只有在 P1 稳定后再进入主实验。

---

## 4. 必做消融矩阵

### 输入先验消融

| 编号 | warm-start | mask 通道 | 目的 |
|---|---|---|---|
| A0 | bilinear | 无 | 基础输入重建能力 |
| A1 | bilinear | 有 | 单独验证 mask 先验 |
| A2 | Wiener | 无 | 单独验证物理初始化 |
| A3 | Wiener | 有 | 验证 mask + Wiener 组合 |

### 结构融合消融

| 编号 | 结构 | 目的 |
|---|---|---|
| B0 | concat | 普通通道拼接 baseline |
| B1 | gated prior fusion | 验证轻量结构融合收益 |
| B2 | FiLM / adapter | 第二阶段增强 |
| B3 | mask-aware attention bias | 第三阶段增强 |

### 预处理消融

| 类别 | 对照项 |
|---|---|
| 归一化 | `minmax` / `fixed` / `zscore` |
| 初始化 | bilinear / Wiener |
| PSF | no PSF / PSF channel / PSF loss / Wiener |
| 噪声 | real low-high / synthetic Poisson + readout |

---

## 5. 指标分层协议

细胞图像背景占比高，不能只看全图 PSNR/SSIM。主表应至少分四层：

| 层级 | 指标 | 用途 |
|---|---|---|
| Global fidelity | PSNR、SSIM、MS-SSIM、MAE、sSNR、FwPSNR | 全图保真和可比性 |
| Foreground structure | PSNR_fg、SSIM_fg、MS-SSIM_fg、NCC_fg | 细胞/结构区域恢复 |
| Background stability | BG mean、BG std、BG residual std、false positive rate | 避免背景假阳性 |
| Local profile | FWHM、profile RMSE、profile correlation、peak shift | 局部分辨率和结构宽度 |

FWHM 不应在全图直接计算，应在有效局部 profile 上计算。自动 profile 结果只作为第一轮结构诊断，关键论文图建议人工复核 ROI。

---

## 6. 结果解释原则

1. 主结果按场景拆分：2D `x/y/xy`、2.5D `z`、3D `xyz`。
2. 不把 2D、2.5D、3D 放进一个 PSNR 总排名。
3. 不把 pollen 上的 2.5D/3D 结果写成 3D benchmark。
4. Wiener + mask + gated fusion 若提升 foreground/profile 指标，即使全局 PSNR 提升较小，也应作为结构恢复收益讨论。
5. 若全局指标提升但 foreground/profile 不提升，应视为背景主导的伪提升。
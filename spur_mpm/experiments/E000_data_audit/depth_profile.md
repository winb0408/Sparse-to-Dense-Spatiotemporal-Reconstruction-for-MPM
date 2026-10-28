# 深度剖面（每卷）

由 `scripts/profile_depth.py` 生成；数值来自原始 16-bit 数据，未经归一化。

| sample | 层数 | 深度跨度 (µm) | 首/末层均值比 | 最大饱和比例 | 最深 p99/最浅 p99 |
|---|---|---|---|---|---|
| M1_FOVII_S1_vessels | 202 | 1005 | 0.78 | 0.00% | 0.73 |
| M1_FOVII_S2_vessels | 190 | 945 | 0.90 | 0.00% | 0.84 |
| M1_FOVI_S1_vessels | 202 | 1005 | 0.76 | 0.00% | 0.70 |
| M1_FOVI_S2_vessels | 188 | 935 | 0.79 | 0.00% | 0.74 |
| M2_FOVII_S1_vessels | 203 | 1010 | 0.80 | 0.00% | 0.78 |
| M2_FOVII_S2_vessels | 202 | 1005 | 0.78 | 0.00% | 0.72 |
| M2_FOVI_S1_vessels | 201 | 1000 | 0.92 | 0.00% | 0.86 |
| M2_FOVI_S2_vessels | 152 | 755 | 0.83 | 0.00% | 0.77 |
| M3_FOVI_S1_vessels | 158 | 785 | 0.91 | 0.00% | 0.82 |
| M3_FOVI_S2_vessels | 168 | 835 | 0.79 | 0.00% | 0.73 |

## 数据缺陷

（无：无饱和、无空白层）

## 分析约束（不是缺陷，但必须在分析中体现）

- **M1_FOVII_S1_vessels**
  - p99 falls to 0.71 of its maximum across the stack (depth 0->1005 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M1_FOVII_S2_vessels**
  - p99 falls to 0.73 of its maximum across the stack (depth 0->945 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M1_FOVI_S1_vessels**
  - p99 falls to 0.70 of its maximum across the stack (depth 0->1005 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M1_FOVI_S2_vessels**
  - p99 falls to 0.73 of its maximum across the stack (depth 0->935 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M2_FOVII_S1_vessels**
  - p99 falls to 0.75 of its maximum across the stack (depth 0->1010 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M2_FOVII_S2_vessels**
  - p99 falls to 0.72 of its maximum across the stack (depth 0->1005 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M2_FOVI_S1_vessels**
  - p99 falls to 0.72 of its maximum across the stack (depth 0->1000 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M2_FOVI_S2_vessels**
  - p99 falls to 0.76 of its maximum across the stack (depth 0->755 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M3_FOVI_S1_vessels**
  - p99 falls to 0.75 of its maximum across the stack (depth 0->785 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately
- **M3_FOVI_S2_vessels**
  - p99 falls to 0.72 of its maximum across the stack (depth 0->835 µm): the stack mixes an SNR regime, so depth-pooled metrics average over it and bands must be reported separately

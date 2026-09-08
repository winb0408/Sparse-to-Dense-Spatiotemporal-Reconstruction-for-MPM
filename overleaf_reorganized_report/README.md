# Overleaf 上传说明

这是重整后的 3PM 稀疏采样重建报告独立上传包。

## 目录结构

```text
overleaf_reorganized_report/
├── main.tex              # Overleaf 主文件
├── figures/              # 报告实际引用的 8 张图片
├── manifest.txt          # 图片来源清单
└── README.md             # 本说明
```

## Overleaf 使用方式

1. 在 Overleaf 新建项目。
2. 上传整个 `overleaf_reorganized_report/` 目录，或上传 `overleaf_reorganized_report.zip` 后解压。
3. 设置主文件为 `main.tex`。
4. 编译器选择 `XeLaTeX`。

## 依赖说明

- `main.tex` 已改为只从 `figures/` 读取图片。
- 不依赖原项目中的 `eval_*`、`docs/`、`report/` 等目录。
- 中文排版依赖 `ctex`，Overleaf 使用 XeLaTeX 通常可直接编译。

## 图片命名

图片已从原始实验目录复制并重命名为论文式名称，便于后续维护：

- `cidc25_k2k4_qual_xy.png`
- `cidc25_k2k4_qual_z.png`
- `cidc25_k2k4_qual_xyz.png`
- `cidc25_k2_oblique_xy.png`
- `cidc25_k2_oblique_z.png`
- `cidc25_k2_oblique_xyz.png`
- `mdc25_k2_best5_modes.png`
- `pollen_fov1_4xXY_32p5mW_best5.png`
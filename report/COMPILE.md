# 如何编译 `report/reorganized_main.tex` / `report/main.tex` / `report/main_v2.tex`

## 推荐报告

当前推荐阅读和编译：`report/reorganized_main.tex`。

该文件依据重整后的实验主线编写，包含实验目标、总体设计、具体方法、结果分析、不足和后续工作。报告正文以文字与表格为主，文末附录集中放置关键图集，便于按结论核对图片证据。旧版 `main.tex` / `main_v2.tex` 保留为历史阶段报告。

图集依赖以下项目内图片路径：

- `eval_sparse3d_k2_fullimg/qualitative/*.png`
- `eval_sparse3d_k4_fullimg/qualitative/*.png`
- `eval_mdc25_sparse/qualitative/*.png`
- `eval_pollen_sparse/qualitative_pseudocolor_v3/*.png`

## Overleaf（推荐）

1. 新建项目，上传 `reorganized_main.tex`。
2. 若在 Overleaf 编译，需要同时上传图集依赖图片；最简单方式是在 Overleaf 中保持如下目录关系，或把所有图复制到与 `reorganized_main.tex` 同级后同步修改 `\\graphicspath`：
   - `eval_sparse3d_k2_fullimg/qualitative/`
   - `eval_sparse3d_k4_fullimg/qualitative/`
   - `eval_mdc25_sparse/qualitative/`
   - `eval_pollen_sparse/qualitative_pseudocolor_v3/`
3. 如需编译旧版报告，再上传 `main.tex`（或 `main_v2.tex`）与相关 `figures/` 图片。
4. 打开 **Menu → Compiler**，选择 **XeLaTeX**（不要选 pdfLaTeX）。
5. 点击 **Recompile**。

文档含中文，必须使用 **XeLaTeX** 或 **LuaLaTeX**；若使用 pdfLaTeX 会出现大量「Unicode character not set up」类错误。

## 本地（TeX Live）

```bash
cd report
xelatex -interaction=nonstopmode reorganized_main.tex
xelatex -interaction=nonstopmode reorganized_main.tex

# legacy reports
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex

xelatex -interaction=nonstopmode main_v2.tex
xelatex -interaction=nonstopmode main_v2.tex
```

## 可选：项目根目录 `latexmk`

在项目根目录放置 `.latexmkrc`：

```perl
$pdf_mode = 5;  # xelatex
$xelatex = 'xelatex -interaction=nonstopmode %O %S';
```

然后执行 `latexmk -cd report/main.tex`。

## 已修复的常见错误

| 问题 | 处理 |
|------|------|
| 中文在 pdfLaTeX 下无法编译 | 改用 `ctexart` + XeLaTeX |
| Unicode 破折号 `—`、`±`、`→` | 已改为 `---`、`$\pm$`、`$\rightarrow$` |
| 表格中 `$\approx$0` | 已改为 `$\approx 0$` |
| 表格中 em dash `—` 作占位符 | 已改为 `---` |
| `hyperref` 与部分宏包顺序 | `hyperref` 已移到宏包列表后部 |
| 未使用的 `subcaption` | 已移除，减少冲突 |

# Report Version History

## v1

- File: `report/main.tex`
- Scope:
  - Stage 1--4 summary
  - K=4 Phase1/Phase2 analysis
  - Overleaf-ready baseline report

## v2

- File: `report/main_v2.tex`
- Scope:
  - Keep v1 as immutable baseline
  - Add K=2 experiment integration
  - Add K=2 vs K=4 cross-sectional and longitudinal comparison
  - Add v2-specific conclusions and risk notes
- Key result source:
  - `eval_sparse3d_k2/summary_all.csv`
  - `eval_sparse3d_k2/summary_all.md`
  - `eval_sparse3d_k2/figures/`

## Compile

- Overleaf/Local both use XeLaTeX.
- For v2:
  - `xelatex report/main_v2.tex`
  - run twice for stable references.

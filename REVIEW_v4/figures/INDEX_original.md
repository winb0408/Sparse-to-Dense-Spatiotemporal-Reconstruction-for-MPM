# PSRAS v4 — paper-level figure set

All figures here are regenerated from committed checkpoints and metrics; no
new experiments. Style: serif, 300 dpi PNG + vector PDF; qualitative panels
use grayscale vmin=0/vmax=1 with axes off.

## 1. Claim -> figure map

| # | claim (v4 innovation) | quantitative figure | qualitative figure |
|---|---|---|---|
| C1 | fixed-budget Y/Z allocation law, (k_y,k_z)=(2,8) is the optimum | `fig1_alloc_law.{png,pdf}` | `qual_yz_2x8`, `qual_yz_8x2`, `xz_yz_2x8`, `xz_yz_8x2` |
| C2 | Wiener prior family + gated fusion beats Direct on all 6 combos | `fig3_gain_over_direct.{png,pdf}` | `qual_yz_2x8`, `qual_yz_2x8_err` |
| C3 | prior quality does not decide network quality (FillFirst < Iter-DC prior-side, > network-side) | `fig2_prior_vs_network.{png,pdf}` | `qual_yz_2x8` (b4i vs b4f columns) |
| C4 | one unified model covers Y/YZ/XYZ; the axial (AZ) prior is what makes XYZ work | `fig4_unified_cross_transfer.{png,pdf}` | `qual_univ_*`, `xz_univ_yz2_K4x4` |
| C5 | (Phase 5) noise robustness and regime weighting | see `../exp027_noise_stress/`, `../exp028_regime_weight/` | `../exp027_noise_stress/panel_*` |

## 2. Quantitative figures

### fig1_alloc_law — C1
(a) PSNR of the 4 YZ arms over the 6 sampling points (test split, mean over
seeds with spread as error bars; 3 seeds at 2×8 and 4×4). (b) equal sampling
budget k_y·k_z = 16: 2×8 vs 4×4 vs 8×2 for the 4 arms.

### fig2_prior_vs_network — C3
(a) analytic prior PSNR per point (EXP-023 sweep, mean over 8 items).
(b) network PSNR for the same warm-start families. (c) paired deltas
Iter-DC − FillFirst: the two sides disagree in sign at 4 of 6 points.

### fig3_gain_over_direct — C2
Gain (dB) of b2/b4i/b4f over the Direct arm at every sampling point.

### fig4_unified_cross_transfer — C4
(a) 4 models (specialists SY/SYZ/SXYZ + unified U15) evaluated on all three
regimes: specialists collapse off-regime (14-24 dB), U15 stays at 33.5-34.8 dB.
(b) unified variants: AZ prior off (`U_noAZ`) collapses on XYZ (11.5 dB);
regime weighting (`U15W`, EXP-028) trades Y for nothing.

## 3. Qualitative figures (val split, fixed grid, items 0-2)

Columns are GT | sparse Y | X0 prior | model outputs; PSNR is annotated under
each reconstruction. In 2.5D panels the query plane is *unmeasured*, so the
"sparse Y" column shows the nearest measured plane instead (a black frame
carries no information).

| file | content |
|---|---|
| `qual_yz_2x8` | EXP-024 optimal allocation: GT vs b1/b2/b4i/b4f |
| `qual_yz_8x2` | worst allocation, same columns — the gap is visible |
| `qual_yz_2x8_err` / `qual_yz_8x2_err` | absolute error maps, common scale (0-0.12) + MAE |
| `xz_yz_2x8` / `xz_yz_8x2` | XZ cross-section at the brightest column; horizontal stripes = unmeasured planes; shows whether the network interpolates between measured planes |
| `qual_univ_y_K4` | unified U15 vs specialist Y on the Y regime |
| `qual_univ_yz2_K4x4` | unified U15 vs specialist YZ on YZ 4×4 |
| `qual_univ_xyz2_K4x4` | unified U15 vs specialist XYZ on XYZ 4×4 |
| `xz_univ_yz2_K4x4` | unified XZ cross-section (axial interpolation quality) |

The raw arrays behind every qualitative panel are written next to it as
`*_data.npz` (`gt`, `sparse_y`/`sparse`, `x0`, `pred_<run>`), so panels can be
re-plotted without re-running the models.  These are generated locally and
are deliberately NOT committed (`.gitignore: *.npz`); rerun the script to
recreate them.

## 4. Numbers behind the figures (`key_numbers.csv`)

Rows are tagged `kind = yz_arm | prior | unified_matrix`.

### YZ arms — PSNR (dB), test split, mean ± seed sd
| arm | 2×2 | 2×4 | 2×8 | 4×2 | 4×4 | 8×2 |
|---|---|---|---|---|---|---|
| b1 Direct | 37.70 | 37.54 | 35.25 ± 0.47 | 33.07 | 32.94 ± 0.16 | 29.23 |
| b2 Bilinear | 38.78 | 39.07 | 36.86 ± 0.34 | 34.14 | 33.78 ± 0.39 | 29.24 |
| b4i Iter-DC | 40.91 | 39.04 | 37.22 ± 0.66 | 34.47 | 33.91 ± 0.29 | 29.81 |
| b4f FillFirst-Wiener | 41.06 | 40.45 | 37.96 ± 0.82 | 34.40 | 34.17 ± 0.20 | 29.82 |

Seed counts: 3 at 2×8 and 4×4, 1 otherwise.

### Gain over Direct (dB)
| arm | 2×2 | 2×4 | 2×8 | 4×2 | 4×4 | 8×2 |
|---|---|---|---|---|---|---|
| b2 Bilinear | +1.08 | +1.53 | +1.61 | +1.07 | +0.84 | +0.01 |
| b4i Iter-DC | +3.21 | +1.49 | +1.97 | +1.40 | +0.97 | +0.58 |
| b4f FillFirst-Wiener | +3.36 | +2.91 | +2.71 | +1.32 | +1.23 | +0.59 |

### Analytic priors (EXP-023 sweep, mean over items)
| prior recipe | 2×2 | 2×4 | 2×8 | 4×2 | 4×4 | 8×2 |
|---|---|---|---|---|---|---|
| P0_bilinear | 36.13 | 36.10 | 35.99 | 33.53 | 33.52 | 30.18 |
| P1_wiener_frozen | 28.51 | 28.51 | 28.49 | 13.18 | 13.18 | 9.76 |
| P4_iterdc_1e-3_5 | 37.80 | 36.91 | 36.26 | 34.01 | 33.76 | 30.27 |
| P2_fill_first | 37.69 | 37.65 | 37.48 | 33.68 | 33.67 | 30.04 |

### Unified / specialist cross-evaluation (mean ± sd over 3 seeds)
| model | Y (K=4) | YZ (4×4) | XYZ (4×4) |
|---|---|---|---|
| UNIV4_SY | 34.23 ± 0.04 | 31.99 ± 0.32 | 13.75 ± 0.29 |
| UNIV4_SYZ | 33.21 ± 0.12 | 33.55 ± 0.31 | 12.20 ± 2.13 |
| UNIV4_SXYZ | 23.53 ± 0.24 | 23.61 ± 0.37 | 35.24 ± 0.20 |
| UNIV4_U15 | 33.43 ± 0.17 | 33.71 ± 0.04 | 34.55 ± 0.26 |
| UNIV4_U_noAZ | 34.04 ± 0.06 | 34.03 ± 0.11 | 9.92 ± 1.15 |
| UNIV4_U15W | 33.17 ± 0.14 | 33.57 ± 0.13 | 34.59 ± 0.10 |

## 5. Reproduction

```bash
cd /data2/wjb/threePM/PSRAS
export TMPDIR=/data2/wjb/tmp
CUDA_VISIBLE_DEVICES=2 python scripts/paper_figs.py quant        # data only
CUDA_VISIBLE_DEVICES=2 python scripts/paper_figs.py qual_yz      # GPU
CUDA_VISIBLE_DEVICES=2 python scripts/paper_figs.py qual_unified # GPU
```

Source runs: `runs_v3/YZ_{ky}x{kz}_b{arm}[_s{seed}]` (EXP-024),
`runs_v3/exp023_prior_sweep/summary.csv` (EXP-023),
`runs_v3/UNIV4_*_s{0,1,2}` (EXP-025/026/028).

## 6. Other figure sets in `eval_viz/` (pre-v4 / Phase 5)

| directory | contents | supports |
|---|---|---|
| `../exp027_noise_stress/` | `noise_curve` (5 eval points, 4 noise levels, U15/U_noAZ + priors) and `panel_{y_K2,y_K4,y_K8,yz2_K4x4,xyz2_K4x4}` (GT / noisy / sparse / X0 / U15 / U_noAZ) | EXP-027: degradation is monotone and comes from noise amplification, not prior collapse |
| `../exp028_regime_weight/` | `exp028_suite_curves` (XYZ 1:1:2 weighting vs U15 vs specialist across 5 suites) | EXP-028: share hypothesis falsified |
| `../severity_z_K2/`, `../severity_z_K16/` | v3 severity panels + XZ cross-sections (z mode, K=2/16) | P1-P3 era baselines |
| `../severity_curve_v2/` | v3 PSNR-vs-K curve | P1-P3 era baselines |
| `../baseline_compare/` | external baseline comparisons (2d_xy, 2p5d_z, 2p5d_xyz) | EXP-020 |
| `../fusion_uncertainty/` | fusion / uncertainty maps (per-regime item 0) | gate diagnostics |
| `../exp017_T_curve/`, `../exp019_photon_curve/` | axial context depth T and photon budget curves | EXP-017 / EXP-019 |

## 7. Conventions and caveats

- Test split = frames [750,1500) of F0; qualitative panels use the val split
  (frames [0,750)) because they must show the patches the models were
  validated on. Absolute PSNRs therefore differ slightly from the test tables
  (2×8 b4f: 37.96 test vs 38.91 val on items 0-2).
- Unified runs store `psf.sigma_z_px = 0.0` in their config (top-level regime
  is `2d`) while the 2.5D training points used `--psf_z 1.5`; qualitative
  unified figures therefore build the dataset with sigma_z = 1.5 to match
  training (insensitivity check: EXP-027 result 5).
- In 2D (`y`) panels every plane is queried, so no measured-plane substitution
  is needed.
- Item labels in `qual_yz_*` are indices into the deterministic val grid
  (`make_val_dataset`, 8 patches per stack, 4 stacks).

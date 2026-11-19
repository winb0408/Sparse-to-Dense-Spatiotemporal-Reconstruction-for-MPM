# E025 results — the Y-only negative control at the same 25 % budget

**Status: complete** (2026-11-17 15:19). Three floors, three trained arms, five
structure reports, two `compare_structure` pairs, the collected table and both
qualitative-figure passes.

Protocol: `sampling.mode=y`, `ky=4` → 25 % of voxels retained and **75 % missing
on one axis only**, context `T = 1`, `patch_hw = 256x256`, 3000 steps, fold 0.
Compare with E026 (`docs/PHASE1_PR3_E026_RESULTS.md`), which removes 75 % as a
two-axis lattice at `T = 3`. The point of this group is that the missing mass is
the same but the geometry is not: if an effect comes from the Wiener prior rather
than from the sampling lattice, it should appear here too; if it appears only
under the YZ lattice, that is a statement about the lattice.

The prior used by the floors is the **train-only frozen selection**
(`runs/E025_adaptive_dual_prior/fold0_y_k4_prior_selection.json`):
`status: selected_on_training_volumes_only`, **identity** (`σ = (0, 0)`),
`λ = 0.001`, chosen from 15 candidates. So the same construction-level
degeneracies as in E026 `real` hold here, and are confirmed numerically below.

## 1. Window-level evaluation (758 windows, fold 0, test animal M2)

| arm | `psnr_missing` | `e_missing` | Dice | clDice | `pred_fraction` |
|---|---|---|---|---|---|
| `I0` | 24.7667 | 0.04562 | 0.5979 | 0.5614 | 0.1869 |
| `W0` | 24.7667 | 0.04562 | 0.5979 | 0.5614 | 0.1869 |
| `ResidualI` | **26.4385** | 0.03764 | 0.5406 | 0.5113 | 0.0620 |
| `P3_interp` | **26.4394** | 0.03761 | 0.5426 | 0.5126 | 0.0628 |
| `P3_wiener` | **26.4394** | 0.03761 | 0.5426 | 0.5126 | 0.0628 |

- `W0 ≡ I0` to every printed digit: identity kernel, `wiener_fill_all` short-
  circuits to `interp_linear`.
- `P3_wiener ≡ P3_interp` to every printed digit: with `H = I` the two arms
  receive the same physics channel. This is construction, not evidence.
- Trained arms over the floor: **+1.67 dB** `psnr_missing` and MAE −0.0080
  (≈ −18 %), against Dice **−0.055** / clDice **−0.049**. Same pair of signs as
  E026 `real` (+1.42 dB, −0.042 Dice), slightly larger in PSNR and slightly
  larger in the Dice penalty.
- The floor's `pred_fraction` (0.1869) sits almost exactly on the window
  protocol's reference fraction (0.1865); the trained arms call a third as much.

## 2. Whole-volume structure (per-plane threshold from the reference only, planes whose reference fills > 15 % gated out)

| arm | Dice | clDice | skelF1 | precision | recall | mean radius µm ref→pred | skeleton components ref→pred |
|---|---|---|---|---|---|---|---|
| `I0` ≡ `W0` | 0.7654 | 0.6484 | 0.5979 | 0.597 | 0.600 | 1.80 → 1.75 | 15543 → 11694 |
| `ResidualI` | 0.8021 | 0.7022 | 0.6248 | 0.785 | 0.530 | 1.80 → 2.08 | 15543 → 5693 |
| `P3_interp` ≡ `P3_wiener` | 0.8050 | 0.7050 | 0.6272 | 0.786 | 0.534 | 1.80 → 2.09 | 15543 → 5319 |

`P3 − I0` = **+0.0396 Dice, +0.0566 clDice, +0.0293 skelF1**, positive on 4 of 4
volumes:

| volume | scored | I0 Dice / clDice | P3 Dice / clDice |
|---|---|---|---|
| M2_FOVII_S1 | 146/203 | 0.8514 / 0.8933 | 0.8766 / 0.9322 |
| M2_FOVII_S2 | 134/202 | 0.7801 / 0.6541 | 0.8172 / 0.7163 |
| M2_FOVI_S1 | 119/201 | 0.8128 / 0.6594 | 0.8505 / 0.7561 |
| M2_FOVI_S2 | 51/152 | 0.6174 / 0.3869 | 0.6758 / 0.4154 |

`P3_wiener − P3_interp` = 0.0000 / 0.0000 / 0.0000.

## 3. What the negative control establishes

1. **The PR3 physics channel has no effect here either**, and here that is a
   structural zero rather than a measured null: `H = I`, so the two arms are the
   same operator. The group's value is not "another null" — it is that the
   *rest* of the comparison reproduces.
2. **The network's effect on intensity and on structure is geometry-independent
   in sign.** Both sampling geometries give +1.4 to +1.7 dB missing-region PSNR,
   a 0.04–0.06 window-Dice penalty, and a +0.04–0.06 whole-volume structure gain.
   That consistency is what makes the sign conflict of §4 below a measurement
   question rather than a fluke of one lattice.
3. **The protocol conflict of E026 §4.3 is not specific to the YZ lattice.**
   The window protocol's reference is 18.65 % foreground here as well, the floor
   again tracks it (0.1869), and the window Dice again penalises the trained
   arms while the whole-volume protocol rewards them. The conflict follows the
   *reference definition* (global Otsu on real stacks), not the sampling geometry
   and not the model.
4. **The depth pattern repeats**: the window-Dice deficit is concentrated below
   600 µm (0–200 +0.032, 200–400 +0.014, 400–600 +0.001, 600–800 **−0.143**,
   800–1000 **−0.222**), i.e. exactly where the global threshold drags the
   reference into haze, while `psnr_missing` improves at every depth
   (24.37 → 25.89, 26.05 → 27.71, 25.54 → 27.17, 26.56 → 28.28, 28.00 → 29.85 dB).

## 4. Qualitative figures

Two passes, one frozen view spec (sample `M2_FOVII_S1`, z = 101, zoom 96 µm,
full-z MIP), seven columns per arm (Y-only sampling has no XZ/YZ measured-plane
pairs). Metrics on the shown crop:

| arm | psnr | Dice | clDice | `e_missing` | `pred_fraction` |
|---|---|---|---|---|---|
| `I0` / `W0` | 24.3303 | 0.6628 | 0.6517 | 0.05220 | 0.1610 |
| `ResidualI` | 25.8115 | 0.6242 | 0.6163 | 0.04350 | 0.0657 |
| `P3_wiener` / `P3_interp` | 25.8116 | 0.6307 | 0.6232 | 0.04351 | 0.0682 |

`P3_wiener` and `P3_interp` are identical here to the last bit, not merely to the
printed digits: the two `summary.json` files differ in exactly two fields,
`model_control.phys_control` (`none` vs `interp_only`) and `checkpoint`, and every
metric matches exactly. That is the strongest form the construction-level
degeneracy can take — the same weights, the same operator, the same numbers.

## 5. What is not claimed

- No Wiener effect of any kind: `H = I` by the train-only selection, so
  `P3_wiener ≡ P3_interp` by construction.
- No structure verdict from either vessel protocol: as in E026 §4.3 they
  disagree in sign and the re-run that would settle it is `NOT_RUN`.
- One test animal, one fold, one seed. The +1.67 dB is a direction across the
  fold, not an effect size. Multi-seed is `NOT_RUN` for the same reason as in
  E026: the initial result shows no credible Wiener gain.
- No latency, throughput or memory number as a benchmark: shared cards.

## 6. Artefacts

| what | where |
|---|---|
| runs | `runs/E025_adaptive_dual_prior/E025_y_k4_fold0_s0_*` |
| frozen train-only selection | `runs/E025_adaptive_dual_prior/fold0_y_k4_prior_selection.json` |
| structure reports / comparisons | `runs/E025_adaptive_dual_prior/structure_E025_y_k4_fold0_s0_<arm>/`, `structure_compare_E025_y_k4_fold0_s0_*.md` |
| collected table | `runs/E025_adaptive_dual_prior/table_E025_y_k4_fold0_s0.md` |
| qualitative figures | `runs/E025_adaptive_dual_prior/figures/E025_y_k4_fold0_s0/` |

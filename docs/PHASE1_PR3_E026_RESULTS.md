# E026 results — the historical PSRAS condition under this repository's observation model

**Status:** `matched_synthetic` complete (six arms); `real` partial — two floors
done, the rest interrupted by a GPU fault and queued, see §4. Every number below
is traced to a run directory that contains its own `config.yaml`,
`provenance.json` and `metrics.jsonl`; nothing here is transcribed from a log by
hand, and every arm that has not produced a summary is marked `NOT_RUN` rather
than left blank.

Protocol: `sampling.mode=yz`, `ky=2`, `kz=2` → exactly 25 % of voxels retained,
context `T = 2*kz - 1 = 3`, `patch_hw = 256x256`, 3000 training steps, fold 0.
Both groups use the same protocol and differ only in the inverse problem:

- `matched_synthetic`: `Y = M*(H*D)`, target `D`, prior kernel `H = σ(1.0, 1.5) px
  = (0.98, 7.5) µm`. A **known-kernel synthetic stress test**, labelled as such
  everywhere; it is not the recorded stacks.
- `real`: `Y = M*D`, no extra forward blur, prior kernel selected on training
  animals only (`status: selected_on_training_volumes_only`).

## 1. How to read the arms (binding)

| label | arm | physics control | what it isolates |
|---|---|---|---|
| `I0` | `P2_I0` | — | interpolation floor, no network, no Wiener |
| `W0_full3d` | `P2_W0` | — | fill-then-Wiener floor, no network |
| `W0_legacy` | `P3_Wlegacy` | — | historical order — **not an independent floor here**, see §2 |
| `ResidualI` | `P2_ResidualI` | — | network on top of the interpolation prior |
| `P3_interp` | `P3_DualAdaptive` | `interp_only` | PR3 dual branch, Wiener term replaced by the interpolation prior, parameter-matched |
| `P3_wiener` | `P3_DualAdaptive` | — | PR3 dual branch with the Wiener prior |

`P3_wiener` vs `P3_interp` is the PR3 question: same parameters, same
architecture, same training budget, only the physics channel differs.

## 2. `W0_legacy` is not a second floor at T = 3

At the protocol's context length the two Wiener orders agree on the **scored
centre plane** to `1e-7`; the ordering only separates for T ≥ 5. Full derivation,
the four checks that ruled out a plumbing bug, and the context-length sweep:
[PHASE1_PR3_E026_OPERATOR_ORDER.md](PHASE1_PR3_E026_OPERATOR_ORDER.md).

Consequence: `W0_legacy` appears in the tables for traceability and is reported
as "identical to `W0_full3d`", never as an independent arm. The zero delta of
the `"W0_legacy W0_full3d"` pair in `structure_compare_*.md` is the check that
this still holds for this run.

## 3. `matched_synthetic` group

All six arms, test fold (`n_windows = 758`, of which
`n_windows_fully_measured = 0` — every window contains voxels the arm never
received):

| arm | psnr_missing | e_missing (MAE) | Dice | clDice | pred_fraction |
|---|---|---|---|---|---|
| `I0` | 26.8938 | 0.0366 | 0.4315 | 0.3908 | 0.0273 |
| `W0_full3d` | 27.0645 | 0.0358 | 0.5085 | 0.4654 | 0.0426 |
| `W0_legacy` | 27.0645 | 0.0358 | 0.5085 | 0.4654 | 0.0426 |
| `ResidualI` | 27.3071 | 0.0349 | 0.6231 | 0.5900 | 0.0932 |
| `P3_interp` | 27.3725 | 0.0348 | 0.6126 | 0.5783 | 0.0846 |
| `P3_wiener` | 27.3249 | 0.0350 | 0.6109 | 0.5759 | 0.0826 |

Reading the table under the claim discipline in force (cross-animal variance
≈ 2.8 dB PSNR / 0.08 Dice; differences below 1 dB / 0.03 Dice are not claimed;
this group has one test animal, so no confidence interval is estimable):

- `W0_full3d` vs `I0`: +0.24 dB PSNR and +0.077 Dice / +0.075 clDice. The PSNR
  difference is inside the 1 dB band and is **not** claimed. The Dice difference
  is above the 0.03 band but comes from a single-volume group; it is reported as
  a direction, not as an established effect.
- `ResidualI` vs `W0_full3d`: +0.24 dB and +0.115 Dice / +0.125 clDice, same
  reading. `pred_fraction` 0.0426 → 0.0932 says the network is far less
  conservative about calling vessel than the floor, which is why Dice and
  clDice move while PSNR barely does.

The PR3 comparison, `P3_wiener − P3_interp` (identical architecture, identically
parameter-matched, identical training budget, only the physics channel differs):

| metric | delta |
|---|---|
| `psnr_missing` | **−0.048 dB** |
| MAE on missing | +0.0002 |
| Dice | **−0.002** |
| clDice | **−0.002** |
| `pred_fraction` | −0.002 |

**Verdict: no credible Wiener gain, even with a known non-identity kernel.**
Every delta is inside the noise band, and the PSNR sign is *negative*. The
validation curve pointed the other way at the same budget
(`P3_wiener` 28.2319 vs `P3_interp` 28.0086 dB `dense_psnr`, +0.22 dB), so the
sign flip between val and test is itself the finding: at this effect size the
selection metric cannot distinguish the two arms, and no claim may be built on
the validation ordering either.

What does hold up in this group is coarser: every trained arm beats both floors
on missing-region error and on vessels (`P3_wiener − I0` = +0.43 dB
`psnr_missing`, +0.179 Dice, +0.185 clDice), and the three trained arms are
mutually indistinguishable (≤0.07 dB, ≤0.012 Dice apart from each other).
`W0_full3d`'s Dice gain over `I0` (+0.077) is in the same direction but from a
single-volume group with no estimable interval.

### 3.1 The structure report of the floor (one arm so far, read it as a baseline shape)

`structure_I0` for this group finished (whole-volume reconstruction, 4 test
volumes, vesselness binarisation with the threshold fitted on the reference
only). It is the parameter-free interpolation floor, so it says what *any* arm
in this group has to improve on, not what the trained arms did.

| volume | Dice | clDice | skelF1 | precision | recall | mean radius µm ref→pred | skeleton components ref→pred | scored planes |
|---|---|---|---|---|---|---|---|---|
| M2_FOVII_S1 | 0.8951 | 0.9156 | 0.8348 | 0.935 | 0.754 | 2.30 → 2.37 | 1580 → 497 | 146/203 (72 %) |
| M2_FOVII_S2 | 0.8204 | 0.6691 | 0.5779 | 0.919 | 0.421 | 1.76 → 2.37 | 11664 → 890 | 134/202 (66 %) |
| M2_FOVI_S1 | 0.8711 | 0.7444 | 0.6630 | 0.931 | 0.515 | 1.89 → 2.38 | 17833 → 973 | 119/201 (59 %) |
| M2_FOVI_S2 | 0.6676 | 0.3066 | 0.2372 | 0.896 | 0.137 | 1.24 → 2.22 | 31094 → 786 | 51/152 (34 %) |
| mean | 0.8136 | 0.6589 | 0.5782 | 0.920 | 0.457 | 1.80 → 2.33 | 15543 → 787 | — |

Three things this table binds, beyond the numbers themselves:

- **The mean is not a comparable scalar.** Dice ranges 0.67–0.90 and clDice
  0.31–0.92 across four volumes of one test animal. Quoting "clDice 0.659"
  without the per-volume spread would invent a precision the data does not have.
- **Every row is scored only on the planes that survived gating.** The
  reference mask fills more than 15 % of a plane on the top and bottom of the
  stack, so both masks are emptied there; between 28 % and 66 % of the planes
  are excluded depending on the volume. The scored share is part of the number.
- **The floor's failure mode is recall, not precision, and it is depth-graded.**
  Precision stays 0.90–0.94 while recall falls 0.75 → 0.14 from FOVII_S1 to
  FOVI_S2; skeleton components collapse from 15 543 to 787 and mean radius
  inflates from 1.80 to 2.33 µm. That is interpolation thickening and merging
  fine branches, and it is exactly the effect the missing-region and
  fine-vessel analysis in §4 has to test — which is why the PSNR table alone
  cannot carry the verdict.

This is 1 of the group's 6 structure reports. The other five, the three
`compare_structure` pairs and the qualitative figures are queued; until they
land, nothing here distinguishes any arm from the floor.

This is a 3000-step mechanism pilot, one fold, one seed, and one test volume.
Scaling to multi-seed / multi-fold is **`NOT_RUN` by design**: the initial
result shows no credible gain, so spending seeds on it would only make a null
more precise.

## 4. `real` group

**Partially run, then interrupted by a hardware fault.** Two of the three floors
are complete and the rest is queued; the state of each arm is stated rather than
implied:

| arm | state |
|---|---|
| `I0` | **done** — `psnr_missing` 25.2246, `e_missing` 0.0435, Dice 0.6256, clDice 0.6003 |
| `W0_full3d` | **done** — identical to `I0` in every field except `arm` |
| `W0_legacy` | `NOT_RUN` — its evaluation was running when GPU 2 fell off the bus (2026-11-13 21:44, `docs/evidence/gpu2_falloff_20261113/INCIDENT.md`) |
| `P2_ResidualI` | `NOT_RUN` |
| `P3_interp` | `NOT_RUN` |
| `P3_wiener` | `NOT_RUN` |
| analysis for the `real` group | `NOT_RUN` |
| analysis for the `matched_synthetic` group | **partial** — `structure_I0` done (§3.1); the other five reports, the three `compare_structure` pairs and the qualitative figures are queued |

The floors were re-run on the frozen train-only selection, i.e. no test animal
touched the kernel choice. Two facts constrain what the remaining arms can show,
established from that selection file
(`runs/E026_psras_condition/fold0_yz_k2_kz2_selection.json`) and from the
operator definitions:

- The train-only selection chose **identity** (`σ = (0, 0)`, `λ = 0.005`,
  `status: selected_on_training_volumes_only`). Every non-identity candidate was
  worse on held-out dense voxels — the best by +7.0 %, the worst by +686 %.
  With `H = I`, `wiener_fill_all` returns `interp_linear` exactly, so
  `W0_full3d ≡ I0` by construction, not by coincidence.
- With `H = I` the `interp_only` control and the uncontrolled PR3 arm receive the
  same physics channel, so `P3_interp ≡ P3_wiener` by construction.

So the informative `real` comparison reduces to `I0` (≡ `W0_full3d`) vs
`P2_ResidualI` vs `P3_DualAdaptive`. All candidates and their errors stay in the
selection file; the kernel is not re-tuned against test data, and no non-zero
kernel is forced to make the search look different.

## 5. What is not claimed

- Nothing here shows that the Wiener prior beats interpolation on real data. On
  real data the selected kernel *is* interpolation.
- The `matched_synthetic` result is about a synthetic inverse problem with a
  known kernel. It does not transfer to the recorded stacks without an
  independent PSF estimate — and with the identity kernel this cohort selects,
  there is nothing for the Wiener term to invert there either.
- No multi-seed or multi-fold claim: fold 0, seed 0 only.
- No arm of the `real` group beyond the two floors has a result yet; nothing in
  §4 is a finding about real data except the identity-kernel collapse itself.

## 6. Artefacts

| what | where |
|---|---|
| runs (6 arms + 4–6 floor/arm evals) | `runs/E026_psras_condition/E026_<group>_yz_k2_kz2_fold0_s0_*` |
| preflight (train volume, Y/M/target/candidates) | `docs/evidence/e026_preflight_20261113/` |
| operator-order diagnostic | `docs/evidence/e026_operator_context_20261113/operator_context.json` |
| structure reports / comparisons | `runs/E026_psras_condition/structure_<TAG>_<arm>/`, `structure_compare_<TAG>_<A>_vs_<B>.md` (scoped by group: the two groups share this directory and an unscoped name made them collide) |
| collected table | `runs/E026_psras_condition/table_*.md` |
| qualitative figures (frozen view spec) | `runs/E026_psras_condition/figures/` |
| full test suite | `docs/evidence/pytest_20261113_pr3_full.log` (425 passed) |

# E026 results — the historical PSRAS condition under this repository's observation model

**Status:** both groups complete — six arms, structure analysis and qualitative
figures each, `real` finishing on 2026-11-16 23:12 and its matched-synthetic
counterpart on 2026-11-17 05:03 after a GPU fault on 2026-11-13 forced a restart
(§4). Every number below is traced to a run directory that contains its own
`config.yaml`, `provenance.json` and `metrics.jsonl`; nothing here is transcribed
from a log by hand, and every measurement that has not been run is marked
`NOT_RUN` rather than left blank.

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

This is 1 of the group's 6 structure reports, and it describes the floor only.

### 3.2 All six arms, whole-volume structure (finished 2026-11-17 05:03)

Same protocol as §3.1, all six arms, plus the three `compare_structure` pairs:

| arm | Dice | clDice | skelF1 | precision | recall | mean radius µm ref→pred | skeleton components ref→pred |
|---|---|---|---|---|---|---|---|
| `I0` | 0.8136 | 0.6589 | 0.5782 | 0.920 | 0.457 | 1.80 → 2.33 | 15543 → 786 |
| `W0_full3d` | 0.8765 | 0.7261 | 0.6829 | 0.926 | 0.567 | 1.80 → 2.10 | 15543 → 3059 |
| `W0_legacy` | 0.8765 | 0.7261 | 0.6829 | 0.926 | 0.567 | 1.80 → 2.10 | 15543 → 3059 |
| `ResidualI` | 0.9091 | 0.8123 | 0.7938 | 0.886 | 0.727 | 1.80 → 1.92 | 15543 → 9396 |
| `P3_interp` | 0.9146 | 0.8104 | 0.7942 | 0.900 | 0.720 | 1.80 → 1.93 | 15543 → 8964 |
| `P3_wiener` | 0.9145 | 0.8070 | 0.7891 | 0.902 | 0.712 | 1.80 → 1.94 | 15543 → 8701 |

Four things follow, and they are consistent with §3's PSNR verdict rather than a
re-reading of it:

1. **The PR3 comparison is still null.** `P3_wiener − P3_interp` = **+0.0001
   Dice, +0.0034 clDice, +0.0051 skelF1**, with per-volume Dice of both signs
   (+0.0003, −0.0005, +0.0016 across the scorable volumes). The structure
   protocol, which §4.3 shows can disagree with the window protocol, here agrees
   with it exactly: the physics channel buys nothing even when the kernel is
   known and non-identity.
2. **`W0_legacy ≡ W0_full3d` again, now with `H ≠ I`.** Δ = 0.0000 on all four
   volumes, on the whole-volume structure protocol. §2's finding — that the two
   Wiener orderings coincide at T = 3 — was measured on a *training* volume with
   a synthetic kernel and then on the real test set. This is a third setting, and
   the nullity does not depend on the kernel being identity.
3. **In this group a non-identity kernel does move structure**, unlike on real
   data: `W0_full3d − I0` = **+0.063 Dice, +0.067 clDice, +0.105 skelF1**, with
   recall 0.457 → 0.567, mean radius 2.33 → 2.10 µm and skeleton fragments
   786 → 3059 against a reference tree of 15 543. That is the Wiener term
   actually inverting a blur it was told about, and it is the one place in E026
   where the physics channel has a measurable effect — on a *synthetic* inverse
   problem. It is still one test animal, and +0.063 Dice sits inside the 0.08
   cross-animal spread, so it is a direction, not an effect size. It also
   reinforces §3's verdict: the gain belongs to the known-kernel floor, and once
   the two PR3 arms are both given the same physics channel that gain is already
   in both of them.
4. **Protocol agreement is conditional on the reference, not on the model.** In
   this group the window protocol and the whole-volume protocol rank the arms
   identically (trained ≫ floor; `P3` = `ResidualI`; `P3_wiener` = `P3_interp`).
   On real data they disagreed (§4.3). The difference between the two settings is
   the reference: 2.18 % foreground here and in §4.2, 18.65 % in the real
   window protocol. That is the evidence behind §4.3's reading — the real-group
   sign flip is a property of that reference, not of the network.

### 3.3 Qualitative figures

Two passes over the same frozen view spec (sample `M2_FOVII_S1`, z = 101,
zoom 96 µm, full-z MIP), nine columns per arm as in §4.4, split by
`phys_control` for the same reason. Metrics on the shown crop:

| arm | psnr | Dice | clDice | `e_missing` | `pred_fraction` |
|---|---|---|---|---|---|
| `I0` | 24.8333 | 0.5413 | 0.5248 | 0.04208 | 0.0344 |
| `W0_full3d` / `W0_legacy` | 25.0679 | 0.6145 | 0.5929 | 0.04125 | 0.0501 |
| `ResidualI` | 25.3860 | 0.7017 | 0.6878 | 0.03984 | 0.0916 |
| `P3_wiener` | 25.4226 | 0.6914 | 0.6755 | 0.03969 | 0.0822 |
| `P3_interp` | 25.4358 | 0.6940 | 0.6791 | 0.03962 | 0.0848 |

The picture is the table: the two Wiener orderings are one image, the two PR3
arms are another that differs from `ResidualI` only in the last decimals, and
the floors differ from the trained arms by a visibly thinner vessel call (the
floor's `pred_fraction` 0.034 is a third of the reference, which is why its
recall is 0.457). Artefacts:
`figures/E026_matched_synthetic_yz_k2_kz2_fold0_s0/qualitative_M2_FOVII_S1_vessels{,_interp}.png`
with the `error_*`, `profile_*` and `support_*` panels and the `.npz` behind each.

This is a 3000-step mechanism pilot, one fold, one seed, and one test volume.
Scaling to multi-seed / multi-fold is **`NOT_RUN` by design**: the initial
result shows no credible gain, so spending seeds on it would only make a null
more precise.

## 4. `real` group

**Complete** (2026-11-13 13:56 → 2026-11-16 23:12: three floors, three trained
arms, six structure reports, three `compare_structure` pairs, the collected table
and both qualitative-figure passes).

### 4.1 Window-level evaluation (758 windows, fold 0, test animal M2)

| arm | `psnr_missing` | `e_missing` | Dice | clDice | `pred_fraction` |
|---|---|---|---|---|---|
| `I0` | 25.2246 | 0.04354 | 0.6256 | 0.6003 | 0.1670 |
| `W0_full3d` | 25.2246 | 0.04354 | 0.6256 | 0.6003 | 0.1670 |
| `W0_legacy` | 25.2246 | 0.04354 | 0.6256 | 0.6003 | 0.1670 |
| `ResidualI` | **26.6106** | 0.03703 | 0.5862 | 0.5595 | 0.0779 |
| `P3_interp` | **26.6458** | 0.03696 | 0.5836 | 0.5569 | 0.0765 |
| `P3_wiener` | **26.6458** | 0.03696 | 0.5836 | 0.5569 | 0.0765 |

All three floors exist and are the third independent confirmation of §2's
operator-order finding at this context length: `I0` 25.2246487021,
`W0_full3d` 25.2246487021 (identical to every printed digit), `W0_legacy`
25.2246486795 — **2.3e-8 dB** between the two Wiener orderings, against the
1.24e-7 centre-plane prediction measured on a training volume. On real data at
`T = 3` the ordering is not a variable.

The floors were re-run on the frozen train-only selection, i.e. no test animal
touched the kernel choice. Two facts constrain what the trained arms can show,
established from that selection file
(`runs/E026_psras_condition/fold0_yz_k2_kz2_selection.json`) and from the
operator definitions:

- The train-only selection chose **identity** (`σ = (0, 0)`, `λ = 0.005`,
  `status: selected_on_training_volumes_only`). Every non-identity candidate was
  worse on held-out dense voxels — the best by +7.0 %, the worst by +686 %.
  With `H = I`, `wiener_fill_all` returns `interp_linear` exactly, so
  `W0_full3d ≡ I0` by construction, not by coincidence.
- With `H = I` the `interp_only` control and the uncontrolled PR3 arm receive the
  same physics channel, so `P3_interp ≡ P3_wiener` by construction. The
  measurement confirms it to every printed digit — in the window table, in the
  whole-volume structure report, in the qualitative crop metrics and in
  `structure_compare_..._P3_wiener_vs_P3_interp.md` (Δ = 0.0000 on all four
  volumes).

So the informative `real` comparison reduces to `I0` (≡ `W0_full3d` ≡
`W0_legacy`) vs `P2_ResidualI` vs `P3_DualAdaptive`.

**What the trained arms do, in the window protocol:**

- `psnr_missing` **+1.39 dB** (`ResidualI`) and **+1.42 dB** (`P3`) over the
  floor, and MAE on missing voxels −0.0065 (≈ −15 %). PSNR is above the 1 dB
  band, so this is the one place in the `real` group where a difference survives
  the claim discipline. One test animal, so it is stated as a direction across
  the fold, not as an established effect size.
- Dice **−0.039 / −0.042** and clDice **−0.041 / −0.043** against the floor,
  i.e. above the 0.03 band in the *negative* direction. This is not a rounding
  artefact: it is the same phenomenon seen from the other side in §4.2, and it
  is a conflict between two vessel protocols, which §4.3 resolves as far as the
  data allows.
- `pred_fraction` 0.1670 → 0.0765: the trained arms call less than half as much
  vessel as the floor on the same reference.

### 4.2 Whole-volume structure (4 test volumes, per-plane threshold fitted on the reference only, planes whose reference fills > 15 % gated out)

| arm | Dice | clDice | skelF1 | precision | recall | mean radius µm ref→pred | skeleton components ref→pred |
|---|---|---|---|---|---|---|---|
| `I0` ≡ `W0_full3d` ≡ `W0_legacy` | 0.7778 | 0.6462 | 0.6103 | 0.550 | 0.690 | 1.80 → 1.63 | 15543 → 21516 |
| `ResidualI` | 0.8329 | 0.7256 | 0.6619 | 0.778 | 0.585 | 1.80 → 2.01 | 15543 → 5914 |
| `P3_interp` ≡ `P3_wiener` | 0.8323 | 0.7233 | 0.6580 | 0.785 | 0.576 | 1.80 → 2.02 | 15543 → 5536 |

Per-volume, with the scored-plane share beside every number (the four volumes
differ by a factor of two and the share of depth that is scorable ranges
119/201–146/203 on the three informative volumes and 51/152 on the fourth):

| volume | scored | I0 Dice / clDice | P3 Dice / clDice |
|---|---|---|---|
| M2_FOVII_S1 | 146/203 | 0.8895 / 0.8893 | 0.9128 / 0.9420 |
| M2_FOVII_S2 | 134/202 | 0.8001 / 0.6541 | 0.8463 / 0.7357 |
| M2_FOVI_S1 | 119/201 | 0.8210 / 0.6415 | 0.8718 / 0.7696 |
| M2_FOVI_S2 | 51/152 | 0.6003 / 0.4000 | 0.6983 / 0.4458 |

`P3 − I0` whole-volume: **+0.0545 Dice, +0.0771 clDice, +0.0477 skelF1**,
positive on 4 of 4 volumes, and the mechanism is visible in the columns rather
than only in the score: the floor's precision is 0.55 with 21 516 skeleton
fragments against a reference tree of 15 543 (it over-calls and shatters), while
the trained arms reach 0.78–0.79 precision with 5.5–5.9 k fragments (they
under-call and merge). Neither is correct; they fail in opposite directions.

### 4.3 The two vessel protocols disagree, and the reason is identifiable

Window-level Dice says the network is **worse** by 0.04; whole-volume Dice says
it is **better** by 0.055. That is not noise — the protocols score different
references:

| | window protocol (`evaluate.py`) | whole-volume protocol (`structure_report`) |
|---|---|---|
| threshold | one global Otsu **0.0290**, fitted on the pooled reference windows, applied to every arm | per-plane threshold fitted on the reference, **median 0.0414** |
| plane gating | none | planes whose reference mask fills > 15 % are emptied on both sides |
| reference foreground | **18.65 %** of scored voxels | **2.18 %** |

A reference mask that covers 18.65 % of the tissue is not a vessel tree — this is
the failure mode `segment_pair`'s own docstring documents ("a mask that fills a
third of a plane turns the skeleton into a surface"). In the window protocol the
floor wins by *matching that over-segmentation* (its `pred_fraction` 0.167 is
close to the reference's 0.1865), not by reconstructing vessels better. In the
whole-volume protocol the reference is a 2 % tree and the trained arms are
closer to it.

So: **the negative window-level vessel delta cannot be read as "the network
degrades vessels"**, because its reference is inflated by haze that the floor
happens to reproduce. Equally, the **positive whole-volume delta cannot be read
as "the network improves structure"**: it is one test animal, and +0.055 Dice
sits inside the 0.08 cross-animal spread quoted by the comparison tool itself.
Both numbers are reported; neither carries a verdict. This is the "unify the
evaluation semantics" item of the PR3 directive, and it stays open — the honest
resolution is a re-run of the window protocol with per-plane thresholds and
gating, which is **`NOT_RUN`** and would change a published metric for every
earlier experiment, so it is not done here on a pilot's authority.

Depth-stratified window Dice, reported because the directive asks for deep
structures and because it shows where the two protocols part company:

| depth | I0 Dice | P3 Dice | difference |
|---|---|---|---|
| 0–200 µm | 0.7184 | 0.7369 | +0.019 |
| 200–400 µm | 0.7431 | 0.7504 | +0.007 |
| 400–600 µm | 0.6716 | 0.6615 | −0.010 |
| 600–800 µm | 0.4934 | 0.4017 | **−0.092** |
| 800–1000 µm | 0.4562 | 0.2917 | **−0.165** |

The entire window-protocol deficit is deep, where the global threshold pulls the
reference furthest into haze; the same windows show `psnr_missing` improving by
~1.4 dB at every depth. Per the standing rule, depth conclusions are made on
vessel metrics, not PSNR — and on this metric the trained arms are worse below
600 µm under the window protocol, better under the whole-volume one.

### 4.4 Qualitative figures

Two passes sharing one frozen view spec (sample `M2_FOVII_S1`, plane z = 101,
y = x = 128, ROI [160, 0, 96], zoom 96 µm, MIP over the full z range), nine
columns per arm: XY, XZ, YZ, a zoomed patch, the z-MIP, plus measured-vs-skipped
plane and row pairs so the acquisition mask is visible next to the error. The
second pass exists because `phys_control` is a property of the model: `P3_wiener`
and `P3_interp` cannot share one figure without one of them being mislabelled.

Metrics on the shown crop (one volume, one plane — descriptive only):

| arm | psnr | MAE | Dice | clDice | `e_missing` | `pred_fraction` |
|---|---|---|---|---|---|---|
| `I0` / `W0full` / `W0leg` | 24.9856 | 0.03683 | 0.6976 | 0.6924 | 0.04954 | 0.1479 |
| `ResidualI` | 26.2055 | 0.03205 | 0.6708 | 0.6626 | 0.04244 | 0.0799 |
| `P3_wiener` | 26.2506 | 0.03178 | 0.6709 | 0.6619 | 0.04231 | 0.0796 |
| `P3_interp` | 26.2506 | 0.03178 | 0.6709 | 0.6619 | 0.04231 | 0.0796 |

The figure is the picture of §4.3 at one plane: the three floors draw the same
image to five decimals, the two PR3 arms draw the same image to five decimals,
and the difference between floor and network is a visibly sparser vessel call on
a reference that includes haze. Artefacts:
`figures/E026_real_yz_k2_kz2_fold0_s0/qualitative_M2_FOVII_S1_vessels{,_interp}.png`
plus the matching `error_*`, `profile_*`, `support_*` panels and the `.npz` behind
each figure.

## 5. What is not claimed

- Nothing here shows that the Wiener prior beats interpolation on real data. On
  real data the selected kernel *is* interpolation.
- The `matched_synthetic` result is about a synthetic inverse problem with a
  known kernel. It does not transfer to the recorded stacks without an
  independent PSF estimate — and with the identity kernel this cohort selects,
  there is nothing for the Wiener term to invert there either.
- No multi-seed or multi-fold claim: fold 0, seed 0 only. The `real` group's
  +1.4 dB missing-region PSNR and its structure deltas are each from one test
  animal.
- No claim that the network improves or degrades vessel structure: §4.3 shows
  the two protocols disagree in sign and says why. The re-run that would settle
  it (per-plane thresholds and gating in the window protocol) is `NOT_RUN`.
- No claim from the qualitative figures beyond "the floors are one image and the
  two PR3 arms are another": they are one plane of one volume, and their crop
  metrics are descriptive.
- No latency, throughput or peak-memory number from this group is reported as a
  benchmark: the cards were shared throughout.

## 6. Artefacts

| what | where |
|---|---|
| runs (6 arms + 4–6 floor/arm evals) | `runs/E026_psras_condition/E026_<group>_yz_k2_kz2_fold0_s0_*` |
| preflight (train volume, Y/M/target/candidates) | `docs/evidence/e026_preflight_20261113/` |
| operator-order diagnostic | `docs/evidence/e026_operator_context_20261113/operator_context.json` |
| structure reports / comparisons | `runs/E026_psras_condition/structure_<TAG>_<arm>/`, `structure_compare_<TAG>_<A>_vs_<B>.md` (scoped by group: the two groups share this directory and an unscoped name made them collide) |
| qualitative figures, both groups (frozen view specs) | `runs/E026_psras_condition/figures/E026_{real,matched_synthetic}_yz_k2_kz2_fold0_s0/` (two passes each, `viewspec_*.json`) |
| collected table | `runs/E026_psras_condition/table_*.md` |
| qualitative figures (frozen view spec) | `runs/E026_psras_condition/figures/` |
| full test suite | `docs/evidence/pytest_20261113_pr3_full.log` (425 passed) |

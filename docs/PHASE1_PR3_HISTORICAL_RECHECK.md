# PR3 / historical PSRAS Wiener verification and revised experiment decision

**Reviewed:** 2026-11-13, connected GitHub repository; PSRAS branch HEAD `4a70d9e`; PR2 HEAD `b637bde`. Historical findings are from **historical records and source code**, not from newly executed re-training. E026 remains `NOT_RUN`.

## 1. Historical provenance checked

- `PSRAS/PROJECT_LOG.md` EXP-024 and `PSRAS/Project_progress_status.md`: six joint line/plane sparse configurations `(ky,kz)=(2,2),(2,4),(2,8),(4,2),(4,4),(8,2)`, all with `kz>=2`. `b4f-b1` test PSNR gains were +3.36, +2.91, mean +2.71, +1.33, mean +1.23 and +0.59 dB. `(2,8)` and `(4,4)` include 3 seeds; other cells mainly one seed. This demonstrates the **complete** FillFirst-Wiener+fusion system vs a Direct network, NOT the isolated Wiener contribution.
- `PSRAS/scripts/train.py`: `b1=(Y,M)`, `b2=(Y,M,I0)`, `b4f=(Y,M,W0,C_s,U_p)+fusion`. Historical defaults were `psf_xy=1.0`, `psf_z=1.5` **in pixels**, `lambda=0.01`. In the EXP-024 YZ mode, context `T=2*kz-1` and 8000 training steps.
- `PSRAS/physics/acquisition.py` 118–152: synthetic forward `Hx` then sparse mask `M`, using `PSRAS/data/datasets.py` 141–162 which supplies the SAME `psf` to the inverse candidate. Target = original `gt_t`, i.e. before the added known blur. This is **matched-known-kernel inverse restoration**. The current real-Dryad main line instead targets recorded dense `D` without added H; it is not the same inverse problem.
- **Crucial correction**: `PSRAS/physics/wiener.py::warm_start` mode `wiener_fill_first` under axial gaps first interpolates missing lateral rows, then applies **lateral-only Wiener on measured Z planes** (`psf.lateral_only()` / `sigma_z=0`), then interpolates skipped Z planes. **It does not perform one global full-3D Wiener after Z interpolation.** Compare with current `P2_W0`/`P3_DualAdaptive` which use `wiener_fill_all` -> interpolated full stack -> 3D FFT. Both should be evaluated; do NOT call them identical PSRAS operators.
- `PSRAS/PROJECT_LOG.md` EXP-021 no-PSF M1: at z-K8, `b1=40.53`, `b2=41.34`, `b3=41.34` dB without extra PSF; with known blur, `b1=38.91`, `b3=44.24`. This shows strong dependence on the known forward degradation but the no-PSF comparison also changes task difficulty; not proof of optical PSF estimation on unknown real images.
- Historical EXP-020/021/022 records show much weaker or negative 2D/Y results, including one 2D `b3-b1=-0.29 dB` and negative frozen-lambda experiments. The correct inference is regime dependence.
- Historical `b4f` versus **bilinear-prior network b2** is less uniform than `b4f-b1`: e.g. +2.28 (2,2), +1.38 (2,4), (2,8) positive across 3 seeds, but (4,4) includes a -0.22 dB seed. This is why matching to a baseline network is mandatory.

**Evidence links:** [Historical PSRAS log](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/PROJECT_LOG.md) | [PSRAS code](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/physics/wiener.py) | [Historical matrix](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/blob/PSRAS/Project_progress_status.md).

## 2. Revised PR3 priority (not an instruction to force Wiener to win)

**E025 remains a Y-only `ky=4,kz=1` 75%-missing NEGATIVE/LOW-GAIN CONTROL.** There is little historical basis to assume that a nonidentity Wiener would dramatically improve Y-only `kz=1`; run it with train-only PSF selection and retain negative results.

**New E026: test the 75%-missing YZ regime known to be relevant historically**, with `ky=2,kz=2` so retained fraction `1/(2*2)=25%`, matching the E025 Y-only measurement budget. Context `T=3` is essential (not `T=1`). Do NOT call Y vs YZ a parameter-matched identical problem; geometry and context are different. Their comparison is a *regime-dependence* study, while within-regime P3_Wiener vs P3_interp is the isolated physics-contribution test.

E026 has **two separate subgroups**:

1. `SOURCE_KIND=matched_synthetic` (first, mechanistic positive control): generate `Y=M*(H_known*D)` from the recorded dense image D with known H `sigma_xy=1 pixel, sigma_z=1.5 pixels` and target D; use exactly that H for inverse prior. The physical values entered for this Dryad cohort are `sigma_xy_um=0.98`, `sigma_z_um=7.5`. This is a *synthetic known-kernel inverse problem*, not calibration, real-data PSF inference or direct reproduction of CIDC25. It must be published only under synthetic stress-test labeling. Strong effect here and not on real recorded D is fully possible and expected.
2. `SOURCE_KIND=real`: genuine current task `Y=M*D`, no extra forward blur, train-fold-only adaptive **effective** kernel (not calibrated optical PSF). A zero-width winner is valid. If kernel is identity, report non-availability of effective Wiener contribution rather than force H!=I.

**New no-learning operator control**: `P3_Wlegacy` in `models/registry.py`, with `warm_start=wiener_fill_first`, scores the historical lateral-Wiener-then-Z-fill prior against the PR3 `P2_W0` full 3D fill-first Wiener. They must NOT be conflated. Not a separate trained network.

**Within each E026 group compare**: `P2_I0`, `P2_W0` (full-3D), `P3_Wlegacy` (lateral-first), `P2_ResidualI`, `P3_DualAdaptive[interp_only]`, `P3_DualAdaptive[none]`. Same train animal split, config, PSF and seed inside group. For synthetic group both P3 dual arms share same actual Y from H_known*D; only W0 input is replaced by I0 in the control. E026 initial training steps are 3000 per arm for **low-cost mechanism pilot**, NOT a 1:1 repetition of the old 8000-step PSRAS experiment. If credible signal appears, run matched 8000 steps and multiple seeds/folds before scientific claims.

## 3. Execution commands (DeepSeek after tests and dry-run)

```bash
# Historical-condition synthetic known-H sanity/positive control:
DRY=1 SOURCE_KIND=matched_synthetic FOLD=0 KY=2 KZ=2 STAGE=all \
  bash spur_mpm/experiments/run_e026_historical.sh

# Run ONLY after full test and visual check of Y/M/GT and 2D vs 3D candidate:
DRY=0 SOURCE_KIND=matched_synthetic FOLD=0 KY=2 KZ=2 STAGE=all GPU=0 \
  bash spur_mpm/experiments/run_e026_historical.sh

# Independently select an effective candidate on TRAIN animals for the real group:
DRY=0 SOURCE_KIND=real FOLD=0 KY=2 KZ=2 STAGE=select \
  bash spur_mpm/experiments/run_e026_historical.sh

DRY=0 SOURCE_KIND=real FOLD=0 KY=2 KZ=2 STAGE=all GPU=0 \
  bash spur_mpm/experiments/run_e026_historical.sh
```

Use an **owned** GPU; `GPU=0` is an example, not authorization. Before the real group, inspect frozen selection JSON, ensure train/test IDs disjoint, no test tuning. Check that the script refuses existing run directories and preserves all valid runs. The old PR3 `run_e025.sh` previously hardcoded a 512×512 eval patch although current examples are 256×256: this was fixed to 256×256 in PR3 commit `c846c8c`.

## 4. Additional required unit and integration checks

- Assert `P3_Wlegacy` under YZ missing planes produces different W0 from `P2_W0` under known nonzero axial H (except specially degenerate inputs). Assert real measured mask is unchanged between all arms in same subgroup.
- Assert known kernel is applied ONCE in synthetic forward and used as prior, and is not applied at all to real group forward. Synthetic target = D before **extra** H. Do not label D as unblurred latent biology: D itself is an optical recording.
- Assert 3D Wiener, legacy lateral-first and simple interpolation outputs are finite and nonidentical for nontrivial known H; quantify PSNR missing and ringing, vessel topology and heldout areas.
- Verify `P3_DualAdaptive` training forward/aux outputs and gradients work with `context_slices=3` and every skipped-center Z phase; use test runner before allocating GPU time.
- Correct the historical assertion that identity PSF in current real data is a *measured optical identity*. It is only the chosen/selected task-effective inversion kernel for the recorded-D target.
- Validate training/qualitative/inference/evaluation semantic agreement for `forward_extra_psf`, `psf_source` and `phys_control`. Qualitative figures must not load model with stale defaults.
- Any substantial apparent improvement: rerun with independent seeds and multiple animal-held-out folds, and report 2D/3D vessel metrics and coverage. No structure gain claim from PSNR alone.
- Preserve original E021/E023/E024 and correct negative E025/E026; run dry-run allowlist audit; don't delete referenced smoke audit evidence.

## 5. Decision tree

- **Synthetic known-H YZ wins but real YZ does not:** code can exploit an *oracle matched inverse*, but no evidence that unknown real PSF is useful; pursue calibration/model identification separately.
- **Only legacy-lateral Wiener wins:** prior construction/order matters; reconsider full-3D Wiener pipeline, do not assert all deconvolutions equivalent.
- **Real YZ effective-Wiener beats interp-only and I0 reproducibly with anatomical fidelity:** strongest support for PR3 hypothesis; continue deeper sparse-rate and Stage 2 neuron validation.
- **Both synthetic and real fail:** inspect operator, DC gain, masks, PSF scaling/units, sample target, and branch gradient before adding complexity.
- **Adaptive selection is identity:** do not force positive H or attribute network learning gains to Wiener.

This document and `docs/PHASE1_PR3_IMPLEMENTATION_AND_RUNBOOK.md` jointly govern DeepSeek execution. On a conflict concerning *historical evidence* or *experimental priority*, this historical recheck takes precedence. **Status: source verified / code committed / GPU tests NOT RUN.**

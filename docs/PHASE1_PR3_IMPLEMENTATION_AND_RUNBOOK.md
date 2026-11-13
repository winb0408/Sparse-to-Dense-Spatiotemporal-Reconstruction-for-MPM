# PR3: Adaptive effective Wiener prior + lightweight dual-branch test (E025)

**Status:** CODE COMMITTED / TESTS NOT YET EXECUTED / GPU TRAINING NOT RUN.  
**Base:** PR #2 head `feature/phase1-dual-prior-20261110` @ `b637bde62ee79303959dfac02759d11a85d40894`.  
**Head:** `feature/phase1-adaptive-prior-pr3-20261112`.  
**Downstream target:** first review PR3 against PR2; do not merge directly to `PUR-MPM` until PR2 is accepted or both are explicitly rebased.  
**Scientific scope:** general 2P/3P sparse reconstruction; vessels are Stage 1 *structural* checks, neurons/morphology/calcium are Stage 2 extensions. No vessel-specific head is added to the shared backbone.

## A. Exact code changes already made in this branch

| Path | Change / why | Acceptance |
|---|---|---|
| `spur_mpm/data/datasets.py` | Independent `ItemConfig.forward_psf` and `forward_psf_source`; `cfg.psf` now only generates the Wiener candidate. Actual virtual acquisition uses `forward_psf`, identity by default; nonidentity forward degradation requires `synthetic_extra_degradation`. | Changing only `prior_psf` must NOT change measured Y. A synthetic extra forward H must be visibly labelled. |
| `spur_mpm/train.py` | Wire forward-extra PSF; add optional missing-pixel L1 auxiliary terms for observation and physics branches, controlled by `loss.branch_obs_weight` and `loss.branch_phys_weight`; main L1+MS-SSIM unchanged. | Both heads get gradient; 0 weights recover original main objective. Reject nonzero weights for legacy arms. |
| `spur_mpm/evaluate.py` | Wire forward-extra PSF identically for evaluation. | Train/eval must use exactly same acquisition model and source labels. |
| `spur_mpm/inference.py` | Wire forward-extra PSF; correct wall-time metadata (does not include saved-array writes, but includes cache/input/transfer and is **not** pure GPU latency). | No claim that model latency includes disk writes. |
| `spur_mpm/scripts/qualitative_figures.py` | Forward `psf_source`, `forward_extra_psf*`, `model.phys_control`, `model.consistency`. | Reconstructing a control uses the same model configuration as its checkpoint. |
| `spur_mpm/scripts/viewspec.py` | Correct denominator/broadcasting for observed versus missing Y rows; compute sampled-row support among visited Z planes. | For `y_k4`, measured-row label 100%, skipped-row label 0%. |
| `spur_mpm/models/dual_prior_reconstructor.py` | Add independent, light 1×1 branch auxiliary heads; **instantiate ONLY for PR3**. Add `phys_control=interp_only` which replaces W0 with I0 and delta with zeros without changing architecture. | Legacy PR2 state_dict keys and checkpoint shapes unchanged; same exact P3 network parameters with/without Wiener. |
| `spur_mpm/models/registry.py` | New isolated `P3_DualAdaptive` arm: obs(Y,M,U_sampling), phys(I0,W0,delta,M,U_physics); gated residual output, two branch auxiliary heads. | No old arm definitions or names overwritten. |
| `spur_mpm/configs/base.yaml` | Add independent identity forward-extra PSF settings. | Current real-recorded-image main task stays physically unchanged. |
| `spur_mpm/scripts/select_adaptive_prior.py` | Training-volumes-only grid selection over effective lateral/axial widths and Wiener λ, on missing-pixel MSE for specified sampling; identity in grid. Records complete candidates and test IDs excluded. | NEVER call this calibrated optics. Identity winner is permitted and must be reported. |
| `spur_mpm/experiments/run_e025.sh` | Reproducible Y 75%-missing test driver; no automatic deletes; named baseline/residual/interp-only/Wiener dual runs; default dry-run; freezes selection JSON. | No E024 directory reused or overwritten. Explicit device required for actual training. |
| `spur_mpm/tests/test_pr3_adaptive_prior.py` | Regression test skeleton for PSF separation, label geometry, unchanged legacy checkpoint keys, auxiliary heads, train-only parameter selection. | DeepSeek must run and fix failures before GPU experiments. |

**Terminology:** prior H = candidate inversion kernel for the target *recorded* dense D; forward-extra H = additional synthetic optical degradation applied to D before sampling. These are separate. We do **not** assume that deconvolving the recorded D should directly reproduce D or that an estimated effective kernel equals the physical microscope PSF.

**Simple objective for E025:**
`L = L1(final,D) + 0.5 MS-SSIM(final,D) + 0.05 L1_missing(obs_aux,D) + 0.05 L1_missing(phys_aux,D)`.
The two heads are trained independently on observed-vs-physics features, but both are ultimately supervised against recorded D. This is a lightweight validation of branch participation, not the full future two-objective topology method. Do not claim calibrated uncertainty or vessel topology confidence.

## B. Mandatory DeepSeek verification BEFORE GPU

1. Inspect every file in §A at PR3 HEAD; do not trust this document alone. Run `python -m pytest spur_mpm/tests -o addopts="" -q`. **No test has been executed by the PR author in this environment.** Archive log and exit code.
2. Run the targeted `test_pr3_adaptive_prior.py`, `test_physics_prior_contract.py`, `test_dual_branch_contract.py`, `test_pipeline_smoke.py`. Fix incompatibilities; document every change. In particular confirm P2 archived checkpoints still `load_state_dict(strict=True)`; the PR3 heads must not appear in P2 state dicts.
3. Test `physics.psf=nonidentity`, `physics.forward_extra_psf=identity` on recorded dense pixels: sampled observation must remain the original recorded values. Separate `synthetic_extra_degradation` positive control should actually change Y. Exercise `y_k4` and `y_k8`, and later z/yz with valid masks.
4. Check the new `select_adaptive_prior` CLI on a tiny TRAIN-only sample (no M2 test volume allowed in fold0). Verify every candidate's input is `Y=M*D`, selection score is on `~M`, source/provenance/selected σ, λ are saved, and identity remains a legal answer. Correct data/sampling phase and no GT leaks.
5. The proposed script uses FFT on interpolated *recorded* D; λ changes DC response for nonzero σ whereas identity early-exit copies I0. Check mean-preserving/DC-normalized alternative as separate reported ablation rather than silently changing previous E024 operators.
6. Check `P3_DualAdaptive` vs `phys_control=interp_only` model parameter counts and input dimensionality are identical. Check actual gate/head gradient norms after a few optimization steps (the zero-initialized decoder head can suppress branch gradients on first step). Compare branch outputs and final output.
7. Check model is task-independent: vessel-only masks/skeletons do not enter P3 main input. Do not rename Z to T for neuron sequences.

## C. Trial E025: 75% missing / 25% retained (`y_k4`)

**Do not confuse "75% sparsity" with "75% retained".** Here `ky=4`: every fourth Y line is measured, **25% retained, 75% missing**. It is virtual line skipping from fully recorded images, not accelerated hardware acquisition.

Preparation (read-only dry-run, no GPU):

```bash
DRY=1 STAGE=all FOLD=0 KY=4 bash spur_mpm/experiments/run_e025.sh
```

Selection (training volumes only; optional CPU/GPU depending on available host):

```bash
DRY=0 STAGE=select FOLD=0 KY=4 bash spur_mpm/experiments/run_e025.sh
cat runs/E025_adaptive_dual_prior/fold0_y_k4_prior_selection.json
```

Before running, inspect candidate table. If identity wins, **record it without forcing nonidentity**. Consider a separate sensitivity test using `estimated_from_data` candidates even if they lose, but not as headline optimized Wiener. On real data do not invent a blur in the acquisition path.

Explicit GPU launch (one job, owned device only):

```bash
DRY=1 STAGE=all FOLD=0 KY=4 GPU=0 bash spur_mpm/experiments/run_e025.sh
DRY=0 STAGE=all FOLD=0 KY=4 GPU=0 bash spur_mpm/experiments/run_e025.sh
```

The script runs:

- `P2_I0`: zero-param interpolation baseline.
- `P2_W0`: zero-param fill-first Wiener with SAME frozen effective prior.
- `P2_ResidualI`: original single-branch trained residual.
- `P3_DualAdaptive[interp_only]`: full P3 architecture and aux loss, but W0 forcibly replaced by I0.
- `P3_DualAdaptive[none]`: EXACT same P3 architecture and aux loss, with selected W0. 

Do NOT present a difference between the last two as a PSF gain if the selected kernel was identity. If W0 is not measurably different from I0 on held-out samples, stop the physics-contribution claim. Additional positive synthetic blur controls must live under a **different run ID**, and cannot be cited as real-data Wiener gain.

Only if the 75%-missing pilot passes basic correctness, extend to `KY=8` (87.5% missing) and `KY=2` (50% missing), and expand to folds 1–2 / multiple seeds. Every fold must select its own prior from its own training animals, never from the test animal. The script defaults to FOLD=0; no silent fold sweep.

## D. Interpretation gates and outputs

**Primary:** missing-pixel PSNR and MAE, 2D Sato-based same-operating-point Dice/clDice, with one fixed protocol, masks, coordinates and thresholding semantics across arms. Compare 2D per-window and 3D whole-volume separately and document their distinct filter dimensionality. Report how many Z planes are gated; include a no-gate sensitivity and full-volume manual QC.

**Qualitative:** same physical ROI and display windows for dense GT, actual sampled Y, I0, W0, residual, P3 interp-only and P3 Wiener; show error map only on missing positions, centerlines/branch overlays, and difficult deep/capillary/branchpoint examples (not just a visually successful image). Visually inspect exported PNG/TIF rather than only verifying it is nonempty. Demonstrate the input measured/skipped row labels are truthful.

**Performance:** on an owned idle GPU, time preprocessing (interpolation, PSF/Wiener), network forward and combined compute, with warmups/synchronize. Report peak allocated/reserved VRAM and network Params/MACs. Exclude ALL disk read/write from compute comparison. Existing E024 shared-card memory reports are not valid benchmarks.

**Success criterion:** P3 Wiener must materially and reproducibly exceed I0 and the matched P3 interp-only control **in missing-pixel quality AND on independently audited structure**, without worse false bridges/breaks. If P3 and interp-only are indistinguishable, faithfully retain this negative result. Do not tune on M2 test outputs.

## E. Archive and cleanup

Keep all correctly executed E021/E023/E024 experiments and correct but negative E025 cells. Invalid experiments require an explicit invalidity note, config/checkpoint/commit identifiers, replacement link, and only then reviewed deletion of redundant LARGE artifacts. Smoke/pilot garbage can be deleted ONLY with a verified, reviewed, path-specific allowlist and dry-run; do not erase cited audit evidence inside a smoke parent. Never `rm -rf runs`, overwrite prior results or delete the user's original data. Re-run `audit_runs.py` after E025 and link new run manifests. Record GPU failures and interrupted experiments as invalid/interrupted, not scientifically negative.

## F. Mandatory response from DeepSeek

Provide (1) code-diff summary per file, (2) unit/regression test logs and unresolved failures, (3) prior-selection full JSON with fold-scoped source and candidate errors, (4) each run's config fingerprint, best checkpoint path and test summary, (5) comparison table, depth profiles, qualitative images including failures, (6) artifact archive status, (7) explicit classification DONE / TESTED / NOT_RUN for every item. Keep PR3 Draft until reviewed. No claims of improved Wiener signal or topology without measured proof.

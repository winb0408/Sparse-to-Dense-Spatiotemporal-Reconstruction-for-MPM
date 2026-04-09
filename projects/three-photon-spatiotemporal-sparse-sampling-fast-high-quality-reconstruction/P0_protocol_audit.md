# P0 Protocol Audit: CIDC25 Evaluation Correctness

## Scope
- Audit target: CIDC25 evaluation path and consistency of data assumptions.
- Files inspected: `eval.py`, `dataset.py`, `options.py`, `README.md`, existing `eval_results/**/summary*.csv`.

## Key Findings

1. **Potential 3D->2D collapse risk (Critical)**
- `dataset.py::load_tiff_float32()` collapses any TIFF with ndim > 2 by repeatedly taking the first slice (`while raw.ndim > 2: raw = raw[0]`).
- If CIDC25 sample files are stacks `[F,H,W]`, current baseline/eval path may only keep frame 0, effectively discarding temporal/depth information.
- This aligns with user concern: CIDC25 is originally 3D-like stack data and pure 2D baseline handling can be scientifically insufficient.

2. **2D-only eval path for CIDC25 in current scripts (High)**
- `eval.py` uses `NucleiDenoisingDataset` / `NucleiGridDataset`, both operating on paired 2D images.
- The current CIDC25 conversion (`prepare_cidc25_pairs.py`) creates 2D paired frames, which is valid for 2D baseline smoke test but not sufficient to claim spatiotemporal reconstruction quality.

3. **Split naming inconsistency risk (Medium)**
- README examples include paths with `Test` and text referring to Validation in nearby section wording.
- This can cause accidental split misuse and misleading zero-shot claims unless explicitly logged in every run artifact.

4. **Metric distribution anomaly indicates either domain shift or protocol mismatch (High)**
- Aggregated table shows strong in-domain metrics in some subsets but severe degradation on `eval_results_CIDC25_zero_val_F0_full` (very low PSNR/SSIM and negative delta_psnr).
- Before interpreting as model weakness, protocol must be validated (pairing, normalization, GT alignment, dynamic range, file matching semantics).

## Verdict
- Current CIDC25 pipeline is acceptable as a **2D baseline benchmark** only.
- It is **not yet sufficient** for final claims about "三光子时空稀疏采样的快速高质量重建" until 2.5D/3D evaluation is added.

## Immediate P0 Fixes
- P0-1: Freeze and document current 2D protocol as baseline-only.
- P0-2: Add an explicit CIDC25 dimensionality note in output artifacts and result tables.
- P0-3: Add 2.5D setting (k-neighbor frame stack as input channels, center-frame target).
- P0-4: Add true 3D/spatiotemporal evaluation track using `--task spatiotemporal_3d` with stack-preserving loader.

## Next Experiment Tracks
- **Track A (2D baseline):** keep current pipeline for reproducible reference.
- **Track B (2.5D):** input `[t-1,t,t+1]` (or 5-frame) -> output center frame.
- **Track C (3D):** input/output clip with temporal consistency metrics.

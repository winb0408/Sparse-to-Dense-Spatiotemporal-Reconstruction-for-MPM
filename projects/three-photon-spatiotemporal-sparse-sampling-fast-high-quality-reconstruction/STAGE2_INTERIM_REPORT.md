# Stage-2 Interim Report (Pilot)

## Scope
- Environment: `conda activate threePM`
- Dataset: `stage2_pilot_data` (Training 800 pairs, Test 200 pairs sampled from CIDC25_paired_A1F0)
- Models: `baseline` vs `spatial_2p5d (temporal_window=3)`
- Training budget: 2 epochs each (pilot quick-turn validation)
- DINO branch: enabled and active (non-zero `loss/dino` in train logs)

## Quantitative Comparison

| metric | baseline2d | spatial_2p5d | 2p5d-baseline |
|---|---:|---:|---:|
| psnr | 27.4128 | 26.2564 | -1.1564 |\n| ssim | 0.7187 | 0.6951 | -0.0236 |\n| ms_ssim | 0.6881 | 0.6551 | -0.0330 |\n| lpips | 0.6603 | 0.6502 | -0.0101 |\n| delta_psnr | 0.9996 | -0.1209 | -1.1205 |\n| delta_ssim | 0.1670 | 0.1445 | -0.0225 |\n| mae | 0.0371 | 0.0438 | +0.0066 |\n
## Observations
- On this pilot split and short budget, baseline2d outperforms 2p5d on PSNR/SSIM/MAE.
- 2p5d still improves over noisy input in SSIM (`delta_ssim>0`), but `delta_psnr` is slightly negative in this pilot run.
- This indicates the current 2p5d configuration is not yet tuned; temporal context does not automatically yield gains under low training budget.

## Artifacts
- 2p5d training: `runs/stage2_2p5d_w3_pilot`
- 2p5d eval: `eval_results/stage2_2p5d_w3_pilot_eval/eval`
- baseline training: `runs/stage2_baseline2d_pilot`
- baseline eval: `eval_results/stage2_baseline2d_pilot_eval/eval`
- figures: `projects/three-photon-spatiotemporal-sparse-sampling-fast-high-quality-reconstruction/figures_stage2/`

## Next Actions
1. Increase 2p5d training budget (epochs and/or data volume).
2. Test temporal_window=5 and compare w3 vs w5.
3. Add temporal-consistency metric and 3D track before claiming spatiotemporal gains.

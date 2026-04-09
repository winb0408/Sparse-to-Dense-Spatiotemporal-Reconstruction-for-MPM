# Stage-2 Extended Report (Round-2)

## Setup
- Env: `conda activate threePM`
- DINO branch: enabled
- Data: stage2_pilot_data
- Budget: extended 6-epoch runs for 2.5D

## Comparison Table

| metric | baseline2d | 2p5d_w3_e6 | 2p5d_w5_e6 | best |
|---|---:|---:|---:|---|
| psnr | 27.4128 | 29.7044 | 28.9355 | 2p5d_w3_e6 |\n| ssim | 0.7187 | 0.7854 | 0.7686 | 2p5d_w3_e6 |\n| ms_ssim | 0.6881 | 0.7495 | 0.7305 | 2p5d_w3_e6 |\n| mae | 0.0371 | 0.0266 | 0.0298 | 2p5d_w3_e6 |\n| delta_psnr | 0.9996 | 3.3270 | 2.5944 | 2p5d_w3_e6 |\n| delta_ssim | 0.1670 | 0.2348 | 0.2191 | 2p5d_w3_e6 |\n| lpips | 0.6603 | 0.6667 | 0.6625 | baseline2d |\n
## Key Findings
- Extended-budget 2.5D significantly improved over baseline on core restoration metrics.
- `w3_e6` outperformed `w5_e6` on this split for PSNR/SSIM/MAE and delta metrics.
- Relative to baseline, `w3_e6` gains: PSNR +2.292, SSIM +0.067, MAE 0.037 -> 0.027.

## Interpretation
- Stage-2 now provides positive evidence that temporal context (2.5D) can improve reconstruction quality when training budget is sufficient.
- However, this is still a pilot-scale split and does not replace full-scale 3D temporal-consistency validation.

## Next Steps
1. Re-run w3_e6 on larger/full split for stability.
2. Add explicit temporal-consistency metrics and failure-case analysis.
3. Proceed to 3D track for final spatiotemporal claim closure.

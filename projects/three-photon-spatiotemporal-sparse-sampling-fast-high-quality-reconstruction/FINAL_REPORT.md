# Final Research Cycle Report

## Topic
三光子时空稀疏采样的快速高质量重建

## Executive Conclusion
- Current evidence supports **2D reconstruction baseline** only.
- Current evidence is **insufficient** to support the full "spatiotemporal" claim.
- The main blocker (silent 3D->2D collapse risk) was fixed at code level in this cycle.

## What Was Completed In This Cycle
- P0 protocol audit completed (`P0_protocol_audit.md`).
- Unified result aggregation completed (`P0_unified_summary_table.csv/.md`).
- External GLM-5 critical review completed (`P0_glm5_reviewer_feedback.md`).
- Fail-fast stack safety implemented in data loading path (`dataset.py`).
- Pre-eval guardrail script added (`scripts/check_eval_protocol.py`).

## Key Quantitative Snapshot (from existing summaries)
### Best 3 by PSNR

| run_tag | psnr | ssim | delta_psnr | summary_file |
|---|---:|---:|---:|---|
| eval_results | 34.5178 | 0.9581 | 14.9982 | /data2/wjb/threePM/eval_results/eval_results_MDC25_patch/eval/summary_run_MDC25.csv |
| eval_results | 33.8210 | 0.9447 | 14.3014 | /data2/wjb/threePM/eval_results/eval_results_DD/eval/summary_run.csv |
| eval_results | 32.4060 | 0.9372 | 10.8114 | /data2/wjb/threePM/eval_results/eval_results_MDC25_full/eval/summary_run_MDC25.csv |

### Worst 3 by PSNR

| run_tag | psnr | ssim | delta_psnr | summary_file |
|---|---:|---:|---:|---|
| eval_results | 6.0789 | 0.1250 | -16.0744 | /data2/wjb/threePM/eval_results/eval_results_CIDC25_zero_train_A1_full/eval/summary_run_DD.csv |
| eval_results | 6.3861 | 0.0948 | -19.2707 | /data2/wjb/threePM/eval_results/eval_results_CIDC25_zero_val_F0_full/eval/summary_run_DD.csv |
| eval_results | 24.8721 | 0.6287 | -0.7846 | /data2/wjb/threePM/eval_results/eval_results_CIDC25_ft_val_F0/eval/summary_run_CIDC25_ft_A1F0.csv |

## Scientific Risks
- Claim-scope mismatch risk: objective mentions spatiotemporal reconstruction, but current evaluated pipeline is predominantly 2D.
- Generalization risk: at least one CIDC25 split shows severe collapse (very low PSNR/SSIM, negative delta_psnr).
- Protocol risk reduced: silent stack collapse is now prevented by default (`--stack_policy error`).

## Decision
- **Do not** freeze final scientific claim yet.
- Freeze current deliverable as: "2D baseline + protocol-hardening completed".

## Next Stage Plan (Required for final claim)
1. 2.5D pilot: [t-1,t,t+1] / [t-2..t+2] -> center-frame reconstruction with identical split and metrics.
2. 3D track: stack-preserving spatiotemporal reconstruction + temporal consistency metrics.
3. Fair comparison: same data split, same normalization, same runtime budget, report speed-memory-quality triplet.
4. External reviewer pass (GLM-5) after 2.5D/3D results are available.

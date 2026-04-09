# Stage-3 Next Actions

Reviewer verdict: **NOT PASS** (GLM-5).

## Required minimal actions before milestone pass
1. Re-check train/val curves to confirm whether 180 epochs caused overfit or underfit.
2. Tune loss weights (especially perceptual branch balance) to address high LPIPS.
3. Run a strict paired comparison against Stage-2/2D baseline on the same split.
4. Re-run one ablation on temporal window / fusion strategy and re-evaluate.
5. Re-submit Stage-3 report to reviewer after metrics update.

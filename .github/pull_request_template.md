# Pull Request Checklist

## What changed

- [ ] Briefly describe this update in 1-3 bullets.
- [ ] Explain why this change is needed.

## Experiment context (if applicable)

- [ ] Related run name:
- [ ] Related commit ID:
- [ ] Related result directory:
- [ ] Main metric changes (PSNR/SSIM/MS-SSIM/LPIPS):

## Validation

- [ ] I ran the related train/eval/inference command.
- [ ] I checked key outputs (metrics CSV / figures / logs).
- [ ] I confirmed no sensitive data is included.

## Risk check

- [ ] Backward compatibility considered.
- [ ] Data path assumptions are documented.
- [ ] Large generated artifacts are excluded by `.gitignore`.

## Merge policy

- [ ] Target branch is correct (`dev` for normal work, `main` only for stable merge).


# Experiment Log Template

Use this template for each important training/evaluation iteration.

## 1) Basic info

- Date:
- Owner:
- Branch:
- Commit ID:
- Run name:
- Task (`baseline` / `spatial_2d` / `spatiotemporal_3d` / `spatial_2p5d`):

## 2) Goal

- What question is this experiment trying to answer?
- Expected outcome:

## 3) Data setup

- Train data path:
- Val/Test data path:
- Split policy:
- Stack policy (if used):

## 4) Key hyperparameters

- Model:
- Epochs:
- Batch size:
- Patch size:
- Learning rate:
- Sparsity factor:
- Temporal window:
- PSF settings:

## 5) Command (copy actual command)

```bash
# train or eval command here
```

## 6) Results

- Main metrics (PSNR/SSIM/MS-SSIM/LPIPS):
- Best checkpoint:
- Output directory:
- Visual result path (figures):

## 7) Conclusion

- What worked:
- What failed:
- Next action:


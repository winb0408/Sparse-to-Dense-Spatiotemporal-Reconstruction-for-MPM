# 3PM Sparse-to-Dense Reconstruction Pipeline

**Three-Photon Microscopy** | Swin-UNet 2D / Video Swin-UNet 3D | DINOv2 + MS-SSIM Loss

---

## Directory Structure

```
threePM/
├── options.py          # All hyperparameters (argparse)
├── dataset.py          # Data loading, PSF blur, noise, masking, warm-start
├── models/
│   ├── __init__.py     # build_model() factory
│   ├── swin_unet_2d.py # 2D Swin-UNet
│   └── video_swin_unet.py  # 3D Video Swin-UNet
├── loss.py             # MSE + MS-SSIM + DINOv2 hybrid loss
├── train.py            # Training loop
└── inference.py        # Sliding-window / temporal-window inference
```

---

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tifffile pillow tensorboard
```

---

## Task Modes

| `--task`             | Input shape        | Model               | Description                        |
|----------------------|--------------------|---------------------|------------------------------------|
| `baseline`           | `[B, 1, H, W]`     | `swin_unet_2d`      | Dense noisy → dense clean          |
| `spatial_2d`         | `[B, 2, H, W]`     | `swin_unet_2d`      | 2D sparse scan reconstruction      |
| `spatiotemporal_3d`  | `[B, 2, T, H, W]`  | `video_swin_unet`   | 3D spatiotemporal reconstruction   |

---

## Usage

### Training

```bash
# Baseline denoising
python train.py --task baseline --model swin_unet_2d \
    --train_dir data/train --val_dir data/val --epochs 100

# 2D sparse reconstruction
python train.py --task spatial_2d --model swin_unet_2d \
    --sparsity_factor 4 --scan_pattern grid \
    --photon_scale 10 --epochs 100

# 3D spatiotemporal reconstruction
python train.py --task spatiotemporal_3d --model video_swin_unet \
    --n_frames 8 --sparsity_factor 4 --scan_pattern interleaved \
    --psf_sigma_lateral 1.0 --psf_sigma_axial 3.0 \
    --epochs 100 --batch_size 2
```

### Inference (large FOV)

```bash
python inference.py --task spatial_2d --model swin_unet_2d \
    --checkpoint runs/my_run/best.pth \
    --input_path data/test/sample.tif \
    --output_dir outputs/

python inference.py --task spatiotemporal_3d --model video_swin_unet \
    --checkpoint runs/my_run/best.pth \
    --input_path data/test/frames/ \
    --output_dir outputs/
```

---

## Ablation Study Guide

Run these back-to-back, then compare TensorBoard curves:

```bash
# Full model
python train.py --task spatial_2d --model swin_unet_2d --run_name full

# No DINOv2 perceptual loss
python train.py --task spatial_2d --model swin_unet_2d --run_name no_dino --no_dino

# No MS-SSIM
python train.py --task spatial_2d --model swin_unet_2d --run_name no_msssim --no_msssim

# MSE only
python train.py --task spatial_2d --model swin_unet_2d --run_name mse_only --no_dino --no_msssim

# No PSF simulation
python train.py --task spatial_2d --model swin_unet_2d --run_name no_psf --no_psf

# No mask channel
python train.py --task spatial_2d --model swin_unet_2d --run_name no_mask_ch --no_mask_ch
```

View all runs:
```bash
tensorboard --logdir runs/
```

---

## Physics Parameters

| Parameter              | Typical 3PM value | Flag                      |
|------------------------|-------------------|---------------------------|
| Lateral PSF sigma (xy) | 0.5–1.5 px        | `--psf_sigma_lateral`     |
| Axial PSF sigma (z)    | 2–5 px            | `--psf_sigma_axial`       |
| Mean photon count      | 5–20              | `--photon_scale`          |
| Read-out noise std     | 0.01–0.05         | `--readout_noise_std`     |
| Sparsity factor        | 2–8               | `--sparsity_factor`       |

---

## Loss Function

```
L_total = λ₁ · L_MSE  +  λ₂ · L_MS-SSIM  +  λ₃ · L_DINOv2
Default:  1.0            0.5                  0.1
```

DINOv2 features are extracted from blocks `{3, 6, 9, 11}` of `dinov2_vitb14` (frozen).  
For offline environments, set `--no_dino` and the backbone will not be downloaded.

---

## CIDC25 Evaluation / Fine-tuning

This project now supports direct evaluation on any `high/low` split directory via:
- `eval.py --eval_split_dir /path/to/split` (must contain `high/` and `low/`)

## Stage-2: 2.5D Pilot (CIDC25)

Use neighboring low frames as multi-channel input to reconstruct center high frame.

### Train (3-frame 2.5D)

```bash
python train.py --task spatial_2p5d --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0 \
  --temporal_window 3 \
  --patch_size 256 --patches_per_image 10 \
  --batch_size 8 --epochs 20 --num_workers 4 \
  --split_val_from_train 0.1 \
  --run_name stage2_2p5d_w3
```

### Evaluate (full-image mode required)

```bash
python eval.py --task spatial_2p5d --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0 \
  --checkpoint /data2/wjb/threePM/runs/stage2_2p5d_w3/best.pth \
  --output_dir /data2/wjb/threePM/eval_results_stage2_2p5d_w3 \
  --eval_split_dir /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0/Test \
  --temporal_window 3 \
  --eval_full_image --no_save_images
```

### 1) Prepare CIDC25 paired frames

CIDC25 provides stacks `[F,H,W]`. Convert them into paired 2D TIFFs first:

```bash
python scripts/prepare_cidc25_pairs.py \
  --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
  --output_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0 \
  --train_gt A0 --train_noisy A1 B1 C2 D2 \
  --val_gt F0 --val_noisy F1 F2 F3
```

If your local Training split has no `A0.tif`, use an available stack as temporary GT
for smoke tests (for example `--train_gt A1 --train_noisy B1 C2 D2`).

### 2) Zero-shot test with an already trained model

Evaluate on CIDC25 Training and Validation separately:

> Protocol safety note: CIDC25 can contain stack TIFFs. Default `--stack_policy error`
> will fail fast when ndim>2 to avoid silent 3D->2D collapse. If you intentionally run
> a 2D baseline, set an explicit policy (e.g. `--stack_policy first --stack_frame_index 0`)
> and document it in the experiment report.

```bash
# Training split test (A0 vs noisy stacks)
python eval.py --task baseline --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0 \
  --checkpoint /data2/wjb/threePM/runs/run_DD/best.pth \
  --output_dir /data2/wjb/threePM/eval_results_CIDC25_zero_train \
  --eval_split_dir /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0/Training \
  --stack_policy first --stack_frame_index 0 \
  --eval_full_image --no_save_images

# Validation split test (F0 vs F1/F2/F3)
python eval.py --task baseline --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0 \
  --checkpoint /data2/wjb/threePM/runs/run_DD/best.pth \
  --output_dir /data2/wjb/threePM/eval_results_CIDC25_zero_val \
  --eval_split_dir /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0/Test \
  --stack_policy first --stack_frame_index 0 \
  --eval_full_image --no_save_images
```

### 3) Fine-tune on CIDC25 Training, then test on Validation

```bash
python train.py --task baseline --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0 \
  --init_checkpoint /data2/wjb/threePM/runs/run_DD/best.pth \
  --epochs 20 --batch_size 8 \
  --patch_size 256 --patches_per_image 50 \
  --split_val_from_train 0 --run_name run_CIDC25_ft

python eval.py --task baseline --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0 \
  --checkpoint /data2/wjb/threePM/runs/run_CIDC25_ft/best.pth \
  --output_dir /data2/wjb/threePM/eval_results_CIDC25_ft_val \
  --eval_split_dir /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A0F0/Test \
  --stack_policy first --stack_frame_index 0 \
  --eval_full_image --no_save_images

### 4) Run protocol guardrail before eval (recommended)

```bash
python scripts/check_eval_protocol.py \
  --split_dir /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0/Test \
  --stack_policy first
```

Replace the split path with your local prepared paired dataset root.
```

---

## Version Iteration Management

To keep project versions organized on GitHub, use:

- Workflow guide: `docs/GIT_WORKFLOW.md`
- Experiment record template: `docs/EXPERIMENT_LOG_TEMPLATE.md`

Recommended daily branch:

```bash
git checkout dev
```

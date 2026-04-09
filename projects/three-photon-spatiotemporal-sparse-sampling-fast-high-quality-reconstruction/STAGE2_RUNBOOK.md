# Stage-2 Runbook (2.5D / 3D)

## Objective
Move from protocol-hardened 2D baseline to evidence for spatiotemporal reconstruction.

## Stage-2A: 2.5D pilot (must run first)

### 1) Protocol guardrail
```bash
python scripts/check_eval_protocol.py \
  --split_dir /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0/Test \
  --stack_policy first
```

### 2) Train 2.5D (window=3)
```bash
python train.py --task spatial_2p5d --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0 \
  --temporal_window 3 \
  --patch_size 256 --patches_per_image 10 \
  --batch_size 8 --epochs 20 --num_workers 4 \
  --split_val_from_train 0.1 \
  --run_name stage2_2p5d_w3
```

### 3) Evaluate 2.5D on Test split
```bash
python eval.py --task spatial_2p5d --model swin_unet_2d \
  --data_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0 \
  --checkpoint /data2/wjb/threePM/runs/stage2_2p5d_w3/best.pth \
  --output_dir /data2/wjb/threePM/eval_results_stage2_2p5d_w3 \
  --eval_split_dir /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_paired_A1F0/Test \
  --temporal_window 3 \
  --eval_full_image --no_save_images
```

### 4) Compare vs 2D baseline
- Compare with the same split and report:
  - PSNR, SSIM, MS-SSIM, LPIPS
  - delta_psnr, delta_ssim
  - runtime and memory footprint

## Stage-2B: 2.5D extension
- Repeat with `--temporal_window 5` and compare w=3 vs w=5.

## Stage-2C: 3D track (next)
- Introduce stack-preserving 3D training/eval path and temporal-consistency metrics.

## Deliverables
- `AUTO_REVIEW.md` append new rounds with evidence
- `REVIEW_STATE.json` status updates
- Stage-2 interim report with baseline-vs-2.5D table

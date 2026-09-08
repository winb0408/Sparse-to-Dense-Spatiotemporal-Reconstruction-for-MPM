"""
options.py  (v2 — Noisy Nuclei edition)
========================================
Central argument parser for the 3PM reconstruction pipeline.
Updated to support the Noisy Nuclei dataset's paired directory layout
and the new NucleiDenoisingDataset arguments.

Usage examples
--------------
# Baseline denoising on Noisy Nuclei:
python train.py --task baseline --model swin_unet_2d \
    --data_root /data/Denoising_Dataset \
    --norm_mode minmax --patches_per_image 50 --epochs 100

# Sparse 2D reconstruction, masking the noisy low image:
python train.py --task spatial_2d --model swin_unet_2d \
    --data_root /data/Denoising_Dataset \
    --sparse_source low --sparsity_factor 4 --scan_pattern grid

# Full ablation suite (run separately, compare in TensorBoard):
python train.py ... --run_name full
python train.py ... --run_name no_dino    --no_dino
python train.py ... --run_name no_msssim  --no_msssim
python train.py ... --run_name mse_only   --no_dino --no_msssim
python train.py ... --run_name no_mask_ch --no_mask_ch
python train.py ... --run_name grid_test  --test_as_grid

# Spatiotemporal 3D with Video Swin (requires separate 3D dataset):
python train.py --task spatiotemporal_3d --model video_swin_unet \
    --train_dir data/3d/train --val_dir data/3d/val --n_frames 8
"""

import argparse
from pathlib import Path


def get_opts() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3PM Sparse-to-Dense Reconstruction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Task / mode ──────────────────────────────────────────────────────────
    p.add_argument(
        "--task",
        type=str,
        default="baseline",
        choices=["baseline", "spatial_2d", "spatial_2p5d", "spatiotemporal_3d"],
        help=(
            "baseline          : dense noisy → dense clean (denoising only)\n"
            "spatial_2d        : 2-D sparse scan → 2-D dense  [B,C,H,W]\n"
            "spatial_2p5d      : multi-frame 2.5D → center-frame dense [B,C,H,W]\n"
            "spatiotemporal_3d : 3-D sparse across time → 3-D dense  [B,C,T,H,W]"
        ),
    )

    # ── Model selection ───────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        type=str,
        default="swin_unet_2d",
        choices=["swin_unet_2d", "video_swin_unet"],
        help=(
            "swin_unet_2d    : 2-D Swin-UNet (baseline / spatial_2d)\n"
            "video_swin_unet : 3-D Video Swin-UNet (spatiotemporal_3d)"
        ),
    )

    # ── Data paths — Noisy Nuclei paired layout ───────────────────────────────
    p.add_argument(
        "--data_root",
        type=str,
        default="/data2/wjb/threePM/ThreePM_openDatasets/Denoising_Dataset",
        help=(
            "Root of the Noisy Nuclei dataset (Denoising_Dataset). Used for\n"
            "validation during training and for eval.py test set. Must contain:\n"
            "  Training/high/  Training/low/  Test/high/  Test/low/"
        ),
    )
    p.add_argument(
        "--train_data_root",
        type=str,
        default=None,
        help=(
            "If set, training uses AI4Life-MDC25 mixed set (large+small) from this path.\n"
            "Expects: large_images/gt, large_images/noisy, small_images/gt, small_images/noisy.\n"
            "Validation and eval still use --data_root (Denoising_Dataset)."
        ),
    )
    # Legacy fallback paths for spatiotemporal_3d (no paired layout)
    p.add_argument("--train_dir", type=str, default=None,
                   help="[spatiotemporal_3d] Directory of training frame TIFFs.")
    p.add_argument("--val_dir",   type=str, default=None,
                   help="[spatiotemporal_3d] Directory of validation frame TIFFs.")

    # Inference paths
    p.add_argument("--input_path", type=str, default=None,
                   help="[inference] Path to input TIFF (file or folder of frames).")
    p.add_argument("--output_dir", type=str, default="outputs",
                   help="[inference] Directory for reconstructed outputs.")

    # ── Dataset-specific parameters ───────────────────────────────────────────
    p.add_argument(
        "--norm_mode",
        type=str,
        default="minmax",
        choices=["minmax", "zscore", "fixed", "minmax255", "fixed255"],
        help=(
            "Per-pair normalisation strategy:\n"
            "  minmax : scale to [0,1] using the pair's global min/max\n"
            "  zscore : zero-mean / unit-std using the clean image's stats\n"
            "  fixed  : no-op (images already in [0,1])\n"
            "  minmax255 : scale to [0,255] using pair min/max (for range-ablation)\n"
            "  fixed255  : treat images as [0,1] then scale to [0,255] (for range-ablation)"
        ),
    )
    p.add_argument(
        "--patches_per_image",
        type=int,
        default=50,
        help=(
            "Number of random crops drawn per image pair per epoch.\n"
            "Effective epoch size = n_pairs × patches_per_image.\n"
            "Recommended ≥ 50 for the 20-pair Noisy Nuclei training set."
        ),
    )
    p.add_argument(
        "--val_root",
        type=str,
        default=None,
        help=(
            "① Independent validation set: path to a directory with high/ and low/.\n"
            "   If set, this is used as validation; no splitting from train."
        ),
    )
    p.add_argument(
        "--split_val_from_train",
        type=float,
        default=0.0,
        help=(
            "② When --val_root is not set: fraction of training pairs to hold out as validation (0–1).\n"
            "   If > 0: split train and use this fraction as validation.\n"
            "   If 0: no validation; use all training data, no val during training (eval on test after)."
        ),
    )
    p.add_argument(
        "--sparse_source",
        type=str,
        default="low",
        choices=["low", "high"],
        help=(
            "[spatial_2d] Which image the sparse mask is applied to:\n"
            "  low  : apply to already-noisy LQ image (realistic)\n"
            "  high : apply to clean GT (physics-informed ablation)"
        ),
    )
    p.add_argument(
        "--test_as_grid",
        action="store_true",
        default=False,
        help=(
            "At test time, tile 1024×1024 images into a non-overlapping "
            "grid of patch_size patches instead of returning the full image.\n"
            "Use this when GPU memory cannot hold the full image."
        ),
    )
    p.add_argument(
        "--stack_policy",
        type=str,
        default="error",
        choices=["error", "first", "mean_projection", "max_projection"],
        help=(
            "How to handle multi-page/stack TIFF (ndim>2):\n"
            "  error           : fail fast (recommended default for protocol safety)\n"
            "  first           : pick one frame by --stack_frame_index\n"
            "  mean_projection : average projection to 2D\n"
            "  max_projection  : max projection to 2D"
        ),
    )
    p.add_argument(
        "--stack_frame_index",
        type=int,
        default=0,
        help="Frame index used when --stack_policy first.",
    )
    p.add_argument(
        "--apply_psf",
        action="store_true",
        default=False,
        help=(
            "Apply 2-D Gaussian PSF blur to the GT target before training.\n"
            "Disabled by default for the Noisy Nuclei dataset (noise is real).\n"
            "Enable for synthetic 3PM simulation experiments."
        ),
    )

    # ── Patch / volume geometry ───────────────────────────────────────────────
    p.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Square patch size for random crops. Must be divisible by 32.",
    )
    p.add_argument("--n_frames",  type=int, default=8,
                   help="[spatiotemporal_3d] Temporal clip length T.")
    p.add_argument(
        "--temporal_window",
        type=int,
        default=3,
        help="[spatial_2p5d] Number of neighboring low frames as input channels (odd number, e.g. 3/5).",
    )
    p.add_argument("--overlap",   type=int, default=32,
                   help="[inference] Spatial overlap for sliding-window tiling.")

    # ── Sparse scanning parameters ────────────────────────────────────────────
    p.add_argument(
        "--scan_pattern",
        type=str,
        default="grid",
        choices=["grid", "random", "stripe", "interleaved"],
        help=(
            "grid        : regular K×K lattice\n"
            "random      : uniformly random pixel locations\n"
            "stripe      : every K-th column (fast-axis scan)\n"
            "interleaved : random-offset grid per frame (3-D temporal diversity)"
        ),
    )
    p.add_argument(
        "--sparsity_factor",
        type=int,
        default=4,
        help="Mask downsample factor K.  Fraction of pixels scanned = 1/K².",
    )
    p.add_argument(
        "--sparse_mode",
        type=str,
        default="xy",
        choices=["x", "y", "xy", "z", "xyz"],
        help=(
            "Sparse sampling mode:\n"
            "  x   : 2D/slice column-wise sparse sampling\n"
            "  y   : 2D/slice row-wise sparse sampling\n"
            "  xy  : 2D spatial subsampling, or per-frame spatial subsampling in 3D\n"
            "  z   : axial subsampling (every K_z-th frame; 3D only)\n"
            "  xyz : combined XY + Z subsampling (3D only)"
        ),
    )
    p.add_argument(
        "--sparsity_factor_z",
        type=int,
        default=4,
        help="Axial (Z) sparsity factor for z/xyz modes.",
    )

    # ── CIDC25 3D volume data ──────────────────────────────────────────────────
    p.add_argument(
        "--cidc25_root",
        type=str,
        default=None,
        help="Path to AI4Life-CIDC25 root for 3D volume training.",
    )
    p.add_argument(
        "--cidc25_gt_stem",
        type=str,
        default="A1",
        help="Stack stem used as GT for 3D sparse reconstruction.",
    )
    p.add_argument(
        "--cidc25_split",
        type=str,
        default="Training",
        choices=["Training", "Validation"],
    )
    p.add_argument(
        "--volume_patch_t",
        type=int,
        default=16,
        help="Temporal depth of 3D volume patches.",
    )
    p.add_argument(
        "--volume_patches_per_stack",
        type=int,
        default=200,
        help="Number of random 3D patches drawn per stack per epoch.",
    )
    p.add_argument(
        "--volume_max_stacks",
        type=int,
        default=None,
        help="Optional cap on number of stacks used for sparse-stack datasets.",
    )

    # ── PSF simulation parameters ─────────────────────────────────────────────
    p.add_argument("--psf_sigma_lateral", type=float, default=1.0,
                   help="Lateral PSF sigma (xy, pixels).")
    p.add_argument("--psf_sigma_axial",   type=float, default=3.0,
                   help="Axial PSF sigma (z/t, pixels).  Typically 2-4× lateral.")
    p.add_argument("--psf_input_channel", action="store_true", default=False,
                   help="Append a PSF kernel-map channel to the model input.")
    p.add_argument("--wiener_init", action="store_true", default=False,
                   help="Use Wiener deconvolution instead of bilinear warm-start.")

    # ── Noise model ───────────────────────────────────────────────────────────
    p.add_argument("--photon_scale",      type=float, default=10.0,
                   help="[synthetic noise] Mean photons at peak intensity.")
    p.add_argument("--readout_noise_std", type=float, default=0.02,
                   help="[synthetic noise] Gaussian read-out noise std.")

    # ── Model architecture ────────────────────────────────────────────────────
    p.add_argument("--embed_dim",     type=int, default=96)
    p.add_argument("--depths",        type=int, nargs="+", default=[2, 2, 6, 2])
    p.add_argument("--num_heads",     type=int, nargs="+", default=[3, 6, 12, 24])
    p.add_argument("--window_size",   type=int, default=8)
    p.add_argument("--window_size_t", type=int, default=2,
                   help="[video_swin_unet] Temporal window size.")
    p.add_argument(
        "--prior_fusion",
        type=str,
        default="none",
        choices=["none", "gated"],
        help=(
            "Prior-aware fusion before patch embedding. "
            "gated uses mask channels to modulate Wiener/warm-start image channels."
        ),
    )

    # ── Loss weights ──────────────────────────────────────────────────────────
    p.add_argument("--lambda_mse",    type=float, default=1.0,
                   help="MSE loss weight (λ₁).")
    p.add_argument("--lambda_msssim", type=float, default=0.5,
                   help="MS-SSIM loss weight (λ₂).")
    p.add_argument("--lambda_dino",   type=float, default=0.1,
                   help="DINOv2 perceptual loss weight (λ₃).")
    p.add_argument("--lambda_psf",   type=float, default=0.0,
                   help="PSF consistency loss weight (λ₄).  0 = disabled.")
    p.add_argument("--dino_layers",   type=int, nargs="+", default=[3, 6, 9, 11],
                   help="DINOv2 block indices for feature extraction.")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=8,
                   help="Recommended 8 for 256×256 patches on a 24 GB GPU.")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--clip_grad",     type=float, default=1.0,
                   help="Max gradient norm for clipping (0 = disabled).")
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many epochs without monitored loss improvement (0 = disabled).",
    )
    p.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum monitored loss decrease required to reset early stopping patience.",
    )
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--use_amp",       action="store_true", default=False)
    p.add_argument("--seed",          type=int,   default=42)

    # ── Eval-only options ─────────────────────────────────────────────────────
    p.add_argument(
        "--eval_full_image",
        action="store_true",
        help=(
            "[eval] Use full-image forward (one 1024×1024 pass per test image) "
            "instead of sliding-window grid inference."
        ),
    )
    p.add_argument("--eval_split_dir", type=str, default=None,
                   help="[eval] Optional directory with high/ and low/ for direct evaluation.")
    p.add_argument("--eval_checkpoints", nargs="+", type=str, default=None,
                   help="[eval] Multiple checkpoints for ablation evaluation.")
    p.add_argument("--eval_run_names", nargs="+", type=str, default=None,
                   help="[eval] Run names corresponding to --eval_checkpoints.")
    p.add_argument("--no_save_images", action="store_true",
                   help="[eval] Do not save npy/tif prediction files.")
    p.add_argument("--cnr_signal_thresh", type=float, default=0.5,
                   help="[eval] Foreground threshold used by CNR metric.")
    p.add_argument("--n_bootstrap", type=int, default=2000,
                   help="[eval] Number of bootstrap rounds for confidence intervals.")

    # ── Logging / checkpointing ───────────────────────────────────────────────
    p.add_argument("--run_name",   type=str, default="run")
    p.add_argument("--log_dir",    type=str, default="runs")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="Load model weights only (for fine-tuning), without resuming epoch/optimizer.")

    # ── Protocol guard ───────────────────────────────────────────────────────
    p.add_argument(
        "--dataset_protocol",
        type=str,
        default="auto",
        choices=["auto", "cidc25_3d", "mdc25_2d", "mdc25_3d", "pollen_2d", "generic_2d", "generic_3d"],
        help=(
            "Dataset/task protocol guard. Use auto for path-based inference. "
            "2D protocols disallow z/xyz and 3D main training unless explicitly redesigned."
        ),
    )
    p.add_argument(
        "--allow_protocol_mismatch",
        action="store_true",
        default=False,
        help="Allow cross-dimensional transfer/sanity runs that are not valid main experiments.",
    )

    # ── Ablation flags ────────────────────────────────────────────────────────
    p.add_argument("--no_psf",      action="store_true",
                   help="[ablation] Disable PSF blur (raw GT used as target).")
    p.add_argument("--no_dino",     action="store_true",
                   help="[ablation] Disable DINOv2 perceptual loss.")
    p.add_argument("--no_msssim",   action="store_true",
                   help="[ablation] Disable MS-SSIM loss.")
    p.add_argument("--no_mask_ch",  action="store_true",
                   help="[ablation] Disable mask as 2nd input channel.")

    opts = p.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    assert len(opts.depths) == len(opts.num_heads), \
        "--depths and --num_heads must have the same number of stages."
    assert opts.patch_size % 32 == 0, \
        "--patch_size must be divisible by 32."
    if opts.prior_fusion != "none" and opts.no_mask_ch:
        p.error("--prior_fusion requires mask channels. Remove --no_mask_ch or use --prior_fusion none.")
    if opts.task == "spatial_2p5d":
        if opts.temporal_window < 3 or opts.temporal_window % 2 == 0:
            p.error("--temporal_window for spatial_2p5d must be an odd number >= 3.")

    data_root_lower = str(opts.data_root).lower()
    inferred_protocol = opts.dataset_protocol
    if inferred_protocol == "auto":
        if opts.cidc25_root:
            inferred_protocol = "cidc25_3d"
        elif "mdc25_sparse/2d" in data_root_lower:
            inferred_protocol = "mdc25_2d"
        elif "mdc25_sparse/3d" in data_root_lower:
            inferred_protocol = "mdc25_3d"
        elif "pollen" in data_root_lower:
            inferred_protocol = "pollen_2d"
        elif (Path(opts.data_root) / "Training" / "stacks").is_dir():
            inferred_protocol = "generic_3d"
        else:
            inferred_protocol = "generic_2d"
    opts.dataset_protocol = inferred_protocol

    if not opts.allow_protocol_mismatch:
        if inferred_protocol in {"mdc25_2d", "pollen_2d", "generic_2d"}:
            if opts.task in {"spatial_2p5d", "spatiotemporal_3d"}:
                p.error(
                    f"--dataset_protocol {inferred_protocol} is 2D and cannot be used as a main "
                    f"{opts.task} experiment. Use --allow_protocol_mismatch only for transfer probes."
                )
            if opts.sparse_mode in {"z", "xyz"}:
                p.error(
                    f"--sparse_mode {opts.sparse_mode!r} has 3D axial meaning and is invalid for "
                    f"--dataset_protocol {inferred_protocol}. Use x/y/xy or mark the run with --allow_protocol_mismatch."
                )
        if inferred_protocol in {"cidc25_3d", "mdc25_3d", "generic_3d"}:
            if opts.task == "baseline":
                p.error(
                    f"--task baseline is a 2D denoising protocol, not a main experiment for "
                    f"--dataset_protocol {inferred_protocol}. Use spatial_2d/2p5d/3d or --allow_protocol_mismatch."
                )
            if opts.task == "spatial_2p5d" and opts.sparse_mode != "z":
                p.error(
                    "The reorganized 2.5D main protocol uses --sparse_mode z. "
                    "Use --allow_protocol_mismatch for xy/xyz transfer or ablation probes."
                )
            if opts.task == "spatiotemporal_3d" and opts.sparse_mode != "xyz":
                p.error(
                    "The reorganized 3D main protocol uses --sparse_mode xyz. "
                    "Use --allow_protocol_mismatch for x/y/xy/z ablation probes."
                )

    # For spatiotemporal_3d:
    # allow three input styles:
    # 1) --cidc25_root (legacy CIDC25 volume path)
    # 2) --train_dir + --val_dir (legacy flat frame dirs)
    # 3) --data_root containing Training/stacks (new sparse-stack flow, e.g. MDC25_sparse)
    if opts.task == "spatiotemporal_3d":
        has_cidc25 = getattr(opts, "cidc25_root", None) is not None
        has_legacy = opts.train_dir is not None and opts.val_dir is not None
        has_stack_root = (Path(opts.data_root) / "Training" / "stacks").is_dir()
        if not has_cidc25 and not has_legacy and not has_stack_root:
            p.error(
                "--task spatiotemporal_3d requires either --cidc25_root "
                "or both --train_dir and --val_dir, or --data_root with Training/stacks."
            )

    opts.log_path = Path(opts.log_dir) / opts.run_name
    opts.log_path.mkdir(parents=True, exist_ok=True)

    return opts


if __name__ == "__main__":
    opts = get_opts()
    for k, v in vars(opts).items():
        print(f"  {k:25s}: {v}")

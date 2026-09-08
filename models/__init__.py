"""
models/__init__.py
==================
Model registry for the 3PM reconstruction pipeline.

Usage
-----
from models import build_model
model = build_model(opts)
"""

from pathlib import Path

from .swin_unet_2d   import SwinUNet2D
from .video_swin_unet import VideoSwinUNet


def build_model(opts):
    """
    Instantiate the appropriate model from options.

    Parameters
    ----------
    opts : argparse.Namespace
        Must include: opts.model, opts.task, opts.embed_dim, opts.depths,
        opts.num_heads, opts.window_size, opts.window_size_t, opts.patch_size,
        opts.n_frames, opts.no_mask_ch.

    Returns
    -------
    nn.Module
    """
    # Input channels depend on task:
    #   baseline      -> always 1 (no noisy+mask dual-channel design)
    #   spatial_2d    -> 2 by default (interp image + binary mask), 1 with --no_mask_ch
    #   spatial_2p5d  -> temporal_window (multi-frame channels)
    #   spatiotemporal_3d -> 1 or 2 (warm-start + optional mask) for 5D input
    #
    # Additional channels:
    #   --psf_input_channel adds +1 for PSF kernel map
    force_in = getattr(opts, "force_in_channels", None)
    use_masked_2p5d = False
    if force_in is not None:
        in_ch = int(force_in)
    elif opts.task == "baseline":
        in_ch = 1
    elif opts.task == "spatial_2p5d":
        tw = int(getattr(opts, "temporal_window", 3))
        # Backward compatibility:
        # - legacy paired-frame 2.5D dataset returns [T, H, W]
        # - new CIDC25 volume 2.5D path can return [2T, H, W] with mask channels
        data_root = getattr(opts, "data_root", None)
        use_stack_2p5d = False
        if data_root is not None:
            dr = Path(str(data_root))
            use_stack_2p5d = (dr / "Training" / "stacks").is_dir()
        use_masked_2p5d = (getattr(opts, "cidc25_root", None) is not None) or use_stack_2p5d
        if use_masked_2p5d and not getattr(opts, "no_mask_ch", False):
            in_ch = tw * 2  # [T interp + T mask]
        else:
            in_ch = tw      # legacy [T] or ablation without mask
    else:
        in_ch = 1 if opts.no_mask_ch else 2

    if getattr(opts, "psf_input_channel", False):
        in_ch += 1

    prior_fusion = getattr(opts, "prior_fusion", "none")
    prior_image_ch = None
    prior_mask_ch = None
    if prior_fusion != "none" and not getattr(opts, "no_mask_ch", False):
        if opts.task == "spatial_2p5d" and use_masked_2p5d:
            tw = int(getattr(opts, "temporal_window", 3))
            prior_image_ch = tw
            prior_mask_ch = tw
        elif opts.task == "spatiotemporal_3d":
            prior_image_ch = 1
            prior_mask_ch = 1
        elif opts.task == "spatial_2d":
            prior_image_ch = 1
            prior_mask_ch = 1

    out_ch = 1   # always single-channel output (greyscale reconstructed image)

    if opts.model == "swin_unet_2d":
        return SwinUNet2D(
            in_channels  = in_ch,
            out_channels = out_ch,
            img_size     = opts.patch_size,
            embed_dim    = opts.embed_dim,
            depths       = opts.depths,
            num_heads    = opts.num_heads,
            window_size  = opts.window_size,
            prior_fusion = prior_fusion,
            prior_image_channels = prior_image_ch,
            prior_mask_channels = prior_mask_ch,
        )

    elif opts.model == "video_swin_unet":
        return VideoSwinUNet(
            in_channels    = in_ch,
            out_channels   = out_ch,
            img_size       = opts.patch_size,
            n_frames       = opts.n_frames,
            embed_dim      = opts.embed_dim,
            depths         = opts.depths,
            num_heads      = opts.num_heads,
            window_size    = (opts.window_size_t, opts.window_size, opts.window_size),
            prior_fusion   = prior_fusion,
            prior_image_channels = prior_image_ch,
            prior_mask_channels = prior_mask_ch,
        )

    else:
        raise ValueError(f"Unknown model: {opts.model!r}")

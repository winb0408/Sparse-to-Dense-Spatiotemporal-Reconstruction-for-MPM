"""
models/__init__.py
==================
Model registry for the 3PM reconstruction pipeline.

Usage
-----
from models import build_model
model = build_model(opts)
"""

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
    #   spatiotemporal_3d follows the same mask-channel rule as sparse tasks
    if opts.task == "baseline":
        in_ch = 1
    elif opts.task == "spatial_2p5d":
        in_ch = int(getattr(opts, "temporal_window", 3))
    else:
        in_ch = 1 if opts.no_mask_ch else 2
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
        )

    else:
        raise ValueError(f"Unknown model: {opts.model!r}")

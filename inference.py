"""
inference.py
============
Sliding-window (2-D) and temporal-window (3-D) inference for the 3PM pipeline.

Handles arbitrarily large field-of-view images / volumes without GPU memory limits:
  - 2-D: overlapping spatial patches with Hann-window blending
  - 3-D: overlapping spatiotemporal clips with Hann-window blending in all 3 dims

Usage
-----
# Reconstruct a large 2D FOV:
python inference.py --task spatial_2d --model swin_unet_2d \
    --checkpoint runs/my_run/best.pth \
    --input_path data/test/sample.tif \
    --output_dir outputs/

# Reconstruct a 3D volume (multi-frame TIFF or folder of frames):
python inference.py --task spatiotemporal_3d --model video_swin_unet \
    --checkpoint runs/my_run/best.pth \
    --input_path data/test/volume_folder/ \
    --output_dir outputs/
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from options  import get_opts
from models   import build_model
from dataset  import (
    load_image_as_tensor,
    apply_psf_blur_2d,
    apply_psf_blur_3d,
    add_acquisition_noise,
    make_mask_2d,
    make_mask_3d,
    warm_start_2d,
    warm_start_3d,
)

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


# ─────────────────────────────────────────────
# Hann window factory
# ─────────────────────────────────────────────

def hann_window_2d(size: int, device: torch.device) -> torch.Tensor:
    """2-D Hann window for smooth patch blending. Shape: [1, 1, size, size]."""
    h = torch.hann_window(size, periodic=False, device=device)
    w = torch.hann_window(size, periodic=False, device=device)
    return (h.unsqueeze(1) * w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)


def hann_window_3d(
    t: int, h: int, w: int, device: torch.device
) -> torch.Tensor:
    """
    3-D Hann window for spatiotemporal blending.
    Shape: [1, 1, t, h, w]
    """
    ht = torch.hann_window(t, periodic=False, device=device)
    hh = torch.hann_window(h, periodic=False, device=device)
    hw = torch.hann_window(w, periodic=False, device=device)
    # Outer product along all three dims
    win = ht.view(t, 1, 1) * hh.view(1, h, 1) * hw.view(1, 1, w)
    return win.unsqueeze(0).unsqueeze(0)                   # [1, 1, t, h, w]


# ─────────────────────────────────────────────
# 2-D sliding-window inference
# ─────────────────────────────────────────────

@torch.no_grad()
def infer_2d(
    model:       torch.nn.Module,
    inp:         torch.Tensor,      # [1, C_in, H, W]  — full FOV input
    patch_size:  int,
    overlap:     int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Sliding-window inference with Hann-windowed blending.

    Handles images larger than the training patch_size without boundary
    artefacts.  The output is a weighted-average of overlapping patches.

    Shape flow
    ----------
    inp      : [1, C_in, H, W]
    patch    : [1, C_in, ps, ps]  → model → [1, 1, ps, ps]
    output   : [1, 1, H, W]
    """
    model.eval().to(device)
    _, C, H, W  = inp.shape
    ps          = patch_size
    stride      = ps - overlap

    output  = torch.zeros(1, 1, H, W, device=device)
    weights = torch.zeros(1, 1, H, W, device=device)
    win2d   = hann_window_2d(ps, device)                   # [1, 1, ps, ps]

    # Pad image so every patch fits (reflect padding avoids edge artefacts)
    pad_h   = (stride - H % stride) % stride
    pad_w   = (stride - W % stride) % stride
    inp_pad = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = inp_pad.shape

    for y in range(0, Hp - ps + 1, stride):
        for x in range(0, Wp - ps + 1, stride):
            patch = inp_pad[:, :, y:y+ps, x:x+ps].to(device)   # [1, C, ps, ps]
            pred  = model(patch)                                  # [1, 1, ps, ps]
            output[:, :, y:y+ps, x:x+ps]  += pred  * win2d
            weights[:, :, y:y+ps, x:x+ps] += win2d

    # Crop back to original size
    output  = (output  / weights.clamp(min=1e-6))[:, :, :H, :W]
    return output                                               # [1, 1, H, W]


# ─────────────────────────────────────────────
# 3-D sliding-window inference
# ─────────────────────────────────────────────

@torch.no_grad()
def infer_3d(
    model:       torch.nn.Module,
    inp:         torch.Tensor,      # [1, C_in, T, H, W]  — full volume
    n_frames:    int,               # temporal clip size (= training T)
    patch_size:  int,
    overlap_s:   int,               # spatial overlap
    overlap_t:   int,               # temporal overlap
    device:      torch.device,
) -> torch.Tensor:
    """
    Spatiotemporal sliding-window inference.

    Iterates over (T, H, W) with configurable overlap and blends predictions
    using a 3-D Hann window to eliminate seam artefacts.

    Shape flow
    ----------
    inp   : [1, C_in, T, H, W]
    clip  : [1, C_in, nf, ps, ps]  → model → [1, 1, nf, ps, ps]
    output: [1, 1, T, H, W]
    """
    model.eval().to(device)
    _, C, T, H, W = inp.shape
    ps            = patch_size
    nf            = n_frames
    stride_s      = ps - overlap_s
    stride_t      = nf - overlap_t

    output  = torch.zeros(1, 1, T, H, W, device=device)
    weights = torch.zeros(1, 1, T, H, W, device=device)
    win3d   = hann_window_3d(nf, ps, ps, device)             # [1, 1, nf, ps, ps]

    # Spatial padding (reflect, no temporal padding for simplicity)
    pad_h   = (stride_s - H % stride_s) % stride_s
    pad_w   = (stride_s - W % stride_s) % stride_s
    inp_pad = F.pad(inp, (0, pad_w, 0, pad_h, 0, 0), mode="reflect")
    _, _, Tp, Hp, Wp = inp_pad.shape

    for t in range(0, max(1, T - nf + 1), stride_t):
        t_end = min(t + nf, T)
        t_sl  = slice(t, t + nf)
        # If clip is shorter than nf at the end, pad temporally
        clip_t = inp_pad[:, :, t_sl]
        if clip_t.shape[2] < nf:
            pad_frames = nf - clip_t.shape[2]
            clip_t = F.pad(clip_t, (0, 0, 0, 0, 0, pad_frames), mode="reflect")

        for y in range(0, Hp - ps + 1, stride_s):
            for x in range(0, Wp - ps + 1, stride_s):
                clip = clip_t[:, :, :, y:y+ps, x:x+ps].to(device)  # [1,C,nf,ps,ps]
                pred = model(clip)                                   # [1,1,nf,ps,ps]
                # Only accumulate the valid (non-padded) temporal frames
                valid_t = t_end - t
                output[:, :, t:t+valid_t, y:y+ps, x:x+ps]  += \
                    (pred * win3d)[:, :, :valid_t]
                weights[:, :, t:t+valid_t, y:y+ps, x:x+ps] += \
                    win3d[:, :, :valid_t]

    output = (output / weights.clamp(min=1e-6))[:, :, :T, :H, :W]
    return output                                               # [1, 1, T, H, W]


# ─────────────────────────────────────────────
# Input preparation helpers
# ─────────────────────────────────────────────

def prepare_input_2d(
    img_path: Path, opts
) -> torch.Tensor:
    """
    Load, PSF-blur, noise, mask, and warm-start a single 2-D image.
    Returns [1, C_in, H, W] ready for inference.
    (In real deployment, input is the raw sparse measurement — here we
    simulate it from a GT image for evaluation purposes.)
    """
    gt = load_image_as_tensor(img_path)            # [1, H, W]
    if not opts.no_psf:
        gt = apply_psf_blur_2d(gt, opts.psf_sigma_lateral)
    noisy  = add_acquisition_noise(gt, opts.photon_scale, opts.readout_noise_std)
    _, H, W = gt.shape
    mask   = make_mask_2d(H, W, opts.sparsity_factor, opts.scan_pattern)
    sparse = noisy * mask
    interp = warm_start_2d(sparse, mask, opts.sparsity_factor)
    interp_n = (interp - 0.5) / 0.5               # normalise to [-1,1]

    if not opts.no_mask_ch:
        inp = torch.cat([interp_n, mask], dim=0).unsqueeze(0)  # [1, 2, H, W]
    else:
        inp = interp_n.unsqueeze(0)                            # [1, 1, H, W]
    return inp


def prepare_input_3d(
    frame_paths: List[Path], opts
) -> torch.Tensor:
    """
    Load a sequence of frames, simulate 3-D acquisition, return [1, C_in, T, H, W].
    """
    frames = [load_image_as_tensor(p) for p in frame_paths]  # list of [1, H, W]
    vol = torch.stack(frames, dim=1)                          # [1, T, H, W]

    if not opts.no_psf:
        vol = apply_psf_blur_3d(vol, opts.psf_sigma_lateral, opts.psf_sigma_axial)
    noisy = add_acquisition_noise(vol, opts.photon_scale, opts.readout_noise_std)

    _, T, H, W = noisy.shape
    mask   = make_mask_3d(T, H, W, opts.sparsity_factor, "interleaved")
    sparse = noisy * mask
    interp = warm_start_3d(sparse, mask, opts.sparsity_factor)
    interp_n = (interp - 0.5) / 0.5

    if not opts.no_mask_ch:
        inp = torch.cat([interp_n, mask], dim=0).unsqueeze(0)  # [1, 2, T, H, W]
    else:
        inp = interp_n.unsqueeze(0)                            # [1, 1, T, H, W]
    return inp


# ─────────────────────────────────────────────
# Save output
# ─────────────────────────────────────────────

def save_output(tensor: torch.Tensor, out_path: Path):
    """
    Save a [1, 1, H, W] or [1, 1, T, H, W] tensor as float32 TIFF.
    Values are rescaled from [-1,1] → [0,1] → uint16.
    """
    arr = tensor.squeeze().cpu().numpy()          # [H, W] or [T, H, W]
    arr = (arr * 0.5 + 0.5).clip(0, 1)           # [-1,1] → [0,1]
    arr = (arr * 65535).astype(np.uint16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_TIFFFILE:
        tifffile.imwrite(str(out_path), arr)
    else:
        # Fallback: save as PNG (2D only)
        from PIL import Image
        if arr.ndim == 2:
            Image.fromarray((arr / 256).astype(np.uint8)).save(str(out_path))
        else:
            for i, frame in enumerate(arr):
                Image.fromarray((frame / 256).astype(np.uint8)).save(
                    str(out_path).replace(".tif", f"_f{i:04d}.png")
                )
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────
# Main inference entry point
# ─────────────────────────────────────────────

def main():
    opts   = get_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Inference] task={opts.task}  model={opts.model}  device={device}\n")

    assert opts.checkpoint is not None, \
        "Provide --checkpoint path to a trained model."
    assert opts.input_path is not None, \
        "Provide --input_path (image file or frame folder)."

    # ── Load model ─────────────────────────────────────────────────────────
    model = build_model(opts).to(device)
    ckpt  = torch.load(opts.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch','?')}")

    out_dir   = Path(opts.output_dir)
    inp_path  = Path(opts.input_path)

    if opts.task in ("baseline", "spatial_2d"):
        # ── 2-D inference ────────────────────────────────────────────────────
        files = sorted(inp_path.rglob("*.tif"))  if inp_path.is_dir() \
                else [inp_path]

        for img_file in files:
            print(f"  Processing {img_file.name} …")
            inp   = prepare_input_2d(img_file, opts)        # [1, C, H, W]
            pred  = infer_2d(model, inp, opts.patch_size, opts.overlap, device)
            save_output(pred, out_dir / (img_file.stem + "_recon.tif"))

    elif opts.task == "spatiotemporal_3d":
        # ── 3-D inference ────────────────────────────────────────────────────
        if inp_path.is_dir():
            frame_paths = sorted(inp_path.rglob("*.tif")) + \
                          sorted(inp_path.rglob("*.png"))
        else:
            # Single multi-page TIFF  (load pages as frames)
            if _HAS_TIFFFILE:
                import tifffile as tf
                vol_np = tf.imread(str(inp_path))          # [T, H, W] or [H, W]
                if vol_np.ndim == 2:
                    vol_np = vol_np[None]
                # Save individual pages to temp files
                import tempfile, os
                tmpdir = Path(tempfile.mkdtemp())
                frame_paths = []
                for i, frame in enumerate(vol_np):
                    p = tmpdir / f"frame_{i:04d}.tif"
                    tf.imwrite(str(p), frame)
                    frame_paths.append(p)
            else:
                raise ValueError("Install tifffile for multi-page TIFF inference.")

        print(f"  Found {len(frame_paths)} frames.")
        inp  = prepare_input_3d(frame_paths, opts)          # [1, C, T, H, W]
        pred = infer_3d(
            model, inp,
            n_frames   = opts.n_frames,
            patch_size = opts.patch_size,
            overlap_s  = opts.overlap,
            overlap_t  = max(1, opts.n_frames // 4),
            device     = device,
        )
        save_output(pred, out_dir / (inp_path.stem + "_recon.tif"))

    print("\n  Done.")


if __name__ == "__main__":
    main()

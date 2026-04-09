"""
dataset.py  (v2 — Noisy Nuclei edition)
========================================
Data loading and augmentation for the 3PM reconstruction pipeline,
rigorously adapted for the Noisy Nuclei dataset
(Guillaume Jacquemet, 2021 — SiR-DNA fluorescence, 16-bit TIFF, 1024×1024).

Expected directory layout
--------------------------
Denoising_Dataset/
    Training/
        high/   ←  ground-truth / clean images  (GT)
        low/    ←  noisy acquisitions            (LQ)
    Test/
        high/
        low/

Design decisions
----------------
1. 16-bit TIFF loading via tifffile — PIL silently truncates to 8-bit.
2. Per-pair normalisation (min-max or z-score) to handle fluorescence
   intensity variance across acquisitions.
3. Dihedral-group augmentation (all 8 D4 symmetries) applied identically
   to both images in a pair — prevents the Transformer from over-fitting
   on the tiny 20-pair training set.
4. Virtual index expansion: each __getitem__ draws a fresh random
   256×256 crop from the 1024×1024 image.  With patches_per_image=50
   the effective epoch size is 20×50 = 1 000 samples.
5. Task-aware output:
     baseline    → (low_patch, high_patch)            shapes [1,ps,ps]
     spatial_2d  → (sparse_inpainted_patch, high_patch)
                   shapes [2,ps,ps] or [1,ps,ps] depending on use_mask_channel
6. Test mode returns full 1024×1024 images (batch_size=1) or a
   deterministic non-overlapping 4×4 grid of patches via NucleiGridDataset.

Tensor conventions
------------------
  All tensors are float32.
  Model input/output range: [-1, 1]  (linear map from [0,1]).
  Binary mask channel stays in {0, 1} (not normalised).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Any
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# ── tifffile is mandatory — PIL truncates 16-bit to 8-bit ────────────────────
try:
    import tifffile
except ImportError as exc:
    raise ImportError(
        "tifffile is required for correct 16-bit TIFF loading.\n"
        "Install with:  pip install tifffile"
    ) from exc


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Low-level TIFF I/O
# ═════════════════════════════════════════════════════════════════════════════

def load_tiff_float32(
    path: Path,
    stack_policy: str = "error",
    frame_index: int = 0,
) -> np.ndarray:
    """
    Load any TIFF as a float32 numpy array in **[0, 1]**.

    Handles:
      - uint16  (most fluorescence microscopes, max=65535)
      - uint8   (max=255)
      - float32 / float64  (already normalised or raw)

    Multi-page / multi-channel TIFFs are handled by `stack_policy`:
      - error          : raise ValueError (safe default, avoids silent misuse)
      - first          : use one frame by index (`frame_index`)
      - mean_projection: average across all leading dimensions
      - max_projection : max projection across all leading dimensions

    Returns
    -------
    arr : np.ndarray  [H, W]  float32  ∈ [0, 1]
    """
    raw = tifffile.imread(str(path))   # preserves original dtype (uint16, etc.)

    # Make stack handling explicit so 3D/4D data is not silently collapsed.
    if raw.ndim > 2:
        if stack_policy == "error":
            raise ValueError(
                f"Stack TIFF detected at {path} with shape {tuple(raw.shape)}. "
                "Set an explicit stack policy (first/mean_projection/max_projection) "
                "for 2D baselines, or use a spatiotemporal pipeline."
            )
        if stack_policy == "first":
            # Repeatedly select the requested frame on leading axes until 2D.
            while raw.ndim > 2:
                axis_len = raw.shape[0]
                idx = min(max(frame_index, 0), axis_len - 1)
                raw = raw[idx]
        elif stack_policy == "mean_projection":
            while raw.ndim > 2:
                raw = raw.mean(axis=0)
        elif stack_policy == "max_projection":
            while raw.ndim > 2:
                raw = raw.max(axis=0)
        else:
            raise ValueError(
                f"Unknown stack_policy={stack_policy!r}. "
                "Use: error | first | mean_projection | max_projection."
            )

    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    elif raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    elif np.issubdtype(raw.dtype, np.floating):
        arr = raw.astype(np.float32)
        # If already in [0,1] leave as-is; otherwise min-max rescale
        if arr.min() >= 0.0 and arr.max() <= 1.0:
            return arr
        vmin, vmax = float(arr.min()), float(arr.max())
        return ((arr - vmin) / (vmax - vmin + 1e-8)).clip(0.0, 1.0)
    else:
        # Fallback: cast and min-max
        arr = raw.astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        return ((arr - vmin) / (vmax - vmin + 1e-8)).clip(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Scientific normalisation
# ═════════════════════════════════════════════════════════════════════════════

NormMode = Literal["minmax", "zscore", "fixed"]


def normalise_pair(
    low:  np.ndarray,
    high: np.ndarray,
    mode: NormMode = "minmax",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalise a paired (low, high) image consistently.

    Parameters
    ----------
    low, high : [H, W] float32 arrays in [0, 1] (from load_tiff_float32).
    mode :
      'minmax'  — scale using the pair's global [min, max].
                  Preserves relative intensity; recommended for Noisy Nuclei
                  (SiR-DNA fluorescence has a meaningful zero background).
      'zscore'  — subtract high's mean, divide by high's std.
                  Useful when acquisition gain varies across images.
      'fixed'   — no-op (arrays already in [0, 1]).

    Returns
    -------
    low_n, high_n : normalised float32 arrays.
      - 'minmax' / 'fixed'  →  ∈ [0, 1]
      - 'zscore'            →  zero-mean, unit-variance (can exceed [0,1])
    """
    if mode == "minmax":
        g_min = min(float(low.min()), float(high.min()))
        g_max = max(float(low.max()), float(high.max()))
        denom = g_max - g_min + 1e-8
        return (
            ((low  - g_min) / denom).clip(0.0, 1.0).astype(np.float32),
            ((high - g_min) / denom).clip(0.0, 1.0).astype(np.float32),
        )
    elif mode == "zscore":
        mu  = float(high.mean())
        std = float(high.std()) + 1e-8
        return (
            ((low  - mu) / std).astype(np.float32),
            ((high - mu) / std).astype(np.float32),
        )
    elif mode == "fixed":
        return low.astype(np.float32), high.astype(np.float32)
    else:
        raise ValueError(f"Unknown norm_mode: {mode!r}. Choose minmax | zscore | fixed.")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Physics helpers  (PSF, noise, sparse masks, warm-start inpainting)
#     Kept here so dataset.py is self-contained; also imported by inference.py
# ═════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel for separable convolution."""
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def apply_psf_blur_2d(
    img:            torch.Tensor,   # [1, H, W]
    sigma_lateral:  float,
) -> torch.Tensor:
    """
    Separable 2-D isotropic Gaussian PSF blur.

    Shape: [1, H, W] → [1, H, W].
    Used when apply_psf=True to simulate the optical focal spot.
    """
    if sigma_lateral < 1e-6:
        return img
    k   = _gaussian_kernel_1d(sigma_lateral)
    pad = len(k) // 2
    x   = img.unsqueeze(0)                               # [1, 1, H, W]
    x   = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, pad))
    x   = F.conv2d(x, k.view(1, 1, -1, 1), padding=(pad, 0))
    return x.squeeze(0)                                  # [1, H, W]


def apply_psf_blur_3d(
    vol:           torch.Tensor,    # [1, T, H, W]
    sigma_lateral: float,
    sigma_axial:   float,
) -> torch.Tensor:
    """
    Anisotropic separable 3-D Gaussian PSF  (axial σ >> lateral σ in 3PM).

    Shape flow:
        [1, T, H, W]
         → unsqueeze(0) → [1, 1, T, H, W]
         → separable conv3d in t, h, w
         → squeeze(0)   → [1, T, H, W]
    """
    if sigma_lateral < 1e-6 and sigma_axial < 1e-6:
        return vol
    x = vol.unsqueeze(0)                                 # [1, 1, T, H, W]
    if sigma_axial >= 1e-6:
        kt  = _gaussian_kernel_1d(sigma_axial)
        pad = len(kt) // 2
        x   = F.conv3d(x, kt.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
    if sigma_lateral >= 1e-6:
        kl  = _gaussian_kernel_1d(sigma_lateral)
        pad = len(kl) // 2
        x   = F.conv3d(x, kl.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
        x   = F.conv3d(x, kl.view(1, 1, 1, 1, -1), padding=(0, 0, pad))
    return x.squeeze(0)                                  # [1, T, H, W]


def add_acquisition_noise(
    img:          torch.Tensor,
    photon_scale: float,
    readout_std:  float,
) -> torch.Tensor:
    """
    Poisson shot noise + Gaussian read-out noise.

    img ∈ [0, 1].  Poisson(λ): variance = mean, dominant at low photon counts.
    Returns clamped tensor ∈ [0, 1].
    """
    noisy = torch.poisson(img * photon_scale) / photon_scale
    noisy = noisy + torch.randn_like(noisy) * readout_std
    return noisy.clamp(0.0, 1.0)


def make_mask_2d(
    H: int, W: int,
    sparsity_factor: int,
    pattern: str,
) -> torch.Tensor:
    """
    Binary sparse-scanning mask  [1, H, W].   1 = scanned pixel.

    Patterns
    --------
    grid        : regular K×K lattice  (offset 0,0)
    interleaved : grid with random spatial offset  (for 3-D temporal diversity)
    random      : uniformly random 1/K² pixel selection
    stripe      : every K-th column  (fast-axis scanning)
    """
    K    = sparsity_factor
    mask = torch.zeros(1, H, W)
    if pattern in ("grid", "interleaved"):
        oh = random.randint(0, K - 1) if pattern == "interleaved" else 0
        ow = random.randint(0, K - 1) if pattern == "interleaved" else 0
        mask[:, oh::K, ow::K] = 1.0
    elif pattern == "random":
        n = max(1, H * W // (K * K))
        mask[0, torch.randint(0, H, (n,)), torch.randint(0, W, (n,))] = 1.0
    elif pattern == "stripe":
        mask[:, :, random.randint(0, K - 1)::K] = 1.0
    else:
        raise ValueError(f"Unknown scan_pattern: {pattern!r}")
    return mask


def make_mask_3d(
    T: int, H: int, W: int,
    sparsity_factor: int,
    pattern: str,
) -> torch.Tensor:
    """Binary sparse-scanning mask  [1, T, H, W]."""
    frames = [make_mask_2d(H, W, sparsity_factor, pattern) for _ in range(T)]
    return torch.stack(frames, dim=1)               # [1, T, H, W]


def warm_start_2d(
    sparse: torch.Tensor,   # [1, H, W]  masked (zero-filled) image
    factor: int,
) -> torch.Tensor:
    """
    Bilinear down ↓ factor then up ↑ to fill unscanned pixels.

    Returns [1, H, W].  Provides a smooth initialisation for the network
    rather than zero-padding the missing pixels.
    """
    H, W = sparse.shape[-2:]
    x = sparse.unsqueeze(0)                                          # [1, 1, H, W]
    x = F.interpolate(x, scale_factor=1.0 / factor,
                      mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(H, W),
                      mode="bilinear", align_corners=False)
    return x.squeeze(0)                                              # [1, H, W]


def warm_start_3d(
    sparse: torch.Tensor,   # [1, T, H, W]  masked volume
    factor: int,
) -> torch.Tensor:
    """
    Trilinear warm-start across (T, H, W).

    Shape flow:
        [1, T, H, W]
         → unsqueeze(0) → [1, 1, T, H, W]
         → trilinear ↓  →  trilinear ↑
         → squeeze(0)   → [1, T, H, W]
    """
    _, T, H, W = sparse.shape
    x = sparse.unsqueeze(0)                                          # [1, 1, T, H, W]
    x = F.interpolate(x,
                      scale_factor=(1.0, 1.0 / factor, 1.0 / factor),
                      mode="trilinear", align_corners=False)
    x = F.interpolate(x, size=(T, H, W),
                      mode="trilinear", align_corners=False)
    return x.squeeze(0)                                              # [1, T, H, W]


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Dihedral augmentation
# ═════════════════════════════════════════════════════════════════════════════

def dihedral_augment(
    *tensors: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    Apply one of the 8 D4 dihedral-group symmetries identically to all inputs.

    The 8 elements of D4 are parameterised as (k_rot90 ∈ {0,1,2,3}, flip: bool):
      k=0, no flip  → identity
      k=1, no flip  → 90°  CCW
      k=2, no flip  → 180°
      k=3, no flip  → 270°
      k=0, flip     → horizontal flip
      k=1, flip     → flip then 90°
      k=2, flip     → flip then 180° (= vertical flip)
      k=3, flip     → flip then 270°

    All input tensors must share the same spatial dimensions [..., H, W].
    The same (k, flip) is sampled once and applied identically, preserving
    the paired correspondence between low and high images.

    Returns a tuple of transformed tensors in the same order.
    """
    k    = random.randint(0, 3)
    flip = random.random() > 0.5

    def _transform(t: torch.Tensor) -> torch.Tensor:
        if flip:
            t = torch.flip(t, dims=[-1])           # horizontal flip
        if k > 0:
            t = torch.rot90(t, k, dims=[-2, -1])   # CCW rotation
        return t

    return tuple(_transform(t) for t in tensors)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Pair matching utility
# ═════════════════════════════════════════════════════════════════════════════

def _match_pairs(
    high_dir: Path,
    low_dir:  Path,
) -> List[Tuple[Path, Path]]:
    """
    Match TIFF files in high_dir and low_dir by filename **stem**.

    Files present in one directory but missing from the other trigger a
    printed warning and are excluded from the returned list.

    Returns
    -------
    List of (high_path, low_path) tuples, sorted by stem.
    """
    _EXTS = {".tif", ".tiff"}

    def _index(d: Path) -> Dict[str, Path]:
        return {p.stem: p for p in sorted(d.iterdir())
                if p.suffix.lower() in _EXTS}

    hi = _index(high_dir)
    lo = _index(low_dir)
    common      = sorted(set(hi) & set(lo))
    only_high   = sorted(set(hi) - set(lo))
    only_low    = sorted(set(lo) - set(hi))

    if only_high:
        print(f"[dataset] WARNING — {len(only_high)} file(s) in high/ with no "
              f"matching low/: {only_high[:3]}{'…' if len(only_high)>3 else ''}")
    if only_low:
        print(f"[dataset] WARNING — {len(only_low)} file(s) in low/ with no "
              f"matching high/: {only_low[:3]}{'…' if len(only_low)>3 else ''}")

    return [(hi[s], lo[s]) for s in common]


# ═════════════════════════════════════════════════════════════════════════════
# 6.  NucleiDenoisingDataset  — primary dataset class
# ═════════════════════════════════════════════════════════════════════════════

class NucleiDenoisingDataset(Dataset):
    """
    PyTorch Dataset for the Noisy Nuclei paired fluorescence microscopy dataset.

    Parameters
    ----------
    root : str | Path
        Path to the ``Denoising_Dataset`` root directory containing
        ``Training/`` and ``Test/`` sub-folders.
    split : 'train' | 'test'
        Which sub-directory to load.
    task : 'baseline' | 'spatial_2d'
        Controls the returned sample format (see "Returned items" below).
    mode : 'train' | 'test'
        'train' — random patch crops + dihedral augmentation.
        'test'  — returns full 1024×1024 image tensors without augmentation.
    patch_size : int
        Side length of randomly cropped square patches in train mode.
        Default 256 works well for Swin-UNet on a 24 GB GPU (batch=8).
    patches_per_image : int
        Number of independent random crops drawn per image per epoch.
        Effective epoch size = n_pairs × patches_per_image.
        Recommended: 50 (gives 1 000 samples from 20 training pairs).
    norm_mode : 'minmax' | 'zscore' | 'fixed'
        Per-pair normalisation strategy (see normalise_pair docstring).
    sparsity_factor : int
        [spatial_2d] Mask downsample factor K.  Fraction scanned = 1/K².
    scan_pattern : str
        [spatial_2d] One of 'grid' | 'random' | 'stripe' | 'interleaved'.
    sparse_source : 'low' | 'high'
        [spatial_2d] Which image receives the sparse mask.
        'low'  → realistic: apply to already-noisy LQ image.
        'high' → ablation:  apply to clean GT (physics-informed variant).
    use_mask_channel : bool
        If True (default), the binary mask is appended as a 2nd input channel.
        Set to False with --no_mask_ch for ablation studies.
    apply_psf : bool
        If True, a 2-D Gaussian PSF blur is applied to the GT target before
        it is returned.  Corresponds to NOT passing --no_psf.
    psf_sigma_lateral : float
        PSF lateral sigma in pixels (only used when apply_psf=True).

    Returned items
    --------------
    task='baseline', mode='train':
        inp    : [1, patch_size, patch_size]   noisy low image patch ∈ [-1,1]
        target : [1, patch_size, patch_size]   clean high image patch ∈ [-1,1]

    task='baseline', mode='test':
        inp    : [1, 1024, 1024]
        target : [1, 1024, 1024]

    task='spatial_2d', mode='train', use_mask_channel=True:
        inp    : [2, patch_size, patch_size]   (warm-start, binary mask)
        target : [1, patch_size, patch_size]

    task='spatial_2d', mode='train', use_mask_channel=False:
        inp    : [1, patch_size, patch_size]   warm-start only
        target : [1, patch_size, patch_size]

    Note on model-range normalisation
    ----------------------------------
    Images are scaled to [-1, 1] via  x_model = x_01 * 2 - 1.
    The binary mask channel always remains in {0, 1}.
    To recover [0,1] values from a model output:  x_01 = (x_model + 1) / 2.
    """

    def __init__(
        self,
        root:              str | Path,
        split:             Literal["train", "test"] = "train",
        task:              Literal["baseline", "spatial_2d"] = "baseline",
        mode:              Literal["train", "test"] = "train",
        patch_size:        int       = 256,
        patches_per_image: int       = 50,
        norm_mode:         NormMode  = "minmax",
        sparsity_factor:   int       = 4,
        scan_pattern:      str       = "grid",
        sparse_source:     Literal["low", "high"] = "low",
        use_mask_channel:  bool      = True,
        apply_psf:         bool      = False,
        psf_sigma_lateral: float     = 1.0,
        split_dir_override: Optional[str | Path] = None,
        stack_policy: str = "error",
        stack_frame_index: int = 0,
    ):
        super().__init__()

        root = Path(root)
        if split_dir_override is not None:
            split_dir = Path(split_dir_override)
        else:
            split_dir = root / ("Training" if split == "train" else "Test")
        high_dir  = split_dir / "high"
        low_dir   = split_dir / "low"

        if not high_dir.is_dir():
            raise FileNotFoundError(
                f"GT directory not found: {high_dir}\n"
                f"Check that --data_root points to the Denoising_Dataset folder."
            )
        if not low_dir.is_dir():
            raise FileNotFoundError(f"LQ directory not found: {low_dir}")

        self.pairs = _match_pairs(high_dir, low_dir)
        if len(self.pairs) == 0:
            raise ValueError(
                f"No matched TIFF pairs found under {split_dir}.\n"
                f"Ensure high/ and low/ contain identically named .tif files."
            )

        self.task              = task
        self.mode              = mode
        self.patch_size        = patch_size
        self.patches_per_image = patches_per_image if mode == "train" else 1
        self.norm_mode         = norm_mode
        self.sparsity_factor   = sparsity_factor
        self.scan_pattern      = scan_pattern
        self.sparse_source     = sparse_source
        self.use_mask_channel  = use_mask_channel
        self.apply_psf         = apply_psf
        self.psf_sigma_lateral = psf_sigma_lateral
        self.stack_policy      = stack_policy
        self.stack_frame_index = stack_frame_index

        # Virtual index:  (image_pair_index, crop_repetition_index)
        # Expands 20 pairs × 50 crops → 1 000 unique __getitem__ calls per epoch.
        # Each call re-samples a fresh random crop (not pre-computed).
        self._index: List[Tuple[int, int]] = [
            (img_i, crop_i)
            for img_i  in range(len(self.pairs))
            for crop_i in range(self.patches_per_image)
        ]

    def __len__(self) -> int:
        return len(self._index)

    # ── Core __getitem__ ─────────────────────────────────────────────────────

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (inp, target) tensors for one sample.

        Full pipeline
        -------------
        1. Load both 16-bit TIFFs → float32 [0,1]
        2. Per-pair scientific normalisation
        3a. [train] Random 256×256 crop (same coordinates for both images)
        3b. [test]  No crop — return full 1024×1024
        4. [train only] Dihedral D4 augmentation (applied identically to pair)
        5. Optional PSF blur on GT target
        6. Scale from [0,1] to [-1,1]  (model convention)
        7. [spatial_2d] Generate sparse mask, warm-start inpaint, build input
        """
        img_i, _ = self._index[idx]
        high_path, low_path = self.pairs[img_i]

        # ── Step 1: Load 16-bit TIFFs ───────────────────────────────────────
        high_np = load_tiff_float32(
            high_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
        )   # [H, W]  float32 [0,1]
        low_np  = load_tiff_float32(
            low_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
        )    # [H, W]  float32 [0,1]

        # ── Step 2: Scientific normalisation ────────────────────────────────
        low_np, high_np = normalise_pair(low_np, high_np, mode=self.norm_mode)

        # ── Step 3: Crop / full image ────────────────────────────────────────
        if self.mode == "train":
            low_np, high_np = self._random_crop_np(low_np, high_np)
        # test mode: use full 1024×1024

        # ── Convert to tensors [1, H, W] ─────────────────────────────────────
        low_t  = torch.from_numpy(low_np.copy() ).unsqueeze(0)
        high_t = torch.from_numpy(high_np.copy()).unsqueeze(0)

        # ── Step 4: Dihedral augmentation (train only) ───────────────────────
        if self.mode == "train":
            low_t, high_t = dihedral_augment(low_t, high_t)

        # ── Step 5: Optional PSF blur on GT ─────────────────────────────────
        if self.apply_psf:
            high_t = apply_psf_blur_2d(high_t, self.psf_sigma_lateral)

        # ── Step 6: Scale [0,1] → [-1,1] ────────────────────────────────────
        low_m  = low_t  * 2.0 - 1.0
        high_m = high_t * 2.0 - 1.0

        # ── Step 7: Task-specific output ─────────────────────────────────────
        if self.task == "baseline":
            return low_m, high_m

        # task == "spatial_2d"
        return self._build_spatial2d(low_t, high_t, high_m)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _random_crop_np(
        self,
        low:  np.ndarray,   # [H, W]
        high: np.ndarray,   # [H, W]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random top-left corner (y0, x0) and crop both arrays to
        [patch_size, patch_size].

        A fresh random position is drawn on every __getitem__ call, so
        repeated passes over the same image_pair index yield distinct patches.
        """
        H, W = high.shape
        ps   = self.patch_size
        y0   = random.randint(0, max(0, H - ps))
        x0   = random.randint(0, max(0, W - ps))
        return (
            low [y0:y0+ps, x0:x0+ps],
            high[y0:y0+ps, x0:x0+ps],
        )

    def _build_spatial2d(
        self,
        low_t:  torch.Tensor,   # [1, H, W]  unnormalised [0,1]
        high_t: torch.Tensor,   # [1, H, W]  unnormalised [0,1]
        high_m: torch.Tensor,   # [1, H, W]  model-normalised [-1,1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the (inp, target) pair for the spatial_2d task.

        Pipeline
        --------
        1. Choose masking source: opts.sparse_source in {'low', 'high'}
        2. Generate on-the-fly sparse mask  [1, H, W]
        3. Apply mask to source → zero out unscanned pixels
        4. Bilinear warm-start: down ↓ K then up ↑ to H×W
        5. Optionally append mask as 2nd input channel
        6. Normalise inpainted image to [-1,1]

        Returns
        -------
        inp    : [2, H, W] if use_mask_channel else [1, H, W]
        target : [1, H, W]  (model-normalised clean GT)
        """
        _, H, W = high_t.shape

        # Step 1: masking source
        source = low_t if self.sparse_source == "low" else high_t   # [1, H, W]

        # Step 2: sparse mask
        mask = make_mask_2d(H, W, self.sparsity_factor, self.scan_pattern)
        # mask: [1, H, W]  ∈ {0, 1}

        # Step 3 + 4: mask → warm-start bilinear inpaint
        interp = warm_start_2d(source * mask, self.sparsity_factor)  # [1, H, W]

        # Step 5: normalise inpainted image
        interp_m = interp * 2.0 - 1.0                                # [1, H, W]

        # Step 6: assemble input channels
        if self.use_mask_channel:
            inp = torch.cat([interp_m, mask], dim=0)                 # [2, H, W]
        else:
            inp = interp_m                                           # [1, H, W]

        return inp, high_m


# ═════════════════════════════════════════════════════════════════════════════
# 6b. NucleiMixedDataset  — AI4Life-MDC25 large + small (mixed training)
# ═════════════════════════════════════════════════════════════════════════════

class NucleiMixedDataset(Dataset):
    """
    Mixed training set: AI4Life-MDC25 "2D denoising (Nuclei)" layout.

    - large_images/ (gt/ + noisy/): 1024×1024 pairs, random patch_size crop per epoch.
    - small_images/ (gt/ + noisy/): 256×256 pairs, used as-is (no crop).

    Intended for training only. Validation/test should use Denoising_Dataset
    (NucleiDenoisingDataset or NucleiGridDataset with opts.data_root).

    Supports task='baseline' only (same (inp, target) as NucleiDenoisingDataset).
    """

    def __init__(
        self,
        root:                   str | Path,
        task:                   Literal["baseline", "spatial_2d"] = "baseline",
        mode:                   Literal["train"] = "train",
        patch_size:             int       = 256,
        patches_per_image_large: int       = 50,
        norm_mode:              NormMode  = "minmax",
        apply_psf:              bool      = False,
        psf_sigma_lateral:      float     = 1.0,
        stack_policy:           str       = "error",
        stack_frame_index:      int       = 0,
    ):
        super().__init__()
        root = Path(root)
        large_gt   = root / "large_images" / "gt"
        large_noisy = root / "large_images" / "noisy"
        small_gt   = root / "small_images" / "gt"
        small_noisy = root / "small_images" / "noisy"

        if not large_gt.is_dir() or not small_gt.is_dir():
            raise FileNotFoundError(
                f"AI4Life-MDC25 layout expected under {root}:\n"
                "  large_images/gt, large_images/noisy\n"
                "  small_images/gt, small_images/noisy"
            )

        if task != "baseline":
            raise ValueError(
                "NucleiMixedDataset supports task='baseline' only. "
                "Use Denoising_Dataset (data_root) for spatial_2d."
            )
        self.large_pairs = _match_pairs(large_gt, large_noisy)
        self.small_pairs = _match_pairs(small_gt, small_noisy)
        if len(self.large_pairs) == 0:
            raise ValueError(f"No large pairs in {root}")
        if len(self.small_pairs) == 0:
            raise ValueError(f"No small pairs in {root}")

        self.task                   = task
        self.mode                   = mode
        self.patch_size             = patch_size
        self.patches_per_image_large = patches_per_image_large
        self.norm_mode              = norm_mode
        self.apply_psf              = apply_psf
        self.psf_sigma_lateral      = psf_sigma_lateral
        self.stack_policy           = stack_policy
        self.stack_frame_index      = stack_frame_index

        n_large = len(self.large_pairs)
        n_small = len(self.small_pairs)
        self._n_large_crops = n_large * patches_per_image_large
        self._total = self._n_large_crops + n_small
        self.pairs = self.large_pairs + self.small_pairs  # for compatibility with logging

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < self._n_large_crops:
            pair_idx = idx // self.patches_per_image_large
            high_path, low_path = self.large_pairs[pair_idx]
            high_np = load_tiff_float32(
                high_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
            )
            low_np  = load_tiff_float32(
                low_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
            )
            low_np, high_np = normalise_pair(low_np, high_np, mode=self.norm_mode)
            low_np, high_np = self._random_crop(low_np, high_np)
        else:
            small_idx = idx - self._n_large_crops
            high_path, low_path = self.small_pairs[small_idx]
            high_np = load_tiff_float32(
                high_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
            )
            low_np  = load_tiff_float32(
                low_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
            )
            low_np, high_np = normalise_pair(low_np, high_np, mode=self.norm_mode)
            # small is already 256×256, no crop

        low_t  = torch.from_numpy(low_np.copy()).unsqueeze(0)
        high_t = torch.from_numpy(high_np.copy()).unsqueeze(0)
        low_t, high_t = dihedral_augment(low_t, high_t)
        if self.apply_psf:
            high_t = apply_psf_blur_2d(high_t, self.psf_sigma_lateral)
        low_m  = low_t * 2.0 - 1.0
        high_m = high_t * 2.0 - 1.0
        return low_m, high_m

    def _random_crop(
        self, low: np.ndarray, high: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = high.shape
        ps   = self.patch_size
        y0   = random.randint(0, max(0, H - ps))
        x0   = random.randint(0, max(0, W - ps))
        return (
            low [y0:y0+ps, x0:x0+ps],
            high[y0:y0+ps, x0:x0+ps],
        )


# ═════════════════════════════════════════════════════════════════════════════
# 7.  NucleiGridDataset  — deterministic test-time patch grid
# ═════════════════════════════════════════════════════════════════════════════

class NucleiGridDataset(Dataset):
    """
    Splits every full 1024×1024 test image into a deterministic non-overlapping
    grid of ``patch_size × patch_size`` patches for memory-efficient evaluation.

    Use this when a full 1024×1024 forward pass exceeds GPU memory.

    After running inference, reassemble the full prediction:

    .. code-block:: python

        from collections import defaultdict
        accum = defaultdict(lambda: torch.zeros(1, H_full, W_full))
        for inp, target, meta in test_loader:
            pred = model(inp.to(device)).cpu()   # [1, 1, ps, ps]
            for b in range(pred.shape[0]):
                stem, y0, x0 = meta['stem'][b], meta['y0'][b], meta['x0'][b]
                ps = patch_size
                accum[stem][:, y0:y0+ps, x0:x0+ps] = pred[b]

    Returned items
    --------------
    inp    : [C_in, patch_size, patch_size]
    target : [1,    patch_size, patch_size]
    meta   : dict with keys 'img_idx', 'y0', 'x0', 'H', 'W', 'stem'
             (y0, x0 are positions in the *padded* image; H, W are the
             original un-padded dimensions for final cropping)
    """

    def __init__(
        self,
        root:              str | Path,
        task:              Literal["baseline", "spatial_2d"] = "baseline",
        patch_size:        int       = 256,
        norm_mode:         NormMode  = "minmax",
        sparsity_factor:   int       = 4,
        scan_pattern:      str       = "grid",
        sparse_source:     Literal["low", "high"] = "low",
        use_mask_channel:  bool      = True,
        apply_psf:         bool      = False,
        psf_sigma_lateral: float     = 1.0,
        stack_policy: str = "error",
        stack_frame_index: int = 0,
    ):
        super().__init__()
        root     = Path(root)
        test_dir = root / "Test"
        self.pairs = _match_pairs(test_dir / "high", test_dir / "low")
        assert self.pairs, f"No paired TIFFs found under {test_dir}"

        self.task              = task
        self.patch_size        = patch_size
        self.norm_mode         = norm_mode
        self.sparsity_factor   = sparsity_factor
        self.scan_pattern      = scan_pattern
        self.sparse_source     = sparse_source
        self.use_mask_channel  = use_mask_channel
        self.apply_psf         = apply_psf
        self.psf_sigma_lateral = psf_sigma_lateral
        self.stack_policy      = stack_policy
        self.stack_frame_index = stack_frame_index

        self._items = self._build_grid_index()

    def _build_grid_index(self) -> List[Dict[str, Any]]:
        """
        Pre-compute all (img_idx, y0, x0) patch positions.

        Images are reflect-padded to the nearest multiple of patch_size
        before tiling so no pixels are missed.
        """
        items: List[Dict[str, Any]] = []
        ps = self.patch_size
        for img_i, (high_p, _) in enumerate(self.pairs):
            raw = tifffile.imread(str(high_p))
            H = raw.shape[-2] if raw.ndim > 2 else raw.shape[0]
            W = raw.shape[-1] if raw.ndim > 2 else raw.shape[1]
            H_pad = math.ceil(H / ps) * ps
            W_pad = math.ceil(W / ps) * ps
            for y0 in range(0, H_pad, ps):
                for x0 in range(0, W_pad, ps):
                    items.append({
                        "img_idx": img_i,
                        "y0": y0, "x0": x0,
                        "H": H, "W": W,
                        "H_pad": H_pad, "W_pad": W_pad,
                        "stem": high_p.stem,
                    })
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta = self._items[idx]
        high_path, low_path = self.pairs[meta["img_idx"]]
        ps = self.patch_size

        high_np = load_tiff_float32(
            high_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
        )
        low_np  = load_tiff_float32(
            low_path, stack_policy=self.stack_policy, frame_index=self.stack_frame_index
        )
        low_np, high_np = normalise_pair(low_np, high_np, mode=self.norm_mode)

        # Reflect-pad to (H_pad, W_pad)
        H, W     = meta["H"], meta["W"]
        H_pad    = meta["H_pad"]
        W_pad    = meta["W_pad"]
        pad_h    = H_pad - H
        pad_w    = W_pad - W
        high_np  = np.pad(high_np, ((0, pad_h), (0, pad_w)), mode="reflect")
        low_np   = np.pad(low_np,  ((0, pad_h), (0, pad_w)), mode="reflect")

        y0, x0  = meta["y0"], meta["x0"]
        high_p  = high_np[y0:y0+ps, x0:x0+ps]
        low_p   = low_np [y0:y0+ps, x0:x0+ps]

        high_t  = torch.from_numpy(high_p.copy()).unsqueeze(0)
        low_t   = torch.from_numpy(low_p.copy() ).unsqueeze(0)

        if self.apply_psf:
            high_t = apply_psf_blur_2d(high_t, self.psf_sigma_lateral)

        # Scale to model range [-1,1]
        high_m  = high_t * 2.0 - 1.0
        low_m   = low_t  * 2.0 - 1.0

        if self.task == "baseline":
            return low_m, high_m, meta

        # spatial_2d
        mask    = make_mask_2d(ps, ps, self.sparsity_factor, self.scan_pattern)
        source  = low_t if self.sparse_source == "low" else high_t
        interp  = warm_start_2d(source * mask, self.sparsity_factor)
        interp_m = interp * 2.0 - 1.0
        inp      = torch.cat([interp_m, mask], dim=0) \
                   if self.use_mask_channel else interp_m
        return inp, high_m, meta


# ═════════════════════════════════════════════════════════════════════════════
# 7b. NucleiTemporal2p5DDataset  — 2.5D multi-frame to center-frame
# ═════════════════════════════════════════════════════════════════════════════

class NucleiTemporal2p5DDataset(Dataset):
    """
    2.5D dataset for temporal context learning on paired frame folders.

    Input:  neighboring low frames stacked as channels [T, H, W]
    Target: center high frame [1, H, W]
    """

    _PAT = re.compile(r"^(?P<prefix>.+)_f(?P<idx>\d+)$")

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"] = "train",
        mode: Literal["train", "test"] = "train",
        patch_size: int = 256,
        patches_per_image: int = 50,
        norm_mode: NormMode = "minmax",
        temporal_window: int = 3,
        split_dir_override: Optional[str | Path] = None,
        stack_policy: str = "error",
        stack_frame_index: int = 0,
    ):
        super().__init__()
        if temporal_window < 3 or temporal_window % 2 == 0:
            raise ValueError("temporal_window must be an odd number >= 3.")
        self.temporal_window = temporal_window
        self.radius = temporal_window // 2
        self.mode = mode
        self.patch_size = patch_size
        self.norm_mode = norm_mode
        self.patches_per_image = patches_per_image if mode == "train" else 1
        self.stack_policy = stack_policy
        self.stack_frame_index = stack_frame_index

        root = Path(root)
        split_dir = Path(split_dir_override) if split_dir_override is not None else root / ("Training" if split == "train" else "Test")
        high_dir = split_dir / "high"
        low_dir = split_dir / "low"
        if not high_dir.is_dir() or not low_dir.is_dir():
            raise FileNotFoundError(f"{split_dir} must contain high/ and low/")

        high_map = _index = {p.stem: p for p in sorted(high_dir.iterdir()) if p.suffix.lower() in {".tif", ".tiff"}}
        low_map = {p.stem: p for p in sorted(low_dir.iterdir()) if p.suffix.lower() in {".tif", ".tiff"}}
        common = sorted(set(high_map) & set(low_map))
        if not common:
            raise ValueError(f"No matched stems found in {split_dir}")

        seq: Dict[str, Dict[int, str]] = {}
        for stem in common:
            m = self._PAT.match(stem)
            if not m:
                continue
            prefix = m.group("prefix")
            idx = int(m.group("idx"))
            seq.setdefault(prefix, {})[idx] = stem

        self.centers: List[str] = []
        for _, idx_to_stem in seq.items():
            idxs = sorted(idx_to_stem.keys())
            for c in idxs:
                ok = True
                for d in range(-self.radius, self.radius + 1):
                    if (c + d) not in idx_to_stem:
                        ok = False
                        break
                if ok:
                    self.centers.append(idx_to_stem[c])
        if not self.centers:
            raise ValueError("No valid temporal centers found for requested temporal_window.")

        self.high_map = high_map
        self.low_map = low_map
        self.seq = seq
        # Compatibility attribute used by build_dataloaders split logic.
        self.pairs = self.centers
        self._index = [
            (i, j) for i in range(len(self.centers)) for j in range(self.patches_per_image)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def _neighbor_stems(self, center_stem: str) -> List[str]:
        m = self._PAT.match(center_stem)
        assert m is not None
        prefix = m.group("prefix")
        c = int(m.group("idx"))
        idx_to_stem = self.seq[prefix]
        return [idx_to_stem[c + d] for d in range(-self.radius, self.radius + 1)]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        center_i, _ = self._index[idx]
        center_stem = self.centers[center_i]
        neigh = self._neighbor_stems(center_stem)

        low_frames = []
        for s in neigh:
            low_np = load_tiff_float32(
                self.low_map[s],
                stack_policy=self.stack_policy,
                frame_index=self.stack_frame_index,
            )
            low_frames.append(low_np.astype(np.float32))
        low_stack = np.stack(low_frames, axis=0)  # [T,H,W]

        high_np = load_tiff_float32(
            self.high_map[center_stem],
            stack_policy=self.stack_policy,
            frame_index=self.stack_frame_index,
        ).astype(np.float32)

        # Consistent normalization using center low + center high; then apply to all low frames.
        center_low = low_stack[self.radius]
        center_low_n, high_n = normalise_pair(center_low, high_np, mode=self.norm_mode)
        if self.norm_mode == "minmax":
            g_min = min(float(center_low.min()), float(high_np.min()))
            g_max = max(float(center_low.max()), float(high_np.max()))
            denom = g_max - g_min + 1e-8
            low_stack_n = ((low_stack - g_min) / denom).clip(0.0, 1.0).astype(np.float32)
        elif self.norm_mode == "zscore":
            mu = float(high_np.mean())
            std = float(high_np.std()) + 1e-8
            low_stack_n = ((low_stack - mu) / std).astype(np.float32)
        else:
            low_stack_n = low_stack.astype(np.float32)
        # Keep variable referenced to avoid accidental drift between pathways.
        _ = center_low_n

        if self.mode == "train":
            H, W = high_n.shape
            ps = self.patch_size
            y0 = random.randint(0, max(0, H - ps))
            x0 = random.randint(0, max(0, W - ps))
            low_stack_n = low_stack_n[:, y0:y0 + ps, x0:x0 + ps]
            high_n = high_n[y0:y0 + ps, x0:x0 + ps]

        low_t = torch.from_numpy(low_stack_n.copy())       # [T,H,W]
        high_t = torch.from_numpy(high_n.copy()).unsqueeze(0)  # [1,H,W]

        if self.mode == "train":
            # Apply identical D4 transform to all low-frame channels and target.
            cat = torch.cat([low_t, high_t], dim=0)
            cat_aug, = dihedral_augment(cat)
            low_t = cat_aug[:-1]
            high_t = cat_aug[-1:].contiguous()

        low_m = low_t * 2.0 - 1.0
        high_m = high_t * 2.0 - 1.0
        return low_m, high_m


# ═════════════════════════════════════════════════════════════════════════════
# 8.  DataLoader factory
# ═════════════════════════════════════════════════════════════════════════════

def build_dataloaders(opts) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and (optionally) validation DataLoaders.

    ① If opts.val_root is set: use that directory (high/ + low/) as independent validation set.
    ② Else if opts.split_val_from_train > 0: hold out that fraction of training pairs as validation.
    ③ Else: no validation; val_loader is None (all data for training, evaluate on test set after).

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader or None
    """
    use_mask_ch  = not getattr(opts, "no_mask_ch",  False)
    apply_psf    = getattr(opts, "apply_psf",    False)
    test_as_grid = getattr(opts, "test_as_grid", False)

    train_data_root = getattr(opts, "train_data_root", None)
    val_root        = getattr(opts, "val_root",        None)
    split_val       = getattr(opts, "split_val_from_train", 0.0)
    seed            = getattr(opts, "seed", 42)

    shared = dict(
        root              = opts.data_root,
        task              = opts.task,
        patch_size        = opts.patch_size,
        norm_mode         = getattr(opts, "norm_mode",   "minmax"),
        sparsity_factor   = opts.sparsity_factor,
        scan_pattern      = opts.scan_pattern,
        sparse_source     = getattr(opts, "sparse_source", "low"),
        use_mask_channel  = use_mask_ch,
        apply_psf         = apply_psf,
        psf_sigma_lateral = opts.psf_sigma_lateral,
        stack_policy      = getattr(opts, "stack_policy", "error"),
        stack_frame_index = getattr(opts, "stack_frame_index", 0),
    )

    if opts.task == "spatial_2p5d":
        train_ds = NucleiTemporal2p5DDataset(
            root=opts.data_root,
            split="train",
            mode="train",
            patch_size=opts.patch_size,
            patches_per_image=getattr(opts, "patches_per_image", 50),
            norm_mode=getattr(opts, "norm_mode", "minmax"),
            temporal_window=getattr(opts, "temporal_window", 3),
            stack_policy=getattr(opts, "stack_policy", "error"),
            stack_frame_index=getattr(opts, "stack_frame_index", 0),
        )
        train_info = f"{len(train_ds):>5,} samples  (2.5D window={getattr(opts, 'temporal_window', 3)})"
    elif train_data_root:
        train_ds = NucleiMixedDataset(
            root                    = train_data_root,
            task                    = opts.task,
            mode                    = "train",
            patch_size              = opts.patch_size,
            patches_per_image_large = getattr(opts, "patches_per_image", 50),
            norm_mode               = getattr(opts, "norm_mode", "minmax"),
            apply_psf               = apply_psf,
            psf_sigma_lateral       = opts.psf_sigma_lateral,
            stack_policy            = getattr(opts, "stack_policy", "error"),
            stack_frame_index       = getattr(opts, "stack_frame_index", 0),
        )
        n_large = len(train_ds.large_pairs)
        n_small = len(train_ds.small_pairs)
        n_crops = train_ds.patches_per_image_large
        train_info = (
            f"{len(train_ds):>5,} samples  "
            f"(large {n_large}×{n_crops} + small {n_small} pairs)"
        )
    else:
        train_ds = NucleiDenoisingDataset(
            **shared,
            split             = "train",
            mode              = "train",
            patches_per_image = getattr(opts, "patches_per_image", 50),
        )
        n_pairs = len(train_ds.pairs)
        n_crops = train_ds.patches_per_image
        train_info = f"{len(train_ds):>5,} samples  ({n_pairs} pairs × {n_crops} crops/pair)"

    val_loader: Optional[DataLoader] = None

    # ① Independent validation set
    if val_root is not None and Path(val_root).is_dir():
        high_dir = Path(val_root) / "high"
        low_dir  = Path(val_root) / "low"
        if high_dir.is_dir() and low_dir.is_dir():
            if opts.task == "spatial_2p5d":
                val_ds = NucleiTemporal2p5DDataset(
                    root=opts.data_root,
                    split="test",
                    mode="test",
                    patch_size=opts.patch_size,
                    patches_per_image=1,
                    norm_mode=getattr(opts, "norm_mode", "minmax"),
                    temporal_window=getattr(opts, "temporal_window", 3),
                    split_dir_override=Path(val_root),
                    stack_policy=getattr(opts, "stack_policy", "error"),
                    stack_frame_index=getattr(opts, "stack_frame_index", 0),
                )
            else:
                val_ds = NucleiDenoisingDataset(
                    **shared,
                    split              = "test",
                    mode               = "test",
                    patches_per_image  = 1,
                    split_dir_override = Path(val_root),
                )
            val_loader = DataLoader(
                val_ds,
                batch_size        = 1,
                shuffle           = False,
                num_workers       = opts.num_workers,
                pin_memory        = True,
                persistent_workers= opts.num_workers > 0,
            )
            print(f"[dataset] Train : {train_info}")
            print(f"[dataset] Val   : {len(val_ds):>5,} samples (independent: {val_root})")
        else:
            raise FileNotFoundError(f"--val_root {val_root} must contain high/ and low/.")

    # ② Split part of training data as validation
    elif split_val > 0:
        if train_data_root:
            n_pairs = len(train_ds.large_pairs) + len(train_ds.small_pairs)
            def _pair_id(i):
                if i < train_ds._n_large_crops:
                    return i // train_ds.patches_per_image_large
                return len(train_ds.large_pairs) + (i - train_ds._n_large_crops)
        else:
            n_pairs = len(train_ds.pairs)
            def _pair_id(i):
                return train_ds._index[i][0]

        rng = random.Random(seed)
        order = list(range(n_pairs))
        rng.shuffle(order)
        n_val = max(1, int(n_pairs * split_val))
        val_pair_ids = set(order[:n_val])
        train_idx = [i for i in range(len(train_ds)) if _pair_id(i) not in val_pair_ids]
        val_idx   = [i for i in range(len(train_ds)) if _pair_id(i) in val_pair_ids]

        val_loader = DataLoader(
            Subset(train_ds, val_idx),
            batch_size        = 1,
            shuffle           = False,
            num_workers       = opts.num_workers,
            pin_memory        = True,
            persistent_workers= opts.num_workers > 0,
        )
        train_loader = DataLoader(
            Subset(train_ds, train_idx),
            batch_size        = opts.batch_size,
            shuffle           = True,
            num_workers       = opts.num_workers,
            pin_memory        = True,
            drop_last         = True,
            persistent_workers= opts.num_workers > 0,
        )
        print(f"[dataset] Train : {train_info}  (holdout val {split_val:.2f})")
        print(f"[dataset] Val   : {len(val_idx):>5,} samples (from training pairs)")
        return train_loader, val_loader

    # ③ No validation: all data for training
    else:
        print(f"[dataset] Train : {train_info}")
        print(f"[dataset] Val   : none (no validation; evaluate on test set after training)")

    if val_loader is None:
        train_loader = DataLoader(
            train_ds,
            batch_size        = opts.batch_size,
            shuffle           = True,
            num_workers       = opts.num_workers,
            pin_memory        = True,
            drop_last         = True,
            persistent_workers= opts.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size        = opts.batch_size,
            shuffle           = True,
            num_workers       = opts.num_workers,
            pin_memory        = True,
            drop_last         = True,
            persistent_workers= opts.num_workers > 0,
        )

    return train_loader, val_loader

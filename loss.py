"""
loss.py
=======
Hybrid loss function for 3PM reconstruction:

    L_total = λ₁ · L_MSE  +  λ₂ · L_MS-SSIM  +  λ₃ · L_DINOv2

Components
----------
L_MSE       : pixel-wise mean squared error — enforces pixel fidelity
L_MS-SSIM   : multi-scale structural similarity — preserves high-frequency
               cellular structures (dendrites, spines) at multiple scales
L_DINOv2    : feature-space perceptual loss using DINOv2 ViT-B/14
               intermediate patch-token features — aligns semantic and
               structural content (morphological fidelity) without relying
               on supervised ImageNet features

3-D handling
------------
For spatiotemporal_3d task, pred/target arrive as [B, 1, T, H, W].
We collapse T into B before passing to the 2-D losses:
    [B, 1, T, H, W] → [B*T, 1, H, W]
This keeps all loss implementations 2-D and avoids code duplication.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Compatibility for old torch versions (<1.9) that lack torch.inference_mode.
# DINOv2 hub code decorates methods with @torch.inference_mode().
if not hasattr(torch, "inference_mode"):
    torch.inference_mode = torch.no_grad


# ─────────────────────────────────────────────
# MS-SSIM
# ─────────────────────────────────────────────

def _gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian kernel, used to build a 2-D separable SSIM window."""
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return (g / g.sum()).unsqueeze(0)                      # [1, window_size]


def _ssim(
    pred:   torch.Tensor,          # [B, 1, H, W]
    target: torch.Tensor,          # [B, 1, H, W]
    window: torch.Tensor,          # [1, 1, ws, ws]
    C1:     float = 0.01 ** 2,
    C2:     float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Compute mean SSIM between pred and target.
    Window is applied as a depthwise 2-D convolution.
    """
    mu1 = F.conv2d(pred,   window, padding=window.shape[-1] // 2, groups=1)
    mu2 = F.conv2d(target, window, padding=window.shape[-1] // 2, groups=1)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   window, padding=window.shape[-1] // 2) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window.shape[-1] // 2) - mu2_sq
    sigma12   = F.conv2d(pred   * target, window, padding=window.shape[-1] // 2) - mu1_mu2

    ssim_map  = ((2 * mu1_mu2 + C1) * (2 * sigma12   + C2)) / \
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM loss.

    Images are progressively downsampled (avg-pool factor 2) and SSIM is
    computed at each scale.  Weights across scales are taken from the
    original MS-SSIM paper (Wang et al., 2003).

    L_msssim = 1 - MS-SSIM(pred, target)   ∈ [0, 2]
    """

    _WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 5-scale weights

    def __init__(self, window_size: int = 11, n_scales: int = 5):
        super().__init__()
        self.n_scales = n_scales
        ws  = window_size
        k1d = _gaussian_window(ws)                         # [1, ws]
        k2d = (k1d.T @ k1d).unsqueeze(0).unsqueeze(0)     # [1, 1, ws, ws]
        self.register_buffer("window", k2d)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : [B, 1, H, W]  (2-D) or [B*T, 1, H, W] (collapsed 3-D)
        """
        weights = self._WEIGHTS[: self.n_scales]
        levels_ssim = []

        p, t = pred, target
        for scale in range(self.n_scales):
            ssim_val = _ssim(p, t, self.window.to(p.device))
            levels_ssim.append(ssim_val)
            if scale < self.n_scales - 1:
                # Downsample by 2 for next scale
                p = F.avg_pool2d(p, kernel_size=2)
                t = F.avg_pool2d(t, kernel_size=2)

        # Weighted product (converted to loss = 1 - MS-SSIM)
        ms_ssim = 1.0
        for w, s in zip(weights, levels_ssim):
            # ms_ssim = ms_ssim * (s ** w)
            # 终极修复：强制截断负数，确保底数严格非负！
            s_safe = torch.clamp(s, min=0.0)
            ms_ssim = ms_ssim * (s_safe ** w)
        return 1.0 - ms_ssim


# ─────────────────────────────────────────────
# DINOv2 Perceptual Loss
# ─────────────────────────────────────────────

class DINOv2PerceptualLoss(nn.Module):
    """
    Extracts patch-token features from a frozen DINOv2 ViT-B/14 backbone and
    computes L2 perceptual distance between prediction and target.

    DINOv2 (Oquab et al., 2023) is preferred over supervised ViTs because
    its features are highly sensitive to fine-grained spatial structure —
    crucial for preserving neuronal morphology (dendrites, spines) in 3PM.

    Feature layers: block indices where intermediate tokens are collected.
    Typically {3, 6, 9, 11} for ViT-B (12 blocks total).

    Input handling
    --------------
    DINOv2 was trained on RGB 224×224 images.  Our images are greyscale and
    may be smaller or larger.  We:
      1. Expand 1-ch → 3-ch via weight-shared Conv2d
      2. Bilinear resize to 224×224
      3. Apply imagenet normalisation
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        feature_layers: List[int] = [3, 6, 9, 11],
        model_name:     str       = "dinov2_vitb14",
    ):
        super().__init__()
        self.feature_layers = set(feature_layers)

        # Load DINOv2 from torch.hub
        print(f"[DINOv2Loss] Loading {model_name} from facebookresearch/dinov2 …")
        try:
            dino = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True,
            )
        except Exception as e:
            print(f"[DINOv2Loss] Hub load failed ({e}). Using random-init fallback.")
            # Fallback: smaller ViT-S for development without internet
            dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=False,
            )

        # Freeze all DINOv2 parameters — perceptual loss backbone is fixed
        for p in dino.parameters():
            p.requires_grad_(False)

        # Store the patch embedding and transformer blocks
        self.patch_embed = dino.patch_embed
        self.blocks      = dino.blocks              # nn.ModuleList
        self.norm        = dino.norm

        # Register imagenet stats as buffers (moved to device automatically)
        self.register_buffer(
            "mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1)
        )

        # Greyscale → 3-channel adapter (mean-copy initialisation)
        self.to_rgb = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.to_rgb.weight, 1.0 / 3.0)

        # Freeze the adapter too (optional — unfreeze for fine-tuning)
        for p in self.to_rgb.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract patch-token features at specified block depths.

        x : [B, 1, H, W]  (values in [-1,1])
        returns : list of [B, N_patches, C] tensors, one per feature_layer
        """
        # 1. Greyscale → RGB → resize to 224 → imagenet-normalise
        x = self.to_rgb(x)                                             # [B, 3, H, W]
        x = F.interpolate(x, size=(224, 224), mode="bilinear",
                          align_corners=False)
        x = (x * 0.5 + 0.5)                                           # [-1,1] → [0,1]
        x = (x - self.mean) / self.std                                 # imagenet norm

        # 2. Patch embedding
        x = self.patch_embed(x)                                        # [B, N, C]

        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.feature_layers:
                feats.append(x)                                        # patch tokens

        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : [B, 1, H, W]

        Returns scalar perceptual loss (mean over layers and batch).
        """
        pred_feats   = self._extract(pred)
        target_feats = self._extract(target)
        loss = sum(F.mse_loss(pf, tf) for pf, tf in zip(pred_feats, target_feats))
        return loss / max(1, len(pred_feats))


# ─────────────────────────────────────────────
# Combined Reconstruction Loss
# ─────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """
    L_total = λ₁ · MSE  +  λ₂ · MS-SSIM  +  λ₃ · DINOv2

    Ablation flags (from opts):
        opts.no_dino    → λ₃ forced to 0
        opts.no_msssim  → λ₂ forced to 0

    Shape handling
    --------------
    Both 2-D [B, 1, H, W] and 3-D [B, 1, T, H, W] inputs are supported.
    3-D inputs are collapsed along T before each sub-loss:
        [B, 1, T, H, W] → [B*T, 1, H, W]
    """

    def __init__(self, opts):
        super().__init__()
        self.lambda_mse    = opts.lambda_mse
        self.lambda_msssim = 0.0 if opts.no_msssim else opts.lambda_msssim
        self.lambda_dino   = 0.0 if opts.no_dino   else opts.lambda_dino

        self.msssim_loss = MSSSIMLoss() if self.lambda_msssim > 0 else None
        self.dino_loss   = (
            DINOv2PerceptualLoss(feature_layers=opts.dino_layers)
            if self.lambda_dino > 0 else None
        )

    def _flatten_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        Collapse temporal dimension into batch if input is 5-D.
        [B, 1, T, H, W] → [B*T, 1, H, W]
        [B, 1, H, W]    → unchanged
        """
        if x.ndim == 5:
            B, C, T, H, W = x.shape
            return x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        return x

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns
        -------
        total_loss : scalar Tensor  (backward-able)
        metrics    : dict with per-component float values for logging
        """
        pred_2d   = self._flatten_time(pred)
        target_2d = self._flatten_time(target)

        # ── MSE ──────────────────────────────────────────────────────────────
        mse = F.mse_loss(pred_2d, target_2d)
        total = self.lambda_mse * mse

        # ── MS-SSIM ───────────────────────────────────────────────────────────
        # msssim_val = torch.tensor(0.0, device=pred.device)
        # if self.msssim_loss is not None:
        #     msssim_val = self.msssim_loss(pred_2d, target_2d)
        #     total = total + self.lambda_msssim * msssim_val

        # ── MS-SSIM ───────────────────────────────────────────────────────────
        msssim_val = torch.tensor(0.0, device=pred.device)
        if self.msssim_loss is not None:
            # 修复：将 [-1, 1] 映射回 [0, 1] 以防止负数底数导致 NaN
            p_01 = (pred_2d + 1.0) / 2.0
            t_01 = (target_2d + 1.0) / 2.0
            msssim_val = self.msssim_loss(p_01, t_01)
            total = total + self.lambda_msssim * msssim_val

        # ── DINOv2 Perceptual ─────────────────────────────────────────────────
        dino_val = torch.tensor(0.0, device=pred.device)
        if self.dino_loss is not None:
            # Move dino backbone to same device as input
            self.dino_loss.to(pred.device)
            dino_val = self.dino_loss(pred_2d, target_2d)
            total = total + self.lambda_dino * dino_val

        metrics = {
            "loss/total":      total.item(),
            "loss/mse":        mse.item(),
            "loss/ms_ssim":    msssim_val.item(),
            "loss/dino":       dino_val.item(),
        }
        return total, metrics

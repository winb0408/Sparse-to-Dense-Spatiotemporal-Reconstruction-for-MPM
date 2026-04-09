import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_image(img, vmin=None, vmax=None, pmin=0.1, pmax=99.9):
    """如果提供了 vmin 和 vmax，则作为绝对标尺；否则根据自身图像计算"""
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(img, (pmin, pmax))
        
    denom = vmax - vmin + 1e-8
    img_norm = np.clip((img - vmin) / denom, 0.0, 1.0)
    return img_norm, vmin, vmax

def main():
    parser = argparse.ArgumentParser(description="3PM Reconstruction Visualization")
    parser.add_argument("--data_root", type=str, 
                        default="/data2/wjb/threePM/ThreePM_openDatasets/Denoising_Dataset",
                        help="数据集根目录")
    parser.add_argument("--pred_dir", type=str, default="outputs", help="模型预测结果存放目录")
    parser.add_argument("--out_dir", type=str, default="visualizations", help="可视化输出目录")
    
    # 将归一化模式设为三个明确的选项
    parser.add_argument("--norm_mode", type=str, default="joint", 
                        choices=["independent", "gt", "joint"],
                        help="independent: 各自独立拉伸 (看结构) | "
                             "gt: 严格以 GT 为标尺 (看绝对强度差异) | "
                             "joint: 综合三图的所有像素计算统一公平标尺 (全局视界)")
    opts = parser.parse_args()

    data_root = Path(opts.data_root)
    low_dir   = data_root / "Test" / "low"
    high_dir  = data_root / "Test" / "high"
    pred_dir  = Path(opts.pred_dir)
    vis_out_dir = Path(opts.out_dir)
    vis_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"当前可视化归一化模式: {opts.norm_mode.upper()}\n")

    for pred_path in pred_dir.glob("*_pred.tif"):
        stem = pred_path.name.replace("_pred.tif", "")
        low_path  = low_dir / f"{stem}.tif"
        high_path = high_dir / f"{stem}.tif"

        if not (low_path.exists() and high_path.exists()):
            continue

        print(f"正在处理: {stem} ...")

        img_low  = tifffile.imread(str(low_path))
        img_high = tifffile.imread(str(high_path))
        img_pred = tifffile.imread(str(pred_path))

        # ── 核心：三种归一化策略 ────────────────────────────────────────────
        if opts.norm_mode == "independent":
            # 模式 1：各自独立拉伸（结构最清晰，忽略亮度差异）
            norm_low, _, _  = normalize_image(img_low)
            norm_pred, _, _ = normalize_image(img_pred)
            norm_high, _, _ = normalize_image(img_high)

        elif opts.norm_mode == "gt":
            # 模式 2：严格以 GT 为准（如果 pred 亮度低，就会显得极暗）
            norm_high, gt_vmin, gt_vmax = normalize_image(img_high)
            norm_low, _, _  = normalize_image(img_low, vmin=gt_vmin, vmax=gt_vmax)
            norm_pred, _, _ = normalize_image(img_pred, vmin=gt_vmin, vmax=gt_vmax)

        elif opts.norm_mode == "joint":
            # 模式 3：联合统一标尺（不偏袒任何一列）
            # 将三张图拉平成一维数组拼接，从全局寻找真实的极值分布
            concat_pixels = np.concatenate([img_low.flatten(), img_pred.flatten(), img_high.flatten()])
            joint_vmin, joint_vmax = np.percentile(concat_pixels, (0.1, 99.9))
            
            norm_low, _, _  = normalize_image(img_low, vmin=joint_vmin, vmax=joint_vmax)
            norm_pred, _, _ = normalize_image(img_pred, vmin=joint_vmin, vmax=joint_vmax)
            norm_high, _, _ = normalize_image(img_high, vmin=joint_vmin, vmax=joint_vmax)

        # ── 制作 RGB 伪彩图并绘图 ──────────────────────────────────────────
        rgb_overlay = np.zeros((*norm_high.shape, 3), dtype=np.float32)
        rgb_overlay[..., 0] = norm_pred  # Red
        rgb_overlay[..., 1] = norm_high  # Green
        rgb_overlay[..., 2] = norm_pred  # Blue

        fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
        plt.subplots_adjust(wspace=0.05)
        fig.suptitle(f"Sample: {stem} | Mode: {opts.norm_mode.upper()}", fontsize=20, color='white', y=0.98)

        axes[0].imshow(norm_low, cmap='gray')
        axes[0].set_title("Input (Noisy)", fontsize=16)
        axes[0].axis('off')

        axes[1].imshow(norm_pred, cmap='gray')
        axes[1].set_title("Prediction (Swin-UNet)", fontsize=16)
        axes[1].axis('off')

        axes[2].imshow(norm_high, cmap='gray')
        axes[2].set_title("Ground Truth", fontsize=16)
        axes[2].axis('off')

        axes[3].imshow(rgb_overlay)
        axes[3].set_title("Overlay (GT=Green, Pred=Magenta)", fontsize=16)
        axes[3].axis('off')

        save_path = vis_out_dir / f"compare_{opts.norm_mode}_{stem}.png"
        plt.savefig(save_path, bbox_inches='tight', facecolor='black')
        plt.close(fig)

if __name__ == "__main__":
    main()
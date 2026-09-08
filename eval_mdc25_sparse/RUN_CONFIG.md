# MDC25 K=2 评估与可视化配置

## 1) 统一 Validation 评估命令

```bash
PYTHONPATH="/data2/wjb/threePM" python "/data2/wjb/threePM/scripts/eval_mdc25_sparse_unified.py" \
  --sparse_root "/data2/wjb/threePM/ThreePM_openDatasets/MDC25_sparse" \
  --runs_dir "/data2/wjb/threePM/runs" \
  --output_dir "/data2/wjb/threePM/eval_mdc25_sparse" \
  --max_frames 64 \
  --patch_hw 256
```

评估输出：
- `eval_mdc25_sparse/summary_all.csv`
- `eval_mdc25_sparse/summary_all.md`

## 2) 代表帧可视化命令（GT/Sparse/best-2D/2.5D/3D）

```bash
PYTHONPATH="/data2/wjb/threePM" python "/data2/wjb/threePM/scripts/plot_mdc25_k2_best5.py" \
  --summary_csv "/data2/wjb/threePM/eval_mdc25_sparse/summary_all.csv" \
  --sparse_root "/data2/wjb/threePM/ThreePM_openDatasets/MDC25_sparse" \
  --runs_dir "/data2/wjb/threePM/runs" \
  --output_path "/data2/wjb/threePM/eval_mdc25_sparse/qualitative/mdc25_k2_best5_modes.png"
```

可视化输出：
- `eval_mdc25_sparse/qualitative/mdc25_k2_best5_modes.png`

## 3) 当前 best run（按 mode + 算法族）

- `xy`
  - 2D: `sparse3d_2d_xy_K2_wiener_mdc25`
  - 2.5D: `sparse3d_2p5d_xy_K2_mask_wiener_mdc25`
  - 3D: `sparse3d_3d_xy_K2_wiener_mdc25`
- `z`
  - 2D: `sparse3d_2d_z_K2_wiener_mdc25`
  - 2.5D: `sparse3d_2p5d_z_K2_mask_wiener_mdc25`
  - 3D: `sparse3d_3d_z_K2_wiener_mdc25`
- `xyz`
  - 2D: `sparse3d_2d_xyz_K2_wiener_mdc25`
  - 2.5D: `sparse3d_2p5d_xyz_K2_mask_wiener_mdc25`
  - 3D: `sparse3d_3d_xyz_K2_wiener_mdc25`

## 4) 说明

- 本次统一评估采用全图推理（2D/2.5D）和体块拼接（3D）。
- `n_frames` 在不同算法族不一致（2D=64, 2.5D=14, 3D=16）是由各评估函数可用帧构造规则决定，后续若需“严格同帧对齐比较”，建议固定同一帧索引集合再重评估。

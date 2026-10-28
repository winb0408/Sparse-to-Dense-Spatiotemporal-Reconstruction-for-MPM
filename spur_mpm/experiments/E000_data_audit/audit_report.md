# SPUR-E000 数据审计报告

- 数据集目录: `/ssd1/wjb/Data/Dryad_3PM/wetransfer_data-3p_2025-11-21_1143/Data 3P`
- 文件数: **29**（纳入 10，排除 19，有 issue 的 0 个）
- 动物 (mouse_id): ['1', '2', '3']
- FOV (mouse/fov): ['1/I', '1/II', '2/I', '2/II', '3/I']
- session: ['1', '2']
- Z 覆盖 (µm): [755.0, 785.0, 835.0, 935.0, 945.0, 1000.0, 1005.0, 1010.0]

## 形状与类型（仅纳入的文件）

| shape | 文件数 |
|---|---|
| [202, 512, 512] | 3 |
| [188, 512, 512] | 1 |
| [190, 512, 512] | 1 |
| [201, 512, 512] | 1 |
| [152, 512, 512] | 1 |
| [203, 512, 512] | 1 |
| [158, 512, 512] | 1 |
| [168, 512, 512] | 1 |

| dtype | 文件数 |
|---|---|
| uint16 | 10 |

## 纳入清单（可进入分组划分）

| sample_id | mouse | session | fov | axes | shape | z_step | z_span_um |
|---|---|---|---|---|---|---|---|
| M1_FOVI_S1_vessels | 1 | 1 | I | ZYX | [202, 512, 512] | 5.0 | 1005.0 |
| M1_FOVI_S2_vessels | 1 | 2 | I | ZYX | [188, 512, 512] | 5.0 | 935.0 |
| M1_FOVII_S2_vessels | 1 | 2 | II | ZYX | [190, 512, 512] | 5.0 | 945.0 |
| M1_FOVII_S1_vessels | 1 | 1 | II | ZYX | [202, 512, 512] | 5.0 | 1005.0 |
| M2_FOVI_S1_vessels | 2 | 1 | I | ZYX | [201, 512, 512] | 5.0 | 1000.0 |
| M2_FOVI_S2_vessels | 2 | 2 | I | ZYX | [152, 512, 512] | 5.0 | 755.0 |
| M2_FOVII_S1_vessels | 2 | 1 | II | ZYX | [203, 512, 512] | 5.0 | 1010.0 |
| M2_FOVII_S2_vessels | 2 | 2 | II | ZYX | [202, 512, 512] | 5.0 | 1005.0 |
| M3_FOVI_S1_vessels | 3 | 1 | I | ZYX | [158, 512, 512] | 5.0 | 785.0 |
| M3_FOVI_S2_vessels | 3 | 2 | I | ZYX | [168, 512, 512] | 5.0 | 835.0 |

## 排除清单（显式声明，不进入划分）

| sample_id | 原因 | shape | axes |
|---|---|---|---|
| 3a Plaques | figure panel derived from the dataset: reusing it risks testing on a training animal (see duplicate_of) | [203, 512, 512] | ZYX |
| 3a Vessels | figure panel derived from the dataset: reusing it risks testing on a training animal (see duplicate_of) | [203, 512, 512] | ZYX |
| 3b - GFP neurons | figure panel derived from the dataset: reusing it risks testing on a training animal (see duplicate_of) | [131, 2, 512, 512] | ZCYX |
| M1_FOVI_S1_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [202, 512, 512] | ZYX |
| M1_FOVI_S2_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [188, 512, 512] | ZYX |
| M1_FOVII_S1_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [202, 512, 512] | ZYX |
| M1_FOVII_S2_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [190, 512, 512] | ZYX |
| M2_FOVI_S1_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [201, 512, 512] | ZYX |
| M2_FOVI_S2_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [168, 512, 512] | ZYX |
| M2_FOVII_S1_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [203, 512, 512] | ZYX |
| M2_FOVII_S2_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [202, 512, 512] | ZYX |
| M3_FOVI_S1_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [158, 512, 512] | ZYX |
| M3_FOVI_S2_plaques | plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels | [168, 512, 512] | ZYX |
| neonatal_lung_sampleA1 | power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex | [25, 512, 512] | TYX |
| neonatal_lung_sampleA2 | power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex | [25, 512, 512] | TYX |
| neonatal_lung_sampleA3 | power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex | [25, 512, 512] | TYX |
| neonatal_lung_sampleB1 | power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex | [25, 512, 512] | TYX |
| neonatal_lung_sampleB2 | power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex | [25, 512, 512] | TYX |
| neonatal_lung_sampleB3 | power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex | [25, 512, 512] | TYX |

## 重复文件（像素级一致；这是泄漏源）

| sample_id | 与谁重复 | use | path |
|---|---|---|---|
| 3a Plaques | M2_FOVII_S1_plaques | excluded | Fig 3/3a Plaques.tif |
| 3a Vessels | M2_FOVII_S1_vessels | excluded | Fig 3/3a Vessels.tif |
| M3_FOVI_S2_plaques | M2_FOVI_S2_plaques | excluded | Mouse 3 FOV I/Mouse 3 FOV I Plaques Session 2.tif |

## 警告（不阻塞，但需人工确认）

- `3a Plaques`: excluded: figure panel derived from the dataset: reusing it risks testing on a training animal (see duplicate_of); issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved; mouse_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; session_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; fov_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; pixel-identical to M2_FOVII_S1_plaques (Mouse 2 FOV II/Mouse 2 FOV II Plaques Session 1.tif)
- `3a Vessels`: excluded: figure panel derived from the dataset: reusing it risks testing on a training animal (see duplicate_of); issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved; mouse_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; session_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; fov_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; pixel-identical to M2_FOVII_S1_vessels (Mouse 2 FOV II/Mouse 2 FOV II Vessels Session 1.tif)
- `3b - GFP neurons`: excluded: figure panel derived from the dataset: reusing it risks testing on a training animal (see duplicate_of); issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved; mouse_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; session_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; fov_id missing: grouped splits and longitudinal leakage checks cannot be guaranteed; extreme anisotropy (z_step 10.0 um vs xy 0.98 um): axial context must span enough micrometres for any depth comparison to be fair; ImageJ spacing 15.0 vs spec z_step 10.0 -- spec wins, reconcile manually
- `M1_FOVI_S1_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels
- `M1_FOVI_S2_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels; ImageJ spacing 7.0 vs spec z_step 5.0 -- spec wins, reconcile manually
- `M1_FOVI_S2_vessels`: ImageJ spacing 7.0 vs spec z_step 5.0 -- spec wins, reconcile manually
- `M1_FOVII_S1_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels
- `M1_FOVII_S2_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels; ImageJ spacing 7.0 vs spec z_step 5.0 -- spec wins, reconcile manually
- `M1_FOVII_S2_vessels`: ImageJ spacing 7.0 vs spec z_step 5.0 -- spec wins, reconcile manually
- `M2_FOVI_S1_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels
- `M2_FOVI_S2_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels; ImageJ spacing 7.0 vs spec z_step 5.0 -- spec wins, reconcile manually; duplicated by 1 other file(s): M3_FOVI_S2_plaques
- `M2_FOVII_S1_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels; duplicated by 1 other file(s): 3a Plaques
- `M2_FOVII_S2_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels
- `M2_FOVII_S1_vessels`: duplicated by 1 other file(s): 3a Vessels
- `M3_FOVI_S1_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels
- `M3_FOVI_S2_plaques`: excluded: plaque channel: same scan as vessels, out of reconstruction scope; would leak tissue identity across channels; ImageJ spacing 7.0 vs spec z_step 5.0 -- spec wins, reconcile manually; pixel-identical to M2_FOVI_S2_plaques (Mouse 2 FOV I/Mouse 2 FOV I Plaques Session 2.tif)
- `M3_FOVI_S2_vessels`: ImageJ spacing 7.0 vs spec z_step 5.0 -- spec wins, reconcile manually
- `neonatal_lung_sampleA1`: excluded: power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex; issue: neonatal_lung_sampleA1: axes='TYX' contains a time axis. 3-D structural data must be ZYX; TYX belongs to the calcium pipeline (Phase 6) and must not be loaded as depth.; issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved
- `neonatal_lung_sampleA2`: excluded: power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex; issue: neonatal_lung_sampleA2: axes='TYX' contains a time axis. 3-D structural data must be ZYX; TYX belongs to the calcium pipeline (Phase 6) and must not be loaded as depth.; issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved
- `neonatal_lung_sampleA3`: excluded: power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex; issue: neonatal_lung_sampleA3: axes='TYX' contains a time axis. 3-D structural data must be ZYX; TYX belongs to the calcium pipeline (Phase 6) and must not be loaded as depth.; issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved
- `neonatal_lung_sampleB1`: excluded: power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex; issue: neonatal_lung_sampleB1: axes='TYX' contains a time axis. 3-D structural data must be ZYX; TYX belongs to the calcium pipeline (Phase 6) and must not be loaded as depth.; issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved
- `neonatal_lung_sampleB2`: excluded: power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex; issue: neonatal_lung_sampleB2: axes='TYX' contains a time axis. 3-D structural data must be ZYX; TYX belongs to the calcium pipeline (Phase 6) and must not be loaded as depth.; issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved
- `neonatal_lung_sampleB3`: excluded: power-stability calibration: TYX intensity time series, neonatal lung, not a Z stack of mouse cortex; issue: neonatal_lung_sampleB3: axes='TYX' contains a time axis. 3-D structural data must be ZYX; TYX belongs to the calcium pipeline (Phase 6) and must not be loaded as depth.; issue: mouse_id unresolved; issue: session_id unresolved; issue: fov_id unresolved

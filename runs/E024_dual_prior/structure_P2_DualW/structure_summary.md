| volume | Dice | clDice | skelF1 | skeleton components ref→pred | junctions ref→pred | length µm ref→pred | radius µm ref→pred | planes scored |
|---|---|---|---|---|---|---|---|---|
| M2_FOVII_S1_vessels | 0.9381 | 0.9638 | 0.8997 | 1580 → 1055 | 1703 → 1453 | 143348 → 134446 | 2.295 → 2.360 | 146/203 (72 %) |
| M2_FOVII_S2_vessels | 0.8884 | 0.7913 | 0.7483 | 11664 → 7410 | 2725 → 2038 | 136925 → 111618 | 1.759 → 1.935 | 134/202 (66 %) |
| M2_FOVI_S1_vessels | 0.9081 | 0.8153 | 0.7671 | 17833 → 9283 | 3402 → 2559 | 188305 → 157308 | 1.887 → 2.093 | 119/201 (59 %) |
| M2_FOVI_S2_vessels | 0.7630 | 0.5503 | 0.5711 | 31094 → 18501 | 3271 → 2044 | 155620 → 95486 | 1.240 → 1.354 | 51/152 (34 %) |
| **mean** | 0.8744 | 0.7802 | 0.7465 | 15542.75 → 9062.25 | 2775.25 → 2023.5 | 156049 → 124714 | 1.795 → 1.936 | – |

**Planes gated out** (reference mask fills > 0.15 of the plane, so both masks are emptied): M2_FOVII_S1_vessels 57/203, M2_FOVII_S2_vessels 68/202, M2_FOVI_S1_vessels 82/201, M2_FOVI_S2_vessels 101/152. Every number above is a score on the planes that survived, not on the whole volume; report the share beside the table and spot-check the gated planes in ImageJ.
Binarisation: vesselness response, threshold 0.0413627 fitted on the reference only (per_plane), foreground ref 2.2% / pred 2.1%. Skeleton overlap tolerance 1 voxel.
Components above count 26-connected **skeleton** fragments; the connected mask blobs are recorded separately in the JSON as `n_components_ref` / `n_components_pred`.

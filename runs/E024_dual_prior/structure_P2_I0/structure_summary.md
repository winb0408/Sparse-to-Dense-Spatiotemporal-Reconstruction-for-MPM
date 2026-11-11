| volume | Dice | clDice | skelF1 | skeleton components ref→pred | junctions ref→pred | length µm ref→pred | radius µm ref→pred | planes scored |
|---|---|---|---|---|---|---|---|---|
| M2_FOVII_S1_vessels | 0.9168 | 0.9420 | 0.8686 | 1580 → 2653 | 1703 → 1863 | 143348 → 145761 | 2.295 → 2.246 | 146/203 (72 %) |
| M2_FOVII_S2_vessels | 0.8556 | 0.7543 | 0.7193 | 11664 → 12386 | 2725 → 3178 | 136925 → 151731 | 1.759 → 1.692 | 134/202 (66 %) |
| M2_FOVI_S1_vessels | 0.8755 | 0.7471 | 0.7107 | 17833 → 22403 | 3402 → 4180 | 188305 → 220956 | 1.887 → 1.751 | 119/201 (59 %) |
| M2_FOVI_S2_vessels | 0.7089 | 0.5262 | 0.5664 | 31094 → 32820 | 3271 → 4755 | 155620 → 218731 | 1.240 → 1.196 | 51/152 (34 %) |
| **mean** | 0.8392 | 0.7424 | 0.7163 | 15542.75 → 17565.5 | 2775.25 → 3494.0 | 156049 → 184295 | 1.795 → 1.721 | – |

**Planes gated out** (reference mask fills > 0.15 of the plane, so both masks are emptied): M2_FOVII_S1_vessels 57/203, M2_FOVII_S2_vessels 68/202, M2_FOVI_S1_vessels 82/201, M2_FOVI_S2_vessels 101/152. Every number above is a score on the planes that survived, not on the whole volume; report the share beside the table and spot-check the gated planes in ImageJ.
Binarisation: vesselness response, threshold 0.0413627 fitted on the reference only (per_plane), foreground ref 2.2% / pred 2.2%. Skeleton overlap tolerance 1 voxel.
Components above count 26-connected **skeleton** fragments; the connected mask blobs are recorded separately in the JSON as `n_components_ref` / `n_components_pred`.

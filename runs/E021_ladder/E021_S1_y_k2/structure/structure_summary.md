| volume | Dice | clDice | skelF1 | components ref→pred | junctions ref→pred | length µm ref→pred | radius µm ref→pred | planes scored |
|---|---|---|---|---|---|---|---|---|
| M2_FOVII_S1_vessels | 0.9337 | 0.9574 | 0.8892 | 1580 → 1130 | 1703 → 1503 | 143348 → 135125 | 2.295 → 2.349 | 146/203 (72 %) |
| M2_FOVII_S2_vessels | 0.8705 | 0.7739 | 0.7272 | 11664 → 7654 | 2725 → 2138 | 136925 → 107633 | 1.759 → 1.921 | 134/202 (66 %) |
| M2_FOVI_S1_vessels | 0.9007 | 0.8063 | 0.7597 | 17833 → 10475 | 3402 → 2738 | 188305 → 160394 | 1.887 → 2.053 | 119/201 (59 %) |
| M2_FOVI_S2_vessels | 0.7488 | 0.5527 | 0.5731 | 31094 → 19882 | 3271 → 2547 | 155620 → 113274 | 1.240 → 1.324 | 51/152 (34 %) |
| **mean** | 0.8634 | 0.7725 | 0.7373 | 15542.75 → 9785.25 | 2775.25 → 2231.5 | 156049 → 129107 | 1.795 → 1.912 | – |

**Planes gated out** (reference mask fills > 0.15 of the plane, so both masks are emptied): M2_FOVII_S1_vessels 57/203, M2_FOVII_S2_vessels 68/202, M2_FOVI_S1_vessels 82/201, M2_FOVI_S2_vessels 101/152. Every number above is a score on the planes that survived, not on the whole volume; report the share beside the table and spot-check the gated planes in ImageJ.
Binarisation: vesselness response, threshold 0.0413627 fitted on the reference only (per_plane), foreground ref 2.2% / pred 2.1%. Skeleton overlap tolerance 1 voxel.
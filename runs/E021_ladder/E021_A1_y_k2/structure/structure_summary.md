| volume | Dice | clDice | skelF1 | components ref→pred | junctions ref→pred | length µm ref→pred | radius µm ref→pred |
|---|---|---|---|---|---|---|---|
| M2_FOVII_S1_vessels | 0.9168 | 0.9420 | 0.8686 | 1580 → 2653 | 1703 → 1863 | 143348 → 145761 | 3.161 → 3.096 |
| M2_FOVII_S2_vessels | 0.8556 | 0.7543 | 0.7193 | 11664 → 12386 | 2725 → 3178 | 136925 → 151731 | 2.375 → 2.307 |
| M2_FOVI_S1_vessels | 0.8755 | 0.7471 | 0.7107 | 17833 → 22403 | 3402 → 4180 | 188305 → 220956 | 2.495 → 2.371 |
| M2_FOVI_S2_vessels | 0.7089 | 0.5262 | 0.5664 | 31094 → 32820 | 3271 → 4755 | 155620 → 218731 | 1.896 → 1.852 |
| **mean** | 0.8392 | 0.7424 | 0.7163 | 15542.75 → 17565.5 | 2775.25 → 3494.0 | 156049 → 184295 | 2.482 → 2.407 |

Gated planes (reference mask fills > 0.15 of the plane): 57, 68, 82, 101 (one number per volume); those planes are in neither mask.
Binarisation: vesselness response, threshold 0.0413627 fitted on the reference only (per_plane), foreground ref 2.2% / pred 2.2%. Skeleton overlap tolerance 1 voxel.

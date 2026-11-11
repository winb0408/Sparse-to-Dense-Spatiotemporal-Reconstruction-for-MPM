| volume | planes scored | P2_I0 Dice | P2_I0 clDice | P2_I0 skelF1 | P2_DualW Dice | P2_DualW clDice | P2_DualW skelF1 | Δ(P2_DualW−P2_I0) Dice | Δ(P2_DualW−P2_I0) clDice | Δ(P2_DualW−P2_I0) skelF1 |
|---|---|---|---|---|---|---|---|---|---|---|
| M2_FOVII_S1_vessels | 146/203 | 0.9168 | 0.9420 | 0.8686 | 0.9381 | 0.9638 | 0.8997 | +0.0213 | +0.0219 | +0.0310 |
| M2_FOVII_S2_vessels | 134/202 | 0.8556 | 0.7543 | 0.7193 | 0.8884 | 0.7913 | 0.7483 | +0.0328 | +0.0370 | +0.0290 |
| M2_FOVI_S1_vessels | 119/201 | 0.8755 | 0.7471 | 0.7107 | 0.9081 | 0.8153 | 0.7671 | +0.0325 | +0.0681 | +0.0564 |
| M2_FOVI_S2_vessels | 51/152 | 0.7089 | 0.5262 | 0.5664 | 0.7630 | 0.5503 | 0.5711 | +0.0541 | +0.0241 | +0.0047 |
| **mean** | – | 0.8392 | 0.7424 | 0.7163 | 0.8744 | 0.7802 | 0.7465 | +0.0352 | +0.0377 | +0.0303 |

Planes scored (reference mask fills > 15 % of a plane, so both masks are emptied there): M2_FOVII_S1_vessels 146/203, M2_FOVII_S2_vessels 134/202, M2_FOVI_S1_vessels 119/201, M2_FOVI_S2_vessels 51/152. Every Dice above is on the planes that survived; a volume with a third of its planes scored is reporting a Dice on a third of its depth.

These are **whole-volume** numbers at a threshold fitted on the reference (per plane), not the window-level masked Dice in `docs/PHASE1_PR2_E024_RESULTS.md` §2; the two are not comparable and the threshold is not the same number. Cross-animal spread on this cohort is ~0.08 Dice — treat differences below 0.03 as noise.

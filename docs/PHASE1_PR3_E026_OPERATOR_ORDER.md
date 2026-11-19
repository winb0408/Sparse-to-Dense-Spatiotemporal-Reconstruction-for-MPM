# E026: why `P3_Wlegacy` is not an independent floor under this protocol

**Status:** diagnostic, resolved. Written 2026-11-13 while E026 `matched_synthetic`
was training. No test animal is scored here; the numbers are operator-only and
were produced on a **train** volume.

## 1. The observation that started this

E026 carries six arms, two of which are supposed to be different Wiener floors:

| label | arm | warm start |
|---|---|---|
| `W0_full3d` | `P2_W0` | `wiener_fill_all` — fill every hole, then one global 3-D FFT deconvolution |
| `W0_legacy` | `P3_Wlegacy` | `wiener_fill_first` — lateral fill, lateral-only Wiener on measured planes, then axial interpolation |

On the E026 `matched_synthetic` test fold the two floor evaluations agree to
~1e-7 in **every** metric (PSNR 27.1883717 vs 27.1883716 dB, `psnr_missing`
27.06450894 vs 27.06450887, Dice/clDice and the vessel threshold bit-identical).
Two operators that are claimed to be different cannot produce the same floor to
float32 round-off by coincidence, so the floor was treated as suspect until the
cause was found.

## 2. It is not a plumbing defect

Checked in this order, each on the real evaluation path:

1. `evaluate.py` builds the `ItemConfig` from `arm.warm_start` (`warm_start_kind=arm.warm_start`) —
   confirmed by reading the code and by printing the value at run time:
   `P2_W0 → wiener_fill_all`, `P3_Wlegacy → wiener_fill_first`.
2. `VolumeWindowDataset` really hands the two arms **different** inputs
   (`inputs identical False`), so the warm start is not being skipped or cached.
3. The dataset window at an odd query plane really is `[measured, missing, measured]`:
   `query_z=7, window_z0=6 → per-plane measured fraction [0.5, 0.0, 0.5]`,
   i.e. the lattice phase comes from `z0 % kz` and the centre plane is genuinely
   missing — the case where the two orders are supposed to diverge.
4. Calling both warm starts on that same item: the **centre plane** (the only
   plane an evaluation scores) differs by `1.24e-7`, while the measured
   neighbour planes differ by `9.1e-5`.

So the operators do differ; they differ on planes nothing is scored on.

## 3. The measurement that explains it: the difference is a function of context length

`spur_mpm/scripts/e026_operator_context.py` (added with this note) builds the same
lattice windows on a train volume for a sweep of axial context lengths `T`,
computes both warm starts from the identical input, and reports the centre-plane
disagreement. 24 windows spanning both lattice phases per row, λ = 0.01,
H = σ (1.0, 1.5) px:

| T | n windows | centre mean&#124;Δ&#124; | centre max&#124;Δ&#124; |
|---|---|---|---|
| 3 (E026 protocol) | 24 | 3.2e-08 | 1.1e-07 |
| 5 | 24 | 3.7e-03 | 3.1e-02 |
| 7 | 24 | 1.0e-02 | 7.9e-02 |
| 9 | 24 | 1.7e-02 | 2.5e-01 |
| 15 | 24 | 1.4e-02 | 1.7e-01 |

The ordering difference is **zero at the protocol's context and grows with T**.
The E026 preflight, which ran on a 16-plane stack, therefore reported a
difference (0.003019 vs 0.003770 `mse_missing`, `mean|Δ|` = 0.13) that the
actual evaluation windows cannot exhibit.

Mechanistically this is consistent with the axial direction being under-resolved
at T = 3: with only three samples along Z the axial OTF is sampled at DC and the
Nyquist pair only, and after the fill the missing plane is a linear interpolation
between the two measured knots — the same object in both orders. We state the
measurement, not a derivation: the sweep is the evidence, and it is reproducible
with the script above.

Evidence: `docs/evidence/e026_operator_context_20261113/operator_context.json`.

## 4. Consequence for E026 reporting (binding on the results document)

- `W0_legacy` **must not be counted as a second floor or as an independent arm**.
  It is reported as "identical to `W0_full3d` at T = 3 (≤ 1e-6 in every metric);
  the ordering only matters for longer axial context (see the table above)".
- Any future claim of the form "the historical operator order is better/worse"
  requires an operator-only comparison at a context where the orders differ
  (T ≥ 5), run without training, on train volumes, and reported as a diagnostic —
  not as a method result.
- The group's `compare_structure` pair `"W0_legacy W0_full3d"` is kept: a zero
  delta there is the verification that this note still holds for the run at hand.
- Nothing here changes the comparison that E026 exists to make: `I0` vs
  `P2_W0`/`P3_Wlegacy` vs `P2_ResidualI` vs `P3_DualAdaptive[none]` vs
  `P3_DualAdaptive[interp_only]`.

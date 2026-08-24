# v0.17 Thermal/GPE Ridge Push — Findings

## Goal

Replace the v0.16 constant ridge-push line force with a force tied to the explicit mechanical mantle-lithosphere state, while preserving the calibrated large-scale tectonic regime.

Climate remains out of scope and is intended for a separate, higher-power climate model.

## Physical model

v0.16 already carries, for every surface parcel:

- chemical crust thickness;
- mantle-lithosphere thickness `H`;
- mantle-lithosphere thermal density anomaly `Δρ`.

Slab pull uses integrated negative buoyancy, approximately `Δρ H`. Ridge push is instead a gravitational-potential-energy contrast, so v0.17 uses the first-moment proxy

`U_ridge ∝ max(Δρ, 0) H²`.

For half-space cooling, `H ∝ sqrt(age)`, so `H² ∝ age`, reproducing the classic young-ocean ridge-push trend without introducing a second independent age law.

A ridge boundary cannot use only the cell immediately beside the ridge because that parcel is necessarily very young. Instead v0.17 integrates `U_ridge` over the oceanic footprint of each plate. Twice the area-weighted mean approximates the end-to-end GPE contrast of a simple spreading flank. Physical cell area is used, so the result does not scale with mesh-cell count.

Classic plate-cooling calculations do not allow ridge force to increase linearly forever; the force grows only slowly after oceanic lithosphere matures. v0.17 therefore applies a normalized saturating curve to the raw GPE ratio.

Production parameters:

- reference mantle-lithosphere thickness: 93.5 km (approximately an 80-Myr ocean below 7 km crust);
- reference density anomaly: 64 kg/m³;
- saturation ratio: 0.20;
- calibration gain: 1.014;
- numerical floor/cap: 0.20 / 2.40.

Approximate ideal-flank multipliers:

| Age | Ridge-push factor |
|---:|---:|
| 5 Myr | 0.20 |
| 10 Myr | 0.38 |
| 20 Myr | 0.67 |
| 40 Myr | 0.92 |
| 80 Myr | 1.01 |
| 120 Myr | 1.02 |
| 160 Myr | 1.02 |

Thus v0.17 mainly changes newly opened ocean basins. Mature oceanic plates remain close to the calibrated v0.16 ridge force rather than gaining an ever-larger push.

## Calibration history

An initial nearly half-space-like calibration allowed ridge-side multipliers to reach roughly 0.54–2.20 at 500 Myr and systematically produced too many plate splits. A moderate saturation still shifted a four-seed 300-Myr ensemble from 11.25 plates (v0.16) to about 13 plates.

The final strong plate-cooling saturation removes that systematic shift without restoring a literal constant force.

## Paired 300-Myr validation

Four identical sub3 seeds (`20260806`–`20260809`) were run with v0.16 and final v0.17.

Ensemble means:

| Metric | v0.16 | v0.17 |
|---|---:|---:|
| plate count | 11.25 | 11.75 |
| topology events | 5.25 | 5.75 |
| largest plate fraction | 24.32% | 21.22% |
| continental material | 29.05% | 29.10% |
| mean continental plate speed | 0.2187 deg/Myr | 0.2171 deg/Myr |

The v0.17 shifts are small compared with the seed-to-seed variability already documented for the topology model. There is no longer a systematic fragmentation regime.

Detailed rows are in `V117_RIDGE_PUSH_PAIRED_300MYR.csv`.

## 500-Myr canonical-seed comparison on sub3

For seed `20260806`:

| Metric | v0.16 | v0.17 |
|---|---:|---:|
| final plates | 13 | 14 |
| topology events | 9 | 12 |
| largest plate fraction | 30.58% | 23.31% |
| continental material | 29.82% | 29.70% |
| continental volume | 3.603e9 km³ | 3.634e9 km³ |
| final sea level | -328 m | -454 m |
| land fraction | 28.45% | 29.16% |

The one-seed late-time histories diverge chaotically, as expected after multiple topology events, so the paired ensemble is the stronger calibration test. The 500-Myr run remains physically healthy: no continent loss, no grid-lock, no numerical topography clipping, and conservative material transport remains at ~1e-6 km³ absolute error.

Final ridge-side factors in that world are tightly saturated: mean 1.006, min 0.921, max 1.021. The important departures from unity therefore occur earlier, when new oceanic lithosphere is young.

## Resolution and determinism

- Added a direct cross-resolution test showing area-integrated ridge-push factor convergence across sub2/sub3/sub4.
- Canonical sub5 (20,480 cells) completed a 20-Myr smoke run with 12 plates and no consistency errors.
- `0→40 Myr` and `0→20→resume→40 Myr` are bitwise identical in checkpoint arrays and `meta.json`.

## Tests

**109 / 109 passing.**

## Diagnostics

v0.17 writes:

- `ridge_push_gpe_proxy_final.png`;
- `ridge_push_plate_factor_final.png`;
- `ridge_push_history.png`;
- `ridge_push_age_calibration.png`;
- `mean_ridge_push_factor`, `min_ridge_push_factor`, `max_ridge_push_factor` in long-run diagnostics.

## References used for the physical shape

- Parsons, B. & Richter, F. M. (1980), *A relation between the driving force and geoid anomaly associated with mid-ocean ridges*, Earth and Planetary Science Letters 51, 445–450.
- Recent plate-driving-force formulations likewise describe ridge push as the GPE/thickening force of cooling oceanic lithosphere and recover an age-dependent half-space expression; see *Assessing plate reconstruction models using plate driving force consistency tests*, Scientific Reports (2023).

## Recommendation

Freeze v0.17 ridge push at this calibration. Do not tune it to make one 500-Myr seed reproduce the exact v0.16 topology sequence; the paired ensemble is already statistically close, while the new model adds the desired physical suppression of ridge push in very young oceans.

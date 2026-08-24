# v0.16 Crust / Mantle-Lithosphere Split — validation findings

## Scope

v0.16 separates the chemical crust from the mechanical/thermal mantle part of
lithosphere.  The pre-existing `crust_thickness_km` remains chemical crust.
Two new advected checkpoint fields are added:

- `mantle_lithosphere_thickness_km`
- `mantle_lithosphere_density_anomaly_kg_m3`

The hydrosphere, material-aware topography, continental material transport and
physical topology introduced in v0.11-v0.15 remain intact.

## Thermal structure

Oceanic total thermal lithosphere follows a capped half-space-cooling proxy

`delta = min(delta_max, 2 * sqrt(kappa * age))`

with kappa = 1e-6 m2/s and delta_max = 155 km.  The basaltic crust is not part
of the mantle-lithosphere field, so the oceanic mantle layer is
`max(delta - 7 km, 0)`.

For reference, an ~100 Myr ocean has about 112 km total thermal lithosphere,
therefore about 105 km mantle lithosphere beneath its ~7 km crust.

Continental mantle lithosphere has an independent 125 km target root and is
allowed to thin toward 55 km under accumulated rift extension.  It relaxes on a
250 Myr timescale instead of appearing/disappearing instantly when fractional
continental material changes locally.

Oceanic thermal density anomaly is computed from
`rho * alpha * DeltaT * 0.5`, about 64 kg/m3 for the canonical parameters.
Pure continental mantle lithosphere uses a small compositional buoyancy anomaly
of -8 kg/m3.  Mixed cells blend unresolved endmembers by continental footprint.

## Transport invariant

The mantle-lithosphere fields are transported with the same winning surface
parcel as plate ownership.  They are not fixed-grid properties.  New ridge-gap
ocean starts with approximately zero cold mantle-lithosphere thickness.
Oceanization of a continental rift likewise removes the old coherent mantle
root at the spreading axis.

A dedicated test assigns a unique mantle-root marker to every source cell and
verifies exact source-marker recovery after one lithosphere transport step.

## Slab pull

Ocean-ocean subduction polarity now prefers the side with the larger integrated
local negative-buoyancy proxy

`B = mantle_lithosphere_thickness * max(density_anomaly, 0)`.

When v0.16 fields are present, slab-pull strength no longer multiplies crust age
by one global thermal-lithosphere thickness.  It uses local B instead.

A calibration gain of 1.85 preserves the v0.15 force scale at the ~80 Myr
reference ocean.  This deliberately preserves the calibrated macro-regime while
changing the shape of the law: very young ocean pulls much less, old/cold ocean
pulls about as strongly as before.

The old age/global-thermal path remains only as a backwards-compatible fallback
for legacy states that do not contain v0.16 fields.

## No bathymetry double counting

The new mantle-lithosphere field is **not** independently added to topographic
subsidence.  Oceanic bathymetry still uses the existing cooling-age relation.
A unit test confirms that changing only the new mantle-lithosphere fields leaves
the instantaneous topographic target bit-identical.  Thus the same cooling
signal is used for two different physical consequences (bathymetry and slab
negative buoyancy), not added twice to elevation.

## 500 Myr calibrated comparison, same sub3 seed

| Metric | v0.15 | v0.16 |
|---|---:|---:|
| Final plate count | 13 | 13 |
| Topology events | 9 | 9 |
| Largest plate | 32.92% | 30.58% |
| Continental material | 29.48% | 29.82% |
| Continental volume | 3.590 bn km3 | 3.603 bn km3 |
| Mean continental plate speed | 0.2405 deg/Myr | 0.2410 deg/Myr* |
| Sea level | -337.2 m | -328.1 m |
| Land fraction | 28.01% | 28.45% |

`*` The 0.2410 value is from the actual final 500-Myr integration step.  A later
zero-length re-finalization was used only to generate new maps and therefore
cannot reconstruct dynamics diagnostics from a step that did not run.

This is intentionally close: the goal of v0.16 is to improve the internal
physics of the plate, not retune the entire tectonic regime.

## Final v0.16 local layer statistics (sub3, 500 Myr)

- Mean nearly-pure-ocean mantle-lithosphere thickness: 74.63 km
- p90 nearly-pure-ocean mantle-lithosphere thickness: 147.13 km
- Mean continental/mixed mantle-root thickness: 106.84 km
- Mean oceanic negative-buoyancy proxy: 4796.9 km kg/m3
- Maximum negative-buoyancy proxy: 9523.8 km kg/m3
- Mean nearly-pure-ocean crust age: 77.5 Myr

The p90 near 147 km mantle thickness corresponds to old ocean approaching the
155 km total-lithosphere cap minus ~7 km crust.

## Resolution sanity check

The canonical sub5 mesh (20,480 cells) was run fresh to 20 Myr after the split.
It completed without consistency/topology errors and retained 12 plates.  At
20 Myr the nearly-pure-ocean mantle layer averaged ~65 km with p90 ~99 km,
consistent with the initialized mature ocean-age distribution plus new ridge
crust.

## Checkpoint determinism

A direct 0->40 Myr run and 0->20->resume->40 Myr run are bit-identical for all
NPZ arrays and exactly equal for `meta.json`, including both new mantle-
lithosphere arrays.  A resume-only refresh bug found during development was
removed; existing v0.16 fields are never recomputed merely because a checkpoint
was loaded.

## Tests

104/104 tests pass.

New coverage includes:

- sqrt(age) oceanic thermal-lithosphere growth;
- crust thickness != mantle-lithosphere thickness;
- older ocean has greater integrated negative buoyancy;
- continental rifting thins the mantle root without directly changing crust;
- ocean-ocean subduction polarity chooses the more negatively buoyant mantle
  lithosphere when crust ages are identical;
- mechanical lithosphere advects with surface parcels;
- checkpoint roundtrip preserves both new arrays;
- changing only mantle-lithosphere fields cannot double-count bathymetry.

## Deliberate limitations / next physics

v0.16 is still a 2-D surface plate model.  It does not yet create an explicit
3-D buried slab.  Therefore slab rollback, slab breakoff, slab depth/length,
flat-slab geometry and long-lived subduction-zone memory are not yet represented
as mantle objects.

Also deliberately unchanged for this version:

- ridge push is still an effective boundary term rather than an integral of
  plate cooling/GPE;
- continent-continent collision resistance remains primarily crust/GPE based;
- bathymetry remains age/cooling based rather than being recomputed from the new
  mantle-layer field (to avoid double counting during this split step).

Those can now be improved independently because crust and mechanical
lithosphere are no longer the same state variable.

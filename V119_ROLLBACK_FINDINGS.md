# Moon Tectonics v0.19 — Slab Rollback / Trench Migration / Back-Arc Extension

## Scope

v0.19 extends the persistent 2.5-D slab memory of v0.18 with a deliberately
small effective rollback coupling. It remains a rigid-plate/effective tectonic
model rather than a 3-D mantle solver.

A mature, deep, negatively buoyant *active* slab can now:

1. produce a small landward angular-velocity contribution on its overriding
   plate, representing oceanward trench retreat in the overriding-plate frame;
2. produce a local extensional forcing band behind the trench, which is passed
   into the existing conservative continental-rifting machinery.

The module never creates a new plate boundary directly. Back-arc opening still
requires the ordinary accumulated-extension, duration, thinning, and topology
criteria.

## Production calibration

The first implementation was intentionally tested stronger, then reduced after
it produced too many early topology changes.

Final production values:

- rollback activation requires an active slab older than `20 Myr`;
- weak/shallow slabs remain effectively stationary until roughly
  `550 km` down-dip length / `350 km` depth;
- rollback approaches its full value around `1400 km` length / `900 km` depth;
- maximum effective trench-retreat rate: **2 km/Myr** (`0.2 cm/yr`);
- cumulative diagnostic rollback distance cap: `1600 km`;
- back-arc band: roughly **120–650 km** landward of the trench;
- peak back-arc forcing near **280 km**;
- maximum rollback-only extension forcing: **0.15**.

This is deliberately conservative. Observed global trench-retreat estimates
are often several km/Myr or more, but the present model lacks full mantle flow,
slab-edge toroidal flow, and explicit deformable overriding plates. The small
coefficient prevents rollback from silently becoming a new dominant plate
force.

## Force coupling

The plate-scale rollback contribution is expressed directly as an angular
velocity target on the overriding plate:

`omega_rollback ~ v_rollback / R`

with direction chosen so the overriding plate moves landward and the trench
retreats oceanward in that plate's reference frame.

Multiple slab segments acting on the same overriding plate are weighted by
remembered trench length rather than naively summed.

Back-arc extension is spatially restricted to the overriding plate and to the
landward half-space behind the remembered trench. Its triangular physical
profile peaks near 280 km from the trench. The resulting field enters
`advance_lithosphere()` as an independent tectonic extension forcing and is
combined with existing kinematic extension using `max()`, not addition.

## Validation

### Test suite

**117 / 117 tests passed.**

### Checkpoint / resume

A continuous `0 -> 40 Myr` run and a segmented
`0 -> 20 -> resume -> 40 Myr` run are now:

- bitwise identical in every NPZ array;
- exactly identical in the complete `meta.json`, including rollback and slab
  diagnostic histories.

An initial implementation differed only at ~1e-16 in diagnostic floating-point
means because dictionary iteration order changed after JSON reload. Aggregates
are now computed in sorted slab-key order.

### Canonical-resolution smoke test

`subdivision=5` (`20,480` cells), seed `20260806`, 20 Myr:

- 12 plates;
- no topology/ID inconsistency;
- continental material fraction about 28.04%;
- rollback still incipient: final mean `0.153 km/Myr`, max `0.578 km/Myr`;
- no numerical transport failure.

## Paired four-seed sub3 ensemble, 300 Myr

This comparison uses the rebuilt v0.18 source line and the final v0.19
production calibration, same seeds `20260806..20260809`.

| metric | rebuilt v0.18 | v0.19 |
|---|---:|---:|
| mean final plate count | 12.00 | **11.25** |
| plate-count range | 11–13 | **8–14** |
| mean topology events | 5.50 | **5.25** |
| mean largest-plate fraction | 20.70% | **21.50%** |
| mean plate speed | 0.2552 deg/Myr | **0.2500 deg/Myr** |
| mean continental-material fraction | 28.84% | **28.97%** |

The sign of the topology change is not consistent by seed:

- 20260806: `11 -> 11` plates;
- 20260807: `11 -> 8`;
- 20260808: `13 -> 14`;
- 20260809: `13 -> 12`.

Thus the final calibration does not show a simple systematic fragmentation or
merger bias. The small mean changes are well within the chaotic spread already
present in the topology model.

See `V119_ROLLBACK_PAIRED_300MYR.csv`.

## 500-Myr demonstration, seed 20260806 / sub3

Rebuilt v0.18 baseline:

- final plates: 13;
- topology events: 9;
- largest plate: 23.05%;
- mean plate speed: 0.2636 deg/Myr;
- continental material: 30.02%;
- continental volume: 3.634 billion km3.

v0.19:

- final plates: **14**;
- topology events: **10**;
- largest plate: **23.67%**;
- mean plate speed: **0.2420 deg/Myr**;
- continental material: **29.45%**;
- continental volume: **3.602 billion km3**;
- active rollback zones: ~45 near the final diagnostic step;
- mean rollback rate: **1.72 km/Myr**;
- maximum rollback rate: **2.0 km/Myr**;
- mean integrated effective rollback distance: **300 km**;
- maximum: **889 km**;
- rollback-driven back-arc forcing area: **19.4 million km2**;
- maximum instantaneous back-arc forcing: **0.144**.

The continental budget and largest-plate scale remain in the previous regime.
The new mechanism changes the detailed topology history, as expected for a
chaotic plate system, but does not create runaway speed or fragmentation.

## Important reconstruction note

The temporary source tree for the original v0.18 implementation was removed by
the execution environment before it was archived. The preserved v0.17 archive,
v0.18 findings, paired-validation CSV, diagnostics, and tests survived.

For v0.19 development, v0.18 source was therefore rebuilt from v0.17 plus those
preserved specifications and immediately re-archived. The rebuilt line passes
113/113 v0.18 tests and its own smoke/checkpoint tests, but it does **not**
reproduce every historical v0.18 seed trajectory bit-for-bit. Therefore the
paired v0.18/v0.19 ensemble in this report is explicitly labelled **rebuilt
v0.18 baseline** rather than the historical run stored in the older CSV.

This does not affect the v0.19 internal checkpoint determinism or its own
validation, but it should remain documented for provenance.

## Main remaining slab limitation

Rollback reduces the conceptual problem of a stationary trench, but it does
not solve slab saturation. In the final 500-Myr v0.19 checkpoint:

- remembered slab zones: 65;
- 45 are at the `1800 km` slab-length cap;
- 49 are at the `1100 km` slab-depth cap;
- none has reached the `1600 km` rollback-distance cap.

The next physical step should therefore be **slab breakoff / collision
transition**, not stronger rollback.

## Recommended next work

1. **Slab breakoff and continent-arrival choking**
   - detect buoyant continental lithosphere entering an active trench;
   - accumulate necking/stall state;
   - detach the oceanic slab on a finite timescale;
   - remove residual pull rapidly after breakoff;
   - allow post-breakoff uplift / thermal relaxation proxy later.

2. **Arc geometry from slab depth**
   - place volcanic arcs landward where the remembered slab reaches a
     dehydration-depth window instead of directly on the convergent mesh edge;
   - feed that location into the existing felsic/continental-growth cycle.

3. **Flexural isostasy**
   - introduce non-local elastic bending for trenches, forebulges, mountains,
     forearc and foreland basins.

4. **Conservative erosion and sediment transport**
   - keep climate outside this program;
   - accept precipitation/erosivity from the external climate model later;
   - conserve sediment mass through shelves, basins and trenches.

5. **Continental age/composition/strength and cratons**
   - distinguish young arc crust from ancient strong cratonic lithosphere.

6. **Long multi-seed / sub4-sub5 validation and performance optimization**
   before any final multi-Gyr production ensemble.

A full 3-D mantle solver remains optional rather than required for the intended
world-history generator.

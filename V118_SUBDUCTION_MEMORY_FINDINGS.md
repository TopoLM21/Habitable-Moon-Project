# Moon Tectonics v0.18 — Subduction Memory Findings

## Scope

v0.18 adds persistent 2.5-D memory for subduction zones while keeping the
surface model a rigid-plate effective tectonic solver rather than a full 3-D
mantle-convection code.

A zone is keyed by an ordered `(subducting_plate -> overriding_plate)` pair and
stores:

- active and inactive age;
- approximate down-dip slab length and depth;
- slab dip;
- trench length and midpoint;
- mean convergence rate;
- entry negative-buoyancy proxy;
- cumulative subducted area;
- remembered torque direction.

The state is deterministic, checkpointed, and remapped through plate topology
changes. For splits, trench location is used to attach the remembered slab to
the geographically appropriate child plate.

## Final production calibration

The important calibration decision is that v0.17 already represents the
calibrated effective slab pull of a mature active subduction zone. v0.18
therefore does **not** add a second mature-slab force on top of it:

- active slab multiplier: `1.00 -> 1.00`;
- residual pull gain after contact loss: max `0.05` of calibrated slab pull;
- residual exponential decay timescale: `24 Myr`;
- detach after `80 Myr` inactive;
- slab-length growth efficiency: `0.90`;
- slab length cap: `1800 km`;
- slab depth cap: `1100 km`;
- dip matures approximately `35 -> 55 deg`;
- ocean-ocean polarity hysteresis exists as an optional experiment but is
  **disabled by default**.

Residual torques from multiple remembered trenches on one plate are
trench-length-weighted rather than naively summed. This prevents many small
stored segments from multiplying the plate-scale force.

## Why stronger memory was rejected

Several stronger variants were tested and rejected:

1. Mature active memory adding about 25–30% extra slab pull strongly changed the
   plate-count distribution. This was interpreted as double-counting the same
   negative buoyancy already represented by v0.17.
2. Strong ocean-ocean polarity hysteresis (`~1.75x` override threshold) locked
   zones into old polarities and systematically changed topology.
3. A 70–90% initiation penalty for young active zones also shifted topology by
   giving rifting/merging too much extra time before full slab pull developed.

The residual-only production form is deliberately conservative: it adds only a
piece of physics that v0.17 genuinely could not represent — an already sinking
slab does not cease exerting force in the exact step when surface contact is
briefly reclassified.

## Validation

- Test suite: **113/113 passed**.
- Continuous `0 -> 40 Myr` and segmented `0 -> 20 -> resume -> 40 Myr` runs are
  bitwise identical in all NPZ arrays and have identical `meta.json`, including
  slab memory.
- With `subduction_memory.enabled=false`, the new runner reproduces the v0.17
  physical checkpoint state bitwise.
- Canonical `subdivision=5` (`20,480` cells) 20-Myr smoke run completed with 12
  plates, no topology inconsistency, max conservative material-transport error
  `4.77e-7 km3`, and 50 remembered/incipient slab zones before final remapping.

### Paired sub3 ensemble, same four seeds

At **300 Myr**:

- v0.17 plate counts: `12, 10, 12, 13` — mean `11.75`;
- v0.18 plate counts: `13, 10, 12, 11` — mean `11.50`.

At **500 Myr**:

- v0.17 plate counts: `14, 7, 7, 11` — mean `9.75`;
- v0.18 plate counts: `13, 13, 9, 10` — mean `11.25`.

The 300-Myr distributions are essentially the same regime. By 500 Myr there is
a mild WATCH that residual slab pull may let some plates survive longer or
reach a later disconnect event instead of an earlier merge. With only four
chaotic worlds this is not a justified target for further coefficient fitting.

See `V118_SUBDUCTION_MEMORY_PAIRED_VALIDATION.csv`.

## Final 500-Myr demonstration world

Seed `20260806`, subdivision 3:

- final plates: **13**;
- topology events: **13**;
- continental-material fraction: **29.56%**;
- land fraction: **27.08%**;
- sea level: **-392.8 m**;
- final remembered zones after topology remapping: **55**;
  - active: **37**;
  - residual: **18**;
- slab-zone births: **135**;
- slab detachments: **58**;
- maximum slab length: **1800 km**;
- maximum slab depth: **1100 km**;
- cumulative subducted oceanic area: **1.202 billion km2**;
- final mean residual slab-pull fraction: **1.25%**;
- final maximum residual slab-pull fraction: **3.97%**;
- maximum conservative continental-material transport error:
  **9.54e-7 km3**.

Cumulative subducted area is a flux integral, not a unique-area measure. It may
exceed the moon's surface area because new oceanic lithosphere is repeatedly
created and recycled. In this 500-Myr example it is about 3.4 moon-surface
areas.

## Main WATCH: slab saturation

At the final checkpoint, 36 of 55 remembered zones are at the `1800 km` slab
length cap and 38 of 55 are at the `1100 km` depth cap. This is not treated as a
numerical failure; it marks the physical limit of the current 2.5-D model.
There is no mantle transition-zone interaction, slab rollback, stagnation,
lower-mantle penetration, or mechanically triggered slab breakoff yet.

This saturation is the main reason the next recommended physical step is
**trench/slab rollback and trench migration**, followed by **explicit slab
breakoff/detachment during continent arrival or stalled convergence**.

## Recommended roadmap after v0.18

### High priority

1. **Slab rollback + trench migration + back-arc extension**
   - use remembered slab depth/dip/negative buoyancy;
   - allow the trench to retreat relative to the overriding plate;
   - feed rollback into overriding-plate extension and back-arc rifting.

2. **Physical slab breakoff / collision transition**
   - buoyant continent arriving at the trench should choke subduction;
   - slab necking/breakoff should remove pull rapidly but not instantaneously;
   - avoid slabs merely sitting forever at the depth/length caps.

3. **Arc geometry tied to slab depth**
   - position volcanic arcs landward of the trench where the slab reaches a
     characteristic dehydration depth, rather than directly at every
     convergent edge;
   - feed this into the existing felsic/continental growth cycle.

4. **Flexural isostasy / elastic loading**
   - replace purely local Airy response for trench/forebulge/mountain loads;
   - enables forearc/foreland basins and realistic wavelength of lithospheric
     bending.

5. **Conservative erosion + sediment transport**
   - tectonic base model can use generic erosion/transport;
   - the external climate program can later supply precipitation/erosivity;
   - sediments should affect trenches, shelves and basin fill without putting
     climate physics inside this program.

### Medium priority

6. **Continental composition, age, strength and cratons**
   - young arcs and ancient cratons should not have identical density and
     rheology;
   - preserve thick strong cratonic mantle roots and allow thermal/rift damage.

7. **Long-wavelength dynamic topography**
   - couple the existing fixed-grid mantle-flow background weakly to surface
     elevation;
   - still much cheaper than solving full mantle convection.

8. **Plumes / hotspots**
   - intraplate volcanism, hotspot tracks and a possible breakup trigger;
   - useful but not required for basic plate-cycle plausibility.

9. **Diffuse deformation / transform fault memory**
   - allow broad plate-boundary zones instead of all strain living on a
     one-cell line;
   - especially useful for continental transforms and collision interiors.

### Validation / interface work

10. **Long ensemble validation**
    - more seeds on sub4/sub5 and eventually multi-Gyr runs;
    - WATCH plate survivability, largest-plate distribution, slab saturation,
      continental volume and sea-level statistics.

11. **Performance work for canonical sub5**
    - sparse matching/topology remain expensive for long high-resolution runs;
    - optimize before doing large 4-Gyr ensembles.

12. **Stable export format for the external climate model**
    - elevation and sub-grid land fraction;
    - sea level/water depth;
    - crust/lithosphere thickness and age;
    - plate IDs and velocities;
    - volcanic/subduction/ridge flux proxies;
    - optional tectonic CO2-outgassing proxy for a separate carbon/climate
      model.

### Optional endgame

A true 3-D mantle/slab solver is **not required** for the intended world-history
generator. The 2.5-D approach can plausibly stop after rollback, breakoff,
flexure and sedimentary/arc improvements. Full 3-D slab geometry, transition-
zone rheology and mantle convection should be treated as optional endgame if
higher physical fidelity later justifies the cost.

# v0.21 Slab-Geometry Volcanic Arcs — findings

## Scope
v0.21 replaces boundary-local volcanic-arc placement with a passive 2.5-D field derived from the persistent slab geometry introduced in v0.18–v0.20. It does **not** add a second crust-production system: the field feeds the existing continental cycle and topography.

## Physical rule
- Active arcs require a remembered active slab older than 8 Myr and deeper than 55 km.
- Reference sub-arc slab depth is 105 km. A weak convergence-rate correction is bounded to 70–130 km.
- Horizontal trench-to-arc distance is `target_depth / tan(slab_dip)`, bounded to 55–420 km.
- Every current convergent mesh segment belonging to an oriented remembered slab pair projects its own arc segment onto the overriding plate. This produces an arc **line**, not one hotspot per plate pair.
- Production width is 140 km (outer influence 320 km). If that physical band is narrower than a coarse mesh cell, it is represented as an unresolved sub-grid band rather than silently disappearing.
- Broken slabs may leave a small post-breakoff pulse (22% peak, 10 Myr e-folding, 32 Myr maximum duration). It uses the same old arc position and creates no new mass source by itself.

Observational motivation: arc volcanoes commonly overlie slab depths of order ~100 km, but observed values vary substantially (roughly 65–130+ km) and correlate with kinematics. Central American volcanic centers commonly lie above ~90–110 km slab depths. See England, Engdahl & Thatcher (2004), Grove et al. (2009), and Gazel et al. (2021).

## Coupling to existing geology
For a dimensionless arc field `A` and continental material fraction `f`:
- juvenile/island-arc maturation intensity = `A * (1-f)`;
- continental-arc thickening intensity = `A * f`.

Forearc/subduction erosion remains tied to the plate interface at the trench. Topographic arc uplift is moved from the trench-side boundary cell to the slab-projected arc field.

## Numerical validation
- Full suite: **127/127 tests**.
- Checkpoint determinism: `0→40 Myr` and `0→20→resume→40` are bit-identical in every `state.npz` array and in the complete `meta.json`, including arc diagnostics.
- Canonical sub5 (20,480 cells), 20 Myr smoke: 12 plates, 0 topology events, 39 active arc zones, mean target slab depth 110.4 km, mean trench–arc distance 136.0 km, max forcing 1.019.

### Paired sub3, 4 seeds, 300 Myr
This deliberately coarse mesh under-resolves a 100–300 km volcanic front.
- mean plate count: v0.20 12.25, v0.21 12.25;
- mean topology events: 4.25 vs 5.25;
- mean continental volume: 3.5163 vs 3.4694 billion km³;
- mean generated continental volume: 136.7 vs 104.1 million km³.

The deficit is mainly juvenile island-arc maturation, not continental-arc thickening. This was **not** compensated by an arbitrary productivity gain.

### Paired sub4, 3 seeds, 200 Myr
At a more appropriate spatial resolution the budget converges without gain tuning:
- mean plate count: 13.67 vs 14.00;
- mean topology events: 3.67 vs 3.33;
- mean continental volume: **3.4420 vs 3.4425 billion km³**;
- mean generated volume: **55.31 vs 59.01 million km³**.

This is the main calibration result: the v0.21 geometry preserves the previous continental budget once the arc is spatially resolved.

### 500 Myr coarse diagnostic, seed 20260806
v0.20 → v0.21:
- plates: 14 → 16;
- topology events: 12 → 12;
- continental volume: 3.5806 → 3.4780 billion km³;
- generated volume: 206.4 → 148.1 million km³.

Final v0.21 arc state: 49 active arc zones, 4 post-breakoff pulses, mean target slab depth 110.1 km, mean trench–arc distance 98.4 km. This run is retained as a coarse diagnostic, not as the calibration target.

## WATCH
1. Each oriented plate pair still has one remembered mean slab dip; individual trench segments do not yet have independent dip histories.
2. Subdivision 3 is too coarse for quantitative island-arc maturation. Use sub4/sub5 for arc-budget calibration.
3. The post-breakoff magmatic pulse is an effective heuristic and should remain weak until melt/thermal structure is modeled explicitly.
4. Arc forcing is passive with respect to plate forces; only its crustal/topographic consequences feed back later through existing mass/GPE physics.

## References
- England, P., Engdahl, R. & Thatcher, W. (2004), *Systematic variation in the depths of slabs beneath arc volcanoes*, Geophysical Journal International. https://pubs.usgs.gov/publication/70027074
- Grove, T. et al. (2009), *Kinematic variables and water transport control the formation and location of arc volcanoes*, Nature 459. https://www.nature.com/articles/nature08044
- Gazel, E., Flores, K. & Carr, M. (2021), *Architectural and Tectonic Control on the Segmentation of the Central American Volcanic Arc*, Annual Review of Earth and Planetary Sciences 49.

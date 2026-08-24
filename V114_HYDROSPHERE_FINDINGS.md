# v0.14 Hydrosphere findings

## Scope
v0.14 adds a passive, globally connected hydrosphere to the stabilized v0.13 tectonic model. One water inventory is conserved. At each tectonic step the sea surface is solved from the current topography using exact spherical-shell volume integration for each icosphere cell.

The ocean is intentionally **one-way coupled** in v0.14. It does not yet alter erosion, sedimentation, lithospheric loading, climate, or plate forces.

## Initialization correction discovered by the hydrosphere
The legacy lithosphere initializer assigned age `0 Myr` to every oceanic cell. With plate-cooling bathymetry this made the entire ocean floor cool and subside together during the first tens of Myr, creating an artificial sea-level crash (~1.85 km in the initial coarse smoke test).

Fresh v0.14 runs now seed a mature oceanic-age field:

- oceanic cells on active divergent boundaries start at age 0;
- age grows with within-plate geodesic distance from a ridge using a 30 km/Myr effective half-spreading rate;
- age is capped at 160 Myr;
- oceanic regions on plates without an active ridge start at 120 Myr.

This reduced the same 40-Myr startup sea-level excursion from roughly -1.85 km to about -0.38 km.

## 500-Myr sub3 diagnostic run
Canonical seed `20260806`, subdivision 3 (1280 surface cells), dt=4 Myr.

- calibrated conserved water volume: **1.0880 billion km3**;
- equivalent global water layer: **3097 m**;
- initial land fraction: **27.78%**;
- final land fraction: **27.72%**;
- minimum land fraction during run: **23.51%**;
- maximum land fraction during run: **29.08%**;
- initial sea level: **0 m** by calibration;
- minimum sea level: **-560.7 m**;
- final sea level: **-310.4 m**;
- initial mean ocean depth: **4.293 km**;
- final mean ocean depth: **4.290 km**;
- final maximum water depth: about **9.04 km**;
- maximum relative water-volume solver error: about **1.1e-8**;
- final plate count: **13**.

The sea-level signal is therefore a basin-volume response rather than water creation/loss. Land fraction stays near the intended ~28% on this run even though sea level varies by several hundred metres.

## Resolution snapshot at t=0
With the same seed and mature-ocean initializer:

| subdivision | cells | calibrated water (billion km3) | equivalent global layer (m) | land fraction |
|---|---:|---:|---:|---:|
| 3 | 1,280 | 1.0880 | 3097 | 27.78% |
| 4 | 5,120 | 1.0779 | 3069 | 27.94% |
| 5 | 20,480 | 1.1267 | 3208 | 27.92% |

The land fraction is already very stable. Water inventory differs by several percent because the initial plate/ridge geometry and therefore initial ocean-age/bathymetry field is itself resolution-dependent. That is a property of the tectonic initial-condition generator, not the sea-level root solver.

## Checkpoint determinism
A fresh `0 -> 40 Myr` run and `0 -> 20 -> resume -> 40 Myr` run produced:

- bitwise-identical NPZ arrays;
- identical `meta.json`;
- identical checkpointed water volume;
- identical hydrosphere diagnostic history.

## What v0.14 still does not model
1. Ocean loading/flexure and water mass feedback on isostasy.
2. Erosion relative to the actual shoreline; current erosion still uses the historical topographic datum.
3. Sediment production, transport, continental shelves and sediment-loaded trenches.
4. Climate/glaciation and transfer of water between ocean and ice.
5. Local lakes/endorheic basins; all water is treated as one connected global ocean.
6. Sub-cell coastline geometry. At coarse resolution a whole triangular cell is wet or dry, so shelf area is under-resolved.
7. Topography still uses a largely binary oceanic/continental crust label even though v0.11 material transport is fractional. A later topographic material-mixing pass should make coastlines and continental margins smoother and more physically consistent.

## Recommended next step
Keep the hydrosphere passive for now. The most useful next improvement is to make topographic/isostatic elevation consume the v0.11 `continental_fraction + continental_volume` material layer instead of the legacy binary crust label. That directly improves continental margins and coastlines without feeding new uncertain physics into plate dynamics. After that, shoreline-aware erosion/sedimentation is the natural next coupled process.

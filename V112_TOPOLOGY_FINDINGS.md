# Moon Tectonics v0.12 — physical topology findings

## What was wrong

v0.11 fixed continental-material transport, but topology still mixed physical thresholds with mesh-count thresholds. In the canonical configuration:

- `min_plate_cells = 90`
- `split_min_child_cells = 180`
- `disconnect_min_child_cells = 400`
- `split_min_rift_cells = 8`

A cell at subdivision 3 has ~274,000 km², while a cell at subdivision 5 has ~17,150 km². The same cell-count threshold therefore changed physical meaning by about 16× across the resolution sweep.

Late boundary nucleation had the same problem: minimum rift path length was 20 cells and rift width was one neighbour ring.

## v0.12 changes

Production topology now uses physical thresholds:

- microplate absorption: `min_plate_area_km2 = 1.55e6`
- ordinary split child: `split_min_child_area_km2 = 3.10e6`
- disconnected child: `disconnect_min_child_area_km2 = 6.90e6`
- coherent rift-band minimum span: `split_min_rift_span_km = 1000`
- continental collision contact remains a physical length threshold (900 km), now evaluated using the v0.11 continental fraction field rather than only the legacy binary crust label.

The values above deliberately preserve the approximate physical meaning of the old canonical subdivision-5 calibration. They are calibration constants, not fundamental geology.

Late-rift spatial rules are also physical:

- minimum cross-plate rift path: 2500 km
- maximum path guard: 20,000 km
- rift half-width: 160 km
- collision-seam spread scale: 160 km

The seam field uses sub-grid dilution when a 160 km fault is narrower than a mesh cell, so coarse cells do not receive the same full-cell weakness as resolved fine cells.

## Resolution tests

The automated suite is now 82/82. New tests verify across subdivisions 3, 4 and 5 that:

1. the same ~4.3 million km² microplate is classified the same way by an area threshold;
2. the same two disconnected ~10 million km² plate fragments trigger the same disconnect split;
3. the same physical rift band triggers a split using span and child area rather than cell counts;
4. a physical Dijkstra rift path has convergent length as the mesh is refined;
5. area-integrated unresolved collision-seam weakening is approximately resolution stable;
6. an identical synthetic continent-continent contact welds after the same physical collision/quiet time on all three resolutions.

## Real-seed check (seed 20260806, 500 Myr)

Old v0.11 final plate counts were:

- subdivision 3: 4
- subdivision 4: 8
- subdivision 5: 10

With physical topology thresholds, the completed runs gave:

- subdivision 3: 11
- subdivision 4: 7
- subdivision 5: 12

The important early-time result is stronger than the raw 500 Myr counts. Around 200 Myr the systems still contain approximately 12, 14 and 13 plates respectively, and the first topology event on all tested resolutions is a macroscopic `disconnect_split` (132–168 Myr). The catastrophic early absorption of coarse-grid plates is gone.

At 500 Myr subdivision 4 is still an outlier. Inspection of collision memory shows that by ~200 Myr the actual colliding plate pairs already differ between resolutions. This is not simply the same weld clock crossing a threshold on different meshes; the dynamical trajectories have diverged. Plate tectonic evolution is chaotic, so exact same-seed event histories are not expected to remain identical after hundreds of Myr once small geometric differences have changed previous topology events.

Therefore the next convergence criterion should be statistical: compare event rates, plate-count distribution and largest-plate distribution over an ensemble at each resolution, rather than demand identical event sequences for one seed.

## Current conclusion

The topology *rules* are now expressed in physical units and pass direct cross-resolution synthetic tests. Long-run same-seed trajectories still diverge, as expected for a chaotic discrete plate system. The remaining validation task is ensemble-level convergence; it should be done before retuning physical merge/split coefficients.

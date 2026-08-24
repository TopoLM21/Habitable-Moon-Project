# Moon Tectonics v0.15 — material-aware isostasy/topography

## Scope

v0.15 is a passive relief/hydrosphere refinement on top of v0.13 topology,
v0.11 conservative continental material, and v0.14 conserved water. It does
**not** alter plate forces, plate topology, material transport, mantle flow, or
climate.

## Material-aware base relief

The legacy topography switched abruptly at `crust_type >= 50%`: an otherwise
identical cell could jump from oceanic bathymetry to continental Airy relief
when the visible binary label changed.

v0.15 instead uses the independent v0.11 material fields:

- `continental_fraction = f`;
- `continental_volume_km3 = Vc`;
- material thickness `Hc = Vc / (A f)` for `f > 0`;
- oceanic endmember from the existing thermal depth-age law;
- continental endmember from the existing Airy calibration;
- scalar cell relief is the area-weighted closure
  `z = (1-f) z_ocean + f z_continent`.

Thus `f=0` and `f=1` reproduce the old pure endmembers exactly while mixed
cells vary continuously across the old 0.5 visibility threshold.

Tectonic relief is also continuous in material fraction: ridge relief blends
between oceanic spreading and continental rifting; continental collision
uplift scales with shared continental fraction; residual oceanic convergence
can still produce trench/arc relief.

## Why scalar averaging alone was insufficient

A mixed cell is unresolved footprint, not a physically homogenized single
surface. For example, `f=0.8` means 80% continental footprint and 20% oceanic
footprint. A scalar area-mean height can lie below sea level even while the
continental 80% remains exposed. Treating the entire mean cell as wet produced
artificially large submerged continental areas.

v0.15 therefore adds **sub-grid material hypsometry** to the passive
hydrosphere. Each mixed cell contributes two spherical surface patches to basin
capacity:

- area `(1-f) A` at the oceanic endmember surface;
- area `f A` at the continental endmember surface.

The current topography's relaxation/erosion residual is applied equally to both
patches, so their area-weighted mean remains exactly the stored scalar
`TopographyState.elevation_m`. Water volume and sea level are then solved over
both patches, not over the scalar mean alone.

## Validation

Test suite: **97/97 passing**.

Checkpoint determinism: a sub2 `0→40 Myr` run and `0→20→resume→40 Myr` run
produce bitwise-identical NPZ arrays and identical `meta.json`, including the
hydrosphere history.

### 500 Myr, seed 20260806, sub3

- continental material area: **29.48%**;
- land area: **28.01%**;
- submerged continental material: **1.46%**;
- sea level: **−337.2 m** relative to the original datum;
- mean ocean depth: **4.31 km**;
- maximum ocean depth: **8.87 km**;
- final plate count: **13**;
- topology events: **9**;
- pure conservative transport max volume error: **9.54e−7 km³**;
- numerical topography safety-rail hits: **0**.

The same tectonic/topographic state diagnosed using the rejected one-height
scalar flooding rule would yield only **25.33% land** and **4.40% submerged
continental material**.

### Resolution spot-check at sub4

On the independently evolved sub4 500-Myr state:

- scalar flooding: land **22.36%**, submerged continental material **6.63%**;
- sub-grid hypsometry: land **26.32%**, submerged continental material **2.54%**.

This demonstrates that the correction is not merely hiding coarse-grid
pixelation; it fixes the semantics of fractional material itself.

## Interpretation

The remaining ~1.5–2.5% submerged continental material is now plausible as
continental shelves, low/thinned margins, and locally flooded continental
areas. It is no longer dominated by the numerical rule that a whole mixed cell
must be either wet or dry.

Climate remains intentionally out of scope. Future climate work should consume
the exported land/ocean/elevation fields in the separate climate model rather
than be coupled into this tectonic prototype.

## WATCH

- The scalar `TopographyState` is still one mean height per cell. Sub-grid
  hypsometry is used for water capacity and coast diagnostics, but geomorphic
  erosion is not yet independently evolved on the two sub-cell patches.
- Oceanic age is still one age per cell; mixed cells do not yet store an
  independent oceanic-age subfield.
- Sediment transport, flexural loading, dynamic topography, and true crust vs
  lithospheric-mantle separation remain future geological refinements.

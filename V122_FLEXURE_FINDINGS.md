# Moon Tectonics v0.22 — Flexural Isostasy

## Scope

v0.22 adds a passive variable-rigidity elastic-plate response on top of the
v0.21 slab-geometry volcanic-arc model.  Plate dynamics, conservative crustal
material transport, topology, slab memory, rollback, breakoff and continental
magmatic flux are unchanged by the flexure solver.

The implemented weak-form equation is

`(M + K diag(D/A) K / (Delta_rho*g)) w = M h_local`,

which is the finite-volume spherical-mesh form of

`D nabla^4 w + Delta_rho*g*w = q`.

`D = E Te^3/[12(1-nu^2)]` with E=70 GPa and nu=0.25.

## What is flexed

The solver deliberately does not smooth the complete planet elevation field.
Only mechanically supported/load-like relief is sent through the elastic plate:

- Airy relief from departures of continental thickness from the 35-km reference;
- slab-geometry volcanic-arc uplift;
- continent-continent collision uplift;
- trench deflection.

Thermal ocean-floor subsidence, the reference continental datum and ridge-axis
thermal uplift remain local/background terms.  In particular, the several-km
continent-ocean datum contrast is not spread across coastlines.

## Variable elastic thickness

Effective elastic thickness is derived from the already existing chemical crust
plus mechanical mantle-lithosphere thickness:

`Te = 0.25 * H_mechanical`, clipped to 4–50 km.

Rift extension, collision-seam weakness and tidal damage reduce Te locally.
The production v0.22 sub4 run reaches mean Te ≈25.65 km at 500 Myr and mean
flexural parameter ≈67.87 km.

## Synthetic validation

Four dedicated v0.22 tests were added:

1. Te increases with mechanical-lithosphere thickness and decreases with damage.
2. The elastic solve preserves area-weighted mean load/response.
3. A stiffer plate gives a lower local peak and broader neighboring response.
4. The fourth-order Green response contains an opposite-sign peripheral lobe,
   distinguishing elastic flexure from ordinary diffusion/smoothing.

Full suite: **131/131 PASS**.

## Deterministic checkpoint/resume

Fresh sub3 validation:

- continuous `0 -> 40 Myr`;
- segmented `0 -> 20 -> resume -> 40 Myr`.

All 21 NPZ arrays are bit-identical and `meta.json` is exactly identical.
Checkpoint version is `0.22-flexural-isostasy`.

## Exact v0.21 vs v0.22 sub4 baseline, 500 Myr

Same seed (`20260806`), same 5120-cell mesh, same 4-Myr step.

At 500 Myr:

- plate count: **12 vs 12**;
- continental material raster: **28.95% vs 28.95%**;
- all tectonic/material state arrays are bit-identical;
- only `elevation_m` differs;
- max |elevation difference|: **311.86 m**;
- RMS elevation difference: **20.68 m**.

Sea level:

- v0.21: **-808.60 m**;
- v0.22: **-807.90 m**;
- flexural difference: only **+0.70 m**.

Thus the large late sea-level fall is a property of the evolving basin geometry
already present in v0.21, not a flexure-created global-volume artefact.

The sea-level deltas v0.22-v0.21 were +0.26, +0.30, +0.43, +0.47 and +0.70 m
at 100, 200, 300, 400 and 500 Myr respectively.

## Existing erosion WATCH

The legacy topographic erosion step removes positive relief but does not yet
store or deposit the removed sediment mass.  Over this 500-Myr sub4 run its
integrated diagnostic removal is huge:

- v0.21: **442.01 million km3**;
- v0.22: **439.07 million km3**.

Flexure changes that integral by only -2.94 million km3 (~-0.66%), so it is not
responsible for the problem.  The magnitude exposes the next architectural
requirement: erosion must become a conservative sediment-transport layer rather
than a sink of topographic volume.

## Numerical behavior

At 500 Myr in the production sub4 run:

- max instantaneous |flexural correction|: **284.0 m**;
- RMS correction: **21.53 m**;
- mean Te: **25.65 km**;
- mean flexural parameter: **67.87 km**;
- final CG iterations: **5**;
- all steps converged;
- no tectonic/material array changed relative to v0.21.

A canonical sub5 (20,480-cell) 20-Myr smoke test also passes with 12 plates and
no topology/ID consistency failures.

## Verdict

**v0.22 Flexural Isostasy — PASS / stable.**

Flexure is now a passive, physically scaled relief layer with variable elastic
rigidity, deterministic checkpointing and no hidden recalibration of tectonics.
The next required module is conservative erosion + sediment transport/deposition.

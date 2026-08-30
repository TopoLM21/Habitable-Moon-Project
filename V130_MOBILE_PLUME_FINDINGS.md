# v0.30 — Mobile plume sources and bent hotspot tracks

## Research question

How much does slow motion of the deep plume source alter an age-progressive
volcanic track that is otherwise produced by plate motion over a stationary
tail?

Hotspots are not a perfectly fixed reference frame. Reconstructions infer
substantial relative motion among deep-mantle plumes (Konrad et al., 2018,
<https://doi.org/10.1038/s41467-018-03277-x>), and mantle-flow models can
produce both plume drift and hotspot-track bends (Hassan et al., 2016,
<https://doi.org/10.1038/nature17422>). v0.30 therefore replaces the fixed
source assumption with an explicitly testable mobile-source branch.

## Model added in v0.30

- Every plume has a persistent integer ID, a deep-source tangent direction and
  a linear drift speed sampled deterministically from `8–30 km/Myr`.
- The projected source follows a great-circle arc for `80 Myr`, then receives a
  new tangent direction with `0.65` directional memory. This creates coherent
  paths with finite bends rather than a per-step random walk.
- Each active source stores its current arc segment, speed, cumulative path
  length and cumulative bend angle. Population ledgers retain distance and bend
  contributed by sources that later die.
- At the source location the model separately evaluates the deep-source
  velocity, overlying-plate velocity and their relative velocity. The last is
  the instantaneous kinematic direction and speed of an ideal hotspot track.
- Source paths and aggregate kinematics are written to CSV, checkpointed and
  overlaid on the six-panel hotspot GIF.
- Fixed-source mode remains available as the paired control.

This is a kinematic effective model, not a mantle-convection solution. The
piecewise arcs test the consequences of source mobility without claiming that
the chosen paths are unique reconstructions of deep flow.

## Numerical correction for moving narrow tails

A mobile narrow Gaussian repeatedly changes its alignment with mesh
centroids. v0.29 normalized the raw kernel integral, but melt generation uses a
nonlinear `flux^1.4` productivity law. v0.30 therefore preserves the analytic
integral of the powered Gaussian,

`integral(kernel^q) = 2*pi*sigma^2/q`,

with `q = 1.4`. This keeps the physical tail narrow while removing the
centroid-alignment bias from the quantity that actually generates magma.

For the mobile seed `20260806`, subdivision 3 versus 4 gives:

| Quantity | Relative difference |
|---|---:|
| Head productivity time integral | 0.113% |
| Tail productivity time integral | 0.021% |
| Total generated igneous volume | 0.385% |
| Deep-source path length | exactly 0% |
| Deep-source cumulative bend | exactly 0% |

The final retained surface igneous volume and plate count still diverge
strongly between the two resolutions. As in v0.29, nonlinear topology and
material fate are not yet resolution-independent predictions.

## Verification

- `171/171` tests pass.
- A continuous 8-Myr run and a 4+4-Myr checkpoint/resume run are exactly
  identical in all 59 arrays and complete JSON metadata, including the
  individual source paths.
- A 40-Myr end-to-end Windows smoke produced every map, CSV and GIF.
- A 240-Myr demonstration produced 16 six-panel frames and includes conduit
  reorientations at 80 and 160 Myr.
- Ten 500-Myr validation worlds completed: four fixed/mobile pairs at
  subdivision 3 and one fixed/mobile pair at subdivision 4.
- Maximum absolute igneous-ledger error: `2.79e-9 km3`.
- Maximum absolute continental-ledger error: `9.54e-7 km3`.
- Elevation safety-rail clips: zero in every validation world.

## Paired 500-Myr result

Means use four subdivision-3 seeds; the reported effect is paired by seed.

| Quantity | Fixed source | Mobile source | Mobile − fixed |
|---|---:|---:|---:|
| Generated igneous volume (million km3) | 4.399 | 4.391 | -0.008 |
| Retained surface igneous volume (million km3) | 1.195 | 1.148 | -0.048 |
| Maximum rift extension | 0.359 | 0.289 | -0.070 |
| Maximum elevation (m) | 2246.2 | 2191.0 | -55.1 |
| Land fraction (%) | 22.435 | 21.819 | -0.616 pp |
| Plate count | 12.75 | 13.50 | +0.75 |
| Final age–distance correlation | +0.614 | +0.582 | -0.032 |
| Population source path (km) | 0 | 19,086 | +19,086 |
| Population cumulative bend (deg) | 0 | 653 | +653 |

Mobile sources have a mean final active-source speed of `20.07 km/Myr`. At the
same final time, their mean overlying-plate speed is `14.48 km/Myr`, the mean
relative track speed is `16.14 km/Myr`, and source motion deflects the relative
track direction by `72.9°`.

The age–distance response is not sign-definite: paired changes by seed are
`+0.030`, `-0.225`, `-0.024` and `+0.089`. Source drift can either align with or
oppose the contemporary plate velocity. Thus a bent track is not, by itself,
evidence for a change in plate motion, and source mobility does not
systematically destroy age progression in this ensemble.

## Main limitation and next step

The drift chronology is deterministic and checkpointable but is sampled
independently of the simulated mantle-flow field. v0.31 should couple conduit
motion to the low-degree mantle flow already present in the model, retain a
smaller stochastic residual, and compare flow-coupled paths against this v0.30
kinematic control.

# v0.31 — Mantle-flow-coupled plume sources

## Research question

Does replacing the independent v0.30 plume drift with motion guided by the
model's existing Eulerian mantle-flow memory produce coherent, curved hotspot
tracks without breaking exact restartability or the closed material ledgers?

v0.31 is the final model change planned before the canonical long run. It does
not add a second mantle solver. Instead, it lets the plume conduits read the
fixed-grid mantle angular-velocity field that has already been evolved and
checkpointed since v0.10.

## Model added in v0.31

- Each active source samples the fixed-grid mantle angular velocity with an
  area-weighted Gaussian of `550 km` one-sigma radius. Continuous interpolation
  prevents a conduit from jumping when it crosses a cell boundary.
- The resolved source velocity is `0.35` of the sampled mantle-flow velocity.
  The deterministic v0.30 arc remains as an unresolved residual at `0.30` of
  its original speed. The two tangent vectors are added before the source is
  moved along its great-circle step.
- The residual retains its `80 Myr` persistence and `0.65` directional memory;
  the resolved part responds to the slowly evolving mantle field at every
  model step.
- Bend angles now measure changes in the *effective* source direction, not just
  the scheduled residual reorientations.
- Sampled flow vectors, effective source axes and speeds, aggregate histories,
  and the new diagnostic rows are included in checkpoints.
- A dedicated history plot separates resolved-flow, residual and effective
  speeds and records flow/effective alignment.
- Disabling the coupling preserves the v0.30 drift branch exactly.

This remains an effective 2-D projection of deep motion. It is coupled to a
long-lived Eulerian mantle-memory field, not to a 3-D convection calculation.

## Verification

- `174/174` tests pass.
- Continuous `0 -> 8 Myr` and split `0 -> 4 -> resume -> 8 Myr` runs are
  identical in all **62 arrays** and complete JSON metadata.
- A 40-Myr Windows end-to-end smoke generated all maps, CSV files and GIFs.
- A separate 8-Myr canonical subdivision-5 smoke completed two full steps on
  20,480 cells, wrote a resumable checkpoint and finalized every diagnostic.
- Five new 500-Myr flow-coupled worlds completed: four subdivision-3 seeds and
  one subdivision-4 resolution check. The ten v0.30 fixed/independent worlds
  provide the paired controls.
- Maximum absolute igneous-ledger error across the combined table:
  `2.79e-9 km3`.
- Maximum absolute continental-ledger error: `9.54e-7 km3`.
- Elevation safety-rail clips: zero in every world.

## Three-mode 500-Myr result

Means use the same four subdivision-3 seeds.

| Quantity | Fixed source | Independent v0.30 | Flow-coupled v0.31 |
|---|---:|---:|---:|
| Generated igneous volume (million km3) | 4.399 | 4.391 | 4.373 |
| Retained surface igneous volume (million km3) | 1.195 | 1.148 | 1.105 |
| Maximum rift extension | 0.359 | 0.289 | 0.265 |
| Maximum elevation (m) | 2246.2 | 2191.0 | 1842.0 |
| Land fraction (%) | 22.435 | 21.819 | 22.569 |
| Plate count | 12.75 | 13.50 | 12.00 |
| Final age-distance correlation | +0.614 | +0.582 | +0.728 |
| Population source path (km) | 0 | 19,086 | 11,360 |
| Population cumulative bend (deg) | 0 | 653 | 613 |
| Final active-source speed (km/Myr) | 0 | 20.07 | 9.64 |
| Final relative track speed (km/Myr) | 15.10 | 16.14 | 17.55 |

At 500 Myr, the flow-coupled ensemble has mean resolved, residual and effective
source speeds of `12.99`, `6.01` and `9.64 km/Myr`. Effective motion can be
slower than either scalar contribution because these are vectors and can partly
oppose one another. Mean effective/flow alignment is `+0.575`, and source
motion deflects the ideal track from the plate-only direction by `27.4 deg`.

Relative to independent v0.30 drift, the coherent flow-coupled source path is
`40.5%` shorter and its aggregate bend is `6.2%` smaller. Its mean final
age-distance correlation increases by `0.146`. The paired correlation changes
are not sign-definite (`-0.104`, `+0.159`, `+0.123`, `+0.406`), but improve in
three of four worlds. The experiment therefore supports the intended result:
mantle-guided source motion can preserve or strengthen age progression while
still producing curved tracks, and a bend does not uniquely imply a plate
motion change.

The differences in rifting, elevation, land and plate count are nonlinear
ensemble responses. Four seeds are sufficient for a pre-run engineering check,
not for population-level statistical inference.

## Resolution check and inherited limitation

For flow seed `20260806`, subdivision 3 versus 4 gives:

| Quantity | Relative difference |
|---|---:|
| Head productivity time integral | 0.659% |
| Tail productivity time integral | 0.198% |
| Total generated igneous volume | 2.791% |
| Deep-source path length | 2.560% |
| Final age-distance correlation | 1.160% |
| Population cumulative bend | 20.237% |

Source productivity, total path and final age progression meet the intended
coarse/fine engineering check. The endpoint source speed and cumulative bend
do not converge at the same level, and retained surface volume and topology
remain strongly resolution-sensitive.

A separate interpolation-radius diagnostic shows why merely broadening the
Gaussian is not a valid correction: the legacy random plate generator consumes
a mesh-dependent random sequence while constructing plate geometry, so the
formation-era Euler poles—and hence the initial mantle field—are not identical
experiments across subdivisions. This is an inherited initial-condition issue,
not a cell-crossing discontinuity in v0.31. The canonical long run uses one
fixed subdivision-5 world, for which the model is deterministic and exactly
restartable. Cross-resolution source curvature should not yet be treated as a
physical prediction.

## Verdict and next step

**v0.31 is ready for the canonical long run.** The implementation is
checkpoint-complete, ledger-safe, locally visualized and ensemble-smoked. The
next stage is operational rather than another model version: run the canonical
subdivision-5 world in resumable segments, retain checkpoints, and inspect
tectonic activity, material budgets, plume tracks and relief before extending
each subsequent segment.

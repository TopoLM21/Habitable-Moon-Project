# Moon Tectonics v0.11 — 500 Myr material-layer convergence

Seed: `20260806`; `dt=4 Myr`; identical physical configuration except grid subdivision.

| sub | cells | area start | area 500 Myr | volume start | volume 500 Myr | raw collision max | post-redistribution max | plates | largest plate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1,280 | 28.246% | 29.325% | 3.4725e9 km³ | 3.5639e9 km³ | 106.8 km | 42.1 km | 4 | 46.5% |
| 4 | 5,120 | 27.978% | 28.624% | 3.4397e9 km³ | 3.4711e9 km³ | 102.0 km | 42.5 km | 8 | 31.6% |
| 5 | 20,480 | 27.920% | 28.236% | 3.4325e9 km³ | 3.4392e9 km³ | 114.2 km | 46.9 km | 10 | 56.4% |

## Verdict

- The disappearing-continent failure is removed: material area remains close to the initial ~28% at all three resolutions.
- Collision overlap still *tries* to produce >100 km raster stacks, but material-footprint redistribution keeps effective post-collision thickness around 42–47 km.
- At canonical sub5 the 500-Myr continental volume is essentially stable; explicit recycling/delamination remains zero over this interval.
- Pure transport error is floating-point scale (~9.54e-7 km³).
- Plate topology is not resolution-converged: final plate count and largest-plate fraction vary strongly. This is a separate dynamics/topology WATCH, not a material-conservation failure.

## Next model work

1. Keep the v0.11 material layer fixed as the new baseline.
2. Diagnose topology resolution dependence before re-tuning plate-force coefficients.
3. Add a conserved-water-volume sea-level diagnostic and coastline rendering after topography is stable.
4. Later separate crust thickness from total thermal/mechanical lithosphere thickness and add richer density/composition state.

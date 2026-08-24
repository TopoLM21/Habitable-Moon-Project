# Moon Tectonics v0.13 — topology validation

## Scope

v0.13 completes the resolution-independence pass started in v0.12. The goal is
not identical chaotic histories on different meshes, but statistically similar
plate populations and event rates when all thresholds are expressed in physical
units.

## Changes relative to v0.12

- microplate cleanup uses `min_plate_area_km2` plus a 20 Myr continuous
  persistence requirement;
- the persistence clock resets as soon as a plate recovers above the area
  threshold;
- microplate clocks are checkpointed and remapped through topology events;
- a plate that owns zero surface cells is removed immediately by an explicit
  `vanish` invariant event;
- zero-area ID compaction updates `PlateSystem` and `LithosphereState`
  synchronously;
- progressive collision coupling is now velocity-only and is forbidden from
  compacting IDs or changing topology;
- the long runner checks that owner IDs are compact and inside the current
  plate-array bounds at step start and after topology.

## Validation

- 86/86 tests pass.
- New tests cover microplate persistence/reset, zero-area plate compaction and
  the invariant that collision coupling cannot change plate ownership.
- Seed 20260810: monolithic 0→160 Myr and 0→80→resume→160 Myr are bitwise
  identical in every NPZ array and have identical `meta.json`. This interval
  contains a `disconnect_split` at 72 Myr and a `vanish` at 144 Myr.

## 500 Myr ensemble

All runs use the same physical parameters and `dt=4 Myr`.

| subdivision | cells | seeds | mean plate count | median | mean largest plate | mean topology events | mean continental area |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1,280 | 10 | 11.7 | 11.5 | 29.6% | 14.1 | 28.92% |
| 4 | 5,120 | 10 | 11.5 | 10.5 | 37.1% | 13.3 | 28.50% |
| 5 | 20,480 | 3 | 12.3 | 12.0 | 27.4% | 11.0 | 28.47% |

Plate-count ranges are 8–15 (sub3), 8–19 (sub4), and 11–14 (sub5). The
within-resolution seed scatter is therefore much larger than the difference in
mean plate count between resolutions.

Event means per 500 Myr world:

| subdivision | merge | disconnect_split | absorb | vanish |
|---:|---:|---:|---:|---:|
| 3 | 5.2 | 6.9 | 1.9 | 0.1 |
| 4 | 6.1 | 6.4 | 0.8 | 0.0 |
| 5 | 4.0 | 5.7 | 1.3 | 0.0 |

`absorb` remains somewhat more common on the coarse mesh. This is retained as a
WATCH item rather than tuned away, because the final plate-count distribution
is already stable and an additional heuristic could overfit event labels rather
than improve physical topology.

The sub4 sample has a somewhat larger mean largest-plate fraction. Synthetic
cross-resolution split/disconnect/weld tests do not reproduce a rule-level
bias, so this is also retained as a WATCH for a future longer/larger ensemble.

## Decision

Topology is sufficiently resolution-stable to freeze for the next development
stage. Do not retune split/merge/weld parameters while adding hydrosphere/sea
level. Revisit only if a later 1–4 Gyr ensemble shows a systematic resolution
trend in plate count, largest-plate fraction or topology-event balance.

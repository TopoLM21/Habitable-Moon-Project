# v0.31 GUI canonical subdivision-5 run — 500 Myr

## Result

The first GUI-controlled canonical subdivision-5 integration completed from
0 to 500 Myr with 20,480 surface cells and a 4 Myr internal step. The desktop
controller split it into 20 Myr segments. An intentional controller interruption
at 100 Myr was recovered from the last complete checkpoint, providing a real
production-scale checkpoint/resume test rather than only a synthetic smoke.

Local output: `results/gui_runs/v031_canonical_500myr_sub5` (ignored by Git
because it contains about 224 MiB of generated checkpoints and media).

## Acceptance checks

- 25 complete checkpoints at 20 Myr spacing through 500 Myr;
- 303 PNG artifacts and 10 GIF animations (about 40 MB of GIFs);
- 184/184 repository tests pass after the run;
- zero minimum- or maximum-elevation safety clips;
- flexural solver converged at every step;
- final continental ledger error: `4.77e-7 km3`;
- maximum conservative transport error: `9.54e-7 km3`;
- final igneous ledger error: exactly `0 km3` at summary precision;
- maximum hydrosphere error: `13.19 km3`, or `1.17e-8` of the
  `1.1267e9 km3` water inventory.

## Evolution

| Metric | Start | 500 Myr |
|---|---:|---:|
| Plate count | 12 | 6 |
| Mantle temperature | 1850.0 K | 1814.9 K |
| Tectonic activity factor | about 1.0 | 0.856 |
| Continental material area | 27.92% | 28.27% |
| Land area | about 28% initially | 25.06% |
| Sea level | about 0 m | -615.4 m |
| Elevation range | — | -8.88 to +2.83 km |

All six topology events were mergers, at 192, 232, 284, 312, 328 and
332 Myr. The resulting six-plate state then persisted to 500 Myr. The run
recorded four slab breakoffs, 41 slab detachments, 26 final slab zones and 24
active rollback zones.

This is an active, evolving mature-tectonics world, but it did not nucleate a
new rift or split a plate: late-rift nucleation and breakup area both remained
zero. That is a useful scientific result, not a failed run. It means the
current canonical forcing and thresholds preferentially consolidate the
initial plate system. It also confirms that v0.31 cannot answer how the first
plates formed; the upstream genesis/onset pipeline must test that question
explicitly.

## Plumes and surface

The coupled plume sources travelled a cumulative 12,041 km. At 500 Myr their
mean resolved-flow, residual and effective speeds were 5.49, 6.44 and
9.21 km/Myr, with mean flow alignment `+0.686`. The final hotspot-track
age-distance correlation was `+0.765`. Permanent surface plume igneous volume
was `2.826e6 km3`; generated and deep-recycled volumes close at `5.761e6` and
`2.935e6 km3`. Maximum plume dynamic uplift over the run was 971 m.

The final maps show coherent large-scale oceans, continental blocks, mountain
belts, trenches, six plate domains, and spatially transported hotspot chains.
Subdivision 5 substantially improves spatial detail over the earlier
subdivision-3 smoke while remaining practical on the Windows host. Subdivision
6 remains appropriate for selected convergence checks, not as the default
ensemble resolution.

## Interpretation limit

This is one deterministic canonical realization. It validates the local GUI,
checkpoint workflow, numerical ledgers and production resolution; it does not
calibrate the physical probability of a six-plate end state. The next mature
tectonics experiment should be a small paired sensitivity set around topology
split/nucleation parameters. The separate genesis work should follow
`GENESIS_ARCHITECTURE.md` and preserve this run as its unchanged v0.31 control.

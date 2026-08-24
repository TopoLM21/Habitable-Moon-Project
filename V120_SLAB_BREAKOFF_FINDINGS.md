# v0.20 Slab Breakoff — findings

## Scope
v0.20 adds collision-triggered slab necking and breakoff to the v0.19 rollback / trench-migration model. It remains an effective 2.5-D surface+slab-memory solver, not a 3-D mantle rheology model.

## Production mechanism
A remembered slab can enter necking only after the former active oceanic subduction contact has stalled or reclassified **and** the incoming edge of the former subducting plate is sufficiently continental.

Default physical scales:
- continental footprint onset/full: 0.35 / 0.75;
- front sampling radius: 450 km;
- minimum/full slab length: 650 / 1200 km;
- minimum/full slab depth: 250 / 650 km;
- weak/strong-slab target breakoff time: 10 / 22 Myr;
- necking-damage relaxation: 20 Myr;
- post-breakoff tombstone/cooldown: 40 Myr.

Breakoff damage is cumulative and bounded. Brief continental contact relaxes instead of causing instant failure. When damage reaches 1, the slab is marked broken off; active and residual slab pull for that pair become zero immediately and rollback stops.

The breakoff timing range was chosen to remain within the broad 5–30 Myr range found in numerical collision/breakoff studies, with old/strong slabs generally breaking later than young/weaker slabs. The model intentionally does not represent lateral 3-D tear propagation.

## Tests and determinism
- 122/122 unit/regression tests pass.
- New tests cover sustained-collision breakoff, transient-contact relaxation, minimum slab geometry, zero residual pull after breakoff, cooldown expiry, and JSON persistence.
- 0→40 Myr and 0→20→resume→40 Myr are bit-identical in every NPZ array and in the complete meta.json.
- subdivision=5 (20,480 cells) 20-Myr smoke test completed with 12 plates and no topology/ID consistency errors.

## 300-Myr paired validation (subdivision=3, four identical seeds)

| metric | v0.19 | v0.20 |
|---|---:|---:|
| mean final plates | 11.25 | 12.25 |
| mean topology events | 5.25 | 4.25 |
| mean largest-plate fraction | 0.2096 | 0.1689 |
| mean continental-plate speed (deg/Myr) | 0.2268 | 0.2236 |
| mean continental material fraction | 0.29037 | 0.29057 |
| mean continental volume (km³) | 3.5147e9 | 3.5163e9 |
| mean slab breakoffs | 0 | 9.75 |

The new physics leaves mean plate speed and continental mass budget essentially unchanged. Topology follows different chaotic branches, with a modest +1 plate / -1 event shift in this small ensemble; this remains a WATCH for a larger long-run ensemble rather than a reason to tune one seed.

## 500-Myr seed 20260806
- v0.19: 18 plates, 12 topology events, mean continental-plate speed 0.2366 deg/Myr, continental material fraction 0.29390.
- v0.20: 14 plates, 12 topology events, mean continental-plate speed 0.2310 deg/Myr, continental material fraction 0.29574, 15 slab breakoffs.
- The very different final plate partition is consistent with chaotic branch divergence; event count and bulk budgets remain close.

Most importantly, breakoff reduces the population of slabs stuck on artificial 2.5-D caps:
- length-cap zones: 49 → 40;
- depth-cap zones: 57 → 44.

This does not eliminate cap saturation because long-lived purely oceanic subduction still lacks transition-zone stagnation/penetration physics.

## Remaining limitations exposed by v0.20
1. No mantle transition-zone state (410/660-km interaction, stagnation vs penetration).
2. No lateral 3-D tear propagation after first breakoff.
3. No explicit post-breakoff asthenospheric upwelling / magmatic pulse.
4. Volcanic arc position is still too tightly tied to the surface trench rather than slab dehydration depth.
5. Flexural loading is still local/Airy-dominant rather than elastic-plate flexure.

## Recommendation
Freeze v0.20 breakoff unless a larger multi-seed 500-Myr ensemble shows a systematic topology bias. The next high-value module is slab-depth-controlled volcanic arc geometry; flexural isostasy should follow soon after. A mantle transition-zone module is valuable before multi-Gyr production runs, but it is a larger step than arc geometry.

# Fractional transport: existing coverage and experiment contract

This stage is an independent experimental material-transport pipeline. Mechanics0.5 and its archived trajectories remain unchanged. The proposed comparisons freeze plate angular velocities and thermal/mechanical loading; they are not new plate-speed predictions or a mature-evolution integration.

## Existing0.5 baseline

`baseline.py` reads the already completed0.5 saves, reconstructs acceptance from persisted thermal cohorts, aligns it with post-transport history, and records input hashes. It also writes a test-function catalogue with source line numbers. No evolution is launched.

| Existing arm | Mean speed at50 Myr, mm/yr | Display-convergent boundary first present, Myr | First raster commit and acceptance, Myr | Accepting steps with no raster commit | Zero-acceptance steps after onset |
|---|---:|---:|---:|---:|---:|
| sub4, dt1 | 2.68435 | 8 | 16 | 0 | 9/35 |
| sub4, dt0.5 | 3.34911 | 7 | 15.5 | 0 | 12/70 |
| sub5, dt1 | 3.49826 | 8 | 11 | 0 | 3/40 |

Display classification is not an oracle for weak-boundary flux: the production material law must use signed local relative normal motion and actual material polarity. The observations establish that existing acceptance is gated by raster transport despite already resolved convergent motion. The old traces sample inventory before the current material transaction; saved cohort timestamps provide the independent endpoint acceptance timeline.

`baseline.json` additionally preserves the existing50/100/200/400 endpoints and last100 Myr statistics. It does not repeat the expensive400 Myr run.

## Existing tests to preserve or extend

| Existing file / test | Actual coverage | Missing fractional contract |
|---|---|---|
| `tests/test_conservative_transport.py`: one-to-one assignment, area-scaled continental parcels | Every donor appears once in an individual plate's raster assignment; unequal source/target areas preserve volume | A material can occupy multiple targets and multiple plates can occupy one cell; assignment uniqueness does not prove a fractional budget |
| Same file: `test_subcell_rotation_accumulates_until_material_moves` | Residual quaternion eventually reaches a raster commit | New positive physical acceptance must occur before that commit when convergent flux exists; old test remains valid for the legacy raster path |
| Same file: topology residual remap | Split children inherit unresolved parent rotation | Sparse cohort identities and all extensive amounts must survive relabel/split separately from quaternion memory |
| `tests/test_young_boundary_material.py:188` raster loss transaction | Lost whole-source basalt equals acceptance, surface subtraction and explicit sink; no phantom force | Partial withdrawal, donor remainder, repeated steps and multiple outlet edges need independent tests |
| Same file: duplicate transaction, checkpoint/resume, failure/remesh | Once-only accepted ledger, conservative detached history, old source not mutated | One donor may legitimately contribute several fractions; validate summed budget rather than reject all repeated source identifiers |
| Same file: ordered layers, equal-time deep transfer, refinement | Thermal cohorts retain their identity, order and mass in the slab inventory | Surface-side histories must remain distinct before acceptance too; averaging cell age/damage is insufficient |
| `tests/test_continental_material_transport.py` | Conservative continental footprint and overflow | Experimental importer should explicitly reject unsupported continents; do not silently use a winner-only continental consumer |
| `tests/test_transport_cycle_memory.py` | Felsic history follows one winning material-source index | A single source index cannot represent mixed fractional parcels; old consumers must remain outside the new experiment |
| `tests/test_genesis_starter_ledger.py` | Independent source/sink volume audit, unrecorded errors detected, resume counts only new records | The global basalt ledger cannot detect repeated cold-footprint or density-excess acceptance when basalt alone was debited |
| `tests/test_genesis_continuation_remesh.py` | Extensive vs intensive field remapping, preserved clocks, memories and histories | Every fractional material id must split conservatively; surface fractions must sum to physical cell area after refinement |
| `tests/test_genesis_sinking_compatibility.py` | Persisted versions/force laws, populated frozen/runtime equality, no silent upgrade, endpoint clocks | New transport experiment must not relabel0.5 checkpoints or imply compatibility with mature winner-only evolution |
| `tests/test_genesis_continuation_execution.py` | Actual saved-state CLI resume, CPU parity and rendering independence | New sparse snapshot needs explicit roundtrip and restart equivalence before mature integration |

Exact function names and current line locations are in `test_catalogue.json`. This is a coverage map, not a claim that all tests have been rerun in this new stage.

## Why the donor budget must cover more than basalt

The0.5 `raster_acceptance_events` computes accepted area from the whole source cell's oceanic footprint, then cold volume as area×H and density-excess mass as cold volume×Δρ. These are independent of the remaining `oceanic_volume_km3` in that cell. A patch that subtracts fractional basalt now and later invokes the old whole-cell acceptance can therefore accept the footprint/cold mass twice while the basalt ledger still balances.

The experimental representation should carry each material id's area, basalt volume, cold mantle volume and density-excess mass together. All four must be partitioned by the same withdrawal fraction. Distinct material age, cold thickness/history, fracture damage/stress/strength and provenance travel with each parcel. Donor identities follow parcels, not fixed Eulerian cell indices; a newly created ridge parcel receives its own identity.

## Independent small experiments after the API is ready

1. **Actual source imports.** Load the original Starter and saved0.5 at50 through the source integrity checker. Import the actual lithosphere and fracture memory into sparse parcels. Confirm sums and histories; record source/config hashes. Snapshot import must be side-effect-free. Full400 Myr evolution is unnecessary.

2. **Pre-raster motion.** At fixed source velocities, compare one short fractional step with `build_transport_map` on a copy of the saved raster state. Also record a clearly labelled zero-residual raster control, if the saved residual was already close to committing. When independent signed relative flux is positive and the raster control has zero commits, require positive fractional acceptance and exactly matching donor subtraction. Do not lower the old raster commit thresholds to produce the result.

3. **Material-id budget.** For each extensive quantity q and each material id, verify `initial(q)+newborn(q)=remaining(q)+accepted(q)`. Include simultaneous withdrawals through several edges, repeat steps and a donor close to exhaustion. Require nonnegative remainders; the flux limiter must limit aggregate withdrawal, not each edge independently. A later raster-only probe must not mutate the fractional snapshot or generate another accepted transaction.

4. **Shared rigid rotation control.** Replace all plate velocities by one identical finite angular velocity. Transported surface material should move, while subduction losses and ridge births remain zero to arithmetic tolerance. Per-cell area closure and all global extensive inventories must hold. This separates material advection from relative plate convergence and exposes spurious sources from a non-solenoidal edge discretization.

5. **Time refinement.** Compare dt1, dt0.5 and dt0.25 over the same short physical horizon and frozen velocities. Report cumulative accepted area/volume, births, material-id budget residuals and spatial distribution errors. Equal-time snapshots with different integration schedules need numerical convergence, not bitwise equality. Creation/acceptance times are part of the measured discretization error.

6. **Grid refinement of the same state.** Refine the same sparse snapshot and its material ids conservatively, then apply the same angular velocity vectors and horizon. Compare integrated source/sink volumes and parent-aggregated amounts, not an independently evolved coarse/fine speed. Keep any reconstruction error separate from changed physics.

7. **Snapshot restart.** On one fixed schedule, compare uninterrupted transport with serialize→load between steps. Require exact parcel identity/history preservation and deterministic state/flux results. Corrupt or incomplete snapshots must fail before replacing a valid state. Source arrays and files must remain unchanged.

The new kernel's per-step report should retain acceptance/birth substep times, donor identities, source/receiver plates, all extensive transfer amounts, CFL substeps and budget residuals. The transport experiment has no claim about the eventual slab force, tensile failure rate, or speed until the new representation is connected conservatively to all required consumers.

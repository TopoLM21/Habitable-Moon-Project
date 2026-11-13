# Reconstruction from the available repository, 13 November 2026

Base: `831e6e3e0d58e352efab575f561aa677595a4b7f` (September v0.31).
Working branch: `reconstruction/regional-20261113`. Main is unchanged.
The supplied project conversation is the source of historical statements below;
it is not an executable archive. This is new work, not recovered original code.

## What survives

The v0.31 entry point uses `run_long_evolution_v123.main` through a chain of
version wrappers. Its main loop advances `thermal.advance_thermal_state`,
`mantle.advance_mantle_flow`, `dynamics.update_plate_dynamics`,
`lithosphere.advance_lithosphere`, the continental cycle and topology manager.
`transport.build_transport_map` carries residual rotations and performs discrete
parcel assignments. Continental volume, sediments and igneous products have
ledgers. Topology includes rifting, disconnected areas, welding and microplate
handling. Native checkpoints preserve these fields. Genesis is a documented
upstream architecture, not the missing young runtime in this checkout.

The force coefficients explicitly are NOT SI forces (`tectonics/dynamics.py`).
Cooling scales their prescribed activity; mechanical work is not derived from
that heat reservoir. The split operator can add differential angular velocity
without an elastic energy debit. A continental volume ledger is not a complete
mass/composition/enthalpy ledger. These facts prohibit calling the old coupled
loop an energetically closed model of plate formation.

## Established from the supplied history, absent on disk

The lost October–November branch explored material boundaries, finite reservoirs,
thermal/viscous material cohorts, isostatic and elastic potentials, energetic
candidate tests and fixed regional event quadrature. Detailed contact/bridge
models became reference controls; coarse regional evolution was the agreed main
direction. Do not reintroduce microscopic seam resolution as the main product.

The last recorded experimental 1 Myr regional runs took 148–205 seconds and did
not create new regions. At the borderline amplitude 0.60, the 0.35-radian model
accepted a cut rejected by the detailed reference (release approximately
2.4353e23 versus 2.0567e23 J; cost 2.3014e23 J). Thus event convergence was NOT
established. These are historical reports, NOT new measurements.

An old mechanical test lacked `thermal_error_control`. A reported compatibility
patch added numerical controller defaults; the final paid-event continuation,
tests and summary queue never had a confirmed outcome. None of their saved
states, test logs, maps or final source changes exist here. Commits including
`460f15981d31e57362fbf0742e4bc47108c0b46c` are absent from this checkout's objects.
The historical seven baseline failures cannot be assigned to this older tree.
Unknown exact constitutive implementations, geometric mappings and coefficients
must not be silently guessed, nor can old protected hashes be re-certified.

## First bounded stage and integration route

1. Add an opt-in, restartable thermal integrator in the EXISTING global loop.
   Reuse the existing constitutive heat-loss law and coefficients, replace the
   numerical update only. Account separately for radiogenic input, prescribed
   orbital input and surface loss; never balance a residual by adding heat.
   Preserve small energy increments and controller memory. Verify analytical
   cooling, zero/source-only limits, failure rollback, exact terminal time,
   tolerance sensitivity and full-run restart. Keep v0.31 defaults unchanged.
2. Next attach explicit finite mantle/crust material inventories and specific
   enthalpy to real transport, creation and recycling requests, including age
   and provenance. Audit oceanic parcel reset/overlap losses before converting
   the old geometric raster into inventory authority. Give each exchange a
   donor, receiver and energy debit; reject exhaustion instead of replenishing.
3. Replace proxy forcing in an opt-in regional mode with a dimensional,
   finite-potential force/work interface; carry per-material viscous memory.
   Only then reconstruct paid topology events. Reference/coarse disagreements
   near threshold are explicit uncertainty, not parameter-tuning opportunities.
4. Compare 1–100 Myr scenarios, grid/time sensitivity, material/energy exchange
   scales, restart and measured speed; add an experimental viewer. A label split
   or fixed seed geometry is not physical birth of continents.

Stage 1 controls ONLY thermal integration error. It cannot guarantee fracture
timing, close missing crust/mechanical energies, infer mantle composition/water/
redox, or reproduce the lost paid-cut result. No new spatial convection, melting
law or dynamic planetary radius is introduced. The whole-mantle reservoir is
finite, but prescribed orbital heating has an external source; orbital decay is
not evolved. Radiogenic coefficients and the existing force scalings remain
prototype parameters, not validated lunar properties.

The new thermal update fails outside Tmantle >= Tsurface + 1 K rather than using
the legacy numerical +100 K floor. This changes only the optional mode's domain
handling; no heat is injected to keep it in range. No phase boundary is certified
by this effective single-temperature law.

## Preservation

`baseline_manifest.json` fingerprints all pre-existing tracked files before
edits. Existing archives/checkpoints/results are read-only for this work. New
compact logs, parameters, diagnostics and restart examples are versioned on the
new branch along with code. A result manifest records its source commit. Every
push must be checked against the remote branch SHA; no force push or main update.

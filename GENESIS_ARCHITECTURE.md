# Genesis architecture after v0.31

## Continuing young-shell fracture inside mature dynamics, 2026-09-27

Continuation format 0.2 keeps the starter material damage law alive after its
first partition. `genesis_starter_fracture.py` carries cooling/water/damage and
an event-use latch through the actual mature material donor map. It supplies
valid all-oceanic or boundaryless-shell cuts inside the topology transaction,
without invented rift age, material, velocity kicks or a target plate count.
The duplicate mature tidal damage source is disabled only in this continuation.
`genesis_starter_slab.py` gates mature slab pull with actual convergence-integrated
slab length, so no slab means zero pull but developed slabs are no longer
permanently disabled. This length normalization is an explicit model closure.
The separate fracture checkpoint is integrity checked with both existing states;
older 0.1 continuations must restart from their original starter checkpoint.
See `GENESIS_STARTER_FRACTURE.md` for event timing, validation and physical limits.

## Coarse starter continuation through the actual v0.31 runner, 2026-09-27

`genesis_starter_continuation.py` imports the starter's actual domains into
the existing mature loop. The first solid surface reservoir is explicitly
assumed mafic; its mass was already excluded from the global mantle reservoir.
Mechanical lid depth is not renamed chemical crust. Independent mantle flow,
damage, canonical mesh and absolute clocks are retained; no procedural plate
or continent initialization occurs. Genesis remains the sole heat/orbit/water
owner during this young continuation, including condensation after partition.

The optional starter GUI/CLI path executes a bounded mature segment with both
checkpoints retained. `genesis_starter_ledger.py` independently compares material
process counters with upper-reservoir changes using a stated reference density.
This is an experimental rigid-domain representation, not physical handoff
certification or a validated 4.5 Gyr run. See `GENESIS_STARTER_CONTINUATION.md`.

Validation: 1789 regression tests pass; five actual runner segments pass their
clock/water/heat/material checks. A 1280-cell 3 Myr run is bitwise identical to
1 Myr plus a resumed 2 Myr segment. At 5120 cells, a 10 Myr continuation takes
about 12 s including outputs, but mean surface speed is only 4.42 m/Myr and
no raster transport or new continental material occurs. Halving the mature
step barely changes that speed. Coupling works; sustained mobility remains
the unresolved physical target, not a demonstrated outcome.

## Coarse molten starter and shared mature topology, 2026-09-27

The primary implementation path is now `genesis_starter.py`: an initially
molten, flat, unpartitioned sphere with shared heat/orbit evolution and an
explicit effective mantle/differential-cooling/tidal damage law. Seed controls
smooth loading/strength heterogeneity, never a prescribed plate count.
`genesis_starter_loading.py` resolves inexpensive thermal forcing inside Myr
output intervals; its conductive lid is not differentiated chemical crust.

`genesis_starter_topology.py` accepts already weakened separating bands even
on a boundaryless global shell. Both it and the mature `_attempt_split` reuse
the same ownership/rotation operator. Mature eligibility and behavior remain
unchanged. Starter cuts never cross ineligible material; no synthetic age or
rift memory is created, and its differential-velocity kick is zero.

The new CLI and GUI stop at the first candidate partition. They preserve the
thermal/damage state and diagnose a mantle-flow Euler fit, but do not certify
rigid motion, erupt magma, create continents or launch mature evolution. The
next integration is transfer of actual material/thermal/water/weakness state
and a bounded mature continuation. Detailed contact research remains available
but is no longer a prerequisite to this coarse path. See `GENESIS_STARTER.md`.

Seven bounded controls and 1702 regression tests pass. On 5120 cells, the
default 20 kPa case retains one shell through 10 Myr (7.35 s model time);
an explicitly stronger 50 kPa experiment partitions at 0.99062 Myr (1.50 s).
This is sensitivity evidence, not calibration or proof of mobile plates.
Full provenance, step/mesh comparisons and limits are in `GENESIS_STARTER.md`
and `results/genesis_runs/starter_validation_20260927/validation.json`.

## Moving local contact with continuing solidification, 2026-09-27

`genesis_moving_contact.py` continues the thermally born extrinsic contact on
an updated parent background, retaining the assumed-strain discretization and
a persistent, objectively transported local enrichment. Current geometry owns
force and drag assembly; material vertex IDs own the active interval. This is
a finite-background, small-local-opening formulation, not general finite
sliding or a finite-compatible split-surface element.

`genesis_moving_contact_cohorts.py` preserves the original extrinsic section
and adds separate stress-free ordinary cohorts for newly solidified material.
Initial material areas/history are never rescaled; fresh material across an
already open gap remains unbonded. Material fronts use the minimum bank depth
in birth reference coordinates. Remelting is explicitly unsupported.

The thermal analysis conservatively subdivides columns at the saved birth,
then continues heat, Maxwell memory, orbit and tides. It monitors held-vertex
strength and stops at the next localized event instead of retaining a
supercritical tied constraint indefinitely. Paired 1000-year continuations
with steps of 100 and 50 years and 1556 regression tests pass. Extended runs
locate the same next tensile event at 2326.59 / 2327.52 years after birth,
282.56 km from the first active interval. Independent held-reaction recovery
confirms a separate birth candidate: releasing the intervening subcritical
material by enlarging the first interval is not justified. Work quadrature
and geometric contact work remain explicit, nonzero diagnostics. See
`GENESIS_MOVING_CONTACT.md`; propagation, additional birth and GUI/mature
handoff remain separate work.

## Moving tied shell with physical-time force balance, 2026-09-27

`genesis_moving_tied.py` now solves each physical step on updated material
geometry with corotational Hencky increments, inherited Maxwell memory and
current-area basal drag. Its Newton line search checks forces on every moved
candidate. It does not invoke the static Mobile equilibrium solver, erase
accumulated strain, or move an active contact. Geometry, last-step motion/load/
drag, elastic memory and work ledgers have a self-contained checkpoint format.
Absolute force-resolution allowances for floating geometry are exposed alongside
raw force residuals; the normal force tolerance is not a claim of arbitrary
accuracy for vanishing motion.

`MaterialPathSupport` transports the original support connectivity/barycentric
ancestry and retains initial child material fractions. Moving path observations
inherit current parent forces and isotropic elastic energy without repartitioning
material according to changed areas. In the 5120-cell thermal experiment this
passes the former accumulated-reference-strain guard and reaches a local tensile
strength event, where the existing force-consistent birth law can be inserted.
The path is still prescribed; moving active interfaces and crack networks remain
outside this tied owner. See `GENESIS_MOVING_REFERENCE.md` for results, time-step
controls and the explicit nonzero work-quadrature remainder.

## One traction-preserving local contact birth, 2026-09-26

`genesis_extrinsic_contact_law.py` supplies an initial-stress contact potential
with zero insertion energy, unchanged tensile peak, and post-birth monotone
Mode-I work equal to Gc. Its prestressed unloading rule is an explicit research
constitutive assumption. Signed shear reference energy and return-mapping
remainders are separate from physical fracture work and heat.
`genesis_path_activation.py` selects one minimally resolved support vertex at
an admissible strength event, replacing its measured tied reaction without a
force impulse. It supports subsequent fixed-depth mechanics and self-describing
birth checkpoints, but not propagation or a second event. The ordinary path
law and fingerprints remain unchanged; validation additionally rejects invented
irreversible history on permanently tied traces.

A thermal-gate diagnostic enters before the old instantaneous static solve,
retaining the genuinely unstressed earlier shell. Current solid interface area
must be used in traction recovery: a frozen thin-shell section falsely predicts
early failure. With that correction the thermal experiment reaches the fixed
reference strain guard before onset. The controlled contact-birth experiment
therefore remains distinct from a completed physical genesis run. See
`GENESIS_PATH_ACTIVATION.md` for the constitutive choices, evidence and limits.

## Tied traction onset and opt-in local geometry checks, 2026-09-26

`genesis_path_birth.py` reconstructs tied interface tractions with an explicit
minimum-correction convention, checks the unchanged tensile/frictional limits,
and separately bounds reaction capacity without selecting endpoint tractions.
`genesis_path_onset_event.py` locates a sampled threshold crossing using pure
one-step trials from the same accepted state. It refuses already-supercritical
starts and does not initialize a cohesive law or commit physical activation.

The old source motion stop compared points 515 km apart. Opt-in
`PathGeometryParameters` now checks local deformation gradients, rotations,
finite/linear strain differences and corotated contact jump errors instead of
global maximum displacement divided by the global shortest edge. Existing
strain, sliding and penetration limits remain; reference geometry is never
reset. Default historical behavior and fingerprints remain compatible.
See `GENESIS_PATH_BIRTH.md` for source evidence and the remaining birth-law
and fixed-reference limitations. Automatic mature handoff is still unchanged.

## Embedded path physical-time kernel, 2026-09-26

`genesis_path_basis.py` retains the original background mechanics while adding
area-balanced relative bank motion on the inserted support. It is an explicit
assumed-strain enrichment, not ordinary conforming refinement; constitutive and
displayed geometric strains are both guarded. Tying the new DOFs reproduces the
old stiffness, prestress, forcing and drag. `genesis_path_dynamics.py` advances
Maxwell/cohesive/contact mechanics under supplied physical-time loading, with
rejected-step rollback and a separate mechanics-only checkpoint format.

Several tied steps agree with the existing mechanics under shared thermal/orbit
loading. Prescribed release tests include nonlinear history, temporal refinement
and exact restart. Fresh zero-gap cohesive contact cannot balance the old tie
reactions, so release remains an explicit intervention, not detected nucleation.
Automatic birth/history conversion, changing interface depths, finite motion
and mature handoff are still not promoted into the new kernel. Production path
selection and previous checkpoint formats are unchanged. See
`GENESIS_PATH_DYNAMICS.md` and the source validation artifacts it references.

## Continuous material-interface insertion, 2026-09-26

`genesis_path_mesh.py` now inserts an explicit open minor-arc polyline through
material cell interiors, retaining shared intersections, parent-face ancestry
and prospective front positions on one common mesh. `genesis_path_material.py`
partitions mass/heat/work and objectively transports elastic/weak-plane memory.
The saved 1532 km strong-source ridge gives 5176 child faces from 5120; cutting
this prescribed open support produces no detached two-cell chips. This is a
new diagnostic backend, not a replacement for production onset/selection.

Zero-cut remeshing is explicitly audited: conserving initial material energy
does not preserve discrete equilibrium. The actual refined source includes
slender triangles and its static relaxation violates the motion/reference-edge
limit. Those changes are not counted as fracture release. Same-refined-mesh
contact comparisons pass on 320/1280/5120/20480 controls, but the final energy
change remains 19.9%, so convergence is not claimed. Production integration
requires controlled activation of the new mechanical degrees of freedom and
consistent diffuse-to-contact history transfer. See `GENESIS_PATH_INSERTION.md`.

## Physical time continuation and equilibrium correction, 2026-09-25

The default 20 kPa, 320-cell run now reaches 30 Myr without activating a
weak plane: residual damage decays as load remains below wet tensile strength.
The 50 kPa coupled continuation reaches 102 589 physical years after its
1.4 Myr source, with 7.50 km opening and 4.93 km slip, before the accumulated
reference strain reaches the supported 0.5% limit. The cohesive shell remains
connected. A fresh restart reproduces the stopping state bitwise. This is a
numerical geometry limit in one coarse scenario, not physical tectonic arrest.

The earlier frozen alpha=0 test removed external mantle loading while retaining
prestress. Its large unload response did not establish an invalid source state.
All four intact source states satisfy their original force tolerance. A torque
normalization bug in near-balanced frozen loads is fixed; an independent audit
now reconstructs actual accepted coupled-step forces, including basal drag and
Maxwell memory, and verifies the committed work remainder. Guard utilization
is exposed in normal coupled diagnostics without resetting accumulated motion.
See `GENESIS_LONG_TIME.md` for evidence, limits and reproducible commands.

Longer waiting does not repair the known edge-path extraction defect. Physical
front/interface insertion and a consistent transition from diffuse shear to
bank contact remain the immediate work. The observed reference limit also
requires a justified way to advance geometry if persistent regions have not
formed before it; merely increasing the limit is not a solution.

## Unilateral shell contact and guarded loading, 2026-09-25

`genesis_unilateral.py` adds reversible frictionless nonpenetration to the
frozen-shell virtual-extension experiment. Complementarity, force balance,
material nesting and the contact contribution to released energy are checked.
Opening reproduces the free-bank solution; pure symmetric compression no
longer creates the false release caused by interpenetration. Load continuation
holds prestress fixed and preserves total-reference geometry limits. Strong
sources with the manually imposed notch violate those limits in the
prestress-only unload experiment; this does not invalidate intact source
equilibrium. Numerical load factors do not advance time.
See `GENESIS_UNILATERAL.md`. Physical front evolution still needs a valid
starting mechanical state, integration with the existing evolving solver,
continuous interface insertion and consistent fracture-energy ownership.

## Reversible shell crack-extension energy, 2026-09-25

`genesis_shell_release.py` now compares fixed-load elastic equilibria on
nested explicit edge-cut meshes, using the existing membrane stiffness and
inherited prestress. Material-corner prolongation verifies stiffness, loading
and area-weighted rotation-gauge preservation. Potential release is checked
independently through elastic relaxation energy. Compression/interpenetration,
large reference motions and detached free components cannot authorize growth.
The matched 320/1280/5120/20480 control retains a 4.87% finest-grid change for
a fixed finite extension, so full mesh convergence is not claimed. Real source
probes preserve their checkpoints and explicitly assume a large initial notch;
they do not establish nucleation. See `GENESIS_SHELL_RELEASE.md`. Contact-aware
virtual extension was added in the subsequent stage above; continuous interface
insertion and consistent fracture-work ownership remain ahead of propagation/handoff.

## Seeded energetic front control, 2026-09-24

An immutable reference path and irreversible front history now have an
independent displacement-controlled DCB validation. Finite extensions spend
released elastic energy on fracture; excess release remains an explicit
unresolved ledger. Joint load/extension refinement reduces external-work
error from 4.89% to 0.346%; unloading/reloading and exact restart pass.
This is an analytical specimen backend with an explicit initial notch and
one advancing tip, not a shell fracture-energy estimator or nucleation law.
See `GENESIS_CRACK_FRONT.md`. The subsequent shell energy control is
described above; conservative interface insertion and
unambiguous ownership of cohesive fracture work remain required. Existing production
selection, checkpoint formats and handoff blockers remain unchanged.

## Seeded localization geometry, 2026-09-24

`genesis_ridge.py` now tracks a supplied transverse scalar maximum as a
continuous spherical polyline, without cutting mesh edges. Analytic controls
show reduced mesh error for a straight band, explicit underresolution/constant
field rejection, and a remaining support-scale bias for curved bands. A static
1532 km accumulated-shear trace was recovered from an explicit seed in the
5120-cell strong source. This is not nucleation or physical crack growth;
production contact selection and checkpoints are unchanged. The subsequent
front benchmark above validates history/energetics on a specimen; contact-aware
shell energy evaluation and conservative interface insertion remain required.
See `GENESIS_RIDGE_TRACKING.md` for the measured limits and artifacts.

## Current blocking evidence: crack geometry, 2026-09-24

The matched 320/1280/5120-cell sources exposed a failure of the legacy edge
selector: its 35-degree cones cut two triangular edge families and create
two-cell rhombi even in a smooth, uniform weak-plane control. Geometry and
the first 100 years of contact motion are mesh dependent. Connectivity now
separately counts cut components and surviving cohesive/vertex links; neither
is a plate count. Historical selection and restart remain unchanged.
The next step is validated localization/path tracking before interpreting
coupled fragments as persistent plates. See `GENESIS_SEAM_EXTRACTION.md`.

## Current implementation priority: reuse v0.31, 2026-09-24

The active v0.31 already provides plate transport/topology, oceanic crust
generation, subduction/collision, continental growth/recycling/cratons, mantle
flow/plumes, relief, sediments and the ocean. Genesis must supply justified
initial conditions and transfer state into these existing mechanisms. Earlier
stage descriptions below are historical snapshots, not a requirement to build
a second complete tectonic solver before handoff. The origin/onset matrix and
full spin-orbit program describe broader research extensions; the explicitly
synchronous scenario can continue under its stated assumptions.

The immediate path is: coupled early shell evolution and persistent regions;
conservative transfer of actual crust/material/thermal/water/mantle state;
targeted compatibility fixes in the existing mature solver; then end-to-end
continuity, restart and long-run validation. Full finite-sliding contact is
needed before handoff only if evidence shows plate kinematics still cannot
represent the relevant early motion.

`run_genesis_handoff.py` now observes consecutive contact states and reports
rigid-rotation residuals without turning connected components into artificial
plates. The current 318+2-face result fails that screen. An explicit-state
experimental v0.31 adapter and optional oceanic-volume ledger prepare the
receiving side; they do not certify a physical handoff. Source enthalpy/mass
archives are preserved but not evolved by v0.31. See `GENESIS_HANDOFF.md`.

## Thermal prototype, 2026-09-20

An independent two-reservoir magma/steam/ocean experiment is now available via
`run_genesis.py` and the GUI's thermal-genesis dialog. See `GENESIS_THERMAL.md`.
It has conserved water/energy and its own resumable checkpoints, but no
spin-orbit solution, spatial onset mechanism, or v0.31 handoff. Its lid threshold
and cooling length are diagnostics, not an accepted onset gate. The A/B/C
origin choices below remain disabled; the full acceptance gates still apply.

## Spatial shell prototype, 2026-09-21

`run_genesis_shell.py` and the same GUI dialog now add passive enthalpy columns,
local solid-layer thickness, a freely contracting triangular membrane,
temperature-dependent Maxwell relaxation, prescribed basal shear, and tensile
damage/healing. See `GENESIS_SHELL.md` for controls, checks, and examples.
The thermal history forces the spatial model in one direction; the separate
column energy ledger is not added to the global inventory. A joint NPZ saves
both states. Uniform unconstrained cooling produces no tensile damage.

Damage contours and connected intact regions are diagnostic candidates, not
mobile plates. Basal shear is prescribed, not a computed mantle circulation.
The fixed-geometry membrane stops when a principal total strain exceeds 5%.
Neither continental differentiation nor a conservative v0.31 handoff is
implemented. The onset gates below remain requirements for subsequent work.

## Orbit-assisted onset diagnostic, 2026-09-22

`run_genesis_onset.py` adds an explicitly synchronous, zero-obliquity,
small-e eccentricity-tide experiment using the saved giant/moon orbit.
Isolated eccentricity damping supplies the global heat budget from lost
orbital energy. Phase-averaged tidal overstress, liquid-water access weakening,
and a physical Helmholtz length modify the shell damage update. The solved
membrane displacements now move reference-material markers, with incremental
velocities and rigid-region fit diagnostics. See `GENESIS_ONSET.md`.

This does not satisfy the full spin-orbit/onset/handoff gates: spin locking,
giant tides and resonant forcing are absent; water access is a constitutive
proxy; first displacements retain the original 5% small-strain limit. Finite
material transport, contact, subduction and continental differentiation remain
required before a conservative mature-model handoff can be enabled.

## Moving material shell, 2026-09-22

`run_genesis_mobile.py` and GUI mode «Движущаяся оболочка» now evolve the
actual material mesh with fixed layer masses, variable column thickness and
corotational Maxwell memory. Nonlinear equilibrium is checked on the moved
geometry; incremental, elastic and mesh guards replace the fixed-reference
5% stopping rule only in this new mode. Rejected adaptive steps roll back all
ledgers. Thermal formation precedes activation of a continuous membrane.
See `GENESIS_MOBILE.md` for controls, numerical comparisons and examples.

The topology stays connected: no plate contact, subduction, remeshing,
continental differentiation or mature-model transfer has been enabled.
Mechanical work is not thermalized and the canonical global thermal/orbital
radius remains a one-way approximation to the changing mechanical radius.

## Frictional shear zones, 2026-09-22

`run_genesis_faults.py` and GUI mode «Трение и сдвиг разломных зон» add
persistent material weak planes, pressure-dependent friction, water-access
weakening and irreversible nondilatant shear. The returned stress enters the
same moved-geometry force balance; a consistent, generally nonsymmetric
material tangent supplies Newton corrections. Activation, orientation, shear
history and per-cell work ledgers survive checkpoints and roll back on retries.
See `GENESIS_FAULTS.md`.

This is a finite-width constitutive extension of the connected shell. Equivalent
slip is accumulated shear times an explicit width, not a displacement jump.
Normal compression comes only from the membrane stress; water availability
does not supply a pore-pressure solution. No detached banks, opening, contact,
subduction, continental differentiation or mature-model handoff is enabled.
Frictional and viscous work are reported separately and not added to heat.

## Split-bank contact continuation, 2026-09-23

`run_genesis_contact.py` continues a `fault_checkpoint.npz` in a separate,
bounded mechanical window. Face-corner fans create independent bank nodes;
paired endpoint traces carry irreversible Mode-I cohesion, unilateral penalty
compression and Coulomb/viscous sliding. Physical basal drag regularizes
fragment motion without pinning every fragment. Layer masses, enthalpies and
the complete source snapshot are preserved in an independent checkpoint.

Thermal/orbital fields, diffuse damage, water and Maxwell memory are frozen;
bulk increments use reference-frame elasticity with strict motion guards and
a source-dependent omitted-relaxation limit. This contact experiment does not
yet feed back into the long-term genesis integrator or enable its mature-model
handoff. Fixed original pairs limit it to small sliding, with explicit penalty
penetration and energy remainder diagnostics. See `GENESIS_CONTACT.md`.

## Coupled cooling and split-bank evolution, 2026-09-23

`run_genesis_coupled.py` now advances the original fault snapshot's physical
thermal/orbital clock alongside conductive columns, incremental Maxwell memory,
water access, bulk damage, new persistent cuts and split-bank contact. New cuts
preserve existing material motion and contact histories. A joint checkpoint
supports exact continuation; rejected steps roll back every physical state.
The GUI entry is «Остывание и разломы…» in the contact dialog.

This removes the frozen-field assumption of the separate contact experiment.
It retains reference-geometry small strains/sliding, prescribed basal traction
and separate heat/mechanics ledgers. Bulk diffuse shear return mapping is
replaced by Maxwell/damage and explicit interface sliding in this mode.
The historical version 0.1 uses fixed interface birth areas and stops at 2%
area growth. Its 1000-year example has two new cut edges, but no certified
mature plates. Kinematic refinement is small; frictional work remains timestep
sensitive. See `GENESIS_COUPLED.md` and
`results/genesis_runs/coupled_validation_final_20260923/validation_report.md`.

## Growing material contacts, 2026-09-24

Coupled version 0.2 adds independent contact-depth cohorts as the solid
fraction increases in fixed material-reference columns. Geometric thickening
alone does not create material. Each cohort preserves its own birth offsets,
irreversible damage, slip and accumulated work; Newton forces and tangents
sum the individual laws. A new cohort born between open banks is unbonded
and may subsequently carry compression and friction, without filling the gap
or automatically welding it. Compression uses the actual gap, with any
penalty energy introduced at birth recorded separately.

The default 1 m growth quantum leaves a smaller pending depth per trace;
histories are never merged to reduce memory. Thermal-first/mechanical-second
splitting introduces new material on the trial's initial geometry and requires
timestep and growth-depth refinement. The 2% guard now bounds geometric
reference distortion only. Remelting represented interface material and the
100 000-cohort limit stop the calculation explicitly. Small-sliding/contact
partner limits and separate mechanical/thermal budgets remain in force.
Version 0.2 checkpoints preserve cohort histories; version 0.1 must be restarted
from the original fault source because that history cannot be reconstructed.

The 20 000-year reference continuation crosses the historical 12 391.5-year
growth stop, retains separate energy/mass budgets, and adds four cuts while
still having two connected material components. It is not a mature-plate
certification. Independent 2000-year refinements show small kinematic and
depth-quantum sensitivity, but 10-to-5-year stepping changes frictional work
by 2.566%; dissipation remains unresolved. See the v0.2 results in
`GENESIS_COUPLED.md` and `analysis/genesis_coupled_growth_validation.py`.

## Scope

The existing v0.31 runner is the mature-tectonics model. It starts with a
constructed lithosphere and plates and must remain a reproducible reference.
Genesis is a new upstream pipeline that produces a documented handoff
checkpoint; it does not silently change v0.31 initial conditions.

Satellite origin and plate onset answer different questions and are therefore
independent experiment axes:

1. **Satellite origin (3 histories)**
   - `disk_quiet`: accretion in the giant planet's circumplanetary disk;
   - `disk_impact`: disk accretion followed by a major late impact;
   - `capture_circularization`: capture followed by orbital damping and
     circularization.
2. **Plate onset (5 hypotheses)**
   - `stagnant_lid_control`: no mobile-lid transition in the simulated window;
   - `convective_overstress`: mantle stress exceeds a cooling lid's strength;
   - `impact_triggered`: impact damage supplies the connected weak zones;
   - `tide_assisted`: cyclic tidal stress and heating lower the onset barrier;
   - `hybrid_damage`: convection, impacts and tides accumulate persistent
     damage together.

The full first-pass design is therefore a 3 x 5 matrix of 15 hypotheses, not
three mutually exclusive plate models. Typed stable identifiers live in
`moon_gui/genesis_schema.py`.

## Physical timeline

The calculation advances through six checkpointable phases:

1. `initial_conditions`: mass, composition, formation time, orbit, spin,
   obliquity, impact history and uncertainties.
2. `spin_orbit_evolution`: rotation, semimajor axis, eccentricity, obliquity,
   dissipation and giant-planet tide. Synchronous rotation is an event to
   calculate, not an initial assumption. Disk-born bodies can begin rotating;
   captured bodies can remain asynchronous while the orbit evolves.
3. `magma_ocean_cooling`: melt fraction, radiative/convective heat loss,
   differentiation and tidal/radiogenic/impact heat budgets.
4. `lid_formation`: first continuous crust, lid thickness, thermal stress,
   inherited compositional structure and a spatial damage field.
5. `plate_onset`: apply one of the five hypotheses, recording why and where
   connected mobile boundaries appear—or that the body remains stagnant-lid.
6. `v031_handoff`: conservatively remap the solid state onto the mature
   icosphere mesh and start the unchanged v0.31 dynamics.

These phases have separate clocks and checkpoints. A failed or rejected onset
experiment never has to repeat orbital evolution or magma-ocean cooling.

## Minimum state contract

Every genesis checkpoint must carry:

- provenance: scenario IDs, seed, configuration hash, code version and phase;
- orbit/spin: time, semimajor axis, eccentricity, spin rate, obliquity and
  synchronous-state flag;
- energy: mantle/core temperature, global and cell melt fraction, heat-source
  powers and integrated energy ledgers;
- shell: crust and lid thickness, composition, density, water/volatile proxy,
  strength and persistent damage per cell;
- events: impacts, resonances, synchronization, first continuous lid and onset
  candidates with timestamps;
- numerical ledgers: mass, energy and angular-momentum residuals.

The v0.31 handoff additionally requires the existing plate labels, boundary
states, mantle state, crustal material fields, hydrosphere state and plume
state. New plates must be derived from connected mobile regions and boundary
kinematics; they must not be assigned by a random Voronoi initializer at the
handoff.

## Acceptance gates

1. **Orbit/spin gate:** angular-momentum accounting is closed and locking time
   is resolution/time-step converged.
2. **Thermal gate:** energy accounting is closed; solidification and lid times
   are stable under half time steps.
3. **Onset gate:** each mechanism has an explicit threshold and a stagnant-lid
   negative control; onset is not guaranteed by construction.
4. **Handoff gate:** all material fields remap conservatively and a
   checkpoint/resume split is bitwise equivalent.
5. **Mature-run gate:** the handed-off world runs at least 100 Myr in v0.31
   without safety clips or unexplained ledger drift.

## Implementation sequence

1. Add versioned genesis YAML sections and phase/checkpoint metadata.
2. Build and validate the zero-dimensional spin-orbit/energy integrator.
3. Add magma-ocean cooling and continuous-lid formation on coarse meshes.
4. Implement the five onset hypotheses behind one common interface.
5. Implement the conservative v0.31 handoff adapter.
6. Run the 15-case subdivision-3 screening matrix, then promote only stable,
   scientifically distinct cases to subdivision 4/5.
7. Enable the currently disabled genesis choices in the desktop GUI only when
   their checkpoint and handoff gates pass.

This ordering preserves the current v0.31 baseline and lets each uncertain
physical mechanism be tested or replaced independently.

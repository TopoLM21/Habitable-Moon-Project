# Genesis architecture after v0.31

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

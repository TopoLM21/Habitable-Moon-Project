# Active source heat and real-source eligibility

13 October 2026; `feat/genesis-geological-handoff`, parent
`3a65aba03e580eec063034d5da4f45f034237ec7`. Restore CLOSED. The earlier frozen
experiment, its results and snapshots are preserved.

## What is available from Genesis

`analysis/simplified_genesis_validation/import_source.py` carries material IDs,
cohort masses, face ownership, thermal nodes, cp/latent/offset, the Fourier graph,
orbital state, thermal/tidal parameters, inertia, force and drag projections.
`GenesisParameters` is a bulk silicate/water model. Neither it nor the imported
material catalogue provides elemental fractions or continental differentiation.
Those unknowns are not invented. The new mode represents undifferentiated
silicate with the existing noncontinental native carrier; the field named
`oceanic_volume_km3` is a computational volume slot, not proof of chemical basalt.
Continental fraction is zero and continental differentiation remains inactive.

**Required now:** extensive masses/H, a compatible homogeneous silicate EOS,
water/orbit/source parameters, source thermal graph, material face IDs and
selector ownership. Those data are present in the saved real source. Effective
basal forcing can be inherited as `solve(face_drag, face_torque)` in rad/s and
converted to rad/Myr; this is the projected zero-velocity basal equilibrium
forcing, not a reconstructed 3-D mantle flow or inferred plate velocity.
The synthetic experiment retains its explicitly manufactured independent field.

**Optional future physics:** chemical species fractions, depletion and
continental-generation laws. They do not block an undifferentiated run. No new
physical coefficient is requested for the implemented approximation.

## Active mapping and accounting

`ThermalGeologicalContinuation.from_source` is no longer synthetic-only. It
checks the actual source properties. The complete source remains immutable,
but its mass and H are also copied into a separate **active** thermal inventory.
All interior nodes survive; they are not hidden in an unchanging archive.

The real source has the canonical face connectivity with slightly displaced
coordinates. The mapping preserves each face ID's extensive mass and H and
replaces its footprint by the canonical cell. Maximum centroid relocation is
1.1556433 km. This is an explicit discrete geometry replacement, not exact
spatial overlap remapping or preserved resolved deformation. Already rotated
independent domains/edge strips need another spatial mapping. Variable-size
newborn profiles and heterogeneous silicate EOS likewise need implementation;
they are absent in the inspected real checkpoint and require no choice now.

Native transport supplies the actual winning source indices. Surviving thermal
profiles move unchanged with those indices. Lost material returns **its own**
mass and H to the deep reservoir. New gap material draws mass and donor-specific
H from that reservoir before recycling, using the inherited destination profile
mass proportions. The homogeneous EOS makes this mixing well defined. Transfer
masses are independently compared with the native geological creation/recycling
counters; per-face mass equality, no duplicate thermal donors, local inherited H,
total H and total mass are checked. Insufficient donor mass rejects the entire
trial including native transport. No new latent heat or guessed birth temperature
is added. The active EOS reports the carrier's melt fraction explicitly.

The same `genesis_thermal_ports.apply_ports` supplies radiogenic, orbital tidal,
stellar, giant and OLR doses. An opt-in condensation path uses the **existing**
Genesis vapor-column law and outgoing radiation. `genesis_implicit_enthalpy.solve`
then advances the inherited graph with that water EOS at its last node. Its
previous defaults and hot-vapor behavior remain unchanged. Liquid water in the
native hydrosphere and mantle temperature/time now follow the active reservoirs.

Conductances stay fixed on the inherited reference graph as parcels move. This
is a declared Eulerian approximation, not moving-thickness conduction. No new G
coefficients were chosen. The manufactured graph has no water-to-solid edge;
its water cools through radiation, while its solid/deep nodes exchange heat.
A separate test covers a connected silicate/water graph across condensation.
A second native mantle heat integrator is NOT also run. Native reference flux,
activity and thickness are compatibility diagnostics, not extra energy sources.
No temperature-dependent recalibration of the prescribed mantle forcing is made.

Detailed Maxwell state and its elastic energy are frozen only in the archive.
Momentum changes from resolved inertia to the native thin-shell proxy and
`signed_model_change_kinetic_j = K_proxy - K_source` are explicit. The prior
passive kinetic projection/reserve are retained. **No mechanical term is
silently thermalized.** Mature dynamics is an effective quasistatic velocity
law; SI impulse/work conservation is not claimed. Subsequent changes of its
kinetic/momentum proxies remain separate from the closed thermal/material audit.

## Experiment and checks

`thermal_experiment.py` uses the same irregular manufactured stress belts and
zero initial damage/one owner. Existing selector produces 1→2→3; no forced split.
Before evolution, source EOS/radiation/tidal constants are taken from the existing
Genesis parameter classes; mass/water-volume parameters are derived from the
fixture inventories. The deep reservoir is the same 4e24 kg as the previous
experiment. Source ports are active during the two 25-year birth steps too.

Four 10 Myr native steps completed in **2.98 s**, below the five-minute budget:

- 1666 winning-parcel moves summed across steps; relative omega up to
  0.0131422999646 rad/Myr.
- Deep temperature: **990.000011 → 995.249436 K**.
- Water temperature: **914.419356 → 282.189206 K**, with condensation.
- H carried out of the donor: **5.43283156247e26 J**; H returned by recycling:
  **2.85890011170e23 J**. These are exchanges, not external heating.
- Maximum local/global/transport H residual divided by its numerical allowance:
  **0.0872 / 0.0250 / 0.0231**. Material budgets pass. Mechanical heat is zero.
- Exact step-4 direct/restart equality of native arrays, metadata, source,
  orbital state and the complete active thermal inventory/ledger.
- **89 tests PASS in 16.74 s**, covering both handoffs, native importer/material
  ledger, existing ports/conduction, connected water conduction, rollback,
  explicit nonzero kinetic projection and unknown chemistry.

These coarse steps test integration and bookkeeping; time convergence, moving
boundary thermal accuracy and SI mechanics are not certified. Long radiative
steps subtract enormous incoming/outgoing doses. The opt-in audit includes dose
ULPs and representable-root spacing, instead of comparing only storage ULPs.
Compensated node-local flux sums avoid losing small exchanges against the deep
reservoir. These are floating-point balance checks, not exact-error certificates.
The default hot-vapor audit is unchanged.

Development attempts are preserved under `genesis-geological-thermal-01/02/03`:
first the existing hot-vapor guard stopped condensation; then the audit exposed
missing radiative-dose roundoff; then sequential summation in the independent
local check lost small deep-node exchanges. None was bypassed by changing
physical forcing, heat, transport or time step. The first successful run is
`genesis-geological-thermal-04`; final snapshots are
`/workspace/genesis-geological-thermal-final` (16.52 MB). All experiments and
checks combined remained well below five minutes of computation.

Compact evidence: `thermal_result.json`, `thermal_ledger.json`,
`real_eligibility.json`. Earlier `result.json` and `ledger.json` are unchanged.

## Real eligibility and next step

Read-only check of `/workspace/simplified-genesis-phase-pilot-01/initial.json`:
**0.8684 Myr, 800 original cohort records, 2722 thermal nodes**, including
160 mapped solid nodes and 2561 interior nodes, plus water. File checksum is
recorded and unchanged. The one current blocking condition is **one owner**:
the selector has not yet produced multiple domains. No real transfer, split,
evolution or reinitialization was performed. Chemical fractions are unknown
but unnecessary for the explicitly undifferentiated mode.

Next implementation/authorized experiment should address genuine selector
readiness of that same source history. No request for arbitrary chemistry,
new thermal constants or a new exact contact solver is needed for this mode.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  timeout 300 python -m analysis.genesis_geological_handoff.thermal_experiment /workspace/geological-thermal-new
python -m pytest -q -p no:cacheprovider tests/test_genesis_geological_thermal.py \
  tests/test_genesis_geological_handoff.py tests/test_genesis_implicit_enthalpy.py \
  tests/test_genesis_thermal_ports.py tests/test_genesis_mature.py tests/test_genesis_starter_ledger.py
```

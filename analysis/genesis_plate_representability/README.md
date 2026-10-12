# Read-only representation diagnosis and revised early-regime recommendation

No actual handoff, time advance, source reset, physical-code/default change or
canonical write. Measurement ran in **1.60 seconds**, bounded by60s, producing
about12KB JSON. Three manufactured projection tests PASS. `result.json` preserves
one pinned initial state at0.8684Myr and the existing, unaccepted fine100 endpoint
at0.8685Myr. The second is a diagnostic trial, not new accepted history.

## Existing API and exact scope of the measurement

`genesis_starter_continuation.build_starter_continuation` requires first_partition,
at least two domains, and a fully solid primary surface reservoir. These states
have one connected mafic owner and a thin solid shell with hotter material below.
They do not meet that API's physical preconditions. We neither create plate IDs
nor label the young shell mature crust. `genesis_mature` itself calls the import
experimental and returns physical_handoff_certified=False.

Current coarse motion has three Euler-vector components per plate. It lacks the
252 generalized nodal velocity components and800 conical memory records. Its
surface fit minimizes area-weighted tangential velocity error. The diagnostic
calls the existing `plate_rigid_mantle_fit` with the single existing owner, using
top-face velocity samples on its canonical mesh; normal/radial information is
not supplied to that surface API. No mature checkpoint is instantiated.

For a stronger, generous representability bound, the measurement independently
builds the actual consistent volume mass matrix from all800 cohorts and projects
onto three rotations of the actual displaced geometry. An additional six-DOF
translation+rotation fit is a diagnostic lower bound on rigid residual, **not a
new plate model**. No amplitude correction, target-speed calibration, subdivision
into artificial owners or conservation-forcing fit is performed.

## Measured result

| Quantity | Initial | Existing fine endpoint |
|---|---:|---:|
| Solid mass, kg | 7.26186584e19 | same |
| Solid enthalpy, J | 1.21031806e26 | 1.20373042e26 |
| All thermal reservoirs' enthalpy, J | 4.38330798e30 | 4.38324452e30 |
| Kinetic energy, J | 5014.97953 | 0.309964826 |
| Elastic memory energy, J | 3.44416072e19 | 4.95511187e19 |
| Best mass-weighted rotation residual / original RMS | 0.9999998863 | 0.9999999999994 |
| Even with six rigid DOFs: residual / original RMS | 0.9999956164 | 0.9999972458 |
| Best rotation mass RMS residual, m/s | 1.17524e-8 | 9.23947e-11 |
| Source faces above12MPa tensile threshold | 22 | 40 |
| Maximum damage | 0.0395916 | 0.0439860 |
| Active faults | 0 | 0 |

The existing area-fit leaves essentially100% of the volume velocity RMS as well.
The mass-optimal rotation retains only2.27e-7 and1.19e-12 of kinetic energy,
respectively. Its angular momentum is preserved by the normal equations, while
linear momentum generally is not. Six DOFs preserve both momentum vectors to
roundoff but still discard almost all nonrigid motion. Neither preservation of
a few integrals nor the small absolute kinetic energy makes the fit adequate.

The JSON gives full momentum vectors, first mass moment and inertia tensor for
both states. Initial |P|≈2.49448e9 kg m/s and |L|≈1.67986e15 kg m²/s;
final |P|≈1.57473e7 and |L|≈3.01816e10. The initial inertia diagonals are about
1.236–1.239e33 kg m². These were measured on retained geometry, not invented for a
mature sphere. The existing mature API has no corresponding exact volume-moment
mapping. Small residuals relative to an absolute momentum scale are also recorded
because net moment cancellation makes relative errors misleading.

## What is exact, what is missing

- Summing existing solid masses and enthalpies into the80 existing faces produced
  zero floating-point total residual here. This is bookkeeping, not a validated
  coarse thermal evolution. Non-solid/deep/water reservoirs remain distinct.
- The mature import copies source mass/enthalpy arrays into an **archive**, and
  explicitly states they are not evolved by the mature solver. It can also copy
  the80 damage values without loss at that instant. This preserves damage-gate
  classification, not subsequent damage evolution.
- Volume strain0/strain1, their radial dependence, Maxwell history and cohesive/
  frictional memory have no active target fields in this import. An opaque source
  archive can retain their bytes; it cannot make the mature solver evolve them.
- `intraplate_stress` is initialized to zero. Treating that initialized field as
  source stress would erase the22/40 tensile classifications. This is **not** a
  prediction of a subsequently run mature solver: its alternative stress driver
  was not executed and cannot reconstruct those tensors from the transferred data.
- Rigid motion does not physically imply zero inherited prestress. The loss comes
  from this API's missing state, not a theorem that rigid bodies cannot be stressed.
- The depth-resolved enthalpy/phase profile, steam versus liquid inventory and
  local birth-age-dependent relaxation cannot be recovered from one scalar
  temperature or averaged age. Scalar elastic-energy retention alone cannot
  recover tensor orientation or local threshold crossings either.

**Conclusion for an immediate rigid-plate handoff: not suitable.** An objective
future admission test must bound discarded motion in the declared output metric,
account for discarded mechanical energy/momentum, retain active memory or justify
its closure, preserve relevant event classifications, and satisfy the physical
partition/solidification preconditions. Current residuals near100% and missing
active memory fail this without choosing an arbitrary fit threshold. The current
states do not demonstrate that a later plate-scale regime is impossible.

## Revised user priority: simplify early physics without pretending it is rigid

The user's goal is planetary cooling, ocean appearance and later large-scale
tectonics, rather than the geometry of every crack in an80m crust. On that goal,
an explicitly approximate **early thermal/solidification regime with aggregate
strength/history** is a better candidate than forcing the young shell into rigid
plates. This is a proposed model choice, not an implemented transition or a
retroactive reinterpretation of accepted history.

A minimal candidate active state would retain:

1. Silicate and water masses in relevant reservoirs; enthalpy/latent energy and
   solid fraction in a small vertical representation, with the existing thermal/
   orbital clocks and source/debit ledgers. Temperature alone is insufficient.
2. Solid mass/age or birth-history summaries and thermal gradients needed for
   buoyancy and strength. Keep a small spatial field if regional differences
   determine the eventual plate partition; a global mean cannot do so.
3. Aggregate weakening: mean damage plus weak-area fraction or an appropriate
   distribution measure; permeability/water access where active. The existing
   equations contain damage² and nonlinear threshold exceedance, so the mean
   alone is not an exact closure. Any rate closure must be labelled approximate.
4. An explicit residual mechanical-memory/energy record, or a demonstrated bound
   that its later effect is negligible. Do not deposit lost memory as invented
   heat or reset stress simply because a time/thickness milestone was reached.

Early cracks matter later if they alter heat transfer, water penetration and
weakening, leave persistent preferred weak zones, or change the first stable
partition. Their individual paths can be parametrized only at the price of losing
predictions of exact fracture locations and activation times. If cracks heal and
stress relaxes before affecting those observables, retaining every path is not
necessary. Current Maxwell times of3–4kyr, versus a shell-diffusion estimate of
~200yr, do **not** yet demonstrate immediate forgetting. A memory-decay bound can
use the actual integrated relaxation exposure, rather than an arbitrary age cut.

An appropriate handover trigger is the combination of a connected load-bearing
solid shell, persistent stress on the timescale of changing loads, and resolved
large-scale heterogeneity whose influence exceeds the user's declared coarse
accuracy requirement. Use the actual temperature/viscosity profiles, forcing
variation, damage/healing rates and memory bounds. No universal thickness or
fixed age is assigned. The decision is also affected by ocean/water feedback;
full solidification of the mantle is not a prerequisite.

Existing code that can be reused: Genesis enthalpy/phase and source/orbit ledgers,
solidification chronology, existing material/thermal inventories and mature
plate machinery after its applicability conditions are met. The current
solidification/Starter route already has explicit approximations and preserves
history. It is a starting point, not proof that the new reduced regime is
physically equivalent. New aggregate strength/memory closure and its admission
criterion would be a separate, explicitly authorized approximation.

**User tradeoff:** faster global cooling/ocean/regime evolution, while surrendering
early local stress/crack maps and precise fracture timing. Keep the detailed
history archived, and do not silently splice an unvalidated reduced trajectory
into it. This result closes the representation question: exact same-history
acceleration through the existing rigid API is unavailable now; an early reduced
thermal-strength model is a consciously lower-detail alternative, not a free
optimization. No such model or transition was implemented here.

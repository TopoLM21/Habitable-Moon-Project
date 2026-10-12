# Open boundary energy budget of the saved BE50 trial

Base `7e67a44d85ef834b31517142fcbe9825c69321d9`. Restore CLOSED.
No evolution, temperature update, archive operation or default change.
Read-only result: `/workspace/genesis-open-boundary-budget-01.json`.

## What the current equations actually specify

**Prescribed traction is an external work port.**
`genesis_shell.mantle_traction` explicitly prescribes a smooth potential-flow
pattern; it does not solve mantle flow. `genesis_volume_basal` integrates this
as a dead/reference traction on the actual basal surface. Its discrete work is
`Wext = F·du` joules, positive into the resolved system. The donor is the
prescribed external mechanical driver. No fictitious deep-mantle debit or new
mantle dynamics is necessary to make this an honest open system. The independent
quadrature recomputes F and its work from the saved state.

**Basal drag is effective interface resistance.**
The implemented law is `t_drag = -beta P_t v_b`, with beta in Pa·s/m and a
stationary unresolved reference substrate. The matrix integrates `beta NᵀP_t N`
over area, not a mantle volume. Its loss is `dt*vᵀDv` joules. There is no lower
mantle velocity, shear-layer thickness, temperature or viscosity field tied to
beta. Thus identifying beta with `eta/H` would be an *additional* Couette-layer
model, not recovery of a model already present. Integrating the positive loss
on the interface establishes its location as an unresolved boundary channel;
it does not determine a fraction heating the crust versus the substrate.
The audit records 80 face powers and areas. No 50/50 split or deposit into the
existing deep thermal reservoir is justified by the code.

**Damage changes elastic free energy, but has no fracture-energy potential.**
The elastic law is `psi_el = 1/2 e : C(d) : e`, with
`C(d) = [r+(1-r)(1-d)^2] C0`. At fixed old elastic memory the operator records
`DeltaU_damage = 1/2 integral e_old : (C_after-C_before) : e_old dV`.
For increasing d this is negative; the released amount is `-DeltaU_damage`.
The kinetic law mixes stress/tidal driving with temperature-dependent healing.
It contains no fracture surface energy Gc, stored damage-energy potential,
entropy law, latent term or heat-conversion fraction. A positive damage-energy
change under healing would require a specified energy donor. A scalar journal
alone cannot prove a constitutive law is thermodynamically closed.

For this saved trial the independent per-cohort integral finds **800 nonnegative
releases and zero healing cohorts**. Its separate nonthermal accumulator is a
valid accounting destination for this interval. It is **not** a calibrated
recoverable fracture free energy and cannot automatically finance healing.
The new port helper rejects aggregate unfunded healing; the audit also checks
local signs to prevent cancellation from hiding it. Future healing requires a
constitutive rule and local funding check, not a silently negative heat source.

## One minimal bounded hypothesis, with its limits

Retain the existing unresolved interface/damage channels as **thermally isolated
from resolved cells over this diagnostic interval**. This is one temporary
no-return-heat hypothesis, not a numerical estimate of a heat partition and not
a universal physical law. It introduces no new architecture or arbitrary split.
It is sufficient for an auditable forced-system budget; it does not settle
future long-time heating or healing. This task makes the hypothesis explicit
without applying any additional heat or advancing a state.

The distinction between fracture expenditure and heat is physically motivated:
[NASA TM-89032](https://ntrs.nasa.gov/citations/19870007657) explicitly separates
crack separation, plastic dissipation and residual strain-energy contributions.
That material study does not calibrate this planet. In a specified cohesive
law, fracture energy instead comes from the traction-opening integral and its
history, as illustrated by the author's
[FEniCS cohesive-zone implementation](https://comet-fenics.readthedocs.io/en/latest/demo/cohesive_zone_model/cohesive_zone_model.html).
Those ingredients are absent from the present damage kinetics. Neither source
establishes that all damage release here is heat, nor that the no-return-heat
hypothesis remains valid beyond the bounded control.

## Completed open budget

Choose the resolved domain as thermal reservoirs + K + U + unresolved damage
accounting store. Keep the interface outside that domain:

`Delta(H+K+U+Sdamage) + Nnumerical = Wtraction - Qdrag`.

Maxwell and thermal work are internal, paired exchanges and cancel; they must
not appear again as sources. Alternatively include the interface accounting
store inside the domain and replace `-Qdrag` at the boundary by `+DeltaSinterface`
on the left. **Never count both.** Both domain choices give the same residual.

| Quantity over 50 years | Joules | Mean power |
|---|---:|---:|
| External traction input | +1.202005e15 | +0.762 MW |
| Drag exported to unresolved interface | −9.846185e15 | −6.240 MW |
| Damage nonthermal storage increase | +8.208626e15 | 5.202 MW |
| Numerical sink, never heat | +8.207145e15 | 5.201 MW |
| Located Maxwell → thermal | +9.639596e17 | 610.921 MW |
| Thermal work into mechanics | −3.349820e16 | −21.230 MW |

For the mechanical-feedback stage (thermal baseline is the saved conduction
endpoint), net boundary input is **−8.644180e15 J**. Domain energy change including
damage storage is **−1.685133e16 J**; numerical sink closes the remaining debit.
Residual **3.583396e8 J**, within the previously established **2.024413e11 J**
allowance from touched-cell endpoint rounding and mechanical audit. No tolerance
is borrowed from untouched deep/water energies. This is finite-precision
accounting closure, not proof of exact continuous physics.

For the **whole existing split interval**, the initial thermal state is the
pinned conservative cut-cell initialization, including the projected deep/water
reservoirs. Direct endpoint readback gives total thermal change
**9.9906953641e17 J** and open-budget residual **1.6117758141e15 J** within the
summed stage allowance **9.3634641485e15 J**. The difference is the already
measured conduction residual **1.6117754558e15 J** from enormous thermal endpoint
numbers; it is not new external heat. This broad global bound cannot resolve
the traction work by itself. Independent face-work, local heat, and mechanical
checks remain mandatory and are not replaced by this larger allowance.
Supplement: `/workspace/genesis-open-boundary-full-split-01.json`.

## Significance for the next step

Actual top conduction into water is **5.489404e23 J / 3.478974e14 W**; bottom
conduction into deep mantle is **1.915327e23 J**. They are internal reservoir
exchanges, not extra external inputs. Drag and damage release are respectively
**1.794e-8 / 1.495e-8** of the top-conduction energy, but **1.021% / 0.852%** of
Maxwell loss. They are small thermally here, yet should remain in mechanical
accounting. Initial elastic energy is 3.444161e19 J; their fractions are
**0.0286% / 0.0238%**. Numerical loss is of similar size to damage release,
so a small temperature effect does not justify reclassifying numerical loss.

The retained-solid sensible capacity is **8.714239e22 J/K**; all-column sensible
capacity is **5.058142e25 J/K**. Diagnostic *hypothetical complete heating* gives:

- all drag spread over retained solid: **1.130e-7 K** mean; the largest
  face-homogenized value, using that face's own drag, is **3.075e-7 K**;
- all damage release spread over retained solid: **9.420e-8 K** mean; using
  each cohort's own release/capacity gives at most **1.594e-6 K**.

These are sensitivity scales, **not applied transfers or pointwise temperature
upper bounds**: without a heat-receiving thickness, localization could be much
stronger. They do not establish that a 1e-8 K coupling criterion would absorb
all possible missing heating. Near a fracture/healing threshold, the allocation
still matters even if global conduction dominates.

The external-driver question is resolved by an explicit boundary condition;
solving the entire mantle is not a prerequisite. The remaining physical question
is **how unresolved interfacial/damage energy returns to resolved material, and
what funds healing**. For the next bounded numerical accuracy control it can
remain explicitly isolated. Before claiming physical long-time heating or
healing, specify that constitutive transfer. Separately test the already open
conduction/mechanics splitting and initial-profile sensitivity. Four spatial
controls and the original incomplete plan stages remain open.

## Verification

`tectonics/genesis_energy_ports.py` implements both control-volume budgets as
opt-in accounting only. Manufactured tests cover signed external work, exclusion
of internal exchanges, domain equivalence, and rejection of unfunded healing,
negative sinks, nonfinite data and undeclared material ports. Source snapshots
and saved candidate hashes are authenticated; per-face work/drag and per-cohort
damage agree with independent integrals. **74 tests PASS.** The machine summary records results;
no existing candidate or canonical file was rewritten.

# Source flow, live coupling, and thermal activity: read-only follow-up

No production physics, constants, or checkpoint files were edited. Reproduce with
`python analysis/slow_motion_followup/origin_probe.py`. Source SHA256 and exact
results are in `origin_probe.json`. All current-loading fields below are frozen
counterfactual evaluations of existing formulas, not simulated predictions.

## Confirmed inconsistency between flow and fracture forcing

`genesis_shell.mantle_traction` defines
`tau(x,h) = tau0 c(h) grad_s(Phi)`, `c(h)=1-exp(-h/2 km)`.
`genesis_starter_material.independent_mantle_omega` divides this already coupled
traction by constant `beta=1e14 Pa s/m` once at first partition. The resulting
field enters `MantleFlowState`; the later mantle solver has no lid-thickness input.

In contrast, `StarterModel._material_sample`, called throughout continuation by
`YoungShellFracture.advance`, reevaluates `c(h)` at every thermal sample for its
stress proxy `tau0 * L/h * c(h)`. Thus the two branches cease to use the same
source-amplitude convention immediately after import.

| Time after partition, Myr | Live lid, km | Live coupling | Live/source coupling |
|---:|---:|---:|---:|
| 0 | 0.216162257 | 0.102445224 | 1 |
| 1 | 5.746049828 | 0.943472324 | 9.209529630 |
| 10 | 22.529406636 | 0.999987183 | 9.761188915 |
| 400 | 34.595806207 | 0.999999969 | 9.761313731 |

At 400 Myr the actual local mantle RMS is **1.392256624 km/Myr**. Evaluating the
same existing loading formula with current lid thickness gives **14.188734661**.
The corresponding rigid-fit mean velocities are **1.041145444** and
**10.610388322**. This establishes sensitivity to an imported historical factor;
it does not establish which field is physically correct.

The stress proxy itself does not grow tenfold: its additional `1/h` decreases
the scalar stress scale from 37.914 MPa at formation to 2.312 MPa at 400 Myr.
The inconsistency concerns the coupling source, not a claim of increasing stress.

## Thermal activity compares different transport regimes

`project_thermal` sets the permanent reference to total mantle-to-surface heat
flux at partition: **53.997372430 W/m2**, dominated by the partly molten transport
blend. One Myr later total flux is **2.710022211**, ratio **0.050188039**, and the
activity proxy clips to **0.35**. Yet the shared *solid convection* flux increases
from **0.002964143** to **0.016385855 W/m2**, and Ra increases from
**7.77756e6** to **1.97655e7**. A cooled surface changes the temperature difference
as well as the transport regime, so neither total nor solid heat-flux ratio alone
is a demonstrated speed law.

At 400 Myr the total-flux ratio is only **0.000537700**, while solid-convection
flux remains **0.012778598**, above its formation value, with Ra **1.03339e7**.
All sampled late states therefore receive the same activity=0.35. The runner
multiplies empirical boundary drive by that value. This contributes no present
velocity loss because the present boundary force terms are exactly zero, but
will affect any future bootstrap experiment.

For mantle evolution, activity=0.35 gives a target amplitude fraction
`0.72+0.28*0.35**0.35 = 0.913901803`. The observed source-relative amplitude is
**0.957819965** after 400 Myr. Thus the thermal modulation explains about **4.22%**
of local amplitude decrease since formation, not the order-of-magnitude deficit.

## Mechanisms excluded or still assumptions

- Formation-to-400-Myr velocity-pattern cosine is **0.999999972764**. The mantle
  model mostly retains its formation pattern. Calling it a convection response
  to growing plates or evolving lid would be incorrect.
- The invisible radial component introduced by graph averaging contains only
  **1.1287e-8** of stored omega squared at 400 Myr. Its effect cannot explain the
  slow velocities in this run.
- Constant beta does not use the shared thermal viscosity. A physically derived
  drag law requires a stated shear-layer geometry and rheology. Using `eta/D`
  without that choice could make the system slower; no increase is guaranteed.
- Because initial traction is a degree-two potential field, nearly zero rigid
  torque on the initial two large domains need not be numerical loss. At five
  Myr even current-coupling reevaluation gives rigid-fit mean only **0.023818**,
  versus local RMS **14.169294**. The subsequent four-domain geometry represents
  much more of the same flow. A velocity multiplier cannot recover deformation
  that is outside a rigid plate's degrees of freedom.

## Concrete correction plan, before any coefficient tuning

1. **Define one basal interaction.** Choose and document whether `tau0` prescribes
   traction or a free mantle velocity. For a prescribed free velocity, an explicit
   candidate is `tau_b=beta*c(h)*(u_free-u_plate)`. Its driving and resisting terms
   must use the same live coupling. If traction is prescribed instead, use
   `tau_b=c(h)*tau0*g - beta_eff*u_plate` and define beta_eff explicitly. These
   alternatives have different thin-lid limits; they cannot be silently mixed.
   Root's SI torque audit covers the unexplained 0.22 and missing opposing torque.
2. **Separate source pattern from boundary interaction.** Store the prescribed
   Eulerian source pattern/amplitude independently of formation lid thickness in
   `mantle.py`; provide the current mechanical sample to a shared basal-load
   evaluator. Route starter fit, continuation plate dynamics, and fracture loading
   through that evaluator. Preserve geometry under plate relabeling and splitting.
   Do not repair by merely dividing the old checkpoint field by 0.102445.
3. **Replace ambiguous thermal-reference semantics in Genesis.** Keep the scalar
   heat solver unchanged. Introduce an explicit mechanically relevant mantle
   amplitude/rheology closure using consistent material state. Until that closure
   is derived, expose the existing flux proxy as a legacy assumption rather than
   treating it as a measured convective speed. Separate this from the old mature
   mode and from chemistry-rate activity where appropriate.
4. **Version the field interpretation and checkpoint state.** Store source-law ID,
   amplitude units, coupling treatment, reference state, and current mechanical
   coupling. Older archives need an explicit compatibility/migration path, using
   their saved origin metadata, so resume cannot silently reinterpret velocity as
   traction or change a historical source amplitude.
5. **Validate limiting cases.** Exact rigid imposed flow; no-slip torque balance
   without other forces; vanishing-lid coupling applied consistently to force and
   drag; identical physical state reached via different handoff times; symmetry
   and relabeling; slip-dependent stress; smooth thermal regime transition; no
   positive mechanical power without its source; deterministic resume; unchanged
   ordinary mature mode and closed thermal/material ledgers.
6. **Then rerun the same 0.08-MPa source.** Compare individual changes before their
   combination at 5, 50, 100, 200, 400 Myr. Save basal torque/resistance/power,
   current coupling, local and rigid speeds, signed boundary displacement and
   slab geometry, plus thermal histories. Report any remaining slow regime rather
   than choosing constants to obtain an Earth-like result.

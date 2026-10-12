# Coupled SDIRK2 prototype — verified properties, admission blocked

Base: `33085fe0c2b5d333bc8e7d0acecc8d98be1b5691`. No production hook,
planetary step/series, canonical write, archive operation, push, new physical
parameter or changed acceptance tolerance. Manufactured finite integrations are
verification problems, not additions to the Genesis trajectory. The rejected
real 25-year pair remains rejected. No speedup of a real accurate step is claimed.

## One method and its actual scope

The chosen standard method is stiffly accurate two-stage SDIRK2:

```
gamma = 1 - 1/sqrt(2)
A = [[gamma, 0], [1-gamma, gamma]]
b = [1-gamma, gamma], c = [gamma, 1]
```

The scheme and its order/L-stability properties are documented in Kennedy and
Carpenter, NASA/TM–2016–219173, §4.1.2, equation (221):
https://ntrs.nasa.gov/api/citations/20160005923/downloads/20160005923.pdf

Both stages solve enthalpy, momentum and Maxwell memory together. One-step RK
needs no multistep history when dt changes; equal diagonal coefficients allow a
future reusable preconditioner. Its stability function is
`R(z)=(1+(1-2gamma)z)/(1-gamma*z)^2`, tending to zero for stiff decaying modes.
There is no exact-memory substitution or thermal-first split. These benefits do
not imply positivity, passivity, nonlinear convergence or a safe large dt.

The implemented coupled equations are the existing **fixed-geometry,
fixed-catalogue, constant-C/relaxation** equations, with the existing enthalpy law
and local Maxwell/thermal-work exchange. This is a restricted coefficient
prototype, not a complete new planetary integrator. Geometry/damage-dependent
constitutive coefficients, evolving source/orbit stages, birth and remelt are
not approximated by pretending they are handled. Their absence blocks admission.
In particular, state-dependent H would require its energy-rate contribution;
this audit must not be applied unchanged to that case.

For thermal node energy `E=m h(T)`, one memory block at that node obeys:

```
M v' = F - D v - sum(B.T H e)
e'   = B v - U T' - lambda e
E'   = Qsource - graph_flux + sum(lambda e.T H e + e.T H U T')
```

`U` is the existing thermal-expansion coefficient in stable conical memory
coordinates. The last two terms are exactly Maxwell heat and the opposite of
mechanical thermal work, following `genesis_cutcell_exchange`. Thus the thermal
solve uses effective capacity `m dh/dT - sum(e.T H U)`; nonpositive effective
capacity is rejected. Drag remains a separately accounted reservoir, not invented
heat. Numerical work is never routed into enthalpy. Thermal inversion/phase law,
conical B operators, radial mean/variance energy metric, consistent mass and basal
drag reuse production functions. No new conductivity, viscosity, phase threshold
or heat partition is introduced.

`core.py` uses bounded dense nonlinear stage solves. `model.py` also offers a
mass-form residual, avoiding explicit inversion of thin-cohort inertia in Newton.
`audit.py` independently reconstructs stage momentum, thermal node exchanges,
memory equations and work using the saved stages/rates; it does not call the
producer RHS. This is a prototype audit, not a replacement for the existing full
independent planetary auditor. Its manufactured residual checks are separate
from, and do not modify, existing physical acceptance budgets.

## Energy result: strict nonnegative numerical dissipation fails

For fixed quadratic mechanical energy metric W, define
`Gij=bi*aij+bj*aji-bi*bj`. RK algebra gives exactly, up to solve/roundoff error:

```
N = dt^2/2 sum_ij Gij <ki,kj>_W
Delta(Ethermal + K + Uelastic) = Qsource + Wexternal - Qdrag - N
```

Memory and kinetic contributions to N are reported separately. Maxwell heat and
signed thermoelastic work cancel between the actual thermal and mechanical
reservoirs using the same stages and weights. Conduction exchanges cancel node
by node. Impulse uses the same weighted stage forces.

For this tableau G has eigenvalues **-0.0857864376269, +0.0857864376269**.
It is not algebraically stable: N need not be nonnegative. The recorded decay
example yields **N=-0.00367752698892 J**, with balance residual
**-1.19e-14 J**. The deficit is real discretization error, not an omitted physical
reservoir, roundoff, or something to relabel as heat. The positive Maxwell source
is computed from stage states; its quadrature can overdraw the exact available
mechanical energy, and the signed N exposes that effect. No clipping/compensating
heat fixes it. Under the unchanged requirement of strict nonnegative numerical
dissipation, this scheme is **not admissible**, even though it is L-stable and
second order. Reducing dt is not a general proof that N becomes nonnegative.

Temperature positivity is also conditional: for scalar cooling, R(-x) becomes
negative for x>1+sqrt(2). A manufactured x=3 thermal step is rejected; x=.1
remains positive. Every solved stage and endpoint is checked for positive energy/
temperature and the selected phase interval. Phase fractions use the existing
piecewise enthalpy law, not a correction of a negative computed temperature.
Stage checks alone do not certify the entire path.

## Evidence and failure retained

**135 scoped tests PASS, including 13 new prototype tests.**

`validation.json` is a byte-identical copy of the finite result at
`/workspace/genesis-sdirk2-validation-01.json`, with source manifest and input
hashes. Its status is **PROTOTYPE_VERIFIED_WITH_ADMISSION_BLOCKERS**.

- Uniform dt analytic errors decrease by **3.966, 3.983** on halving dt.
- Alternating variable steps decrease by **3.871, 3.933**. Energy identities are
  within **1.34e-14 J** in those examples.
- A separate fully coupled manufactured solution checks second order for all
  three fields under nonzero thermoelastic feedback and forcing.
- Stiff decay, internal-force translation momentum, paired conduction, independent
  corruption detection, mass-form equivalence and invalid capacity are tested.
- Near-phase smooth heating retains order two. A crossing is rejected; a separately
  labelled manufactured restart at the known phase boundary is exact for constant
  heating, with no numerical heat. That test does not authorize remelt/birth.
- A manufactured patch using production conical strain/energy, consistent inertia
  and basal operators passes, including independent radial-energy reconstruction.
- A second patch extracts **actual saved cohort 1** geometry/C/mass/relaxation and
  shell thermal parameters with pinned hashes, then applies manufactured states
  and forcing. It is not an extraction of a complete global basal problem.
  Its mass condition number is **1.0631e12**. The mass-form stage solve stops at
  scaled residual **1.29062e-9**, exceeding the unchanged prototype **2e-10**
  criterion after 441 evaluations. This is an explicit failure, not second-order
  evidence on that real patch. Earlier diagnostic attempts with an explicit
  inertia inversion also failed; no tolerance or coefficient was changed to pass.

The root solve is intentionally a bounded prototype, not a scalable solver for
800 coupled cohorts. One targeted future implementation would use the existing
mechanical block elimination/scaling with a thermal Schur complement. It must
first solve the same pinned local problem with unchanged residual criteria.
There is no reason to launch a planetary series to investigate this failure.

## What would bound phase/birth/gate throughout a real step?

Stage values alone, even at positive temperatures, are insufficient. Retain stage
E,T,phase fraction, T', v, memory/stress, source/orbit variables, damage/drivers,
geometry, stage residuals and their source/code/ID bindings. A possible second-
order continuous extension uses
`beta1(theta)=(theta-theta^2/2)/(1-gamma)` and
`beta2(theta)=(theta^2/2-gamma*theta)/(1-gamma)`.
This polynomial is an approximation, **not an enclosure of the exact solution**.
A usable enclosure additionally needs a bound on its ODE defect and on the local
Jacobian/one-sided growth over a validated state box, including nonlinear-solve
error. No such real bound is implemented or claimed here.

- **Phase:** turn enclosed node E into a temperature interval using the monotone
  existing enthalpy inverse; compare all old/new material and water margins to
  their existing thresholds. A latent-law kink invalidates the smooth order proof.
  Stop/localize there; do not step across and continue to label it second order.
- **Birth:** use exact source-defined scheduled times where available; thermal
  birth requires the same phase enclosure. New IDs, mass and memory need audited
  conservative transfer and new cache bindings. No interpolation through birth.
- **Damage gate:** enclose stress/temperature and the existing damage-rate driver
  together with damage over the whole interval. Constitutive nonsmoothness and
  unresolved geometry/contact crossings return UNRESOLVED. Endpoint flags or the
  two stage values alone cannot exclude an interior crossing/touch.

These are the concrete inputs/obligations for `EventPolicy.enclose`, not a newly
certified provider. The event harness remains synthetic. Cache use remains the
existing explicit `FastPath` option for invariants only; there is no stage-result,
physical-audit or dynamic-coefficient caching added here.

## Condition for exactly one real comparison

**No real comparison is admitted now.** Before one bounded comparison against the
saved BE reference: (1) resolve the strict passivity incompatibility without
silently changing its criterion — as long as that requirement stands this tableau
cannot be generally admitted; (2) pass the pinned thin-cohort nonlinear solve and
independent stage balances without loosening tolerances; (3) provide consistent
stage treatment of actual source/orbit, temperature-dependent coefficients and
damage plus their energy terms; (4) certify a phase/birth/gate-free interval or
stop at its localized boundary; (5) retain the existing significant-output/work
criteria and compare time-to-accuracy, including all stage/audit cost, rather than
assuming the two-stage method is cheaper. This is a finite no-go decision under
the current strict dissipation requirement, not permission for a method search
or a months-long integration campaign.

Reproduction (new output path; no physical evolution):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_sdirk2_prototype.validate \
 --output /workspace/genesis-sdirk2-validation-NEW.json
```

The validation CLI has a 120-second deadline and refuses an existing output file.
It retains the real-patch failure in the report. A successful CLI execution means
that the bounded verification completed, not that planetary admission passed.

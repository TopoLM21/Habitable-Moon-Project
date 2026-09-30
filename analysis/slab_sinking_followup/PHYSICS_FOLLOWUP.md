# Follow-up physics audit: slab survival and ordered thermal forces

Scope: the follow-up to `young-mechanics-0.4`. The activated `0.5` correction
resolves the distribution of thermal mass along the existing prescribed slab
shape. The instantaneous neck-failure rule is retained and remains a major
physical limitation. No coefficient was fitted to Earth plate speeds.

## What the 91% number establishes

The recorded 0.4 run transferred 105.0493 million km³ of 115.6154 million km³
accepted material to the mechanical-detachment ledger by 400 Myr: 90.861%.
That accounting result is not itself evidence that real slabs on this body
would detach at the same rate. The old final speed also rose after a cluster
of failures released resisting constraints, so a faster final snapshot cannot
be interpreted as stronger sustained slab pull.

The force audit finds no sign or units error in the transmitted neck force:

`T = F_gravity - C_bending q - mantle_feed_row · omega + lambda`.

Here `q` is incoming plate speed relative to the trench, `T*q` is the
work-conjugate feed power, and `lambda >= 0` enforces the explicitly chosen
no-eduction boundary condition. The mantle term includes motion of the
trench/overriding plate. Replacing this expression by total slab weight, or
omitting the mantle term, would change the force balance incorrectly.

The questionable inference is the failure threshold `T > sigma_surface W H`.
`StarterModel._material_sample` computes `sigma_surface` from a nominal surface
tensile strength, global mean lid temperature, water access and inherited
damage. Its own shear-law comment explicitly excludes vertical overburden,
fault dip and pore pressure. The runtime correctly transports that field, but
applying the resulting roughly 4–4.5 MPa scalar through an entire 10–34 km cold
mantle section and deleting its attachment immediately is an additional model
assumption, not a resolved slab rheology.

Pressure-dependent friction and thermally activated creep enter lithospheric
strength envelopes differently and depend on water, temperature and strain
rate. [Kohlstedt, Evans and Mackwell (1995)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/95JB01460)
reviews these experimentally constrained mechanisms and their uncertainties.
Thermomechanical slab-detachment models find viscous necking and localized
shearing more representative than homogeneous brittle opening across a whole
slab. [Duretz, Schmalholz and Gerya (2012)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011GC004024)
provides the relevant mechanism comparison. Neither paper determines this
satellite's material properties or validates importing Earth rupture times.

## Frozen failure decomposition

`physics_failure_probe.py/.json` evaluates saved 0.4 states without advancing
heat, motion or material and verifies their SHA256 hashes before and after.
The richer trace produces 0, 1, 1 and 10 new failures at the saved 50, 100, 200
and 400 Myr states respectively. These 12 examples are **not** a decomposition
of the original 1023 events during evolution.

All 12 exceed the old capacity even after subtracting their own no-eduction
reaction at that same solution. Consequently reaction-only rupture is not an
adequate explanation of these probes. The subtraction is not a re-solve without
constraints: other contacts and velocities would change in that experiment.
Some reactions nevertheless strongly amplify tension. At one 400 Myr contact,
thermal pull contributes about 113 MPa and the reaction about 427 MPa, giving
539 MPa total after resistance. Other failed contacts have zero own reaction
and 24–40 MPa pull. At 100 Myr, pull 28.83 MPa minus bending 21.78 MPa and mantle
1.39 MPa leaves 5.66 MPa tension, already above 4.35 MPa capacity.

## Offline pressure and creep sensitivity

`physics_strength_probe.py/.json` is analysis code and is never called by
production. It reuses the actual gravity 7.12 m/s², existing shell density
3000 kg/m³, local transported age/reference-column temperature, Shell Arrhenius
viscosity, live water access and weakened cohesion. It invents no new strain
rate or strength calibration.

The illustrative stress convention is compression positive, vertical principal
stress `sigma1=P`, horizontal tensile differential `d`, and `sigma3=P-d`.
For `phi=atan(mu)`, the normal-faulting Mohr-Coulomb condition gives

`Y_MC(z) = 2 [c cos(phi) + P(z) sin(phi)] / [1 + sin(phi)]`,

with the opening cap `Y_open(z)=P(z)+sigma_surface`. We integrate
`Y=min(Y_MC,Y_open)` through cold mantle only. The chemical crust contributes
overburden, not load-bearing neck area in this diagnostic. Pressure uses
`P=rho g z` with **zero pore pressure**. Reusing the live surface weakening for
cohesion is another stated diagnostic assumption; no depth-dependent hydration
or resolved stress tensor is available.

The mean confined capacity is 47–127 MPa, or 10.8–31.8 times the old scalar
capacity, across these 12 sections. Nine lie below that illustrative capacity
and three remain above it. This is sensitivity evidence that the current
whole-thickness surface threshold is poorly justified; it does not establish a
correct replacement threshold or prove nine real slabs would survive.

For a further constitutive check, the script solves

`F = W integral min(4 eta[T(z)] * strain_rate, Y(z)) dz`

for a uniform plane-strain axial rate at the observed force. The factor four
is the differential-stress coefficient of incompressible Newtonian plane
strain. No arbitrary strain rate is prescribed. The nine sub-capacity examples
produce unit-strain times from about 287 to 75 million Myr. These very long
values reflect the existing high cold Shell viscosity (up to 1e28 Pa s), its
strong variation with depth and the imposed uniform strain rate. They are
**not breakoff times**, and demonstrate why this diagnostic should not simply
be activated as a complete neck model. Nonlinear mantle creep, localization,
water-dependent rheology and evolving section thickness remain unresolved.

The 16-to-32-node-per-profile-interval quadrature comparison changes integrated
capacity by at most 2.3e-16 relative and inferred times by at most 4.6e-4. Four
analytic tests check the pressure integral, constant-viscosity extension,
fully yielded non-uniqueness and the absence of tensile extension in compression.
Both source checkpoint hashes and the input force-probe hash are saved.

## Activated ordered thermal force correction

The previous law applied the same mean excess mass per unit length everywhere,
including the shallow bend. Cohort age already controls thermal mass decay, so
that averaging can place recently accepted, cold material artificially deep
and old, warmed material artificially shallow.

Version 0.5 orders retained cohorts from newest/shallowest to oldest/deepest.
For each contiguous retained arc interval `[s0,s1]` of mass excess `dm`, its
feed force is

`Fg_i = g * dm/(s1-s0) * integral[s0,s1] sin(theta(s)) ds`.

Equal acceptance-time batches are aggregated before allocating intervals;
partial deep-transfer fractions contribute neither force nor retained length.
The smooth bend and resistance matrices are unchanged. Empty layer metadata
retains the old exact uniform arithmetic for older versions. Detailed traces
also retain the old uniform counterfactual force for direct comparison.

The independent frozen final-400 Myr inventory audit gives total ordered force /
uniform force = 0.9999899024 (33 sections, nine with multiple cohorts). Therefore
this actual-state correction is tiny; it is not the main cure for detachment.
Controlled mixed cold/warm bend tests show why the correction matters in general.
There are 28 force tests, including independent quadrature and power, constant
linear mass density, reordered/split layers and malformed inventory rejection.

## Numerical support correction discovered during validation

A new sub5 run near 29 Myr exposed a numerical issue in the secondary reaction
allocation, which minimizes `sum(lambda_i^2 / (W_i H_i))` at unchanged torque.
SLSQP left inactive normalized forces around 1e-11 beside four true forces of
order 0.2–1.0. A single support threshold included those inactive variables and
exact least-squares polishing then found negative forces on the false support.

Polishing now follows equality-constrained minimum-norm directions to the first
blocking nonnegative bound, and releases a bound if its KKT multiplier requires
it. The bound, equality, KKT and physical torque guards are unchanged. The
captured case recovers the independently verified four-force optimum with
orthonormal-coordinate equality residual about 3.1e-17. The actual fixture,
primal optimization oracle, permutation, area-weighted section split and three
analytic active-set cases are included among 29 constraint tests. The combined
force/constraint suite passes 57 tests; the inventory agent separately checked
240 splits of this captured case. This changes the numerical solution process,
not a physical failure threshold.

## Next physically closed implementation

A finite neck must distinguish plate feed from slab-body sinking and account
for the material between them. A viscous extension law would dissipate positive
power `T*(w-q)`; for a uniform Newtonian neck of length `ell`, its Rayleigh
coefficient is `C_neck = 4 W integral eta dz / ell`. Merely adding that extra
velocity or a failure clock would leave the current accepted-material geometry
inconsistent.

Before activation, the implementation needs an explicitly owned neck volume,
advection between incoming plate/neck/slab body, evolving length and thickness
with conserved volume, and re-evaluated buoyancy/drag along the deformed material
coordinates. It also needs a documented rheology/thermal provenance and a
constitutive localization or detachment condition, with its dissipated work
accounted for. Tests must cover material/thermal accounting, positive power,
mesh splitting, timestep convergence, reattachment and checkpoint/resume.

Those state/geometry changes are deliberately deferred in 0.5. The current
instantaneous scalar failure law is retained visibly; no unsupported stronger
threshold, arbitrary waiting time or Earth-speed target has been substituted.

# Slab sinking: work-conjugate reduced-order design

Research/design note, 2026-09-29. The finite-bend force assembly described below
is implemented in `tectonics/young_slab_sinking.py`, with independent checks in
`tests/test_young_slab_sinking.py` (20 passed). The convex branch solver is in
`tectonics/young_slab_constraints.py` (23 independent tests passed). Production
integration owns checkpoint versioning and material detachment. No parameter
is selected to obtain an Earth-like plate speed.

## Main recommendation

Use accepted, still connected material to supply gravitational work. Solve its
viscous sinking and bending resistance simultaneously with the existing basal
drag. A force fraction applied after the solve, a velocity cap, or subtracting
an old-speed resistance would not establish that balance.

For a local trench with outward unit radius `r`, horizontal unit direction `n`
from the subducting plate to the overriding plate, and dip `theta`, define

```
v_s = R omega_s x r
v_o = R omega_o x r
q   = n . (v_s - v_o)
t   = cos(theta) n - sin(theta) r
v_slab = v_o + q t
```

`q` is along-slab feed speed, equal to incoming convergence under the
inextensible, prescribed-shape assumption. It is **not** the slab's horizontal
speed at depth. This closure ties trench motion to the overriding plate; it
does not solve rollback as an additional degree of freedom.

Let `M` be the current excess mass of attached material, after warming and deep
transfer. For a straight slab gravity releases power `P_g = g M sin(theta) q`.
For the implemented finite curved slab replace `sin(theta)` by its along-length
mean. Uniform excess mass per length is an explicit aggregate approximation.
Thus the generalized
surface driving force is `g M sin(theta)`, acting with opposite signs on the two
plate variables. Using `g M`, `g M cos(theta)`, or applying the force to only the
subducting plate while using relative convergence for power violates this
particular kinematic closure. A different trench assumption requires a new
work calculation.

The pair force has no net torque for common rotation, as expected for gravity
acting radially on this internal sinking coordinate. Ambient mantle drag still
resists common motion of the connected slab.

## Resistance and the matrix solve

Define a row `a` mapping all plate angular velocities in rad/s to `q`, and a
three-row matrix `J` mapping them to `v_slab`. For a positive bending coefficient
`C_b` and positive-semidefinite mantle resistance tensor `Z_m`, minimize

```
1/2 omega^T D_basal omega - b_basal^T omega
 + 1/2 C_b (a omega)^2
 + 1/2 (J omega - u_m)^T Z_m (J omega - u_m)
 - g M sin(theta) a omega - b_ridge^T omega.
```

Here `u_m` is the explicitly chosen deep ambient velocity. Current prescribed
surface forcing does not supply a resolved deep mantle flow: `u_m=0` in its
fixed reference frame is a transparent approximation. Reusing a shallow basal
velocity as deep mantle flow would be another unvalidated assumption.

The resulting global operator adds `C_b a^T a + J^T Z_m J` to the existing basal
drag. Every resistance contribution is symmetric positive semidefinite. Its
sign is valid for either direction of motion. Solve the coupled `3*N` system;
independent per-plate solves cannot represent the cross terms correctly.

Power checks are direct: `P_bend=C_b q^2 >= 0`,
`P_mantle=(v_slab-u_m)^T Z_m (v_slab-u_m) >= 0`,
and drive work minus all dissipation is zero at the equilibrium target.
The former 0.3 model's 45 Myr velocity relaxation intentionally has a nonzero
residual. The new 0.4 model instead uses the quasistatic solution: there is no
physical plate inertia here that justifies such a long relaxation. Old declared
checkpoint models retain their former behavior.

## Bending closure and parameters

Buffett's thin viscous sheet gives force per trench length

```
f_b = (2/3) eta_b (H / R_b)^3 q
C_b = W (2/3) eta_b (H / R_b)^3 .
```

Both bending and unbending contribute. The more general coefficient is
`W eta_b H^3 / 3 * integral (dK/ds)^2 ds` for a prescribed curvature profile.
The numerical factor above belongs to its approximate curvature profile; it
must not be represented as an exact coefficient for every shape.

The implementation uses an explicit smooth curvature profile instead of applying
the fully developed approximate coefficient to a newborn slab:

```
l_b = 2 theta_infinity R_b
kappa(s) = theta_infinity/l_b * (1 - cos(2 pi s/l_b))   for 0 <= s <= l_b
kappa(s) = 0                                          for s > l_b
theta(s) = theta_infinity * [s/l_b - sin(2 pi s/l_b)/(2 pi)]
```

The angle is constant beyond the bend. Maximum curvature is exactly `1/R_b`.
Integrating `kappa'(s)^2` only over accepted length gives the bending Rayleigh
coefficient; tangent velocity and gravity are integrated over that same existing
curve. The straight tail is integrated exactly. At fixed thickness and density,
as a short slab length approaches zero its drive scales as `L^4`, bending
resistance as `L^3`, and mantle resistance as `L`. No birth switch is needed.

There is one hinge per physical contact, not per thermal cohort. Production uses
incoming cold mantle thickness when positive. A zero newborn source cell falls
back to retained `sum(A_i H_i^3)/sum(A_i)`, explicitly treating the connected old
hinge as still present. A chemical ocean parcel with no cold mantle has no
operator in this thermal-mantle-only closure.

Neither effective bending viscosity nor bend radius is dynamically resolved.
Explicit new assumptions are `eta_b/eta_m=100`, `R_b/H=3`, and shear distance
`ell_m=D/2`. Their values are not fitted to the output speed. A dimensionless
viscosity and curvature description avoids importing Earth's kilometre scale;
it still needs sensitivity runs and is not a measured satellite property.

Do not automatically use the shell's cold Arrhenius viscosity (up to 1e28 Pa s)
for a faulted bending hinge: it represents another constitutive assumption.
Likewise its 12 MPa tensile damage strength is not a measured plastic shear
yield stress. Wu et al. discuss effective bending viscosity as incorporating
unresolved damage/non-Newtonian behavior and show strong sensitivity outside
roughly 1–100 times mantle viscosity in their Earth analysis. Those Earth
constraints do not calibrate this satellite.

## A finite, extensive mantle-drag approximation

A transparent first closure is a two-face viscous shear envelope:

```
Z_m = C_m I
C_m = 2 eta_m A_attached / ell_m .
```

`ell_m` is a physical shear-decay distance, for example an explicit fraction
of the physical mantle depth. This is a Couette-type reduced-order estimate,
not a solved three-dimensional Stokes flow. An anisotropic tensor is possible
later when normal displacement and lateral return flow are resolved.

Use Genesis's existing `mantle_viscosity_pa_s(T_m, params)` and physical mantle
depth, not numerical slab-length caps. The default shear distance must be
recorded. A viscosity/length multiplier sensitivity is required because this
flow geometry is not predicted by the present model.

Advantages: zero accepted area gives zero drag; splitting a section into
identical smaller widths preserves its total operator; increasing material
increases both its buoyancy and drag rather than producing unlimited force at
unchanged resistance. **Do not apply separate Stokes-sphere drag to every mesh
edge:** its non-extensive size scaling would change the total force when the
mesh is refined.

Mantle drag based solely on `q` would ignore common translation of slab and
overriding plate through the mantle. It is only defensible if a specifically
trench-comoving ambient-flow approximation is declared. The full `J` mapping
above avoids that omission at little extra matrix cost.

## Inventory and finite geometry requirements

- Aggregate current connected contact and plate polarity, not historical
  plate-pair totals. Use current trench directions; keep cancelling directions.
- Accepted area divided by actual trench width defines the equivalent slab
  length. Heated material still has volume and viscous resistance; thermal
  buoyancy alone decreases. Detached/unresolved or physically deep-transferred
  cohorts must not pull a surface plate.
- Reattachment cannot be an unconstrained nearest-boundary search: a slab
  behind an unrelated region would transmit forces through no material bridge.
- A prescribed dip is a closure. Age-based dip maturation is not an evolved
  force-balanced shape. For a very short accepted slab, a fully mature finite
  dip/bend should be identified as an initiation approximation.
- The inventory currently represents positive thermal mantle excess mass.
  Compositional buoyancy of oceanic crust is potentially important for very
  young thin lithosphere. Its correction needs a consistent mantle/crust
  reference-density provenance; the repository currently contains several
  density conventions. Do not substitute an arbitrary Earth density contrast.
- Signed buoyancy is permitted in the force balance. Clipping a buoyant slab to
  zero removes gravitational resistance and biases the result toward sinking.
  If only thermal anomaly is implemented, label that limitation explicitly.
- `q<0` describes withdrawal in the prescribed geometry. The current accepted
  inventory does not return material to the surface, so appreciable persistent
  negative feed is an eduction-model limitation, not evidence of a solved
  reversible material law. Do not hide it with a speed classification gate.

## Meaningful checks before making the mode default

1. Analytic two-plate scalar case: expected `q` and target power balance.
2. Positive dissipation, symmetric operator, nonnegative eigenvalues, and a
   vanishing buoyancy work contribution under common rotation.
3. Invariance to plate labels, global spatial rotation, and splitting an
   otherwise identical section into smaller widths/cohorts.
4. Zero accepted material gives exactly the former basal/ridge solution.
   A heated slab loses drive but retains viscous resistance while connected.
5. Larger bending viscosity raises bending dissipation and lowers feed in the
   scalar reference problem; a smaller dip reduces its gravitational work.
6. Checkpoint/resume, changing mesh, removal of a contact, and deep-transfer
   volume conservation. Old checkpoints keep their declared force model.
7. Runs at more than one time step and grid size, separately reporting speed,
   accepted volume, connected inventory, target residual, gravitational power,
   and both resistance powers. Parameter sensitivity is separate from numerical
   convergence. Neither establishes full geophysical validation by itself.

## Irreversible feed constraint and finite tensile neck

The unconstrained force law permits negative feed: it then consistently consumes
gravitational work to withdraw material. The accepted-material ledger does not
yet have an eduction transaction. If production imposes `a omega >= 0` to select
an irreversible branch, it is an explicit no-eduction boundary condition, not a
result of the smooth bending or mantle-drag law. KKT multipliers must be reported
as additional constraint reactions; active reactions do zero work at zero feed.

Without a strength bound that condition could support unlimited tensile reaction
and prevent opening. The adopted integration instead checks total transmitted
neck force, including gravity and mantle/bending resistance as well as the
extra no-eduction reaction:

```
T_neck = F_gravity - C_b q - mantle_feed_drag_row . omega + lambda
mantle_feed_drag_row = (2 eta_m W / ell_m) integral t(s)^T J(s) ds
T_limit = tensile_strength_pa * W * H
```

The force has units N. The mantle row has units N s, since its input is rad/s.
This total neck test also catches excessive pull during positive feed; checking
only the no-eduction multiplier would miss that failure mode. Production uses
the existing live Starter tensile-strength field rather than inventing a new
Earth yield stress, detaches failed contact inventory without deleting its
volume, and re-solves. Compression is not a tensile failure.

This is an ideal rigid, brittle tensile connection. It is not a resolved
viscoelastic neck, neck-thinning instability, fracture-energy release, or
eduction model. Report unconstrained negative feeds, active constraint count,
reaction torques, neck force/strength ratios and detached volume. These make the
effect of the imposed irreversible branch visible.

The constraint helper solves the scaled dual nonnegative least-squares problem
using rank-revealing BVLS subproblems. A normal-equation NNLS implementation
generated a spurious near-null circulation of roughly 1e33 N in an actual
rank-deficient trench configuration; this was a numerical failure, not slab
strength or a physical driving force. The objective and force bounds are
unchanged by the BVLS replacement.
Dependent constraints can admit multiple force allocations even though velocity
is unique. A second minimum-complementary-energy problem selects the reaction
that minimizes `sum(lambda_i^2/(W_i H_i))` without changing generalized torque.
Thus splitting an otherwise identical hinge divides its force in proportion
to cross-sectional area and preserves its tensile stress/failure threshold.
The helper reports primal feasibility, stationarity, complementary work and
unconstrained feed; it never mutates material.

SLSQP locates the bound-active support for that secondary distribution. Its
absolute equality stopping error must not be accepted as a physical torque
error: another actual case amplified a roughly 5e-13 equality error into a
roughly 2e-9 normalized torque error. The final implementation re-solves the
positive support by exact least squares and independently checks nonnegativity,
equality feasibility and complementary-energy KKT conditions. The original
physical torque guard remains in force. Both actual inputs are retained as
small JSON regression fixtures, with tests against an independent primal QP,
constraint permutation and section-split tensile-stress invariance. A further
100 randomized four-plate constraint systems and ten spatial rotations of
each recorded case passed the numerical audit.

## Primary references

- [Buffett (2006), Plate force due to bending at subduction zones,
  doi:10.1029/2006JB004295](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JB004295).
  Thin-sheet force/dissipation relationship, bending and unbending, and the
  approximate `2/3` law. Also explicitly distinguishes slab weight from force
  transmitted after other resistance.
- [Wu et al. (2008), Reconciling strong slab pull and weak plate bending,
  doi:10.1016/j.epsl.2008.05.009](https://www.clintconrad.no/papers/Wu_etal_EPSL2008.pdf).
  Effective bending viscosity interpretation and Earth-specific sensitivity to
  viscosity and observed curvature. These are not satellite measurements.
- [Li and Ribe (2012), Dynamics of free subduction from 3-D boundary element
  modeling, doi:10.1029/2012JB009165](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012JB009165).
  Distinguishes ambient-viscosity-controlled sinking from internally resistant
  slab bending. Its stiffness parameter includes viscosity ratio and the cube
  of thickness divided by bending length.
- [Khabbaz Ghazian and Buiter (2013), A numerical investigation of continental
  collision styles](https://earthdynamics.org/papers-ED/2013/2013-KhabbazGhazian-Buiter-GJI.pdf).
  Equations 13–15 separate viscous bending, mantle resistance, and compositional
  buoyancy; their order-of-magnitude drag is proportional to ambient viscosity
  and velocity. This supports the scaling, not the exact geometry coefficient
  of the proposed shear-envelope closure.
- [Bercovici et al. (2018), A simple toy model for coupled retreat and detachment
  of subducting slabs](https://people.earth.yale.edu/sites/default/files/files/Long/bercovici_et_al_2018_jg.pdf).
  An explicit reduced-order example separates mantle shear and normal drag,
  trench motion, slab neck strength, and geometry-dependent mantle-flow scales.

The matrices and the moving-trench power derivation in this note are derived
for the proposed repository closure; they are not quoted formulae or a claim
that these papers validate this implementation.

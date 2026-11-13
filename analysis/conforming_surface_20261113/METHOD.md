# Bounded shared-boundary prototype

This stage tests the minimum geometric/material closure suggested by the
10.7% gap / 10.6% overlap diagnosis in `contact_time_20261113`. It does not
reinterpret a rotated old checkpoint as a closed surface. The implementation
is a reusable `tectonics.conforming_surface` operator using the project's mesh,
material arrays, compensated ledger and native zero-time inventory. Its own
restart format is explicit; the existing global runner remains unchanged.

## Alternatives and selected limit

* Independent rigid footprints preserve each area, but produce open/overlapping
  surface under general independent rotations. Raster reassignment conserves
  totals without defining mechanical contact or filling that space.
* Shared normal motion with independent tangential sides is the longer-term
  regional target. It also needs separate material coordinates on each side
  and a work-conjugate contact law; current Euler poles do not determine it.
* Here shared vertices carry **one complete tangential velocity**. Adjacent
  regions are welded, deformable material patches with fixed labels. This is a
  deliberate no-slip limit, not a transform-fault model or independent rigid
  plates. All triangle faces share exactly the same edges; changing physical
  areas changes column thickness while volumes and identities stay attached.

Only oriented minor spherical triangles are allowed. Total area must be 4πR²;
it is checked, never renormalized. Negative/folded cells and impending collapse
are rejected. A degree-one orientation-preserving deformation of this closed
mesh tiles the sphere; an independent point-coverage check supplements this
topological/area invariant. No remeshing or new boundary creation is supplied.

The source literature distinguishes conforming Lagrangian/ALE interfaces from
Eulerian tracking and discusses their stability limitations. See the
[Underworld technical note](https://www.underworldcode.org/free-surface-in-underworld/)
and [Gray, Tackley & Gerya (2026)](https://gmd.copernicus.org/articles/19/8877/2026/).
They motivate checking geometry and conserved material separately; neither
validates this prototype's coarse rock rheology or its parameters for the Moon.

## Two explicit mechanical controls

1. **Prescribed motion comparison.** Project the old regional Euler velocities
   onto a shared vertex field, with incident physical-area weights. Each step
   follows the resulting node rotations. This minimizes a weighted velocity
   mismatch; it changes the old kinematics and is not a force solution. Work
   of the imposed deformation is independently integrated with 5/9-point
   Gaussian quadrature of the potential gradient along the geometric path.
   It is recorded as external work, not obtained by balancing a heat residual.
2. **Unforced gravity/drag control.** Derive tangential forces from finite column
   gravitational energy. Basal resistance is β times velocity, with β in
   Pa·s/m. Positive dissipation goes into an explicitly recorded crustal heat
   increment. No prescribed mantle velocity, sustained external work or old
   dimensionless force proxy drives this control.

The base radius is fixed. Felsic and basalt fractions denote side-by-side
unresolved columns; their footprint fractions follow the material cell, while
their volumes and ages remain independent. With A in m², V in m³, h in m and
constant prescribed g in m/s², exact radial-sector measures are

```
V = A [h + h²/R + h³/(3R²)]
U = ρ g A [h²/2 + 2h³/(3R) + h⁴/(4R²)]
(∂U/∂A)_V = −ρ g [h²/2 + h³/(3R) + h⁴/(12R²)]
```

There is no floating isostatic base, mantle deformation or self-gravitational
readjustment. The fixed radial reaction does no work. The explicit validity
domain is h/R < 0.02; basal-drag-dominated creeping motion excludes internal
shear strength, bending and brittle failure. The underlying shell is continuous
and the material patches are deformable. It is not elastic lithosphere.

For unit-sphere vertex x_i, the exact spherical-area derivative gives G_i=∂U/∂x_i
in joules. Tangential force is −P_i G_i/R in newtons. Lumped incident area A_i
gives angular resistance β A_i R² in J·s:

```
dx_i/dt = −P_i G_i / (β A_i R²)
D = Σ |P_i G_i|² / (β A_i R²) >= 0, in watts
dU/dt = −D;  dQ/dt = D.
```

Subtracting a uniform pressure before gradient assembly is a gauge because
ΣdA=0; it prevents cancellation and does not remove a residual net torque.
Uniform rotations must independently give zero potential work and internal
torque. The initial potential is newly declared from the specified columns;
the earlier inventory did not track gravitational energy, so this is **not**
a conservative conversion of an established old total-energy history.
Inherited sensible heat is retained, with Q as a separate increment.

R=5287 km, g=7.12 m/s², densities 2800/2900 kg/m³ and cp=1200 J/kg/K are from
the existing synthetic configuration. β=1e17 Pa·s/m is a new declared control
parameter, with sensitivity at 1e16 and 1e18, not lunar calibration. It can be
interpreted as η/d for a simple shear layer; neither η nor d is inferred.
Changing β changes the time scale, not the available initial energy.

## Numerical and physical acceptance

Projected explicit Heun updates shared vertices and integrates D using the
same two stages. The **remaining energy defect is reported**, not added to Q.
Step doubling controls vertex displacement, column thickness and the sum of
absolute half-step work defects; rejected trials do not alter material stocks
or accepted geometry. Defaults are coarse-model accuracy controls, not
micrometre or millikelvin accuracy. Restart stores next step and the complete
ledger. Fixed-step runs separately measure numerical convergence.

Independent controls: radial integration of U, finite-difference virtual work,
rotational covariance, positive dissipation, quiet uniform layer, and the
linear spherical-harmonic diffusion limit. For a homogeneous column define
J(h)=h+h²/R+h³/(3R²); small degree-l perturbations decay at
λ=ρg J(h0)² l(l+1)/[β J'(h0)R²]. Mesh convergence is measured against this limit.
Smooth initial fields use one common fine reference quadrature and conservative
aggregation, so comparisons do not silently change total mass with grid size.

All labels and extensive inventories are retained; no fracture threshold is
chosen. Source creation, recycling, external cooling and topology events stay
off in this prototype. Their old laws depend on velocity jumps between rigid
labels and cannot be reused unchanged when the normal boundary velocity is
common. A later regional coupling must derive fluxes from the deforming finite
boundary zones and their work, and explicitly address tangential slip.

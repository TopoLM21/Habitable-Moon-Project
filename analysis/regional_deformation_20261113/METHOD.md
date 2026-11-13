# Regional deformation and thermal/material work contract

Parent: `e8f9dded7a961673f956e382ffaf072a1025645d`. This stage preserves the
conforming reference and existing production runner. It adds a coarse viscous
dilation law, then couples its dissipation to the existing finite two-box
thermal inventory. No independent Euler motion is imposed.

## Constitutive choice

For tangential physical velocity v, the Rayleigh dissipation is

```
D = beta integral |v|² dA + zeta integral (div_s v)² dA
zeta = beta ell²;  beta [Pa s/m], zeta [Pa s m], ell [m].
```

This is the dilatational part of an effective viscous surface-layer law,
with basal drag supplying resistance to all velocity modes. It excludes
elastic storage, shear strength, slip, failure, vertical momentum and mantle
convection. The physical length ell is explicit and independent of cell size;
it must not be selected separately on each grid to force agreement. Values
0, 500, 1000 and 2000 km are synthetic sensitivity controls, not calibration.
The Boussinesq--Scriven surface-fluid literature establishes an energetic
framework for such surface viscosity, not its validity for lunar rocks:
[Barrett, Garcke & Nürnberg (2014)](https://arxiv.org/abs/1404.5519).

Let J be the exact derivative of each spherical triangle's unit area with
respect to two tangent coordinates at its vertices, s its unit area and w
the incident lumped nodal unit area. The positive definite force solve is

```
[diag(w) + (ell/R)² J^T diag(1/s) J] qdot = -grad(U)/(beta R^4).
```

The same J defines area change, the virtual work and the dilation heat.
It is not an extra force filter followed by unrelated material fluxes.
Potential and geometry remain those of the finite radial column reference.
Both dissipation terms are independently evaluated; their sum is compared
with -grad(U) dot qdot. The finite-step energy error is reported, not heated.
In the small degree-l spherical-harmonic limit, the previous gravity/drag
decay rate is divided by `1 + ell² l(l+1)/R²`, an independent analytic check.

## Thermal and material interface

The old `MaterialExchangeBudget` conserves mass and sensible enthalpy but
explicitly excludes mechanical energy. It cannot pay the gravitational cost
of depositing new material merely because its donor is hot. The new coupling
must deposit actual viscous dissipation once, retain all signed thermal
fluxes, and report both the mechanical defect and the combined energy defect.
Its thermal owner is the existing finite mantle/crust model, not a second
duplicate inventory in the surface operator.

Transfer controls use material volume changes at fixed actual area and keep
the existing donor-enthalpy/provenance operator. Independent work integration
uses `dU/dV = rho g h`; a positive placement work must have explicit funding.
An unfunded request is rejected atomically. Returning material to the declared
base reference can release gravitational energy as heat. This reference is
an effective shallow depot at R, not a resolved deep mantle or subduction law.
Prescribed transfers and explicitly supplied work are controls, not spontaneous
crust generation. No latent heat, mineral partitioning or hot melt availability
is inferred from the inherited sensible-heat model.

Automatic creation/recycling and fracture remain disabled until their work
source and kinematic relation to deforming boundary zones are justified.
In a welded surface, the two traces of boundary-normal velocity are equal;
reusing an old jump between independent plate rotations would invent opening.
Area expansion/compression is diagnosed directly from J and actual areas.

The acceptance sequence is analytic work/rate/rotation tests, material and
thermal controlled transfers, exact restart, then identical physical smooth,
oblique and sharp initial states across grids and time steps. Spatial error,
time error and parameter uncertainty are reported separately. Earlier
protected artifacts and benchmark outcomes are never overwritten.

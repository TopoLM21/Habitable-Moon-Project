# Finite material/thermal reconstruction, 13 November 2026

Parent source: `06a077af58bc0731017ad33e4bc559a0482ada4d` on
`reconstruction/regional-20261113`. This is new reconstructed work, not a replay
of the lost regional solver or its unconfirmed paid-cut continuation.

## Bounded change in the existing v0.31 runtime

The optional `material_exchange.enabled` mode adds explicit extensive basalt
volume to the existing continental/sediment volumes and conservative parcel
transport. A winning source carries its entire volume, even between unequal
cells; thickness follows volume/footprint. Lost parcels are explicit recycling
requests, gaps explicit production requests. Covered basaltic basement stays in
the inventory. Legacy mode continues to reset visible oceanic thickness to its
reference value and is unchanged for reproducibility.

The arc maturation equation dP/dt = S - P/tau is integrated analytically for
piecewise constant forcing. Threshold time and the remaining interval for
thickening are computed inside each step. The old per-step thickness cap becomes
a rejected increment in this opt-in mode, not a clipped production rate. This
does not localize all coupled events or remove raster transport aliasing.

The experimental thermal owner has two temperatures and three mass pools:
mantle, continental/sediment rock-equivalent, and basalt. The latter two share
one crust temperature. The existing mantle thermal mass is partitioned to fund
the crust: total mass and total sensible energy do not increase. Initial crust
temperature is the steady conductive midpoint; mantle temperature is solved to
preserve the old total energy. This is a declared new initialization, never a
reconstructed crust thermal history. The unmodelled planetary mass is unchanged.

For spherical area A, mean crust thickness h = Vcrust/A and the existing effective
thermal boundary-layer thickness L, conductances are kA/(L-h/2) between mantle
and crust and 2kA/h between crust and surface. They split the old resistance
L/(kA), rather than count it twice. Require 0 < h <= L. Heat exchange can reverse.
Radiogenic input is allocated by current pool mass; prescribed tidal heating
enters the mantle and is explicitly an external orbital input. Surface loss
leaves the system. Exponential midpoint with step doubling controls the thermal
update; an independent high-accuracy ODE and reversible two-body solution test it.

Every geological production/recycling request transfers m*cp*Tdonor and conserved
origin masses. Production is processed first, returns second; insufficient donor
inventory rejects the request. The outer geological split is first order and
must be checked separately from thermal tolerance. No constant replenishment,
residual-to-heat repair or duplicate thermal owner is allowed. Native checkpoints
retain high/low energy sums, origin masses, basalt volume and controller state.
Old checkpoints cannot silently acquire an invented enthalpy history.

## What this budget does NOT establish

Sensible heat uses constant cp and fixed rock-equivalent densities (2800 and
2900 kg/m3). Sediment porosity, melting/latent heat, chemical fractionation and
pressure work are absent. Provenance is conserved globally among three origin
classes, not a spatial element-by-element chemistry model. Apparent basalt
creation and felsic maturation remain prescribed coarse process requests, not
predicted melt extraction. Plume magmatism must be disabled because its extra
igneous reservoirs are not yet reconciled with this owner. Sediment/topography
densities are legacy proxies, not a validated mass/volume EOS.

Legacy forces/velocities, collision coupling, rift damage and relief remain
kinematic/scaling approximations. Their mechanical and gravitational energies
are not closed by this sensible-heat ledger. Hence nonempty-region splits,
welds and microplate absorption are disabled in the new mode with
`plate_topology.allow_topology_changes: false`. Empty label compaction after
transport is still permitted; it is not physical fracture. Disconnected labels
are possible and must not be described as connected independently formed plates.

An honest paid cut needs a dimensional potential depending on real geometric
degrees of freedom, reference/viscous material memory, reciprocal forces/work,
and a boundary fracture cost. Those lost components are absent in this checkout.
Debiting total mantle heat or splitting a label would not reconstruct them.
This stage preserves the old topology control and does not claim paid fracture.

## Sources and scope of validation

- Korenaga (2008), *Urey ratio and the structure and evolution of Earth's mantle*,
  https://doi.org/10.1029/2007RG000241: planetary thermal budgeting separates
  heat production and loss; it does not validate these prototype coefficients.
- ASPECT official heating-model documentation,
  https://aspect-documentation.readthedocs.io/en/v3.0.0/parameters/Heating_20model.html:
  phase-change latent heat and shear/adiabatic heating are separate physical
  terms. We do not claim that sensible enthalpy alone includes them.

The series-resistance reduction and finite-pool discretization here are our
explicit engineering approximation, not a fitted lunar constitutive law. No
water/redox/composition is selected. Machine conservation is not geological
accuracy. Compare thermal, transport and threshold errors separately. Independent
random initial meshes are scenario sensitivity, not strict spatial convergence.

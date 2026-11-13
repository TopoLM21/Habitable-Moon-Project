# Persistent material areas and direct strip intersections

This is a numerical continuation of the reconstructed regional inventory, not
recovery of the lost historical tectonic solver. Source widths, effective
production rates, maturation thresholds and the finite sensible-heat owner are
unchanged. Previous point-column and legacy modes remain available.

Each regional material parent is a spherical triangle in reference coordinates.
A quaternion stores its owner's rigid rotation. Near a physical source strip,
triangles are subdivided geodesically until a specified numerical edge length
is reached. Child areas use the spherical metric. Extensive inventories and age
moments split conservatively; maturation potential is prolonged as the existing
piecewise-constant field. It is never reset, averaged back from the map, or
coarsened away. This is not a material fracture or a new tectonic plate.

The coarea strip rule computes exact longitudinal intervals cut by the three
triangle half-spaces at each cross-strip distance. For unit angular coordinates
the measure is `R² cos(delta) length(delta) ddelta`. Vertex latitudes, edge
extrema and intersections with finite strip ends divide the remaining 1D
quadrature into smooth pieces. Adjacent quadrature orders provide a diagnostic
error estimate. The complementary material regions are integrated separately;
the source residual is retained and checked, not normalized into matter or heat.
Post-breakoff caps retain their earlier explicitly reported point quadrature.

Local refinement avoids assigning the full parent area to one sampled source
value. Each leaf still has a finite-volume average and a single effective
maturation variable. Thus the nonlinear threshold has a residual spatial error
that must be measured independently of the much smaller source-integral error.
The numerical edge length is not a modified physical arc width or a rheological
parameter. Increasing the tree budget changes no physics: exhaustion stops the
calculation rather than discarding history.

Basalt line requests are assigned to the same coarse regional reservoirs as
before. Supply is divided among their oceanic leaf areas; their common depletion
rate acts on each available inventory. This retains the declared within-parent
mixing approximation. Source/sink volumes and sensible donor enthalpy still use
the finite mantle/crust ledger; no local heat capacity is counted a second time.

Only parent aggregates are mapped to the receiver mesh, preventing material
quadrature refinement from multiplying the capacitated map's unknowns. That map
still influences the old force and boundary laws. Independently rotating regions
can overlap or leave gaps, and nonlocal map allocation is not a contact solution.
Accordingly this work can improve source integration without proving convergence
of the whole moving mechanical model. Dimensional force/work, latent/chemical
energy, elastic/viscous memory and energy-paid topology remain unclosed.

Initialization remains an explicit zero-time material state; evolved point-only
states are not reinterpreted as recovered area histories. Native checkpoints
store leaf geometry, tree identity, owner rotations, all extensive inventories,
age/maturation memory and numerical settings. Changing the strip rule at restart
is rejected. Validation compares physical source integrals, mature areas, finite
budgets, rotations, mesh/step choices, restart and old-mode regressions.

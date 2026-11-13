# Deformation-driven exchange with a finite pressure store

Parent: bc2ee6aa2b6ea9c824f78f4dba10e1be0ef69fdb, published on the working
branch. This is a new coarse reconstruction experiment, not restored evidence
from the lost session. Old runtime modes, snapshots and reports are unchanged.

## Limited physical closure

Shared spherical material triangles still move from the dimensional gravity,
basal drag and surface dilation solve. No Euler rotations are imposed. At each
accepted outer interval the actual area ratio bounds exchange of existing
basalt footprints:

    q_limit_i = gamma * w_i * V_i * (A_end_i/A_start_i - 1).

Positive q permits insertion; negative q permits return. gamma is a declared
0..1 coarse exchange efficiency. It does not prescribe the attained transfer.
w_i is the area average of a compact distance kernel about the initial actual
shared region-boundary arcs. It uses a physical width, unions overlapping
supports, and is advected as a material fraction. It is not recalculated from
a raster map or a jump between independent regional rotations. Quadrature is
separate from mechanical resolution. This stage does not maintain constant
Eulerian band width under large finite strain or integrate a detailed conduit.

A compliant, well-mixed effective shallow store has mechanical potential

    E_p = C P^2 / 2,  P = P0 - net_volume_out / C.

C has units m^3/Pa, P is Pa, and E is J. The store uses the existing mantle
material/thermal inventory; its reference volume is an applicability scale,
not another mass or heat pool. Withdrawal reduces pressure, return increases
it. No permanent inflow, prescribed pressure, extraction of arbitrary thermal
energy, or hidden external work is allowed. P0=0 starts without pressure energy.
Nonzero P0 explicitly adds a **new declared initial mechanical energy**, not
energy recovered from the old checkpoint or derived from its temperature.
Small compliance strain C*P/V_reference must remain <=0.05 in the controls.
The reference volume must fit inside the existing mantle inventory.

The basic pressure/compliance relation is a coarse approximation with research
precedents in reservoir models, e.g. [Mastin et al. (2009)]
(https://volcanoes.usgs.gov/vsc/file_mngr/file-72/Mastin%20et%20al_MSH%20volatiles_GRL_2009.pdf).
That Earth study supports the use of pressure, finite volume and compliance as
model variables, **not** our global connectivity, activation rule, parameter
values or applicability to this moon. No terrestrial pressure is adopted as a
measured lunar property. The new exchange law below is explicitly our reduced
closure, not a claim to implement that paper's eruption model.

## Passive transfer and independently evaluated work

At fixed end geometry, minimize U_columns(V)+E_p subject to the directional
volume bounds. Since dU/dV=rho*g*h, a scalar monotone pressure root gives the
convex KKT solution: each final volume is the hydrostatic volume at the common
pressure, clipped to its local allowed interval. This is the limit of rapid
passive equilibration of already activated channels, **not a resolved flow
velocity or a constitutive law for solid subduction**. A geometry quota can
remain unused if pressure cannot pay for the requested direction.

Pressure work is integrated exactly over net withdrawal; column work and
irreversible loss are independently integrated with Gauss 8/16 quadrature:

    W_p = integral P dV_net;
    W_column = sum integral rho*g*h_i dV_i;
    Q_hydraulic = sum integral (P-rho*g*h_i) dV_i >= 0.

Convexity guarantees nonnegative total loss for the constrained relaxation.
The straight interpolation is a work quadrature, not a claimed local transient
hydraulic path. Global connectivity and rapid relaxation are explicit strong
approximations; local pressure/flow times have not been established.

Q is deposited once in the existing crust thermal owner. The separately
measured residual delta(U)+Q-W_p is retained as an error, never as heat.
The compliant-store energy changes by -W_p. The surface ledger calls W_p an
external port to the surface; the combined model exposes it as **internal
pressure work**, with zero external mechanical work. Initial E_p and its
compensated changes are included in the complete budget and checked against EOS.

The inherited two-box operator moves donor sensible enthalpy and pool-averaged
material provenance. Returned local age is removed proportionally; inserted
material starts at age zero. Felsic material is only deformed, not manufactured
or recycled here. Basalt is rock-equivalent supply from the finite mantle pool;
melting, latent heat, chemistry, phase availability and solid rheology are
unresolved. Therefore extrusion/return are conditional effective exchanges,
not demonstrated magmatism or subduction. No fracture or new owner is created.

## Numerical and acceptance scope

The inherited thermal/mechanical step is followed by a conservative endpoint
exchange. This outer split is first order, and needs explicit step comparison;
there is no claim of event localization or second-order transfer. The updated
mass and temperature feed the next force/thermal solve. Failed trials are
atomic. The existing geometry and thermal applicability checks remain active.
An unsupported donor exhaustion or compliance limit stops the calculation,
rather than adding matter or extending the pressure law.

Acceptance: independent thin-column solution; equilibrium and blocked flow;
zero-pressure charging and finite-pressure release; positive dissipation;
high-precision small-work comparison; band rotation/integration; material and
energy ledgers; exact save/load/continue; identical-data step and grid series;
bounded long run with actual wall time. Initial pressure, compliance, exchange
efficiency and physical band width are sensitivity parameters, not fitted to
obtain a desired flow, continent or split. Spatial, temporal and model errors
are reported separately. No new GUI or main-runner switch is implied.

# Fractional transport: physical closure and activation boundary

This stage provides an independent experimental ocean-only transport pipeline.
It does not replace the active 0.5 raster solver, change the default mechanics
version, or assert a new dynamically sustained plate speed.

## Why a scalar pre-commit reservation is insufficient

The current raster scheme assigns every noncontinental part of a cell to one
visible plate. Tracked `oceanic_volume_km3` is an extensive chemical reservoir,
but its footprint is still `(1-continental_fraction)*cell_area`. Subtracting an
early slab sink from volume alone makes the surviving material a thinner
full-footprint column. Local mechanics subsequently recomputes cold mantle as
`total_lid_thickness(age)-chemical_crust_thickness`; this can increase cold
mantle thickness after a basalt debit. Directly reducing H is also overwritten
by that refresh. Neither operation represents a removed horizontal parcel.

A second difficulty is lineage. Winner-only `material_source_index` transports
one donor's age and fracture memory; it cannot represent surviving minorities.
`accept_slab_material` also rejects two events from the same source face even
though a fractional donor can split between multiple targets. Reducing whole
raster loss by a global previously-accepted total would debit unrelated donors
and hide duplication rather than reconcile it.

The experiment instead makes persistent sparse material components primary.
Each carries cell, physical plate owner, origin ID, occupied area, basalt and
cold-mantle volume, thermal excess mass, age and unaveraged material fields.
Specific volume/mass-per-area signatures survive splitting, preventing floating
point ratio noise from fragmenting an otherwise identical history. Historical
origin IDs remain registered after loss so they cannot be reused as newborns.

## Exact edge flux and positive transport

Let u,v be unit endpoints of a shared great-circle edge, and let
`n=sign*(u cross v)/|u cross v|` point from cell a to b. At radius R and angular
velocity omega, the signed area flux is

`Q_ab = R² integral (omega cross r) dot n d(alpha)`

`     = R² * sign * omega dot (u-v)`.

The second equality follows by integrating the great-circle tangent. The
endpoint differences telescope around each triangular cell, so rigid rotation
has zero discrete divergence to floating-point precision. This prevents
fictitious opening or collision when differently labeled materials share one
angular velocity. The implementation is independently tested against quadrature
of the original velocity integral and against tracer rotation direction.

For a component of area a in a cell of area A, each outgoing edge receives the
fraction `dt*max(Q,0)/A`; the remainder stays. The timestep is divided until the
sum of outgoing fractions is at most one for every plate/cell. Identical
fractions apply to all extensive quantities; age advances once per substep and
all other material memories follow their actual donor. No global correction,
nearest-donor replacement or tiny-parcel deletion is used.

Flux-form conservation and streamfunction-compatible nondivergent spherical
transport are established numerical principles; see
[Skamarock and Gassmann (2011), Conservative Transport Schemes for Spherical Geodesic Grids](https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_47.pdf).
The endpoint formula and the multiphase overlap rules here are derived for this
repository. This first-order implementation does not claim that paper's
high-order accuracy.

## Local collision and ridge transactions

After advection, arriving area S is compared with cell capacity A.

- If S>A, reject only the excess. Greater negative-buoyancy mass per area has
  subduction priority; age resolves unequal thermal ties. Physically identical
  priority groups share rejection in proportion to their incoming area, so
  arbitrary plate labels do not choose a loser. Every rejected piece records
  actual origin/history, source cell, accepted fraction, time and the surviving
  other plate owners. Receiver fractions are proportional to their retained
  area. A no-receiver rejection raises an error.
- If S<A, fill the actual vacancy with newborn ocean. Newborn ownership is
  divided across positive pre-to-post occupancy losses of the departing plates.
  This local first-order ridge ownership rule is symmetric and remains defined
  for an empty target. The caller supplies chemical thickness and fresh history;
  the kernel requires age, cold mantle volume and excess mass to start at zero.
- Differences at the scale of arithmetic roundoff retain every parcel unchanged
  and are reported as capacity residuals. They do not create microscopic sinks
  or births. In particular, subtracting a fully rejected opposing parcel can
  leave a rounded positive remainder; it must not consume the sole survivor.

Before each immutable commit, retained plus lost fractions must partition every
donor. For each origin and extensive quantity, initial plus newly born material
must equal remaining plus rejected material. The independent validation checks
these per-origin identities, not merely global totals. Births later subducted
within the same requested interval remain in both ledgers with their own IDs.

There is no later raster material commit to “pay” again in this experiment.
A future visible-owner projection must be a view of the same sparse state.
Serializing, refining or reprojecting it cannot emit another acceptance event,
reset thermal ages, or replace minority histories by the visible winner.

Refinement must also use one authoritative cell area. The first actual sub4 to
sub5 probe exposed an inconsistency between conserved parent area and separately
rounded spherical child capacities: a single-owner cell arrived about 6.35e-10
km² above its nominal 16176 km² capacity. The transport guard correctly refused
to call this self-overlap subduction. Child capacities are now the persisted
parent area split by normalized geometric weights, and parcels use those same
weights. This preserves parent material and full coverage simultaneously;
the raw spherical quadrature differs only at its verified floating-point scale.
The transport guard was not relaxed. The exact old failure input remains saved
under `failure_refined_dt1`, and a full 20480-cell co-rotation regression checks
that refinement creates neither subduction nor ridge material.

## Current tests and limits

The transport module has 19 focused tests: independent edge integration,
cellwise zero divergence, common rotation with heterogeneous ages/thicknesses,
independently moving minority components, tracer direction, zero motion,
per-origin budgets, positive fractional acceptance before any raster jump,
CFL substeps, physical polarity/tie symmetry, plate relabeling and temporal
convergence to an independently exponentiated semidiscrete advection operator.
Primitive, snapshot and refinement tests are maintained separately.

Spatial upwind diffusion broadens interfaces, and exact retained histories can
increase component count. These are measured limitations; components are not
silently dropped or averaged to hide them. The transport experiment advects
cold volume and thermal excess mass while aging material; it does not solve new
thermal production, slab thermal decay, rheology or force evolution.

Activation in the real runner still requires component-aware thermal/fracture
updates, basal torque and drag weighted by each owner's actual area, geometric
contact/trench reconstruction within mixed cells, safe plate split/merge rules,
and accounting for newborn crust extraction in the global material/energy
owner. Current cellwise fields and winner-only topology cannot supply these
without further work. A fractional loss at a target cell therefore deliberately
carries no invented trench edge, width or torque axis. The backend is a tested
conservative foundation, not a claim that these consumers have been converted.

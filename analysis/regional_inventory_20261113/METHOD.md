# Independent regional material inventory: limited reconstruction

Parent source and evidence: 9e6628a9d8069c7c9618c4e689c8208003d4b94d.
This is a new optional regime in the existing v131 runner, not recovery of the
lost regional code. Prior controls remain available and unchanged.

## Chosen approximation

Moving regional columns own positive material quadrature areas, basalt/felsic
volumes, their volume-age moments and a basalt-mass-weighted maturation state.
Columns rotate continuously with their owning regions; their histories are
never replaced by map-cell averages or a key naming a pair of plates.
The initial material quadrature is a nested subdivision of a new zero-time
state; no evolved history is migrated. Refinement and receiver resolution are
chosen independently. This remains a coarse column model, not seam mechanics.

The areas are fixed integration weights. They are NOT explicitly resolved
free-boundary polygons, ridge-opening space or independent plate fragments.
Basalt supplied at a ridge enters finite columns; their volume/thickness changes.
Thus this stage tests active-material memory and conservative mapping, not a
physical solution of newly opened surface area. This limitation cannot be
removed by a conservation test alone.

Physical arc strip/cap fields are evaluated before receiver-cell averaging.
For basalt V'=S-kV. Its maturation moment Y=V*P follows
Y'=q*V-(k+1/tau)*Y; newborn basalt has zero maturation history. Both frozen
equations are integrated analytically (a positive quadrature is used only for
a cancellation-prone small-k divided difference). Threshold time is localized,
including an interior maximum that an endpoint test would miss. Maturation
returns the remaining basalt and creates only its column's oceanic footprint
as juvenile material, paid from the same finite global inventory. Existing
felsic material survives and thickens. Delamination retains the old effective
threshold/rate and returns material. No fracture is claimed.

Basalt boundary requests are line integrals assigned to same-owner columns.
Within a column, new/old material is a declared mixed reservoir; stored age is
an exact moment, not a recovered complete cohort distribution. Regional
composition is basalt/felsic; global origin classes still belong to the existing
finite inventory. Elemental chemistry is not resolved.

Projection is a positive capacitated transport problem minimizing angular
distance to receiver cells. Material rows are conserved independently of
cell capacities; the latter have a checked tolerance. It does not feed its
mixed history back into the columns. Projection distance measures the spatial
averaging needed to cover the map; it is not physical displacement or work.
If a material region becomes invisible or the graph cannot satisfy capacities,
the runtime fails rather than deleting it. This map is not a contact solution.

## Explicit restrictions

The regime requires the finite material/sensible-enthalpy owner, integral source
law, and disabled nonempty-region topology. Sediments, plume/craton evolution,
late rifts, gravitational collapse, continental erosion, rift loading and tidal
damage must be explicitly disabled: their memory transfer to the new columns
is not implemented. This isolates the authorized inventory question; it does
not silently claim all old processes migrated. Comparisons must use these same
conditions. Previously initialized diagnostic craton/plume fields may remain
as frozen initial properties, not an evolved new history.

The old force proxies, cooling/activity coupling, subduction memory and two-box
heat owner still run. Sensible enthalpy follows transferred mass at the common
reservoir temperature; no extra column heat capacity is added. Latent, chemical
and mechanical/GPE energies remain unclosed, and no artificial heat pays for
fracture. A source magnitude or threshold is never fitted for a positive event.

## Tests before interpreting long runs

Independent ODE and transient-event controls, positive projection, finite donor,
rotation, age/history conservation, receiver-grid and material-grid comparisons,
native restart and old-mode regressions. Completed measurements and failures
will be recorded in RESULT.md. The material quadrature and projection are
numerical approximations whose errors must be shown, not hidden as geological
uncertainty. Full mechanical energetics remains conditional on those checks.

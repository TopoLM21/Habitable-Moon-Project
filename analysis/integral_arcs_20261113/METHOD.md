# New reconstruction: physical boundary integrals

Parent: 88d5b52ed72b2f6655d227f500344ac69719064c. These are new calculations,
not recovered results from the lost environment. The old modes and evidence
are retained. The two switches are explicit opt-ins and saved with restart.

## Model change and numerical correction

The old arc painter enlarged a Gaussian to fit the receiving cell and combined
edge samples by a maximum. That altered the total source when the mesh changed.
The new approximation integrates additive, oriented trench strips of fixed
physical width. For a geodesic segment, the surface measure is
cos(d/R) ds dd. The Gaussian has sigma 140 km, cutoff 320 km, and the existing
slab-dependent landward offset. Each source's dimensionless amplitude uses its
actual normal convergence/reference speed (no minimum-speed production floor).
The baseline reference rate and all physical thresholds are unchanged.
Sources from distinct boundaries add, including overlaps. This differs from
the legacy maximum rule and is not a claim of calibrated magma productivity.

Quadrature spacing (200 km) and order (8) are independent of receiving cells.
Points are located inside actual spherical triangles. Source outside the
overriding region is reported, not rescaled back into it. Without an active
boundary there is no invented active point source. The old localized
post-breakoff pulse remains a finite physical cap with measure R sin(d/R) dd dθ.

The integrated field divided by receiving area is used once in the continental
volume equation. This removes an erroneous second continental-footprint factor
in thickening. Maturation preserves old continental volume and creates only the
missing footprint. Its threshold remains a **cell-average approximation**;
source-integral conservation is not proof of event convergence.

Basalt exchange is separately integrated on physical ridge/trench lengths.
At a ridge, source S = v_n L h_ref times the adjacent ocean fractions (half on
each side). At a trench the legacy polarity chooses the receiving mantle's
donor column, with depletion k = |v_n| L/A and V' = S-kV. The frozen update is
analytic, positive and finite-donor. Its age moment M obeys M'=V-kM, including
zero-age births. The corresponding gross source and sink enter the finite
mantle/crust sensible-enthalpy ledger. Raster overlaps retain all basalt
parcels; they are not an extra sink, and raster gaps are not an extra source.
Unfilled oceanic footprint is reported. This is a mixed-column closure;
slab geometry, fractional material boundaries and subcell composition are not
resolved. The raster's old created/subducted *area* diagnostics remain geometric
and must not be interpreted as these new physical volume exchanges.

## Verification plan and limits

Independent analytic quadrature, ODE integration, partition/rotation controls,
finite-donor/age semigroup checks, and mixed-footprint mass/enthalpy controls
precede integrated runs. Refinement comparisons will use a newly initialized
80-cell state conservatively subdivided into 320/1280 cells: same material
regions, Euler poles, total inventories and initial history, rather than newly
randomized worlds. This is initial-state refinement only, not migration of an
evolved checkpoint. Time comparisons and split restart use native checkpoints.
Runs are bounded individually; raw and compact results are kept separately.

The existing reconstruction has sensible heat and material pools, but no latent
or chemical differentiation energy and no common SI mechanical free energy.
Force proxies and prescribed mantle patterns are not a closed convection source.
Topology stays disabled. Paid fracture and elastic memory cannot be honestly
restored from labels or funded by arbitrary removal of bulk heat. No result here
will be presented as a lunar history or evidence for spontaneous continent birth.

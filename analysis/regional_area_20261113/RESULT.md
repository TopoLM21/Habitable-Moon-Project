# Recovered environment and persistent material areas

The previous working copy recovered. Inspection established that the stalled
strip-intersection patch had **not** executed; no calculation was duplicated.
Surviving changes were committed/pushed as `287759e`, then the newly implemented
operator and its initial tests as `4e714a4`. Both remote SHAs were checked.
This is a new reconstruction from `a6e7894`, not reproduction of unavailable
historical results. Branch: `reconstruction/regional-20261113`; main unchanged.

## Implemented and verified

The existing global runner now has an opt-in material-area mode. Persistent
spherical triangles carry volume, area, age moments, maturation and owner
rotation independently of the receiver map. Refinement conservatively splits
these inventories without resetting history. It is numerical quadrature, not
fracture or creation of a plate. The earlier point inventory remains available.

The physical source strip is intersected with actual triangles. Exact
longitudinal interval lengths and a one-dimensional cross-strip quadrature
replace deposition of isolated sample points. The complementary material
domain is integrated independently. No residual is normalized into matter or
heat. `METHOD.md` gives the contract, including the retained within-parent
basalt mixing and point-quadrature treatment of post-breakoff caps (no such
caps occurred in these coupled runs).

No physical source width, rate, threshold, density or rheology was fitted.
Numerical refinement is separate from the 140 km Gaussian source width. These
are synthetic effective laws for the configured 5287 km-radius habitable moon,
not measured properties or an inferred history of Earth's Moon.

## Separate convergence questions

Frozen 100 Myr source: independent continuum mature area is
1,151,975.30723 km². The source and material are initialized explicitly; this
is an operator control, not a self-generated arc or a mechanical fracture.

| Maximum material edge | Aligned mature-area error | Oblique error |
|---:|---:|---:|
| 250 km | -38.368% | -17.366% |
| 125 km | -11.102% | -11.107% |
| 62.5 km | -5.399% | -6.709% |
| 31.25 km | -3.009% | -3.544% |

Orders 8/16 (using paired 16/32 for the actual integral) give the same areas to
roundoff. The previous point-deposition alias is removed; the nonlinear
threshold still has finite-volume error. The finest two orientations use
10,427/10,871 local elements and 5.17/5.76 seconds for each order-16 probe.
This is evidence of approach to the continuum, not a strict event-error bound.

Coupled runs use identical physical initial fields, poles and boundaries,
conservatively subdivided onto receiver maps. All start at zero time; no evolved
point history was migrated. Each is capped at 120 wall seconds; none timed out.

| Comparison, 20 Myr | Felsic thickening, million km³ | Wall seconds |
|---|---:|---:|
| Material edge 250 km; map 320; dt 4 Myr | 1.643231216 | 17.49 |
| Material edge 125 km; map 320; dt 4 Myr | 1.643231216 | 25.93 |
| Material edge 62.5 km; map 320; dt 4 Myr | 1.643231216 | 54.08 |
| Edge 125 km; map 320; dt 2 Myr | 1.412059553 | 36.73 |
| Edge 125 km; map 320; dt 1 Myr | 1.320097011 | 61.98 |
| Edge 125 km; map 80; dt 4 Myr | 1.576706221 | 28.38 |
| Edge 125 km; map 1280; dt 4 Myr | 2.594731100 | 44.08 |

The linear integrated thickening changes by <3e-15 relative across material
edges. This does **not** prove convergence of a threshold event: no juvenile
maturation occurred in these coupled cases. The maximum maturation potential
across them was 0.439, below the unchanged threshold 0.75. Local peak thickness
changes from 36.114 to 36.307 to 36.358 km; integral and peak are different
questions. All runs retain the 12 prescribed material regions, with topology
disabled. No paid fracture or new tectonic plate is demonstrated.

## Confirmed remaining limits

The receiver-map dependence is substantial: +57.90% thickening from 320 to
1280 cells. It is not the strip-integral error. At 8 Myr the full source
integrals on those maps are 14.596 and 14.584 million km²; at 20 Myr they have
diverged to 18.640 and 59.191 million km². The independently checked relative
source residual remains <=9.25e-14.

`source_diagnosis_verified.json` shows the structural problem. All three
initial maps have the same boundary length, 159,538.613 km, and 12 connected
regions. At 20 Myr their boundary lengths are 159,539 / 191,599 / 348,062 km
and their numbers of connected **map components** are 12 / 19 / 114. The
material owners remain 12. A positive, globally conservative capacitated map
does not preserve connected plate geometry. Reading its hard labels back into
the force, slab and source laws creates additional discrete boundary segments.
This identifies a mechanism for the error; it is not a complete attribution
of every difference in the coupled mechanical trajectory. The source residual
and quadrature estimator cannot certify the physical correctness of that map.

Time refinement changes thickening by -14.07%, then -6.51%; full temporal
convergence is not established. There is a concrete inherited endpoint-gating
defect: slab age/depth are advanced before the whole-step source is applied.
The 8 Myr age gate therefore supplies material over [4,8], [6,8], or [7,8] Myr
for dt 4/2/1. It should not act before its crossing. The measured apparent
order 1.33 combines this defect, changing source strength and map dynamics;
it is not a verified algorithmic order. These raw runs are retained as evidence,
not relabelled as converged by decreasing a tolerance or a physical threshold.

## Budgets, restart and old modes

The default 20 Myr run creates 278,553,461.778 km³ basalt, returns
205,522,156.278 km³ basalt and adds 1,643,231.216 km³ felsic material, all as
thickening of existing felsic areas. Finite mantle inventories supply the mass
and donor sensible enthalpy. Mantle temperature ends at 1858.494834 K and the
lumped crust temperature at 410.472756 K. Initial temperature and all powers
are in the saved state/configuration, not reconstructed geological history.

Across the coupled suite, independent sums of native volumes and reaction
ledgers leave at most 3.56e-7 km³ felsic and 1.90e-7 km³ basalt residual.
Rounded pool mass error is <=5.12e-17 relative. Independent 80-digit arithmetic
on the compensated stored terms gives zero origin-mass and sensible-heat
ledger residual. Source support-area residual is <=2.97e-14 relative;
geometric subdivision error <=3.57e-13; receiver capacity error <=1.13e-14.
The quadrature-order difference is an estimator, <=4.21e-9 relative here,
not a mathematical guarantee on all local or future events.

These small numbers certify accounting only. Latent/chemical energy, work of
the proxy force law and gravitational/elastic energy are not jointly closed.
There is no justified energy available for a fracture calculation in this
reconstructed branch. Bulk sensible heat must not be spent as arbitrary work.
The three provenance categories (initial mantle/felsic/basalt) are conserved;
this is not an oxide-resolved petrological model.

Native 12 -> 20 Myr restart equals uninterrupted 20 Myr in all 86 arrays and
metadata. Rotation/refinement preserve material memory, with analytic strip
and rotation-covariance tests. The earlier point-mode 20 Myr regression equals
its protected saved result in all 73 arrays and metadata. Full suite:
**316 passed**, one third-party pytest deprecation warning, 15.38 seconds.
The seven failures mentioned in lost historical environments are not present
in this recovered baseline; they were not silently counted as repaired.

## Cost and practical decision

The default run covers 20 Myr in 25.93 s end to end (16.14 s integration),
0.771 Myr per wall second, at 209.01 MiB peak RSS. Its tree grows from 1280 to
21,995 elements. The 62.5 km check reaches 73,598 elements and 260.0 MiB RSS.
The nine area runs, including the short/restart controls, total 302.69 seconds;
the old-mode regression adds 13.45 seconds. Shared-container timings are
measurements, not dedicated-machine speed guarantees. No 100 Myr/Gyr run or
long-horizon cost extrapolation was performed in this new mode. Persistent
refinement never discards history and can reach its explicit resource ceiling.

The new material-area/source operator is useful and reproducible. It does not
make the whole tectonic model resolution-independent. Further material
refinement is not the next useful task. The shortest next work is to localize
age/depth source activation within the time interval, then stop using
fragmented receiver labels as physical contacts. Forces and line fluxes need
a common coarse material boundary representation, while the capacitated map
remains a display/thermal receiver. Overlap/contact and dimensional work must
be addressed explicitly before adding energy-paid regional fractures.

`convergence.png` and `area20.png` were visually checked. Ten native states
(20 files, 17,434,671 bytes) are copied, not hardlinked, alongside logs, exact
parameters, summaries and source hashes. Prior artifact manifests and the
original baseline are checked by `preservation_check.json`. No old result,
archive, checkpoint or protected data was removed or modified. The incomplete
`source_diagnosis.json` is retained as a labelled reporting-only serialization
failure; use `source_diagnosis_verified.json` for the completed diagnosis.

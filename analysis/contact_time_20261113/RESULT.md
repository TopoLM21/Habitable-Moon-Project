# Source activation and material-contact reconstruction

The premature arc-source activation is corrected in an explicit regional mode.
Material banks now have a contact registry independent of raster labels, with
an audit and a strict geometric validity check. **The contact registry is not
yet a replacement force/evolution solver.** Its independent checks expose a
structural gap: independently rotating rigid material footprints cease to tile
the sphere. Continuing the proxy controls is not a physically closed history.

This is new work on `reconstruction/regional-20261113`, following `df23c61`.
Source localization was committed/pushed as `be47df2`; the bank registry and
initial evidence as `0e45555`; interval-union coverage as `c70daf5`. Remote SHAs
were verified. Main, earlier evidence and available protected files are kept.
Unavailable historical calculations have not been reproduced or relabelled.

## What changed

`tectonics/arc_time.py` follows the existing frozen-boundary slab growth law.
Each outer interval is split at the existing age gate, depth crossing and the
end of a previously detached slab's source pulse. Source strength is sampled
at each subinterval midpoint; material rotation, source-strip integration and
finite-volume reaction span exactly those subintervals. The receiver map is
updated once at the outer end. No interval or energy residual is turned into
artificial heat. No source width, rate, density or physical threshold changed.

Use `regional_material.active_area.source_time_rule: localized`. The old
`endpoint` mode remains for exact regression. A newly occurring breakoff is
rejected explicitly because its detachment time has not been localized. This
is not a general event solver, nor a proof of order for the coupled thermal,
force and map updates. Restart retains and checks the chosen numerical rules.

`tectonics/material_contacts.py` derives two banks per root material boundary
from saved owners, reference geometry and owner quaternions. It pairs actual
coincident, oppositely oriented great-circle arcs, including changed neighbours
under tangential slip. There is no new stock, area, mass or heat in the registry.
Stable identity does not depend on receiver labels or source-tree refinement.
The 1e-10 rad coincidence tolerance is numerical, not a fitted contact width.

An additional audit defect was corrected during review: summed overlap lengths
could let double coverage of one segment cancel an uncovered segment. The
registry now measures the interval union and excess coverage separately.
Independent half-covered/duplicated-bank controls exercise that distinction.
Final saved contact diagnostics are unchanged by this correction; see
`coverage_validation.json`. Old evidence was not rewritten.

`contact_policy: audit_material_banks` records the geometry alongside the
explicit label that forces still use the legacy receiver proxy.
`require_shared_support` checks the current and predicted endpoint banks
before modifying material inventories, and refuses separated or multiply
covered banks. This endpoint guard is not a guarantee against an intermediate
contact event or a complete transactional rollback of every outer-runner
field. A failed run is not saved as an accepted continuation.

## Time and space comparisons

All cases retain the same physical initial fields and parameters, including
the 5287 km radius and the 8 Myr source-age gate. These are synthetic effective
laws for a habitable-moon experiment, not calibrated properties of Earth's
Moon. The previous endpoint method activated the source over [4,8], [6,8] or
[7,8] Myr for steps of 4, 2 or 1 Myr. The corrected cases give **zero felsic
production through 8 Myr** and begin the active interval at 8 Myr. Independent
age/depth-dose tests include a crossing at 10.3 Myr and noncommensurate steps.

| 20 Myr control | Felsic thickening, million km³ | Wall seconds | Peak RSS, MiB |
|---|---:|---:|---:|
| Previous endpoint, 320 cells, dt 4 Myr | 1.643231216 | 25.13 | 207.95 |
| Localized, 320 cells, dt 4 Myr | 1.245652579 | 47.43 | 215.77 |
| Localized, 320 cells, dt 2 Myr | 1.229037131 | 57.99 | 218.00 |
| Localized, 320 cells, dt 1 Myr | 1.228186391 | 90.18 | 216.03 |
| Localized, 80 cells, dt 4 Myr | 1.200852418 | 33.70 | 227.00 |

The consecutive time-refinement changes are -1.334% and -0.0692%, compared
with the previous -14.07% and -6.51%. This demonstrates removal of the early
activation defect; three coupled values do not establish a formal order or
convergence of threshold events. Crust temperature, for example, still changes
410.448 -> 404.953 -> 402.730 K at 20 Myr. Mantle temperature changes much less:
1858.493 -> 1858.485 -> 1858.485 K. Other operators remain time-dependent.

The corrected 80 -> 320 receiver comparison changes thickening by +3.731%.
A 1280-cell attempt has **no accepted result**. Its tool session became unknown;
inspection found no active simulation, no result JSON, no resource result and
no native checkpoint. The configured ceiling was 120 seconds, but the terminal
completion/timeout message was unavailable: neither its actual cost nor cause
is claimed. It was not blindly repeated. Its YAML, empty log and explicit
`gate20_g3_uncompleted.json` status are retained. Thus three-grid spatial
convergence of the corrected evolution remains unestablished. Earlier
1280-cell endpoint results are plotted as earlier results only.

## The physical-contact blocker

The initial material partition has 12 owners, 408 banks and 159,538.613 km of
shared contacts. After independent rotations, coincident line contact is lost;
the 319,077.226 km total bank length is unmatched in the saved controls.
This is not permission to use new raster components as plates or contacts.

An independent spherical sampling calculation checks material-footprint
coverage, without using receiver labels or inferring area from bank distance:

| Time, dt 4 Myr | Samples | Uncovered surface | Multiply covered surface |
|---|---:|---:|---:|
| 8 Myr | 320 | 0.000% | 0.000% |
| 8 Myr | 1280 | 3.486% | 3.519% |
| 8 Myr | 5120 | 4.801% | 4.814% |
| 20 Myr | 320 | 9.462% | 8.871% |
| 20 Myr | 1280 | 10.320% | 10.129% |
| 20 Myr | 5120 | 10.670% | 10.626% |

These are area-quadrature diagnostics, not exact unions or a debit of matter.
At 20 Myr the sampling material-measure residual is +0.0333% of sphere area;
the large uncovered/overlap signal is not that residual. At 5120 samples the
dt 2 and dt 1 cases still have approximately 10.69% uncovered area. Coarse
sampling at 8 Myr can entirely miss it. Tightening a numerical tolerance or
deleting small map components cannot repair the finite geometry defect.

The strict run rejects its first predicted 4 Myr step with
`Material contacts not geometrically closed`. Both initial checkpoint files
retain their hashes and no completed strict checkpoint is written. The
20 Myr runs above are permissive numerical proxy controls. Their mass ledger
can be correct even though their rigid footprints are not a physical surface.

## Inventories, energy scope and restart

The 320-cell dt 4 control creates 278,496,130.815 km³ of basalt, returns
206,304,240.051 km³ and adds 1,245,652.579 km³ to existing felsic areas.
No juvenile area matures: the largest potential in the corrected suite is
0.3232 versus the unchanged 0.75 threshold. All 12 prescribed owners remain
12, with topology disabled. This is not an energy-paid fracture demonstration.

Across eight accepted runs, independent sums of native volumes and reaction
ledgers leave at most 2.62e-7 km³ felsic and 2.02e-7 km³ basalt residual.
Rounded pool-mass error is <=7.23e-17 relative. Independent 80-digit evaluation
of the stored compensated terms gives zero residual for each of the three
origin-mass categories and the explicit sensible-heat ledger. The largest
relative source-deposition residual is 8.14e-16. Full tables are in
`comparison.json` and `comparison.csv`; zeros in these accounting identities
are not physical-accuracy estimates.

The thermal ledger includes finite donor/receiver sensible enthalpy, explicit
radiogenic and orbital input, and surface output. It is not an isolated system.
Latent/chemical energy, gravitational/elastic energy and work of the inherited
non-SI mechanical proxy are **not jointly closed**. No arbitrary subtraction
of mantle heat is used to pay mechanical work or fracture. Provenance consists
of the initial mantle/felsic/basalt categories, not oxide-resolved petrology.

Native 8 -> 20 Myr continuation equals uninterrupted evolution in all 86 arrays
and metadata. The unchanged endpoint control equals the protected previous
20 Myr result in all 86 arrays and metadata. Derived contact identity and
native audit restore agree; analytical tests cover rotating/sliding hemisphere
contacts, changed neighbours, rotation covariance and virtual-work algebra.
That algebraic force/torque check does not supply missing SI units to runtime.

The final suite is **340 passed**, one third-party pytest deprecation warning,
15.54 seconds. Earlier logs with 327 and 338 passes remain evidence of their
own commits. The seven failures mentioned in the lost historical environment
are absent from the recovered baseline; they are not claimed as repaired.

## Cost, preservation and next useful step

The corrected 320-cell dt 4 control covers 20 Myr in 47.43 wall seconds,
37.55 seconds in integration: 0.422 Myr per end-to-end wall second on this
shared machine. Eight accepted executions total 321.20 seconds; strict
rejection, tests and the unmeasured 1280-cell attempt are separate. Peak RSS
among accepted runs is 227.0 MiB. No 100 Myr or Gyr claim is extrapolated.
Source refinement grows the default tree from 1280 to 20,594 material elements;
these are integration elements, not new plates.

Eight native states, 16 files and 10,021,254 bytes, are copied (not hardlinked)
with parameters, logs, summaries and source provenance. `preservation_check.json`
checks 485 earlier manifested artifacts and the 264-file recovered baseline.
Unavailable historical archives cannot be recertified. Both figures were
visually inspected. `RESTORE.md` gives a non-overwriting restart command and
the limits of each contact policy.

Further source-tree refinement is not the shortest path to a usable model.
The recommended next closure is a coarse **conforming deformable regional
partition**: one common normal boundary motion, distinct material coordinates
for tangential slip, actual evolving areas, conserved volume mapped to
thickness, and explicit deformation/work memory. It can stay coarse and does
not require a global 3D solver, but cannot be inferred just from today's
independent rigid rotations. It needs a declared material/force closure and
initialization before energy-paid new boundaries can be defended.

An alternative retains rigid patches and explicitly represents open and
overlapping areas, finite filling/recycling stocks, pressure and their work.
That adds more physical assumptions and material bookkeeping. Neither option
is implemented silently here. Expanding a coincidence tolerance, inventing
matter in gaps or treating overlap residuals as heat would not be a solution.
The completed source correction and independent contact guard are usable;
physical contact dynamics and a converged self-forming plate history remain
the exact uncompleted part of this reconstruction.

# Regional material inventory: verified reconstruction and its limit

2026-11-13. New work in `reconstruction/regional-20261113`; the lost historical
regional experiments have not been recovered or reproduced. Parent completed
stage: `9e6628a9d8069c7c9618c4e689c8208003d4b94d`.

**The material-history and conservation part works in the existing v131 runtime.
The chosen point-column discretization is not yet an adequate coarse tectonic
model.** It removes receiver-cell averaging from maturation history, but narrow
source sampling and the map-derived forces remain resolution dependent. The
global mapping is costly and increasingly nonlocal. These measured limitations
block interpreting the run geologically or adding an energy-paid fracture on top.

## What is implemented

- Optional `regional_material` mode; old modes remain available unchanged.
- Moving columns own area weights, basalt/felsic volumes, volume-age moments,
  owner identity and basalt-mass-weighted maturation memory. Regional histories
  do not get reset when the coarse display cell changes ownership.
- Frozen finite basalt exchange and maturation are integrated together:
  `dV/dt=S-kV`, `d(VP)/dt=qV-(k+1/tau)VP`. New basalt enters with zero maturation.
  A threshold is localized even when an interior peak exceeds it but the endpoint
  does not. Existing effective coefficients are unchanged.
- Gross production/recycling requests pass through the finite mantle/crust mass,
  provenance and sensible-enthalpy owner. No second heat capacity is added.
- Positive capacitated projection conserves material rows and checks receiver
  areas independently. Native checkpoints preserve all regional fields/history.
- Complete exhaustion of a pool transfers its actual compensated provenance
  components and sensible heat. It does not discard a small residual as rounding.

**Area weights are not resolved moving material polygons.** Ridge input changes
column thickness, not explicitly opened geometrical area. The projection fills
map capacities; it is not a solution of contact, overlap, opening or mechanical
work. Its mixed ages are diagnostic only, while its map/labels still affect the
old runtime's force and boundary calculations. That remaining feedback is a
major source of grid dependence.

This control explicitly disables sediment evolution, evolving plume/craton
properties, late rifts, continental erosion/collapse, rift loading and tidal
damage. Their memory transfer to these columns is not implemented. Initially
stored craton/plume properties can remain frozen background fields. Nonempty
region topology is disabled. Material columns are not newly created plates.
Basalt/felsic classification and three global origin pools are not elemental
chemistry. Within-column mixing is an effective approximation, not full cohorts.

## Actual coupled history

Same physical zero-time state as the preceding stage, refined conservatively
from 80 parent cells. No old evolved history was migrated. Twelve prescribed
initial regions, 1280 material columns, 320 receiver cells, outer step 4 Myr.

| Quantity at 100 Myr | Measured result |
|---|---:|
| Basalt created | 2,290,418,073.839 km³ |
| Basalt returned to mantle | 1,916,864,096.865 km³ |
| Felsic material added | 23,806,167.537 km³ |
| Of that, juvenile maturation | **zero** |
| Of that, thickening existing felsic material | 100% |
| Final total basalt / felsic inventory | 2.141344 / 3.478950 billion km³ |
| Maximum actual felsic column thickness | 36.0733 km |
| Maximum displayed dominant-material thickness | 46.2233 km |
| Mantle temperature | 1860.73585 → 1848.25668 K |
| Final crust reservoir temperature | 437.69820 K |
| Final largest maturation variable / threshold | 0.333694 / 0.75 |
| Region count | **12 → 12** |

The maximum displayed thickness is not necessarily continental thickness.
The run contains motion, basalt creation/recycling and felsic thickening;
it does not demonstrate formation of a new continent, plate or cooling fracture.
Its absence of maturation is a result of this discretized history, not evidence
that the physical effective law forbids maturation. Under an independent steady
forcing the same law does mature material, as checked below.

See [regional100.png](regional100.png), visually inspected, for material
quadrature positions, their conservative receiver map and thermal/volume history.
The scatter points are intentionally not presented as resolved coastlines.

## Convergence and uncertainty: negative findings retained

All coupled comparisons below use the same physical initial geometry and stocks,
with no coefficient fitting. Production is in **million km³ over 20 Myr**.

| Material columns | Receiver cells | dt, Myr | Felsic addition | Basalt created | Basalt returned | Wall, s |
|---:|---:|---:|---:|---:|---:|---:|
| 1280 | 320 | 4 | 1.817272 | 278.553463 | 205.522157 | 12.53 |
| 1280 | 320 | 2 | 1.590775 | 280.141875 | 210.926400 | 15.49 |
| 1280 | 320 | 1 | 1.478356 | 280.374873 | 211.109689 | 21.55 |
| 320 | 320 | 4 | 1.187511 | 298.905033 | 254.529629 | 9.65 |
| 1280 | 80 | 4 | 1.783958 | 273.461149 | 178.144543 | 18.50 |
| 1280 | 1280 | 4 | 2.732802 | 426.612980 | 330.819862 | 14.67 |
| 5120 | 320 | 4 | no completed result | — | — | stopped at 120 s |

The temporal differences are approximately first-order over these three steps;
they are not enough to establish an asymptotic order or a converged answer.
Even 2→1 Myr changes felsic addition by 7.6% relative to the smaller-step answer.
Receiver refinement 320→1280 changes it by 50.4% with identical material
resolution. It changes basalt creation by 53.2%. Separating material memory
from display cells therefore has **not** eliminated the mechanical/map feedback.

The frozen source control has a 1-radian arc, offset 105 km, Gaussian sigma
140 km, cutoff 320 km and amplitude 0.7. A separate one-dimensional spherical
integral gives source integral 989,360.403667 km² and mature area at 100 Myr
**1,151,975.307230 km²**. Rotating the whole source leaves this analytic answer
unchanged. Relative numerical mature-area error is:

| Columns | Aligned source | Oblique source |
|---:|---:|---:|
| 320 | +303.79% | −11.00% |
| 1280 | +102.40% | −4.92% |
| 5120 | +80.70% | −2.84% |
| 20480 | +37.10% | −4.43% |

This is quadrature phase error from a narrow thresholded source, not chaotic
plate history. Whole column weights are selected by point samples; they do not
resolve partial intersections with the physical active strip. Increasing the
global column count is an expensive and unreliable way to repair this error.
The independent operator test has no capacitated map and finished in seconds;
its 20480-point check is not a completed global 20480-column runtime.

At fixed 1280-point oblique quadrature, dt 4/2/1 Myr yields the same first event
68.139093987789 Myr and felsic volume 29,256,868.0087415 km³ to roundoff. Thus
the frozen reaction update is not the source of the coupled time-step error.
Changing the maturation rate by factors 0.5/1/2, as an **uncalibrated sensitivity
range**, gives analytic mature areas 0 / 1.151975 / 1.610704 million km².
Physical-law uncertainty and numerical error are reported separately. No value
was chosen to force a positive event.

See [convergence.png](convergence.png), [comparison.csv](comparison.csv) and
[operator_verified.json](operator_verified.json). Full convergence is not claimed.

## Accounting, restart and regression evidence

The native 100 Myr final inventory has rounded pool mass residual
−203,423,744 kg against approximately 2.03×10²⁴ kg: relative −1.002×10⁻¹⁶.
This is explicitly shown, not described as zero physical error. Summing the
stored compensated origin entries and heat ledger independently with 80-digit
decimal arithmetic gives zero origin and sensible-heat ledger residuals for
the completed controls. Maximum receiver-capacity relative error is
1.313×10⁻¹⁴; the 100 Myr felsic-volume ledger residual is 9.537×10⁻⁷ km³.

**This is sensible-heat bookkeeping, not complete physical energy closure.**
Latent heat, differentiation energy, mechanical work and gravitational energy
remain unclosed. Radiogenic and prescribed orbital heating remain explicit.
No arbitrary debit from the mantle's bulk heat pays for fracture. The force
law still uses the recovered normalized/proxy formulation, not a verified SI
stress/work potential. Water/redox and lunar rheology have not been calibrated.

Two confirmed arithmetic/state defects were repaired:

1. The v124 wrapper reinitialized already saved craton fields at load. With
   frozen regional background memory this changed subsequent forces. The old
   resumed 100 Myr result gave 24.101784 million km³ felsic addition instead of
   23.806168. It is retained as `m3_restart100`, explicitly invalid for comparison.
   Complete saved fields are now preserved. Corrected 48→100 Myr continuation
   matches the uninterrupted run **bitwise: 73 arrays and all metadata**.
2. A complete basalt drain could leave ~10⁶ kg of roundoff and fail the check
   against zero, despite ~10²¹ kg being transferred. An explicitly empty target
   now transfers every stored origin component, with its donor sensible heat.
   The request must match the full available pool within the existing 64-ulp
   arithmetic scale; materially inconsistent requests are rejected atomically.
   Both undershooting/overshooting roundoff, a mixed-origin pool, empty whole
   crust rejection and restart are tested. No mass is deleted and no residual
   becomes artificial heat. Independent uniform-forcing exhaustion now passes.

`verified_pytest.log`: **302 passed**, one dependency deprecation warning.
This recovered checkout originally had 218 passing tests. The historical seven
baseline failures belong to the unavailable former environment and are not
present here; they have not been silently marked repaired.

Additional exact comparisons:
- Previous integral mode through 100 Myr: 64 arrays and metadata identical.
- Original legacy mode through 20 Myr: 62 arrays and metadata identical.
- New mode through 20 Myr before/after exhaustion fix: 73 arrays and metadata
  identical. The correction does not perturb the nonexhausting case.
- Reprojecting the saved state preserves authoritative material history and
  reproduces eight projected state fields exactly.

Initial failed test/diagnostic logs are retained. `operator_checks.log` is a
diagnostic-script syntax error; `operator_checks_v2.log` records the real
exhaustion defect. `operator_checks_v3.log` is the development confirmation;
`operator_verified.json` is the repeated check at committed code `db39fed`.

## Cost and architectural consequence

Measured 100 Myr run: **85.36 s end-to-end**, 75.85 s for the integration loop,
210.59 MiB peak RSS; about **1.17 Myr per wall second** end-to-end. Completed
20 Myr comparisons take 9.65–21.55 s, at most 218.57 MiB. The 5120-column test
was stopped at the predeclared 120 s cap, without a usable checkpoint/result;
its peak RSS was not recorded. No claim is made about how far it progressed.
An earlier full dt=2 Myr run was deliberately terminated after the dt=4 cost
became known; the diagnostic is retained in `budget_stop.json`. No Gyr run.

At the actual saved 100 Myr state, one identical conservative projection takes
2.74 s. Explicit wall timers locate **99.52%** inside the two linear-program
calls (8-neighbor graph infeasible; 16-neighbor graph feasible). The initial
cProfile table did not account for native solver time; its zero native total
must not be interpreted as zero cost. `projection_timing.json` records the
independent wall timing and exact output/history equivalence.

More seriously, final RMS projection distance is **1246 km**, maximum 2405 km
for that projection (2562 km over the history). This is far larger than the
140 km source width. Filling the receiver map conservatively does not restore
physical local contact geometry when independently rotated fixed-area columns
overlap or separate. A faster linear solver would not remove that model error.

The next useful step is therefore **boundary-supported area/history integration
at the regional scale**, not a larger global point mesh or more detailed seams:
retain actual partial strip areas and short age/maturation cohorts; integrate
their intersections and exchange conservatively; keep the receiving map as a
view rather than the owner of the physical force/boundary geometry. That requires
a compatible coarse boundary representation in this recovered runtime. Test
rotation-invariant strip integrals and threshold areas before another long run.
Only after that should a common dimensional mechanical potential and explicit
elastic/viscous memory be connected to an energy-paid split. The current result
does not justify adding paid fracture to proxy forces or declaring the broader
tectonic objective completed.

## Durable evidence and continuation

Implementation commits: `7f75817` (regional inventory), `5eacbb4` (saved-memory
load fix), `db39fed` (complete finite-pool transfer). All were pushed and remote
SHAs verified on the reconstruction branch. Main was left at
`831e6e3e0d58e352efab575f561aa677595a4b7f`.

Native states, parameters, logs, comparisons and the two figures are saved in
this directory and its `saved_states/` subdirectory. `saved_states_manifest.json`
records exact file hashes, source commits and invalid diagnostic states.
`artifact_manifest.json` binds the complete compact evidence to implementation
files. `preservation_check.json` verifies older committed evidence and original
checkout inventory. Protected unavailable historical archives cannot be
recertified from this recovered checkout. Nothing was deleted or overwritten
through an archival hardlink. See [RESTORE.md](RESTORE.md) for native continuation.

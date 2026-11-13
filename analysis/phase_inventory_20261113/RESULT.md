# Result: finite phase supply, stage closed

Reconstruction baseline: `ea58c142b8e5f38ed6158e1c8184ff23049bee48`.
This report, code, parameters, raw results and restart files are committed
together on `reconstruction/regional-20261113`. Case `source_commit` fields
record that baseline while the worktree was under test; the release manifest
records the exact runtime bytes. These are new tests, not recovered old runs.

## Implemented and checked

An opt-in phase owner now couples to the existing deforming spherical surface
and finite pressure source. Hot mantle is not automatically molten: supply
requires actual liquid in an explicitly tagged finite inventory. Selective
extraction preserves depletion of each effective constituent and its origin.
Full donor enthalpy includes latent heat. Cold returned basalt must pay for
remelting from the existing thermal stock. Initialization adds **zero mass and
zero energy**. In the smooth case chi=1e-5 changes initial mantle temperature
from 1864.4970169 to 1864.4959752 K at unchanged enthalpy.

The old mode is retained. New phase checkpoints are versioned separately and
cannot be silently interpreted as sensible-only states. Restart preserves
component inventories, all heat/work/provenance ledgers, pressure, geometry,
ages and adaptive-controller memory. Liquid crust is rejected by the welded
solid mechanical mode; no molten solid mechanics is silently continued.

**393 tests passed**, one existing third-party deprecation warning, **20.20 s**
(`release_guard_tests.log`). Fourteen are new. Independent tests cover the Stefan
plateau, reversible enthalpy inversion, selective depletion, full transferred
enthalpy, isolated exchange entropy, a linear implicit two-box limit and a
65-digit reference for small heat exchange. All earlier tests remain passing.
The seven historical baseline failures belonged to the lost environment and
are not present in this recovered suite; they are not claimed fixed here.

An initial test compared a finite backward-Euler heat flux to its zero-time
approximation. Its small discrepancy was removed by replacing the reference
with an independent 65-digit finite-step solution, not by loosening tolerance.
The initial failing log and later passing logs are preserved.

## Short linked results

All 13 linked cases ran to **2 Myr** on the existing spherical surface. Phase
temperature envelope 1700–2200 K, latent heat 400 kJ/kg, retention 0.02 and the
eligible fractions below are explicitly synthetic, not calibrated for the
Moon. Smooth/oblique tests use the previously declared finite 500 MPa pressure
initialization; the native case starts at zero pressure. No new force source
was introduced or amplified.

| Case | Cells | New basalt, km³ | Returned basalt, km³ | Integration time |
|---|---:|---:|---:|---:|
| No initially eligible source, smooth | 1280 | 0 | 0 | 0.645 s |
| Limited source chi=1e-7, smooth | 1280 | 20,688.020787 | 0 | 0.640 s |
| Available source chi=1e-5, smooth | 320 | 49,887.952471 | 0 | 0.217 s |
| Same available source | 1280 | 52,150.702811 | 0 | 0.604 s |
| Same available source | 5120 | 51,970.197450 | 0 | 3.122 s |
| Available source, oblique | 1280 | 80,423.118470 | 0 | 0.570 s |
| Native geometry, zero initial pressure | 1280 | 0 | 1,053,348.699730 | 0.582 s |

The limited smooth case consumes its initial mobile allowance of
5.99952602836e16 kg down to a 2.66e5 kg representational remainder (about
92 m³, 4.44e-12 of the allowance). It does not reset melt availability at the
next step. Remaining retained liquid is separately accounted and is not
available feedstock. The native case charges the pressure source by return;
return is still the inherited effective crust-transfer law, not resolved
subduction or a demonstrated transport law for solid mantle.

The first linked attempt exposed a real rounding error: at exhaustion, adding
tiny quotas to large cell inventories could promise 29,500 kg more than the
mobile stock. Positive volume bounds are now rounded **downward** to a
representable amount; the donor retains sub-ULP requests. No stock tolerance,
geometry criterion or energy threshold was relaxed. The failed log is kept.

### What convergence does and does not show

- With ample stock, 1280→5120 cells changes production by **-0.346%**;
  320→1280 changes it by +4.54%. This is a short smooth control, not proof for
  all geometries or future topology changes.
- At exhaustion, 320/1280/5120 cells and outer steps 1/.5/.25 Myr all give
  approximately 20,688.020787 km³. This agreement follows from the finite stock
  cap and does **not** independently establish mechanical spatial convergence.
- The existing-layer thermal step is now first order with error control.
  Targeted tests show decreasing heat-flux/return errors with timestep
  refinement. Phase/error control is not a strict short-event certificate.
- At fixed small stock, 4/16/64 effective phase constituents give
  16,267.503525 / 20,688.020787 / 21,793.150103 km³. The last difference is
  **+5.342%**: phase discretization is not yet quantitatively converged. An
  independent continuous-envelope unit control confirms convergence of the
  caloric approximation as bins are refined, but a coarse bin count cannot be
  called a unique physical prediction.
- The linked two-Myr histories cool the mantle only about 0.27 K and do not
  sweep the full melting envelope. Unit tests exercise the transitions;
  long evolving phase histories are not claimed verified.

## Budgets, preservation and cost

Across the 13 linked cases, material-origin and phase-origin residuals are
zero at recorded precision. Total mass residual is at most 1.007e8 kg against
an approximately 2.03e24 kg inventory (below 5e-17 relative); this comes from
separately summed floating-point totals. Maximum compensated thermal residual
is 0.087 J. Maximum combined energy residual is **1.631e17 J** in the native
case, **4.641e-8** of its mechanical dissipation. Smooth cases are below
7.1e-11 of mechanical dissipation; the oblique case is 2.13e-10. Residuals are
reported, never turned into heat. Explicit radiogenic input/surface output
remain present; external mechanical work and orbital input are zero.

All 13 cases have exact complete-checkpoint restart at 1→2 Myr. Nine saved
states (initial, 1 Myr and 2 Myr for three representative cases) are retained.
Saved restarts were rechecked with the final downward-rounded stock accessor.
All **886** earlier manifested artifacts retain their hashes. No old snapshot,
archive, result or main-branch state was modified.

Actual integration cost is **0.211–3.122 s per 2 Myr**, depending on grid and
step; the 13 integration timers sum to 12.40 s, excluding setup and restart
replays. Reported process peak memory is at most **99.9 MiB**; it is cumulative
within each queue process, not an isolated per-case measurement. Results and
nine compact restart files occupy about **1 MiB**. These timings are short-run
measurements, not a forecast of Gyr performance or an accuracy guarantee.

## Remaining boundary and stop

This completes the requested bounded phase-supply stage. It makes the existing
material exchange thermally accountable, but does not restore fracture
mechanics. The surface still has shared welded boundaries and no thermally
generated elastic stress reservoir. Restoring paid fractures requires a
compatible regional elastic/viscous memory and actual topology work, not a new
label or a temperature threshold. Fixed phase envelopes, eligible fractions,
retention, placement pressure and return laws remain uncertain effective
parameters; no lunar water/redox choice was made.

No new regions formed in these tests. No GUI/default-runner switch, further
mechanical stage, long history, or new investigation is started after this
report. Work stops after committing, pushing and verifying the remote SHA,
as requested.

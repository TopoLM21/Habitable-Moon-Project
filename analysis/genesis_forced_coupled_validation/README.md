# One coupled 25-year step with the inherited physical heat sources

Base `b5555e49fda253388f3fe5e87278e9b5cade77d7`. Restore CLOSED.
No new coefficients, mesh refinement, canonical/default update or publication.
This is one isolated step from the existing migrated/cut-cell initial state,
not another subdivision of the completed BE50 versus 2×25 control.

## Source inventory, signs, units and receiving domains

All coefficients come verbatim from the authenticated source packet.
`genesis.py:fluxes/_rhs` defines the thermal terms; `genesis_tides.py` defines
the finite eccentricity reservoir; `genesis_onset.advance_orbit_thermal` supplies
the convention that orbit-derived heating replaces, rather than adds to, the
prescribed thermal tide parameter. A nonzero prescribed tide is rejected here.

| Port | Positive sign / receiver | Integration and units |
|---|---|---|
| Radiogenic | Into deep mantle | Original total silicate mass × W/kg × exponential decay, analytically integrated in seconds |
| Tidal | Into deep mantle, paired with orbital debit | Original `advance_tidal_orbit`; mean W/m² × spherical area × seconds, exactly once |
| Absorbed stellar | Into surface water/atmosphere node | Stellar W/m² × (1−albedo)(1−eclipse)/4 × spherical area × seconds |
| Giant absorbed | Into same surface node | Original W/m² × area × seconds; actual coefficient is **zero** |
| OLR | Out of surface node | Original grey/steam/hot-window law, endpoint W/m² × area × seconds, negative entry |

The area is the inherited **3.512598284e14 m² spherical reference area**, not the
sum of planar FEM face areas. Initial actual fluxes are radiogenic 0.02889654,
tidal 0.000740507, stellar 224.7260125, giant 0 and OLR 282 W/m².

Receiving domains preserve the old *lumped reservoir roles*: global-mantle
sources now enter the deep residual mantle; radiation enters the surface
water/atmosphere boundary node, which exchanges with rock through the already
accepted conductive network. This is an explicit representation mapping, not
a recovered spatial radioisotope or tidal-dissipation field, and not a claim
that the migrated trace equals the old two-temperature trace. No arbitrary
crust/mantle heat fraction is introduced. Surface rock already belongs to the
columns and is not duplicated as another reservoir. The fixed cut-cell closure
and the existing source coefficients remain unchanged.

There is no separate core, accretion, chemical, obliquity or resonant-pumping
source in this accepted model; none is invented. Old global mantle-to-surface
transport is **internal**, not an external source: adding it in parallel would
change the accepted fixed-grid conduction closure. `net_cooling` is a derived
combination, and the old energy[2:4] fields are cumulative counters, not extra
energy reservoirs or inputs. Latent heat is already in h(T). Maxwell and
thermal work remain internal paired mechanical exchanges; numerical loss is
never physical heat.

## Discrete coupled pipeline

A source-first split integrates radiogenic decay exactly and advances the
existing orbit once. The scalar surface radiation equation is backward Euler
on the supported hot-vapor branch, with the same OLR law; it is solved without
changing any emissivity or cap. Thereafter the already validated implicit
conduction, phase guards, BE predictor/damage/final correction and local
Maxwell/thermal-work loop run on the same 3360 cells. The actual endpoint orbit
now enters the inherited damage law instead of a frozen-orbit snapshot.

The source stage is an intermediate energy dose over the same interval; the
shared thermal time advances once, in conduction. It is not a second physical
interval. Source/conduction/mechanical splitting remains an engineering
approximation; the no-source time-control result does not prove convergence of
this newly forced split. Neither source nor mechanical correction is repeatedly
added on successive correctors.

Radiogenic energy is an external nuclear-power port: no isotope inventory is
invented. Stellar/giant radiation and OLR are open environmental ports. Tidal
energy is external to the thermal/mechanical domain, but cancels as an internal
exchange when the orbital reservoir is included. Both budgets are audited.
Traction remains external work, drag an unresolved interface export, and damage
release a nonthermal accounting store. No return heat from unresolved channels
is assumed for this window only. Local unfunded healing, unsupported birth,
old-material melting, water-phase departure or activation stops the step;
the physical healing law is not disabled.

## Tests and independent evidence before evolution

Manufactured tests check radioactivity over a half-life, W/m²→J and receiving
indices against the original flux law, zero terms, existing nonzero giant
coefficient, signed OLR and stellar terms, one paired orbital debit, duplicate
tide rejection, source-state preservation and tampered-port rejection.
The independent auditor uses Gaussian time integration for radioactivity,
reconstructs the radiation law and water enthalpy, verifies orbital angular
momentum/eccentricity energy and journal debit, and checks cellwise and summed
source-energy changes. It never calls the source evolution routine.

Saved-state preflight (`preflight.py`) authenticates the original state and
completed control files without evolving them. **87 tests PASS before the
single real step**. `source_preflight.json` records the actual coefficients and
fluxes used; all five ports are explicit even when a coefficient is zero.

## Thermal-work sensitivity is a limitation, not a thermal blocker

The previous 9.90% work difference is **3.3156931904e15 J** between BE50 and
2×25. Assigning each signed work difference to its own material heat capacity
would change temperature by at most **6.98779e-7 K**, capacity-averaged
**3.80491e-8 K**. It is **6.04017e-9** of the saved surface-conduction energy.
Thus it is not a meaningful obstacle for current thermal outputs or a reason
for another sweep. The relative work sensitivity remains reported, since a
small thermal effect does not prove mechanical work/time accuracy.

## Runbook

One isolated pipeline action, with a new output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_forced_coupled_validation.run \
 --output /workspace/genesis-forced-coupled-25-NEW
```

Limits: **one 25-year step**, 600 seconds total, 4 GiB, at most six mechanical
correctors. Existing paths are never overwritten. The action writes source
snapshots/ports, independent source and conduction audits, the coupled candidate,
independent mechanics/local-heat audit, typed-memory/orbit/fault restart and the
whole budget. No default hook or automatic long evolution is installed.
Read saved results instead of rerunning the step for inspection.

## Confirmed real-source result

**PASS** in `/workspace/genesis-forced-coupled-25-01/report.json`.
Exactly one coupled interval, **0.8684 → 0.868425 Myr**, 102.81 seconds,
509916 KiB peak RSS, 4 mechanical correctors. Source coefficients remain pinned;
800 old IDs/masses persist, no births or retirements, all old material remains
solid, no unfunded healing and no activation. Maximum damage **0.04002669**.

| Integrated port | Energy into thermal domain, J |
|---|---:|
| Radiogenic | +8.007894005e21 |
| Tidal | +2.052023704e20 |
| Absorbed stellar | +6.227673081e25 |
| Giant absorbed | 0 |
| OLR | −7.814866598e25 |
| **Net** | **−1.586372208e25** |

The independent eccentricity-energy difference is **2.052023704160e20 J**;
the thermal tidal entry is **2.052023704159e20 J**. The stored cumulative orbital
journal difference also agrees within its explicitly computed rounding allowance.
No extra heat is added from its cumulative value. Eccentricity evolves from
4.3803273852e-5 to **4.3801358876e-5**, and that endpoint orbit drives damage.

Surface temperature is **1377.7940517 → 1374.5182089 K**. The source-only
intermediate surface value is 1374.4400387 K; subsequent conduction supplies
heat back to the surface. OLR stays on the inherited 282 W/m² steam branch
throughout these endpoints. The finite-volume mantle/column system is retained;
this is not a replay of the old global two-reservoir temperature trajectory.

Maxwell heat **4.992730378e17 J** and thermal work into mechanics
**+1.435372691e18 J** give net mechanical-feedback heat **−9.360996534e17 J**.
These are internal exchanges, separately audited per cohort. Independent
mechanical momentum residual is **5.01e-14** relative, energy residual **−3200 J**,
memory error **2.49e-15** relative. Geometry/damage/phase audits pass.

Whole forced-domain energy residual is **6.052682662e14 J**, within the sum of
stage endpoint/mechanical allowances **1.838825589e16 J**. Including the orbital
reservoir cancels its tidal transfer and gives the same residual. This broad
thermal roundoff bound does not replace independent source-dose checks, local
heat balances or the tight mechanical audit. Source residual is 4.713982e13 J;
conduction residual is 5.581305e14 J; local-feedback combined residual is
−3.388860e8 J. Finite precision is explicitly accounted for, not fictitious heat.

Typed memory, fault history and actual orbital state restore with byte-identical
full packet serialization. Candidate/restart raw SHA256:
`e6ad0ed1372b18a73c3852f639edafd959776e6cb0d3a747aa75bcffe8c365a5`.
Canonical raw state is unchanged. No second physical step was used for restart.

## Engineering readiness and remaining limitations

The existing physical scenario now has an operational end-to-end forced pipeline:
actual inherited sources → conservative thermal/phase → BE mechanics → defined
local thermal feedback → open budget → restart. It is usable for bounded guarded
engineering trials, not yet a demonstrated accurate long-time planetary run.

A new material limitation is visible under real cooling: BE numerical memory
loss is **3.165279724e17 J**, **63.40% of physical Maxwell loss** (0.9031% of final
elastic energy). It is recorded only in the numerical sink. The large fraction
of step dissipation matters even though the external radiative budget is much
larger; small global residuals cannot certify this time accuracy. This task does
not launch another step-size sweep or silently change BE/default.

The former 9.90% work difference is thermally negligible for the previous control
and is retained as a limitation, not a blanket blocker. The new forced case has
its own time/splitting accuracy question. Source-first splitting, the lumped
source-to-reservoir mapping and initial-profile sensitivity remain explicit.
Unresolved drag/damage do not return heat in this window; healing requires a
funded constitutive law. Unsupported phase/birth/activation events must stop.
The four spatial controls and incomplete physical-fracture stages remain open.

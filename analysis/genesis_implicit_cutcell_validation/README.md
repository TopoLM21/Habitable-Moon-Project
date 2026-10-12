# Fixed cut-cell implicit enthalpy: thermal loading ready

One isolated 50-year backward-Euler step was completed from the conservative
initialization at `e5440e3`: **3360 fixed cut-cells**, plus deep mantle and water.
No grid refinement, new material ID, retirement, mechanical evolution, default
change or canonical advancement occurred. Restore is CLOSED.

The closure is explicitly a numerical hypothesis: accepted old material mean
enthalpies anchor their existing intervals; unowned material receives the
remaining original cell energy. It is not a recovered unique historical profile.
The initialization sensitivity and nonuniqueness documented in
[the preceding study](../genesis_cutcell_feasibility/README.md) remain open.

The solver uses `h = cp*T + L*clip((T-Ts)/(Tl-Ts),0,1) + offset` and solves
`m*(h1-h0) + dt*sum(G*(T1_i-T1_j)) = 0`. Conductances use the inherited
conductivity, density, actual displaced face areas and cut-cell thicknesses;
internal edges use half-thickness sums. Water stays on its hot-vapor branch.
Newton has finite iteration/backtracking budgets, positive temperatures and a
maximum-principle check. Phase-law derivatives enter the implicit Jacobian.
Each edge contributes equal and opposite energy; no numerical heat is added.
External radiation and orbital forcing are disabled in this isolated control.

## Saved real result

`/workspace/genesis-implicit-cutcell-01/{after,restart,report}.json`:

- One solve, 50 years, 1 Newton iteration, 0.512 s, peak RSS 190672 KiB;
  execution budget 60 s / 1 GiB.
- 800 old IDs retained; their temperatures are 1382.868364–1399.767409 K.
  Their liquid fractions are exactly zero in this phase law; no retirement.
- Column liquid mass decreases by **5.0157976441e17 kg**.
- Global energy residual **1.6117754558e15 J**, within the explicitly summed
  endpoint-roundoff allowance **9.3632617073e15 J**. Maximum local residual /
  allowance is **0.02827**. The local bound includes endpoint and paired-flux
  arithmetic uncertainty; no global-energy-relative tolerance is substituted.
- Fully solid surface-connected cut-cell prefixes match the accepted fronts
  (`q = 0.00170202965…0.00174171573`). Connectivity is a separate diagnostic
  from liquid fraction, and is only a statement about this fixed-cell closure.
- Endpoint and restart bytes agree; parent canonical raw SHA remains
  `62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7`.
- Endpoint raw SHA256:
  `632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8`.

`prepare_saved_loading` authenticates the thermal packet and parent memory,
checks exact ID coverage and old temperatures, rejects mixed old material,
and calls the existing `prepare_volume_loading` with unchanged geometry,
damage and radial moments. This succeeds for all 800 IDs, with zero removed
energy and maximum thermal strain 7.0379299464e-7. It prepares prospective
loading only: mechanical memory remains at 0.8684 Myr, thermal endpoint at
0.86845 Myr. This is not yet a shared accepted thermomechanical state.

## Verification without another physical step

From the repository with the project venv and one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_implicit_cutcell_validation.audit_saved
```

The independent readback reconstructs phase enthalpy and geometry-derived
conductances, verifies all 3440 paired exchanges and node/global balances,
checks exact mass/geometry/parent preservation, and prepares existing loading.
It makes **zero solver calls**. Readback PASS.

Manufactured two-cell analytic BE, phase crossing, constant state and order
monotonicity tests plus cut-cell, reservoir, volume runtime, BE adapter and
BE time-control regressions: **58 PASS**. The saved restart check demonstrates
serialization/readback identity, not a second advanced physical trajectory.
`run.py` is the original bounded producer; do not rerun merely to read results.

## Remaining blocker and next step

The prior false-front mapping blocker is removed **for this chosen closure**.
The thermal input is usable for a bounded mechanical loading trial. The next
step is to test that trial and explicitly account for reciprocal thermal work
and physical dissipation feedback. Drag/damage destinations, complete external
forcing and closure sensitivity remain unresolved; no global coupled energy
closure or physical fracture admission is claimed. The four spatial controls
and all unfinished original plan stages remain open.

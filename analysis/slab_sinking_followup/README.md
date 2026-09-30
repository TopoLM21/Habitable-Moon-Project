# Ordered thermal-cohort buoyancy: validation

These are independently evolved mechanics0.5 runs from the original Starter, compared with immutable archived0.4 runs. Only the version and the two buoyancy selectors differ in the fresh configurations. Geometry constants, strength law and viscosity were not retuned. Speeds are area means; 1 km/Myr = 1 mm/yr.

| Elapsed Myr | 0.4 speed, mm/yr | 0.5 speed, mm/yr | 0.5 cumulative mechanical detachments | 0.5 unresolved/detached accepted volume |
|---|---:|---:|---:|---:|
| 50 | 4.48305 | 2.68435 | 24 | 51.13% |
| 100 | 4.84816 | 2.79388 | 69 | 66.94% |
| 200 | 6.32652 | 2.30233 | 167 | 80.19% |
| 400 | 5.41047 | 3.11438 | 452 | 91.70% |

These are endpoint snapshots, not steady velocities. Over 300<elapsed<=400 Myr (100 uniform samples), the mean/median/range of area-mean speed is 2.86448/2.70605/2.08205–5.68254 mm/yr for0.5, compared with 2.63989/2.56838/1.87862–5.41047 for0.4.

Final0.5 accepted oceanic volume is 67,341,968.126 km³; attached 4,788,099.548; deep transferred 801,338.137; unresolved/detached 61,752,530.442. The unresolved category includes lost contact connectivity and mechanical neck failure. A smaller speed or detached fraction is not proof of realistic stable subduction.

Specifically, committed mechanical neck failures account for 53,208,063.333 km³, or 79.012% of accepted volume. Other unresolved connectivity accounts for 8,544,467.109 km³. In0.4 the mechanical fraction was 90.861%; this is distinct from its97.373% total unresolved/detached fraction.

## Resolution sensitivity

- dt0p5: 3.34911 mm/yr (+24.76% from default at50).
- sub5: 3.49826 mm/yr (+30.32% from default at50).

The 50 Myr numerical comparisons remain sensitive to timestep and grid. [The archived baseline investigation](validation_baseline.md) shows whole-cell acceptance timing, different first-force thermal weights and different bend-length regimes. Ordered buoyancy corrects mass placement; it does not make those discretizations converge.

## Force and failure evidence

The new trajectory contains 452 recorded neck failures: 142 have a material no-eduction reaction and 310 do not. Rich records retain gravity, bending, mantle drag, reaction and live capacity before removal. Their maximum relative decomposition error is 0. Subtracting a force term at fixed velocity is only a diagnostic, not an independently solved alternative trajectory.

The [local frozen counterfactuals](frozen04_counterfactual_solver_fixed/summary.json) load old0.4 states and change only selectors on in-memory copies. They do not relabel checkpoints or commit hypothetical detachments. Local speed changes are small on these particular saved states even though evolved histories diverge. The first physical speed difference occurs at18 Myr; the first failure-count difference occurs at26 Myr (0.4:2,0.5:0).

## Checks and artifacts

- [Every-step force audit](validation_step_audit.json): balance of torque and power, zero-work constraints, positive drag/dissipation, feasible feed and finite surviving tensile capacity; all seven continuation checks and source integrity at saved endpoints.
- [Resume compatibility](validation_compatibility.json): old0.4 state arrays/history are bitwise equal; the only metadata additions are10 failure-diagnostic fields. Populated0.5 CPU1/CPU4 states, metadata and physical reports are bitwise equal.
- [Final-solver fresh50 replay](validation_solver_recheck.json): bitwise equal to the original0.5 first50 Myr state, including physical metadata and report.
- [Results and exact numbers](validation_results.json), [failure decomposition](validation_failure_decomposition.json), [source/code provenance](validation.json), [comparison figure](comparison.png).
- Main continuation: `D:\Moon Project\mantle-convection\analysis\slab_sinking_followup\runs\ordered_sub4_dt1_solver_fixed\elapsed_0400`. The first200 Myr are in `runs/ordered_sub4_dt1`;200→400 is in `runs/ordered_sub4_dt1_solver_fixed`.

The initial sub5 attempt stopped on a numerical active-support error; its exact solver inputs and log remain in `runs/ordered_sub5_dt1`. The completed fresh rerun is `runs/ordered_sub5_dt1_solver_fixed`. The support solver was corrected without relaxing force, residual or bound guards. Earlier completed segments retain their original production hashes; the changed file is the numerical constraint helper. Tests and exact source hashes are recorded in `validation.json`.

Reproduce a fresh0.5 series from the repository root:

```powershell
& .\.venv\Scripts\python.exe analysis/slab_sinking_followup/validation_case.py series --output NEW_OUTPUT_DIRECTORY --ages 50 100 200 400
```

`validation_case.py` defaults fresh imports to0.5. Saved-state resume and frozen probes preserve the saved version. The historical `slab_sinking_validation/run_case.py` remains pinned to fresh0.4. Physical overrides are applied before creating the initial inventory.

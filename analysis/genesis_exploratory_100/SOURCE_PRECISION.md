# Source-stage correction: same equation and unchanged energy audit

The previous failure was caused by the nonlinear solver's temperature stopping
criterion, not missing physical heat or an underestimated energy-roundoff bound.
No audit tolerance was increased. The radiation/enthalpy law and its source
routing remain unchanged.

On the saved 100-year input:

| Quantity | Joules |
|---|---:|
| Surface H before | 9.066414225016783e27 |
| Corrected H after | 9.002926484307950e27 |
| Actual delta H | -6.348774070883293e25 |
| Absorbed stellar Q | 2.4910692322627724e26 |
| Outgoing radiative Q | -3.1259466393511065e26 |
| Net Q | -6.348774070883341e25 |
| One ULP of either stored H | 1.099511627776e12 |
| One ULP of stellar Q | 3.4359738368e10 |
| One ULP of outgoing Q | 6.8719476736e10 |
| Unchanged 8×(ULP Hbefore + ULP Hafter) allowance | 1.7592186044416e13 |

The original root T=1364.3779999024125 K differs from the corrected root
1364.3779999023914 K by 2.1146e-11 K. Multiplication by the actual heat capacity
4.732222406336507e24 J/K gives 1.000664e14 J, explaining the former
9.943708e13 J residual. In this supported branch OLR is exactly the inherited
282 W/m² plateau, so the scalar solution also has an independent linear
high-precision reference. The physical source quadrature was not changed.

`surface_endpoint` evaluates the same balance with `math.fsum`, normalized by
heat capacity. Brent's numerical convergence is tightened to machine-relative
precision, and adjacent representable temperatures are checked against that
same residual. This corrects arithmetic/solve accuracy; it is not a relaxation
of output, physical or conservation criteria. The final energy remains stored
in the existing format, and the original independent audit is unchanged.

Corrected surface residual: **4.81036337152e11 J**, or **0.02734375** of the old
allowance. Subtracting large enthalpies still incurs representational rounding,
but the old bound covers it. No new roundoff bound is needed. Other reservoirs
and the independent orbital/source debit audit also pass unchanged.

Tests use the actual saved inputs without loading mechanics, 70-digit Decimal
solutions for the constant-OLR branch, a separately derived nonlinear hot-window
polynomial solved by 70-digit bisection, and manufactured reservoir scales.
They compare the solution and original energy allowance, not only a solver flag.
The corrected source100 packet and independent audit are saved as
`/workspace/genesis-exploratory-100-01/corrected_source100.json`.

Continuation reuses this packet after source/clock/predecessor checks. It starts
from the same saved initial snapshot, not from a failed state. The original
process no longer holds a model object, so a fresh process must restore the
immutable model once; its cost is charged to the remaining budget. The original
54.399753383 seconds and a conservative 10-second charge for isolated diagnosis/
tests leave **535.600246617 seconds** for restoration and all remaining work.
The original failed source attempt is retained. No independent fresh 600-second
budget, automatic reduction, canonical promotion or full event certificate is
introduced. Aggregate artifact accounting includes the prior folder's files.


## Completed continuation — calculable, not acceptable

All three trials (100, 50, 50 years) completed with independent source, thermal,
mechanical, feedback/energy and restart checks. Four correctors per trial.
All three audits PASS. Final decision: **exploratory output/work budgets FAIL**;
accepted=false, canonical_advanced=false. No automatic smaller step was run.

| Whole100 vs fine2x50 | Maximum difference | Worst budget ratio |
|---|---:|---:|
| Temperature | 0.721311644 K | 1.44262 |
| Specific enthalpy | 865.573973 J/kg | 1.44262 |
| Stress tensor | 505952.845 Pa | 2.03785 |
| Physical velocity | 3.49373743e-12 m/s | 1.77478 |
| Physical displacement increment | 0.0692370512 m | 5.10495 |
| Damage | 0.00211354890 | 2.11355 |
| Numerical work, global | 6.06145735e18 J | 12.23027 |
| Numerical work, local | largest local loss 6.33368260e16 J | 15.29517 |

The largest local loss and largest budget ratio need not belong to the same
cohort. Fine physical Maxwell heat is 2.38013777e18 J. Passing the energy identity
does not make the separate numerical loss acceptable. No budget was changed.

No unsupported event was detected in the inspected stages. Old-solid endpoint
margins are 6.17528 K (whole), 2.06290 K (half1), 5.45542 K (half2); the fine
birth margin is 187.618 K and water margin 717.659 K. The largest frozen-driver
damage is .0460931 (whole), .0409934 (half1), .0439860 (half2), leaving >.60 to
the gate. These stage/subproblem statements are **not a full-path certificate**.
The real temperature/stress/damage interval envelopes remain unproved.

Continuation cost **131.239 s**, including **54.885 s** immutable-model restoration.
Original failed attempt **54.400 s** plus conservative isolated-work charge **10 s**
gives **195.639 s total**, below the original 600-second budget. Exclusive main
costs: mechanics42.111s, coupled audits27.716s, conduction0.141s, typed decode0.326s.
Peak RSS606484KiB (592.27MiB). Continuation artifacts39,378,418 bytes before its
report, plus the retained prior files; combined files remain well below200MB.
The corrected coarse source is reused, not integrated again during continuation.

**148 scoped tests PASS**. Raw artifacts and the measured source manifest were
verified after the run. Source-stage correction is complete; the remaining
obstacle to accepting this100-year comparison is measured output/work error,
in addition to absent full-path event certification. Completion of2x50 does not
validate coarse100, and the fine numerical-work failure prevents admitting that
fine path under the unchanged criteria. There is no new general audit or method
project implied by this result.

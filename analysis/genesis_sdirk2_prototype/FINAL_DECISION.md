# Final decision: SDIRK2 is not suitable under the unchanged requirements

Base `8c55cef036d74feb166597486bf6018af7512f86`. **Stop development of this
prototype.** The thin-block numerical solve is repaired; the negative numerical
energy is structural, not a coupling bug. Therefore both original problems are
not removable by minimal numerical correction while preserving the scheme and
strict nonnegative numerical-dissipation requirement. No replacement scheme was
searched or implemented. No planetary state was advanced.

## Thin block: solved without changing the tolerance or operators

The previous 1.29062e-9 residual was a momentum/Newton conditioning failure:
thermal residual 4.58e-14, memory 2.23e-11, momentum 1.29e-9. It was not evidence
that the requested 2e-10 criterion was physically unreasonable.

`thin_diagnosis.py` performs exact stage algebra for the restricted one-node,
one-block, fixed-solid-phase, state-independent-source verification problem.
It is not a general adapter for state-dependent orbital/thermal ports. Given
stage base `(E0,v0,e0)`, diagonal stage time `s`, and `x=deltaT`, eliminate memory:

```
a=1/(1+s*lambda)
e=a*(e0+s*B*v-U*x)
(M+s*D+s^2*a*B.T*H*B)*v = M*v0+s*F-s*a*B.T*H*e0+s*a*B.T*H*U*x
```

This gives v and e affine in x. The existing thermal feedback then gives one
quadratic equation `(capacity-e.T*H*U)*x = s*(Q+lambda*e.T*H*e)`, solved with the
stable quadratic-root formula. The stage equations and physical parameters are
unchanged. No mass lumping, extra nodes, positivity projection, viscosity floor,
force adjustment, or weakened residual normalization is used.

The 18x18 linear system is diagonally equilibrated, factorized in float64 and
corrected using a longdouble residual (nmant=63 in this environment). One
refinement reduces componentwise backward errors from about 4.5e-17/6.5e-17 to
2.85e-20/3.59e-20. The equilibrated stage matrix condition number is **2.98e10**;
raw mass conditioning remains **1.063e12**. Small backward error is not a bound
on forward accuracy or on input-data sensitivity.

Both SDIRK stages are rechecked using the **original** residual function and
normalization, with the **same 2e-10 threshold**. Their maximum residuals are
**8.01e-14 and 1.14e-14**. BE with the same stage algebra gives **1.22e-13**.
Independent impulse backward residual is **4.84e-17**. The energy identity residual
is -1.16e8 J against an explicit 8-ULP stored-energy allowance of 2.15e9 J. The
large absolute ULP budget comes from storing a thermal reservoir near 1e23 J;
it is reported, not confused with time-integration accuracy. Existing physical
acceptance tolerances were not modified or applied to admit this local test.

A diagnostic longdouble reassembly from identical primitive geometry gives
relative matrix differences **6.13e-23 (mass)** and **1.70e-16 (drag)**. Both solves
retain the original float64 matrices. This checks ordinary assembly roundoff,
not an exact-geometry guarantee; the stiff/null directions can amplify tiny
input perturbations. No small eigenvalue was clipped to change the equations.

## Negative energy: a structural, bounded discretization contribution

For fixed quadratic mechanical metric W and stage derivatives ki,

```
N = dt^2/2 * sum_ij Gij <ki,kj>_W
Gij = bi*aij + bj*aji - bi*bj
|N| <= dt^2/2 * ||G||2 * sum_i ||ki||W^2
```

For this SDIRK2, G is diagonal with eigenvalues ±0.0857864376269. In unforced
scalar decay, the second-stage derivative norm is smaller than the first, so N
is negative. A new test derives this directly from scalar stage formulas and
checks `N = energy_before-energy_after-physical_loss`, independently of heat,
force or coupling routines. This is known lack of algebraic stability/discrete
energy monotonicity, not an inconsistent exact-memory/thermal-work mixture.

The prior unforced example has **N=-0.00367753 J**, absolute bound
**0.00563106 J**: 4.451% of its Maxwell work, or 4.188% of drag+Maxwell work.
Thus a small balance residual does not make its sign admissible.

The saved-operator test has **N=-4.74657e10 J**, absolute bound **6.15272e12 J**.
Its magnitude is **6.15e-6 of drag work** and **8.10e-10 of source heat**, but
about **1.71e7 times its tiny Maxwell work** (2775 J). The choice of denominator
matters. None of these ratios changes the sign requirement. N is not physical
heating, is not fed into enthalpy and is not absorbed into a fitted tolerance.
The method cannot be made generally passive by scaling or iterative refinement.

## One completed saved-operator comparison, not a planetary run

`final_decision.json` preserves the completed finite comparison. One 12.5-second
manufactured step uses the unchanged saved cohort-1 geometry and operators,
identical source specification, stable stage elimination and known target.
This is not the actual planetary forcing/history. One initial reporting attempt
failed on a dict/array formatting bug after the solves; the same case was retried
to produce the report. No alternate steps, schemes or physical states were tried
as an accuracy search. The completed timing comparison was not repeated.

| Quantity | BE | SDIRK2 |
|---|---:|---:|
| Temperature target deviation, K | 6.50025e-4 | 1.96215e-5 |
| Maximum velocity target deviation, m/s | 3.23737e-3 | 3.37794e-4 |
| Maximum memory target deviation | 6.50025e-9 | 1.96215e-10 |
| Explicit scaled state norm of target deviation | 748.80 | 78.13 |
| Solve wall time, seconds | 0.000544 | 0.001776 |

The measured norm reduction is **9.58×** for **3.26×** solve cost; SDIRK audit
costs another 0.000258 s and setup 0.382 s is excluded. Millisecond single samples
are not a statistically established cost ratio. The velocity deviation still
exceeds the manufactured velocity amplitude: neither path demonstrates practical
accuracy on this step. These norm scales are not the planetary controller's
acceptance budgets.

`forcing_precision.json` separately records a static, nonintegrating check:
rounding the manufactured force loses about **29217 N**, only 6.14e-17 of total
force but **5.32e-7 of the inertial force** being recovered from cancellation.
Consequently target deviations mix discretization and input/conditioning effects;
9.58× must not be advertised as a clean time-discretization accuracy improvement
or as a measured planetary time-to-accuracy gain.

## Practical alternative and cost

Keep the independently audited **BE** path, the explicitly enabled invariant
cache, and error/event-limited dt growth. BE has a nonnegative quadratic numerical
loss; the cache already reduced the saved algebra workload by 1.81× without
changing it. Use the demonstrated scaling/elimination lesson only after an
appropriate full-adapter validation, not by replacing the production runner with
this one-cell helper. No new implementation or run is started here.

The price is first-order temporal accuracy: forced, rapidly changing intervals
need more accepted BE steps than a successfully validated second-order method
would. Quiet intervals may permit growth, but only when unchanged significant-
output/work tests and a justified full-path event bound permit it. The existing
25-year rejection remains binding. Under the current constraints there is no
verified shortcut to thousand-year steps; promising one would hide the accuracy
or energy requirement. This is the finite practical choice, not another scheme
search or a long checklist of speculative development.

**141 scoped tests PASS (6 new decision tests).** They check unchanged equations,
original stage threshold, refinement, independent budgets, the scalar negative-
energy counterexample and explicit restricted scope. Report/input hashes and
both measured-source hashes are verified. Defaults, canonical history, physical
parameters, tolerances and node count remain unchanged. Restore CLOSED; no push,
paid resources or deletion.

# Source coverage and exact angular mapping

This lane continues the four authenticated spatial controls at their original
radii, `(3,5)` angular orders, `2e-6` own signed-component criterion and work
limits (3400 lines, 2000 evaluated cells, total depth 24). It does not advance
Genesis or admit physical creation/release. Original frozen evidence is read
without modification. Large results are external to the repository.

## Initial support completeness

`tectonics/genesis_volume_source_coverage.py` bounds the whole source before
subdivision. For every original cohort lambda, let `Y = lambda C + D`, using
the original binary64 `C=X+P`, `D=A-P` operands. For every point `x` of the
closed source ball, Cramer's rule gives `N_i = x · cofactor_i(Y) / det(Y)`.
Centered exact polynomial coefficient bounds enclose the determinant and
cofactors over the entire lambda slab. Cauchy–Schwarz, with an integer-square-
root upper bound verified by squaring, encloses every ball numerator. Exact
interval division handles either determinant sign. A denominator containing
zero retains the full simplex. Outward 80-bit dyadic rounding only enlarges
coordinate bounds; it limits later rational cost, not physical accuracy.

Exact halfplane clipping intersects these bounds with the original simplex.
The resulting convex polygon, triangulated as an exact fan, contains every
source point with positive volume. This is a proof of **initial source
coverage**, separate from the event module's exact partition of supplied cells.

`coverage_audit.py` independently reconstructs cofactors from polynomial minors
and the determinant from signed permutations. It checks the norm inequalities
and independently enumerates all feasible bounding-line intersections. It does
not call the production clipper or coverage builder.

## Node mapping and radial uncertainty

The new lane takes the same literal binary Gauss parameters/weights as the old
control. Its Duffy map is calculated with rational arithmetic on exact vertices:
`N = (1-s) v0 + s(1-t) v1 + st v2`. Every mapped N sums exactly to one and is
used unchanged by the coefficient and radial-root calculations. There is no
conversion of N to native floating point and thus no uncharged N rounding
residual. This certifies the **new exact map**, not the previous native mapper,
and does not certify errors of binary Gauss parameters relative to exact
Legendre roots.

Only the local lambda midpoint/halfwidth is represented in long double. Exact
source roots are enclosed and transformed onto that represented chart. The
existing rational fields, directed tensor contractions and limited mass repair
then evaluate the exact N. The new `evaluate_prepared` entry separates chart
preparation from the unchanged contraction body; the original native-node
entry remains available. Independent sample audits use a direct discriminant
with integer square roots at 768 fractional bits and the separate frozen
polynomial/endpoint algebra, without runtime fields or moment vectors.

## Angular integration and remaining scope

The verified initial support is partitioned by exact source-event Bernstein
bounds. Only cells with a certified empty radial intersection are omitted.
Unresolved event cells are retained. Exact Jacobians and literal quadrature
weights are enclosed before directed aggregation. Source and aggregation errors
consume the original own-component allowance; no positive floor is introduced.
Initial partition levels count toward total depth 24.

A conservative whole-cohort volume envelope keeps possible support visible
when quadrature samples miss it. This bound may be loose. The embedded
`8*abs(I5-I3)` remains a convergence observation, not a rigorous angular error
bound. Even finite embedded agreement does not certify general angular
accuracy or close stage 1. Identically-zero Gamma on a positive radial line
and unresolved exact tangencies are refusals, never fabricated zero integrals.

## Reproduction

From the repository with the prepared environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/habitable-moon-venv/bin/python -m analysis.genesis_volume_mapped_validation.control \
  --output /workspace/FRESH-mapped-control
/workspace/habitable-moon-venv/bin/python -m analysis.genesis_volume_mapped_validation.coverage_audit \
  --control /workspace/FRESH-mapped-control --output /workspace/FRESH-coverage-audit
```

The control checks a 256 MiB free output reserve, authenticates existing source
inputs and all 469 frozen source hashes, snapshots its own numerical code,
records exact node ratios/digests, verifies parent K/M byte identity, and checks
source pins again on completion. It never restores or copies the large data
archive. `summary.json` records the completed run, tests and remaining limits.

## Completed control and final refusal guard

`/workspace/genesis-mapped-full-03` completed 13,430 nodes with zero local or
summed source-budget failures. All four angular controls remain OPEN (1992
failed components each). The eight independent samples covered 20,288
components without a containment miss. See `angular_diagnostics.json` for
comparison with the pinned prior control; median normalized embedded errors
improved by factors 1.12–1.64, which does not meet the tolerance.

After the full run, the only numerical-file change was an explicit refusal
for a nonempty-or-unresolved cell lying wholly in literal Gamma. The guard
must prevent both banks from being silently classified as empty. Two semantic
tests exercise this refusal and a genuinely empty cell. `guard_compatibility.py`
checks that removing exactly this guard recovers the full-run snapshot byte
for byte. Exact affine Gamma values on the simplex prove the guard unreachable
on every positive-area descendant of all four original source regions. It also
checks oriented convex polygon boundaries and exact fan areas. Thus the four
results are unchanged by proof; the expensive control was not repeated after
the guard. Its report and final regression are pinned in `summary.json`.

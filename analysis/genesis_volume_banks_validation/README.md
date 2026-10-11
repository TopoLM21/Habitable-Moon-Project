# Bank bounds and event-guided angular integration

**Result: bank bounds verified; the trial angular strategy is rejected as a
general convergence fix.** The completed four-control run retains 1992 angular
failures in every case. Do not promote this trial lane to physical admission.

This additive numerical lane follows commit dcbc8cf. It preserves the literal
source geometry, reference measure, accepted history, frozen evidence, original
component tolerance 2e-6, orders (3,5), and caps 3400 lines / 2000 cells / depth24.
It does not authorize physical creation, opening, or a new Genesis interval.

## Whole-cell volume bounds

For each triangle and cohort slab, the bilinear birth map and its displacement
from the centroid line are convex combinations of six endpoint corners.
Their exact rational squared norms bound displacement by epsilon (an outward
integer-square-root bound). The centroid-line sphere of radius R+epsilon
contains every source fiber; radius R-epsilon is contained in every fiber.
The minimum and maximum signed plane deviations at the same six corners give
necessary and sufficient Gamma inequalities. Intersecting those intervals with
the original cohort and integrating det(X)*lambda^2 gives separate lower and
upper volume bounds for both banks over the whole angular cell.

The previous strict sphere/cohort Bernstein exclusion is retained and intersected
with these new bounds. Dropping it makes the centroid tube too loose near the
small sphere: the stopped pilot evaluated 35 already-proven-empty cells among
its first 40 cells in case001. A literal regression fixture preserves that case.

Two additional exclusion proofs tighten a bank: its Gamma sign is inactive at
all six corners, or positive Bernstein bounds for A and the Gamma/sphere
resultant exclude intersection and the exact closest-point plane sign fixes
the opposite bank. A box-distance lower bound can exclude the whole sphere.
Only two zero upper bounds permit removal of a cell. A positive or unknown
upper bound is never interpreted as empty from zero quadrature samples.

Certificates retain exact vertices, interval endpoints, source event bounds,
measure bounds and their reasons. Readback rebuilds every certificate. A
separate discriminant / 768-bit integer-root auditor checks both banks at
vertices and centroids. This is a sample audit; the whole-cell argument is the
convexity and interval construction above. Entire unresolved cells on Gamma
are refused; isolated whole-Gamma audit fibers are explicitly recorded.

## Event integration and conserved measure

Initial cells are cut along affine Gamma/cohort endpoint events. To limit
rational growth, each crossing EDGE PARAMETER is rounded to 40 dyadic bits.
The same point on the exact old edge is used by both children. Both sides are
retained; the mesh never assigns bank identity. Near-vertex cuts that would
collapse a child retain the old cell. Therefore event alignment is approximate
but source-domain partition and its measure are exact. Remaining crossing
slivers are evaluated against the actual source geometry.

The exact rational Duffy map and directed source/weight aggregation are
unchanged. Refinement prioritizes unsampled possible-positive banks before
the original component-error priority. Only certified-empty children avoid
quadrature. Initial event splits count toward depth24. The final exact area
ledger includes active cells and all certified-empty discarded cells.

This is not a rigorous angular remainder certificate: 8*abs(I5-I3) remains
an embedded observation. Curved sphere boundaries are not exactly fitted by
these affine cuts. Original own-component allowances and the unresolved-cell
refusal remain unchanged. Even finite agreement cannot close physical stage1.

## Diagnosis and reproduction

`diagnose.py` recovers the old final triangles from authenticated actual node
traces and verifies every recorded mapped coordinate. It separates proven-empty
banks, banks with strict positive rational witnesses, and still-open banks;
it does not rerun tensors or alter the old verdict. The 1992 original angular
failures are two volume and 1990 tensor/work components in each control, with
zero source-budget failures. A missing-bank correction alone does not prove
tensor angular convergence.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/habitable-moon-venv/bin/python -m analysis.genesis_volume_banks_validation.diagnose \
  --output /workspace/FRESH-bank-diagnosis
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/habitable-moon-venv/bin/python -m analysis.genesis_volume_banks_validation.control \
  --output /workspace/FRESH-bank-control
```

The full control uses four processes, checks 256 MiB free space, authenticates
the existing inputs and 469 frozen sources, snapshots numerical code, and
checks those hashes again after completion. No archive restoration occurs.
Exact node traces and large certificates remain outside the repository.

## Completed comparison and the failed strategy

`/workspace/genesis-banks-full-02` completed 13,498 nodes (3400/3400/3366/3332),
with zero local and global source-budget failures. Eight independent actual
nodes passed 20,288 component checks. All parent K/M bytes and 469 frozen
sources matched; accepted history remains 0.8684 Myr /800 cohorts/4 intervals.
There are 3,240 independent bank-fiber checks over 405 final active/discarded
cells. Independent exact clipping verified containment, no overlap and total
measure, including 1,784 candidate pair-intersection checks.
The four initial coverage certificates are byte-identical to the independently
audited baseline; this check is separate from the final partition audit.

| Case | Missing candidate cells before → after | New missing-volume upper bound, m3 | Median old/new normalized embedded error |
|---|---:|---:|---:|
| 0 | 6 → 19 | 5.1070e-8 | 0.1432 |
| 1, small radius | 41 → 1 | 0.7162983 | 0.1669 |
| 2 | 3 → 22 | 2.4146e-5 | 0.04755 |
| 3 | 6 → 13 | 4.3558e-4 | 0.8509 |

Ratios below one mean worse convergence observations. These are embedded
differences, not certified true tensor errors. Cell counts refer to different
meshes and must not be interpreted as counts of lost physical bodies. The
small-radius volume error observations improve 43–74 times, but the overall
component observations worsen: unconditional missing-first refinement is not
a successful global allocation policy.

Rounded affine event intersections introduce real, extremely narrow Gamma
banks. Exact rational witnesses show positivity. `sliver_audit.py` proves
46 banks (18/0/18/10) cannot be sampled by **any** allowed midpoint descendants
under the remaining depth24 budget with the original literal (3,5) rules.
For an affine endpoint field, let delta be the largest nonnegative vertex
value, -m the least-magnitude negative value, w the least Duffy barycentric
weight, and d the remaining depth. Every nondegenerate midpoint descendant
has negative-vertex barycentric mass at least w*2^-d, so every node's field
value is at most delta-(delta+m)*w*2^-d. A nonpositive bound, together with a
nonpositive other cohort endpoint, proves no quadrature node can enter that
bank. A separate strict source witness proves the bank is not empty. Five
tests enumerate all nodes through depth3 for three sign configurations and
check refusal to claim unreachability for a resolvable positive corner.

An independent half-ball oracle confirms that the sphere boundary also remains
unresolved. With exact rational Machin bounds for pi and exact containment in
the literal cohort, the new two bank-volume relative errors are 0.00195724
and 0.00086300 at 3332 nodes; the previous lane gives 0.00185858 and 0.00111629
at 3366 nodes. Both remain OPEN at 2e-6, despite zero missing candidate cells.

**Next:** construct exact affine Gamma intersections without the rounded-edge
slivers; replace unconditional missing-first priority while retaining the
original refusal and error accounting; then handle curved sphere/cohort events
and establish the angular remainder. Do not declare positive slivers empty,
relax 2e-6, or treat this bounded-measure result as physical admission.

The stopped run01 and its source snapshots are preserved. Its omitted old
sphere/cohort exclusion was corrected before completed run02. Numerical
sources were unchanged throughout run02. Independent read-only audit helpers
and their tests were added separately and are pinned as additional validation
sources in `summary.json`; they do not alter the completed runtime.

Evidence: [summary](summary.json), [group comparisons](angular_comparison.json),
[old missing-bank diagnosis](diagnosis.json). Tests: 67 main regression tests
plus 5 sliver-proof tests. Reproduce additional audits with fresh output paths:

```sh
python -m analysis.genesis_volume_banks_validation.partition_audit \
  --control /workspace/genesis-banks-full-02 --output /workspace/FRESH-mesh-audit
python -m analysis.genesis_volume_banks_validation.sliver_audit \
  --control /workspace/genesis-banks-full-02 --case 0 --output /workspace/FRESH-sliver-audit
python -m analysis.genesis_volume_banks_validation.sphere_oracle \
  --output /workspace/FRESH-half-ball
python -m analysis.genesis_volume_banks_validation.sphere_oracle \
  --baseline --output /workspace/FRESH-baseline-half-ball
```

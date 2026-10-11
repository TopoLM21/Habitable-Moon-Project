# Source precision control

The runtime entry point is
`tectonics.genesis_volume_precision_integrals.prepare_precision_integrals`.
It preserves the old component ordering and source geometry, computes exact
source coefficients and basis/probe projections before conversion to directed
`long double` intervals, and returns represented centers with outward absolute
errors. It does not invoke planetary initialization, evolution, or activation.

The 469 frozen source files and archived historical results are unchanged.
The new rational field implementation does not import the independent
coefficient/contraction oracle. Source geometry bounds and directed contraction
operations reuse the existing implementation. Their existing scope limits
(including angular node mapping and candidate geometry) still apply.

Mass contractions with relative enclosure error above `2**-40` trigger exact
polynomial contraction and integration against rational source moment bounds.
At most 64 components per line receive this additional work. This trigger is
an internal precision target, not an acceptance tolerance or a claim that
signed angular sums satisfy their budget. Unrepaired and unresolved errors
remain in the returned error vector. Exact zero components have no scale floor.

Final acceptance still requires the original own-component allowance `2e-6`.
In particular, local relative accuracy does not imply relative accuracy after
cancellation in a signed angular sum. The original angular control independently
charges both rules' source and aggregation errors against its own allowance;
the embedded angular discrepancy is an additional error, not credit.

## Reproduce

Use the restored Linux environment with `np.finfo(np.longdouble).nmant == 63`,
the authenticated archive and its original `/workspace` paths.
All output directories must be new and outside the repository.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/test_genesis_volume_precision_integrals.py
python -m analysis.genesis_volume_precision_validation.control --output /workspace/NEW_SELECTED
python -m analysis.genesis_volume_precision_validation.control --output /workspace/NEW_FULL --angular-cap 3400
```

`--angular-cap 340` is a pilot, explicitly not a completed original control.
The full run retains orders `(3,5)`, 2000 cells, depth 24, the original radii
and at most 3400 radial evaluations per support. It does not replay an
archived exact node list: the old archive contains only rolling node hashes.
The fresh adaptive control records every literal node in `nodes.jsonl` with
an output digest and explicit pointwise/fallback diagnostics.

An independent sample audit rebuilds complete rational target integrals on
two deterministically selected actual pilot nodes per support:

```sh
python -m analysis.genesis_volume_precision_validation.audit_nodes --control /workspace/NEW_FULL --output /workspace/NEW_AUDIT
```

This audit is explicitly sampled, not an independent replay of every node.
The four original selected-line comparisons use hash-authenticated archived
exact certificates; the sampled-node audit recomputes new exact targets.
Neither check grants angular convergence, creation work, or physical release.

The control reserves 512 MiB free output space. It creates small reports and
node traces without copying the original source archives or full certificates.

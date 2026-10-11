# Exact source-event partition

This lane follows the completed source-precision control in
`../genesis_volume_precision_validation/summary.json`. It reuses the preserved
event/Bernstein prototype and verifies equality of the source polynomials.

```sh
python -m pytest -q tests/test_genesis_volume_event_partition.py
python -m analysis.genesis_volume_event_partition_validation.control --output /workspace/NEW_EVENT_CONTROL
```

Use a new external output directory and the original restored `/workspace`
anchors. The control keeps the four original source supports and radii. It
reserves 128 MiB; the actual four certificate files total approximately 2 MiB.
It writes source snapshots, exact leaf vertices, signs and canonical hashes.

The module certifies polynomial signs over a whole supplied triangle using
exact Bernstein coefficients. Midpoint subdivision preserves all three source
N components without normalization. The four affine child triangles tile
their parent; exact signed areas are checked per original root. Work caps
retain all uncertain cells, including identically zero events.

For a positive leading coefficient, signs of the sphere quadratic, its
discriminant and endpoint derivatives can certify empty or positive radial
support throughout a cell. These classifications are reported without dropping
any cell or declaring a physical Gamma event.

The completed depth-two control produced 232 leaves: 57 with strict signs for
every listed event. The radial-support classifier independently reports 51
empty, 68 positive and 113 unresolved cells. These are different classifications;
an empty sphere cell can still cross an irrelevant Gamma event.

**Limits:** the initial candidate cells still come from the old shared
geometry routine. Exact partition of those cells does not prove coverage of
the physical source. Native node mapping and angular integral accuracy remain
open. This control does not perform angular quadrature or planetary evolution.
The next integration step must address source coverage and mapped-node error
before presenting the event partition as a certified spatial integral.

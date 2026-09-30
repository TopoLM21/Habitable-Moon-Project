# Independent geometric-contact validation

This folder records a separate geometric experiment. Existing fractional area
checkpoints do not contain polygons and cannot provide exact contact geometry.

The pre-change `baseline.json` hashes the existing production code and the
original Starter plus saved 0.5 source. Existing uncommitted work is the baseline;
comparison with Git HEAD would not establish isolation for this change.

Required independent checks:

| Property | Oracle |
| --- | --- |
| Pure-cell import | The saved source triangle itself, with unchanged lineage and four extensive quantities. |
| Mixed-only import refusal | Two distinct geometries can have identical fractional areas. No implicit reconstruction is possible. |
| Common rotation | A rigid rotation preserves area, contact length, and all material inventories; it creates no gaps or overlap loss. |
| Frame covariance | Apply a second rigid coordinate rotation to all geometry and angular-velocity vectors. Scalar measures remain fixed and vectors rotate with the frame. |
| Actual contact inside a cell | Bisect one triangle in two different directions with the same owner fractions. Contact directions must differ. |
| Contact refinement | Dividing one arc at an interior point preserves total length and its integrated lever arm. |
| Local receiver | Only a fragment sharing the geometric overlap can receive removed material; another owner elsewhere in the same cell cannot. |
| Polarity symmetry | Renumbering owners and material IDs and changing tuple order does not select a different physical donor. Physical ties remain explicit. |
| Material balance | For every material ID and every extensive quantity, initial plus births equals retained plus archived removals. |
| Rebinning | A different integration mesh changes bins, not physical polygons, material histories, contact identities, or inventories. |
| Restart | Serialization followed by continuation reproduces geometry, lineage, passive archives, and history. |
| Input preservation | All immutable source SHA256 values and existing production SHA256 values remain unchanged. |

Analytic toy geometry is used alongside the actual saved source. Measured runtime
and convergence are reported separately from conservation; conservation alone
does not establish spatial or temporal accuracy.

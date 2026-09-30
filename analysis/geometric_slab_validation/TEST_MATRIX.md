# Independent validation: persisted contact polarity and passive slab inventory

Baseline captured in `baseline.json` before this stage changes code. The scope
contains 176 established production/configuration files, 54 immutable source
files, five geometry files from the preceding stage, and four geometry test files.
Expected edits to the experimental geometry code are reported separately from
unexpected changes to the established runtime or source states.

## Acceptance matrix

| Contract | Independent check | Failure that must remain visible |
| --- | --- | --- |
| Physical polarity memory | A declared local subducting side persists when later local buoyancy/age ordering reverses | Memory cannot silently flip with scalar rounding or material renewal |
| Locality | Same owner pair has opposite polarity on separated arcs | Pair-wide memory cannot resolve an unrelated contact |
| Symmetry | Rotate coordinates and all vectors; permute owner labels and fragment enumeration | Coordinate sorting, owner numbers, and integration-cell numbers cannot choose the side |
| Birth of polarity | Unequal physical properties produce the declared fallback; equal properties require explicit evidence | Missing evidence and contradictory evidence stay unresolved |
| Arithmetic tie | One-ULP differences in specific buoyancy/age | Floating-point noise must not create physical direction |
| Attachment | Loss touches an actual retained donor/receiver contact | Shared cell, shared owner pair, nearest remote arc, or an unproven historical ID is insufficient |
| Exactly-once acceptance | Replay identical and altered event IDs after checkpoint; several fragments from one source cell | No duplicate inventory; atomic rejection of replay/conflict |
| Contact lineage | Split one physical supporting arc into child arcs, then merge/reorder; remove support | Sum of all four extensive quantities preserved; unsupported material stays archived and mechanically inactive |
| Cohort history | Different acceptance times and material histories at the same contact | No averaging of age, thermal thickness, damage, or acceptance time |
| Ledger | Surface + removed = initial + born for all four extents, per material identity | Geometric removal is not performed a second time during acceptance |
| Restart | Serialize after an intermediate transaction, continue, compare with uninterrupted execution | Contact/polarity/cohort IDs and histories reproduce deterministically |
| Passive scope | Input motion and source state remain unchanged; no torque enters the established mechanics path | Passive inventory must not be described as a speed prediction |

## Real-state validation

Use the immutable Starter checkpoint and saved continuation at 50 Myr from the
previous geometry stage. Record how many actual contacts and overlap candidates
have sufficient local physical evidence and how many remain unresolved. Report
pairwise overlap sums as candidate geometry, never accepted subduction area.
Where the source evidence cannot certify a complete differential step, keep the
last valid state and record the precise refusal. Common rotation remains a
zero-relative-motion control, not evidence of successful subduction initiation.

Synthetic closed-sphere partitions exercise accepted differential transactions
without changing actual-source physics. Check full ledgers, contact lineage, and
restarts on those controlled states. The established geometry and fractional
material tests provide the regression set; a full old-runtime suite is needed
only if established runtime code changes.

## Positive accepted-material artifact

`passive_connection_case.py` creates a reproducible 20-face closed-sphere
experiment with unequal, explicitly declared material buoyancy and ages. The
first 0.1 Myr uses differential plate motion; the following 0.1 Myr is common
rigid motion. Results are in `passive_connection_case/report.json` with three
joint checkpoints.

The differential transaction removes 20 km² and stores its 12 actual losses
once: seven have a certified geometric contact and five remain explicitly
unattached. The common rotation preserves those seven contacts, all cohort
payloads, all material histories, and support lineage. The maximum relative
per-material residual across the four extensive budgets is 1.59e-16. Direct
and resumed final checkpoints are byte-identical. Replaying the accepted
transaction is rejected; all inspected code hashes remain unchanged during
the run. This artifact contains no active slab force or recalculated plate
speed.

The independent test files are `tests/test_geometric_polarity.py` (39 cases),
`tests/test_geometric_boundary_io.py` (13 cases), and
`tests/test_geometric_slab_integration.py` (seven cases). They exposed and
retained regressions for malformed callback decisions, semantically invalid
but checksummed checkpoint geometry, false partial support under common rigid
rotation, and improper expansion of a half-contact polarity condition after
a geometric merge. The final combined project test log is owned by the main
implementation task.

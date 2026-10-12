# Explicit invariant cache: one measured fast path, no new physical steps

Base `31b3a95f7ba491d25a3f3916b648c63e6362ea65`. The existing BE equations,
physical parameters, audit tolerances, controller requirements and canonical
history are unchanged. **No nonlinear correction, new time step, planetary
series, integrator change, publication or archive operation was performed.**
The earlier real forced pair remains engineering-rejected.

## Result and scope of the measurement

**97 scoped tests PASS (12 new).** On authenticated saved inputs from the whole25
and two12.5 packets, uncached versus cached replay measured:

| Stage, identical workload | Baseline, s | Fast path, s |
|---|---:|---:|
| Initial geometry/basal assembly for three saved cases | 5.673 | 4.242 |
| Repeated geometry/basal assembly | 65.546 | 31.566 |
| Loading saved coefficients | 0.111 | 0.113 |
| BE matrix/RHS construction and scaling | 0.984 | 0.964 |
| **24 pure linear solves** | **0.0226** | **0.0246** |
| Memory update and energy ledgers | 0.109 | 0.116 |
| **Three independent coupled audits** | **41.736** | **26.142** |
| **Measured replay wall time** | **114.256** | **63.281** |

**1.8055× speedup**, or **44.6% less time** for this measured replay workload.
The isolated linear solve is about one millisecond; the bottleneck is rebuilding
geometric data and performing/checking integrals, not solving a stiff matrix.
Tiny differences in the unchanged millisecond kernels are timing noise, not an
algorithmic change. Assembly (including its initial calls) is roughly 1.99×
faster; independent audits about 1.60× faster.

One-time model loading/validation took **54.537 s** and input JSON reading
**0.699 s**. They are excluded from the 1.8055× ratio and are not claimed removed.
The complete benchmark took **232.949 s**, peak RSS **522,944 KiB**. Small report
writes are outside the phase timers; this benchmark does not time writing new
full trajectory checkpoints. The previous real pair's I/O/controller overhead
remains separately measured in its report.

The workload contains 24 frozen linear replays per path: for each of three
saved packets, four alternating visits to its saved converged predictor/final
inputs. This reproduces the assembly/cache access pattern of four corrections,
**not** their evolving intermediate states. No temperature, orbit, damage or
clock is advanced. Source/conduction/mechanical step entrypoints are guarded
against invocation. Therefore this is a measured cost reduction for the saved
current-step workload, **not an end-to-end speedup measurement of a newly evolved
step or a forecast for a Myr trajectory**.

## Cache contract and independent control

`cache.py` provides one explicit `FastPath` context. Nothing is installed in the
default runner. It is intended for one active Python worker in a process:
function bindings are scoped to the context and restored on exit, including
exceptions. No thread/concurrent-context use is claimed.

Every entry key contains:

- Geometry SHA256, including the literal immutable geometry and its embedded
  source/history binding.
- Material SHA256: cohort catalogue, actual intervals/masses and density.
- Source SHA256.
- Solver/code SHA256, including the relevant Python source manifest and runtime
  versions; additionally the particular function's file SHA256 and qualified name.
- The function's full typed argument content. Array dtype/shape/bytes and
  longdouble scalar bytes are preserved, rather than rounded through float64.

`bindings_for_geometry` constructs these bindings from the actual typed geometry.
A changed geometry at the basal-assembly boundary stops with
`GEOMETRY_BINDING_CHANGED`; exit and bind a new context. Rebinding a store on any
of the four SHA fields invalidates its entries. Other changed numerical arguments
miss naturally. There is no reuse based only on an object address.

Two completely separate stores are used:

| Producer store | Independent auditor store |
|---|---|
| Conical B0/B1 and vertex operators, radial moments | Its own angular/radial quadrature, conical scales and Cartesian strain operators |
| Geometry/source-bound mass, basal drag and traction assembly | Its independently computed mass, prescribed traction/drag and face geometry |

The producer basal cache excludes C/damage because the original basal assembler
does not consume it. Its key includes all consumed operator geometry/source/
clock/DOF/ID fields, contact parameters and actual operator type. Dynamic
constitutive tensors, Maxwell ratios, RHS, factorization, solution, damage,
thermal work and acceptance results are **not cached**. Outer source/geometry replay validation remains in the unchanged operator
assembler. On a basal invariant hit, its duplicate internal assembly checks are
replaced by the full consumed-input binding and geometry-SHA guard; changed
content misses or stops. No cached physical/audit verdict substitutes for the
state-dependent checks.

The independent auditor never reads the producer's cached matrices/operators.
It caches only results of its own independent invariant calculations, then
recomputes memory, forces, energies, balances and admissibility from each supplied
state. With the cache warm, deliberately corrupting the real saved strain is
still rejected as **BE memory mismatch**. A manufactured test also corrupts a
returned producer buffer and confirms it cannot affect the independent result.

Stored values are private immutable byte payloads produced in this process.
Each return is deserialized into a detached object; mutations of a returned
array cannot poison a later hit. No external pickle or persistent cache file is
read. Each store has a 64 MiB payload limit with eviction (metadata overhead is
additional); there is no unbounded history cache. Measured payloads were:

- Producer: **3,440,405 bytes**, 961 entries; basal assembly 1 miss / 22 hits;
  conical strain operators 160 misses / 36,640 hits.
- Independent auditor: **4,998,586 bytes**, 2651 entries; mass matrix 2 misses /
  16 hits; independent strain operators 80 misses / 7121 hits; angular rule
  1 miss / 16,806 hits. These are invariant hits, not skipped physical checks.

## Numerical and provenance verification

All 24 cached solver replays match the uncached replay **bit for bit**, including
velocity, displacement increment, memory and returned energy ledgers. Saved
velocity/memory values are also reproduced exactly. All three full independent
audit result digests match the uncached results. Raw canonical/pair input hashes
are unchanged. The original engineering rejection is not modified or relaxed.

Tests cover all four binding invalidations, changed array/scalar inputs,
longdouble distinction, bounded storage, detached return values, independent
store separation, forged-memory rejection, geometry-binding failure and restoring
original functions after normal/exceptional exits. `summary.json` binds the exact
raw reports in `/workspace/genesis-invariant-cache-replay-01/`:
`provenance.json`, `baseline.json`, `cached.json`, `report.json`.

The environmentDisconnected notification during the task was checked by one
read-only action. Access was intact and the existing process continued; no
benchmark or archive operation was restarted because of that notification.

## Opt-in use and reproduction

The tested fast path wraps the existing functions; it does not contain a second
solver or rewrite their arithmetic. After loading the authenticated model, an
explicit caller can scope its existing assembly/solver/audit work:

```python
from analysis.genesis_invariant_fastpath.cache import FastPath, bindings_for_geometry
from analysis.genesis_invariant_fastpath.replay import code_identity

solver_sha, _, _ = code_identity()
with FastPath(bindings_for_geometry(parent.geometry, solver_sha)):
    # Existing fixed-geometry assembly / solve / independent-audit callbacks.
    ...
```

Only the saved/replay workload was executed in this task. An existing physical
runner is not silently switched to this option. To reproduce the finite,
read-only-in-time benchmark, with a new report directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_invariant_fastpath.replay \
 --output /workspace/genesis-invariant-cache-replay-NEW
```

The benchmark caps wall time at 600 seconds and address space at 4 GiB. Its
`model._cache.clear()` only resets a process-local implementation cache for a fair
starting point; it deletes no files or accepted state. This is one before/after
measurement, not a statistical timing distribution. Full source/geometry replay
validation remains a significant uncached cost; no further broad refactor was
undertaken to inflate the measured speedup.

## Finite next proposal: events, then time accuracy — not implemented

1. Add a bounded **event-aware growth contract** to the existing BE controller:
   allow growth on smooth intervals while preserving output/work tests; bracket
   approaching solidus/new-material/water/fault boundaries and stop at the last
   admissible state if the transition is unsupported. Check intermediate stage/
   path indicators, not only endpoint signs. An error-estimate PASS alone must
   not allow an unobserved crossing. Start with manufactured event cases and the
   saved boundary margins; do not launch a long series as the test.
2. Separately prototype a two-stage **L-stable second-order SDIRK** treatment of
   the coupled source/conduction/mechanics problem, with explicit physical heat,
   numerical-work accounting and event checks. Merely upgrading mechanics while
   retaining first-order source splitting would not resolve the measured forced
   error. Use analytic relaxation/forcing/energy tests and the BE step-doubling
   reference. L-stability is not a proof of energy passivity or accuracy: reject
   an unexplained signed energy defect rather than relabeling it as heat.

Neither proposal, new integrator, event locator nor larger step is implemented
here. BE's stability already avoids mandatory seconds-scale resolution. This
cache improves cost, not the rejected 25-year accuracy. Large quiet-regime steps
and kyr/Myr throughput still require demonstrated output accuracy and correct
handling of the events actually reached, without changing physics or secretly
loosening the engineering requirements.

### Next implementation boundary: event contract before larger dt

Deliver an opt-in event contract and manufactured tests alongside the existing
transactional controller; do not change its default configuration or its current
50-year supported ceiling. This is a controller capability proposal, not approval
of a larger physical step. A later ceiling change needs separate evidence.

Each event observation must identify kind, cohort/face ID, physical clock, signed
margin with units, source SHA, supported transition handler (if any), and the
validity interval of any bound used to exclude a crossing:

- **Phase:** distance to the existing phase boundary from the actual thermal
  state and existing constitutive law, including intermediate thermal states.
- **Birth:** the next source-defined cohort/material creation time and its source
  binding; never extrapolate the old geometry through creation. A supported birth
  requires audited conservative state transfer and fresh cache bindings.
- **Gate:** existing damage/contact/water admissibility margins and identities;
  no replacement threshold or automatic permission to cross a closed gate.

The proposed controller first clips a candidate to known scheduled events, then
combines its existing error-based growth limit with a justified event-distance
bound. Samples alone cannot certify that a path is event-free: an unbounded or
ambiguous path returns UNRESOLVED and blocks growth. Same-sign endpoints do not
exclude an interior crossing or grazing contact.

For a bracketed event, localize from isolated copies of the last accepted state,
with explicit finite evaluation/time limits and a separately recorded localization
resolution. Never silently reuse an acceptance tolerance as an event tolerance.
A failed independent audit remains a STOP, not a localization hint. An unsupported
transition stops with the accepted cursor unchanged; an admissible prefix can
be committed only after its own complete error/work/audit checks. A bracket or
trial endpoint is not itself accepted history. Simultaneous events remain grouped
until the model defines a justified transition ordering.

Acceptance tests for this next implementation: smooth no-event growth under the
unchanged limits; exact scheduled birth clipping; interior double crossing;
grazing phase boundary; simultaneous phase/gate events; missing bounds; exhausted
localization budget; unsupported transition; audit failure; input/source mismatch;
and rollback/restart equality. Include a test that birth invalidates geometry and
material cache bindings. These are manufactured/controller tests first, with
saved-state diagnostics only: **no planetary series is part of this next scope**.

Only after that contract is demonstrated should the separate coupled order-two
prototype be assessed. Neither event handling nor L-stability establishes that
thousand-year steps meet the unchanged output and energy requirements.

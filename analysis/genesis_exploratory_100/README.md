## Executed outcome: STOPPED at source energy audit

The single authorized attempt is retained in `/workspace/genesis-exploratory-100-01/`.
It started coarse100 and stopped with **Source endpoint energy audit**, before
conduction/mechanics. No half1/half2 was dispatched, no complete trial was produced,
and no output-difference or numerical-work acceptance result exists for this pair.
No fallback, smaller step, repeat pair or tolerance relaxation was performed.

Elapsed **54.400 s**, setup **54.192 s**; peak RSS **477504 KiB** (466.31 MiB).
Before the final report, artifacts used **2046768 bytes**. The final report and
small static failure diagnosis are additionally hashed/size-listed in summary.json;
total remains well below 200 MB. Canonical bytes are unchanged.

A separate scalar algebra reconstruction, **without apply_ports or orbital
integration**, identifies a failing surface reservoir: T=1364.3779999 K,
energy residual **99437082836992 J**, unchanged 8-ULP allowance
**17592186044416 J**, ratio **5.65234375**. This does not rerun a physical trial.
It is an energy-accounting/numerical-accuracy blocker, not demonstrated remelt or
water departure. The surface root's temperature stopping tolerance does not by
itself guarantee the tighter energy-residual requirement. No source solver fix
or looser audit is included in this task.

Thus larger dt remains **unconfirmed**, not physically disproved. A source-stage
numerical correction that passes the same conservation audit is the concrete
prerequisite for any separately authorized retry. Full-path event certification
is still absent; even a subsequent output PASS would not automatically admit
history. The new exploratory budgets cannot bypass conservation.

# One exploratory 100-year diagnostic — isolated from verified/default policy

The run is exactly coarse100 versus half50+half50, starting from the pinned
0.8684 Myr canonical state. No retry, automatic reduction, continuation series or
canonical promotion is implemented. All outputs are diagnostic artifacts.
The previous verified25 rejection remains unchanged.

`config.json` explicitly selects temperature .5 K, specific enthalpy 600 J/kg,
stress 10000 Pa+10%, velocity 1e-12 m/s+5%, damage .001 and physical displacement
.01 m+5% of the interval increment. These are exploratory engineering requirements,
not physical constants or global-error guarantees. The existing 5%-of-remaining-
damage-margin requirement and 1e-4 gate exclusion margin remain. Global/local
numerical-work limits are copied exactly from the verified configuration and
checked for equality on startup. Conservation, signs, source/orbit debit, phase,
geometry, damage/healing, independent audits and typed restarts are unchanged.

The only legacy-path edit adds an explicit `diagnostic_max_step_years=100` option
to the mechanical helper. Its default remains 50. The verified config/parser and
runner remain unchanged. The new runner does not invoke adaptive integration;
it dispatches precisely the three authorized calls or stops at the first failure.
The fixed six-corrector limit remains. Full-path event certification is required
for acceptance, so this diagnostic runner always returns `accepted=false`.

## Real event checks and limits

`checks.py` reconstructs the actual fixed-cut conduction graph from current
physical face areas, cell masses, source conductivity/density and thermal fields.
It evaluates signed instantaneous heat-flow rates and the actual enthalpy inverse.
It also tests an invariant upper box: the source-free conduction maximum principle
bounds all temperatures by their global maximum. At an old node's solidus, the
worst allowed neighbor temperature gives an upper bound on its incoming flux.
If that bound is positive, the proposed box cannot prove that the node stays
below solidus. A negative instantaneous rate is never extrapolated as a bound.

At this input the closest old material is 736: margin .2462175 K and instantaneous
conduction cooling -.00684470 K/year. The conduction-only global upper bound is
2306.396 K; the solidus-boundary flux upper bound is positive, so it cannot certify
the old cohort. Source and thermoelastic/Maxwell feedback also need bounds.
These checks use real operators and fields but explicitly return UNCERTIFIED,
not a synthetic PASS or a claimed complete event locator.

For completed trials the same checks inspect actual source/conduction/mechanical
stage snapshots. The independent physical damage law supplies the actual frozen
loading/healing coefficients. Its exponential subproblem is monotone toward its
equilibrium, so its maximum is bounded by the before/after damage under that
frozen driver. The report labels this restricted bound explicitly: no envelope
for stress/temperature between stages means no full-path damage certificate.
Actual old-material remelt, new solid material, water departure, active/candidate
faults, healing or other existing physical-guard failure stop the run immediately.
No unsupported transition handler is added.

## Bounded runbook

Requires real /workspace, Linux, longdouble nmant=63, source-pinned inputs and
at least 2.2 GB free space. Hard wall-time limit 600 seconds includes setup,
solves, audits and report preparation; individual trials are at most 300 seconds
within that enclosing limit. Address-space limit 4 GiB; artifact allowance
200,000,000 bytes including a reserved final report, and 2 GB disk reserve.
The cache is the explicit verified FastPath context with separate producer and
independent-auditor stores, bound to source/geometry/material/code hashes.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_exploratory_100.run \
 --output /workspace/genesis-exploratory-100-NEW
```

This command is for an explicitly authorized new diagnostic only. Do not rerun
the completed/failed pilot to obtain a PASS. Output directory must be new;
trial files are read back before subsequent use, and every completed trial is
independently audited. A half1/half2 result, if produced, is never evidence that
a single coarse100 step is accurate or admitted.

Restore CLOSED. No publication, paid service, deletion, physics/default/tolerance
change or canonical evolution is part of this task.

Final validation: **144 scoped tests PASS**, including three new exploratory tests.
Run artifacts and all measured source hashes were checked after documentation.

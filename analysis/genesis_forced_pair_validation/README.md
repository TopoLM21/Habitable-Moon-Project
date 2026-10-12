# One actual forced25 / 2×12.5 control — COMPLETE, ENGINEERING REJECTED

Base `d5c01fe11f185134e491fc885a8ea852c4d6ff6a`. Exactly **two new 12.5-year
trials**, plus the reused existing forced25. No retry, smaller step, 100-year
series, changed budget, new physics, canonical/default change, publication or
archive operation. **44 targeted tests passed before execution.**

The pair starts from the pinned **0.8684 Myr** canonical state and its pinned
cut-cell thermal migration. This is a comparison of the original forced25
window, not the 100-year runner's later private starting point at 0.868425 Myr.
No HEAD switch or history reset was used.

## Whole-step compatibility and reuse

Original raw forced25 SHA256:
`e6ad0ed1372b18a73c3852f639edafd959776e6cb0d3a747aa75bcffe8c365a5`.
The exact old bytes remain untouched. `compatibility.json` records:

- All **356 compute Python files** under tectonics, BE solver/prototype,
  coupled split and forced validation are unchanged from producer commit
  `30b7154c4aa251a0ccdd4c73e20eb691608a3592`; individual hashes are included.
- Physical parameters SHA256 `6c4812689aa33380c8337593e4cd4faaa6d808645cd36cc88500676fc74c376f`.
- Reconstructed initial thermal packet SHA256 `0a6bef291ed30aeda1542a99777ea5da7a1bc432b02b0c118d816b459bf160e5`.
- Fixed cuts SHA256 `13a39cf9f31fd46781155c705f7f6eedc94e2706e601a71e853536f48fd394d1`.
- Coarse solver-input SHA256 `45a45648516e5bfd6236c416f7981c6683c4c5f0a541adce3aa0359f75875291`.
- Engineering configuration raw SHA256 `f2f4155160c2bd94cb81809aa84fbc6571de622426a5b7294f699471b1828700`.

The last configuration is a post-trial acceptance policy, not a parameter of
the old physical solve. Its engineering thresholds remain literal. Only horizon,
attempt and resource limits are narrowed for this pair. The derived whole view
has new wrapper/predecessor metadata and its own seal; its state, solver inputs,
predictor, thermal packets, ports and exchanges equal the original exactly.
The whole is independently audited again, but is **not solved again**.

## Limits, real resources and measured cost

Before execution: **6,009,958,400 free disk bytes**; cgroup limit **17,179,869,184
bytes (16 GiB)**. Its high current use was overwhelmingly file cache:
16,269,520,896 bytes file-backed, including 12,436,021,248 inactive-file bytes;
anonymous memory 49,741,824 bytes. Host available memory was 18,074,292,224 bytes.
No cache/drop/delete command was used. The in-run resource snapshot is saved.

Limits: **600 seconds total**, **200,000,000 artifact bytes**, **4 GiB address
space**, and 2 GB free-disk reserve. Result: **158.219 s**, peak RSS
**580,337,664 bytes (553.45 MiB)**, all pair artifacts **39,392,647 bytes**.
No limit was exceeded.

| Measured stage | Calls / work | Seconds |
|---|---|---:|
| Initial loading, binding and setup | Once | 54.916 |
| New mechanics | 2 trials, 4 correctors each; 16 predictor/final BE solves | 47.293 |
| Independent coupled audit | 3, including the reused whole | 41.544 exclusive |
| Source audits | 5: pre-mechanics for each half, plus all three coupled audits | 0.0043 |
| Source integration | 2 | 0.0022 |
| Conduction and local audit | 2 | 0.0929 |
| Observable extraction | 3 | 1.381 |
| Instrumented typed decoding | 2 parent restores + 2 new endpoint restores | 0.181 |
| Other serialization, checkpoint IO, copies, hashes and controller work | Includes reused-whole restart check | 12.805 |

There are **three endpoint typed-restart checks and three disk readbacks** in
all (including the reused whole). The second half consumes the disk-decoded
first-half checkpoint. No redundant byte-identical restart copies are stored.
Each trial receives the normal independent coupled audit; no tolerance/audit
was removed to meet the resource budget. Source-audit timings are separated
from the inclusive coupled-audit timing to avoid double counting.

The earlier **62–72 minute** figure was 36–42 × 102.8103 seconds: 11 accepted
windows × 3 trials plus 1–3 rejected windows × 3 trials. At four correctors per
trial this meant 144–168 correctors / 288–336 BE solves, 36–42 coupled audits
and 36–42 endpoint restart checks. It incorrectly treated the complete isolated
run's **one-time setup** as a recurring trial cost.

Here setup is 54.9 s and the controller portion 103.2 s for one reused plus two
new trials. Mechanics + audit + observable/IO overhead suggest roughly **40–44 s
per further new trial**, not 102.8 s. At the *old assumed trial count*, about
**26–32 minutes including setup** would be a better rough estimate. However,
this pair fails output tolerances, so the old trial-count assumption itself is
not validated; it must not be advertised as a measured 100-year runtime.

## Engineering outcome: reject, with successful audits and rollback

All three independent audits passed: sources/orbital debit, momentum, Maxwell
memory, local heat, whole energy, geometry and damage. All dissipation signs
pass. The 800 IDs remain; no birth, phase change, gate event or unfunded healing
was admitted. The engineering step-doubling test nevertheless **rejects**:

| Output / budget | Measured whole-versus-fine difference | Worst budget ratio |
|---|---:|---:|
| Projected world stress, tensor norm | 163,981.43 Pa maximum | **12.81498** |
| Physical corner velocity | 1.15388e−12 m/s maximum | **1.01141** |
| Damage | 4.97413e−5 maximum | 0.49741 |
| Temperature | 0.160821 K maximum | **3.21643** |
| Specific enthalpy | 192.986 J/kg maximum | **3.21643** |
| Fine-path numerical work, global | 1.39707e17 J | 0.40031 |
| Fine-path numerical work, local | Maximum cohort loss 1.91366e15 J | **4.42337** |

Maximum temperature difference is in cut-cell 1009, face 24, material 105.
The small surface difference (−0.002774 K) would conceal this larger material
error. This is real forced-path sensitivity, unlike the earlier frozen-forcing
material probe. The decisive metric is now **stress**, not N/Maxwell alone.

Fine Maxwell heat is 4.964742741e17 J versus whole 4.992730378e17 J; fine thermal
work 1.102604983e18 versus 1.435372691e18 J. Fine endpoint stored energy is
3.489732065e19 versus 3.504945532e19 J (−0.4341%). Fine maximum damage is
0.03997717, safely below the physical gate 0.65. Positive dissipation and passing
balance audits do not imply sufficient temporal accuracy.

The controller records **REJECT**, one attempt, zero accepted substeps, and
matching base/rollback hashes. Its accepted cursor stays at the beginning of
**this comparison**; previous saved trials/reports are untouched. It proposes
h≈**5.5869 years** from the existing error rule. **That step was not run.** No
threshold was changed to obtain a PASS and no long series was launched.

## Artifacts and next action

All working packets are retained in `/workspace/genesis-forced-pair-25-01/`:

- `001_whole.json`: derived view of the original whole, freshly audited.
- `001_half1.json`: raw SHA256 `f9f0dff6dad2dedfd52529536939d2c8eacf3554dba1a552a96839091ef7c5fa`.
- `001_half2.json`: working fine endpoint, raw SHA256 `5e8d430efa592c06be35ea967757ad81e63f0a0191706fa8f612173182e75fdb`.
- `initial.json`, exact `config.json`, compatibility/resource manifests,
  three audit events, rejection/stop events, and `report.json`.

The fine endpoint is a reproducible **trial result, not an engineering-accepted
continuation**. `report.json` raw SHA256 is
`8722bf15dcd276279409d7c480432eff7d1fb200bc8561ab2fa16076d29d3ba0`.
`summary.json` also binds the read-only verification report. Verification rechecks
all raw hashes/seals, the two-half thermal/orbit chain, unchanged budgets and
zero accepted progress, and reproduces the same assessment with **zero solves**.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_forced_pair_validation.verify \
 --folder /workspace/genesis-forced-pair-25-01 \
 --output /workspace/genesis-forced-pair-verification-NEW.json
```

Do not rerun the completed pair for reporting. The next possible finite control
is the proposed smaller macro-window with unchanged budgets, if separately
requested. Long-series admission remains pending this output-accuracy issue;
there is no requirement here to first resolve every unknown physical mechanism.
Canonical history and the four spatial controls retain their previous status.

## Large-step recommendation after the measured pair (code review only)

**A large step can be stable and still inaccurate or miss an event.** BE already
has stiff damping; no seconds-scale CFL restriction is imposed by this controller.
The tested window nevertheless fails accuracy at 25 years. Its material Maxwell
τ≈3–3.6 kyr does not alone prescribe a safe step: cooling-induced stress and the
source/conduction/mechanical split change appreciably much sooner. Neither a
small physical-heat difference nor a passing energy identity waives that error.

Current capabilities and limits, verified in code:

- `analysis/genesis_adaptive_control/controller.py` grows h by up to 1.6 when
  errors permit. The 50-year ceiling is an engineering diagnostic guard; the
  same ceiling is asserted inside `genesis_coupled_split_control/mechanical.py`.
  Removing it alone does not make 100/1000-year steps accurate.
- Phase/new-material/healing/fault guards **stop** candidate steps. They are not
  root-locating event integration. Large quiet intervals require both an error
  estimate and detection/localization of the next phase/front/gate event;
  endpoint-only checks cannot prove no interior event was missed.
- The older lumped `tectonics/genesis.py:advance` already uses implicit Radau,
  adaptive error control and event roots, with a default upper step of 0.01 Myr
  (10,000 years). That is an upper bound, not its actual step size. It evolves a
  much smaller reservoir model, not this 800-cohort coupled state; it cannot be
  silently substituted as an equivalent detailed solver.

The selected 0.5% stress/1 kPa, 0.05 K, damage and work budgets, safety factors,
50-year cap and attempt limits are engineering choices. Actual transitions such
as freezing/new cohort creation, water phase departure, damage activation and
unsupported healing are domain/event limitations. Today the measured limiter is
accuracy (stress ratio12.8), not an observed gate event or BE instability.
Changing output sampling intervals also does not change the integration cost.

Safe optimization candidates, **not implemented or assigned an unmeasured speedup**:

1. In `genesis_volume_dynamics._assembly`, the cache keeps only the latest damage
   for a geometry. Alternating predictor/updated damage can evict reusable work.
   Basal mass/drag/traction and geometric B/radial data depend on the fixed material
   geometry/source, not on that damage update. Separate immutable-geometry and
   constitutive caches, invalidating them on every relevant input change.
   In the BE solver, precompute fixed geometric contractions rather than rebuilding
   conical blocks for each cohort/corrector. Do not reuse a factorization when the
   actual matrix has changed. Verify optimized outputs against this saved pair.
2. Independent final-mechanics and local-heat audits repeat closely related
   cohort quadrature. Compute those independent local quantities once and feed
   both sets of assertions; keep source, conservation, sign, geometry and gate
   checks. Retain an independent reference replay to catch cache/invalidation
   errors. Simply dropping audits is not proposed.
3. Reuse the loaded model across trials (already done), and share immutable
   geometry/source by hashes in future checkpoint formats. This reduces redundant
   serialization without discarding state or restart evidence. Here only 39.4 MB
   was needed; writing alone was not the dominant measured cost.

The **47.3 s mechanical stage is not a timing of matrix factorization alone**:
its eight correctors include sixteen BE solves, loading, assembly, stress/damage
work and certificate construction. A finer saved-window profile is needed before
claiming which kernel dominates. Step doubling costs three trial calls for two
accepted fine substeps — 50% extra solves compared with that fine path. A validated
embedded estimator could reduce that overhead, but is not currently installed.

Practical route to kyr/Myr intervals: preserve the current working BE reference,
remove verified duplicate work, then evaluate a **higher-order stiffly stable,
energy-audited time integrator with event localization** on a short reference
window. In smooth regimes allow h to grow based on measured output error; shorten
only near fast forcing or events. A multirate or quasi-static slow-regime reduction
may later avoid repeatedly resolving already-damped fast mechanics, but needs
its own comparison/error criterion and must retain Maxwell memory, work and damage.
The rejected exact-memory coefficient swap is not such a validated method.

Thousands/millions of years cannot be promised by multiplying the current h:
new phases/cohorts/fault events require supported state transitions when reached.
No requirement is imposed to resolve all fast modes in seconds, and no blanket
requirement to solve every unknown physical mechanism before optimizing this
window. No additional performance or large-step experiment was run beyond the reported
pair; this is the recommendation before selecting the next finite task.

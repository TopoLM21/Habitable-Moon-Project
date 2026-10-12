# Opt-in adaptive controller for the working forced BE pipeline

**Subsequent real pair: audits PASS, engineering REJECTED.** See
[the forced25 / 2×12.5 report](../genesis_forced_pair_validation/README.md).
Exactly two new trials were measured in 158.2 s / 39.4 MB. The controller
proposes 5.5869 years, not executed. The original implementation-only evidence
below is historical; the real forced pair now exists and fails output budgets.
The configured private coarse seed has not gained engineering admission for a
long run. No default/seed change is made silently.

Base `614a7cbcde35ea5fd2d85e6677dc22be6ea010cb`. **No planetary evolution was
run for this implementation.** Production, canonical history, physical laws,
source coefficients and the BE solver are unchanged. The rejected exponential
coupling prototype is not imported or integrated.

This is a usable, separate runner for a later bounded **100-year private trial**
from the authenticated forced25 endpoint **0.868425 Myr**. It does not restart
Genesis. The original 0.8684 Myr / 800-cohort / four-interval canonical history
stays immutable. The 800 existing IDs and actual evolved orbit are carried into
each candidate. The model is still restricted to the already supported fixed
material, solid, inactive-fault window.

## What the requirements mean

`config.json` is an explicit **engineering pilot configuration**, not a set of
physically derived tolerances, validated global error bounds, or production
defaults. The numbers express reviewable output/work requirements. They are not
chosen from the fastest eigenmode or from a claimed 63.4% stress error. The
previous 63.4% N/Maxwell ratio is a work diagnostic, not an output error metric.

For every output, compare a whole BE step h against two BE steps h/2. For this
first-order method the fine local-error estimate is |fine−whole|/(2¹−1).
Accept the actual fine path; **do not Richardson-extrapolate state or damage**.
Endpoint comparisons alone do not prove continuum/global accuracy.

| Metric | Pilot requirement and units | Reason / spatial coverage |
|---|---|---|
| Stress | 1000 Pa + 0.005 × larger tensor norm | World-space projected face stress; Frobenius norm counts shear twice. Each face and global RMS must pass. This is the existing in-plane damage-driving stress, not a newly inferred full 3-D stress. |
| Physical velocity | 1e−12 m/s + 0.01 × larger speed | Compare velocity **vectors**, not raw generalized coefficients or just speeds. Six conical corner values per cohort include radial shape terms; convexity gives the maximum on the affine conical field. Also report global RMS of these samples (not a volume-weighted RMS). |
| Damage | Absolute difference ≤1e−4 | Check every face; also require difference ≤5% of remaining gate margin. |
| Temperature | Absolute difference ≤0.05 K | Every cut-cell and both reservoirs; global RMS also reported. |
| Specific enthalpy | Absolute difference ≤60 J/kg | Every cell/reservoir; detects latent-energy changes even when T is insensitive. Roughly matches 0.05 K × inherited 1200 J/kg/K for silicate, but remains an engineering requirement. |
| Global numerical work | N ≤ 1e12 J/year × h + 0.01 × max stored energy | Sum memory loss and kinetic numerical loss over the **two fine steps**; reference is elastic + kinetic energy. No division by Maxwell heat. |
| Local numerical work | N_i ≤ 1e9 J/year × h + 0.10 × max elastic energy_i | Every cohort; sum both fine-step memory losses. Absolute terms protect zero/small reference energies. The 10% cap is deliberately a coarse pilot work requirement, not a 10% stress-error claim. |

Absolute work allowances are **rates multiplied by the macro-window duration in
years**, so splitting does not buy repeated fixed allowances. Relative allowances
refer to the largest stored energy at the fine-step boundaries, not a sum that
would double the allowance. Local old strain is evaluated in each substep’s
endpoint damage metric, as in the BE memory-work identity. Both global and local
gates are mandatory. The
relative work budget is per accepted window, not a guaranteed cumulative
100-year relative bound; the run journal retains every window's work for review.
Physical Maxwell heat, drag, numerical kinetic loss and every local memory loss
must be finite and nonnegative for **all three trials**, including the discarded
whole step. Negative dissipation stops the run; it is never repaired by clipping
or returned as heat. Existing independent audits still check balance and signs.

## Transaction, event guards and bounded adaptation

1. Deep-copy the last accepted snapshot. Calculate/audit the whole step.
2. Calculate/audit half1 from the **same** snapshot, then half2 from half1.
3. Check output errors, work budgets, signs and gate margins. Emit the acceptance
   journal only after all audits and checks pass. Commit half1 + half2 together.
4. On an ordinary budget rejection, retain the attempted files, verify the base
   snapshot hash is unchanged, and propose a smaller h. The accepted cursor,
   orbit debit, thermal sources, memory and damage do not accumulate trial work.

Local output error scales approximately as h². For work the controller solves
`N r² ≤ A r + R` for the admissible step multiplier r, where A is the absolute
rate allowance and R the relative stored-energy allowance. Use the tightest
prediction, safety 0.8, and a multiplier between 0.2 and 1.6. Recheck actual
results; these scaling predictions are not convergence proofs. This avoids
mandatory repeated halving or resolving superfast damped modes in seconds.

A reached gate, nonzero candidate age/active fault, unsupported phase/birth,
unfunded healing, failed audit, input mutation or solver exception causes a
**STOP at the previous accepted cursor**, not a blind refinement loop. The
existing predictor/final checks apply at each candidate; additionally keep
absolute damage clearance 1e−4 and an error allowance inside that clearance.
A whole-step gate event also stops conservatively; there is no event-crossing
or event-location implementation. Endpoint/estimator checks do not certify the
unobserved continuous path. They never license stepping across the model gate.

Hard pilot limits:

- Macro h: 0.25–50 years, initial 25 years. Finest trial half-step is 0.125 years.
- Horizon: at most 100 years. At most 24 attempts, 20 accepted macro windows,
  five consecutive rejected attempts. Existing mechanics caps each trial at
  six correctors. No progress beyond these limits.
- Wall clock: **5400 s total**, including setup; **300 s per trial**; **4 GiB**
  address space. Nested alarms preserve the enclosing total deadline.
- New artifacts: **2,000,000,000 bytes**, reserving 1 MB for final status;
  retain at least **2,000,000,000 free bytes**. Existing paths are refused.
  Rejected evidence is kept; nothing is automatically deleted.

Whole/half snapshots are saved under attempt labels. Acceptance events include
canonical JSON snapshot SHA256 values (via `controller.fingerprint`, **not** raw
file hashes including the trailing newline), and identify the accepted attempt's
half1/half2. The half2 checkpoint contains its predecessor checkpoint seal. Each
file is flushed/fsynced before its successful write returns. On STOP use the
last acceptance event; if none exists, use the seed. A failed final status write
does not license an unjournaled acceptance. No canonical pointer is updated.

## Verification performed without a new series

**31 new tests; 85 scoped tests PASS.** Manufactured tests cover BE fine-path
acceptance, audit-before-acceptance, rejection rollback, half2 failure, negative
dissipation, gate clearance/error margin, near-zero absolute tolerances, local
loss hidden by a global sum, bounded attempts/rejections/minimum step/runtime,
durable-journal failure, nested clocks, world tensor rotation and stiff drag
without a relaxation-time step restriction. Existing BE/conical/ports/restart
and isolated exact-Maxwell tests remain green.

Read-only replay authenticates and independently audits saved BE50, both saved
25-year steps, and forced25 (including its actual source/orbit audit). The real
unforced step-doubling pair **passes**, worst budget ratio **0.520943**:

| Saved output difference | Measured value |
|---|---:|
| Maximum world projected-stress tensor norm | 7164.59 Pa |
| Maximum physical corner-velocity vector difference | 4.89446e−14 m/s |
| Maximum damage difference | 2.09924e−5 |
| Maximum temperature difference | 0.00763590 K |
| Maximum specific-enthalpy difference | 9.16308 J/kg |
| Fine-path global/local numerical-work budget ratios | 0.011854 / 0.074558 |

The saved **single** forced25 passes the global work budget (0.9030) but fails
local work (9.3922). This does not fabricate a forced step-doubling pair: **no such
saved pair existed at the implementation stage**. The subsequent real pair is
now linked above; it rejects the larger window under unchanged budgets.
The adapter was initialized with evolution functions replaced by failure guards;
its real checkpoint, typed memory and full orbit restore pass with **zero**
evolution calls. Linux/x86_64, Python 3.12.14 and longdouble nmant=63 verified.
All pinned raw input hashes remain unchanged. `summary.json` links exact report
paths/hashes and the configuration hash.

## Next bounded run and expected resources

The runner implementation supports a **separate** bounded 100-year trial,
not executed here; the subsequent rejection must be accounted for before
claiming an engineering-admitted start and step schedule. Existing physical uncertainty is recorded rather than made an artificial
prerequisite for this already supported pipeline. The run still stops on its
actual phase/fault/healing/material/audit/resource boundaries.

From the saved forced rate, two half steps would approximately halve numerical
work. The local budget predicts a macro-window near **11.5 years**, or **9.2 years
with safety**. This is a cost estimate from the older forced25 interval, **not**
a forecast that the new starting state's rates remain identical. Roughly 11
accepted windows plus 1–3 rejected attempts means **36–42 trial calls**, about
**62–72 minutes at the prior 103 s/trial**, plus setup/IO differences.
This historical estimate overcounted recurring setup: the subsequent measured
pair gives about 40–44 s per additional trial, but rejects the old step/count
assumption, so this is not a validated 100-year resource forecast. Expect
roughly 1–1.5 GB of snapshots. Current read-only preflight found **6,010,347,520
free bytes**; caps are enforced again while writing. 90 minutes is a hard stop,
not a guarantee that all 100 years will finish. Output errors may demand smaller
steps or a phase/gate may stop the run earlier.

The subsequent pair has now exercised the actual sequential adapter and
measured the forced whole/half difference. It rejects the large window, so
engineering output accuracy at an admissible step remains to be demonstrated. It may produce a justified partial endpoint instead of 100 years.
Spatial closure/admission and unresolved heat return/healing laws remain open;
no physical plan stage is declared complete by this controller.

Read-only replay:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_adaptive_control.replay \
 --output /workspace/genesis-adaptive-replay-NEW.json
```

Future opt-in run (not executed in this task):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_adaptive_control.run \
 --config analysis/genesis_adaptive_control/config.json \
 --output /workspace/genesis-adaptive-100yr-NEW
```

The frozen seed is the existing forced25; this command is not a general-purpose
resume or automatic long-evolution service. The final report and acceptance
journal provide the exact last accepted private checkpoint for the next task.

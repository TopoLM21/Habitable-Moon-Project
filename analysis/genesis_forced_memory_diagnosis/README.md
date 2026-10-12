# Saved forcing diagnosis and isolated exact-Maxwell prototype

Base: `30b7154c4aa251a0ccdd4c73e20eb691608a3592`. Restore CLOSED.
**No new planetary steps. No production/default change.** All three inputs are
pinned in `summary.json`; the 800 material IDs match the canonical 0.8684 Myr
state. No births are present. The earlier forced25 and unforced25/50 remain
unchanged. This diagnosis does not accept any additional history.

## Cause of the 63.4% memory loss

With the existing positive conical metric H, BE gives
`Δe = BΔu − Δthermal − z e1`, `z = Δt/τ`, and
`Nmemory = ½ ||Δe||²_H`. This is a discrete work/energy residual, **not** an
estimate that the endpoint stored energy itself is wrong by 63.4%.

| Saved trial | Nmemory, J | Nmemory / Maxwell heat |
|---|---:|---:|
| Unforced 25 yr, same initial state | 1.985129805e15 | 0.4058% |
| Unforced 50 yr | 8.207145388e15 | 0.8514% |
| Forced 25 yr | 3.165279724e17 | 63.3978% |

For forced25, the squared contributions are deformation **2.682562e17 J**,
cooling **5.931355e17 J**, relaxation **1.781486e15 J**. Their signed cross terms
are deformation/cooling **−5.365191e17**, deformation/relaxation **−9.098491e13**,
and cooling/relaxation **−1.003511e16 J**. They sum to the reported Nmemory;
these are overlapping quadratic terms, not independently additive heat channels.
The same-dt unforced relaxation term is 1.766070e15 J. It is the strong cooling
and associated constrained deformation that change the result most.

All material is old. Temperature changes by −1.618 to −0.439 K in these cohorts.
τ is **3068–3634 years**, z **0.006879–0.008148**: unresolved rapid *Maxwell
relaxation* is not the principal cause. The global ratio hides spatial severity:
local N/Q median **6.114**, maximum **59.065**; local N/U_after median **0.09625**,
maximum **0.94138**. A small mean thermal response cannot waive this mechanical
work defect. Conversely, large N/Q in a low-Q cohort is not itself a relative
stress-error estimator.

## Exact material comparison at the SAME saved forcing

`exponential.py` solves `de/ds=f−ze`, with constant f = BΔu−Δthermal and frozen
endpoint H and τ. It integrates physical heat `∫ z eᵀHe ds` and mean stress
work analytically. Saved displacements/temperature/damage remain fixed; **no
alternative mechanical equilibrium or thermal trajectory is solved**.

| Forced25 quantity | BE | Exact material probe |
|---|---:|---:|
| Endpoint stored energy, J | 3.504945532e19 | 3.505272538e19 |
| Maxwell heat, J | 4.992730378e17 | 4.942223522e17 |
| Thermal work into mechanics, J | 1.435372691e18 | 1.116979440e18 |

Stored energy changes +3.270058e15 J (+0.00933%); heat −5.050686e15 J (−1.012%).
The much larger thermal-work change is **−3.183933e17 J (−22.18%)**, essentially
the old numerical-memory scale. Mean stress, rather than endpoint stress, must
be used for the exact path work. Using the exact endpoint in the old work
formula would give 1.437566e18 J and retain the work inconsistency.

Maximum face-stress change is **3435.75 Pa**, relative face-array L2 **0.03368%**;
local material energy-norm change peaks at **0.38455%**. Re-evaluating the inherited
damage law with exact *predictor* stress gives max |Δdamage| **4.0681e−7**;
maximum damage 0.0400266912 → 0.0400270953, below activation 0.65. This is a
fixed-forcing sensitivity probe, not accepted damage or proof of trajectory
accuracy. Endpoint viscosity differs from starting viscosity by 0.405–1.540%
(read directly from saved cohort temperatures and inherited Arrhenius law), so
freezing it is an approximation of comparable order to the heat difference.
Neither intermediate forcing curvature nor the true continuous path is saved.

The exact material balance is checked per cohort, and physical heat independently
integrated with 24-point temporal Gauss quadrature. Summary/full-report hashes
are in `summary.json`. Absolute summed material residual is about 243 J against
3.5e19 J stored energy. Raw artifacts are never rewritten.

## One minimal prototype, and its actual limitation

`prototype.py` couples endpoint BE momentum to the exact constant-rate memory:
`[M/h+D+h b BᵀHB] v1 = Mv0/h+F−BᵀH(a e0−b Δthermal)`,
where `a=exp(−z)`, `b=(1−a)/z`. Only small manufactured systems are exercised.
There is no Genesis adapter, birth support or production import.

It closes the total balance with exact physical Maxwell heat, time-mean thermal
work, positive BE kinetic loss, and the **signed** coupling term
`C = (e1−mean(e))ᵀH BΔu`. C is never heat. For the saved forced kinematics its
material probe is **−3.118600e13 J**; that state is not a solution of the new
momentum equation. An independent manufactured forcing case proves that even
`C+Nkinetic` can be negative. Thus this simple swap is **not an unconditionally
energy-dissipative replacement** and is not admitted to production.

For a normalized scalar Maxwell/elastic mode, amplification has characteristic
polynomial `λ²−(1+a)/(1+h²b) λ+a/(1+h²b)`. Jury stability holds and both roots
tend to zero in the stiff-step limit, including the elastic γ=0 limit. This
checks stiff damping of the minimal coupled mode; it is not a proof for every
noncommuting matrix/damage/thermal system. Using mean stress in momentum would
close the work directly but loses this stiff elastic damping (a mode tends to
−1). These limits explain why merely replacing the BE Maxwell coefficient is
insufficient.

## Single recommended next path — a work-budget controller, not a dt sweep

Retain the already audited BE scheme for the next *separately authorized*
bounded control. Gate acceptance using its exact nonnegative numerical ledger:
`ΣN_i ≤ εQ ΣQ_i` **and** `N_i ≤ εU U_after,i` for every material cohort. If a
reference energy vanishes while N>0, reject; do not hide it behind a denominator
floor. The second gate prevents the global sum from masking local mechanical
loss. Numerical loss remains outside physical heat. Also retain existing
phase/birth/healing/activation guards. These are proposed numerical tolerances,
not derived physical coefficients or a claim of proven solution error.

For approximately fixed rates N scales as h², Q as h. Choose a prospective step
from `h * min(εQ/(N/Q), sqrt(εU/max(N_i/U_i)))`, then check the actual ledgers
and measured changes of temperature-dependent τ and forcing. Endpoint data do
not bound curvature; reject if the rate prediction fails. No sweep or new
trial was run here.

Illustrative εQ=5%, εU=1% gives **h≈1.97 years**, at least **13 steps per 25 years**;
εQ=1%, same local budget gives **h≈0.394 years**, at least **64 steps**. At the
previous 103 s/step these are rough **22 min / 110 min** costs before retries;
corrector counts and evolving rates can change them. Imposing 5% of *local Q*
on every cohort instead would demand ≈7.73 days / 1182 steps (≈34 hours), and
is not justified solely by its tiny local dissipation denominator. This cost
comparison is why a local stored-energy gate is proposed explicitly instead.

The next implementation work is that bounded work-budget controller and its
manufactured acceptance/rejection tests. It is not installed by this diagnosis.
Spatial controls, source/conduction splitting, damage/drag heat return and
funded healing remain open; no physical plan stage is closed here.

## Reproduce without advancing the planet

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_forced_memory_diagnosis.diagnose \
 --output /workspace/genesis-forced-memory-diagnosis-NEW.json
/workspace/habitable-moon-venv/bin/python -m pytest -q tests/test_genesis_exact_maxwell_diagnosis.py
```

Existing output paths are refused. Tests cover independent matrix exponential
and heat quadrature, pure relaxation, zero/large z, manufactured forcing and
thermal work, stiff scalar modes, and the negative signed-defect counterexample.

Validation: **20 new tests; 54 scoped tests total PASS**, including existing BE,
conical adapter, thermal ports and coupled restart checks. No real-step runner
was called. Canonical and all three input raw hashes were rechecked unchanged.

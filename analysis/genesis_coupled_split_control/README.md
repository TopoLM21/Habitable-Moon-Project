# Bounded complete coupled-chain control: saved 50 years versus 2 × 25

Base `f4354e0589baeea9ed9d137459ecf8170f0bbf5f`. Restore CLOSED.
The starting thermal state is the same authenticated migrated cut-cell
initialization as the saved BE50 control; the mechanical parent is the same
accepted 0.8684 Myr checkpoint. There are exactly 3360 fixed material cut-cells,
plus deep mantle and water. No new mesh, source profile, initial memory or
production/default state is substituted.

## Single reproducible validation action

From the repository, using a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /workspace/habitable-moon-venv/bin/python -m analysis.genesis_coupled_split_control.run \
 --output /workspace/genesis-coupled-split-25-NEW
```

This one isolated command loads pinned dependencies, performs **at most two new
coupled 25-year steps**, independently audits each, decodes its restart, and
compares the final endpoint with the saved 50-year trial. It never reruns BE50.
The directory must not exist; existing results are not overwritten. Maximum
wall compute budget **600 s**, memory **4 GiB**, maximum **6 correctors per step**.
No sweep or automatic retry follows failure. Do not rerun simply to inspect the
completed control: read the saved report and its hashes instead.

Dependencies are explicitly pinned in `run.py`: original canonical raw state,
conservative cut-cell candidate, migrated reservoirs, saved conduction50 and
saved coupled BE50. The command requires the existing project venv and these
files at their current paths; it does not restore archives or fetch anything.

## What each step does

1. Implicit enthalpy conduction with existing conductivity and actual displaced
   face areas at the beginning of the step, fixed masses and cut boundaries.
   Independently sum phase enthalpy, paired edge energies and each node's balance.
2. Reject old-material melting, a newly fully solid unowned cut-cell requiring
   unsupported birth, or departure of water from the supported hot-vapor branch.
3. BE predictor, inherited damage evolution, **local healing guard**, gate before
   activation, final BE; repeat corrections from the same predecessor memory.
   Maxwell heat goes to its unique material cut-cell; thermal work has the
   equal-and-opposite thermal entry. Every correction starts from the same
   conduction endpoint, so no loss/work/heat is repeatedly accumulated.
4. Require temperature mismatch <=1e-8 K and per-cohort exchange change within
   1000 J + 3e-8 times the exchange magnitude. Reject unsupported phase changes
   after feedback too. Gate and inherited elastic/momentum limits remain active.
5. Independent predictor/final spatial audits of memory, source rheology,
   impulse and work; independent damage law and geometry certificate; 800 local
   heat/work checks and open-boundary budget. No unverified state is admitted.
6. Serialize, verify seal, decode typed memory/fault history, check byte-identical
   restart; step 2 actually uses the decoded step-1 arrays and memories.

The private mechanical view retains the immutable reference chart; the new
packet's `thermal` arrays are authoritative. Legacy thermal fields are used
only as a private clock/orbit carrier, never as input enthalpy, and this view
is not published as a production checkpoint. The predecessor byte hash and
step index preserve chain identity. Internal energy exchanges are not summed
again in the external budget; cumulative work/losses and endpoints are checked
separately at the end.

## Explicit boundaries and stop conditions

Prescribed traction is an open mechanical-work port. Basal drag is exported to
an unresolved interface. Damage release remains a separate nonthermal accounting
store. **No heat returns from unresolved drag/damage during this window** is a
bounded hypothesis, not a universal physical law. Numerical loss never heats.
Frozen orbital parameters drive the inherited damage law; no new orbital/tidal
heat source is introduced. Water remains vapor, ocean fraction zero.

The actual healing law is evaluated, not disabled. Any local decrease in damage
stops before final BE, since no constitutive healing-energy donor is specified;
a gain on another face cannot hide it. New material, phase events or activation
outside the supported fixed-catalogue branch also stop with an explicit blocker.
These guards limit the runner's domain of validity, not the physical laws.

## Interpretation of the comparison

Compare temperature, specific enthalpy, melt fraction/column liquid mass, the
surface-connected fully solid front, velocity, face stress and damage. Report
absolute and relative differences rather than declaring a percentage tolerance
that has no physical justification. The two grids here are identical; the two
time discretizations differ in both BE errors and conduction/mechanics splitting.
Thus their difference measures **combined temporal/splitting sensitivity**, not
an independently isolated splitting error or proof of convergence. No extra
control is silently added to separate those effects. The feedback temperature
increments are an additional splitting-lag scale, not an error bound.

Initial-profile sensitivity, the original four spatial controls, physical
fracture admission and unresolved thermal return/healing remain open. A passing
control supports only bounded sequential trials that satisfy these guards;
it does not authorize or establish long planetary evolution.

## Confirmed saved result

**PASS, exactly two new coupled steps**, 0.8684 → 0.868425 → 0.86845 Myr.
Saved files: `/workspace/genesis-coupled-split-25-01/`.
Each step needed **4 correctors**; no healing, unsupported phase/birth or gate
activation occurred. All 800 material IDs and masses persist. Maximum damage
is 0.040178094, below the existing 0.65 gate. The nonlinear temperature mismatch
is zero in the final stored arithmetic on each step.

The producer took **122.35 s** before an audit-only cancellation stop: predictor
of step 2 had identical old/new damage, but independently summing two elastic
energies around 3.39e19 J and subtracting them yielded 8192 J instead of zero.
The new independent audit integrates `1/2 e_old : (C_after-C_before) : e_old`
before summation. **No tolerance was increased, no physics was changed, and no
third physical step was run.** The older auditor remains intact. Manufactured
large-energy/zero-damage and small-stiffness-change tests cover this correction.
Original `report.json` and failure evidence remain preserved; `completion.json`
is the successful read-only completion, taking 32.85 s. Both restart files are
byte-identical to their step files; step 2 was already run from decoded step 1.

Final endpoint raw SHA256:
`ad8810c6f6c537e6252deeaccafda569dcc403431a3442651934b7086992ad6d`.
**81 tests PASS.** A final single read-only package verification also passed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1  /workspace/habitable-moon-venv/bin/python -m analysis.genesis_coupled_split_control.verify  --directory /workspace/genesis-coupled-split-25-01
```

This verifies the saved predecessor chain, packet seals, independent mechanical,
local-heat/damage/geometry audits, typed restart bytes and endpoint comparison.
Both thermal and mechanical solve entrypoints are forbidden in this action.
Its captured result is `/workspace/genesis-coupled-split-final-verification-01.json`.
There is no need to rerun the two-step producer for inspection.

### 2 × 25 minus saved 1 × 50 years

| Field | Maximum absolute difference | Relative norm |
|---|---:|---:|
| Temperature | 0.00763590 K | 1.7513e-6 |
| Specific enthalpy | 9.16308 J/kg | 1.5470e-6 |
| Liquid fraction | 1.12142e-7 | 2.0952e-8 |
| Surface-connected solid front q | 0 | 0 |
| Face stress | 5182.88 Pa | 0.050116% |
| Damage | 2.09924e-5 | 0.058556% |
| Velocity coefficients A/P | 3.16836e-10 m/s | 10.6733% |

Liquid column mass differs by **−1.227573e14 kg**. Norms in the table are ordinary
array L2 differences normalized by the larger endpoint norm; absolute values
and full fields are retained. They are not global physical energy norms.

In the actual mass-integrated physical velocity field, RMS speeds are
**6.308576e-11 / 6.308995e-11 m/s** for 50 / 2×25 years; RMS difference is
**3.702802e-14 m/s**, relative **0.058691%**. The large raw coefficient difference
must not be mistaken for the same relative physical speed difference: P is
multiplied by lambda−1 in the conical field. It is nevertheless reported, not
hidden by the physical norm.

The feedback temperature increments after conduction are **2.174012e-5 K**
and **2.249094e-5 K**. These quantify the lag scale of the split thermal source;
the **0.007636 K** endpoint difference is the observed combined time/splitting
sensitivity, not a pure splitting-error estimate. Solver convergence at 1e-8 K
does not establish time-discretization accuracy.

Cumulative mechanical exchanges also change: Maxwell **+0.746%**, drag
**+0.706%**, damage release **+3.823%**, magnitude of thermal work **−9.898%**;
BE numerical memory loss **−50.247%**. Small temperatures alone therefore do
not close energy/work accuracy. Final kinetic, elastic and thermal energies
are endpoints, while only interval works/losses are summed.

The whole two-step open budget residual is **1.016979e15 J**, within the summed
thermal-endpoint/feedback allowance **1.872693e16 J**. This broad whole-thermal
bound is supplemented, not substituted, by tight per-step mechanical impulse,
energy, memory and local heat audits. Thermal maximum node balance / allowance
is **0.03276 / 0.02882** for the two steps. The global roundoff allowance cannot
by itself resolve the smaller mechanical work terms.

### Readiness and remaining conditions

The complete chain is usable for **bounded sequential trials in the validated
fixed-catalogue, hot-water, inactive-fault regime**, including restart between
steps. No further sweep is needed to establish that operational result.
It is not an unconditional planetary evolution runner: local healing without a
donor, new fully solid unowned material, old-material melting, water-phase
change, gate/elastic/momentum failure or correction-budget exhaustion must stop.

Keep the no-return-heat assumption for unresolved drag/damage explicit. Temporal
and splitting accuracy, initial-profile sensitivity and spatial/physical-fracture
admission remain open, especially the observed work sensitivity. Neither 50 nor
25 years is promoted to production/default; canonical history is unchanged.

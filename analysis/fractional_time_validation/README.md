# Validation of fractional event timing

`baseline.json` records the seven fractional production-file hashes and the
six results from the preceding `fractional-heat-basal-coupling-1` experiment.
`baseline_code/` contains the local immutable source snapshots captured before
this correction. The original result folders and source saves are preserved.

The new cases use the same sources, durations and outer timesteps as that
baseline. `run_validation.py --case NAME` creates a new output directory and
records its code/report hashes. `late_restart_second` additionally requires
`--resume` pointing at `late_restart_first/fractional_checkpoint.json`.
`summarize_validation.py --plot` checks all source/code/output hashes, verifies
the exact restart, and compares timestep sensitivities with the baseline.

These are short **basal-only** experiments. They do not contain slab pull,
ridge push, evolving fractures or a new interface reconstruction. Two Myr of
temporal refinement does not establish long-term or spatial convergence.

## Independent numerical oracle

For constant boundary temperatures, the existing half-space solidus closure
has depth `h(a) = K sqrt(a)` until reaching the reference lid. A constant
creation flux during a step of duration `dt` produces a uniform distribution
of endpoint ages on `[0, dt]`. For chemical crust thickness `H`, the exact
average cold-mantle thickness is

```
a0 = (H / K)^2
mean(Hcold) = [2 K / 3 * (dt^(3/2) - a0^(3/2)) - H * (dt - a0)] / dt
```

when `a0 < dt` and the lid cap is inactive; otherwise the corresponding
zero/capped intervals must be included. The unit test uses a positive small
`H` so neither a zero chemical reservoir nor a lid-cap branch hides the
birth-time integration error. It checks convergence of 2, 4 and 8 positive
Gauss nodes against the closed integral. It separately verifies mean age
`dt / 2`, second age moment `dt^2 / 3`, and actual birth callback times.

For `H` approaching zero the 2-node relative quadrature error is about
1.0831%, and the 8-node error about 0.0253%. A 2-node rule is an improvement
over endpoint births, not an exact integration of square-root cooling.

## Checks with the actual Genesis thermal model

`tests/test_fractional_time_coupling.py` also checks that every removed
primordial parcel receives the reference-lid thickness and density evaluated
at its own removal time; its age, area, chemical volume and material memory
remain consistent. Previously removed material stays unchanged on subsequent
steps. Checkpoint/resume reproduces all cohort clocks and cumulative source
ledgers exactly. Existing coupling tests independently check that Genesis
retains sole ownership of global heat and no second cooling sink is added.

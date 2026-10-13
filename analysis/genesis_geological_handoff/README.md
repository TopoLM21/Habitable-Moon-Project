# Simplified Genesis → geological continuation

13 October 2026. Opt-in branch `feat/genesis-geological-handoff`, base
`2a9f1a6483d92dc00b9d775fa3e88664984e1b90`. Restore is CLOSED.

The earlier mistaken task came from a repeated inbound restoration delegation.
The old CLOUD_HANDOFF header also proposed further exact contact work; it is
now explicitly historical. This directory and the new header describe the
approved geological alternative. No security or AGENTS instructions changed.

## Implemented path

`born_synthetic()` starts with one owner and zero damage. Existing simplified
cooling/Maxwell/damage and persistent domain selector produce 1→2→3 owners in
two 25-year steps. No daughter owners or velocity kick are prescribed.
The inherited two stress belts are manufactured initial data, not spontaneous
stress from a real planet. Cooling, thermal expansion and damage remain on.
The canonical grid has 1280 faces and irregular discrete domain boundaries.

Two explicit fixture changes happen BEFORE evolution: applied face torques are
zero (stationary source geometry) and the deep thermal reservoir is 4e24 kg,
with its energy adjusted to retain 990 K. The independent manufactured mantle
field at transition is omega_z = 0.04*x rad/Myr. All original solid nodes are
explicitly declared basalt at 3000 kg/m³; liquid water is zero. No real chemical
composition is inferred. Source crust age is the elapsed 50 years, zero relief,
zero mantle-lithosphere anomaly; scalar deep temperature comes from the source.

`GeologicalContinuation.from_synthetic` calls the existing `genesis_mature`
explicit-state importer. Native `update_plate_dynamics` (default
`legacy_effective` parameters) then feeds native `advance_lithosphere`, including
its conservative residual transport and normal geological gap/overlap rules.
No custom contact solver, speed kick or relaxed transport threshold is added.
This is a kernel subset, NOT the entire v131 runner: thermal evolution,
topography, continental cycle, plumes and topology changes are inactive, and
those substates deliberately retain their frozen timestamps. The wrapper's
restart must be used; it is not a certified full-runner restart contract.

## Measured experiment

Four 10 Myr steps completed in 2.26 seconds (first experiment: 2.28 seconds),
well below the five-minute budget. Large geological steps test the plumbing,
not time convergence or physical continuity from 50-year Genesis.

- Maximum relative omega: 0.0131447732083 rad/Myr.
- Native winning parcel indices record 1666 moved cells summed over steps;
  cells can appear on more than one step. Corresponding volume-step sum:
  92,435,211.6758 km³, not a unique-material total.
- Created oceanic volume: 152,026,100.7179 km³; recycled: 80,000 km³.
  This large growth comes from the existing 7 km gap-fill rule versus the thin
  synthetic initial crust. It is explicitly funded from the interior reservoir.
- Maximum independent material residual: 2.23517417908e-8 km³; native budget PASS.
- First mature step holds material in native residual transport; the next three
  commit actual parcel moves. Saving after step 3 and taking step 4 on direct
  and loaded paths gives exactly equal native arrays, metadata, source and ledger.

`result.json` and `ledger.json` retain compact evidence. Full exclusive snapshots
are at `/workspace/genesis-geological-handoff-02` (~15.8 MB); the first experiment
is preserved at `/workspace/genesis-geological-handoff-01`.

## Accounting contract

| Quantity | Active / transformed | Frozen archive / unclosed part |
|---|---|---|
| Mass | 3.84e18 kg initial basalt → native material volume; generated/recycled counters debit/credit bounded 4e24 kg interior | 1e20 kg surface inventory inactive; archive copies never counted twice; common reference density is not species mass closure |
| Composition | Explicit synthetic basalt, native oceanic volume | No elemental differentiation or inferred real chemistry; scalar source damage reused as mature tidal weakening has a changed constitutive meaning |
| H | Deep temperature projected into frozen scalar thermal state | All source-node enthalpies retained exactly; material exchange enthalpy NOT closed |
| Momentum | Source omega converted rad/s → rad/Myr; native effective velocities evolve | Source inertia/momentum archived; per-step thin-shell linear/angular momentum proxies and their changes recorded; no SI force balance claimed |
| Energy | Kinematic kinetic-energy proxy measured each step | Retired elastic energy 1.54060545825e15 J and passive reserve 12345 J archived; native proxies do not provide SI work; changes remain unclosed |
| Maxwell memory | No active detailed constitutive consumer | Entire config/state frozen with checksum; zero conversion to heat |

Negative readiness gates reject a real `source_mode`, a single domain, unknown
composition, rotated/deformed geometry needing remapping, additional phase
material and edge strips. Failed material accounting or exhausted interior
cannot mutate the accepted checkpoint. Checksum verification precedes native
restart loading, and existing output directories cannot be overwritten.

This meets the synthetic geological handoff goal; it does NOT admit a real
Genesis handoff or prove equivalence to resolved mechanics. Next implementation
work must define real chemical crust/remapping and a thermal/material exchange
contract before allowing real-state transfer. Canonical state, defaults,
`feat/simplified-genesis` and the detailed branch remain unchanged.

## Reproduction and tests

From repository root, in the project environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  timeout 300 python -m analysis.genesis_geological_handoff.experiment /workspace/geological-handoff-new
python -m pytest -q -p no:cacheprovider tests/test_genesis_geological_handoff.py tests/test_genesis_mature.py tests/test_genesis_starter_ledger.py
```

The first regression pass had 67 passes and one erroneous exhaustion test:
step 1 correctly held the native transport and consumed no reservoir. The test
now advances to the first actual crust-creation step before exhausting the
reservoir. No transport criteria were changed to satisfy the test.

Final regression: **69 tests passed in 8.95 seconds** (new handoff, existing
explicit importer and independent material ledger). `git diff --check` passed.

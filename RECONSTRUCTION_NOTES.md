# v0.11 forward-development note

Everything described below under the v0.10 reconstruction remains provenance for
the reconstructed baseline.  **v0.11-material is not an attempted recovery of
lost historical code**: it is a new change made after reconstruction.

The new material layer decouples continental footprint/volume from discrete plate
ownership.  It was introduced after resolution tests showed that the reconstructed
one-cell continental representation could create 100-300 km collision columns,
which then caused artificial delamination and continent loss.  The v0.11
sub3/sub4/sub5 500-Myr convergence test removes that failure mode while preserving
intrinsic conservative transport.

Current validation: **76/76 tests** plus bitwise deterministic checkpoint/resume.

---

# Moon Tectonics — v0.10 reconstructed from surviving v0.9.7 source

## Provenance

The last surviving source archive was `moon_tectonics_v0.9.7.zip`.  The original
v0.9.8, v0.9.9 and v0.10.0 source archives were no longer available, but their
README/history and long-run assessment files survived.

This tree is therefore a **functional reconstruction**, not a byte-identical
recovery of the lost v0.10.0 source.

## Recovered with high confidence

The surviving reports specify these pieces in enough detail to reconstruct the
architecture directly:

- v0.9.7 passive-margin/rift and collision/weld calibration remains intact;
- v0.9.8 mantle memory is a fixed-grid field independent of mutable plate IDs;
- topology events must not overwrite mantle memory with current plate velocity;
- collision/transform resistance damps relative tectonic drive, not the common
  mantle-advection component;
- explicit-mantle runs do not quantise slow plates to exactly zero;
- v0.9.9 accumulates unresolved plate rotation as a residual quaternion;
- the canonical v0.9.9 sub-grid commit thresholds are restored verbatim;
- split/merge/absorb remap residual transport state instead of resetting it;
- continental-cycle `felsic_potential` follows the exact same source map as
  crustal material;
- v0.10 uses sparse one-to-one source/target assignment independently per plate;
- source-column thickness is multiplied by `A_source/A_target`, so pure rigid
  transport conserves `A*h` volume intrinsically;
- continent/continent overlap keeps full incoming column volume and locally
  redistributes grid-scale collision overflow rather than deleting it;
- the old global post-hoc continental-volume correction is not used.

## Reconstructed assumptions (not recovered historical constants)

The reports preserve the *existence and direction* of the following v0.9.8
feedbacks, but not their exact source coefficients.  Their values in
`configs/canonical_moon.yaml` are therefore explicitly marked reconstruction
assumptions:

- slow thermal evolution/smoothing of the fixed-grid mantle-flow field;
- thermal-lithosphere scaling of slab pull;
- thickness/buoyancy scaling of continent-continent collision resistance;
- GPE spreading torque from over-thickened continental crust;
- slow conservative gravitational collapse of thick continental crust.

Do **not** expect this reconstructed tree to reproduce the historical v0.10
4-Gyr seed numerically until those coefficients are re-calibrated.

## Validation completed during reconstruction

- Original v0.9.7 archive before modifications: **67/67 tests passed**.
- Reconstructed tree: **74/74 tests pass**.
- New tests cover one-to-one assignment, intrinsic `A*h` conservation,
  cumulative sub-grid movement, topology inheritance, exact felsic-memory
  advection, conservative gravitational collapse, and checkpoint round-trip of
  mantle + residual transport state.
- Canonical 0→40 Myr monolithic and 0→20→resume→40 Myr integrations produced
  identical NPZ checkpoint arrays and identical `meta.json`.
- A segmented canonical integration has been exercised through 60 Myr.
- Maximum pure-transport continental-volume error observed in that 60-Myr smoke
  run was `4.76837158203125e-07 km^3` (floating-point scale).

## Recommended next experiment

Do not tune plate-force coefficients from the old single v0.10 seed yet.  Run a
small coarse multi-seed ensemble first:

```bash
PYTHONPATH=. python run_ensemble_v100_reconstructed.py \
  --subdivisions 3 \
  --end-time 4000 \
  --dt 4 \
  --segment-myr 500
```

The ensemble driver records the two historical v0.10 WATCH items directly:

1. whether >50% superplates occur systematically;
2. whether mean continental thickness systematically falls below ~30 km.

Only after that should the reconstructed v0.9.8 force/collapse assumptions be
calibrated.

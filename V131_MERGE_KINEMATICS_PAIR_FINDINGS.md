# v0.31 paired merger-kinematics replay — 300 to 500 Myr

## Experiment

Both branches resume the same canonical subdivision-5 checkpoint at 300 Myr
and run to 500 Myr with 20,480 cells, `dt=4 Myr`, checkpoint spacing of 20 Myr
and otherwise identical configuration. The original paired experiment ended at
400 Myr; both branches were then continued in place from their own 400 Myr
checkpoints.

- `area_weighted`: the historical v0.31 control, in which the welded angular
  velocity is the surface-area-weighted vector mean;
- `inertia_tensor`: an opt-in diagnostic rule that treats each cell as a
  uniform thin-shell surface element, computes the two full inertia tensors,
  and solves `I_merged * omega_merged = L_a + L_b`.

The inertia case is intentionally not the new default. It accounts for plate
geometry, but not yet spatial differences in lithosphere thickness, density or
slab mass.

Local results: `results/diagnostics/v131_merge_kinematics_300_400_sub5`. The
historical directory name is retained even though it now includes checkpoints
through 500 Myr.

## Reproducibility and numerical acceptance

The control branch is bitwise identical to the original canonical checkpoints
at 320, 340, 360, 380, 400 and 500 Myr across every checkpoint array. This
verifies that adding the optional rule and continuing from 400 Myr did not
change historical v0.31 behaviour.

The inertia rule has a unit test that closes its geometry-weighted angular
momentum equation. The full repository suite passes `185/185`. Both branches
have zero elevation safety clips, converged flexure and conservative transport
error no larger than `9.54e-7 km3`. At 500 Myr the continental ledger errors
are `4.77e-7 km3` for the control and `1.43e-6 km3` for the inertia case.

## Low-motion sector result through 400 Myr

The investigated sector is 0–90°E, 60°S–0°.

| Time | Area-weighted mean | Inertia mean | Speed change | Area below 2 km/Myr: control → inertia |
|---:|---:|---:|---:|---:|
| 340 Myr | 2.30 km/Myr | 3.06 km/Myr | +32.9% | 34.6% → 19.8% |
| 380 Myr | 3.44 km/Myr | 3.90 km/Myr | +13.3% | 15.4% → 12.4% |
| 400 Myr | 4.53 km/Myr | 4.74 km/Myr | +4.8% | 12.6% → 11.4% |

At 340 Myr the control's nearest Euler pole is 40.26°E, 29.10°S and the
inertia branch's is 46.00°E, 36.81°S. At 380 Myr the positions are
46.40°E, 25.62°S and 48.44°E, 21.90°S. The pole therefore remains inside the
sector under both laws.

The quiet zone is consequently robust in sign and location: it is not created
solely by surface-area averaging. The historical rule does make it deeper and
broader immediately after the mergers.

## Continuation from 400 to 500 Myr

The sector does not stay permanently frozen. The two branches recover on
different schedules:

| Time | Control mean | Inertia mean | Control area below 2 km/Myr | Inertia area below 2 km/Myr |
|---:|---:|---:|---:|---:|
| 400 Myr | 4.53 km/Myr | 4.74 km/Myr | 12.6% | 11.4% |
| 420 Myr | 5.30 km/Myr | 4.13 km/Myr | 9.8% | 17.7% |
| 440 Myr | 5.94 km/Myr | 3.74 km/Myr | 7.3% | 25.1% |
| 460 Myr | 6.20 km/Myr | 4.63 km/Myr | 6.7% | 8.2% |
| 480 Myr | 5.96 km/Myr | 6.47 km/Myr | 8.3% | 0.0% |
| 500 Myr | 5.22 km/Myr | 7.92 km/Myr | 15.2% | 2.3% |

The control undergoes a partial thaw, peaking at 460 Myr, then slows again by
500 Myr. Its nearest Euler pole remains close to the sector: the pole distance
from the sector centre grows from 17.9 degrees at 400 Myr to 28.0 degrees at
460 Myr, then falls to 24.3 degrees at 500 Myr.

The inertia case first becomes quieter through 440 Myr, then recovers strongly.
Its nearest pole moves from 16.0 degrees away at 400 Myr to 77.7 degrees at
480 Myr and 89.1 degrees at 500 Myr. This geometric departure explains the
late acceleration without any new split or merger after 400 Myr.

Continental-material-weighted sector speed in the control is 3.66 km/Myr at
400 Myr, reaches 4.76 km/Myr at 460 Myr and is 3.18 km/Myr at 500 Myr. In the
inertia case it falls from 3.87 to 2.69 km/Myr at 440 Myr, then rises to
7.39 km/Myr at 500 Myr. The fractional continental footprint between 400 and
500 Myr has a minimum area-weighted rearrangement of 24.4% in the control and
26.1% in the inertia case. This is a distribution-change measure, not a
Lagrangian tracer distance, but it confirms that continental material is being
transported rather than numerically held in place.

The total fractional continental area is conserved to floating-point precision
over the continuation. Transport commits rise from 944 to 1082 in the control
and to 1109 in the inertia case; maximum hold age remains only 16 Myr versus the
configured 120 Myr forced-transport limit.

## Topological divergence

Both branches repeat the mergers at 312, 328 and 332 Myr. The inertia branch
then triggers a `disconnect_split` at 372 Myr, while the control remains at six
plates. The detached domain contains 536 cells and `8.98e6 km2` (2.58% of the
globe) at birth, then persists and grows to 562 cells and 2.75% at 400 Myr. The
branches keep six and seven plates respectively through 500 Myr, with no new
topology events after 400 Myr. It is therefore a persistent macroscopic
topology change, not a one-cell numerical crumb.

At 400 Myr:

| Metric | Area-weighted | Inertia tensor |
|---|---:|---:|
| Plate count | 6 | 7 |
| Sea level | -623.0 m | -546.4 m |
| Land fraction | 26.30% | 24.82% |
| Mean plate–mantle slip | 0.289°/Myr | 0.330°/Myr |
| Mean continental plate speed | 0.236°/Myr | 0.258°/Myr |

The branches differ in plate ownership for 12.95% of cells by 400 Myr and
their elevation fields have an RMS difference of 1.45 km. These are expected
nonlinear consequences once the seventh plate appears; they should not be
interpreted as a direct 1.45 km effect of the merger formula alone.

## Conclusion

The original low-motion sector has two layers of causation:

1. **Robust geometry:** the merged plate's Euler pole lies inside the sector in
   both branches, so a local speed minimum is unavoidable in rigid-plate
   kinematics.
2. **Rule sensitivity:** area weighting produces stronger immediate vector
   cancellation; inertia weighting raises the sector speed and later changes
   the topology.

The continuation confirms that the quiet sector is transient rather than a
permanent numerical lock. In the canonical control it only partly thaws and
begins slowing again; in the inertia branch it becomes fully mobile after the
Euler pole leaves the region.

Do not replace the canonical rule yet. The 372 Myr disconnect and the strong
late pole migration are scientifically important enough to test at subdivision
4 and 5. The next physical refinement should also compare uniform-area inertia
with a mass-weighted tensor using lithosphere thickness and density.

## Commands

Run the pair:

```text
python analysis/run_merge_kinematics_pair_v131.py \
  --resume results/gui_runs/v031_canonical_500myr_sub5/gui_checkpoint_000300_Myr \
  --output-root results/diagnostics/v131_merge_kinematics_300_400_sub5 \
  --end-time 400 --subdivisions 5
```

Continue both branches safely from their latest complete checkpoints:

```text
python analysis/continue_merge_kinematics_pair_v131.py \
  --pair-root results/diagnostics/v131_merge_kinematics_300_400_sub5 \
  --end-time 500
```

Build the paired diagnostic figure and CSV:

```text
python analysis/compare_merge_kinematics_pair_v131.py \
  --pair-root results/diagnostics/v131_merge_kinematics_300_400_sub5 \
  --initial-root results/gui_runs/v031_canonical_500myr_sub5 \
  --output results/diagnostics/v131_merge_kinematics_300_400_sub5/comparison \
  --map-time 500
```

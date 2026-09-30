# Transport, fracture, and slow-motion follow-up

Read-only production-code audit, 2026-09-29. New files and the 400→405 Myr
continuation are analysis artifacts. Physical parameters and production code
were not changed. All ages below are absolute unless identified as elapsed.

## Confirmed second delay: continuous motion versus raster material

`transport.build_transport_map` accumulates every Euler step in a quaternion.
It does not set plate velocity to zero between material remaps. Nevertheless,
surface material, crust birth, and consumed material remain unchanged until a
conservative parcel assignment is accepted.

At corrected elapsed 400 Myr (age 400.878121948), subdivision 4 has a median
centroid spacing of **208.233163 km**. The normal trigger requires both 18% of
cells choosing a different nearest target and p75 displacement ≥62.469949 km.
The nominal 120 Myr maximum hold is conditional: the forced trigger still
requires 4% choosing a different nearest target.

| Plate | Accumulated maximum displacement, km | Changed nearest target fraction | Hold, Myr |
|---|---:|---:|---:|
| 0 | 92.017233 | 0 | 400 |
| 1 | 100.295692 | 0.000749625 | 400 |
| 2 | 109.279705 | 0.020900322 | 400 |
| 3 | 116.823279 | 0.035746202 | 400 |

A frozen-motion extrapolation predicted plate 3 reaching the forced trigger
in another 1.68609 Myr. A real continuation to elapsed 405 Myr, retaining all
physics, produced its first commit at age **402.878121948**. This is a numerical
material-resolution delay, not a direct damping of angular velocity.

Even that first commit is partial: of plate 3's 1119 cells, **one** changes its
assigned target. The fitted represented rotation is 0.003127854° versus a trial
rotation of 1.273167029°; residual rotation remains 1.270327051°. Thus counting
commits alone substantially overstates resolved material transport. The
one-to-one assignment can still prefer identity for almost every interior cell
even after the nearest-neighbor trigger admits a commit.

That commit creates a real oceanic gap of 65471.703915 km² and consumes an
oceanic overlap of 65266.730696 km²; created and subducted volumes are both
65471.703915 km³. Slab memory still has **zero** zones and zero integrated area,
because its initiation remains gated by the inactive boundary classification.
The mean speed at age405.878 is 0.229594290 km/Myr, essentially unchanged.

The time integral of actual returned plate velocities on the unchanged
four-plate boundary grid, for inputs7.878→399.878 Myr, gives maximum opening
191.783878 km and maximum closure **209.221993 km**. This is a kinematic integral,
not proof that this much lithosphere physically subducted: discrete material
remaps have not represented it. If the previously proposed 0.9 closure-to-slab
length rule were used, the largest potential length would be188.299794 km and
development fraction0.104611, not an immediately mature slab.

## New material loses its local mechanical age

At age405.878 the newborn cell has chemical age3 Myr, while old cells have
age405 Myr. Both have exactly the same mantle-lithosphere thickness
**33.597094644 km**, density anomaly **66.008555227 kg/m³**, and crust thickness1km.
`young_shell.mechanical_transition` is absent. Before that transition,
`YoungWorldCoupling.mechanical_fields` replaces local fields with the global
Genesis column and a uniform temperature deficit on every call. It therefore
erases the zero mechanical thickness assigned to the newborn gap in
`advance_lithosphere`. Local ridge/flank thermal contrast cannot develop by
crust age during this phase. A change to boundary thresholds alone cannot
repair this mechanical-state closure.

## Fracture and geometrical cancellation

At elapsed5 Myr the two initial plates cover49.6528% and50.3472% of the sphere.
Each differs from its best centroid hemisphere by only1.1200% of total area.
The source mantle field is the gradient of a degree-two scalar potential.
Its global rigid torque vanishes; its torque density is even under antipodal
mapping, so the integral over either exact hemisphere also vanishes. A
synthetic non-axis-aligned hemisphere partition of the saved field gives
fitted omega of order10⁻¹⁹ rad/Myr and zero represented kinetic fraction.

Accordingly, the actual early two-plate best rigid RMS is0.002538 km/Myr for
a local mantle RMS1.452522 km/Myr. That is real geometric cancellation in the
prescribed model, not another error in least squares. At late four-plate
states, the best fit retains about78% of local RMS, so this extreme cancellation
does not explain the remaining late slowness. There is no basis for an arbitrary
repartition or rotational kick to suppress this valid cancellation.

Continued fracture does not simply stop executing. At elapsed400 Myr it reports
max damage0.97236, max yield ratio1.29803, damaged area9.6433%, but **zero fresh
eligible rupture cells**: surviving high-damage bands have already been used.
At elapsed50/100/200 Myr fresh eligible cells are also zero. Its stress proxy
is the prescribed potential-field tensor scaled by `traction * stress_length /
H * coupling`, not the evolved per-cell mantle–plate shear. Late scale is
2.31242MPa at H34.5958km, versus6.05892MPa at elapsed5Myr/H13.1856km. Changes in
actual slip cannot change this forcing. A physical shear/stress implementation
must distinguish net rigid torque from deforming membrane load; injecting the
norm of the entire residual as an arbitrary damage/speed source is unjustified.

## Proposed work, ordered by dependency

1. Add conservative continuous boundary opening/closure memory, with donor
   inventories and signed kinematics independent of display classification.
   Keep raster residuals and material ledgers; prevent double counting when
   a parcel commit catches up. Measure resolved displacement, not just commits.
2. Give post-partition newborn material its own thermal/mechanical history
   before whole-column exhaustion. Preserve the primordial column owner and
   match local birth/cooling energy in a declared local ledger. Do not reset old
   primordial material's age or change global heat production to create motion.
3. Use actual supported convergence and available negatively buoyant material
   to build finite slab geometry continuously; ridge drive needs an actual
   young-to-old mechanical contrast. Do not manufacture a mature slab or floor.
4. Feed resolved mantle–plate shear into a mechanically consistent membrane
   stress/Maxwell/damage path, after separating net torque handled by dynamics.
   Preserve fracture latches and donor transport; additional plate counts must
   follow real separating damage bands.
5. Validate subdivision4/5/6, small/large timesteps, long holds, one-cell commits,
   zero motion, closed reversals, common rigid rotation, conservative resume,
   new-crust thermal contrast, and no duplicate mass/energy/slab accounting.
   Keep the two-hemisphere potential cancellation as a diagnostic regression,
   not a behavior to tune away.

## Reproduction

`analysis/slow_motion_followup/transport_fracture_probe.py` reads saved corrected
5/50/100/200/400 Myr and the real110.878 Myr GUI checkpoint, verifies that the
four-plate ownership is unchanged in later archives, and writes the combined
`transport_fracture_probe.json`. This contains frozen predictions clearly
separated from the actual first-commit replay and saved405 Myr material fields.

The actual extra continuation was:

```powershell
& .venv/Scripts/python.exe analysis/plate_velocity_experiments.py segment `
  --source analysis/plate_velocity_validation/experiments/velocity_least_squares/elapsed_0400 `
  --output analysis/slow_motion_followup/transport_405 `
  --duration-myr 405 --mode velocity_least_squares --resume
```

All six continuation ledger/clock checks passed. No production changes or
commits were made by this follow-up.

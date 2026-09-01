# v0.31 low-motion sector diagnosis

## Question

The 500 Myr surface animation shows unusually little lateral motion in the
sector 0–90°E, 60°S–0°, especially after roughly 340–380 Myr. The saved
subdivision-5 checkpoints contain enough state to distinguish real plate
kinematics from a rendering or transport failure.

## Finding

The anomaly is real in the model and has a specific geometric cause: after
three late mergers, the sector is almost entirely carried by plate 0 and the
nearest Euler pole of that plate lies inside the sector. Rigid-plate velocity
is

`v = R * omega x r`,

so surface speed tends to zero at either Euler pole even when the plate remains
active elsewhere.

At 380 Myr:

- plate 0 owns 99.88% of the selected sector and 48.87% of the globe;
- its angular speed is only 0.08396 degrees/Myr;
- its nearest Euler pole is at 46.40°E, 25.62°S, only 4.55° from the sector
  centre;
- area-weighted sector speed is 3.44 km/Myr, versus 12.79 km/Myr globally;
- the sector's p10–p90 speed range is 1.61–4.97 km/Myr;
- 15.39% of the sector is below 2 km/Myr;
- mantle-flow speed in the same fixed sector averages 37.83 km/Myr.

The quiet interval begins abruptly after the mergers at 312, 328 and 332 Myr.
At 340 Myr plate 0 owns the whole sector, its angular speed is 0.057 degrees/Myr
and mean sector speed reaches its minimum of 2.30 km/Myr. The pole and angular
velocity subsequently migrate, so mean sector speed recovers to 5–6 km/Myr by
420–500 Myr, although the visually central area remains close to the pole.

## Why the merged plate slowed

The first important merger combines plates 0 and 2 at 312 Myr. At 300 Myr they
cover 19.2% and 15.1% of the globe and have substantially opposed horizontal
angular-velocity components. The topology rule constructs a welded plate using
the area-weighted vector mean of both angular velocities. Their opposing
components cancel, producing the slow plate-0 vector seen at 320 Myr. The two
additional mergers enlarge that plate to 50.9% of the globe and move its
near-zero pole into the selected sector.

This is deterministic behaviour of the current merger rule, not random GIF
jitter. It is nevertheless a modelling choice worth testing: surface-area
weighting is a stable effective rule, but it is not a full angular-momentum and
moment-of-inertia calculation for a welded spherical shell.

## Transport and rendering checks

There is no evidence of a stalled numerical transport queue:

- plate 0's checkpointed transport hold age alternates between 0 and 4 Myr
  from 340 to 500 Myr, far below the configured 120 Myr forced-commit limit;
- cumulative transport commits rise from 917 at 380 Myr to 1082 at 500 Myr;
- between those checkpoints, 17.3% of fixed sector cells change continental
  fraction, 97.6% change sediment volume and every cell changes elevation;
- mantle flow remains strong in the same region, demonstrating that the saved
  vector diagnostics themselves are not zeroed or missing.

The animation therefore reflects a slowly moving region near an Euler pole.
It is not a frozen frame, missing checkpoint or plotting defect.

## Reproducible diagnostics

Run:

```text
python analysis/diagnose_stagnant_sector_v131.py \
  --input results/gui_runs/v031_canonical_500myr_sub5 \
  --output results/gui_runs/v031_canonical_500myr_sub5/diagnostics/stagnant_sector \
  --map-time 380
```

The command writes `stagnant_sector_timeseries.csv` and
`stagnant_sector_diagnostics.png`. The figure maps the 380 Myr rigid-plate
speed and Euler pole, then compares sector plate speed, global plate speed,
local mantle speed, merger times, plate ownership and the sub-2 km/Myr area.

## Recommended follow-up

Do not alter the accepted canonical run. Use its 300 Myr checkpoint for a
paired diagnostic experiment:

1. replay the current area-weighted merger rule as the control;
2. test an inertia-aware welded-plate angular-momentum rule;
3. compare Euler-pole residence time, plate-mantle slip, boundary work and the
   0–90°E sector displacement through 400 Myr.

If both formulations retain the quiet sector, it is a robust consequence of
the collision geometry. If only the current rule produces it, the anomaly is a
merge-kinematics sensitivity rather than a general tectonic prediction.

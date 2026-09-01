# v0.31 GUI resolution benchmark

Local Windows benchmark performed through the same v0.31 runner and generated
GUI runtime configuration used by the desktop application. Every timed sample
starts a fresh deterministic world, advances with `dt = 4 Myr`, and writes a
full v0.31 checkpoint. Rendering and final GIF assembly were disabled so the
comparison measures numerical startup, integration, and checkpoint cost.

| Subdivision | Cells | Steps | Simulated time | Wall time |
|---:|---:|---:|---:|---:|
| 3 | 1,280 | 5 | 20 Myr | 11.60 s |
| 4 | 5,120 | 5 | 20 Myr | 13.70 s |
| 5 | 20,480 | 5 | 20 Myr | 23.45 s |
| 6 | 81,920 | 5 | 20 Myr | 90.70 s |

The sub-6 sample completed successfully but costs `3.87x` the canonical sub-5
sample for the same five steps. With a conservative 20-Myr GUI checkpoint
interval, repeating these segments to 500 Myr implies roughly 10 minutes for
sub-5 and 38 minutes for sub-6 before animation/finalization overhead. Longer
checkpoint segments reduce repeated process startup, while denser frame output
adds rendering cost.

## Decision

Use **subdivision 5** for the first full 500-Myr GUI run. It is the calibrated
20,480-cell canonical resolution, already covered by v0.31 smoke and
checkpoint/resume validation, and is fast enough for frequent safe checkpoints
and complete GIF output. Keep subdivision 6 available in the GUI for later
short, high-resolution convergence checks. A visually finer mesh is useful for
boundary geometry and narrow hotspot tracks, but it does not automatically
make uncalibrated physical predictions more accurate.

# Default young mechanics: timestep/grid sensitivity at +50 Myr

Source: the original 0.08-MPa Starter archive, SHA256
`b4584dd66208ff80aea18a178c38da2b96e353c723663251091886dee49e87f6`.
Original input files and production code were not changed by these experiments.
New continuation processes use default basal + thermal-contrast ridge dynamics;
slab force remains `disabled_pending_closure` while accepted material is tracked.

| Case | Cells | dt, Myr | Plates | Mean speed | Maximum speed | Commits | Accepted ocean volume, km3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference subdivision 4 | 5120 | 1 | 4 | 6.528218 | 9.460387 | 20 | 5,000,196.66 |
| Same archive, half step | 5120 | 0.5 | 4 | 6.243905 | 9.314436 | 52 | 5,858,435.45 |
| Same archive, refined to 5 | 20480 | 1 | 4 | 6.623383 | 9.478579 | 33 | 7,826,694.66 |
| Rebuilt Starter at 3 | 1280 | 1 | 3 | 3.307240 | 8.950217 | 0 | 0 |

Speed units: km/Myr = mm/year. Every case ended at 50.878121948 Myr absolute
age, 50 Myr after its first partition. Every reported thermal/material/clock
check passed. Maximum absolute thermal energy residual was 2.87e-14;
maximum absolute relative material-volume residual was 1.49e-16.

The half-step mean differs by **−4.36%**. Subdivision 5 differs by **+1.46%**;
the subdivision-3 rebuild differs by **−49.34%** and has a different plate count.
This is a sensitivity result, **not a demonstration of complete numerical
convergence**. Accepted material volume differs by 17.16% for half step and
56.53% for refinement to subdivision 5, despite closer mean speeds.

The temporal discrepancy appears with raster transport: before any commit, at
+30 Myr, reference mean is 4.316416924 and half-step mean is 4.316405522
(relative difference about 2.6e-6). Their continued fractures occur at the same
+6 and +7 Myr with the same selected bands. First commits occur at +38 and
+38.5 Myr; after transport/young-crust thermal feedback starts, the mean-speed
difference grows. This localizes the temporal sensitivity to the material and
geometric coupling, rather than the basal SI units or exponential relaxation.

Subdivision 5 commits earlier, at +28 Myr, and selects slightly different
daughter areas at the same fracture ages. Subdivision 3 was rebuilt from the
exact original thermal, tidal, shell and Starter parameters with ONLY
`shell.subdivisions: 4 -> 3`; it was not obtained by coarsening a partition.
It produces only one subsequent fracture at +7 Myr and no raster commits by
+50 Myr. Therefore it is also a check of the resolution sensitivity of fracture
geometry, not merely a velocity quadrature check on identical plates.

`comparison.json` stores exact metrics, code hashes at summary, and recursive
parameter differences. Equation-config differences are empty for every case;
source-parameter differences are empty for half step and only subdivision for
the spatial cases. Commands, source hashes, wall times and exit codes are in
the per-case `.metrics.json` files. Fresh source 3 provenance is in
`source_sub3/parameters.json`. Each run has a separate log and checkpoint tree.

Reproduction (new output folders are required; the runner refuses overwriting):

```powershell
& .\.venv\Scripts\python.exe analysis/young_mechanics_validation/convergence/run_case.py sub4_dt0p5
& .\.venv\Scripts\python.exe analysis/young_mechanics_validation/convergence/run_case.py sub5_dt1
& .\.venv\Scripts\python.exe analysis/young_mechanics_validation/convergence/run_case.py sub3_dt1
& .\.venv\Scripts\python.exe analysis/young_mechanics_validation/convergence/summarize.py
```

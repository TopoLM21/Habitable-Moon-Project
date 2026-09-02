# CPU performance branch — v0.31

## Safety boundary

- Stable numerical release: `v0.31-cpu-stable` / `3c80eff`.
- Stable desktop plus ETA: `main` / `79dde63`.
- Experimental branch: `perf/cpu-parallel`, in its own Git worktree.
- Original runner remains `run_long_evolution_v131.py`.
- Opt-in runner: `run_long_evolution_v131_cpu.py --cpu-workers 1|2|4|8`.
- The experimental GUI labels this workspace and defaults to CPU optimizations
  with one worker. Worker count applies to plate transport, not every operation.
- Optimized GUI outputs are constrained to this workspace's `results` folder.
  Selecting an external checkpoint never defaults to writing into its parent.
- Separate `.venv`; `requirements-performance-lock.txt` records the same package
  versions as the stable Windows environment. Do not alter the stable `.venv`.

The full user's 0–700 Myr run was backed up before any numerical changes. All
400 files (245,514,904 bytes) passed SHA-256 comparison; all 62 arrays in the
latest three checkpoints were readable. Run data is ignored by Git, so the
backup is separate from commits/tags.

## Changes under test

1. Bounded per-process geometry cache, keyed by mesh identity, not just cell count.
2. One spatial tree and one exact cell-spacing calculation per fixed mesh.
3. Reused pixel-to-cell mappings, independently keyed by raster resolution.
4. Batched fixed-grid mantle neighbour averaging, preserving neighbour order.
5. Persistent optional thread pool for independent per-plate transport plans.
   Workers read the previous state; the coordinator commits in plate-ID order.
   All plans complete before mutation, so a worker error cannot partially commit.
6. Single-worker spatial queries inside optimized mode to avoid repeatedly
   creating an all-core thread set and nesting it inside the plate pool.

No timestep, mesh resolution, material model, topology rules, history, output
frequency or rendering quality is reduced. Geometry caching assumes the fixed
Eulerian mesh used by v0.31. Topology mutates plate ownership, not mesh geometry.

The first candidate also limited BLAS threads. Exact integration validation
rejected that candidate: at 720 Myr it changed 4 arrays at floating-point-rounding
scale (maximum elevation difference 1.82e-12 m). Removing that override restored
byte-for-byte equality of all 62 arrays and exact metadata/history equality.
Keep native-library settings identical to the reference.

## Reproducing the comparison

Run `analysis/benchmark_cpu_modes.py` with `--config`, `--resume`, and a **new**
`--output` directory. It measures separate processes serially, alternates case
order over repetitions, keeps logs and timings, and rejects any difference in
array bytes or the complete checkpoint metadata/history. `--reference` can point
to an endpoint generated independently by the stable checkout.

Default comparison: baseline and optimized 1/2/4/8 workers, two repetitions,
20 Myr at dt=4, full animation frames at the endpoint, normal segment maps and
checkpoint writing. Final aggregate GIF generation is not part of these segment
timings, just as in the non-final GUI segments. Profile data and generated results
are stored under ignored `results/cpu_performance`, not in the stable run.

Wall-clock speed is the acceptance criterion, not the number of busy threads.
More workers may lose on small plates or GIL-bound/library work. Do not promote
the experimental default based only on a single microbenchmark or on cProfile
timings, and do not claim long-term equivalence from one short integration.

## Validation results

- Full suite: 209 tests passed. Re-running the existing suite under an active
  four-worker context passed 200 tests (the 9 context/lifecycle-specific tests
  are excluded there because they deliberately create their own contexts).
- The canonical 20,480-cell segment 700→720 Myr matched the independent stable
  checkpoint in all 62 array byte sequences and complete metadata/history for
  baseline and optimized 1/2/4/8 workers, two repetitions each.
- At 708 Myr, a new optimized checkpoint was resumed to 720 Myr both by the
  optimized runner and the **unmodified original stable checkout**. Both results
  matched the same continuous reference exactly.
- Real Qt QProcess smoke: fresh sub-3 run 0→4→8→12 Myr, four workers, safe pause
  after the first checkpoint, resume, completion and four-frame surface GIF.
- Offscreen GUI layout inspected; settings are not horizontally clipped.
- The two canonical cpu1 repetitions produced the same 40 PNG files by SHA-256.
- User-run intermediate GIFs: 36 frames each, 0..700 Myr, assembled without physics.
- Canonical topology-event checks also matched all 62 array byte sequences and
  full metadata/history against the user's pre-existing stable checkpoints:
  320→340 Myr (8→6 plates) and 620→640 Myr (6→8 plates), four workers.
  These two correctness jobs ran concurrently; their wall times are **not**
  used as performance measurements.

The initial timing series experienced a large wall-time shift late in the run
(cpu1 66.54→33.30 s; baseline 71.70→43.16 s) while model state and PNG outputs
remained identical. The cause of that host-performance change was not established.
Do not rank worker counts by averaging across those different conditions.
Raw initial timings are retained in `performance_reports`; follow-up measurements
pair each optimized case with an immediately preceding baseline.

### Follow-up paired full-segment timings

Same 700→720 Myr interval, dt=4, sub-5, full maps and normal checkpoint output.
Each pair was sequential; one pair per worker count, no simultaneous model jobs.

| Plate workers | Immediately preceding baseline, s | Optimized, s | Time saved |
| --- | ---: | ---: | ---: |
| 1 | 75.619 | 66.526 | 12.0% |
| 2 | 59.393 | 66.235 | -11.5% |
| 4 | 75.131 | 66.582 | 11.4% |
| 8 | 74.784 | 65.944 | 11.8% |

All eight endpoints matched the independent stable reference exactly. These
data show a modest benefit under several observed conditions, not a guaranteed
speedup. In particular the adverse two-worker pair and the earlier host-time
shift must not be hidden. One worker remains the conservative GUI default:
2/4/8 do not demonstrate a convincing full-segment advantage over it here.

The separate in-process transport microbenchmark (`transport_kernel.json`)
records one cold call and three warm calls per mode. Warm medians in seconds:
baseline 0.767; cpu1 0.654; cpu2 0.565; cpu4 0.637; cpu8 0.626. These are only
transport-plan timings, with fixed input copies and exact-result checks; they
do not represent the whole integrator or include final map rendering. Cold
initialization and timing variability remain visible in the raw measurements.

## Scope and remaining limits

Only transport-plan preparation is distributed across the persistent pool.
Inter-plate collision resolution, topology updates and final map rendering still
run in their original order. The caches and batched neighbour averaging also
benefit the one-worker mode. This is **not** an all-core rewrite of the integrator.

Do not infer 4.5-Gyr or cross-platform bitwise reproducibility solely from these
short Windows integrations. Longer ensembles and Linux timing are
separate validation work; package/native-library settings matter for rounding.

Potential next steps after this isolated preview: remove duplicate prototype
initialization, profile compiled per-cell kernels, and design bounded separate-
process rendering from immutable snapshots. GPU support is not installed or
implemented by this branch. None of these next steps changes the stable checkout.

## Intermediate GIF without recomputation

`analysis/assemble_saved_gifs.py --source-run <saved-run> --through-myr 700
--output <new-separate-folder>` assembles existing surface/plate PNG frames.
The cutoff excludes any frames beyond the selected checkpoint. It never runs
the integrator, edits the source run, or overwrites an existing output folder.
Missing old PNG frames cannot be reconstructed from a single final checkpoint.

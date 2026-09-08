# Experimental CUDA backend

2026-09-05. Branch `perf/gpu-compute`, based on CPU commit `eb75568`.
The stable `main` checkout and the `perf/cpu-parallel` worktree are unchanged.

## Current scope

`run_long_evolution_v131_gpu.py` wraps the optimized CPU runner in one
`GpuExecution` context. The context imports CuPy lazily, selects one CUDA device,
compiles FP64 kernels and keeps immutable mesh connectivity on the device.
Sediment routing executes on CUDA with byte-exact CPU ordering. The opt-in
`--gpu-surface` mode also keeps erosion, sediment reworking, routing, basin spill
and relief response on the device within one surface-physics block. Volcanic-arc
painting is available through the separate `--gpu-arcs` research flag; it
uses deterministic maxima but CUDA transcendental functions require tolerance-
based rather than byte-exact comparison. All other physics, topology,
checkpointing and rendering remain on the optimized CPU path.

This is a limited integration, not a claim that the whole model is
GPU-accelerated. Conservative advection and budget/diagnostic reductions remain
on CPU. The surface block packs six input fields into one upload and returns six
fields in one download; intermediate phases do not transfer whole fields. Routing
still downloads a stopping-test scalar per sweep and, exceptionally near its
cutoff, a full mobile field to preserve the CPU decision. State is **not** retained
on GPU across whole simulation time steps.

Install the optional environment separately from the stable CPU environment:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Example continuation:

```powershell
$env:CUPY_CACHE_DIR = 'D:\Moon Project\work\gpu-kernel-cache'
$env:CUDA_CACHE_PATH = 'D:\Moon Project\work\cuda-driver-cache'
.\.venv\Scripts\python.exe run_long_evolution_v131_gpu.py `
  --gpu-device 0 --gpu-surface --cpu-workers 1 --render-workers 4 --cell-kernels `
  --config configs\canonical_moon.yaml --resume <checkpoint> `
  --end-time 720 --dt 4 --output <new-output> --checkpoint <new-checkpoint>
```

Omit `--gpu-surface` to use only the initial CUDA routing implementation.
Add `--gpu-arcs` only for the tolerance-based research mode. It is deliberately
not the default because its whole-segment gain on the current sub-5 model has
not been established and it no longer promises byte-identical checkpoints.

The GPU runner fails clearly if CuPy, the CUDA runtime or the requested device is
unavailable; it never silently labels a CPU calculation as GPU work. Use
`run_long_evolution_v131_cpu.py` as the fallback runner.

## Native CPU optimization options (2026-09-08)

The two validated CPU candidates now have ordinary integration in this branch.
Add `--assignment-columns --boundary-forces` to the CPU or GPU runner to enable
them. Both default off; `--no-assignment-columns` and `--no-boundary-forces`
disable them independently. GUI defaults remain unchanged. These optimizations
do not require CUDA themselves and do not change checkpoint formats or physics
parameters. Native execution uses no research AST/function replacement.

`render_timings.json` records flags, actual calls and inclusive timings under
`numerical_execution.assignment_columns` and `.boundary_forces`. For the full
instructions and post-integration validation see
[NATIVE_CPU_OPTIMIZATIONS.md](NATIVE_CPU_OPTIMIZATIONS.md). The stage4/stage6
sections below describe historical research runs, not current availability.

## Determinism and ownership

- Inputs and results remain NumPy-owned at the current integration boundary.
- Neighbor and incoming-edge tables are uploaded once per mesh identity and the
  cache retains at most two meshes.
- Source contributions are gathered in ascending source order. No unordered
  floating-point atomics are used.
- Kernels use FP64 and compile with fused multiply-add contraction disabled.
- Basin spill preserves the interleaving of each source's own subtraction and
  contributions from its neighbours, not just the order of positive additions.
- One reusable surface workspace owns scratch buffers. Every call refreshes
  evolving values, including geometry-dependent areas; a different mesh identity
  replaces the workspace. Results are detached NumPy arrays, and state updates
  are committed only after the numerical block succeeds.
- The near-zero early-exit threshold is rechecked with the original NumPy
  reduction order before it can change control flow.
- GPU execution takes precedence over the optional CPU sediment kernel only while
  its explicit context is active. Importing normal model modules does not import
  CuPy or initialize CUDA.
- Arc painting assigns one CUDA thread to each target cell and takes the maximum
  over candidate fronts without atomics. Its `acos` and `exp` results can differ
  from NumPy at approximately machine precision.
- `render_timings.json` records the device, CUDA/CuPy versions, calls, cell count,
  elapsed routing time and transfer bytes. Its `surface_pipeline` report records
  block calls, buffer size, packed transfers and transfer-inclusive block time.

## Validation on this host

Host GPU: NVIDIA GeForce RTX 4080, 16 GiB, compute capability 8.9; CuPy 14.2.0.

- Six edge cases and a regular random field matched the original CPU loop byte
  for byte through the production `GpuExecution` dispatch.
- A real 700→704 Myr continuation produced a checkpoint exactly matching the
  optimized CPU reference: all 62 arrays and full checkpoint metadata matched.
- The CPU-only environment passed 278 tests; three CUDA-dependent tests were
  skipped there. The CUDA-enabled environment passed all four GPU tests.
- On a warmed 20,480-cell routing benchmark, the production path measured a
  0.627 ms median including field upload/result download versus 9.610 ms for the
  batched CPU kernel, or 15.3× for this isolated operation. The first cold call
  was about 68 ms, so short runs must not be judged by cold timing.

The second candidate study is recorded in
`performance_reports/gpu_backend_stage2.json`:

- The CUDA flexure CG solution differed by only `3.64e-12 m`, but its 36.88 ms
  median was slower than the 15.91 ms SciPy CPU solve. It is not integrated.
- At 700 Myr, 2,056 real volcanic fronts took 61.98 ms on the full CPU path.
  CPU task preparation plus CUDA painting was estimated at 28.33 ms, about 2.19×
  faster for that function. CUDA painting itself took about 0.737 ms.
- A real 700→720 Myr integration kept all integer/topology arrays exact. Floating
  state differences remained below `8.15e-10` in their stored units; the maximum
  elevation difference was `1.71e-10 m`. One derived hydrosphere ledger diagnostic
  differed by `2.38e-7 km³` through cancellation of much larger totals.
- The previous 0.3-second/~1% whole-segment improvement estimate is withdrawn:
  complete process wall times were not recorded. Tool wait durations are not a
  reliable measure of elapsed subprocess time. Arc CUDA remains opt-in; the
  isolated estimates above do not establish a full-simulation speedup.

Generated validation outputs are under ignored `results/gpu_integration_*`
directories and are not part of the source branch.

## Resident surface block (stage 3)

The exact surface pipeline is selected by `--gpu-surface`, independently of
the approximate `--gpu-arcs` flag. Stage-3 comparisons leave arc painting on CPU.
The reference runner is in the separate, unchanged `cpu-parallel` worktree, so
the CPU refactor in this branch cannot make both sides share an unnoticed change.

On the canonical 20,480-cell mesh, all 62 arrays (dtype, shape and bytes) and
complete checkpoint metadata/history matched for these continuations:

- 700→720 Myr, three independent pairs; all 40 generated PNG files also matched
  byte for byte in the first pair.
- GPU 700→708 followed by CPU 708→720, compared with the uninterrupted CPU run.
- 320→340, including merges at 328 and 332 Myr.
- 620→640, including disconnected-component splits at 628 and 640 Myr.
- 700→900 (50 steps), three pairs including a split at 844 Myr. The third
  pair's CPU endpoint also matched the first pair's CPU endpoint.

Input configuration/checkpoint hashes remained unchanged. The production
surface report confirmed five device block calls and one 3,276,800-byte scratch
allocation per 20-Myr run (immutable mesh tables add 491,520 bytes).

An isolated public `advance_sediments` probe uses checkpoint-700 fields, identity
advection, uniform erosivity and 15 warm repetitions. It includes host preparation,
diagnostics and all required transfers, but excludes loading, input copies and
CUDA context entry. All outputs matched byte for byte:

| Backend for the surface step | Median | Relative to CPU |
| --- | ---: | ---: |
| Optimized CPU | 17.479 ms | 1.00× |
| Only routing on GPU | 9.973 ms | 1.75× |
| Resident five-phase GPU block | 1.785 ms | 9.79× |

These are **surface-block**, not whole-model, speedups. For the full 700→720
subprocess including periodic maps, median CPU time was 11.727 s and GPU time
12.097 s (three samples each). This short workflow did not demonstrate an overall
speedup. The first CPU sample took 13.828 s, versus 11.699/11.727 s thereafter,
illustrating why an individual first pair is not sufficient evidence. Drivers
reported background GPU utilization even before child execution; no strict machine
isolation or statistical significance is claimed. Raw timings and telemetry are
preserved under `results/gpu_surface/stage3_700_720/summary.json`.

The longer 700→900 workflow omits periodic maps but still renders final reports.
Its CPU samples were 49.310, 44.987 and 45.857 s; GPU samples were 44.354, 44.254
and 44.423 s. Combined medians are **45.857→44.354 s**, an observed **3.28%** time
reduction. The third pair reused the first study's Matplotlib cache. All samples,
including the slower first CPU run, remain in the report. This modest local
result is not a statistically isolated estimate or a forecast for a full 4-Gyr run.

Final checks: **327 passed, 15 CUDA skips** in the CPU-only environment;
**65 passed** for the GPU-specific, lifecycle-contract and validation-harness
suites in the CUDA environment. They include a byte-exact 81,920-cell synthetic
public step, non-finite input rejection without state mutation, invalid device
buffers, workspace replacement and cleanup failures. Dense-mesh correctness is
not a dense-mesh speed measurement. The compact tracked report is
`performance_reports/gpu_backend_stage3.json`.

Reproduce correctness and full process measurements in an optional GPU environment:

```powershell
python analysis\validate_gpu_surface.py `
  --cpu-python <cpu-python.exe> --gpu-python <gpu-python.exe> `
  --cpu-project-root <independent-cpu-worktree> `
  --config configs\canonical_moon.yaml --resume <checkpoint-700> `
  --end-time 720 --dt 4 --repeat 3 --frames --midpoint 708 `
  --output results\gpu_surface\new_validation

python analysis\benchmark_gpu_surface.py `
  --config configs\canonical_moon.yaml --checkpoint <checkpoint-700> `
  --repeat 15 --output results\gpu_surface\new_probe.json
```

Both tools reject existing output destinations. Full-process validation alternates
CPU/GPU order, verifies input hashes and checks that the surface backend actually
executed; a failed comparison stops the study without a speedup result. The
isolated probe keeps contiguous backend batches to retain contexts and buffers,
so its per-context cold figures are not independent cold-process startup costs.

## Dynamics and topography study (stage 4)

A fresh exact-validated 700→900 Myr profile found boundary forces at 14.09%
of physics time, versus 1.50% for tectonic topography forcing and 0.23% for
the continental GPE neighbor loop. Isolated CUDA candidates for the latter
two remain research-only because their whole-model opportunity is small.

The strongest new candidate caches exact edge geometry and batches boundary
forces on **CPU**. Transferring only its ordered sums to CUDA was slower than
CPU reduction. In three alternating full-process pairs, adding the CPU candidate
to the same `--gpu-surface` mode reduced median time from **45.266 to 40.714 s**
(observed **10.06%**, not a new GPU speedup or a forecast for every workload).
All six endpoint checkpoints, full metadata/history and 32 PNG per run matched
the independent CPU reference. Additional merge/split continuations also matched.

This candidate is available only through `analysis/run_boundary_candidate.py`;
the ordinary dynamics implementation, GUI and defaults are unchanged. The
research-only AST wrapper is not the intended production integration. Current
validation: **390 passed, 19 CUDA skips** in the CPU environment and **132 passed**
for GPU-specific and new harness suites in the CUDA environment. See the
[complete study](GPU_DYNAMICS_TOPOGRAPHY_STUDY.md) for raw timings, exactness
scope, the optional environment's full-suite limitation and next steps; compact
results are in `performance_reports/gpu_backend_stage4.json`.

## Combined CPU candidates (stage 6, research only)

The assignment-column compaction and prepared boundary-force candidates were
measured together on 2026-09-08. Three fresh alternating full-process pairs
reduced median 700→900 Myr time from **44.910 to 34.215 s (23.81% less time)**.
The reference mode already uses the exact GPU surface pipeline; these two new
improvements run on CPU. This is neither a comparison against main/CPU Parallel
nor a sum of earlier isolated percentages. All six endpoints, complete histories
and 32 PNG per run were exact; combined merge/split continuations were also exact.

Use `analysis/run_assignment_candidate.py --with-boundary` only as an isolated
research runner, or `analysis/validate_assignment_candidate.py --with-boundary`
for guarded paired trials. Ordinary runner/GUI defaults remain unchanged.
Tests: 439 passed / 19 CUDA skips in the full CPU suite, and 161 passed in the
overlapping GPU-specific/research suites. The first baseline was slower and
background GPU activity was present; this is a local observation, not a forecast
for subdivision 7/8. See [the combined study](COMBINED_OPTIMIZATION_STUDY.md) and
`performance_reports/gpu_backend_stage6.json` for exactness scope and raw sources.

## Architectural limits and next decisions

The surface mode implements the first coherent group of resident phases, with
explicit CPU boundaries on either side. It does not eliminate transfers between
time steps. CPU remains responsible for GUI/I/O, topology changes, flexure and
the SciPy sparse bipartite assignment. Keeping fields resident across an entire
step would require migrating more consumers and defining synchronization at
frame, checkpoint and topology boundaries. Residency alone does not guarantee a
speedup: the earlier reports do not establish transfer time as the dominant cost.
Each newly migrated phase must pass an endpoint correctness audit before timing
comparisons are used for decisions. GPU mode is CLI-only and remains opt-in.

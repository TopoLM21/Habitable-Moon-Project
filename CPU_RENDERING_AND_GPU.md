# CPU rendering, batched sediments, and a bounded GPU prototype

2026-09-02. All changes stay in `perf/cpu-parallel`; stable `main` is untouched.
The scientific checkpoint version remains v0.31. No physics parameter, timestep,
mesh resolution, image resolution, frame frequency or output count is reduced.

This report records the stage at `203419d`. The subsequent extension to
6/8/12 render processes and selectable lower CPU priority is documented in
[CPU_RENDER_SCALING.md](CPU_RENDER_SCALING.md); the historical timings below
remain unchanged.

## Ready to use in the experimental GUI

Run `launch_gui.bat` in this worktree. Defaults now are:

- Optimized CPU mode, **1 plate-transport worker**.
- **4 separate render processes** (`Процессов для карт`). Select 1 for serial rendering.
- **Batched sediment routing enabled** (`Пакетный перенос осадков`). It can be disabled independently.
- The original CPU mode disables both new optimizations and remains available.

The existing copied run in `results/gui_runs/stable_copy_0_700` can be resumed
from `gui_checkpoint_000700_Myr`. The original run and its separate backup are
not continuation targets for this experimental workspace.

CLI equivalent:

```powershell
.\.venv\Scripts\python.exe run_long_evolution_v131_cpu.py --cpu-workers 1 --render-workers 4 --cell-kernels --config <config> --resume <checkpoint> --end-time <Myr> --dt 4 --frame-interval 20 --output <new-output> --checkpoint <new-checkpoint>
```

CLI defaults remain conservative: serial rendering, cell kernels disabled unless
explicitly requested. GPU is **not** a GUI or full-integrator backend yet.

## Whole-segment timings

Same canonical 700→720 Myr segment, 20,480 cells, dt=4, full frames and normal
segment maps/checkpoint. No final aggregate GIF in this intermediate-segment test.
Each comparison used separate, sequential processes; case order was reversed
on the second repetition. Every endpoint and PNG matched the stable reference.

| Rendering | Normal Windows scheduling, seconds | 8 performance cores only, seconds |
| --- | --- | --- |
| Serial | 26.525 / 27.008 | 26.712 / 26.640 |
| 2 processes | 19.700 / 20.023 | 20.113 / 19.860 |
| 4 processes | 17.055 / 16.865 | 17.339 / 17.020 |

Rendering alone reduced elapsed time by about **36%** in these comparisons.
The older 39–67-second variability is real, but it is not the baseline for these
speedup claims. Its complete cause remains unproven: the normal-scheduling
follow-up was also fast. No global power plan or application affinity was changed.
Windows identified 8 higher-performance and 16 efficiency-class cores; only the
controlled benchmark used mask `0xff`. Workers confirmed inheriting that mask.

With 4 render processes, a separate interleaved sediment comparison gave:

| Batched sediment kernel | Whole-segment seconds |
| --- | --- |
| Off | 17.586 / 17.185 |
| On | 15.821 / 15.666 |

That is another roughly **9% reduction** relative to parallel rendering alone.
The observed combined result is about 16 seconds versus 27 seconds for the
previous serial-render optimized-CPU configuration; this is not a timing
guarantee for all ages, hosts, background loads, or a full 4.5-Gyr run.

## Rendering contract and safety

- Each job receives a synchronously serialized snapshot, before the simulation
  can mutate it. The pool feeder never serializes live arrays later.
- Independent Matplotlib work uses spawned processes, not competing GUI threads.
- The queue is bounded to twice the worker count and 128 MiB of queued serialized
  data; individual oversized snapshots are rejected rather than growing silently.
- Repeated writes by one renderer to the same target preserve submission order.
- Worker PNGs are written to `.part` files and atomically replaced on completion.
  The GUI does not discover `.part` files. An abrupt kill can leave an ignored
  temporary file, but not replace the last complete PNG with partial data.
- Barriers run **before frame-directory globbing**, not just before the GIF writer.
- Context exit drains all rendering before the process reports success. Safe pause
  therefore waits for the current segment's frames as well as its checkpoint.
- Windows Job Objects kill render workers when their owner is forcibly terminated.
  Linux uses parent-death signalling; that path is implemented but not runtime-tested
  on Ubuntu here. Normal exit joins workers and restores patched runner aliases.
- The legacy volcanic-arc image function also returns diagnostics, so it remains
  synchronous. Value-returning renderers are not silently converted to async calls.
- A stale delayed GUI kill callback cannot target a later process with another PID.
- Four rendering workers peaked at roughly 300–313 MiB each in the controlled
  run, plus the simulation and queued snapshots. The queue limit is not a total-RAM limit.

The first full-GUI candidate rejected the value-returning volcanic-arc renderer;
that failure was fixed and the complete scenario rerun. Initial rejected output is
retained under ignored results, not promoted as a completed run.

## CPU cell kernel

Only sediment routing is batched in this stage. Per-cell drop, deposition and
flux calculations operate on arrays; the sequential sweep dependency remains.
The final scatter additions retain ascending-source / original-neighbour order.
All computations stay in float64. No fast-math, unordered atomics or physical
approximation is introduced. This is vectorization, not an all-core physics rewrite.

## GPU feasibility result — not a full model speedup

A separate environment was created at `D:\Moon Project\work\gpu-probe-env`.
It has CuPy 14.2.0 and a minimal set of official CUDA component wheels. The
working CPU environments, NVIDIA driver and global CUDA installation were not
changed. A full optional toolkit download was stopped; unneeded FFT/random/solver
libraries were not installed. Dependencies are in `requirements-gpu-probe.txt`.

The prototype runs two FP64 kernels: emit independent neighbour fluxes, then
gather them in the CPU's original source order. It avoids unordered floating-point
atomics and disables fused multiply-add contraction. Only closed triangular meshes
with reciprocal, unique three-neighbour connectivity are supported by this probe.

RTX 4080, driver 616.56; five warm repetitions, CPU restricted to performance
cores for comparison. Inputs are fixed synthetic topography/material fields, not
an integrated planetary trajectory. GPU timings below include field uploads and
result download, but reuse prepared mesh connectivity and the CUDA context.

| Cells | Batched CPU median | GPU median with field transfers |
| --- | ---: | ---: |
| 20,480 | 10.44 ms | 0.637 ms |
| 81,920 | 40.08 ms | 0.763 ms |
| 327,680 | 171.34 ms | 9.475 ms |

All three outputs and six small edge cases were byte-identical to the original
CPU loop. Warm GPU timings fluctuate: at 327,680 cells, transfer-inclusive samples
ranged from 5.7 to 16.0 ms. The machine was not an exclusive GPU benchmark host.
First-call/mesh preparation and resident-only samples are retained in the raw report.
Disk kernel caching was enabled; do not call these pristine-install cold timings.

This demonstrates that the chosen numerical operation can run efficiently on GPU
despite the earlier weak Python-thread result. It does **not** demonstrate the same
speedup for the whole simulation. After CPU batching, sediment routing is already
small; moving just this kernel into production GPU execution would save little
total time. A useful full GPU backend needs a larger group of cell operations and
careful CPU↔GPU state ownership, checkpointing and topology validation.

Probe command (use a new output directory):

```powershell
$env:CUPY_CACHE_DIR = 'D:\Moon Project\work\gpu-kernel-cache'
$env:CUDA_CACHE_PATH = 'D:\Moon Project\work\cuda-driver-cache'
& 'D:\Moon Project\work\gpu-probe-env\Scripts\python.exe' analysis\gpu_sediment_probe.py --output results\cpu_performance\gpu_probe_new --subdivisions 5 6 7 --repeat 5
```

## Validation and evidence

- Full CPU suite: **236 tests passed**, including snapshot isolation, real separate
  workers, queue limits, repeated-target ordering, worker failure, atomic output,
  force-killed owner, alias restoration and GUI mode/provenance checks.
- Rendering benchmark: 12 endpoints; combined-cell benchmark: 4 more endpoints;
  all 62 array byte sequences, full metadata/history and all 40 PNGs match.
- Combined mode exactly matches pre-existing checkpoints through 320→340 Myr
  mergers and 620→640 Myr splits.
- A combined-mode 708-Myr checkpoint resumes to the exact same 720-Myr result
  both in the optimized runner and in the unmodified original stable checkout.
- Real Qt controller: 0→4→8→12 Myr, safe pause/resume and finalization. All
  **127 PNGs and 10 GIFs** match the serial GUI reference byte-for-byte, including
  frame counts (the legacy plume-rift GIF has 3 frames; the others have 4).
  This full GIF comparison passed twice: 4 render processes with cell batching
  disabled, and a final repeat with both new GUI-default optimizations enabled.
- Real GUI immediate-stop check in combined mode retained the completed 4-Myr
  checkpoint unchanged; progress was not falsely advanced to the interrupted segment.
- The GPU prototype has not produced or validated full-model GPU checkpoints.
- Final GUI layout was inspected with the new default controls. All 400 files
  of the original run again matched both the backup and the continuation copy
  by SHA-256; stable `main` remained clean and unchanged at `79dde63`.

Raw reports are retained under `results/cpu_performance/` and selected small
JSON evidence is mirrored under `performance_reports/`. Key directories:
`render_benchmark_normal`, `render_benchmark_pcores`, `render_cells_benchmark`,
`render_cells_topology`, `render_cells_resume`, `gpu_sediment_scaling`,
`render_gui_smoke_fixed`, `render_gui_serial_reference`, `render_cells_gui_final`,
`render_cells_stop_smoke`.

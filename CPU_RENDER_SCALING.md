# Render-process scaling and lower CPU priority

Follow-up to `203419d`, in `perf/cpu-parallel` only. The numerical model,
checkpoint format, image resolution and output frequency are unchanged.

## Controls

- `Процессов для карт`: 1, 2, 4, **6, 8, 12**. These are render workers, in
  addition to the numerical coordinator and the desktop GUI. This number is
  not a process affinity mask or a limit on all native-library threads.
- `Уступать CPU другим приложениям`: enabled by default in optimized CPU mode.
  Applies lower CPU scheduling priority to the numerical coordinator **before
  importing scientific libraries**, and explicitly to every render worker before
  its imports. The GUI itself keeps its original priority.
- Turning that checkbox off preserves the normal launch environment; it never
  requests high/realtime priority. Both new options belong to the experimental
  CPU runner; the reference mode stays unchanged.
- Execution controls are locked during calculation and safe pause. Stop/start
  from a completed checkpoint to change them; Resume keeps the captured RunSpec.
  Stop is also available while safely paused and does not interrupt any process.
- Run provenance (`gui_run.json`) stores worker count and requested priority.
  The latest `render_timings.json` stores the coordinator's and each worker's
  actual OS priority. Setting priority errors out visibly if it cannot be applied.

Windows uses `BELOW_NORMAL_PRIORITY_CLASS`. Unix uses an idempotent nice value
of at least 5, retaining an already lower inherited priority. There are no changes
to global Windows settings, affinity, BLAS thread settings, I/O priority, memory
priority or the working Python environments. The Linux code is unit-tested with
mock OS calls, but has not been runtime-tested on Ubuntu here.

Lower CPU priority helps normal-priority interactive applications compete for
CPU time. It is **not** a CPU usage cap or a promise of a freeze-free desktop:
disk pressure, RAM exhaustion and unrelated applications can still cause stalls.
With little competing work, the simulation can still use available CPU capacity;
with competing work, its ETA can increase. See the
[Windows priority documentation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass)
and [Python Unix priority API](https://docs.python.org/3.12/library/os.html#os.setpriority).

The existing queue still limits serialized pending snapshots to 128 MiB and at
most twice the selected worker count. This is not the total RAM used by all
processes. Atomic PNG writes, pre-GIF barriers and owner-bound worker lifetimes
are retained. No automatic 24-worker or all-core mode is introduced.

## Reproduce the timing comparison

Use a new output directory and do not run other model jobs concurrently:

```powershell
.\.venv\Scripts\python.exe analysis\benchmark_render_modes.py --output results\cpu_performance\scaling_new --modes 4 6 8 12 --cell-kernels --priorities normal below_normal --repeat 2
```

Each case resumes the same 700-Myr checkpoint to 720 Myr, sub-5 (20,480 cells),
dt=4, full frame output and checkpoint writing. Plate workers stay at 1 and
batched sediments stay enabled. Case order is reversed in the second repetition;
the two priority modes are interleaved within each worker count. All 62 arrays,
complete metadata/history and all 40 PNGs are checked against an independent
stable reference. This times normal intermediate GUI segments, not the final
assembly of all accumulated GIF frames.

## Measured results (2026-09-02)

Normal Windows scheduling, no performance-core pinning. Seconds for each full
segment; both repetitions are shown, including the slower cases:

| Render processes | Normal priority | Below-normal priority |
| --- | --- | --- |
| 4 | 15.916 / 15.811 | 15.922 / 15.911 |
| 6 | 14.830 / 22.824 | 14.948 / 25.559 |
| 8 | 14.687 / 14.837 | 33.562 / 15.145 |
| 12 | 29.544 / 29.180 | 22.299 / 23.286 |

All **16 endpoints** matched the stable reference in all 62 arrays and complete
metadata/history; all 40 PNGs matched byte-for-byte. Each pool actually used the
requested number of distinct workers. For every below-normal case, the coordinator
and every worker reported Windows priority class `0x4000`; affinity stayed
`0xffffff`. The serialized queue peaks were 44.8–64.0 MiB, below the 128-MiB cap.

Eight workers at normal priority saved about 7% versus four in this sample;
twelve were slower in both priority modes. Six and below-normal eight had large
timing variability. These measurements do not establish its cause, and the slow
samples must not be discarded or attributed solely to the priority setting.
The computer was not an isolated benchmark host. This is not evidence that
lower priority itself makes simulation faster.

**Default: retain 4 render workers, with lower CPU priority enabled.** Four were
consistent in this series and use fewer worker processes. Larger counts remain
manual choices, not recommendations to occupy all available cores. The benefit
of lower priority for desktop responsiveness follows OS scheduling semantics;
interactive foreground latency under controlled contention was not benchmarked.
Nothing here guarantees the same speed or ETA throughout 4.5 billion years.

Raw evidence: `results/cpu_performance/render_scaling_priority/measurements.json`,
mirrored in `performance_reports/render_scaling_priority.json`.

## Validation

- **247 tests passed** on Windows, including expanded worker choices, command
  forwarding/provenance, real OS priority checks, idempotence, parent-process
  priority preservation, GUI control locking, and stopping a safe pause.
- Actual Qt QProcess run with **12 render workers + below-normal priority**:
  0→4→8→12 Myr, safe pause/resume, finalization. All 62 arrays and full history
  matched the serial reference; all **127 PNGs and 10 GIFs** were byte-identical.
- A separate actual-controller immediate-stop test with 12 lower-priority
  workers retained the last completed 4-Myr checkpoint unchanged. Its reported
  progress did not advance to the interrupted segment. Existing worker
  owner-death and atomic-output tests passed in the full suite.
- Offscreen GUI layout was visually inspected; the new checkbox is fully visible.
- Stable `main`, numerical physics and user-run output folders were not edited.

Evidence summaries: `performance_reports/render_scaling_validation.json` and
`performance_reports/render12_low_stop.json`. Complete local test outputs remain
under `results/cpu_performance/render12_low_gui` and `render12_low_stop`.

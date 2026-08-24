# v0.23 Conservative Sediments — Findings

## Scope

v0.23 replaces the old non-conservative topographic erosion sink with an explicit sediment/material cycle while keeping climate outside the tectonic program. The default run uses uniform erosivity; an external climate model can later provide a spatial `erosivity_field` without changing the tectonic architecture.

The production version is `0.23-conservative-sediments`, runner `run_long_evolution_v123.py`.

## Material reservoirs

The tracked continental-material ledger now distinguishes:

1. **continental bedrock** — `continental_volume_km3` on the surface lithosphere;
2. **surface sediment** — `sediment_volume_km3`, advected with the winning lithospheric parcel;
3. **deep recycled sediment** — sediment carried away when its surface parcel is lost to subduction;
4. **rift recycled continental material** — continental crust removed by rift thinning or final breakup and transferred to a deep recycling reservoir;
5. **continental-cycle recycled material** — the pre-existing cycle reservoir for explicit cycle recycling.

The global invariant is

`initial continental + generated continental = bedrock + surface sediment + deep sediment + rift recycled + cycle recycled`.

The runner evaluates this invariant every tectonic step and aborts if its error exceeds the configured numerical tolerance.

## Erosion and routing

- The legacy `topography.erosion_diffusion_per_myr` path is disabled in the v0.23 production runner, preventing double erosion.
- Bedrock erosion is restricted to tracked continental material and removes a real rock volume rather than only lowering a display elevation.
- Sediment is routed deterministically downhill over the spherical mesh using repeated conservative sweeps.
- Deposition uses the **actual previous solved sea level**, not the historical zero datum, so late-time dry lowlands are not incorrectly treated as submarine basins.
- Sediment already on the surface can be reworked independently of fresh bedrock erosion.
- Very thick deposits are handled by a conservative soft burial-spill closure rather than hard clipping. Production parameters use a 10 km soft limit with 35% spill per step.
- Sediment contributes both geometric thickness and lithospheric loading to material-aware topography/flexure.

The tuned production conversion `bedrock_volume_multiplier = 1.50` maps the historical surface-denudation calibration to excavated bedrock volume after isostatic rebound. It is not a separate climate or erosion-speed multiplier.

## Old ledger defects exposed and fixed

The sediment ledger revealed two older bookkeeping defects that had been hidden while erosion simply destroyed relief.

### Rift thinning and breakup

Physical rift thinning removes `A * continental_fraction * dH`, but the old diagnostic could record `A * dH`, and in breakup cells it could be evaluated after `continental_fraction` had already been cleared. v0.23 records the actual material volume **before** breakup and separately records the residual continental column removed by final breakup. Both go to the explicit rift-recycled reservoir.

### Arc generation/thickening diagnostics

Arc thickening previously recorded full-cell `A*dH` even in mixed material cells. It now uses `A*continental_fraction*dH`. Juvenile-arc generation is likewise recorded as the actual increase of tracked `continental_volume_km3`, rather than subtracting pre-existing oceanic basalt that was never part of the continental-material ledger.

These fixes affect bookkeeping of tracked material and close the global invariant; they do not reintroduce fractional plate ownership.

## Determinism and tests

Final test suite: **136/136 passing**.

A fresh deterministic checkpoint test compared:

- monolithic `0 -> 40 Myr`, and
- `0 -> 20 Myr -> checkpoint -> resume -> 40 Myr`.

All **22/22 arrays** in `state.npz`, including `sediment_volume_km3`, are bitwise identical and `meta.json` is exactly identical. The sediment budget and complete sediment history therefore survive resume without trajectory dependence.

A canonical subdivision-5 smoke run also passed at **20,480 cells / 20 Myr**, with 12 plates and no topology/ID/ledger failures.

## Tuned 500 Myr subdivision-4 validation

The main long validation uses 5,120 cells and fixed 4 Myr internal steps. Late runs were checkpointed in shorter wall-clock segments because conservative sediment routing is more expensive; deterministic resume means this does not change the simulated trajectory.

Final values at 500 Myr:

| Quantity | Value |
|---|---:|
| Plate count | 8 |
| Topology events | 16 |
| Largest plate fraction | 42.46% |
| Continental-material area | 28.37% |
| Initial continental volume | 3.439693 billion km3 |
| Generated continental volume | 105.776 million km3 |
| Current continental bedrock | 3.218529 billion km3 |
| Surface sediment | 118.645 million km3 |
| Deep recycled sediment | 121.812 million km3 |
| Rift-recycled material | 86.483 million km3 |
| Cycle-recycled material | 0 |
| Cumulative bedrock erosion | 240.456 million km3 |
| Cumulative sediment reworking | 179.059 million km3 |
| Mean sediment thickness | 337.8 m |
| Maximum sediment thickness | 11.06 km |
| Surface area with >1 km sediment | 8.84% |
| Final sea level | -1000.8 m |
| Final land fraction | 27.79% |
| Mean ocean depth | 4.255 km |
| Maximum ocean depth | 8.194 km |
| Hydrosphere relative volume error | -5.58e-10 |
| Last conservative parcel-transport error | -4.77e-7 km3 |
| Global continental-ledger error | **-9.54e-7 km3** |

The full ledger at 500 Myr is:

- LHS `initial + generated = 3,545,468,396.263559 km3`
- RHS `bedrock + surface sediment + deep sediment + rift recycled + cycle recycled = 3,545,468,396.263558 km3`

The residual is approximately one cubic metre in a multi-billion-km3 ledger and is pure floating-point roundoff.

The soft burial-spill closure matters. An earlier diagnostic formulation concentrated sediment to >16 km in a few numerical sinks. Using real sea level for basin classification plus conservative soft spill reduced the final maximum to ~11.1 km while retaining deep sedimentary basins rather than clipping them.

## Paired topology validation

Because v0.23 physically removes bedrock and redistributes its mass, unlike passive v0.22 flexure it is allowed to alter subsequent plate trajectories. The correct convergence question is therefore statistical, not exact-event equality.

Four identical subdivision-3 seeds (`20260806` through `20260809`) were run to 200 Myr in v0.22 and v0.23.

| Metric | v0.22 mean | v0.23 mean |
|---|---:|---:|
| Plate count | **11.75** | **11.75** |
| Topology events | 1.75 | 2.25 |
| Largest plate fraction | 17.41% | 17.30% |
| Sea level | -650.4 m | -639.0 m |

Individual plate-count changes have both signs: `10->11`, `12->11`, `12->11`, `13->14`. Thus the erosion/sediment cycle does **not** show a systematic plate-count regime shift in this small paired ensemble.

One subdivision-4 seed diverges more strongly by ~300 Myr. This is retained as a WATCH rather than used as a calibration target because the multi-seed result shows the expected chaotic branching rather than a one-directional drift.

See `V123_SEDIMENT_PAIRED_200MYR.csv` for the per-seed values.

## Resolution and performance notes

Sedimentary basins are physically narrower than many subdivision-3 cells, so final sediment thickness should not be calibrated against sub3. The production long validation uses sub4 and the canonical target remains sub5.

Conservative routing adds several mesh sweeps per 4 Myr step. Long sub4/sub5 runs are therefore more expensive than v0.22 and may be operationally preferable in deterministic checkpoint segments. This is a performance issue, not a numerical-stability issue.

## Climate boundary

No climate solver is included. v0.23 uses uniform erosivity by default and exposes an external erosivity field for the separate climate model. Hydrology, precipitation, vegetation and weathering climate feedbacks remain outside this program.

## Status and next stage

v0.23 is considered **stable** for the current architecture: the surface sediment cycle is conservative, rift losses are explicit, the global continental-material ledger closes over 500 Myr, checkpointing is exact, and there is no detected systematic topology regime shift in the paired validation.

The next major geological stage should be **continental lithosphere maturation / cratons**: a transported continuous memory of age/depletion/strength so old continental roots resist rifting and subduction while young arc-derived lithosphere remains weak and can mature over time.

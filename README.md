# Moon Tectonics v0.31 — Mantle-Flow-Coupled Plume Sources

## Desktop laboratory GUI

The repository now includes a native PySide6 control surface for Windows and
Ubuntu. It launches the production `run_long_evolution_v131.py` runner in a
separate Python process, divides long integrations into deterministic
checkpoint segments, previews newly written PNG/GIF artifacts, and reads live
metrics from the latest safe checkpoint. The UI process never owns numerical
simulation state, so closing or restarting the interface cannot corrupt the
last completed checkpoint.

Windows setup and launch:

```text
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
launch_gui.bat
```

Ubuntu setup and launch (the numerical code and GUI contain no Windows-only
paths or APIs):

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python launch_gui.py
```

Install or refresh dependencies with `python -m pip install -r
requirements.txt`. The lightweight `PySide6-Essentials` distribution supplies
the Qt Core, GUI and Widgets modules used here without unrelated WebEngine/3D
downloads. A fresh run must use an empty output directory. To continue
a run, select one of its `gui_checkpoint_*_Myr` folders; the GUI reads its time
and mesh resolution and resumes with the existing output directory. `Pause
safely` finishes the current segment before stopping. `Stop now` interrupts the
active process but leaves the previous checkpoint intact.

The GUI shows active elapsed time and an approximate ETA after two completed
segments, using the latest five segment timings. Pauses are excluded and a
checkpoint resume estimates only the remaining simulated interval. If a
segment exceeds its prediction, the display explicitly recalibrates rather
than claiming a false zero; final map/GIF assembly may take extra time. This is
GUI-only bookkeeping, not a change to the numerical integrator. An already
open GUI retains its loaded version: use the new display after the current
run finishes and the GUI is restarted. Do not edit numerical modules or runtime
configuration during a live run, because each segment starts a fresh process.

The scenario selector intentionally enables only mature v0.31 tectonics. The
three genesis histories (disk/quiet, disk/impact, capture/circularization) are
visible as planned modes so their future orbital, thermal, and plate-onset
parameters can enter without redesigning the application.

The local resolution timing and the sub-5 decision for the first full run are
recorded in `V131_GUI_RESOLUTION_BENCHMARK.md`. The completed 500 Myr
subdivision-5 run and its numerical/scientific acceptance checks are recorded
in `V131_GUI_CANONICAL_500MYR_FINDINGS.md`.

The apparent 340–500 Myr low-motion region at 0–90°E, 60°S–0° is traced to a
merged plate's Euler pole—not a frozen renderer or transport queue—in
`V131_STAGNANT_SECTOR_DIAGNOSIS.md`. Its reproducible checkpoint analysis is
`analysis/diagnose_stagnant_sector_v131.py`.

The paired 300–500 Myr replay comparing historical area-weighted welding with
an opt-in full inertia-tensor rule, including the late recovery of the quiet
sector and continental-material transport diagnostics, is documented in
`V131_MERGE_KINEMATICS_PAIR_FINDINGS.md`.

The post-v0.31 genesis pipeline, including the independent 3 x 5 matrix of
satellite-origin and plate-onset hypotheses, is specified in
`GENESIS_ARCHITECTURE.md`.

---

v0.31 couples mobile deep plume sources to the checkpointed Eulerian mantle
flow already present in the model. Each conduit continuously samples the
fixed-grid field, follows `0.35` of its resolved tangent velocity and retains
`0.30` of the deterministic v0.30 drift as an unresolved residual. The two
vectors determine the effective source path and bends; the overlying plate and
transported volcanic material remain independent.

Validation: **174/174 tests**, exact equality of all 62 arrays and metadata
across checkpoint/resume, complete subdivision-3 and canonical 20,480-cell
Windows smokes, and five new 500-Myr flow-coupled worlds compared with the ten
v0.30 controls. Across four
subdivision-3 seeds, mean source path decreases from `19,086` to `11,360 km`
relative to independent drift while final age-distance correlation increases
from `+0.582` to `+0.728`. Material ledgers remain closed to `2.79e-9 km3`
(igneous) and `9.54e-7 km3` (continental), with zero elevation clips.

Primary runner: `run_long_evolution_v131.py`. See
`V131_FLOW_COUPLED_PLUME_FINDINGS.md`,
`V131_FLOW_COUPLED_PLUME_VALIDATION_500MYR.csv`, and
`V131_FLOW_COUPLED_PLUME_VALIDATION_500MYR.png`.

---

# Moon Tectonics v0.30 — Mobile Plume Sources and Bent Hotspot Tracks

v0.30 lets deep plume sources drift independently of the overlying plates.
Each checkpointed source follows persistent great-circle arcs, reorients at
finite intervals and records its path and bend angles. Diagnostics now separate
source velocity, plate velocity and their relative hotspot-track velocity;
source trajectories are overlaid on the GIF. A powered-Gaussian quadrature
normalization keeps nonlinear narrow-tail productivity stable as a moving
source crosses differently aligned mesh cells.

Validation: **171/171 tests**, exact equality of all 59 arrays and metadata
across checkpoint/resume, a 240-Myr 16-frame GIF, and ten paired 500-Myr
worlds. The mobile subdivision-3/4 source check differs by `0.021%` in the tail
productivity time integral and `0.385%` in generated igneous volume; source
paths are exactly resolution-independent. Maximum absolute igneous and
continental ledger errors are `2.79e-9` and `9.54e-7 km3`, with zero elevation
clips.

Primary runner: `run_long_evolution_v130.py`. See
`V130_MOBILE_PLUME_FINDINGS.md`,
`V130_MOBILE_PLUME_VALIDATION_500MYR.csv`, and
`V130_MOBILE_PLUME_VALIDATION_500MYR.png`.

---

# Moon Tectonics v0.29 — Plume Heads, Tails and Hotspot Tracks

v0.29 separates each mantle plume into a short broad head and a narrow,
persistent tail. The source creates permanent volcanic material that moves with
the plates, while transported heat weakens lithosphere, active rifts focus melt
into dykes, and old underplate can eclogitize and delaminate into the existing
deep-recycled reservoir. Component source kernels are area-normalized to reduce
grid-dependent plume production.

Validation: **167/167 tests**, exact equality of all 53 arrays and metadata
across checkpoint/resume, a subdivision-5 canonical smoke, a 120-Myr six-panel
GIF, and ten 500-Myr paired validation worlds. The maximum absolute igneous
ledger error is `1.86e-8 km3`, continental ledger error is `9.54e-7 km3`, and
no elevation clips occur. At subdivision 3, the full model produces a mean
age–distance correlation of `+0.752`; its area-normalized integrated source
changes by only `1.65%` from subdivision 3 to 4 for the resolution-check seed.
Late plate topology and retained surface volume remain resolution-sensitive.

Primary runner: `run_long_evolution_v129.py`. See
`V129_HOTSPOT_TRACK_FINDINGS.md`,
`V129_HOTSPOT_TRACK_VALIDATION_500MYR.csv`, and
`V129_HOTSPOT_TRACK_VALIDATION_500MYR.png`.

---

# Moon Tectonics v0.28 — Permanent Plume Magmatism and Volcanic Tracks

v0.28 turns plume decompression-melt productivity into permanent, transported
extrusive-basalt, dyke/sill and mafic-underplate reservoirs. The mantle source
stays fixed while the material moves with its plate, leaving a checkpointed
volcanic track and age field. A closed generated/surface/deep-recycled ledger
prevents material loss, and reservoir densities control flexed crustal support.

Validation: **161/161 tests**, exact continuous versus checkpointed equality
for all 45 arrays and complete metadata, a 40-Myr smoke, and a full 2×2×2
500-Myr subdivision-3 factorial isolating dynamic uplift, permanent magmatism
and mechanical plume-rift coupling. Magmatic runs retain `9.04–9.33 million
km3` at the surface, reach `2.594–2.601 km` added thickness and `224 m` local
density-aware support. All eight worlds have zero elevation safety-rail clips;
the maximum absolute igneous-ledger error is `1.86e-8 km3`.

Primary runner: `run_long_evolution_v128.py`. See
`V128_PLUME_MAGMATISM_FINDINGS.md`,
`V128_PLUME_MAGMATISM_FACTORIAL_500MYR.csv`, and
`V128_PLUME_MAGMATISM_FACTORIAL_500MYR.png`.

---

# Moon Tectonics v0.27 — Transient Plume Dynamic Topography

v0.27 turns the plume uplift diagnostic into delayed, reversible surface
relief. Convective support enters the non-flexed topographic background, affects
the solved ocean and conservative surface-process system, and creates no crustal
material. Its area-weighted degree-zero component is removed every step, so a
local plume swell does not change the planet's mean radius.

Validation: **156/156 tests**, byte-identical continuous versus checkpointed
continuation, a six-GIF local smoke, and four paired 500-Myr subdivision-3
control/dynamic comparisons. Production plume swells remain below the `1 km`
ceiling; all paired runs have zero elevation safety-rail clips, close the final
continental ledger to `<=1.43e-6 km3`, and keep the zero-mean dynamic
displacement residual below `1.91e-9 km3`.

Primary runner: `run_long_evolution_v127.py`. See
`V127_DYNAMIC_TOPOGRAPHY_FINDINGS.md`,
`V127_DYNAMIC_TOPOGRAPHY_PAIRED_500MYR.csv`, and
`V127_DYNAMIC_TOPOGRAPHY_PAIRED_500MYR.png`.

---

# Moon Tectonics v0.26 — Plume-Driven Mechanical Rifting

v0.26 gives the v0.25 mantle plumes an independently switchable mechanical
effect. Broad plume-head extension and flank localization are passed through
the existing progressive continental-rift solver, so the plume cannot directly
create a plate boundary. Diagnostic dynamic uplift and magmatic productivity
are recorded but do not yet alter topography or crustal volume.

Validation: **151/151 tests**, byte-identical continuous versus checkpointed
continuation, a local five-GIF smoke run, and sixteen 500-Myr subdivision-3
runs spanning control, weakening-only, forcing-only and combined modes.
Mechanical forcing raises ensemble-mean maximum rift extension from `0.126` to
`0.797–0.839`; successful breakup appears in half the mechanically forced
worlds and none of the non-mechanical worlds. All ensemble runs have zero
elevation safety-rail clips and close the final global continental ledger to
`<=1.43e-6 km3`.

Primary runner: `run_long_evolution_v126.py`. See
`V126_PLUME_RIFTING_FINDINGS.md`, `V126_PLUME_RIFTING_4MODE_500MYR.csv`, and
`V126_PLUME_RIFTING_4MODE_500MYR.png`.

---

# Moon Tectonics v0.25 — Mantle Plumes and Metasomatic Root Modification

v0.25 adds a deterministic, checkpointable population of approximately mantle-fixed plume heads. Plates move across the plume field, so the affected material naturally forms plume tracks. Continental lithosphere above an active plume is thermally rejuvenated, melt/fluid-refertilized and basally eroded; its transported v0.24 age, depletion and craton-strength memory records the encounter after it moves away. Plume flux, cumulative exposure and imposed root erosion have dedicated maps and histories.

Validation: **147/147 tests**, a full canonical-sub5 smoke at 20,480 cells, exact bitwise `0->12` versus `0->4->resume->12` equality for both the compressed numerical state and complete JSON metadata, and four paired subdivision-3 500-Myr control/plume runs. The plume runs have zero elevation safety-rail clips and close the global continental ledger to `<=9.54e-7 km3`. Direct local erosion reaches 21–47 km, while the coupled ensemble reorganizes nonlinearly and finishes with 3.25 fewer plates on average. The current coupling does not itself nucleate a plume-centered rift, so these first ensemble statistics diagnose model feedback rather than constitute a calibrated planetary prediction.

Primary runner: `run_long_evolution_v125.py`. See `V125_PLUME_FINDINGS.md`, `V125_PLUME_PAIRED_500MYR.csv`, and `V125_PLUME_PAIRED_500MYR.png`.

---

# Moon Tectonics v0.24 — Continental-Lithosphere Maturation and Cratons

v0.24 adds transported continuous memory that distinguishes juvenile arc-derived continent from old cratonic lithosphere. Each continental parcel now carries effective lithosphere age, mantle-depletion fraction, and a derived craton-strength field. Quiet roots mature, new arc volume dilutes old memory, and sustained rifting or thermal weakening rejuvenates it.

The strength field thickens and increases the compositional buoyancy of continental mantle roots, raises continent-continent collision resistance, and redirects both kinematic and late-nucleated extension toward weak juvenile/orogenic belts. A non-zero extension floor means sufficiently persistent forcing can still rupture a craton; the model does not make old continents indestructible.

Validation: **142/142 tests**, exact bitwise `0->12` versus `0->4->resume->12` checkpoint equality for all **25 arrays** plus exact metadata, canonical subdivision-5 smoke at 20,480 cells, four paired subdivision-3 500-Myr runs, and a complete 5,120-cell subdivision-4 500-Myr validation. Across the four paired seeds, mean continental mantle-root thickness increases `106.2 -> 142.8 km`, mean maximum rift extension falls `0.169 -> 0.089`, and final mean cratonic share is 68.2% of continental material; mean plate count remains in the same regime (`11.25 -> 11.75`). The detailed sub4 ledger closes to `4.77e-7 km3` with zero elevation safety-rail clips.

Primary runner: `run_long_evolution_v124.py`. See `V124_CRATON_FINDINGS.md` and `V124_CRATON_PAIRED_500MYR.csv`.

---

# Moon Tectonics v0.23 — Conservative Sediments and Closed Material Ledger

v0.23 replaces the legacy non-conservative topographic erosion sink with an explicit continental bedrock/sediment/recycling cycle. Surface sediment is transported with the lithospheric parcel, subducted sediment enters a deep-recycled reservoir, and continental material removed by rift thinning/breakup enters a separate rift-recycled reservoir. The runner checks the complete continental-material ledger every step. Climate remains external; a future climate model may supply an `erosivity_field`.

Production routing uses the solved previous sea level for basin deposition and a conservative soft burial-spill closure, avoiding hard sediment clipping. Sediment thickness and density feed material-aware topography and flexural loading. The old topography erosion path is disabled in the v0.23 runner.

Validation: **136/136 tests**, exact bitwise `0->40` versus `0->20->resume->40` checkpoint equality for 22 arrays plus exact metadata, canonical sub5 smoke at 20,480 cells, a closed 500-Myr subdivision-4 global material ledger (error ~`9.5e-7 km3`), and a four-seed paired sub3 comparison whose mean plate count is unchanged (`11.75 -> 11.75`). See `V123_CONSERVATIVE_SEDIMENTS_FINDINGS.md` and `V123_SEDIMENT_PAIRED_200MYR.csv`.

Primary runner: `run_long_evolution_v123.py`.

---

# Moon Tectonics v0.22 — Flexural Isostasy

v0.22 adds a variable-rigidity spherical elastic-plate response for tectonic/isostatic loads. Effective elastic thickness is derived continuously from the mechanical lithosphere and weakened by rifting, seam damage and tidal damage. Thermal ocean subsidence and ridge buoyancy remain separate rather than being flexed twice.

Validation: **131/131 tests**, exact checkpoint/resume, sub5 smoke, and a 500-Myr sub4 baseline in which all tectonic/material arrays are bitwise identical to v0.21 and only elevation changes; final sea-level difference is ~0.7 m. See `V122_FLEXURE_FINDINGS.md`.

Primary runner: `run_long_evolution_v122.py`.

# Moon Tectonics v0.21 — Slab-Geometry Volcanic Arcs

v0.21 moves volcanic-arc magmatism and arc topography off the trench and onto a slab-geometry front. Arc position follows remembered slab dip and a ~105 km dehydration/melting depth, producing a trench-parallel projected arc on the overriding plate. The field feeds the existing juvenile-felsic and continental-arc cycle; forearc erosion remains at the trench. A weak, short post-breakoff magmatic pulse is included.

Validation: **127/127 tests**, exact bitwise checkpoint/resume, a canonical sub5 smoke run, four-seed sub3 and three-seed sub4 paired comparisons. The sub4 ensemble preserves the v0.20 continental volume essentially exactly without productivity retuning. See `V121_VOLCANIC_ARC_FINDINGS.md`.

Primary runner: `run_long_evolution_v121.py`.

# Moon Tectonics v0.20 — Collision-triggered Slab Breakoff

v0.20 extends the v0.19 rollback model with finite-time slab necking and detachment when buoyant continental lithosphere reaches a formerly oceanic subduction zone.  Breakoff is deliberately an effective 2.5-D process: it removes the calibrated surface slab-pull contribution after a sustained collisional stall, but it does not yet model 3-D tear propagation through the slab.

Production invariants:
- an otherwise active oceanic trench does **not** neck merely because continental material is nearby; the remembered oceanic contact must first stall/reclassify;
- a mature slab must exceed minimum length/depth thresholds before necking can accumulate;
- breakoff damage is time-integrated and relaxes after brief contacts;
- detached slabs enter a 40-Myr tombstone/cooldown, immediately losing slab pull and rollback;
- v0.19 slab pull/ridge push amplitudes are unchanged before actual breakoff.

Primary runner: `run_long_evolution_v120.py`.

# Moon Tectonics v0.19 — Rollback / Trench Migration

v0.19 adds conservative 2.5-D slab rollback and rollback-driven back-arc extension on top of v0.18 persistent slab memory. Production rollback is deliberately weak: max 2 km/Myr, with a 120–650 km landward back-arc forcing band peaking near 280 km and max extension forcing 0.15. It does not directly create new boundaries.

Validation: **117/117 tests**, exact bitwise checkpoint/resume including diagnostic history, a 20-Myr canonical sub5 smoke run, and a four-seed paired 300-Myr sub3 comparison. See `V119_ROLLBACK_FINDINGS.md`.

Recommended runner:

```bash
PYTHONPATH=. python run_long_evolution_v119.py --end-time 500 --dt 4 --finalize
```

---

# Moon Tectonics v0.18 — Subduction Memory

v0.18 stores persistent oriented slab states and a small residual pull after short-lived loss of surface contact. Active mature slab pull is unchanged from v0.17, avoiding double counting. Production residual gain is 5%, decay timescale 24 Myr, detach after 80 Myr. The original temporary v0.18 source was lost before archiving; this branch contains the documented reconstruction used as the base for v0.19. See `V118_SUBDUCTION_MEMORY_FINDINGS.md`.

---

# Moon Tectonics v0.17-ridge-push

v0.17 makes ridge push use the explicit mechanical mantle-lithosphere layer
introduced in v0.16 instead of one constant line-force multiplier at every
divergent boundary. Climate remains intentionally outside this program.

Key additions:

- thermal ridge-push GPE proxy `max(Δρ,0) * H_mantle^2`;
- area-weighted plate-side oceanic GPE so force does not scale with mesh-cell count;
- strong plate-cooling saturation for mature oceanic lithosphere;
- newly opened/young ocean basins therefore have weak ridge push, while mature
  ridge flanks rapidly approach the calibrated v0.16 force level;
- mixed continental/oceanic ridge faces blend continuously back toward the
  legacy continental-rift drive;
- ridge-push min/mean/max diagnostics and final GPE/factor maps;
- legacy states without v0.16 mantle-lithosphere arrays retain the old constant
  ridge-push behaviour.

The production calibration uses an 80-Myr mantle-lithosphere reference
(`H≈93.5 km` below 7 km of oceanic crust), a plate-cooling saturation ratio of
0.20, and gain 1.014. An ideal flank therefore has approximate multipliers:
5 Myr `0.20`, 10 Myr `0.38`, 20 Myr `0.67`, 40 Myr `0.92`, 80 Myr `1.01`,
160 Myr `1.02`. Thus the physically important change is concentrated in young
ocean; mature ocean does not acquire an ever-growing artificial push.

Validation: **109/109 tests**. A paired four-seed sub3 comparison at 300 Myr
gives 11.25 plates / 5.25 topology events for v0.16 and 11.75 / 5.75 for
v0.17, with nearly unchanged mean continental-plate speed (0.2187 vs
0.2171 deg/Myr) and continental-material area (29.05% vs 29.10%). A 20-Myr
canonical sub5 (20,480-cell) smoke run also passes. Checkpoint/resume remains
bitwise deterministic. See `V117_RIDGE_PUSH_FINDINGS.md`.

Recommended runner:

```bash
PYTHONPATH=. python run_long_evolution_v117.py --end-time 500 --dt 4 --frame-interval 52 --surface-only-frames --finalize
```

Physical basis: classic thermal ridge-push theory treats the force as the GPE
contrast of cooling oceanic lithosphere. Half-space cooling gives an
approximately age-linear force, while plate-cooling models saturate strongly
for mature seafloor (Parsons & Richter, 1980). v0.17 intentionally uses the
saturating form because v0.16 already caps mechanical lithosphere thickness.

---

# Moon Tectonics v0.15-material-topography

v0.15 makes relief and sea level use the conservative continental-material
layer introduced in v0.11. Climate is intentionally **not** part of this model;
climate fields are expected to be computed later by the separate higher-power
climate program.

Key additions:

- continuous material-aware Airy/isostatic relief from `continental_fraction`
  and `continental_volume_km3`;
- no topographic jump at the legacy 50% `crust_type` visibility threshold;
- material-aware ridge/collision/trench/arc relief amplitudes;
- two-patch sub-grid hypsometry for mixed continental/oceanic cells when solving
  conserved water volume and sea level;
- coastline/land diagnostics based on real exposed sub-grid area rather than
  the sign of one averaged mixed-cell elevation;
- majority-patch filled surface rendering for readable coastline GIFs.

Validation: **97/97 tests** and bitwise-identical `0→40 Myr` versus
`0→20→resume→40 Myr` checkpoint state/history. In the 500-Myr sub3 diagnostic,
continental material finishes at 29.48% of the surface, actual land at 28.01%,
and submerged continental material at 1.46%. See
`V115_MATERIAL_TOPOGRAPHY_FINDINGS.md`.

Recommended runner:

```bash
PYTHONPATH=. python run_long_evolution_v115.py --end-time 500 --dt 4 --frame-interval 52 --save-frame --surface-only-frames --finalize
```

---

# Moon Tectonics v0.14-hydrosphere

v0.14 adds the first real global ocean to the stabilized v0.13 tectonic model.
The hydrosphere is intentionally **passive** in this version: tectonics and
topography reshape the basins, while one conserved water inventory determines
a globally solved sea level. Water does not yet feed back into erosion,
sedimentation, lithospheric loading, climate, or plate forces.

Key additions:

- conserved global `water_volume_km3`;
- exact spherical-shell water-volume integration on the icosphere cells;
- sea level solved from the current topography at every tectonic step;
- land/ocean, shallow-sea and deep-ocean area diagnostics;
- exposed vs submerged continental-material diagnostics;
- hydrosphere state/history persisted in checkpoints;
- filled shoreline/surface GIF (`surface_history.gif`);
- mature initial oceanic age field based on distance from active divergent ridges, replacing the legacy all-zero-age ocean that caused an artificial whole-basin subsidence transient.

Fresh runs default to `water_volume_km3: null`: the inventory is calibrated once
so the initial sea level equals the historical 0-m topographic datum, then is
held fixed forever. A v0.14 checkpoint must be used for resume so water is never
silently recalibrated.

Validation: **91/91 tests**, plus bitwise-identical `0→40 Myr` versus
`0→20→resume→40 Myr` checkpoints including hydrosphere history. A 500-Myr
sub3 diagnostic run keeps land near 28% while sea level evolves with basin
geometry (see `V114_HYDROSPHERE_FINDINGS.md`).

Recommended runner:

```bash
PYTHONPATH=. python run_long_evolution_v114.py --end-time 500 --dt 4 --frame-interval 52 --finalize
```

---

# Moon Tectonics v0.13-topology-hysteresis

v0.13 is the topology-stabilization checkpoint built on the v0.11 conservative continental-material layer and the v0.12 physical-unit topology rules.

Key additions:

- a plate must remain below the physical microplate-area threshold for 20 Myr before cleanup absorption;
- zero-area plates are removed explicitly as `vanish` topology events;
- collision coupling changes velocities only and can never compact IDs;
- owner/plate-ID consistency is asserted during long runs;
- topology-manager microplate memory is checkpointed and survives resume.

Validation: **86/86 tests**, bitwise deterministic checkpoint/resume through both split and vanish events, and a 500-Myr resolution ensemble (10×sub3, 10×sub4, 3×sub5). Mean final plate counts are 11.7, 11.5 and 12.3 respectively. See `V113_TOPOLOGY_VALIDATION.md`.

Recommended runner:

```bash
PYTHONPATH=. python run_long_evolution_v113.py --end-time 500 --dt 4
```

---

# Moon Tectonics v0.11-material

v0.11 is the first **new forward-development version** after the functional
reconstruction of the lost v0.10 source.  It fixes a grid representation defect
that could turn several colliding continental parcels into one unrealistically
thick raster column and then trigger artificial delamination/recycling.

## Core representation change

Plate kinematics remain discrete: every surface cell still has one plate owner.
Continental material is now a separate conservative layer:

- `continental_fraction` stores how much of a cell's surface footprint is
  occupied by continental material;
- `continental_volume_km3` stores continental-crust volume independently of the
  visible plate/crust owner;
- effective continental thickness is derived from `volume / (area*fraction)`;
- overlap at continent-continent collision is spread locally into available
  footprint while conserving both continental area and volume;
- the binary visible `crust_type` is derived from the material layer for legacy
  mechanics/plots, rather than being the sole source of material bookkeeping.

This specifically prevents the former numerical path:
`2+ continental cells -> 1 raster cell -> 100-300 km crust -> delamination ->
artificial loss of continents`.

## 500-Myr resolution convergence — seed 20260806

Same physics and `dt=4 Myr` at three icosphere resolutions:

| subdivision | cells | initial material area | final material area | final continental volume | max raw overlap | max post-redistribution |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1,280 | 28.246% | 29.325% | 3.564e9 km³ | 106.8 km | 42.1 km |
| 4 | 5,120 | 27.978% | 28.624% | 3.471e9 km³ | 102.0 km | 42.5 km |
| 5 | 20,480 | 27.920% | 28.236% | 3.439e9 km³ | 114.2 km | 46.9 km |

At canonical subdivision 5, continental volume changes only from 3.4325e9 to
3.4392e9 km³ over 500 Myr.  Explicit recycling/delamination is zero in this
interval, and pure conservative-transport error remains at ~9.54e-7 km³.
The catastrophic continental-area collapse seen before v0.11 is therefore not
present and the material budget improves with resolution.

Plate topology itself is **not yet resolution-converged** (final plate count and
largest-plate fraction differ substantially between sub3/sub4/sub5).  Treat that
as a separate WATCH item; do not tune material conservation to compensate for it.

## Visualization

History frames now render filled rasterized Mollweide maps instead of point
clouds. `plate_history.gif` also includes representative Euler-motion arrows.
This is diagnostic visualization only and does not alter the simulation grid.

## Validation

- **76/76 tests pass**;
- new tests cover conservation of material area/volume and collision-footprint
  redistribution;
- 0→40 Myr monolithic and 0→20→resume→40 Myr runs are bitwise-identical,
  including the material layer;
- 500-Myr convergence runs completed at subdivision 3, 4 and canonical 5.

Recommended runner:

```bash
PYTHONPATH=. python run_long_evolution_v110.py --end-time 500 --dt 4
```

---

# Moon Tectonics v0.9.7

v0.9.7 keeps the **v0.9.6 collision/weld calibration unchanged** and calibrates the long-term continental-crust **surface-area budget**.

## Why v0.9.7 was needed

The v0.9.6 4-Gyr run remained tectonically healthy, but continental surface area drifted from **27.92% to 19.73%**. Diagnostics showed essentially zero surface-area loss from subduction erosion; the dominant loss was completed continental breakup (**32.29 million km²**).

The important bug was not simply an excessive rift-rate constant. After a successful continental breakup opened an oceanic axial strip, later spreading gaps beside the new passive margin could be interpreted again as fresh continental rifts. The spreading centre could therefore erode the continent row by row over hundreds of Myr.

## Changes

- broad failed-rift halos may thin, but completed oceanic breakup is restricted to the axial high-extension core (`forcing >= 0.75`);
- progressive continental-rift accumulation and thinning scale linearly with the thermal tectonic-activity factor (`activity^1.0`), matching the late-time decline already applied to crust-production processes;
- an opening gap drives **continental** extension only while continental material from **two distinct plate flanks** borders the gap;
- once an oceanic axial strip exists, subsequent opening is treated as oceanic spreading and no longer recursively consumes a single passive continental margin;
- diagnostics record active two-flank continental-rift gaps and suppressed passive-margin gaps;
- v0.9.6 collision zones, mechanical coupling and weld thresholds are unchanged.

Relevant configuration:

```yaml
continental_rifting:
  extension_rate_per_myr: 0.012
  extension_relaxation_myr: 90.0
  min_duration_myr: 32.0
  extension_threshold: 0.70
  thinning_km_per_myr: 0.16
  min_breakup_thickness_km: 19.0
  breakup_min_extension_forcing: 0.75
  extension_requires_two_plate_flanks: true
  activity_scaling_exponent: 1.0
  tidal_thinning_boost_max_fraction: 0.25
```

## 4-Gyr calibration result

Canonical moon, `dt = 4 Myr`, deterministic checkpoint/resume.

- plate count: **12 → 19**;
- topology events: **87** (`42 disconnect_split`, `30 merge`, `10 absorb`, `5 split`);
- last topology event: **3828 Myr**;
- largest final plate: **31.87%** of the surface;
- continental area: **27.92% → 29.01%**;
- completed continental breakup: **0.481 million km²**, down **98.5%** from v0.9.6;
- cumulative continental thinning: **0.124 billion km³**, down about **90.0%** from v0.9.6;
- juvenile/generated continental area: **4.59 million km²**;
- a genuine late-rift nucleation occurs at **2472 Myr**, followed by real `split` events at 2576, 2588, 2820, 3064 and 3824 Myr: the passive-margin fix therefore does not disable continental breakup;
- weighted median oceanic-crust age: **72 Myr**;
- ocean older than 200 Myr: **35.9%**; older than 2 Gyr: **4.8%**;
- mantle temperature: **1850 → 1677.8 K**;
- tectonic activity factor: **1.000 → 0.439**;
- final relief: **−9.66…+7.77 km**;
- numerical elevation safety-rail hits: **0**.

### Verdict

**Continental surface-area budget: PASS.** The former systematic disappearance of continental area was caused mainly by recursive passive-margin breakup and is removed without freezing plate tectonics or suppressing genuine late rifts.

**WATCH — continental volume/thickness budget.** Although area is now stable, continental volume grows from **3.43 to 4.86 billion km³** (+41.7%). Area-weighted mean continental thickness rises from **35.0 to 47.7 km**, with a median near **50 km**. This is a separate mass/thickness calibration, probably dominated by repeated collision/accretion thickening versus delamination, and should be addressed before treating the crustal volume history as final.

The final ocean-age distribution is also somewhat older than v0.9.6, but it does not show the multi-Gyr oceanic lock-up seen in v0.9.3.

Full diagnosis: `outputs_v097_4000_final/long_run_assessment_v097.json`.

## Tests

```bash
python -m pytest -q
```

**67/67 tests pass.**

# Moon Tectonics v0.9.6

Долгосрочная геодинамическая модель проекта **обитаемой луны газового гиганта**.

Канонический мир без изменений: `0.50 M_earth`, `R≈5287 km`, `g=7.12 m/s²`, синхронный период `47 h`, гигант `5 M_J`, RMS `e=0.00047` либо внешняя история `e(t)`.

## Задача v0.9.6

v0.9.4 и v0.9.5 дали две противоположные крайности:

- v0.9.4: поздняя тектоника жива, но крупнейшая плита достигала ~49.5% поверхности и collision/weld ещё выглядел слишком липким;
- v0.9.5: progressive collision устранила визуальное слипание, но weld стал слишком редким — финал 37 плит, крупнейшая лишь 6.8%, континентальная площадь упала до 13.2%.

v0.9.6 сохраняет двухстадийную схему `collision zone → mechanical coupling → quiet weld`, но ищет промежуточный режим.

## Изменения

Эффективные параметры collision/weld:

- coupling начинается после `20 Myr`;
- characteristic coupling time: `55 Myr`;
- максимальная доля сближения Euler-motion за шаг: `0.14`;
- минимальный возраст зрелой коллизии: `112 Myr`;
- отдельная quiet weld phase: `40 Myr`;
- weld допускается при относительной скорости ≤ `15 km/Myr` и малом нормальном расхождении.

Дополнительно v0.9.6 вводит **локальное подавление рифтинга в зрелой континент–континентальной collision-zone**. Это не запрещает постколлизионный рифтинг вообще: suppression действует только на сам сжимающийся контакт и один соседний пояс, а вне collision-zone правила рифтинга прежние.

## Проверка 500 Myr

Перед 4-Gyr прогоном получено:

- 15 плит;
- крупнейшая плита 19.6%;
- continental area `27.92% → 27.84%`;
- два настоящих weld/merge уже к 500 Myr;
- рельеф устойчив, safety-rail clips = 0.

## Полный прогон 4 Gyr

`dt = 4 Myr`, deterministic checkpoint/resume.

Итог:

- плит: **28**;
- topology events: **100**;
  - `merge`: 19;
  - `split`: 16;
  - `disconnect_split`: 42;
  - `absorb`: 23;
- последнее topology event: **3992 Myr**;
- крупнейшая плита: **28.85%** поверхности;
- 3 плиты >5%, 17 плит >2%, 24 плиты >1%;
- continental area: **27.92% → 19.73%**;
- continental volume: **3.43 → 2.27 млрд km³**;
- cumulative continental breakup: **32.29 млн km²**;
- weighted median oceanic-crust age: **40 Myr**;
- ocean older than 200 Myr: **13.7%**;
- ocean older than 2 Gyr: **7.4%**;
- mantle temperature: **1850 → 1677.8 K**;
- tectonic activity: **1.000 → 0.439**;
- final relief: **−9.62…+4.80 km**;
- numerical elevation clips: **0**.

### Вердикт

**Collision/weld calibration — PASS.** Старое визуальное «столкнулись, склеились и замерли» не возвращается, но и v0.9.5-подобной равномерной сверхфрагментации нет. Размеры плит остаются сильно неодинаковыми, перестройки продолжаются до самого конца 4-Gyr истории.

**WATCH:** континентальная кора всё ещё медленно убывает. Это отдельная следующая калибровка бюджета континентальной коры (ювенильное образование/аккреция против рифтинга и переработки), а не причина снова менять collision/weld.

Полный диагноз: `outputs_v096_4000/long_run_assessment_v096.json`.

## Запуск

```bash
python run_long_evolution_v096.py --end-time 100 --dt 4 --output outputs_v096_long --checkpoint outputs_v096_long/checkpoints/cp0100 --save-frame
```

Продолжение:

```bash
python run_long_evolution_v096.py --resume outputs_v096_long/checkpoints/cp0100 --end-time 200 --dt 4 --output outputs_v096_long --checkpoint outputs_v096_long/checkpoints/cp0200 --save-frame
```

## Тесты

```bash
python -m pytest -q
```

**66/66 тестов проходят.**


## v0.16 — crust / mantle-lithosphere split

`crust_thickness_km` now remains the chemical crustal layer.  The mechanical
plate also carries `mantle_lithosphere_thickness_km` and
`mantle_lithosphere_density_anomaly_kg_m3`.  Oceanic mantle lithosphere grows
with local cooling age using a capped half-space-cooling proxy; continental
mantle roots are independent of crust thickness and thin progressively with
rift extension.  When these fields are present, ocean-ocean subduction polarity
and slab pull use local integrated mantle-lithosphere negative buoyancy rather
than double-counting crust age plus one global thermal-lithosphere thickness.

This is still a surface plate model: v0.16 does **not** yet advect a 3-D slab
into the mantle or model slab breakoff/rollback as an explicit buried object.


### v0.16 validation snapshot

The calibrated 500-Myr sub3 control remains in the v0.15 macro-regime: 13 final
plates, 9 topology events and ~0.241 deg/Myr mean continental-plate speed.
The canonical sub5 mesh also passes a fresh 20-Myr sanity run.  Checkpoint/resume
is bit-identical with the new layer fields.  See `V116_LITHOSPHERE_SPLIT_FINDINGS.md`.

## v0.22 — Flexural Isostasy

v0.22 adds variable-rigidity elastic-plate flexure on the spherical finite-volume
mesh.  Effective elastic thickness is derived from the local mechanical
lithosphere and weakened by rifting/collision/tidal damage.  Only mechanically
load-like relief (continental thickness anomaly, arcs, collisions and trenches)
is flexed; thermal ocean subsidence and ridge thermal uplift remain local.

The 500-Myr sub4 v0.21/v0.22 baseline is tectonically bit-identical.  Only
`elevation_m` differs (RMS ~20.7 m), while final sea level differs by only 0.70 m.
Checkpoint/resume is bit-identical and the full suite is 131/131 PASS.  See
`V122_FLEXURE_FINDINGS.md` and `V122_FLEXURE_BASELINE_500MYR.csv`.

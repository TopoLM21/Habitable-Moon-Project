# v0.24 Cratonic Memory — Findings

## Scope

v0.24 adds a continuous, transported memory of continental-lithosphere maturation on top of the stable v0.23 conservative sediment/material ledger. The production version is `0.24-cratonic-memory`; the primary runner is `run_long_evolution_v124.py`.

This is an effective spherical plate model, not a 3-D thermo-petrological solver. Its purpose is to represent the first-order geological distinction that v0.23 lacked: juvenile arc-derived continental lithosphere should remain weak, while old depleted roots can become thick, buoyant and resistant without becoming indestructible.

## New material-following fields

`LithosphereState` now optionally stores:

1. `continental_lithosphere_age_myr` — effective age of the coupled crust/root column;
2. `mantle_depletion_fraction` — normalized melt-depletion/compositional-buoyancy proxy;
3. `craton_strength` — a derived 0–1 maturity field requiring both age and depletion.

All three fields use the same winning-parcel source map as the existing lithosphere memories. Divergent newborn ocean and fully recycled/broken continental material reset them to zero. Checkpoints store all three arrays and the full diagnostic history.

## Maturation, dilution and rejuvenation

- Quiet continental material ages chronologically and approaches the configured maximum depletion on a 900-Myr effective timescale.
- Strength is the geometric mean of age maturity and normalized depletion, blended continuously by continental footprint fraction. Both an old column and a depleted root are therefore required.
- New juvenile or continental-arc volume is mixed with zero-age, zero-depletion material. The old memory is diluted by the retained old-volume fraction rather than being copied unchanged into newly generated crust.
- Sustained rift extension and strong thermal weakening reduce effective age and depletion exponentially. This represents root rejuvenation/refertilization or loss, not reverse chronological time.
- Oceanic cells and cells with no tracked continental footprint carry zero craton memory.

## Mechanical feedbacks

The derived strength feeds four existing mechanisms without changing plate ownership or the continental mass ledger:

1. **Mantle roots:** continental mantle-lithosphere target thickness grows from the v0.23 baseline toward an additional 75 km at full strength. Depletion makes the root up to 28 kg/m³ more buoyant than the non-cratonic continental endmember.
2. **Kinematic rifting:** craton strength multiplies the real tectonic extension field by a bounded resistance factor. The production floor is 0.28, so sufficiently persistent forcing can still rupture an old root.
3. **Late rift nucleation:** strong cells reduce the path score and slow maturation/thinning of a band that crosses them. Weak juvenile/orogenic corridors become preferred paths.
4. **Continental collision:** strong roots add a moderate drag multiplier at continent-continent convergent boundaries.

Continental material is still not explicitly subducted as a full 3-D buoyant slab. In v0.24, “resistance to subduction” is represented by thicker buoyant roots and stronger collision drag; a future continuum model would be needed for continental slab penetration, delamination and tearing.

## Physical basis and calibration boundary

The qualitative design follows observational and numerical results that cratonic mantle lithosphere is thick, compositionally buoyant, cold and strong, commonly exceeding 200 km, while rifts preferentially exploit weaker orogenic belts. Strong thermal/magmatic modification can nevertheless thin or destroy cratonic roots.

Primary research used as design guidance:

- Celli et al. (2020), *African cratonic lithosphere carved by mantle plumes*, Nature Communications: https://www.nature.com/articles/s41467-019-13871-2
- Heron et al. (2020), *Weak orogenic lithosphere guides the pattern of plume-triggered supercontinent break-up*, Communications Earth & Environment: https://www.nature.com/articles/s43247-020-00052-z

The numerical timescales and normalized strength law are prototype calibration parameters for this 0.5-Earth-mass moon. They are not claimed as direct measurements of lunar or terrestrial rheology.

## Tests and deterministic resume

Final suite: **142/142 tests passing** (the inherited 136 plus six v0.24-specific tests).

New tests cover:

- continental-only initialization;
- quiet maturation and strengthening;
- juvenile-volume dilution;
- rift rejuvenation;
- thicker/more buoyant roots and bounded extension resistance;
- exact transport by `material_source_index`.

A canonical subdivision-5 smoke run passed at **20,480 cells / 4 Myr**.

Checkpoint determinism compared:

- continuous `0 -> 12 Myr`, and
- `0 -> 4 Myr -> checkpoint -> resume -> 12 Myr`.

All **25/25 arrays** in `state.npz` are bitwise identical, including the three v0.24 fields. `meta.json` is exactly identical, including all four craton-history rows.

## Four-seed paired 500-Myr validation

Four subdivision-3 seeds (`20260806` through `20260809`) were run for 500 Myr with identical v0.23/v0.24 initial conditions.

| Metric | v0.23 mean | v0.24 mean |
|---|---:|---:|
| Final plate count | 11.25 | 11.75 |
| Topology events | 10.25 | 11.25 |
| Continental-material area | 28.586% | 28.594% |
| Mean continental mantle-root thickness | 106.19 km | 142.77 km |
| Maximum rift extension | 0.1688 | 0.0890 |
| Mean craton strength | — | 0.6398 |
| Cratonic share of continental material | — | 68.21% |
| Mean effective root age | — | 886.0 Myr |

The intended feedback is consistent in every seed: roots are thicker and maximum accumulated rift extension is lower. Plate-count changes have both signs (`13->15`, `11->7`, `10->12`, `11->13`), and the ensemble mean remains close. v0.24 therefore changes chaotic event timing without producing a one-directional frozen-plate regime in this sample.

Across all eight runs, the maximum absolute continental-ledger residual is `9.54e-7 km3`; maximum per-step sediment error is `5.96e-8 km3`. Per-seed values are in `V124_CRATON_PAIRED_500MYR.csv`.

## Detailed subdivision-4 500-Myr validation

The 5,120-cell production validation completed all 125 four-Myr steps and finalized the complete diagnostic map set.

| Quantity | Value |
|---|---:|
| Final plate count | 11 |
| Topology events | 15 |
| Largest plate fraction | 42.42% |
| Continental-material area | 28.36% |
| Mean continental mantle-root thickness | 133.56 km |
| Mean effective root age | 873.96 Myr |
| Mean mantle depletion | 0.4292 |
| Mean craton strength | 0.6354 |
| Cratonic share of continental material | 64.91% |
| Maximum rift extension | 0.1870 |
| Final sea level | -857.3 m |
| Final land fraction | 24.64% |
| Global continental-ledger error | `+4.77e-7 km3` |
| Maximum sediment step error | `2.98e-8 km3` |
| Maximum hydrosphere volume error | 11.87 km³ (~`1.1e-8` of inventory) |
| Elevation safety-rail clips | 0 |
| Flexure convergence | all steps converged |

The full machine-readable result is included as `demo_v124/summary_v124.json`; the craton maps and diagnostic history are included beside it.

## Performance note

The unit suite itself remains small (about seven seconds in the validation environment). The expensive operation is a geological integration: the finalized subdivision-4 500-Myr run advances 5,120 cells through 125 conservative transport, sediment-routing, topology, flexure and hydrosphere steps and took roughly 75 seconds in the current runtime. Deterministic segmented checkpoints remain the recommended way to run subdivision-5 or longer experiments.

## Status and next stage

v0.24 is considered **stable** for the current effective architecture. The new memory is transported and checkpointed exactly, its material dilution is explicit, mass conservation is unchanged, the intended mechanical signal is repeatable across seeds, and no systematic frozen-plate regime appears.

The most natural next geological stage is explicit **mantle-plume/metasomatic root modification**: couple localized mantle thermal anomalies and magmatism to depletion loss, root thinning and possible craton destruction instead of relying only on the existing supercontinent-heat and rift proxies.

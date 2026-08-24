# Moon Tectonics v0.25 — Mantle-Plume and Metasomatic-Root Findings

## Outcome

v0.25 is a working, deterministic extension of the stable v0.24 cratonic-memory model. It adds mantle-fixed plume heads without duplicating or rewriting the long v0.23/v0.24 integration loop. Moving lithospheric material crosses the fixed plume field and retains the resulting age, depletion, strength and mechanical-root changes after leaving it.

The implementation is numerically stable and checkpoint-exact. Its first four-seed coupled ensemble also exposes an important physical omission: local plume damage is clear, but the present model does not apply plume-generated extensional stress or explicitly nucleate a plume-centered rift. As a result, local weakening can reorganize the global plate system without consistently increasing fragmentation. This is a useful diagnostic result, not a calibration failure to tune away.

## Scientific basis and model scope

The v0.25 forcing follows four constraints from primary literature:

1. Stable depleted cratonic roots are difficult to erode by heating alone. Numerical models require substantial metasomatic loss of buoyancy/strength for rapid plume-assisted thinning ([Wang et al., 2015](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015GC005784)).
2. Progressive melt metasomatism, refertilization, densification and heating can make a craton vulnerable to later rifting ([Wenker & Beaumont, 2018](https://www.sciencedirect.com/science/article/abs/pii/S0040195117302597)).
3. Tomography and kimberlite evidence indicate plume-associated cratonic-root loss can occur concurrently or tens to roughly 100 Myr after plume interaction, and that not every plume-adjacent root is destroyed ([Celli et al., 2020](https://www.nature.com/articles/s41467-019-13871-2)).
4. Metasomatic modification may be channelized and persistent within 100–200 km-deep cratonic lithosphere rather than spatially uniform ([Roots et al., 2025](https://www.nature.com/articles/s41467-025-62912-6)).

The new model is an effective 2.5-D surface projection, not a 3-D mantle-convection, melt-migration or phase-equilibrium calculation. A plume is represented by a mantle-fixed unit-vector center, finite lifetime, smooth temporal envelope, Gaussian surface radius and normalized peak flux. Births and properties are generated independently from deterministic plume IDs, making the sequence insensitive to checkpoint boundaries.

Three coupled effects are applied only where continental material overlies the plume:

- effective lithosphere age decays exponentially with plume exposure;
- the depleted-root proxy decays through melt/fluid refertilization;
- mechanical root thickness is eroded at a bounded flux-dependent rate.

The first two effects immediately recompute transported craton strength. The existing v0.24 mechanical relaxation then moves density and thickness toward the newly weakened root target. No surface volcanism, dynamic plume uplift, plume heat-reservoir feedback or plume-generated extension is included yet.

## Production calibration used for the first experiment

| Parameter | Value |
|---|---:|
| Initial plume heads | 3 |
| Mean birth interval | 160 Myr |
| Lifetime range | 160–240 Myr |
| Gaussian one-sigma head radius | 650 km, ±20% |
| Peak normalized flux | 0.75–1.00 |
| Age-rejuvenation rate | 0.0040 Myr^-1 at unit flux |
| Refertilization rate | 0.0035 Myr^-1 at unit flux |
| Basal root-erosion rate | 0.35 km/Myr at unit flux |
| Minimum continental root | 60 km |

These are conservative effective parameters chosen to produce tens-of-Myr local modification without instant root deletion. They are not empirical measurements for the modeled moon.

## Verification

- Full suite: **147/147 tests passed**.
- New tests cover deterministic plume generation, spatial localization, coupled age/depletion/strength/root response, disabled-mode identity and exact repeated-step continuation.
- Checkpoint round-trip now includes plume population arrays, fixed-grid flux/exposure/erosion fields, birth chronology and diagnostic history.
- Canonical subdivision-5 smoke: 20,480 cells, `0 -> 4 -> resume -> 8 Myr`, with PNG output through Matplotlib's Windows-safe `Agg` backend.
- Exact integration check on subdivision 3:
  - continuous `0 -> 12 Myr` and segmented `0 -> 4 -> resume -> 12 Myr`;
  - `state.npz` SHA-256 identical: `8BF7239D5EDC6FE886FC431B2414FF4C58FA1A73872F72D7F2639EA8F19EBD8D`;
  - `meta.json` SHA-256 identical: `A24CDC4F5440E28817215B932FC3271399551C175E14FEA57C6A71ED2803C2D0`.
- Four paired subdivision-3 runs, 500 Myr each, use identical initial plate geometry within each control/plume pair.
- All four v0.25 runs have zero numerical elevation clips and final absolute continental-ledger error `<=9.54e-7 km3`.

## Four-seed 500-Myr ensemble

The full table is in `V125_PLUME_PAIRED_500MYR.csv`; the paired plot is `V125_PLUME_PAIRED_500MYR.png`.

| Final/accumulated metric | v0.24 mean | v0.25 mean | Mean change |
|---|---:|---:|---:|
| Mean craton strength | 0.6222 | 0.6315 | +0.0092 |
| Cratonic share of continental material | 59.45% | 65.89% | +6.45 percentage points |
| Mean continental mantle-root thickness | 137.81 km | 146.96 km | +9.16 km |
| Maximum rift extension | 0.1262 | 0.1725 | +0.0463 |
| Final plate count | 14.75 | 11.50 | -3.25 |
| Mean surface plume exposure | — | 4.94 Myr | — |
| Maximum cell exposure | — | 163.44 Myr | — |
| Maximum local imposed root erosion | — | 33.42 km | — |
| Summed global-mean imposed root erosion | — | 1.35 km | — |

The direct response has the intended sign everywhere it is applied: plume exposure reduces age, depletion, craton strength and root thickness locally. Maximum local cumulative erosion ranges from 21.3 to 47.0 km across seeds.

The fully coupled final state does not have a single monotonic global response. Final mean strength changes range from -0.022 to +0.041; maximum rift-extension changes range from -0.071 to +0.153. All four plume runs do, however, end with fewer plates, and all four have greater global mean continental-root thickness despite local erosion.

## Interpretation

The counterintuitive global root thickening is an emergent trajectory effect, not a sign error in the local plume operator. Once plume damage changes strength, root buoyancy and plate forces, topology histories diverge. The current model can erode a root that happens to cross a plume but cannot create a new plume-centered divergent boundary. Large unexposed continental domains can therefore continue maturing while the changed dynamics favor plate coarsening and mergers. Their increased contribution to the final weighted mean can exceed the direct local loss.

The robust first result is therefore:

> Metasomatic/root weakening alone is sufficient to alter the global tectonic trajectory, but it is not sufficient in this model to reproduce systematic plume-driven fragmentation.

This distinction is consistent with the literature: weakening makes cratons vulnerable, while removal or rifting also depends on imposed deformation, convection and pre-existing structure.

## Recommended v0.26 question

The next controlled experiment should add a separate plume-head mechanical forcing rather than retune the verified v0.25 weakening rates:

1. derive a radial/annular extensional field from plume-head flux and local root-thickness contrast;
2. pass it through the existing external continental-extension input before breakup;
3. add plume magmatic productivity and dynamic uplift as separately diagnosed fields;
4. compare `weakening only`, `forcing only`, and `combined` ensembles;
5. require ensemble robustness before promoting any plume calibration to the production moon.

That experiment will determine whether the present reduction in plate count is a real consequence of the modeled moon's regime or simply the expected result of omitting active plume-driven rifting.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_long_evolution_v125.py --config configs\canonical_moon.yaml --end-time 500 --dt 4 --output results\v125_canonical --finalize
.\.venv\Scripts\python.exe analysis\compare_v125_ensemble.py
```

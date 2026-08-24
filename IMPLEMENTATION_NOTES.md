# v0.23 implementation notes

- Added `tectonics/sediment.py` and `visualization/sediment.py`.
- Added per-cell `sediment_volume_km3` transported with the winning lithospheric parcel.
- Lost sediment parcels enter a persistent deep-recycled sediment reservoir rather than disappearing.
- Surface erosion now removes tracked continental bedrock volume and routes the same mass conservatively downhill.
- Deposition uses the actual previous solved sea level; a soft 10-km burial spill redistributes overthick deposits without clipping mass.
- Sediment geometry and density contribute to material-aware topography and flexural loading.
- Legacy topographic erosion is disabled in the v0.23 production runner to avoid double erosion.
- Added explicit rift-recycled material for continental thinning and final breakup.
- Fixed material-aware rift and arc generation/thickening diagnostics exposed by the new global ledger.
- Checkpoint format `0.23-conservative-sediments` stores sediment state/budget/history.
- Validation: 136/136 tests; 22/22 arrays + exact metadata on checkpoint resume; canonical sub5 smoke; 500-Myr sub4 ledger residual ~9.5e-7 km3; paired four-seed sub3 plate-count mean 11.75 -> 11.75.

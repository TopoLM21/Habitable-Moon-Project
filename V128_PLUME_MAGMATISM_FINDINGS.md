# Moon Tectonics v0.28 — Permanent Plume Magmatism and Volcanic Tracks

## Result

v0.28 converts the v0.26 decompression-melt productivity diagnostic into
permanent igneous material. Each step partitions extracted melt into extrusive
basalt, crustal dykes/sills and mafic underplate. These physical volumes move
with the winning lithospheric parcel while the plume source remains fixed in
the mantle, leaving an age-ordered volcanic track as the plate moves away.

The new material is independent of the reversible v0.27 dynamic-topography
field and of whether v0.26 mechanical plume forcing is coupled into the rift
solver. This permits a genuine uplift × magmatism × rifting factorial.

## Physical basis and calibration

White and McKenzie (1989) showed that decompression melting of anomalously hot
mantle beneath stretched lithosphere produces both flood basalts and large
intrusive additions to or beneath the crust. Crisp's (1984) global compilation
found intrusive/extrusive ratios typically near 10:1 in continental settings.
The default partition therefore assigns 9% to extrusive basalt and 91% to
intrusions, divided into 21% dykes/sills and 70% underplate.

The v0.28 ceiling of `0.018 km/Myr` at unit productivity is a coarse-cell,
time-averaged emplacement rate, not an instantaneous eruption rate. It is
modulated by the v0.26 productivity field, a `0.025` productivity threshold,
an extraction factor of `0.25 + 0.75 × plume_extension`, and a smooth 25-km
local accommodation ceiling. This keeps the long calculation compatible with
the observation that most LIP volume is emplaced in short 1–5 Myr pulses while
the numerical step is 4 Myr and the effective plume head persists longer.

Density-aware isostasy uses `2900 kg/m3` for surface basalt, `2950 kg/m3` for
dykes/sills, `3050 kg/m3` for mafic underplate and `3300 kg/m3` for mantle.
The underplate density is consistent with the `3.05 g/cm3` high-velocity mafic
lower-crustal layer inferred in recent seismic/gravity models. Extrusive basalt
adds geometric thickness and a flexed downward load; intrusions replace mantle
and provide positive density-contrast support.

Primary sources:

- White and McKenzie (1989), *Magmatism at rift zones: The generation of
  volcanic continental margins and flood basalts*,
  <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JB094iB06p07685>.
- Crisp (1984), *Rates of magma emplacement and volcanic output*,
  <https://www.sciencedirect.com/science/article/pii/0377027384900398>.
- Bryan and Ernst (2008), *Large Igneous Provinces (LIPs): Definition,
  recommended terminology, and a hierarchical classification*,
  <https://doi.org/10.1016/j.earscirev.2007.07.005>.
- Mittal et al. (2021), *The Magmatic Architecture of Continental Flood
  Basalts I: Observations From the Deccan Traps*,
  <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB021808>.
- Zhou et al. (2024), *Crustal melting and continent uplift by mafic
  underplating at convergent boundaries*,
  <https://www.nature.com/articles/s41467-024-53435-7>.

## Implementation

`tectonics/plume_magmatism.py` owns five transported fields: the three volume
reservoirs, age since last emplacement and current emplacement productivity.
The same material-source index used by conservative sediment transport moves
the reservoirs. If a source is duplicated its volume is split among targets;
if it disappears its full remaining volume is added to the corresponding
deep-recycling counter.

The global ledger is

`cumulative generated = surface reservoirs + deep recycled`.

No hard material clipping is used. Local accommodation reduces future
production as the 25-km ceiling is approached, so it cannot create or destroy
existing volume. Checkpoints store all five grids, all six cumulative
generated/recycled counters and the complete diagnostic history.

`tectonics/topography.py` now accepts optional extrusive thickness, extrusive
load and intrusive-support fields. All default to exact zero, preserving old
versions. The permanent magmatic terms are flexed as crustal loads/support;
v0.27 convective support remains a separate non-flexed background anomaly.

Dedicated output includes reservoir/thickness/support maps, a transported
track-age map, volume/ledger histories and a four-panel plate / productivity /
permanent-thickness / total-surface GIF.

## Verification

- `161/161` tests pass.
- Tests cover partitioning, the disabled control, conservative duplicate/loss
  transport, deep recycling, ledger closure, density-aware topography,
  deterministic continuation and checkpoint round-trip.
- Continuous `0→12 Myr` and `0→8→resume→12 Myr` runs are byte-identical for all
  45 checkpoint arrays and exactly equal in complete JSON metadata.
- State SHA-256:
  `D1053E199821AA189BACF4AB9263D9FCB25EFD8492C5660854546EF39AFDCBCF`.
- Metadata SHA-256:
  `DD8836C960E9818AE9B941C0651F41AECE892E45C97EE328691AB10C6D47BD01`.
- The 40-Myr smoke produces 1.177 million km3 of retained igneous material,
  reaches 0.255 km maximum added thickness and closes its ledger to
  `1.89e-10 km3`.
- All eight 500-Myr factorial worlds have zero elevation safety-rail clips.
- Maximum absolute 500-Myr igneous-ledger error is `1.86e-8 km3`.

## 2×2×2 factorial at 500 Myr

One subdivision-3 seed was run in all combinations of dynamic topography (d),
permanent magmatism (m) and mechanical plume-rift coupling (r). Root weakening
and the mantle plume chronology remain common to all eight worlds.

| Metric | Magmatism off | Magmatism on | v0.28 range/effect |
|---|---:|---:|---:|
| Cumulative generated igneous volume | 0 | 31.24–31.39 million km3 | permanent source |
| Final surface igneous volume | 0 | 9.04–9.33 million km3 | +9.13 million km3 main effect |
| Deep recycled igneous volume | 0 | 21.91–22.35 million km3 | explicitly closed sink |
| Maximum added igneous thickness | 0 | 2.594–2.601 km | +2.598 km main effect |
| Maximum local Airy-limit support | 0 | 223.6–224.3 m | density-aware |
| Maximum-elevation main effect | — | — | +1.7 m |
| Land-area main effect | — | — | +0.015 pp |
| Plate-count main effect | — | — | 0.0 |

The small global topographic main effect is expected: the 224-m maximum
magmatic support is localized and does not coincide with the world's highest
mountain, while erosion, flexure and sea-level adjustment respond nonlinearly.
This increment intentionally does not yet heat or weaken the mechanical crust,
so matching magmatism-on/off pairs retain the same plate count.

For context, the dynamic-topography main effect in this single-seed factorial
is +427 m maximum elevation and +0.69 percentage points land area. Mechanical
plume-rift coupling raises maximum rift extension from about `0.105` to
`1.17–1.18` and has the largest coupled shoreline/topology effect. These are
single-seed sensitivity contrasts, not ensemble population estimates.

The complete rows are in `V128_PLUME_MAGMATISM_FACTORIAL_500MYR.csv`; the
comparison plot is `V128_PLUME_MAGMATISM_FACTORIAL_500MYR.png`.

## Next research step

v0.29 should separate the broad, short-lived LIP-producing plume head from a
narrower persistent tail. That will turn the current transported swaths into
hotspot chains with explicit along-track age progression. The same increment
should add independently switchable magmatic heating/weakening, dike
localization along active rifts, and cooling/eclogitization of old underplate,
followed by multi-seed and subdivision-4 convergence tests.

# Moon Tectonics v0.27 — Transient Plume Dynamic Topography

## Result

v0.27 converts the v0.26 diagnostic uplift proxy into delayed, reversible
surface relief. The field is convective mantle support: it shifts equilibrium
topography and therefore affects the solved ocean, coastlines, erosion and
sediment routing, but it is not flexed as a surface load and never adds crustal
material.

The production field has zero area-weighted mean at every step. A positive
plume swell is balanced by a weak distributed negative anomaly, removing the
spherical degree-zero component and preventing a change in mean planetary
radius. In the 40-Myr smoke, local realized support reaches `890 m`, the
compensating minimum is only `-20 m`, global RMS relief is `106 m`, and the
numerical displacement-volume residual is `1.2e-9 km3`.

## Physical basis and calibration

Dynamic topography from plume buoyancy is broad, time-dependent and uncertain.
Gurnis et al. (2000) obtained examples near `500 m`, models leveling near
`700 m`, and an isolated rising plume increasing from roughly `490` to
`1000 m` over 15 Myr. Hartley et al. (2011) reconstructed three transient
`200–400 m` uplift steps followed by reburial. Conversely, Peate and Bryan
(2008) showed that the often-assumed `500–1000+ m` pre-volcanic dome is not
unambiguously present in the Emeishan geological record. The v0.27 default is
therefore a ceiling rather than a guaranteed amplitude:

- maximum instantaneous central support: `1000 m`;
- plume-flux saturation: `0.75`;
- response time: `8 Myr`;
- decay time after weakening support: `20 Myr`;
- mapped positive-support threshold: `50 m`;
- numerical anomaly rail: `±1200 m`.

Primary sources:

- Gurnis et al. (2000), *Constraining mantle density structure using geological
  evidence of surface uplift rates: The case of the African Superplume*,
  <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/1999GC000035>.
- Hartley et al. (2011), *Transient convective uplift of an ancient buried
  landscape*, <https://www.nature.com/articles/ngeo1191>.
- Peate and Bryan (2008), *Re-evaluating plume-induced uplift in the Emeishan
  large igneous province*, <https://www.nature.com/articles/ngeo281>.
- Moucha and Forte (2011), *Changes in African topography driven by mantle
  convection*, <https://www.nature.com/articles/ngeo1235>.

The response and decay times are effective surface-model parameters, not a
claim that every real plume follows one relaxation law.

## Implementation

`tectonics/plume_dynamic_topography.py` converts the mantle-fixed plume-flux
field into an instantaneous target, removes its area mean and evolves a
checkpointed realized anomaly using exact exponential response factors. The
realized field enters `tectonics/topography.py` as a non-flexed background
term. It is included before the hydrosphere solve and the conservative sediment
step, so shoreline migration and surface-process feedbacks are explicit.

The checkpoint stores target relief, realized relief, cumulative positive
support and the complete diagnostic history. `run_long_evolution_v127.py`
wraps v0.26 without altering the mechanical plume-rifting calibration.

Dedicated outputs are:

- target, realized and cumulative-support maps;
- time histories of uplift, subsidence, RMS relief and vertical rate;
- a three-panel plate / dynamic-topography / total-surface GIF;
- summary diagnostics including the global displacement-volume residual.

## Verification

- `156/156` tests pass.
- Tests cover localization, zero area mean, delayed response, reversible decay,
  the disabled control, non-mutation of lithospheric material, deterministic
  continuation and checkpoint round-trip.
- Continuous `0→12 Myr` and `0→4→resume→12 Myr` calculations are byte-identical:
  state SHA-256
  `ECB7FA095C285416341B91E70D8BEDC6BF1C060B08E993F12FF953B9CC9C27E9`
  and metadata SHA-256
  `6DB9D9B6C6ED526DE417071815F6DC8FD2A52109C74AA7971E343A160D962329`.
- The 40-Myr smoke produces six GIFs, including the new dynamic-topography
  animation and the retained v0.26 plume-rifting animation.
- All eight paired 500-Myr runs have zero elevation safety-rail clips.
- Maximum absolute final continental-ledger error is `1.43e-6 km3`.
- Maximum zero-mean displacement-volume residual is `1.91e-9 km3`.

## Paired four-seed 500-Myr ensemble

Each pair shares the same initial plates, mantle plume chronology, mechanical
plume rifting and root weakening. Only transient dynamic topography changes.

| Metric | No dynamic topography | Transient plume support | Paired mean change |
|---|---:|---:|---:|
| Maximum surface elevation, m | 1734.9 | 1969.8 | +234.9 |
| Final land area, % surface | 20.28 | 20.70 | +0.42 pp |
| Final solved sea level, m | -743.6 | -747.1 | -3.5 |
| Cumulative bedrock erosion, million km3 | 446.56 | 443.06 | -3.50 |
| Final surface sediment, million km3 | 143.38 | 142.04 | -1.35 |
| Final plate count | 12.50 | 12.25 | -0.25 |

The direct imposed field is stable and bounded: mean maximum realized uplift
over the four dynamic runs is `994.6 m`. The 500-Myr coupled differences are
not simple static offsets. Dynamic relief changes shorelines and sediment
loads; in two seeds the later topology remains identical, while two seeds
eventually follow different nonlinear trajectories. The mean elevation and
land-area changes are strongly influenced by seed `20260806`, so this ensemble
is a sensitivity test, not a calibrated prediction for one planet.

Per-seed values and ensemble statistics are in
`V127_DYNAMIC_TOPOGRAPHY_PAIRED_500MYR.csv`; the paired comparison is
`V127_DYNAMIC_TOPOGRAPHY_PAIRED_500MYR.png`.

## Next research step

The next increment should make plume magmatism explicit while keeping it
independently switchable from dynamic support and mechanical extension:

1. decompression-melt productivity from the existing diagnostic field;
2. separate dyke, extrusive basalt and underplate reservoirs;
3. a closed igneous-volume ledger and density-aware crustal thickening;
4. transported plume-track age and cumulative volcanic-volume diagnostics;
5. a factorial ensemble isolating uplift, magmatism and mechanical rifting.

Dynamic topography should remain reversible; only the later magmatic module
will leave permanent new igneous crust.

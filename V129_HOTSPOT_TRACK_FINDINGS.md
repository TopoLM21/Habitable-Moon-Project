# v0.29 — Plume heads, tails and age-progressive hotspot tracks

## Research question

Can a short, broad plume head followed by a narrow, persistent tail create
age-progressive volcanic tracks while preserving the exact material ledgers and
checkpoint reproducibility of v0.28?

The experiment follows the classic plume-head/plume-tail interpretation of
flood-basalt provinces and hotspot chains (Richards et al., 1989,
<https://doi.org/10.1126/science.246.4926.103>). It still treats the deep
source as stationary in this version; observed hotspot motion motivates v0.30
(Konrad et al., 2018, <https://doi.org/10.1038/s41467-018-03277-x>).

## Model added in v0.29

- A broad head acts during the first `0.18` of a plume lifetime
  (`29–43 Myr` for the configured lifetimes). Its radius is `520–780 km`.
- A narrow tail has `0.23` of the head radius and `0.55` of its flux scale.
  Both Gaussian components are area-normalized, making their integrated source
  strength much less sensitive to grid subdivision.
- Transported plume heat relaxes over `90 Myr` and can add at most `0.32` to
  the local extension forcing.
- Between rift extension `0.18–0.70`, up to 30 percentage points of melt are
  transferred conservatively from underplate to localized dykes, consistent
  with extension-focused dyke emplacement (Abebe et al., 2004,
  <https://doi.org/10.1016/S0377-0273(03)00318-4>).
- Underplate begins eclogitizing after `120 Myr`, with a `120 Myr` transition
  and `180 Myr` relaxation, capped at 65%. Density rises from `3050` to
  `3450 kg/m3`; material above a 42% eclogite fraction can delaminate on a
  `120 Myr` timescale into the existing deep-recycled ledger. This is an
  intentionally reduced representation of lower-crustal eclogitization and
  delamination (Li et al., 2017, <https://doi.org/10.1002/2016JB013106>), while
  allowing long-lived dense lower crust (Kukkonen et al., 2021,
  <https://doi.org/10.1038/s41467-021-26878-5>).

Each new mechanism is independently switchable. The legacy branch therefore
remains a controlled comparison rather than a separate model.

## Numerical verification

- `167/167` tests pass.
- A continuous 24-Myr run and a 12+12-Myr checkpoint/resume run are exactly
  identical in all 53 state arrays and JSON metadata.
- A canonical subdivision-5 smoke run completed on 20,480 cells.
- The 120-Myr demonstration produced a 16-frame six-panel GIF.
- Ten 500-Myr validation worlds completed: four paired seeds at subdivision 3
  plus one paired seed at subdivision 4.
- Maximum absolute igneous-ledger error: `1.86e-8 km3`.
- Maximum absolute continental-ledger error: `9.54e-7 km3`.
- Elevation safety-rail clips: zero in every validation world.

## Paired 500-Myr result

Means below use the four subdivision-3 seeds. `Full − legacy` is paired by
seed.

| Quantity | Legacy | Full v0.29 | Paired effect |
|---|---:|---:|---:|
| Generated igneous volume (million km3) | 23.957 | 4.351 | -19.606 |
| Surface igneous volume (million km3) | 6.184 | 1.030 | -5.154 |
| Head-generated volume (million km3) | — | 3.901 | — |
| Tail-generated volume (million km3) | — | 0.450 | — |
| Maximum rift intensity | 0.784 | 0.344 | -0.440 |
| Maximum elevation (m) | 1971.3 | 2074.9 | +103.6 |
| Land fraction (%) | 20.718 | 21.556 | +0.838 pp |
| Plate count | 12.25 | 13.50 | +1.25 |
| Final age–distance correlation | +0.339 | +0.752 | +0.413 |

For the full model, the mean peak transported thermal anomaly is `0.146`, the
mean maximum dyke-localization factor is `0.377`, the final eclogitized
underplate volume is `0.066 million km3`, and `2023 km3` has delaminated.

The large reduction in generated volume is deliberate: v0.28 applied a broad
plume kernel throughout each event, whereas v0.29 separates a short broad head
from a narrow lower-flux tail and normalizes component integrals. This is a
change in source interpretation, not a material-ledger loss.

## Resolution check and limitation

For the paired seed `20260806`, integrated full-model source generation changes
from `5.770` to `5.866 million km3` between subdivisions 3 and 4 (`1.65%`), and
head generation changes by about `0.36%`. The area-normalized source therefore
meets the intended grid-convergence target.

The final retained surface volume (`1.934` versus `1.034 million km3`) and late
plate topology (`15` versus `8` plates) do not converge at those two
resolutions. The legacy control is also topologically divergent (`15` versus
`9` plates), indicating nonlinear late plate evolution and material fate rather
than a residual plume-source normalization error. These quantities should not
yet be interpreted as resolution-independent predictions.

## Next: v0.30

Replace the stationary source assumption with deterministic, checkpointed
plume-conduit drift. Fixed-source and mobile-source twins will separate plate
motion from mantle-source motion, quantify path curvature, and test whether the
age-progressive tracks survive realistic source mobility.

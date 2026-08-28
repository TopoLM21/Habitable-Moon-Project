# Moon Tectonics v0.26 — Plume-Driven Mechanical Rifting

## Result

v0.26 closes the main gap identified in v0.25: mantle plumes can now apply a
mechanical extensional load to the continental lithosphere instead of only
rejuvenating, refertilizing and eroding its mantle root. The new load is
independently switchable, deterministic and checkpointed. It enters the
existing progressive rift solver as an external extension field; it does not
directly edit crust type, create plate boundaries or force a breakup event.

In the four-seed, four-mode subdivision-3 ensemble, mechanical plume forcing
raises the mean final maximum rift extension from `0.126` in the control to
`0.797` in forcing-only runs and `0.839` in combined runs. Breakup occurs in
two of four forcing-only and two of four combined worlds, but in none of the
control or weakening-only worlds. The remaining mechanically forced cases
retain strong aborted rifts, which is preferable to making every plume a
guaranteed plate-boundary generator.

## Physical basis and model boundary

The effective forcing represents two first-order contributions:

- broad extension from plume-head radial flow and uplift-related gravitational
  potential energy;
- enhanced localization near the plume-head flank, with a modest boost across
  neighboring mantle-root thickness contrasts.

This follows the qualitative mechanical picture in Westaway (1993), where
radial plume-head flow provides viscous force and plume-related uplift provides
a buoyancy force for extension, and the later thermomechanical modeling of
Koptev and coauthors. Those models show that plume duration/buoyancy and the
state of the lithosphere determine whether active rifting succeeds or aborts.

Primary sources:

- Westaway, R. (1993), *Forces associated with mantle plumes*, Earth and
  Planetary Science Letters, <https://doi.org/10.1016/0012-821X(93)90142-V>.
- Koptev et al. (2018), *Contrasted continental rifting via plume-craton
  interaction: Applications to Central East African Rift*, Tectonophysics,
  <https://www.sciencedirect.com/science/article/pii/S0040195117301245>.
- Koptev et al. (2026), *Successful and Failed Continental Rifts: The Role of
  Mantle Plume-Lithosphere Interactions*, Journal of Geophysical Research:
  Solid Earth, <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JB033048>.

The implementation remains an effective surface projection, not a
momentum-conserving mantle-convection calculation. Dynamic uplift and magmatic
productivity are currently diagnostics only and do not yet modify topography or
add crustal volume.

## Implementation

`tectonics/plume_rifting.py` maps the v0.25 plume-flux field to a bounded
extension field. Production parameters use a smooth core response from flux
`0.10` to `0.75`, a Gaussian flank-localization term centered at flux `0.35`,
and a maximum forcing of `1.0`. The local field is combined with rollback
back-arc forcing using the maximum of the two, avoiding additive double
counting where they overlap.

Two independent switches define four validation modes:

| Mode | Plume chronology | Root weakening | Mechanical rifting |
|---|---:|---:|---:|
| Control | off | off | off |
| Weakening only | on | on | off |
| Forcing only | on | off | on |
| Combined | on | on | on |

The fixed-grid checkpoint adds the last extension, cumulative extension
impulse, diagnostic uplift and diagnostic magmatic-productivity fields, plus
their complete history rows. The runner is `run_long_evolution_v126.py`.

## Verification

- `151/151` tests pass, including localized forcing, the four independent
  modes, deterministic continuation and checkpoint round-trip coverage.
- A continuous `0→12 Myr` run and `0→4→resume→12 Myr` run are byte-identical:
  state SHA-256
  `029AF48A3D218DD198B8D794F29EF0B2405CAD258477E2EE1942EF272D948966`
  and metadata SHA-256
  `4E8349C5D83E1498054846FBD5392C3F16A422004A4039B3D19B8570203BEFB6`.
- The 40-Myr smoke creates all five local GIF products, including the new
  three-panel `plume_rift_history.gif`.
- All sixteen 500-Myr ensemble runs record zero elevation safety-rail clips.
- The worst absolute final global continental-ledger error is
  `1.43e-6 km3`.

## Four-mode 500-Myr ensemble

Values are means over seeds `20260806`–`20260809`.

| Mode | Max rift extension | Breakup, million km² | Thinning, million km³ | Final plates | Mean root, km | Mean craton strength |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0.126 | 0.000 | 96.3 | 14.75 | 137.8 | 0.622 |
| Weakening only | 0.173 | 0.000 | 84.5 | 11.50 | 147.0 | 0.631 |
| Forcing only | 0.797 | 0.188 | 138.1 | 14.50 | 134.8 | 0.559 |
| Combined | 0.839 | 0.260 | 145.9 | 12.50 | 142.6 | 0.566 |

The paired mechanical increment is remarkably similar in the two relevant
comparisons: `+0.671` maximum rift extension for forcing-only minus control,
and `+0.667` for combined minus weakening-only. Mechanical forcing also adds
about `42–61 million km3` of mean cumulative continental thinning. Root
thickness and plate count are not monotonic because changed rifting reorganizes
later plate interactions; they should not be interpreted as direct local plume
responses. The direct local diagnostics behave as intended: forcing-only has
zero imposed root erosion, whereas combined runs reach `35.5 km` mean maximum
local imposed root erosion.

Per-seed values and ensemble statistics are in
`V126_PLUME_RIFTING_4MODE_500MYR.csv`; the comparison figure is
`V126_PLUME_RIFTING_4MODE_500MYR.png`.

## Next research step

The next model increment should connect the already recorded diagnostic
magmatic-productivity and dynamic-uplift fields to explicit observables, while
keeping both independently switchable:

1. transient dynamic topography with zero long-term mass creation;
2. dyke/underplate-driven crustal addition with a closed igneous-volume ledger;
3. plume-track diagnostics in material coordinates;
4. a paired ensemble separating uplift, magmatism and mechanical extension.

That sequence will test whether rift success in this model depends on the same
competition between plume lifetime, lithospheric strength and magmatic
weakening seen in the thermomechanical literature, without hiding those
effects inside one combined tuning parameter.

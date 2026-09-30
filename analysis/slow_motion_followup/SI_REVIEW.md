# Independent SI closure review

Reviewed `tectonics/young_plate_dynamics.py` against dimensions, an analytic
whole-sphere solution, an independent velocity-space least-squares oracle,
mechanical power, frame covariance, relabeling, and actual geometric forces.

`tests/test_young_plate_dynamics.py`: **16 passed**. The earlier combined run of
the initial SI tests, shared basal tests, and all starter tests had **56 passed**.

## Verified equations

For beta in Pa s/m, radius R in m and cell area A in m2:

- `D = beta R^2 sum A(I-rr^T)` has units N m s.
- `b = beta R sum A(r cross u)` has units N m.
- The solve returns rad/s; multiplying by seconds/Myr gives stored rad/Myr.
- On a complete symmetric sphere `D = (8 pi / 3) beta R^4 I`; the code matches.
- A rigid imposed flow is recovered without the legacy memory fraction or speed
  cap. Mixed flows match the independently stacked velocity design matrix.
- Target torque residual is below 5e-15 of driving torque in the oracle test.
- The returned transient still has torque residual because the retained temporal
  relaxation filters the equilibrium target. The trace distinguishes the two.
- Mechanical power closes with basal, ridge and slab input minus basal drag
  dissipation equal to transient net torque dotted with angular velocity.
- Common rotation added to BOTH mantle and plates preserves relative motion
  and slip; no plate-only gauge subtraction is applied.
- Plate relabeling leaves the physical solution unchanged.
- Uniform thermal geometry gives zero ridge moment even if ages differ. Actual
  contrast produces continuous forces independently of display thresholds.
- Accepted slab moment is `R*g*buoyancy_moment_kg`; reducing accepted buoyancy by
  1e-8 reduces its target contribution by 1e-8, with no active-length denominator.

Two review findings were fixed by the module owner: reject nonpositive/nonfinite
timesteps, and include ridge/slab/total power in the trace.

## Explicit physical limits that tests do not remove

The new mode solves the chosen constitutive model, not a complete mantle/plate
mechanics problem. Beta is a prescribed constant. Slab viscous resistance,
bending resistance, continental collision and transform friction do not enter
this new torque solve. Mantle source amplitude/pattern remain prescribed.

Ridge force uses the boundary cell against a whole-plate oceanic mean as a flank
surrogate. This is nonlocal and not a resolved pressure gradient across an
observed ridge profile. Young age and a thermal contrast identify candidates;
their existence does not by itself validate the hydrostatic approximation.

Slab force transmits the FULL current positive buoyancy weight as tangential pull. This
is explicitly an upper-bound closure. Radial gravity itself has zero moment
about the planet center; actual plate torque requires redirection of slab stress
through its geometry and interaction with mantle. Curvature, slab/bending
drag and force transmission are not solved. Accumulating attached mass can
therefore produce large velocities while every algebraic torque test passes.
Long-run experiments must report this distinction and monitor accumulated
accepted mass, attached geometry, and velocities.

## Actual upper-bound experiment and default decision

The initial opt-in full-transmission experiment, before thermal cohorts were
added (`young-boundary-material-1`), reached 50 and 100 Myr after
partition. These are actual captured dynamics steps, not counterfactual fields.
At 50 Myr the final reported post-transport mean was 16.9611 km/Myr, but the
equilibrium target was already 86.0379. The retained 45-Myr filter concealed a
large and growing force target behind a relatively moderate current velocity.

| Time after partition | Basal component mean | Ridge component mean | Slab component mean | Total target mean | Returned before transport |
|---:|---:|---:|---:|---:|---:|
| 50 Myr | 10.5792 | 1.0804 | 74.6673 | 86.0379 | 16.9468 |
| 100 Myr | 7.0445 | 7.9903 | 429.0320 | 442.4766 | 201.8742 |

Units are km/Myr = mm/year. Component norms are not arithmetically additive.
At 100 Myr the final post-transport mean was 196.6209 km/Myr. Slab mechanical
power rose from 54.505 GW at 50 Myr to 3.171 TW at 100 Myr, while basal source
power was respectively 6.937 and 48.870 GW. Slab moments at 100 Myr were
4.7–6.0e26 N m per plate; basal moments were about 8.5e24 N m.

Cumulative accepted oceanic volume increased from 7.636e6 to 350.483e6 km3.
Attached buoyancy excess increased from 1.679e19 to 1.3945e20 kg. At 50 Myr
there were 86 attached and 2 detached segments; at 100 Myr, 475 attached and
1712 detached. Attached inventory is therefore not strictly monotonic: remapping
can lose the material anchors and detach entries. This geometric/raster culling
is not a thermal-equilibration or slab-resistance law. Continental-collision
breakoff cannot regularize this wholly oceanic experiment.

**Decision:** `young_slab_force_model="disabled_pending_closure"` is the default.
It retains accepted material/slab inventory and diagnostics but contributes no
unvalidated slab force. `"full_transmission_upper_bound"` remains an explicit
experiment. Tests now request that upper-bound option explicitly; a new default
test verifies zero applied slab force and unchanged stored accepted inventory.
Root is rerunning the corrected basal + thermal-contrast ridge path separately.

This is a physical-model boundary, not a speed-tuning parameter. Before promoting
slab force to a default, attached parcels need a finite thermal/geometry history,
an attachment/detachment model and a justified transmission/resistance closure.
Existing thermal diffusivity can inform passive warming, but a convenient decay
time, arbitrary viscosity or velocity cap should not be introduced to suppress
this acceleration. Positive source/drag power balance alone does not validate
the physical source of the large slab power.

## Subsequent thermal-cohort refinement

`young-boundary-material-2` now records accepted cohorts, warms their thermal
buoyancy with the existing diffusivity, and transfers the oldest attached material
beyond the physical mantle-depth capacity into a retained deep ledger. The
previous 50/100-Myr figures are NOT measurements of this later refinement.

The heating equation is the mean temperature deficit of an initially uniform
finite sheet of full thickness H with both faces held at a fixed bath:

`f = (8/pi^2) sum_{n odd} exp(-pi^2 n^2 kappa t/H^2)/n^2`.

Its short-time heat-content limit `1-4 sqrt(kappa t/(pi H^2))` is correct for two
faces. The switch at Fourier number 0.001 has negligible exponential overlap
error. Independent evaluation against a converged Fourier series gave maximum
absolute error below 1e-14 over Fourier numbers 1e-7 to 10. No fitted cooling
timescale is introduced. The slab-force test fixture now includes a real cohort
evaluated at its acceptance time, so its analytic dimensional check has f=1.

This remains a passive equivalent-uniform-sheet approximation: the initial
through-slab temperature profile, changing bath temperature, bending, viscous
resistance and the conversion of radial buoyancy into tangential plate force
are not resolved. Cohort heating makes the buoyancy history finite but does
not validate full tangential transmission. The default therefore remains
`disabled_pending_closure`; the full-transmission mode remains explicit.

# Physical scope of explicit oriented-fault admission

2026-10-01. The implemented law is a **local forced-underthrust admission
criterion under prescribed loading**. It does not establish self-sustained
subduction, predict a fault's dip, solve plate bending, or infer a missing
three-dimensional stress tensor from scalar material damage.

## Primary sources and the claims they support

- [Toth and Gurnis (1998), Dynamics of subduction initiation at preexisting
  fault zones](https://authors.library.caltech.edu/records/ev1w2-6v917),
  DOI 10.1029/98JB01076: their numerical experiments prescribe a dipping weak
  fault and investigate compression with force or velocity boundary conditions.
  This supports explicitly distinguishing inherited geometry and loading from
  an initiation result. Their numerical thresholds are specific to their
  experiments; none are copied into this implementation.
- [Gurnis, Hall and Lavier (2004), Evolving force balance during incipient
  subduction](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003GC000681),
  DOI 10.1029/2003GC000681: section 3 uses a Coulomb law with cohesion, friction,
  normal stress and pore pressure. Their calculations include elastic bending,
  plastic failure, viscous flow and heat transport. Sections 66–67 distinguish
  fault resistance from the bending resistance that must also be overcome.
  Therefore a local frictional yield decision alone cannot establish sustained
  subduction or determine its eventual speed.
- [Byerlee (1978), Friction of rocks](https://pubs.usgs.gov/publication/70012542),
  DOI 10.1007/BF00876528: experimental rock friction depends on normal stress
  and, especially at lower stress, surface properties and gouge. This is a
  reason to require explicit friction/cohesion parameters with provenance,
  rather than present one coefficient as universal for every boundary.

## Implemented reduction and units

The following local equations are a direct tensor projection and Coulomb
reduction used by this project, not a reproduction of any cited simulation.

`Sigma_eff` is a finite symmetric, **compression-positive**, effective stress
tensor in global Cartesian coordinates at the contact midpoint, in Pa.
The explicit spatial boundary model is
`contactwise_parallel_transport_from_midpoint`: along the shared great circle,
the tensor is rotated about the contact normal together with the radial and
fault basis. Dip, cohesion and friction are uniform on the contact. Thus the
resolved local tractions are constant on that arc. An arbitrary measured
midpoint stress is not sufficient without this declared homogeneous loading
assumption; no internal spatial stress problem is being solved here.
Lithostatic stress, if used, is part of its total stress, and pore pressure
has already been subtracted: `Sigma_eff = Sigma_total - p_f I`. The evaluator
does not add an invented depth, density, lithostatic load or pore pressure.

For outward radial unit vector `r`, horizontal unit vector `h` pointing from
the proposed subducting plate toward the overriding plate, and explicit dip
`0 < delta < pi/2`, define

```text
d = cos(delta) h - sin(delta) r              down-dip direction
n = sin(delta) h + cos(delta) r              oriented fault-plane normal
sigma_n = n.T Sigma_eff n                   effective compression, Pa
tau = d.T Sigma_eff n                       signed down-dip loading, Pa
Y = C + mu max(sigma_n, 0)                  shear strength, Pa
margin = tau - Y                           yield excess, Pa
```

`C >= 0` is cohesion in Pa and `mu >= 0` is dimensionless friction. Significant
effective tension is a separate rejected state. `abs(tau)` is never used:
reverse shear cannot admit down-dip motion. At exact yield the candidate stays
locked. Tolerances are floating-point bounds, not fitted physical thresholds.

For a local cross-section with `Sigma_hh=p+q`, `Sigma_rr=p`, `Sigma_hr=t`,

```text
tau = q sin(delta) cos(delta) + t cos(2 delta)
sigma_n = p + q sin(delta)^2 + 2 t sin(delta) cos(delta)
```

Mirror candidates change `h` to `-h`. They have identical assessments when
`t=0` and other inputs agree; a tie remains a tie. The test case with
`p=100 MPa`, `q=60 MPa`, `t=10 MPa`, `delta=30 degrees`, `C=2 MPa`, `mu=0.2`
gives margins `+4.248711 MPa` and `-2.287187 MPa`. These are analytic test
inputs, not a fitted planetary initiation model.

Normal velocity is evaluated over the complete shared arc using the exact
sinusoid `R [normal x (omega_b-omega_a)] dot r(theta)`, including interior
extrema. Units are km/Myr for `R` in km and angular velocities in rad/Myr.
Resolved separation anywhere rejects whole-arc admission. A finite convergent
arc with an isolated stagnant endpoint is allowed; a wholly stagnant,
divergent or transform arc is not. Existing cached midpoint velocities do not
replace this check.

## Relation to existing source fields

`genesis_starter.py` reconstructs an effective in-plane stress from prescribed
mantle loading, cooling and tidal stress. It explicitly lacks fault dip and a
vertical overburden/pore-pressure profile. `genesis_faults.py` stores plane
normals with shape `[face,2]`, representing tangent-plane shear orientation,
not radial fault dip. Neither provides this evaluator's missing full input.

`ContactLawParameters` contains a separate existing cohesive/friction law.
Its numerical defaults are not silently imported. Its `water_access` modifies
friction and cohesion and is explicitly not pore pressure; this distinction
also applies here. Likewise, the transported `strength_pa` is not a measured
compressive stress or a fault's complete shear-strength law.

The caller must keep stress frame, spatial-model and time provenance. A
material-attached tensor under rigid rotation `Q` transforms as
`Q Sigma Q.T`. The same tensor transformation defines the explicit parallel
transport along one contact above. A general spatial stress field would require
its own sampling and along-arc mechanical checks and is outside this API.
This evaluator does not itself evolve or silently prolong a loading snapshot.

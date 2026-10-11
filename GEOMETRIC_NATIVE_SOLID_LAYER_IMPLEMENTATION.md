# Finite solid layers, elastic face closure and finite-pose force bounds

Work continues on `fix/genesis-mantle-convection`. The original six roadmap
items are unchanged. Item 1 still has two completed working blocks and one
substantial final physical-admission/accepted-continuation block. This release
adds three mechanisms needed inside that last block; it accepts no new native
material interval. Items 2–6 remain open.

## Finite-thickness local placement

`tectonics/geometric_native_solid_layer_placement.py` declares a local affine
material chart with strike coordinate `s`, bank coordinate `x`, and depth `z`:

```
X(s,x,z) = (s, x + z cot(dip), -z).
```

Both bodies use the same depth shear. Their common dipping face is `x=0` in
material coordinates. The physical shear determinant is exactly one, so the
complete volumes remain `strike_length * bank_width * layer_thickness`.
No receiver wedge is deleted and no artificial neighbour-thickness depth
shift is assigned to the donor. Very narrow material banks remain meaningful.

The separation helper bounds the entire donor prism against the receiving
solid halfspace under finite relative SO(3), including strike, full depth,
inward width, and the supplied whole-clock origin translation. Checking only
the surface or only the common face can miss an interior intersection.
Touching is retained, and positive tiny material is never discarded.

Native adapters authenticate material thickness from its actual source:
original live material has 2 km crust plus 8 km cold mantle; HOT material has
2 km crust. Elastic beam thickness, column length, and force contact height
are different quantities. In this proposed full-face model the existing two
born force carriers cover only 1 km of a 2 km common physical face.
`require_complete_physical_contact_face` refuses that incomplete coverage.
Extending the lower force face needs its own source/energy/history ownership;
the existing once-paid upper-face history is not silently rescaled.

The local chart is not a reconstruction of the actual spherical material
mask. Its bank widths are explicit model declarations, not native mask
widths inferred from a point or an area bound. Global curved embedding,
compatibility between neighbouring charts, the implicit signed correction,
and source-bound whole-clock chart-origin motion still require integration.

## Elastic placement-to-anchor compatibility

`tectonics/geometric_solid_contact_closure.py` retains the existing two
column compliances and normal contact compliance:

```
C = L_sub/E_sub + L_over/E_over + 1/K_normal
p = -min(g_free, 0)/C
g_free = physical_void - shortening_sub - shortening_over - indentation_fault
E/A = p^2 C/2.
```

These equalities are evaluated with exact Fractions. Compression has zero
empty void and separately stored elastic shortening; a negative constitutive
gap is not an absolute layer depth. Every pressure partition is checked,
so reproducing the total gap with a wrong strain split also fails.

The basic closure invents no contact-zone rest thickness. An optional
compaction check requires an independently declared length and refuses a
fully consumed or rebased reference. Actual spatial placement must embed
these deformations consistently, or supply an equivalent physical face
deformation model; local algebra alone does not authorize an underpass.

Tangential closure preserves the supplied signed plastic history and checks
elastic compatibility without replacing the existing return law. C5→C1's
positive increment arises as its radial bank separation decreases from
about 9.277 cm to 8.656 cm during bend unloading. Reverse sliding is a lawful
constitutive branch. Its genuinely incoming material still lacks the actual
solid first-entry certificate required for adoption. The old down-dip gate
and its refusal remain unchanged.

## Finite-pose reaction and energy errors

`tectonics/geometric_native_birth_finite_force_bounds.py` differentiates the
actual exponential rotations using rational Fréchet series and outward
remainders. Receiving-normal rotation, both radial ports, native frame
uncertainty and all five angular basis functions are retained:

```
1, cos(theta), sin(theta), cos(2 theta), sin(2 theta).
```

The unilateral law is globally Lipschitz across opening and closing.
Reaction and potential errors therefore do not assume that a narrow affine
closing tail keeps the same active set under finite rotation. Native measure
and endpoint aliases remain separate errors.

An independent rational enclosure integrates the entire affine reference
arc. Old root/part endpoints are only partition knots: signs are proved anew,
closed cells use outward analytic moments, and unresolved cells retain
clipped-pressure bounds. Distance from the stored Decimal reaction/potential
is explicitly included. This fixes an independently reproduced counterexample:
with exact stiffness `1/3` and zero geometric motion, stored Decimal160 energy
still differs from the exact potential by about `6.6667e-163 J`.

Each normalized coordinate needs its own force budget. The native controller
reconstructs all 19 actual affine balancing scales and compares the born
finite-pose correction with each original tolerance. A ratio to total mantle
torque cannot replace this check. Original forces remain affine in that
diagnostic; complete finite-pose network equilibrium stays uncertified.

## Validation and continuation

Fresh tests and the genuine source controller run in one interpreter under
`analysis/geometric_native_solid_placement_validation/release_validation_01/`.
The entry point is `validate.py`; the compact authoritative result is
`report.json`, with `native_control/report.json` giving physical findings.
The 70 scoped tests belong only to the three new modules. Historical 135,
69 and 137 test results are not included in that count.

The final scoped gate is **PASS: 70/70 tests in 138.86 s**. The genuine source
control is **PASS in 523.350 s**, and the complete same-process suite takes
667.378 s. PASS here includes reproducing the physical refusal. All 16 old
series closures match their real states, source thickness is bound for all
13 components, and the accepted input remains unchanged.

The born finite-force error is only `5.2001e-42` relative to total external
torque, but its bounds exceed the existing individual coordinate budgets:
coordinate 14 has relative bound `0.377919`, and coordinate 16 has
`2.32179e10`. These are bounds, not measured residual values. Neither complete
finite-force equilibrium nor material admission is established.

The cold-born whole-arc finite gap is negative, about `-1.288e-18 m`, although
the affine model has a mostly open arc. A direct exact-Fraction diagnosis
finds a saved-bank normalization alias `1.4873898553785514e-18 m`; its normal
offset is `-1.288117400089088e-18 m`. The before-coordinate builder currently
divides physical `w` by `R` in binary64 before making born anchor offsets.
The finite-pose owner correctly retains the actual saved physical `w`.
Their reference mismatch therefore matters to the microscopic HOT balance.
The source-bound diagnosis is `radial_normalization_diagnosis.json` beside
the validation entry point. It does not claim that a corrected force was solved.

All 778 inherited Python bodies and all 1633 prior artifact bytes are checked
before and after the run. Eight new Python files form the current 786-source
map. The genuine candidate's complete physical payload is compared with the
previous release, excluding only its top-level provenance and selfseal.
Accepted input state is checked unchanged. Geometry classification for its
42 pairs remains inherited evidence for that identical physical candidate;
it is not counted as a new geometric run or a solid admission.

The remaining work is still the last original block of item 1: embed these
finite bodies and elastic faces over the actual spherical material masks,
couple the complete physical faces to forces and once-paid preparation,
and prove whole-clock admission for all nonexcluded pairs. Then adopt a mixed
continuation that preserves all original and born histories, later cohorts,
clock offsets and ledgers, and verify joint saving/restoration. Local prism
PASS and endpoint force bounds do not replace those conditions.

The immediate next implementation is concrete: create canonical Decimal
before-coordinates from the actual saved radial banks with an explicit
quotient-error bound, and recompute the born finite-pose/foundation forces
against every coordinate's own balance. Do not rebase paid preload or memory,
reuse old force certificates for a changed candidate, or interpret a broad
force bound as an observed force residual. Full layer-mask/face coupling and
accepted continuation remain in the same final original block afterward.

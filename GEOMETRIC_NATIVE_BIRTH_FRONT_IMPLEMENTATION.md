# Full birth-contact arcs and all-pair temporal geometry

Work continues on `fix/genesis-mantle-convection`. The six original roadmap
items are unchanged. Item 1 remains partial: two of its three established
blocks are complete; the final physical material-admission block is open.
Items 2–6 are not closed by this release.

## What this release changes

The genuine next mantle load now drives one joint conservative mechanical
solve with the unchanged 16 original contacts and 14-coordinate beam block,
five independently supported HOT patches, and two complete initial shared
arcs. There are 19 coordinates and no born midpoint force atoms. Each arc
uses the declared inclined measure `C dθ`, with `C=R h/sin(dip)`, and its own
three-coefficient normal/shear history. Analytic trigonometric integration
retains both opening portions and narrow closing tails. A stored root band
has explicit force and energy uncertainty; it is not dropped by an area
tolerance. The preparation energy is paid once at zero newborn age and its
implicit unloading remainder remains in the energy ledger.

The genuine development solve took 18 Newton iterations. Its maximum
stored-reaction coordinate residual was `5.2895e-17`. An independent
240-digit standard-library audit checks every coordinate, original history,
continuum potential, reaction, Hessian and hydrostatic support ledger. These
checks authenticate the declared first-order mechanical model; they do not
grant material admission.

The new source-bound adapter retains the actual previous original histories
and advances their clocks and spatial memory without resetting them. The
first-epoch adapter explicitly requires zero radial HOT birth references.
The native material path uses the actual newly solved literal Ω and dt;
Decimal-to-literal angular errors and radial representation aliases are
retained. Full SO(3) degree-two normal, shear and radial-ordering fields are
evaluated independently of the affine force fields. The original finite
anchor tolerance is not used as proof of a tiny HOT opening/closing branch.

## Temporal geometry result

All 42 opposing-owner material pairs have been geometrically classified on
the actual newly solved path: 33 have exactly zero incoming material area
over the entire interval, seven have a positive endpoint subset, and two
have a positive interior-time subset despite disappearing by the endpoint.
Among the 26 pairs involving HOT material, 24 are excluded and two are
positive. Zero incoming material does not erase an existing physical contact
or its force history.

The whole-time exclusions preserve the common time variable, divide out
only a proved exact initial incidence, and allow proved positive combinations
of receiver support planes. The signed curved correction remains implicit:
its true initial outside-archive condition and negative subsequent derivative
exclude HOT5→C1. Neither the curve nor a tiny transient is replaced by a
polygon area cutoff.

The HOT6→HOT2 front has a unique moving flux-zero point and a complete
time-dependent incoming arc. Strict whole-clock concavity certifies first
entry, with the other receiving components excluded. A fixed midpoint or
the sign at just one endpoint is not substituted for that proof.

## Why item 1 remains open

There are concrete physical issues, not a network or environment blockage:

1. A positive-width C5→C1 incoming slice belongs to the existing receiver
   history on trace 066. The actual new whole-trace plastic increment is
   upward, with a conservative lower bound above `0.0031056 m`. This remains
   positive after both epochs' finite-anchor allowances. Its real incoming
   slice cannot be admitted through the unchanged compressed down-dip gate,
   omitted for having a small area, or reassigned to another bank.
2. The independently declared HOT contact preload is not an absolute solid
   depth or a collision-free layer trajectory. Full layer/edge placement,
   its relation to the contact anchors, and finite-pose constitutive margins
   need source-bound proof over the whole material footprint and clock.
3. Once those physical conditions pass, a new mixed accepted-state adapter
   must carry all old and HOT histories, pending newborn cohorts and
   once-paid ledgers through the next material interval and joint restart.
   The existing first-interval acceptor must not be repurposed by appending
   a receipt or resetting the new supports.

One original native interval remains accepted; this release accepts zero new
native intervals. Mechanical and geometrical PASS results do not override
the physical refusals. The remaining work is still the same final block of
item 1, not an additional roadmap stage.

## Evidence and preservation

The final scoped gate passed all **135 new tests in seven modules**, with
`--import-mode=importlib`, in **476.49 seconds** of pytest time. The historical
69 and 137 tests were not rerun and are not included in this count. Original
and additive source bytes remained unchanged throughout this gate.
The same-process native controls passed in **458.112 seconds**, including
fresh force replay, all-pair geometry, the causal front, endpoint SO(3) fields
and whole-clock SO(3) fields. Four independent standard-library audits passed
in a separate process in **6.583 seconds**, with no `tectonics` imports.
The physical admission remains **REFUSED**; this is not an accepted material
interval. The accepted input state is byte-identical before and after.

New evidence is under `analysis/geometric_native_birth_front_validation/`.
The consolidated summary and artifact manifest are in `release_validation_01/`,
the genuine native report is `control_release_01/report.json`, and independent
audit receipts are in `independent_release_01/` within that evidence directory.
The frozen map contains 778 Python source bodies: all 751 inherited bodies
plus 27 additive bodies. Its raw SHA-256 is
`ce4f84e4953e394c513a1fce819871dd92fa19b52d20dcece1e9674fdbb91fcb`.
Development attempts retain their own source-body snapshots and refusals;
the final frozen-source gate and consolidated summary identify the release
authority. All inherited 751 Python source bodies, 1271 artifacts and nine
cloud source files are preserved. The preceding birth-contact implementation
document and release remain immutable; `CLOUD_HANDOFF.md` records the latest
continuation point.

# Static plate membrane: validated foundation, not live fracture coupling

The new `tectonics/genesis_plate_membrane_diagnostic.py` is not called by the
continuation runner. It solves an independent static elastic problem for each
connected load-bearing piece of each plate. It introduces no speed, damage,
fracture, material transport, or thermal modification.

Each plate receives separate copies of its boundary vertex degrees of freedom.
Tangential displacements live on the existing spherical surface; the radial
foundation is prescribed. Free edges have natural zero membrane traction. The
input traction is Cartesian Pa at each face's three vertices; thus shared
geometric vertices can have independent loads on opposite sides of a boundary.

For radius-normalized displacement d=u/R, the assembled equations divide both
physical sides by E R²:

```
K' = Σ A_unit H Bᵀ D B              [m]
f' = Σ A_unit N R τ / E             [m]
```

The returned stress components are σ11, σ22, σ12 in Pa. The face basis is the
existing Genesis shell basis: e1 along vertex0→vertex1 and e2=n_face×e1.
Engineering shear γ12 is used for strain. Output displacement is Cartesian m;
work and elastic energy are J; removed torques are N·m.

For each connected domain the three rotation vectors form Q. With lumped
nodal area W, the load projection is

```
f_balanced = f' − W Q (Qᵀ W Q)⁻¹ Qᵀ f'
```

The removed torque is reported, never converted to extra stress. Plate dynamics
owns that rotational balance. The augmented solve imposes Qᵀ W d=0. No small
stiffness floor connects molten faces: a zero-depth face with nonzero prescribed
load is rejected. Separate disconnected support pieces receive separate gauges.

With spatially uniform basal drag, subtracting a different rigid plate velocity
changes only the removed rotational load. It cannot by itself change deforming
stress. The domain geometry and nonrigid spatial structure of traction matter;
the norm of mantle–plate slip is not an appropriate pointwise stress multiplier.

Validation has nine tests: exact zero/rigid forcing, added rigid load,
independent boundary DOFs, Cartesian rotation/relabeling, separate disconnected
gauges, no zero-thickness support floor, SI torque quadrature and work=2×energy,
modulus/thickness scaling, and convergence to the analytical degree-two
tangential membrane stress on a closed sphere. Constant element stress from
linear displacement elements converges at first order in spatial resolution.

The read-only probe of the corrected-projection checkpoint at400.878Myr uses
current cold-lid thickness and the configured0.08MPa prescribed source. It
reports maximum principal tension9.258615MPa, area mean1.478955MPa, and8.99604%
of area above the stored tensile strength. Maximum strain is1.51869×10⁻⁴;
equilibrium relative residuals are6.6–9.2×10⁻¹⁴. Removing the saved Euler velocity
changes deforming stress by only5.60×10⁻⁸Pa. These are elastic basal-only
diagnostics, not predicted damage or proof of an imminent separating fracture.
The probe intentionally omits damaged-stiffness redistribution for this initial
comparison, and it omits ridge/slab/contact edge loads.

Before enabling a live stress/damage path, the remaining work is:

1. Add compatible line tractions from ridge, slab, and contacts, including their
   signs, free/closed edges and torque/work accounting shared with dynamics.
2. Introduce constitutive stress/eigenstrain/Maxwell history per material parcel.
   Cool-growth dilution, stress relaxation, damage-dependent stiffness and
   physically defined acceptance of a proposed step must remain consistent.
3. Transport tensor memory with material rotation and tangent-basis changes;
   implement split/merge inheritance, remeshing, new-material initialization,
   versioned checkpoint persistence and deterministic restart. Copying the three
   old face components to a new face is insufficient.
4. Replace the prescribed tensor load in the existing yield/damage integration
   with the accepted mechanical stresses, preserving tidal-cycle quadrature and
   existing rupture eligibility/latches. Avoid evaluating a second independent
   mantle stress source in the same damage update.
5. Verify zero-load Maxwell relaxation, rigid-rotation objectivity, free-edge
   response, virtual work, small-strain failure guards, substep convergence,
   tensor donor transport, and full conservative/restart regression before
   enabling it in a physical comparison.

The local cooling implementation is a separate completed adapter change:
`genesis_local_mechanics.py` reconstructs a passive half-space profile from the
already transported crust age, bounded by the primordial reference column.
It preserves old material exactly and has an explicit version gate. Current
Ts/Tm are quasi-static boundary values; this is not a solved per-parcel enthalpy
history. Its heat-content deficit is diagnostic, not a second global heat sink.

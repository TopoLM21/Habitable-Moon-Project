# Accepted-material boundary implementation

Implemented as the explicitly selected `subduction_memory.model:
accepted_material_v1`, with persisted inventory schema
`young-boundary-material-2`. Ordinary mature mode remains `legacy`.

The signed physical view integrates closure/opening area for every actual
inter-plate edge, independently of display thresholds. These integrals are
**candidates**, never proof that material has subducted. The present producer
accepts only oceanic parcels that the conservative raster material transaction
actually removed. `advance_lithosphere(..., young_subduction_sink=callback)`
reports those exact lost source volumes to the inventory after the material
update. It does not subtract them a second time. Its callback is
`callback(events, new_state)`.

Polarity follows the actual donor and receiver. Young ocean-overlap resolution
retains the less negatively buoyant, then younger parcel; ties use existing
surface ownership and local source position, never the ordering of plate IDs.
The final material-position tie is a deterministic raster convention, not a
physical theory of perfectly symmetric subduction initiation. Unresolved
accepted contacts are retained in the volume ledger without an invented
attached trench or torque.

Each attached segment keeps accepted area, ocean volume, mantle volume, and
the **vector sum** of buoyancy moments. Each material transaction adds a
thermal cohort with its acceptance time and original mantle thickness.
There is no lower buoyancy floor and no division by numerical `slab_length_cap`.
The optional full-transmission torque is `R*g*ΣΔρV(r×towards_trench)` in N m.
It is explicitly an upper-bound transmission closure, not a bending/viscous
slab solution; the new dynamics default keeps this force disabled pending that
closure. The legacy empirical breakoff and rollback laws do not act on the
new inventory.

Cold buoyancy is distinguished from cumulative accepted material. The passive
finite-sheet model warms each cohort from both sides in a mantle-temperature
bath using the existing diffusivity κ. Its mean deficit is the analytical
Dirichlet heat-equation solution. It introduces no fitted thermal decay time
and no second heat source in the global Genesis energy solver. This is a
passive mechanical approximation; global backreaction of slab warming is not
implemented. A physical mantle depth, when supplied from the interior model,
limits current attached area by `depth/sin(dip)`; oldest material is transferred
to a separately retained deep fraction. Dip follows the existing explicit
35°→55° geometric closure and is not inferred from a resolved slab shape.

Checkpoint/resume preserves all cohorts. Intermediate experimental schema v1
is rejected because it lacks parcel ages; its material cannot be assigned a
fabricated thermal history. Historical ordinary checkpoints have no new
inventory and remain on their existing path. Topology relabeling preserves
physical anchors; mesh refinement divides extensive material and candidate
quantities over descendant edges, preserving vector torque and thermal ages.

Remaining work is explicit: true conservative fractional/subcell acceptance
requires a footprint/overlap transport representation; accumulated negative
normal velocity alone is not used as an arbitrary subduction sink. Initiation,
bending resistance, transmitted pull, and consistent breakoff/rollback remain
separate physical closures. The implementation does not claim these are solved.

Validation command:

```powershell
& .\.venv\Scripts\python.exe -m pytest `
  tests/test_young_boundary_material.py tests/test_genesis_starter_slab.py `
  tests/test_subduction_memory_v018.py tests/test_slab_breakoff_v020.py `
  tests/test_rollback_v019.py -q
```

Result: **48 passed**. Coverage includes signed low-speed geometry, no phantom
material, exact raster-loss volume reconciliation, no duplicate transactions,
SI dimensions, vanishing buoyancy, vector cancellation, all 24 plate-ID
permutations, checkpoints/resume, physical-depth FIFO conservation, independent
finite-volume heat-equation comparison, and mesh-refinement conservation.

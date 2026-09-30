# Polarity evidence and limits

This stage preserves an explicitly oriented local contact and passively accounts
for accepted slab material. It does not solve spontaneous subduction initiation.

## Primary research

- [Toth and Gurnis (1998), Dynamics of subduction initiation at preexisting fault zones](https://authors.library.caltech.edu/records/ev1w2-6v917),
  DOI 10.1029/98JB01076. Their thermomechanical experiments begin with a dipping
  weak fault and imposed force or velocity conditions. They demonstrate how
  fault resistance and plate loading affect initiation. This supports keeping
  an explicit oriented fault as a boundary condition; it does not justify
  inferring its dip from a scalar damage number.
- [Gurnis, Hall and Lavier (2004), Evolving force balance during incipient subduction](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2003gc000681),
  DOI 10.1029/2003GC000681. Their two-dimensional experiments include elastic
  bending, viscous deformation, plastic failure and heat transport. Initiation
  resistance depends on plate structure and accumulated convergence; their
  homogeneous-plate cases did not produce self-sustaining subduction. These
  results concern their modeled conditions and are not a universal prohibition.

Both entries are paraphrases. No external numerical parameters have been fitted
or copied into the current geometry experiment from these papers.

## What the saved model actually supplies

`tectonics/fractional_surface_io.py::surface_from_lithosphere` carries scalar
fracture damage, cooling stress, water access, yield ratio and strength. None of
these is a dipping fault plane or a signed underthrust displacement. Spatial
variation can identify weakness but does not by itself specify which side dips.

`tectonics/genesis_starter_material.py::independent_mantle_omega` and
`tectonics/genesis_young_mechanics.py` represent prescribed tangential mantle
traction through a velocity or angular-velocity equivalent. There is no radial
downwelling state available to resolve this missing direction. Treating a radial
component of an angular velocity vector as vertical velocity would be incorrect.

`tectonics/young_boundary.py::young_ocean_overlap_order` retains the least
negatively buoyant candidate, then the younger material. If those agree, it uses
the previous owner, nearest source, and finally a lexicographic position
convention. These last choices are numerical raster conventions. An old slab
therefore provides *inherited model orientation*, not independent evidence that
physical initiation has been resolved. The older generic
`tectonics/subduction_memory.py::choose_subducting_side` also has a plate-ID tie
breaker; it must not supply a new geometric polarity decision.

## Reproducible source audit

Run from the repository root:

```powershell
& .\.venv\Scripts\python.exe analysis/geometric_slab_validation/source_polarity_audit.py
```

The script calls the read-only source loader:

```python
from run_fractional_transport_probe import load_probe_source
mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
memory = checkpoint.subduction_memory
inventory = None if memory is None else memory.young_boundary_state
# inventory.contacts: dict[str, YoungContact]
# inventory.segments: dict[str, YoungSlabSegment]
# segment.thermal_cohorts: list[SlabThermalCohort]
```

For the saved 50 Myr continuation, the serialized source is
`analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050/mature_checkpoint/meta.json`,
at `subduction_memory.young_boundary_state`. Its contacts and segments are JSON
lists; the loader reconstructs dictionaries. Starter is loaded from
`results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz` and
has an empty slab inventory.

The companion `source_polarity_audit.json` includes source hashes, all segment
directions and per-contact retained area, oceanic volume, original cold volume,
initial excess mass and thermally surviving excess mass. Locality checks verify
the exact two shared mesh vertices, current face owners and present contact,
rather than selecting the nearest same-pair contact elsewhere on the sphere.

An inherited orientation is admissible only as an explicitly selected model
condition and only where every valid local attached history agrees. Opposing
histories remain conflicts; choosing the direction with greater mass would be
a new unvalidated initiation law. Missing orientation remains unresolved.

The Starter has no existing orientation. The saved 50 Myr state contains 50
attached segments on 30 contact keys: 22 keys have one direction and 8 have both
directions. Thus the conflict is present in retained history, not merely in
absent or detached contacts. This audit does not migrate old slab mass into the
new experiment or mutate a saved source.

## Exact geometric translation

`tectonics/geometric_polarity_source.py::legacy_polarity_evidence` is the
explicitly invoked initial-import adapter. It requires exactly one original
triangle per provenance cell and rejects moved or mixed footprints. Each
retained attached segment maps by its two fragment identities and actual shared
edge, rather than by closest midpoint or plate-pair identity alone. Conflicting
directions produce separate evidence records with the original segment IDs.

Reproduce its source mapping with:

```powershell
& .\.venv\Scripts\python.exe analysis/geometric_slab_validation/source_polarity_mapping.py
```

`source_polarity_mapping.json` contains all accepted and excluded mappings plus
before/after inventory hashes. Starter supplies zero evidence on its 122
geometric contacts. The saved 50 Myr source has 245 geometric contacts; all 50
attached segments map to their exact 30 contacts (22 oriented and 8 conflicting),
and all 50 detached segments are excluded. No legacy material or force is
imported. The 14 dedicated adapter tests cover valid mapping, conflicting
directions, stale contacts, wrong owners, detached/deep/invalid cohorts,
noninitial surfaces and unchanged inputs.

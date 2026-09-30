# Geometry and contact validation

This is a fixed-velocity geometry experiment. It does not change the GUI, the
0.5 force solver, or the previous coupled fractional thermal experiment.

`validation_summary.json` records the measured cases, final test log hash, new
code hashes, original-source integrity, and the unchanged pre-existing code.
`baseline.json` captured 176 production/configuration files and 54 source files
before this change. `source_summary.json` describes the actual saved material.

The independent checks are described in `TEST_MATRIX.md`:

- `static_contact_probe_final.json`: exact contact counts and arc lengths from the
  original pure mesh, plus frame rotation of the actual 5120-fragment source.
  This final run records the frozen production hashes; the earlier
  `static_contact_probe.json` is retained as an intermediate measurement.
- `starter_common_sub4_v2/report.json`: complete geometric rotation, exact
  checkpoint restart, mixed-cell projection, coverage, and per-origin budgets.
- `saved50_common_sub4/report.json`: the same experiment with nonzero cold
  mantle and excess mass, to check all four extensive quantities.
- `starter_actual_motion_final/report.json` and
  `saved50_actual_motion_final/report.json`: actual saved plate velocities.
  These report physically unresolved polarity explicitly and leave the
  source unchanged; they are not successful coupled evolution runs.
- `final_tests.log`: the full final selection of geometry and previous
  fractional material/thermal tests.

The initial `starter_common_sub4` attempt is preserved. It exposed a false
positive overlap of `3.18e-36` steradians at a shared vertex after a common
rotation. The exact input pair is in `first_false_overlap.json`. The primitive
was repaired using an edge-anchored incidence predicate; the successful v2 run
used the repaired code. Small polygons were not removed with an area cutoff.

`equal_area_different_contacts.png` illustrates two exact half-triangle
partitions with equal owner areas but different boundary directions.
`actual_mixed_contact.png` draws actual saved Starter polygons after a common
rotation of 0.25 Myr; its close-up shows the physical contact inside an
integration cell. Both figures use a gnomonic view of spherical polygons.

To repeat the actual common-rotation check, use a new output directory:

```powershell
& .\.venv\Scripts\python.exe analysis\geometric_contact_validation\run_common_rotation.py --source starter --target-subdivisions 4 --output analysis\geometric_contact_validation\starter_common_sub4_new
& .\.venv\Scripts\python.exe analysis\geometric_contact_validation\run_common_rotation.py --source saved50 --target-subdivisions 4 --output analysis\geometric_contact_validation\saved50_common_sub4_new
```

The full 5120-cell Python polygon projection takes several minutes. A finer
integration mesh is exercised in small exact tests; an additional full-source
subdivision-5 projection is not claimed here. Local checkpoints and generated
case folders remain available but are excluded from version control by the
folder's `.gitignore`.

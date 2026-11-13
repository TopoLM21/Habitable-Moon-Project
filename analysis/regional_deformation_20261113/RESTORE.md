# Reproduce or continue this experimental coupled mode

The runtime is committed in `9435de7c50e2ecd35b42e2d653e703d40082c36f` and was
verified on remote `reconstruction/regional-20261113`. The local results commit
is `1004735`; its push failed with missing Git transport credentials. Do not
assume that the later report/results commits are on GitHub. Use the final
backup receipt and the included bundle/manifest when restoring elsewhere.

The old production runner and prior mode remain unchanged. These are new
reconstruction results, not recovered results of the lost November session.

## Inspect and continue a saved state

From a checkout containing this stage:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
from pathlib import Path
from tectonics.deformation_exchange import DeformationExchange
p = Path('analysis/regional_deformation_20261113')
state = DeformationExchange.load(p / 'states/native_g3_20')
state.advance(100.)
reference = DeformationExchange.load(p / 'states/native_g3_100')
assert state.checkpoint() == reference.checkpoint()
print(state.diagnostics())
state.save(p / 'states/new_independent_resume100')  # refuses overwrite
PY
```

Both `surface.npz` and `meta.json` are required. The latter contains physical
and controller parameters, the finite material/thermal inventory, component
provenance, material age and transfer ledgers, and the work/heat histories.
The surface data has its own version identifier; loading it as the older
gravity-only reference is explicitly rejected. The native coupled restart
checks geometry, transferred mass, densities, heat capacity, time and whether
mechanical heat was deposited exactly once.

`states/transfer_dt1_event` is the externally funded transfer checkpoint at
2 Myr. Its initial energy/work history differs from the unforced control.
Do not substitute one for the other or call the prescribed event spontaneous.

## Re-run without overwriting evidence

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  analysis/regional_deformation_20261113/verify.py independent_native20 \
  --scenario native --sub 3 --length 1000000 --dt 1 --end 20
.venv/bin/python -m pytest -q
```

Use a new case name. The harness refuses existing JSON or state directories.
`queue.py` records the completed comparison definitions; its original summary
is immutable evidence, not an invitation to overwrite it. `transfer_control.py`
defines the heterogeneous externally funded impulse controls. Full parameters
and exact runtime hashes are included in every case JSON.

## Verify the artifact manifest

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
p = Path('analysis/regional_deformation_20261113')
m = json.loads((p / 'artifact_manifest.json').read_text())
for name, entry in m['artifact_files'].items():
    data = Path(name).read_bytes()
    assert len(data) == entry['bytes'], name
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], name
for name, expected in m['runtime_files_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('All bytes and runtime hashes verified.')
PY
```

If an incremental Git bundle is delivered, it requires the already published
base commit `9435de7c50e2ecd35b42e2d653e703d40082c36f`. Verify it using
`git bundle verify PATH`; fetch its branch into a **new recovery branch** to
avoid overwriting current work. The companion archive also contains the new
stage files and runtime/tests with a manifest, without credentials or unrelated
repository history. Do not run force-push, reset or deletion commands to repair
the transport-authentication failure.

# Restore the bounded phase-supply stage

Use this report's enclosing commit on `reconstruction/regional-20261113`.
`artifact_manifest.json` gives the baseline and exact runtime/artifact hashes.
Both `meta.json` and `surface.npz` are required per checkpoint. Keep old
`StrainExchange` snapshots separate; only `PhaseStrainExchange` loads these.
New phase initialization is allowed only at time zero, never as an inferred
thermal/fractionation history in an evolved old snapshot.

Read-only load and reproduce the saved one-to-two-Myr continuation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
from pathlib import Path
from tectonics.phase_exchange import PhaseStrainExchange
p=Path('analysis/phase_inventory_20261113/states')
x=PhaseStrainExchange.load(p/'limited_g3_mid')
x.advance(2.)
y=PhaseStrainExchange.load(p/'limited_g3_final')
assert x.checkpoint()==y.checkpoint()
print(x.diagnostics())
PY
```

`verify.py` records all 13 case definitions and refuses to overwrite results.
It is not an automatically running queue. Any later reproduction must use new
case names (import `run` and pass a new name); it must not replace these files.
Case JSON records the parent commit used during development. The runtime
hashes in the manifest identify the released working code; the final case
checkpoint comparisons also passed after the final conservative rounding.

The retained early failure logs document the corrected flux-reference test and
the real rounded-quota stock overdraw. `release_guard_tests.log` is the authoritative
final suite: 393 passed. `audit.json` contains all compact comparisons, budget
checks, exact saved restarts and protected-artifact hash checks.

Verify the released bytes without rerunning calculations:

```bash
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('analysis/phase_inventory_20261113')
m=json.loads((p/'artifact_manifest.json').read_text())
for name,e in m['artifact_files'].items():
    raw=Path(name).read_bytes()
    assert len(raw)==e['bytes']
    assert hashlib.sha256(raw).hexdigest()==e['sha256'],name
for name,h in m['runtime_files_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
print('Verified phase-stage artifacts and runtime')
PY
```

No continuation beyond the bounded verification is authorized by this restore
document. The user requested stopping after this stage and its publication.

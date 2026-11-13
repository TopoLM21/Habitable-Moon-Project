# Continue the new finite-pressure experiment

Code baseline: **278c6354b6610ba57fe42019aa5e8d8e2fe4db10** on
`reconstruction/regional-20261113`. Later results/report commits retain the
same runtime hashes. Main and earlier controls are unchanged. The old
bc2ee6a publication failure was resolved before this stage; use the latest
remote branch and the verified SHA reported with the result.

The mode is imported as `tectonics.strain_exchange.StrainExchange` rather than
silently enabled in the production runner. Both `meta.json` and `surface.npz`
are required. The checkpoint includes the finite thermal/material pool,
pressure energy and net withdrawal, band weights, initial-source provenance,
geometry, ages, controller state and all work/heat ledgers. Do not reinitialize
pressure when resuming or reinterpret an old checkpoint as this new mode.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
from pathlib import Path
from tectonics.strain_exchange import StrainExchange
p=Path('analysis/strain_exchange_20261113/states')
o=StrainExchange.load(p/'native_g3_20')
o.advance(100.)
assert o.checkpoint()==StrainExchange.load(p/'native_g3_100').checkpoint()
print(o.diagnostics())
o.save(p/'new_independent_resume100')  # refuses overwrite
PY
```

Fresh bounded case, using a **new** name:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  analysis/strain_exchange_20261113/verify.py new_native_check \
  --scenario native --sub 3 --dt 1 --pressure 0 --end 20
.venv/bin/python -m pytest -q
```

`queue.py` records the completed 20-case definitions. Do not overwrite its
original outputs. `audit.py` proves exact 20→100 Myr restart and checks prior
manifests. `independent_work.py` independently evaluates an actual transfer
with 70-digit arithmetic. `transition.py` preserves the common 50 Myr state
and three subsequent time-step controls. `benchmark.py` measures a fresh
process with standard-library resource accounting; the first failed attempt
to use absent `/usr/bin/time` is retained in `native_fresh100.log` and did not
start a duplicate calculation. `native_fresh100_run.log` is the completed run.

`comparison.csv` contains 23 preflight/grid/step/sensitivity/long/fresh cases.
The later `native_compliance2e8_100.json` and `transition.json` add the finite
pressure-limitation continuation and its time-refinement checks. The source
stores have different declared initial energy/compliance: do not substitute
one for another. All runs are experimental conditional exchange, not a
reconstruction of lunar thermal/convection history.

Verify bytes after retrieval:

```bash
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('analysis/strain_exchange_20261113')
m=json.loads((p/'artifact_manifest.json').read_text())
for name,entry in m['artifact_files'].items():
    raw=Path(name).read_bytes()
    assert len(raw)==entry['bytes']
    assert hashlib.sha256(raw).hexdigest()==entry['sha256'],name
for name,expected in m['runtime_files_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
print('Verified artifacts and runtime')
PY
```

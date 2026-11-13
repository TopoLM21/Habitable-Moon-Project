# Reproduce the bounded material-area continuation

Use the working branch `reconstruction/regional-20261113`, with runtime code
from `4e714a4` or its subsequent evidence commit. The new mode has explicit
`regional_material.active_area.enabled: true` and `strip_rule: coarea` in its
saved YAML. Do not switch an evolved point-only state to this mode: its missing
material-area history cannot be inferred.

The accepted restart control starts at 12 Myr and ends at exactly 20 Myr.
Choose new output paths rather than overwriting evidence:

```bash
MPLCONFIGDIR=/tmp/hm_reconstruction_mpl .venv/bin/python run_long_evolution_v131.py \
  --config analysis/regional_area_20261113/area_restart20.yaml \
  --resume analysis/regional_area_20261113/saved_states/area12 \
  --dt 4 --end-time 20 \
  --output outputs_v131_regional_area/user_repeat20 \
  --checkpoint outputs_v131_regional_area/user_repeat20/restart
```

`saved_states/area20` is the uninterrupted reference;
`saved_states/area_restart20` is its bitwise-equal continuation. Saved numerical
settings, quaternions, tree paths, extensive stocks and memory must be retained.
Changing settings at restart is rejected. Zero-time initial states used for
cross-map comparisons remain in the previous stage's
`analysis/regional_inventory_20261113/saved_states/initial_same_g1/g2/g3`.

The matrix harness refuses existing output names. It has a 120-second ceiling
per child and no automatic retries. Do not rerun a named suite in place. A new
study should give new names and address the explicit source-time and contact
geometry defects in `RESULT.md`, not extend a state as an asserted lunar history.
`coarea_probe.py` is an independent frozen-source numerical test, not a coupled
fracture history. The old point mode remains available and regression checked.

`artifact_manifest.json` binds parameters, reports, logs, numerical summaries,
figures, native states and runtime file hashes. The incomplete
`source_diagnosis.json` is not a restart or an accepted result; it records an
initial reporting-only NumPy scalar serialization failure. Its verified
replacement is `source_diagnosis_verified.json`. The earlier failed development
tests and point-deposition probes are similarly preserved with their original
names; the acceptance results are `final_tests.log`, `comparison.json`,
`coarea_probe.json` and the two native comparison JSON files.

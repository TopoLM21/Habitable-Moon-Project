# Reproduce the source and contact checks

Use `reconstruction/regional-20261113` at the evidence commit containing this
file. The source-time operator is from `be47df2`, the material-bank registry
from `0e45555`, and its interval-union correction from `c70daf5`. Exact runtime
hashes are in `artifact_manifest.json`; per-case source provenance is retained.
The gate8 smoke test used the working tree based on `df23c61`, and the dt1
control used `be47df2` plus inactive contact diagnostics. Their result JSON
files give exact tested hashes rather than pretending the tree was clean.

The original defaults remain `source_time_rule: endpoint` and
`contact_policy: legacy_map`. New configurations explicitly select
`regional_material.active_area.source_time_rule: localized`. Do not change
saved numerical settings on an evolved restart; the runtime rejects that.
All map comparisons here start from the unchanged zero-time physical states
under `analysis/regional_inventory_20261113/saved_states/initial_same_g1/g2/g3`.

The accepted restart comparison starts at 8 Myr and ends at 20 Myr. Choose new
output paths. Do not overwrite the archived states or rerun named harness
cases in place:

```bash
MPLCONFIGDIR=/tmp/hm_reconstruction_mpl .venv/bin/python run_long_evolution_v131.py \
  --config analysis/contact_time_20261113/gate_restart20.yaml \
  --resume analysis/contact_time_20261113/saved_states/gate8 \
  --dt 4 --end-time 20 \
  --output outputs_v131_contact_time/user_repeat20 \
  --checkpoint outputs_v131_contact_time/user_repeat20/restart
```

`saved_states/gate20` is the uninterrupted reference;
`saved_states/gate_restart20` is its bitwise-equal continuation.
`gate20_dt2`, `gate20_dt1` and `gate20_g1` retain their own parameters.
`old_area20` is the unchanged endpoint regression, not an improved history.
The derived contact registry uses the native reference geometry, material
owners and quaternions; it adds no stock that must be reconstructed separately.

`audit_material_banks` observes material geometry while preserving the legacy
map-force control. `require_shared_support` rejects separated bank geometry
at the current/predicted endpoint before a material-inventory update. In the
specified strict4 case this is an expected rejection, not an accepted step.
It does not solve pressure, gaps, overlaps, contact work or arbitrary events
inside a long step. A new slab detachment also requires separate localization
before the localized-source mode can continue it.

Run the regression suite with `.venv/bin/python -m pytest -q`. The final
acceptance result is `final_tests.log` (340 passed). `gate_restart_comparison.json`
and `old_area_comparison.json` compare native arrays and metadata independently.
`coverage_validation.json` verifies that the final coverage correction leaves
saved final contact diagnostics unchanged. `comparison.json` is independent
ledger extraction; it does not certify full physical energy closure.

The matrix harness has a 120-second child ceiling and refuses duplicate output
names. The gate20_g3 session outcome was unavailable; no result/state exists
and no actual duration is asserted. Inspect processes and outputs before any
future retry. Its status file is evidence of incompleteness, not a success.
No long run should be represented as physical tectonics until the geometry
and mechanical-work limitations in `RESULT.md` have been addressed.

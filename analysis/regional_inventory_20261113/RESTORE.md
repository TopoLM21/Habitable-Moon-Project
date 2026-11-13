# Restore only the explicitly labelled reconstructed experiment

Use the implementation on `reconstruction/regional-20261113` at `db39fed` or the
subsequent evidence commit. Install no new scientific packages for these checks;
the project environment used NumPy/SciPy/PyYAML/matplotlib/pytest.

To repeat the confirmed native restart from the saved 48 Myr state, choose a
**new** output directory (do not overwrite any evidence):

```bash
MPLCONFIGDIR=/tmp/hm_reconstruction_mpl .venv/bin/python run_long_evolution_v131.py \
  --config analysis/regional_inventory_20261113/fixed_restart100.yaml \
  --resume analysis/regional_inventory_20261113/saved_states/m3_first48 \
  --dt 4 --end-time 100 \
  --output outputs_v131_regional/user_repeat100 \
  --checkpoint outputs_v131_regional/user_repeat100/restart
```

`m3_g2_dt4` is the uninterrupted 100 Myr reference; `fixed_restart100` is its
bitwise-equal corrected continuation. `m3_restart100` is retained **only as a
known invalid pre-fix restart diagnostic**. It is not the accepted state.

The zero-time `initial_same_g1/g2/g3` states have identical physical fields and
boundaries, conservatively subdivided, not independent randomized initial data.
New material history is initialized only at zero time. Do not reinterpret an
older evolved checkpoint as having this material history.

Only the settings in the saved regional control are supported. In particular,
topology and the explicitly excluded sediment/craton/plume/rift processes cannot
be enabled and represented as validated with these states. The reported mass
and sensible-enthalpy checks do not validate mechanical/latent/chemical energy.

The verification scripts create distinct outputs and refuse overwriting a run.
The completed short suite includes a recorded 120 s resource stop. Do not rerun
the entire suite in place or automatically increase that budget. A new study
should select new output names and address the numerical/model limits in
`RESULT.md`, rather than extending this state into an asserted lunar history.

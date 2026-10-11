"""Prospective producer guards: do not recover cohort strain from averages."""
from dataclasses import replace
import json

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters
from tectonics.genesis_mobile import (MobileModel, _RetryStep,
    save_mobile_checkpoint, load_mobile_checkpoint)
from tectonics.genesis_onset import OnsetParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_tides import TidalParameters


def model():
    return MobileModel(ShellParameters(subdivisions=1), GenesisParameters(),
                       OnsetParameters(), TidalParameters(enabled=False))


def corrupt(path, edit):
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(payload["metadata"]))
    edit(meta)
    payload["metadata"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **payload)


def test_molten_producer_starts_empty_unactivated_cohort_memory(tmp_path):
    source = model()
    state, thermal, orbit = source.initial()
    memory = state.cohort_mechanics
    assert memory.source_identity == state.bulk_birth_history.source_identity
    assert memory.time_myr == 0.
    assert not memory.established and not memory.cohorts
    assert memory.elastic_energy_j == 0.
    state, thermal, orbit, _ = source.step(state, thermal, orbit, .002)
    assert not state.membrane_established
    assert state.cohort_mechanics.accepted_intervals == state.accepted_steps
    path = tmp_path/"molten.npz"
    save_mobile_checkpoint(path, source, state, thermal, orbit, {}, {})
    restored, loaded, heat, tide, _ = load_mobile_checkpoint(path)
    assert loaded.cohort_mechanics == state.cohort_mechanics
    direct = source.step(state, thermal, orbit, .004)
    resumed = restored.step(loaded, heat, tide, .004)
    assert direct[0].cohort_mechanics == resumed[0].cohort_mechanics
    assert direct[0].bulk_birth_history == resumed[0].bulk_birth_history
    np.testing.assert_array_equal(direct[0].column_enthalpy, resumed[0].column_enthalpy)


@pytest.mark.parametrize("edit", [
    lambda meta: meta.__setitem__("format", "genesis-mobile-0.2"),
    lambda meta: meta["state"].pop("cohort_mechanics"),
    lambda meta: meta["state"].__setitem__("cohort_mechanics", None),
])
def test_missing_constitutive_source_cannot_be_relabelled_or_filled_with_zero(tmp_path, edit):
    source = model()
    path = tmp_path/"missing.npz"
    save_mobile_checkpoint(path, source, *source.initial(), {}, {})
    corrupt(path, edit)
    with pytest.raises(ValueError):
        load_mobile_checkpoint(path)


def test_material_history_cannot_be_removed_while_mechanics_remains(tmp_path):
    source = model()
    state, thermal, orbit = source.initial()
    with pytest.raises(ValueError, match="birth history"):
        save_mobile_checkpoint(tmp_path/"invalid.npz", source,
            replace(state, bulk_birth_history=None), thermal, orbit, {}, {})


def test_rejected_formation_trial_does_not_publish_material_or_constitutive_history(monkeypatch):
    source = model()
    initial = source.initial()
    trial = source._trial
    def reject(*args):
        candidate = trial(*args)
        assert candidate[0].cohort_mechanics.accepted_intervals == 1
        raise _RetryStep("forced_cohort_rollback")
    monkeypatch.setattr(source, "_trial", reject)
    state, thermal, orbit, _ = source.step(*initial, source.mobile_p.min_step_myr)
    assert state.stopped_reason == "forced_cohort_rollback"
    assert state.accepted_steps == 0 and state.rejected_steps == 1
    assert state.bulk_birth_history is initial[0].bulk_birth_history
    assert state.cohort_mechanics is initial[0].cohort_mechanics
    assert thermal == initial[1] and orbit == initial[2]

"""Exact restart must never silently change the Genesis heat-transfer law."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from tectonics.genesis import (
    MODEL_VERSION, GenesisParameters, initial_state, load_checkpoint, save_checkpoint,
)
from tectonics.genesis_contact import ContactModel
from tectonics.genesis_coupled import CoupledModel
from tectonics.genesis_faults import FaultModel, load_fault_checkpoint, save_fault_checkpoint
from tectonics.genesis_mobile import MobileModel, load_mobile_checkpoint, save_mobile_checkpoint
from tectonics.genesis_onset import OnsetModel, OnsetParameters, load_onset_checkpoint, save_onset_checkpoint
from tectonics.genesis_shell import (
    ShellParameters, initialize_shell, load_shell_checkpoint, save_shell_checkpoint,
)
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_starter_continuation import load_starter_source
from tectonics.genesis_tides import TidalParameters
from tectonics.mesh import build_icosphere
from run_genesis_starter import _resume_model


def _rewrite_version(path, version):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    metadata = json.loads(str(arrays["metadata"]))
    if version is None:
        metadata.pop("thermal_model_version")
    else:
        metadata["thermal_model_version"] = version
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


@pytest.fixture(params=["shell", "onset", "mobile", "fault", "starter"])
def checkpoint_case(request, tmp_path):
    # A nondefault rheology parameter must survive all wrapper formats exactly.
    thermal = GenesisParameters(viscosity_reference_pa_s=3.2e21)
    shell = ShellParameters(subdivisions=1)
    tides = TidalParameters(enabled=False, spin_state="synchronous_zero_obliquity")
    path = tmp_path / "checkpoint.npz"
    kind = request.param
    if kind == "shell":
        mesh = build_icosphere(shell.subdivisions)
        save_shell_checkpoint(path, initialize_shell(mesh, shell, thermal), initial_state(thermal),
                              shell, thermal, {}, {})
        return path, load_shell_checkpoint, lambda value: value[3], thermal
    if kind == "starter":
        model = StarterModel(build_icosphere(shell.subdivisions), thermal, tides, shell)
        model.save_state(path, model.initial_state())
        return path, model.load_state, lambda value: model.thermal, thermal
    model_type, save, load = {
        "onset": (OnsetModel, save_onset_checkpoint, load_onset_checkpoint),
        "mobile": (MobileModel, save_mobile_checkpoint, load_mobile_checkpoint),
        "fault": (FaultModel, save_fault_checkpoint, load_fault_checkpoint),
    }[kind]
    model = model_type(shell, thermal, OnsetParameters(), tides)
    save(path, model, *model.initial(), {}, {})
    return path, load, lambda value: value[0].thermal, thermal


def test_current_wrapper_preserves_thermal_model_and_parameters(checkpoint_case):
    path, load, parameters, thermal = checkpoint_case
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
    assert metadata["thermal_model_version"] == MODEL_VERSION
    assert parameters(load(path)) == thermal


@pytest.mark.parametrize("version", [None, "genesis-thermal-0.1", "future-thermal-law"])
def test_wrappers_reject_missing_old_and_unknown_thermal_laws(checkpoint_case, version):
    path, load, _, _ = checkpoint_case
    _rewrite_version(path, version)
    with pytest.raises(ValueError, match="Regenerate.*initial conditions"):
        load(path)


@pytest.mark.parametrize("load", [_resume_model, load_starter_source])
def test_starter_source_loaders_reject_old_parameters_before_loading_state(tmp_path, load):
    metadata = {"format": "genesis-starter-run-0.1", "thermal": asdict(GenesisParameters())}
    (tmp_path / "parameters.json").write_text(json.dumps(metadata), encoding="utf-8")
    # No NPZ or shell configuration: rejection must precede model construction.
    with pytest.raises(ValueError, match="Regenerate.*initial conditions"):
        load(tmp_path / "checkpoint.npz")


def test_starter_source_and_cli_load_same_versioned_configuration(tmp_path):
    shell = ShellParameters(subdivisions=1)
    thermal = GenesisParameters(viscosity_reference_pa_s=3.2e21)
    model = StarterModel(build_icosphere(1), thermal,
        TidalParameters(enabled=False, spin_state="synchronous_zero_obliquity"), shell)
    path = tmp_path / "checkpoint.npz"
    model.save_state(path, model.initial_state())
    metadata = {"format": "genesis-starter-run-0.1", **model.configuration}
    (tmp_path / "parameters.json").write_text(json.dumps(metadata), encoding="utf-8")
    for load in (_resume_model, load_starter_source):
        restored, state, _ = load(path)
        assert restored.thermal == thermal
        assert restored.fingerprint == model.fingerprint
        assert state.time_myr == 0.


@pytest.mark.parametrize("model_type", [ContactModel, CoupledModel])
def test_embedded_fault_consumers_reject_legacy_thermal_source(tmp_path, model_type):
    model = FaultModel(ShellParameters(subdivisions=1), GenesisParameters(), OnsetParameters(),
        TidalParameters(enabled=False, spin_state="synchronous_zero_obliquity"))
    path = tmp_path / "fault.npz"
    save_fault_checkpoint(path, model, *model.initial(), {}, {})
    _rewrite_version(path, None)
    with pytest.raises(ValueError, match="Regenerate.*initial conditions"):
        model_type(path.read_bytes())


def test_standalone_genesis_rejects_previous_thermal_law(tmp_path):
    path = tmp_path / "checkpoint.json"
    thermal = GenesisParameters()
    save_checkpoint(path, initial_state(thermal), thermal)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["model_version"] = "genesis-thermal-0.1"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="[Tt]hermal|[Gg]enesis"):
        load_checkpoint(path)

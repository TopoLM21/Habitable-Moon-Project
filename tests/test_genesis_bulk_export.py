"""Source replay and independent constitutive checks for Genesis bulk export."""
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis import GenesisParameters
from tectonics.genesis_bulk_export import (
    FORMAT, export_genesis_bulk_material, plane_stress_material_tensors,
    validate_genesis_bulk_material,
)
from tectonics.genesis_faults import FaultModel, save_fault_checkpoint
from tectonics.genesis_material import face_frames
from tectonics.genesis_onset import OnsetParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_tides import TidalParameters


@pytest.fixture(scope="module")
def molten_source(tmp_path_factory):
    model = FaultModel(ShellParameters(subdivisions=1), GenesisParameters(),
                       OnsetParameters(), TidalParameters(enabled=False))
    state, thermal, orbit = model.initial()
    path = tmp_path_factory.mktemp("bulk_export")/"actual_molten_fault.npz"
    save_fault_checkpoint(path, model, state, thermal, orbit, {},
                          {"initialization": "FaultModel.initial() from molten birth"})
    return path.read_bytes(), model, state, thermal, orbit


@pytest.fixture(scope="module")
def molten_packet(molten_source):
    return export_genesis_bulk_material(molten_source[0],
        {"purpose": "actual molten producer export contract"})


def _reseal(packet):
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    packet["packet_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_actual_molten_export_preserves_source_and_has_no_invented_solid_memory(molten_source,
                                                                           molten_packet):
    source_bytes, model, state, thermal, orbit = molten_source
    packet = molten_packet
    assert packet["format"] == FORMAT
    assert base64.b64decode(packet["source"]["checkpoint_base64"], validate=True) == source_bytes
    assert packet["source"]["checkpoint_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert packet["time_myr"] == state.time_myr == thermal.time_myr == orbit.time_myr == 0.
    assert packet["birth_source_identity"] == state.bulk_birth_history.source_identity
    assert state.bulk_birth_history.live_mass_kg == 0.
    assert not state.bulk_birth_history.cohorts
    assert packet["solid_birth_cohort_material_catalogue"] == []
    np.testing.assert_array_equal(packet["current_columns"]["layer_mass_kg"], state.layer_mass_kg)
    np.testing.assert_array_equal(packet["current_columns"]["specific_enthalpy_j_kg"],
                                  state.column_enthalpy)
    np.testing.assert_array_equal(packet["current_columns"]["connected_surface_solid_front_fraction"],
                                  np.zeros(model.mesh.cell_count))
    np.testing.assert_array_equal(packet["current_geometry"]["vertices_unit_xyz"], state.vertices)
    assert packet["instantaneous_plane_stress_3d"] is None
    assert not packet["capabilities"]["plane_stress_defined"]
    assert not packet["capabilities"]["physical_force_work_admission"]
    assert all(value is None for value in packet["unknown_fields"].values())
    catalogue = packet["original_material_layer_catalogue"]
    assert len(catalogue) == model.mesh.cell_count*model.p.column_layers
    assert len({row["material_id"] for row in catalogue}) == len(catalogue)
    assert all(row["current_owner"] is None for row in catalogue)
    assert sum(row["layer_mass_kg"] for row in catalogue) == pytest.approx(state.layer_mass_kg.sum())


def test_json_readback_reconstructs_complete_embedded_source(molten_packet):
    decoded = json.loads(json.dumps(molten_packet))
    replay = validate_genesis_bulk_material(decoded,
        expected_source_checkpoint_sha256=molten_packet["source"]["checkpoint_sha256"])
    assert replay == molten_packet
    assert replay is not decoded
    decoded["current_columns"]["layer_mass_kg"][0][0] *= .9
    assert replay == molten_packet


def test_independently_supplied_source_hash_binds_readback(molten_packet):
    with pytest.raises(ValueError, match="independently supplied source"):
        validate_genesis_bulk_material(molten_packet, expected_source_checkpoint_sha256="0"*64)


@pytest.mark.parametrize("mutation", ["clock", "mass", "velocity", "identity", "geometry",
                                      "duplicate_layer", "false_plane_stress", "false_historical_memory"])
def test_resealed_claims_cannot_replace_embedded_producer(molten_packet, mutation):
    packet = deepcopy(molten_packet)
    if mutation == "clock":
        packet["time_myr"] = .1
    elif mutation == "mass":
        packet["current_columns"]["layer_mass_kg"][0][0] *= .9
    elif mutation == "velocity":
        packet["current_shell_memory"]["velocity_km_myr_world"][0][0] = 10.
    elif mutation == "identity":
        packet["birth_source_identity"] = "f"*64
    elif mutation == "geometry":
        packet["current_geometry"]["radius_km"] += 1.
    elif mutation == "duplicate_layer":
        packet["original_material_layer_catalogue"][1] = deepcopy(
            packet["original_material_layer_catalogue"][0])
    elif mutation == "false_plane_stress":
        packet["capabilities"]["plane_stress_defined"] = True
    else:
        packet["unknown_fields"]["historical_natural_metric"] = [[1., 0.], [0., 1.]]
    _reseal(packet)
    with pytest.raises(ValueError, match="full source replay"):
        validate_genesis_bulk_material(packet)


def test_resealed_source_hash_and_invalid_source_bytes_are_rejected(molten_packet):
    altered_hash = deepcopy(molten_packet)
    altered_hash["source"]["checkpoint_sha256"] = "0"*64
    _reseal(altered_hash)
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_genesis_bulk_material(altered_hash)
    malformed = deepcopy(molten_packet)
    malformed["source"]["checkpoint_base64"] = "not canonical base64"
    _reseal(malformed)
    with pytest.raises(ValueError):
        validate_genesis_bulk_material(malformed)


def test_old_fault_format_cannot_acquire_bulk_history_by_relabelling(molten_source):
    source_bytes = molten_source[0]
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as data:
        payload = {name: data[name].copy() for name in data.files}
    metadata = json.loads(str(payload["metadata"]))
    metadata["format"] = "genesis-faults-0.1"
    payload["metadata"] = np.asarray(json.dumps(metadata))
    target = io.BytesIO()
    np.savez_compressed(target, **payload)
    with pytest.raises(ValueError):
        export_genesis_bulk_material(target.getvalue(), {"purpose": "reject old missing history"})


def test_explicit_missing_bulk_history_is_refused(molten_source, tmp_path):
    _, model, state, thermal, orbit = molten_source
    missing = replace(state, bulk_birth_history=None, cohort_mechanics=None)
    path = tmp_path/"mechanical_state_without_birth_history.npz"
    save_fault_checkpoint(path, model, missing, thermal, orbit, {},
                          {"purpose": "legacy mechanical diagnostic, no birth genealogy"})
    with pytest.raises(ValueError):
        export_genesis_bulk_material(path.read_bytes(), {"purpose": "requires calculated history"})


@pytest.mark.parametrize("provenance", [{}, [], {"bad": float("nan")}, {"tuple": (1, 2)}])
def test_export_requires_explicit_literal_json_provenance(molten_source, provenance):
    with pytest.raises(ValueError):
        export_genesis_bulk_material(molten_source[0], provenance)


def test_export_requires_actual_immutable_source_bytes(molten_source):
    with pytest.raises(ValueError):
        export_genesis_bulk_material(bytearray(molten_source[0]), {"purpose": "invalid mutable source"})


def test_plane_stress_extension_matches_independent_three_dimensional_hooke_energy():
    # This is an analytical constitutive control, not a historical planet state.
    strain = np.array([[2e-4, -3e-4, 4e-4], [-5e-4, 1e-4, -8e-4], [1e-3, 1e-3, 0.]])
    damage = np.array([0., .4, .95])
    E, nu, residual = 73e9, .29, .015
    rotations = Rotation.from_rotvec([[.3, -.1, .4], [-.5, .2, .1], [.1, .1, -.3]]).as_matrix()
    result = plane_stress_material_tensors(strain, damage, rotations[:, :, :2],
        young_modulus_pa=E, poisson_ratio=nu, residual_stiffness=residual)
    local_strain, local_stress = result["elastic_strain_local"], result["stress_local_pa"]
    effective = E*(residual+(1-residual)*(1-damage)**2)
    lame = effective*nu/((1+nu)*(1-2*nu))
    shear = effective/(2*(1+nu))
    hooke = (lame*np.trace(local_strain, axis1=1, axis2=2))[:, None, None]*np.eye(3)
    hooke += 2*shear[:, None, None]*local_strain
    np.testing.assert_allclose(local_stress, hooke, rtol=2e-14, atol=2e-8)
    np.testing.assert_array_equal(local_stress[:, 2, :], 0.)
    np.testing.assert_array_equal(local_stress[:, :, 2], 0.)
    # Independent shell D matrix gives the original engineering-shear energy.
    D = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])/(1-nu**2)
    shell_stress = effective[:, None]*(strain@D.T)
    shell_energy = .5*np.sum(strain*shell_stress, axis=1)
    np.testing.assert_allclose(result["elastic_energy_density_j_m3"], shell_energy,
                               rtol=2e-14, atol=0.)
    world_energy = .5*np.einsum("fij,fij->f", result["elastic_strain_world"],
                               result["stress_world_pa"])
    np.testing.assert_allclose(world_energy, shell_energy, rtol=2e-14, atol=0.)


def test_plane_stress_extension_is_objective_under_world_rotation(molten_source):
    mesh = molten_source[1].mesh
    frames = face_frames(mesh)
    strain = np.column_stack((np.linspace(-1e-3, 1e-3, mesh.cell_count),
                             np.full(mesh.cell_count, 2e-4),
                             np.linspace(5e-4, -2e-4, mesh.cell_count)))
    damage = np.linspace(0, 1, mesh.cell_count)
    p = molten_source[1].p
    options = {"young_modulus_pa": p.young_modulus_pa, "poisson_ratio": p.poisson_ratio,
               "residual_stiffness": p.residual_stiffness}
    original = plane_stress_material_tensors(strain, damage, frames, **options)
    Q = Rotation.from_rotvec([.71, -.41, .12]).as_matrix()
    rotated = plane_stress_material_tensors(strain, damage, Q@frames, **options)
    for key in ("elastic_strain_world", "stress_world_pa"):
        np.testing.assert_allclose(rotated[key], Q@original[key]@Q.T, rtol=2e-12, atol=1e-8)
    np.testing.assert_array_equal(rotated["elastic_energy_density_j_m3"],
                                  original["elastic_energy_density_j_m3"])


def test_plane_stress_rejects_nonorthonormal_basis():
    with pytest.raises(ValueError, match="orthonormal"):
        plane_stress_material_tensors(np.zeros((1, 3)), np.zeros(1),
            np.array([[[1., 0.], [0., 2.], [0., 0.]]]),
            young_modulus_pa=60e9, poisson_ratio=.25, residual_stiffness=.02)

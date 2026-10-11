"""Strict export of a Genesis material history from its actual checkpoint.

This is a separate producer contract, not a native943 packet adapter.  The
source bytes remain embedded so a readback reconstructs the material record
against the real saved producer rather than trusting a re-sealed JSON claim.
An instantaneous plane-stress extension of shell elastic memory is explicitly
labelled as such; it does not recover a historical three-dimensional strain,
placement, natural metric, or force/work admission.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math

import numpy as np

from .genesis import temperatures
from .genesis_faults import load_fault_checkpoint
from .genesis_material import face_frames


FORMAT = "genesis-bulk-material-export-0.2"
SCOPE = "GENESIS_BULK_BIRTH_AND_PROSPECTIVE_COHORT_MAXWELL_MEMORY"
PLANE_STRESS_MODEL = "ISOTROPIC_CURRENT_SHELL_ELASTIC_PLANE_STRESS_EXTENSION"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _require_json(value):
    """Require literal JSON types; bool/int and typed aliases are distinct."""
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("Genesis bulk export object keys must be strings")
        for member in value.values():
            _require_json(member)
    elif type(value) is list:
        for member in value:
            _require_json(member)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Genesis bulk export requires finite JSON numbers")
    elif type(value) not in (str, int, bool, type(None)):
        raise ValueError("Genesis bulk export requires literal JSON types")


def plane_stress_material_tensors(elastic_strain, damage, face_frame_xyz, *,
                                 young_modulus_pa, poisson_ratio,
                                 residual_stiffness):
    """Extend the *current* shell elastic tensor under explicit plane stress.

    Input strain is ``[xx, yy, gamma_xy]`` in orthonormal chord-face axes.
    Engineering shear is divided by two in the tensor.  Isotropic plane stress
    determines ``epsilon_zz = -nu/(1-nu)*(epsilon_xx+epsilon_yy)``; normal and
    transverse shear stresses vanish by this declared closure.  The resulting
    tensors are not total deformation or a recovered bulk birth history.
    """
    strain = np.asarray(elastic_strain, dtype=float)
    damage = np.asarray(damage, dtype=float)
    frames = np.asarray(face_frame_xyz, dtype=float)
    if (strain.ndim != 2 or strain.shape[1] != 3
            or damage.shape != (len(strain),)
            or frames.shape != (len(strain), 3, 2)
            or not np.isfinite(strain).all() or not np.isfinite(damage).all()
            or not np.isfinite(frames).all()
            or np.any((damage < 0) | (damage > 1))):
        raise ValueError("Plane stress requires finite face strain, damage and frames")
    for value in (young_modulus_pa, poisson_ratio, residual_stiffness):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError("Plane-stress parameters must be finite physical numbers")
    E, nu, residual = float(young_modulus_pa), float(poisson_ratio), float(residual_stiffness)
    if E <= 0 or not 0 < nu < .5 or not 0 < residual <= 1:
        raise ValueError("Plane-stress elastic parameters are outside their physical range")
    if not np.allclose(np.einsum("fji,fjk->fik", frames, frames),
                       np.eye(2), rtol=0, atol=2e-12):
        raise ValueError("Plane-stress face axes must be orthonormal")
    basis = np.concatenate((frames,
        np.cross(frames[:, :, 0], frames[:, :, 1])[:, :, None]), axis=2)
    effective_E = E*(residual+(1-residual)*(1-damage)**2)
    local_strain = np.zeros((len(strain), 3, 3))
    local_strain[:, 0, 0], local_strain[:, 1, 1] = strain[:, 0], strain[:, 1]
    local_strain[:, 0, 1] = local_strain[:, 1, 0] = strain[:, 2]/2
    local_strain[:, 2, 2] = -nu/(1-nu)*(strain[:, 0]+strain[:, 1])
    local_stress = np.zeros_like(local_strain)
    factor = effective_E/(1-nu*nu)
    local_stress[:, 0, 0] = factor*(strain[:, 0]+nu*strain[:, 1])
    local_stress[:, 1, 1] = factor*(strain[:, 1]+nu*strain[:, 0])
    local_stress[:, 0, 1] = local_stress[:, 1, 0] = effective_E*strain[:, 2]/(2*(1+nu))
    world_strain = basis@local_strain@basis.transpose(0, 2, 1)
    world_stress = basis@local_stress@basis.transpose(0, 2, 1)
    energy = .5*np.einsum("fij,fij->f", local_strain, local_stress)
    if not np.isfinite(world_strain).all() or not np.isfinite(world_stress).all() or not np.isfinite(energy).all():
        raise ValueError("Plane-stress tensor extension overflowed")
    return {"basis_xyz": basis, "elastic_strain_local": local_strain,
            "elastic_strain_world": world_strain, "stress_local_pa": local_stress,
            "stress_world_pa": world_stress, "elastic_energy_density_j_m3": energy}


def export_genesis_bulk_material(checkpoint_bytes, provenance):
    """Replay a saved Fault producer and export its required bulk birth record."""
    if type(checkpoint_bytes) is not bytes or not checkpoint_bytes:
        raise ValueError("Genesis bulk export requires the actual immutable checkpoint bytes")
    if type(provenance) is not dict or not provenance:
        raise ValueError("Declare Genesis bulk export provenance as a nonempty JSON object")
    _require_json(provenance)
    model, state, thermal, orbit, metadata = load_fault_checkpoint(io.BytesIO(checkpoint_bytes))
    history = getattr(state, "bulk_birth_history", None)
    if history is None:
        raise ValueError("The source has no calculated bulk birth history; repeat Genesis from molten birth")
    # The typed birth-history serialization and source binding are supplied by
    # genesis_bulk_birth.  A complete serialized history is never synthesized
    # from the source's aggregate age or its depth-averaged shell strain.
    from .genesis_bulk_birth import bulk_birth_to_dict, validate_bulk_birth

    mesh = model.mesh_for(state)
    frames = face_frames(mesh)
    tm, ts = temperatures(np.asarray(thermal.energy), model.thermal)
    fraction, _, _ = model._phase(state.column_enthalpy, ts, tm)
    history = validate_bulk_birth(history, layer_mass_kg=state.layer_mass_kg,
                                 time_myr=state.time_myr, front_fraction=fraction)
    history_dict = bulk_birth_to_dict(history)
    from .genesis_cohort_mechanics import cohort_mechanics_to_dict
    mechanics_dict = json.loads(_json(cohort_mechanics_to_dict(state.cohort_mechanics)))
    source_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    parameters = metadata["parameters"]
    birth_source_identity = history.source_identity
    layer_catalogue = [{
        "material_id": f"genesis:{birth_source_identity}:face:{face}:layer:{layer}",
        "original_face_index": face, "layer_index": layer,
        "layer_mass_kg": float(state.layer_mass_kg[face, layer]),
        "current_owner": None,
        "ownership": "original material face association retained; plate owner unresolved"}
        for face in range(mesh.cell_count) for layer in range(model.p.column_layers)]
    column_mass = state.layer_mass_kg.sum(axis=1)
    cohort_catalogue = [{
        "material_id": (f"genesis:{birth_source_identity}:face:{cohort.face_index}:"
                        f"layer:{cohort.layer_index}:cohort:{cohort.id}"),
        "original_layer_material_id": (f"genesis:{birth_source_identity}:"
            f"face:{cohort.face_index}:layer:{cohort.layer_index}"),
        "source_cohort_id": cohort.id,
        "original_face_index": cohort.face_index, "layer_index": cohort.layer_index,
        "formed_mass_kg": float(column_mass[cohort.face_index]*(cohort.q_hi-cohort.q_lo)),
        "live_mass_kg": float(column_mass[cohort.face_index]*(cohort.live_q_hi-cohort.q_lo)),
        "current_owner": None,
        "mass_coordinate": "fraction of fixed original face column mass"}
        for cohort in history.cohorts]
    instantaneous, cohort_tensors = None, None
    if state.membrane_established:
        closure = plane_stress_material_tensors(state.elastic_strain, state.damage, frames,
            young_modulus_pa=model.p.young_modulus_pa, poisson_ratio=model.p.poisson_ratio,
            residual_stiffness=model.p.residual_stiffness)
        solid_volume = state.layer_mass_kg.sum(axis=1)*fraction/model.p.density_kg_m3
        instantaneous = {"model": PLANE_STRESS_MODEL,
            "scope": "instantaneous constitutive extension of volume-mean shell elastic memory",
            "historical_three_dimensional_memory_recovered": False,
            "normal_stress_and_transverse_shear_zero_by_plane_stress_closure": True,
            "solid_volume_m3": solid_volume.tolist(),
            "elastic_energy_of_face_mean_j": float(np.dot(closure["elastic_energy_density_j_m3"], solid_volume)),
            **{key: value.tolist() for key, value in closure.items()}}
        records = state.cohort_mechanics.cohorts
        indices = np.asarray([record.face_index for record in records], dtype=int)
        local = plane_stress_material_tensors(
            np.asarray([record.elastic_strain for record in records]), state.damage[indices], frames[indices],
            young_modulus_pa=model.p.young_modulus_pa, poisson_ratio=model.p.poisson_ratio,
            residual_stiffness=model.p.residual_stiffness)
        volumes = np.asarray([column_mass[record.face_index]*(record.live_q_hi-record.q_lo)
            /model.p.density_kg_m3 for record in records])
        cohort_tensors = {"model": PLANE_STRESS_MODEL,
            "scope": "current live cohort plane-stress memory from declared mechanical activation",
            "source_cohort_ids": [record.id for record in records],
            "original_face_indices": indices.tolist(), "live_volume_m3": volumes.tolist(),
            "elastic_energy_j": float(np.dot(local["elastic_energy_density_j_m3"], volumes)),
            "full_three_dimensional_formation_history_recovered": False,
            **{key: value.tolist() for key, value in local.items()}}
        if not math.isclose(cohort_tensors["elastic_energy_j"],
                state.cohort_mechanics.elastic_energy_j, rel_tol=2e-12, abs_tol=1e-6):
            raise ValueError("Cohort export energy disagrees with its constitutive source")
    ledgers = {name: getattr(state, name) for name in (
        "initial_column_energy_j", "boundary_energy_j", "tidal_heat_received_j",
        "friction_work_j", "viscous_fault_work_j")}
    packet = {"format": FORMAT, "scope": SCOPE,
        "source": {"checkpoint_sha256": source_sha,
            "checkpoint_base64": base64.b64encode(checkpoint_bytes).decode("ascii"),
            "checkpoint_format": metadata["format"],
            "parameter_hash": metadata["parameter_hash"],
            "parameters": parameters,
            "original_provenance": metadata["provenance"]},
        "birth_source_identity": birth_source_identity,
        "birth_source_context": {
            "configuration": json.loads(history.configuration_json),
            "initial_radius_km": history.initial_radius_km,
            "solidus_k": history.solidus_k,
            "initial_vertices_unit_xyz": [list(row) for row in history.initial_vertices],
            "initial_faces": [list(row) for row in history.initial_faces],
            "birth_metric_semantics": "chord-edge Gram metric at recorded birth geometry; not a natural metric"},
        "time_myr": state.time_myr,
        "clocks_myr": {"material": state.time_myr, "thermal": thermal.time_myr,
                       "orbit": orbit.time_myr},
        "original_material_layer_catalogue": layer_catalogue,
        "solid_birth_cohort_material_catalogue": cohort_catalogue,
        "bulk_birth_history": history_dict,
        "cohort_mechanics": mechanics_dict,
        "current_geometry": {"radius_km": state.radius_km,
            "vertices_unit_xyz": state.vertices.tolist(), "faces": mesh.faces.tolist(),
            "face_area_sr": mesh.areas_unit_sphere.tolist(),
            "face_frames_xyz": frames.tolist(),
            "vertex_and_face_order": "unchanged source material connectivity"},
        "current_columns": {"layer_mass_kg": state.layer_mass_kg.tolist(),
            "specific_enthalpy_j_kg": state.column_enthalpy.tolist(),
            "connected_surface_solid_front_fraction": fraction.tolist()},
        "current_shell_memory": {
            "elastic_strain_xx_yy_gamma_xy": state.elastic_strain.tolist(),
            "strain_convention": "corotational in-plane elastic memory with engineering shear gamma_xy",
            "basis": "current material triangle chord-face orthonormal axes",
            "velocity_km_myr_world": state.velocity_km_myr.tolist(),
            "velocity_convention": "source finite-difference tangent diagnostic rate",
            "damage": state.damage.tolist(), "water_access": state.water_access.tolist(),
            "membrane_established": bool(state.membrane_established)},
        "instantaneous_plane_stress_3d": instantaneous,
        "instantaneous_cohort_plane_stress_3d": cohort_tensors,
        "source_work_and_heat_ledgers": ledgers,
        "unknown_fields": {"historical_three_dimensional_total_strain": None,
            "historical_three_dimensional_prestress": None,
            "historical_three_dimensional_material_placement": None,
            "historical_natural_metric": None,
            "current_plate_ownership": None,
            "physical_force_work_handover": None},
        "capabilities": {"source_checkpoint_replay_required": True,
            "calculated_bulk_birth_history_present": True,
            "calculated_cohort_constitutive_memory_present": True,
            "plane_stress_defined": bool(state.membrane_established),
            "full_historical_three_dimensional_memory_present": False,
            "physical_force_work_admission": False,
            "native943_source_adopted": False,
            "new_native_equilibrium_accepted": False},
        "provenance": json.loads(_json(provenance))}
    _require_json(packet)
    packet["packet_sha256"] = _digest(packet)
    return packet


def validate_genesis_bulk_material(packet, *, expected_source_checkpoint_sha256=None):
    """Rebuild every field from the embedded source, including re-sealed packets."""
    _require_json(packet)
    if type(packet) is not dict or packet.get("format") != FORMAT:
        raise ValueError("Invalid Genesis bulk material export format")
    try:
        body = {key: value for key, value in packet.items() if key != "packet_sha256"}
        if packet["packet_sha256"] != _digest(body):
            raise ValueError("Genesis bulk export packet seal changed")
        raw = packet["source"]["checkpoint_base64"]
        if type(raw) is not str:
            raise ValueError("Genesis bulk export source must use canonical base64 text")
        source_bytes = base64.b64decode(raw, validate=True)
        if base64.b64encode(source_bytes).decode("ascii") != raw:
            raise ValueError("Genesis bulk export source has noncanonical base64 encoding")
        if hashlib.sha256(source_bytes).hexdigest() != packet["source"]["checkpoint_sha256"]:
            raise ValueError("Genesis bulk export source checkpoint hash changed")
        if (expected_source_checkpoint_sha256 is not None
                and (type(expected_source_checkpoint_sha256) is not str
                     or packet["source"]["checkpoint_sha256"] != expected_source_checkpoint_sha256)):
            raise ValueError("Genesis bulk export differs from the independently supplied source checkpoint")
        replay = export_genesis_bulk_material(source_bytes, packet["provenance"])
        if _json(packet) != _json(replay):
            raise ValueError("Genesis bulk export differs from the full source replay")
        return replay
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed Genesis bulk material export") from error

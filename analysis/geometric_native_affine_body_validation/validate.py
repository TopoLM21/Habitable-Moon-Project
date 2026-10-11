"""Fresh supported affine-body checks preserving the sealed 820-source epoch.

This gate checks a finite dimensional body assembly on authentic source masks.
It does not rebuild the genuine case, solve the mixed force problem, approve
material admission, or accept a new physical interval.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_normal_support_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules
from analysis.geometric_native_spherical_solid_validation.validate import _verify_outputs

RELEASE = ROOT / "analysis/geometric_native_normal_support_validation/release_validation_01"
EXPECTED_REPORT = "3c028106720bd00b8d31bf8ef4e050d7aaaeb5cd34b47ada8f2f252fef51560a"
EXPECTED_REPORT_SEAL = "ea6ea35006e92d8cea21ebf29e5d3f722f8e93b905aca9f5a52447599b9f18f9"
EXPECTED_NATIVE_REPORT = "8cff7debbd4b6a274860ca8042ad8b9d8462d59fe8f2ed5e5aefcd7369078bd5"
EXPECTED_NATIVE_REPORT_SEAL = "5b051b4d2e6b3b12448b6088d4dc1922fa7048d61da7772035249de34ed0b530"
EXPECTED_PACKET = "f4f3703cbc463d4130096dba827a7425fc865851fe5c86f337f98dfbefd8f7ca"
EXPECTED_PACKET_SEAL = "e2db27aa164161482117993f4f23010293eae45100c91d3a52ff0508075688ee"
EXPECTED_SOURCE_MAP = "20ab172bee8cc7b865834817a20f7415e971bd1d60964c853519f9d43d250fc6"
PHYSICAL_COMPONENT_SCOPE = "COMPONENT_AFFINE_BODY_ONLY"
STANDALONE_EVIDENCE = {
    "independent_audit.json": "45eabeb97f8dca5abac94c26f044f8e0babd47077c6dab6bb976637af2989678",
    "source_shortened_columns_refusal.json": "10964b5609415a60b8f3e0eda0822ed37378c787393d48f4a9e83117cbf287f0",
    "source_normal_independent_audit.json": "cdb8964417ff1c781850c900717b9a541fff918c7c33cbbee1ca06a3a3202097",
}
SOURCE_FIELDS = previous.SOURCE_FIELDS
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_affine_body.py",
    "tectonics/geometric_native_affine_bulk_assembly.py",
    "tests/test_geometric_native_affine_body.py",
    "tests/test_geometric_native_affine_bulk_assembly.py",
    "analysis/geometric_native_affine_body_validation/validate.py",
    "analysis/geometric_native_affine_body_validation/native_control.py",
))

encoded = previous.encoded
digest = previous.digest
value_digest = previous.value_digest
read = previous.read
sealed = previous.sealed
source_map = previous.source_map
parse_counts = previous.parse_counts
_literal_fraction = previous._literal_fraction
_interval = previous._interval


def verify_prior():
    """Authenticate all 820 sources, historical outputs and standalone proofs."""
    inherited = previous.verify_prior()
    path = RELEASE / "report.json"
    if digest(path) != EXPECTED_REPORT:
        raise ValueError("Authoritative normal-support regression report changed")
    report = sealed(path, "report_sha256")
    if (read(path).get("report_sha256") != EXPECTED_REPORT_SEAL
            or report.get("status") != "PASS"
            or report.get("regression_validation_status") != "PASS"
            or report.get("physical_candidate_status") != previous.PHYSICAL_REJECTION
            or report.get("source_count") != 820
            or report.get("pytest_counts") != {"passed": 57}
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or set(report.get("added_python_sources", {})) != previous.ADDED_SOURCE_PATHS
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("Normal-support source generation or physical rejection changed")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 820 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The original 820-source generation differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    _verify_outputs(RELEASE, report, 4)
    expected_files = set(report["outputs"]) | {"report.json"}
    if {str(path.relative_to(RELEASE)) for path in RELEASE.rglob("*")
            if path.is_file()} != expected_files:
        raise ValueError("The sealed normal-support release gained or lost a file")
    native_path = RELEASE / "native_control/report.json"
    packet_path = RELEASE / "native_control/source_packet.json"
    if (digest(native_path) != EXPECTED_NATIVE_REPORT
            or digest(packet_path) != EXPECTED_PACKET
            or read(native_path).get("report_sha256") != EXPECTED_NATIVE_REPORT_SEAL
            or read(packet_path).get("packet_sha256") != EXPECTED_PACKET_SEAL):
        raise ValueError("The authoritative normal-support evidence changed")
    previous._check_native_control(RELEASE)
    for name, expected in STANDALONE_EVIDENCE.items():
        path = RELEASE.parent / name
        if digest(path) != expected:
            raise ValueError("A preserved standalone source proof changed: " + name)
        row = read(path)
        if "report_sha256" in row:
            sealed(path, "report_sha256")
        if (row.get("original_point_1_completed") is not False
                or row.get("material_admission_authorized", False) is not False):
            raise ValueError("A standalone refusal gained physical admission")
    return inherited


def _check_component_assembly(native, packet):
    """Authenticate the component-only domain, stiffness and evaluation evidence."""
    from tectonics.geometric_native_normal_support import load_native_normal_support_source

    rows = native.get("body_components")
    if (type(rows) is not list or len(rows) != 5
            or encoded(rows) != encoded(packet.get("affine_body_components"))):
        raise ValueError("All five identical source HOT-body proofs are required in both outputs")
    epoch = packet["before"]["snapshot"]["current_material_epoch"]
    support = packet["full_reference_face_birth_support"]
    reference = packet["saved_joint_reference_field"]
    components = {row["component_id"]: row for row in epoch["components"]
                  if row["material_kind"] == "hot_birth"}
    if len(components) != 5 or {row.get("component_id") for row in rows} != set(components):
        raise ValueError("An affine assembly omitted, repeated or substituted an actual HOT bank")
    source = load_native_normal_support_source()
    source_banks = {bank.component_id: bank for bank in source.banks}
    for row in rows:
        identity = row["component_id"]
        component = components[identity]
        domain = row.get("domain")
        if type(domain) is not dict:
            raise ValueError("An actual HOT bank requires its complete source geometry receipt")
        bank = source_banks[identity]
        births = [value for value in epoch["source_material_state"]["endpoint_vacancies"]
            if value["newborn"] is not None
            and value["newborn"]["vacancy_domain_id"] == identity]
        groups = [group for group in reference["groups"] if identity in group["component_ids"]]
        if len(births) != 1 or len(groups) != 1:
            raise ValueError("A true HOT bank lost its unique physical birth or saved group")
        birth, group = births[0], groups[0]
        corrections = [value for value in epoch["paired_corrections"]
            if identity in (value["negative_component_id"], value["positive_component_id"])]
        decoded_corrections = [json.loads(value)
            for value in domain.get("paired_corrections_source_json", [])]
        if (domain.get("component_id") != identity
                or domain.get("material_kind") != "hot_birth"
                or domain.get("material_id") != component["material_id"]
                or domain.get("original_parent_id") != component["original_parent_id"]
                or domain.get("owner") != component["owner"]
                or json.loads(domain["complete_mask_source_json"]) != component
                or json.loads(domain["complete_birth_source_json"]) != birth
                or encoded(decoded_corrections) != encoded(corrections)
                or json.loads(domain["saved_field_json"]) != reference
                or domain.get("source_snapshot_sha256") != source.source_snapshot_sha256
                or domain.get("source_support_sha256") != support["support_sha256"]
                or domain.get("source_epoch_sha256") != epoch["epoch_sha256"]
                or domain.get("source_packet_raw_sha256") != previous.EXPECTED_PACKET
                or domain.get("saved_field_declaration_precision") != 128
                or tuple(_exact(value) for value in domain["reference_origin_m"]) != (Fraction(0),)*3
                or domain.get("coordinate_order") != ["xx", "yy", "zz", "xy", "xz", "yz"]
                or tuple(_matrix(value) for value in domain["symmetric_generators"]) != _basis()
                or not str(domain.get("stressfree_reference_provenance", "")).strip()
                or not str(domain.get("provenance", "")).strip()):
            raise ValueError("An affine body changed its complete actual mask, allocation or saved reference")
        body_seal = dict(domain)
        expected_geometry_seal = body_seal.pop("geometry_source_sha256", None)
        if expected_geometry_seal != value_digest(body_seal):
            raise ValueError("An authentic affine body geometry selfseal differs")
        radius = _exact(domain["radius_m"])
        thickness = _exact(domain["areal_material_thickness_m"])
        roof = _exact(domain["roof_radial_displacement_m"])
        axes = tuple(_exact(value) for value in domain["angular_rate_per_physical_volume_depth_m"])
        group_axes = [family.shared_depth_axis_per_m for family in source.families
            if family.family_id in group["family_ids"]]
        expected_axes = group_axes[0] if group_axes else (Fraction(0),)*3
        newborn = birth["newborn"]
        properties = tuple(_literal_fraction(value) for value in newborn["specific_properties"][:2])
        expected_thickness = sum(properties, Fraction(0))*1000
        area = _interval(component["area_bounds_sr"])
        volume = _scaled_interval(area, radius**2*thickness)
        if (radius != source.radius_m or thickness != expected_thickness
                or thickness != bank.areal_material_thickness_m
                or roof != bank.roof_radial_displacement_m
                or axes != expected_axes or any(value != expected_axes for value in group_axes)
                or radius <= 0 or thickness <= 0 or (radius+roof)**3-3*radius**2*thickness <= 0
                or area[0] <= 0 or area != _interval(birth["area_bounds_sr"])
                or _interval(domain["actual_area_bounds_sr"]) != area
                or _interval(domain["reference_volume_bounds_m3"]) != volume):
            raise ValueError("Actual whole-mask geometry, volume or saved depth field differs")
        closure = row.get("bulk_closure")
        if type(closure) is not dict:
            raise ValueError("The finite body requires an explicit source-derived new bulk closure")
        columns = closure.get("source_columns")
        source_columns = [carrier["declaration"]["parameters"]["columns"]
                          for carrier in support["interface_carriers"]]
        source_bank = closure.get("source_bank")
        if (source_bank not in ("sub", "over")
                or not any(encoded(columns) == encoded(value) for value in source_columns)
                or closure.get("bank_source_id") != identity
                or closure.get("source_support_sha256") != support["support_sha256"]
                or closure.get("lateral_closure") != "laterally_free_standard_isotropic"
                or not str(closure.get("lateral_closure_provenance", "")).strip()):
            raise ValueError("The new HOT body tensor changed its declared authentic moduli")
        young = Fraction.from_float(columns[source_bank+"_normal_modulus_pa"])
        shear = Fraction.from_float(columns[source_bank+"_shear_modulus_pa"])
        coordinates = tuple(_exact(value) for value in row["probe_coordinates_m"])
        direction = tuple(_exact(value) for value in row["probe_direction_m"])
        first_declaration = support["interface_carriers"][0]["declaration"]
        probe_columns = first_declaration["parameters"]["columns"]
        probe_law = first_declaration["parameters"]["law"]
        compliance = sum((Fraction.from_float(probe_columns[name+"_length_m"])
            /Fraction.from_float(probe_columns[name+"_normal_modulus_pa"])
            for name in ("sub", "over")), Fraction(0))
        compliance += 1/Fraction.from_float(probe_law["normal_stiffness_pa_m"])
        pressure = -Fraction.from_float(first_declaration["birth_gap_free_m"])/compliance
        poisson = young/(2*shear)-1
        amplitude = radius*pressure/young
        expected_coordinates = tuple(amplitude*x for x in
            (Fraction(-1), poisson, poisson, Fraction(1, 7), Fraction(-1, 11), Fraction(1, 13)))
        if (len(coordinates) != 6 or len(direction) != 6
                or coordinates != expected_coordinates or direction != coordinates
                or _exact(row["source_pressure_probe_pa"]) != pressure
                or row.get("probe_is_new_mixed_force_solution") is not False
                or row.get("affine_volume_and_basal_probe_use_same_six_mode_field") is not True
                or row.get("independent_radial_boundary_offset_fixed_zero") is not True
                or row.get("radial_mean_volume_extension_implemented") is not False
                or row.get("complete_mixed_boundary_conditions_satisfied") is not False
                or row.get("old_scalar_foundation_energy_added_again") is not False):
            raise ValueError("The diagnostic must exercise nonzero complete six-coordinate inputs")
        for key, amplitudes, probe_direction in (
                ("probe_bulk_response", coordinates, direction),
                ("zero_bulk_response", (Fraction(0),)*6, None)):
            response = row[key]
            rotation = _matrix(response["spatial_rotation"])
            replay = _bulk_replay(radius, young, shear, amplitudes, rotation)
            if tuple(_exact(value) for value in response["coordinates_m"]) != amplitudes:
                raise ValueError("An exported finite-body response substituted its diagnostic amplitudes")
            for name, expected in (
                    ("body_deformation_gradient", replay["body_gradient"]),
                    ("deformation_gradient", replay["gradient"]),
                    ("green_strain", replay["green"]),
                    ("second_piola_stress_pa", replay["second_piola"]),
                    ("first_piola_stress_pa", replay["first_piola"])):
                if _matrix(response[name]) != expected:
                    raise ValueError("A complete Green/StVK tensor differs from independent exact replay")
            if (_exact(response["energy_density_reference_j_m3"]) != replay["energy_density"]
                    or _exact(response["volume_jacobian"]) != replay["jacobian"]):
                raise ValueError("The actual-mask finite body lost its full density or exact Jacobian")
            _check_outward(response["physical_volume_bounds_m3"],
                _scaled_interval(volume, replay["jacobian"]))
            _check_outward(response["energy_bounds_j"],
                _scaled_interval(volume, replay["energy_density"]))
            for value, density in zip(response["generalized_gradient_bounds_n"], replay["gradient_density"]):
                _check_outward(value, _scaled_interval(volume, density))
            if (len(response["generalized_gradient_bounds_n"]) != 6
                    or len(response["generalized_hessian_bounds_n_m"]) != 6
                    or any(len(values) != 6 for values in response["generalized_hessian_bounds_n_m"])):
                raise ValueError("All six force rows and all36 Hessian terms are required")
            for values, densities in zip(response["generalized_hessian_bounds_n_m"], replay["hessian_density"]):
                for value, density in zip(values, densities):
                    _check_outward(value, _scaled_interval(volume, density))
            if probe_direction is not None:
                first = sum((a*b for a, b in zip(replay["gradient_density"], probe_direction)), Fraction(0))
                second = sum((probe_direction[i]*replay["hessian_density"][i][j]*probe_direction[j]
                    for i in range(6) for j in range(6)), Fraction(0))
                _check_outward(response["first_directional_derivative_bounds_j"], _scaled_interval(volume, first))
                _check_outward(response["second_directional_derivative_bounds_j"], _scaled_interval(volume, second))
            elif (response["first_directional_derivative_bounds_j"] is not None
                    or response["second_directional_derivative_bounds_j"] is not None):
                raise ValueError("An absent directional probe acquired unrequested work")
        stiffness = _matrix(row["bulk_stiffness_per_reference_volume_n_m4"], 6, 6)
        if stiffness != replay["stiffness_density"]:
            raise ValueError("Actual integrated stiffness differs from all six independent tensor modes")
        enclosed = row["bulk_stiffness_bounds_n_m"]
        if len(enclosed) != 6 or any(len(values) != 6 for values in enclosed):
            raise ValueError("Actual whole-mask stiffness requires every source-volume scaled coefficient")
        for values, densities in zip(enclosed, stiffness):
            for value, density in zip(values, densities):
                _check_outward(value, _scaled_interval(volume, density))
        _check_basal_component(row, packet, source)
    _check_source_faces_and_compliance(native, packet, source, rows)


def _check_basal_component(row, packet, source):
    """Replay source basal geometry, then independently check its full six-mode law."""
    from tectonics.geometric_native_affine_body import (
        bind_native_affine_hot_body, bind_native_affine_basal_source,
        native_affine_basal_source_to_dict,
    )

    raw = row["basal_source"]
    body = bind_native_affine_hot_body(row["component_id"], source=source,
        provenance=row["domain"]["provenance"])
    replayed_source = bind_native_affine_basal_source(body,
        precision=raw["precision"], order=raw["order"])
    if encoded(native_affine_basal_source_to_dict(replayed_source)) != encoded(raw):
        raise ValueError("An entire basal mask differs from its actual source geometry replay")
    radius = _exact(row["domain"]["radius_m"])
    area = _interval(raw["actual_area_bounds_sr"])
    reference_area = _scaled_interval(area, radius**2)
    if (_interval(raw["reference_measure_area_bounds_m2"]) != reference_area
            or area != _interval(row["domain"]["actual_area_bounds_sr"])):
        raise ValueError("A basal law replaced its full source R-squared footprint")
    coefficient = _exact(raw["density_kg_m3"])*_exact(raw["gravity_m_s2"])
    base_radius = _interval(raw["base_radius_bounds_m"])
    direction = tuple(_interval(value) for value in raw["whole_mask_unit_direction_bounds"])
    covector = []
    for generator in _basis():
        quadratic = _interval_sum(_scaled_interval(_interval_square(direction[i]), generator[i][i])
                                 for i in range(3))
        quadratic = _interval_add(quadratic,
            _interval_sum(_scaled_interval(_interval_multiply(direction[i], direction[j]),
                2*generator[i][j]) for i in range(3) for j in range(i)))
        covector.append(_interval_multiply(_scaled_interval(base_radius, 1/radius), quadratic))
    covector = tuple(covector)
    foundation = row["basal_assembly"]
    if (foundation["dof_names"] != ["xx", "yy", "zz", "xy", "xz", "yz"]
            or _exact(foundation["foundation_stiffness_density_pa_m"]) != coefficient
            or _interval(foundation["reference_measure_area_bounds_m2"]) != reference_area
            or tuple(_interval(value) for value in foundation["normal_displacement_covector_bounds"]) != covector):
        raise ValueError("The six affine volume modes acquired a different basal field")
    weighted_area = _scaled_interval(reference_area, coefficient)
    stiffness = tuple(tuple(_interval_multiply(weighted_area,
        _interval_square(first) if i == j else _interval_multiply(first, second))
        for j, second in enumerate(covector)) for i, first in enumerate(covector))
    _check_interval_matrix(foundation["stiffness_bounds_n_m"], stiffness)
    response = row["basal_probe_response"]
    coordinates = tuple(_exact(value) for value in row["probe_coordinates_m"])
    variation = tuple(_exact(value) for value in row["probe_direction_m"])
    if (tuple(_exact(value) for value in response["coordinates_m"]) != coordinates
            or _exact(response["external_offset_m"]) != 0):
        raise ValueError("A basal probe added an unsupported radial mean body mode")
    displacement = _interval_sum(_scaled_interval(value, coordinate)
                                 for value, coordinate in zip(covector, coordinates))
    displacement_variation = _interval_sum(_scaled_interval(value, coordinate)
                                           for value, coordinate in zip(covector, variation))
    _check_outward(response["whole_mask_normal_displacement_bounds_m"], displacement)
    _check_outward(response["energy_bounds_j"], _scaled_interval(
        _interval_multiply(weighted_area, _interval_square(displacement)), Fraction(1, 2)))
    gradient = tuple(_interval_multiply(_interval_multiply(weighted_area, displacement), value)
                     for value in covector)
    if len(response["generalized_gradient_bounds_n"]) != 6:
        raise ValueError("The supported basal component requires all six actual force rows")
    for actual, expected in zip(response["generalized_gradient_bounds_n"], gradient):
        _check_outward(actual, expected)
    _check_interval_matrix(response["generalized_hessian_bounds_n_m"], stiffness)
    _check_outward(response["first_directional_derivative_bounds_j"], _interval_multiply(
        _interval_multiply(weighted_area, displacement), displacement_variation))
    _check_outward(response["second_directional_derivative_bounds_j"],
        _interval_multiply(weighted_area, _interval_square(displacement_variation)))
    bulk_stiffness = tuple(tuple(_interval(value) for value in values)
                           for values in row["bulk_stiffness_bounds_n_m"])
    combined_stiffness = tuple(tuple(_interval_add(a, b) for a, b in zip(values, other))
                              for values, other in zip(bulk_stiffness, stiffness))
    _check_interval_matrix(row["combined_hot_reference_stiffness_bounds_n_m"], combined_stiffness)
    combined = row["combined_hot_probe_response"]
    bulk_response = row["probe_bulk_response"]
    for key in ("energy_bounds_j", "first_directional_derivative_bounds_j",
                "second_directional_derivative_bounds_j"):
        _check_outward(combined[key], _interval_add(_interval(bulk_response[key]), _interval(response[key])))
    if len(combined["generalized_gradient_bounds_n"]) != 6:
        raise ValueError("Combined HOT energy requires the same complete six body forces")
    for actual, bulk, basal in zip(combined["generalized_gradient_bounds_n"],
            bulk_response["generalized_gradient_bounds_n"], response["generalized_gradient_bounds_n"]):
        _check_outward(actual, _interval_add(_interval(bulk), _interval(basal)))
    combined_hessian = tuple(tuple(_interval_add(_interval(a), _interval(b))
        for a, b in zip(values, other)) for values, other in zip(
            bulk_response["generalized_hessian_bounds_n_m"], response["generalized_hessian_bounds_n_m"]))
    _check_interval_matrix(combined["generalized_hessian_bounds_n_m"], combined_hessian)


def _check_source_faces_and_compliance(native, packet, source, bodies):
    """Bind whole source face traces and independently reconstruct shared B K^-1 B^T."""
    from tectonics.geometric_native_affine_body import (
        bound_native_affine_source_face_trace, native_affine_source_face_trace_to_dict,
    )

    faces = native.get("native_face_traces")
    shared = native.get("shared_hot_bank_compliance")
    if (type(faces) is not list or len(faces) != 3
            or encoded(faces) != encoded(packet.get("native_face_traces"))
            or type(shared) is not list or len(shared) != 1
            or encoded(shared) != encoded(packet.get("shared_hot_bank_compliance"))):
        raise ValueError("All three real HOT face traces and the shared-bank proof are required")
    body_by_id = {row["component_id"]: row for row in bodies}
    expected_pairs = {(family.family_id, bank) for family in source.families
        for bank, identity in (("donor", family.donor_component_id), ("receiver", family.receiver_component_id))
        if identity in body_by_id}
    if {(face["family_id"], face["bank"]) for face in faces} != expected_pairs:
        raise ValueError("A source pressure trace substituted, repeated or omitted a physical bank")
    by_body = {}
    for raw in faces:
        replay = bound_native_affine_source_face_trace(raw["family_id"], raw["bank"], source=source,
            depth_cells=raw["depth_cell_count"], precision=raw["precision"], order=raw["order"])
        if encoded(native_affine_source_face_trace_to_dict(replay)) != encoded(raw):
            raise ValueError("A pressure trace differs from its complete source activation face")
        by_body.setdefault(raw["component_id"], []).append(raw)
    shared_ids = {identity for identity, values in by_body.items() if len(values) > 1}
    if {row["component_id"] for row in shared} != shared_ids:
        raise ValueError("A shared bank lost its actual cross-contact coupling")
    for row in shared:
        body = body_by_id[row["component_id"]]
        traces = by_body[row["component_id"]]
        operators = tuple(tuple(_interval(value) for value in trace["generalized_normal_force_bounds_n_per_pa"])
                          for trace in traces)
        if (row["face_ids"] != [trace["family_id"] for trace in traces]
                or tuple(tuple(_interval(value) for value in values)
                    for values in row["face_operators_bounds_m2"]) != operators):
            raise ValueError("A common body compliance replaced the actual outward pressure operators")
        closure = body["bulk_closure"]
        bank = closure["source_bank"]
        columns = closure["source_columns"]
        young = Fraction.from_float(columns[bank+"_normal_modulus_pa"])
        shear = Fraction.from_float(columns[bank+"_shear_modulus_pa"])
        poisson = young/(2*shear)-1
        lame = 2*shear*poisson/(1-2*poisson)
        inverse = _inverse_bulk_stiffness(_exact(body["domain"]["radius_m"]), shear, lame)
        if _matrix(row["stiffness_inverse_per_reference_volume_m4_n"], 6, 6) != inverse:
            raise ValueError("Shared compliance used an arbitrary or floored body stiffness inverse")
        volume = _interval(body["domain"]["reference_volume_bounds_m3"])
        reciprocal = Fraction(1)/volume[1], Fraction(1)/volume[0]
        responses = tuple(tuple(_interval_multiply(_interval_sum(
            _scaled_interval(operator[j], inverse[i][j]) for j in range(6)), reciprocal)
            for i in range(6)) for operator in operators)
        _check_interval_matrix(row["unit_pressure_body_response_bounds_m_pa"], responses)
        count = len(operators)
        compliance = []
        for i in range(count):
            values = []
            for j in range(count):
                first, second = min(i, j), max(i, j)
                entry = _interval_multiply(_interval_sum(_scaled_interval(
                    _interval_multiply(operators[first][a], operators[second][b]), inverse[a][b])
                    for a in range(6) for b in range(6)), reciprocal)
                if i == j:
                    entry = max(Fraction(0), entry[0]), entry[1]
                values.append(entry)
            compliance.append(tuple(values))
        _check_interval_matrix(row["pressure_compliance_bounds_m3_pa"], tuple(compliance))


def _exact(value):
    if type(value) is list:
        return _literal_fraction(value)
    lo, hi = _interval(value)
    if lo != hi:
        raise ValueError("An exact constitutive input was replaced by an interval alias")
    return lo


def _matrix(value, rows=3, columns=3):
    if (type(value) is not list or len(value) != rows
            or any(type(row) is not list or len(row) != columns for row in value)):
        raise ValueError("A complete exact tensor or stiffness matrix is required")
    return tuple(tuple(_exact(item) for item in row) for row in value)


def _identity():
    return tuple(tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3))


def _transpose(matrix):
    return tuple(zip(*matrix))


def _multiply(first, second):
    return tuple(tuple(sum((a*b for a, b in zip(row, column)), Fraction(0))
        for column in zip(*second)) for row in first)


def _add(first, second):
    return tuple(tuple(a+b for a, b in zip(row, other)) for row, other in zip(first, second))


def _scale(matrix, coefficient):
    return tuple(tuple(coefficient*x for x in row) for row in matrix)


def _trace(matrix):
    return sum((matrix[i][i] for i in range(3)), Fraction(0))


def _contraction(first, second):
    return sum((a*b for row, other in zip(first, second)
        for a, b in zip(row, other)), Fraction(0))


def _determinant(matrix):
    a, b, c = matrix
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
        -a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))


def _scaled_interval(interval, coefficient):
    lo, hi = interval
    return min(lo*coefficient, hi*coefficient), max(lo*coefficient, hi*coefficient)


def _interval_add(first, second):
    return first[0]+second[0], first[1]+second[1]


def _interval_multiply(first, second):
    products = tuple(a*b for a in first for b in second)
    return min(products), max(products)


def _interval_sum(values):
    total = (Fraction(0), Fraction(0))
    for value in values:
        total = _interval_add(total, value)
    return total


def _interval_square(value):
    lo, hi = value
    return (Fraction(0) if lo <= 0 <= hi else min(lo**2, hi**2), max(lo**2, hi**2))


def _check_interval_matrix(actual, expected):
    if (type(actual) is not list or len(actual) != len(expected)
            or any(type(row) is not list or len(row) != len(other)
                   for row, other in zip(actual, expected))):
        raise ValueError("A complete assembled interval matrix is required")
    for values, bounds in zip(actual, expected):
        for value, expected_interval in zip(values, bounds):
            _check_outward(value, expected_interval)


def _inverse_bulk_stiffness(radius, shear, lame):
    """Closed inverse of the six-mode isotropic stiffness density."""
    normal_identity = radius**2/(2*shear)
    normal_shared = -radius**2*lame/(2*shear*(2*shear+3*lame))
    shear_identity = radius**2/(4*shear)
    return tuple(tuple((normal_shared+(normal_identity if i == j else 0)
        if i < 3 and j < 3 else shear_identity if i == j else Fraction(0))
        for j in range(6)) for i in range(6))


def _export_quantum(bounds):
    maximum = max(abs(bounds[0]), abs(bounds[1]))
    if maximum == 0:
        return Fraction(0)
    exponent = maximum.numerator.bit_length()-maximum.denominator.bit_length()
    below = (maximum.numerator < maximum.denominator << exponent if exponent >= 0
             else maximum.numerator << -exponent < maximum.denominator)
    if below:
        exponent -= 1
    shift = exponent-((3322*96+999)//1000+8)
    return Fraction(1 << shift) if shift >= 0 else Fraction(1, 1 << -shift)


def _check_outward(value, expected):
    """Allow outward export grid rounding while refusing broad invented bounds."""
    actual = _interval(value)
    quantum = _export_quantum(actual)
    if (not actual[0] <= expected[0] <= expected[1] <= actual[1]
            or expected[0]-actual[0] > 2*quantum
            or actual[1]-expected[1] > 2*quantum):
        raise ValueError("An assembled coefficient differs beyond outward export rounding")
    return actual


def _basis():
    positions = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    return tuple(tuple(tuple(Fraction(int((i, j) == position or (j, i) == position))
        for j in range(3)) for i in range(3)) for position in positions)


def _bulk_replay(radius, young, shear, coordinates, spatial_rotation):
    """Independently reconstruct all Green/StVK modal forces and Hessian rows."""
    poisson = young/(2*shear)-1
    lame = 2*shear*poisson/(1-2*poisson)
    if (young <= 0 or shear <= 0 or not -1 < poisson < Fraction(1, 2)
            or radius <= 0 or len(coordinates) != 6):
        raise ValueError("The supported body requires stable source-derived bulk coefficients")
    if (_multiply(_transpose(spatial_rotation), spatial_rotation) != _identity()
            or _determinant(spatial_rotation) != 1):
        raise ValueError("The optional affine body rotation must be exact proper SO(3)")
    basis = _basis()
    generators = tuple(_scale(value, 1/radius) for value in basis)
    body_gradient = _identity()
    for value, generator in zip(coordinates, generators):
        body_gradient = _add(body_gradient, _scale(generator, value))
    gradient = _multiply(spatial_rotation, body_gradient)
    jacobian = _determinant(gradient)
    if jacobian <= 0:
        raise ValueError("Individual affine embedding requires a positive exact determinant")
    green = _scale(_add(_multiply(_transpose(gradient), gradient), _scale(_identity(), -1)),
                   Fraction(1, 2))

    def stress(strain):
        return _add(_scale(strain, 2*shear), _scale(_identity(), lame*_trace(strain)))

    second_piola = stress(green)
    first_piola = _multiply(gradient, second_piola)
    density = shear*_contraction(green, green)+lame*_trace(green)**2/2
    variations = tuple(_multiply(spatial_rotation, value) for value in generators)
    gradient_density = tuple(_contraction(first_piola, value) for value in variations)
    hessian_density = []
    piola_variations = []
    for value in variations:
        strain_variation = _scale(_add(_multiply(_transpose(value), gradient),
            _multiply(_transpose(gradient), value)), Fraction(1, 2))
        piola_variations.append(_add(_multiply(value, second_piola),
            _multiply(gradient, stress(strain_variation))))
    hessian_density = tuple(tuple(_contraction(value, other) for other in piola_variations)
        for value in variations)
    if hessian_density != _transpose(hessian_density):
        raise ValueError("Independent full StVK Hessian lost energetic symmetry")
    stiffness_density = tuple(tuple((2*shear*_contraction(a, b)
        +lame*_trace(a)*_trace(b))/radius**2 for b in basis) for a in basis)
    return dict(generators=basis, body_gradient=body_gradient, gradient=gradient,
        green=green, second_piola=second_piola, first_piola=first_piola,
        energy_density=density, jacobian=jacobian, gradient_density=gradient_density,
        hessian_density=hessian_density, stiffness_density=stiffness_density,
        poisson=poisson, lame=lame)


def _check_native_control(out):
    path = out / "native_control/report.json"
    packet_path = out / "native_control/source_packet.json"
    native = sealed(path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    original = sealed(RELEASE / "native_control/source_packet.json", "packet_sha256")
    false_scope = (
        "original_point_1_completed", "whole_clock_material_admission_authorized",
        "complete_three_dimensional_strain_energy_certified",
        "global_physical_solid_noninterpenetration_certified",
        "all_42_surface_pairs_reexecuted", "new_native_curved_force_equilibrium_certified",
        "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement",
        "material_admission_authorized",
    )
    if (native.get("status") != "PASS"
            or native.get("physical_candidate_status") != PHYSICAL_COMPONENT_SCOPE
            or native.get("actual_source_supported_component_affine_body_assembled") is not True
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("original_point_1_remaining_work_blocks") != 1
            or type(native.get("remaining_physical_conditions")) is not list
            or len(native["remaining_physical_conditions"]) != 3
            or native.get("accepted_input_complete_typed_state_unchanged") is not True
            or any(native.get(name) is not False for name in false_scope)
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("accepted_input_unchanged") is not True
            or packet.get("authenticated_normal_source_packet_raw_sha256") != EXPECTED_PACKET
            or packet.get("authenticated_normal_source_packet_seal") != EXPECTED_PACKET_SEAL
            or packet.get("material_admission_authorized") is not False
            or packet.get("inherited_force_certificate_for_placement") is not False
            or packet.get("full_genuine_source_reconstructed") is not False):
        raise ValueError("Native affine-body assembly exceeded its component-only scope")
    if (native.get("actual_hot_whole_mask_bulk_component_count") != 5
            or native.get("actual_hot_basal_boundary_component_count") != 5
            or native.get("actual_hot_native_pressure_trace_count") != 3
            or native.get("actual_hot_shared_contact_body_count") != 1
            or native.get("actual_cold_clamped_body_component_count") != 0
            or native.get("reference_stressfree_bulk_explicitly_new") is not True
            or native.get("independent_radial_boundary_offset_fixed_zero") is not True
            or any(native.get(key) is not False for key in (
                "source_material_floor_used", "source_area_midpoints_substituted",
                "source_normal_column_lengths_used_as_geometry",
                "old_scalar_foundation_energy_added_again",
                "radial_mean_volume_extension_implemented", "cold_beam_energy_replaced"))):
        raise ValueError("An actual HOT component proof changed source data or its declared scope")
    for name in SOURCE_FIELDS:
        if encoded(packet.get(name)) != encoded(original[name]):
            raise ValueError("A complete authenticated source payload differs: " + name)
    for name in ("coordinates_decimal", "literal_owner_axes"):
        if encoded(packet.get(name)) != encoded(original[name]):
            raise ValueError("The canonical saved curved-force pose differs: " + name)
    reference = packet["saved_joint_reference_field"]
    if reference["declaration_precision"] != 128 or reference["component_count"] != 13:
        raise ValueError("The literal saved source reference field differs")
    _check_component_assembly(native, packet)
    return dict(
        native_control_report_raw_sha256=digest(path),
        native_control_source_packet_raw_sha256=digest(packet_path),
        physical_candidate_status=PHYSICAL_COMPONENT_SCOPE,
        actual_source_supported_component_affine_body_assembled=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        accepted_input_complete_typed_state_unchanged=True,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False,
        new_native_curved_force_equilibrium_certified=False,
        material_admission_authorized=False,
        complete_three_dimensional_strain_energy_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False,
        all_42_surface_pairs_reexecuted=False,
        original_point_1_completed=False,
        original_point_1_remaining_work_blocks=1,
        complete_authenticated_source_packet_preserved=True,
    )


def run(out, test_paths, control):
    paths = tuple(test_paths)
    expected_tests = {name for name in ADDED_SOURCE_PATHS if name.startswith("tests/")}
    if (len(paths) != 2 or set(paths) != expected_tests
            or any(not (ROOT / name).is_file() for name in paths)
            or not isinstance(control, str) or ":" not in control):
        raise ValueError("The two new affine-body test modules and a module:function are required")
    inherited = verify_prior()
    before = source_map()
    added = set(before) - set(inherited)
    if len(before) != 826 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly six additive affine-body Python sources are required")
    removed_caches = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    own_path = str(Path(__file__).resolve().relative_to(ROOT))
    purged_modules = _purge_added_source_modules(added - {own_path})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing final imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(
        format="native-supported-affine-body-same-process-validation-1",
        status="FAIL", regression_validation_status="FAIL",
        physical_candidate_status="NOT_CHECKED",
        started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_solid_placement_outputs=5,
        verified_radial_normalization_diagnosis=True, verified_prior_python_sources=820,
        verified_finite_equilibrium_outputs=4, verified_failed_release_outputs=3,
        verified_spherical_solid_outputs=4, verified_joint_spherical_outputs=4,
        verified_curved_equilibrium_outputs=4, verified_normal_support_outputs=4,
        verified_normal_support_standalone_raw_evidence=dict(STANDALONE_EVIDENCE),
        prior_normal_support_report_raw_sha256=EXPECTED_REPORT,
        prior_normal_support_native_report_raw_sha256=EXPECTED_NATIVE_REPORT,
        prior_normal_support_packet_raw_sha256=EXPECTED_PACKET,
        prior_source_map_sha256=EXPECTED_SOURCE_MAP,
        source_count=len(before), source_map_before_sha256=value_digest(before),
        added_python_sources={name: before[name] for name in sorted(added)},
        removed_added_source_bytecode_caches=removed_caches,
        purged_previously_imported_added_source_modules=purged_modules,
        historical_tests_reexecuted=False,
        test_paths=list(paths), control=control,
        execution_mode="same_python_process", pytest_import_mode="importlib",
        preloaded_genuine_case_used=False,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False,
        whole_native_interval_admitted_by_test_gate=False,
    )
    try:
        import pytest
        command = ["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *paths]
        pytest_started = time.perf_counter()
        original_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    exit_code = int(pytest.main(command))
        finally:
            os.chdir(original_directory)
        summary, counts = parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=exit_code, pytest_summary=summary,
            pytest_counts=counts, pytest_seconds=time.perf_counter()-pytest_started)
        if (exit_code != 0 or counts.get("passed", 0) < 1
                or any(counts.get(name, 0) for name in
                    ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh supported affine-body regressions failed")
        if source_map() != before:
            raise ValueError("A Python source changed during pytest")
        module, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter()-control_started
        report.update(_check_native_control(out))
        expected_outputs = {
            "pytest.log", "control_console.log",
            "native_control/report.json", "native_control/source_packet.json",
        }
        if {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()} != expected_outputs:
            raise ValueError("The affine-body gate requires exactly four declared output files")
        report["status"] = report["regression_validation_status"] = "PASS"
    except BaseException as error:
        report["failure_type"] = type(error).__name__
        report["failure_message"] = str(error)
        report["failure_traceback"] = traceback.format_exc()
    finally:
        after = source_map()
        report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after)
            if before.get(name) != after.get(name))
        try:
            verify_prior()
            report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report["prior_artifacts_and_python_sources_preserved_after"] = False
            report["prior_after_error"] = str(error)
        if (report["changed_python_paths"]
                or not report["prior_artifacts_and_python_sources_preserved_after"]):
            report["status"] = report["regression_validation_status"] = "FAIL"
        report["seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(path.relative_to(out)): dict(
            sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(out.rglob("*")) if path.is_file()}
        report["report_sha256"] = value_digest(report)
        with (out / "report.json").open("xb") as stream:
            stream.write(encoded(report)+b"\n")
        print(encoded({key: report.get(key) for key in
            ("status", "regression_validation_status", "physical_candidate_status",
             "pytest_counts", "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("test_paths", nargs="+")
    options = parser.parse_args()
    run(options.out, options.test_paths, options.control)

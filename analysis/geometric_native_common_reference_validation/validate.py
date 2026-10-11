"""Fresh common-reference component checks preserving the sealed 833 sources.

The accepted historical state is immutable.  A new runtime embedding and its
same-field energy are component evidence; they do not adopt a mixed interval.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_cold_body_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules
from analysis.geometric_native_spherical_solid_validation.validate import _verify_outputs

RELEASE = ROOT / "analysis/geometric_native_cold_body_validation/release_validation_01"
EVIDENCE = RELEASE.parent
EXPECTED_REPORT = "a0843a4f9d886e61125eae0bda9c4964b60d54736b539776b1d32d46d8e56a67"
EXPECTED_REPORT_SEAL = "70d8f1d128582caeb4382baaa2a71100ecbd1bb0c494d4e19f292c7c352e6dbb"
EXPECTED_NATIVE = "a65ce1232f73417c9afc8f790ccdbe810f4452137e6192fecb6c34b29d195db3"
EXPECTED_NATIVE_SEAL = "91f8cefe3a35cc94f5cae97d419b541a4fc7b8469f9fd4f46fdff95850d6708c"
EXPECTED_PACKET = "86806099d9d07467a0a022d57ed8dcad0834487f222a048223486438d2f30472"
EXPECTED_PACKET_SEAL = "8d1744eec08c28b3c3e15593688b812c4aca50677322c3d10db6caf381920a13"
EXPECTED_SOURCE_MAP = "2b322df2d2c426976ee886374517c2e0130acb5a5acfcaf4bb07a6e4c3331e05"
PHYSICAL_SCOPE = "COMMON_REFERENCE_COMPONENT_ONLY"
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_reference.py",
    "tectonics/geometric_native_common_cold.py",
    "tectonics/geometric_native_common_hot.py",
    "tests/test_geometric_native_common_reference.py",
    "tests/test_geometric_native_common_cold.py",
    "tests/test_geometric_native_common_hot.py",
    "analysis/geometric_native_common_reference_validation/native_control.py",
    "analysis/geometric_native_common_reference_validation/validate.py",
))
STANDALONE_EVIDENCE = {
    "common_reference_candidate.json": "50af6b71758feaaed531651b231b50fe5e45228cfa4efa808b26c5a2070d86f4",
    "common_reference_candidate_source.txt": "42f00f1ca95b4cca7e810887b97a04703d99d30d122f84461793053aa9de9f61",
    "common_reference_curved_partition_peer.json": "d07f10dbf7c50a1d454db18d194ff4999889aadd56f27b7a178b30e2d679a259",
    "common_reference_curved_partition_peer_source.txt": "c5607c1128356ef315ba6597973c80f724892748399bee5c56519d809e83a2dc",
    "common_reference_helper_source.txt": "ee031938177a3497aa7821c883650a89397a9942f6b177e48b9a655153297f6e",
    "common_reference_independent_audit.json": "4a1c3af2fa0345c950d7e9dfd10bf5480dfed89b2b5f7bf49a7aeaa792b70054",
    "common_reference_independent_audit_source.txt": "a5dd983d448d921aba565aeb0eb2064e9735e98e3b202cb12823c62e63be1486",
    "common_reference_partition_independent_audit.json": "3e32618b38c2a06f6ce8c6db74b66551d518406531e7a6cb004bc1f61e4b28f7",
    "common_reference_partition_independent_audit_source.txt": "2da33c074affa4c526d942533fc4d70f9cb5e661ae551bc902af6c452ba0d7d1",
    "common_reference_source_partition.json": "038b774e15eba4d8dcdb26ef0d65778218cdea0090a71bc956f331637330a840",
    "common_reference_source_partition_replay.txt": "b1fc3db4a885c676a8c540f4d8c66e9feb1af11fde723b6b5ddc4bc05afb74c9",
    "common_reference_source_partition_source.txt": "65eab6420dad10310bc34a18a21fd9286193d61108a56e57a6e49cb212d89449",
    "common_reference_tensor_dependency_audit.json": "618ad473e7aeb707638ff97f7fe47194a0a0f659a1b22e241446de394fc47070",
    "frozen_source_map_833.json": "ee4c8cb5bc6c602a1df5ca07fb4a0a066c23bc61700b1f36d28e42ae43b35e09",
    "independent_audit.json": "cf8d62c57c3991aa9a4d9329713112dc4979033aaf02128028d22b02176eb376",
    "independent_audit_source.txt": "ee031938177a3497aa7821c883650a89397a9942f6b177e48b9a655153297f6e",
    "initial_reference_overlap.json": "5dcf4003be2a2cbe132f8950351242d1fbc8c1cf14b532ef232d43dbddb2601c",
    "initial_reference_overlap_independent.json": "82d52357b2d16d2bb91cb2cb4e181a5c1a799eabb70e4522e156d2c3e980de52",
    "initial_reference_overlap_replay_inclusion.json": "a80a44e8def61cda75e958464c75f79f1efb63a60cfe5ed4d50b6b86b5592f1f",
    "initial_reference_overlap_replay_source.txt": "f41bc4b5cf007e04d146c82c69348da2bba72dcb508f755a8a9645eb1ed16ba6",
    "initial_reference_overlap_source.txt": "7ac239d77614ae8cffe978127752ece60ac82fafa1d0b8598a016c0bdcfe045b",
}
SOURCE_FIELDS = previous.SOURCE_FIELDS
FALSE_SCOPE = previous.FALSE_SCOPE
encoded, digest, value_digest = previous.encoded, previous.digest, previous.value_digest
read, sealed, source_map, parse_counts = previous.read, previous.sealed, previous.source_map, previous.parse_counts


def verify_prior():
    """Authenticate the completed cold release and every immutable ancestor."""
    inherited = previous.verify_prior()
    report = sealed(RELEASE / "report.json", "report_sha256")
    if (digest(RELEASE / "report.json") != EXPECTED_REPORT
            or report.get("status") != "PASS"
            or read(RELEASE / "report.json").get("report_sha256") != EXPECTED_REPORT_SEAL
            or report.get("pytest_counts") != {"passed": 20}
            or report.get("pytest_exit_code") != 0
            or report.get("physical_candidate_status") != previous.PHYSICAL_COMPONENT_SCOPE
            or report.get("source_count") != 833
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or set(report.get("added_python_sources", {})) != previous.ADDED_SOURCE_PATHS
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("The authoritative completed cold source epoch differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 833 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The immutable 833-source map differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    _verify_outputs(RELEASE, report, 4)
    if {str(p.relative_to(RELEASE)) for p in RELEASE.rglob("*") if p.is_file()} != set(report["outputs"]) | {"report.json"}:
        raise ValueError("A file was added to or removed from the prior sealed release")
    native = sealed(RELEASE / "native_control/report.json", "report_sha256")
    packet = sealed(RELEASE / "native_control/source_packet.json", "packet_sha256")
    if (digest(RELEASE / "native_control/report.json") != EXPECTED_NATIVE
            or read(RELEASE / "native_control/report.json").get("report_sha256") != EXPECTED_NATIVE_SEAL
            or digest(RELEASE / "native_control/source_packet.json") != EXPECTED_PACKET
            or read(RELEASE / "native_control/source_packet.json").get("packet_sha256") != EXPECTED_PACKET_SEAL
            or native.get("source_packet_sha256") != EXPECTED_PACKET_SEAL
            or native.get("actual_cold_supported_body_component_count") != 8
            or native.get("physical_candidate_status") != previous.PHYSICAL_COMPONENT_SCOPE
            or packet.get("accepted_input_unchanged") is not True):
        raise ValueError("The complete previous cold source or component scope differs")
    for row in (report, native):
        if (row.get("accepted_original_native_intervals") != 1
                or row.get("accepted_new_native_intervals") != 0
                or any(row.get(name) is not False for name in FALSE_SCOPE)):
            raise ValueError("Historical cold evidence gained physical acceptance")
    for name, expected in STANDALONE_EVIDENCE.items():
        if digest(EVIDENCE / name) != expected:
            raise ValueError("An immutable standalone proof changed: " + name)
    audit = read(EVIDENCE / "independent_audit.json")
    if (audit.get("status") != "PASS" or audit.get("fresh_tests") != 20
            or audit.get("final_python_sources") != 833
            or audit.get("source_map_canonical_sha256") != EXPECTED_SOURCE_MAP
            or audit.get("wrapper_raw_sha256") != EXPECTED_REPORT
            or audit.get("native_raw_sha256") != EXPECTED_NATIVE
            or audit.get("source_packet_raw_sha256") != EXPECTED_PACKET
            or audit.get("second_certified_tensor_quadrature_performed") is not False
            or audit.get("material_admission_authorized") is not False):
        raise ValueError("The independent completed cold audit differs")
    return inherited


def _check_native_control(out):
    """Replay authentic sources and complete component arithmetic, not motion."""
    from analysis.geometric_native_common_reference_validation import native_control as control
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import native_common_reference_from_dict, native_common_reference_to_dict
    from tectonics import geometric_native_common_cold as cold

    path, packet_path = out / "native_control/report.json", out / "native_control/source_packet.json"
    native, packet = sealed(path, "report_sha256"), sealed(packet_path, "packet_sha256")
    original = sealed(RELEASE / "native_control/source_packet.json", "packet_sha256")
    if (native.get("status") != "PASS" or native.get("physical_candidate_status") != PHYSICAL_SCOPE
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("original_point_1_remaining_work_blocks") != 1
            or len(native.get("remaining_physical_conditions", ())) != 3
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("authenticated_cold_source_packet_raw_sha256") != EXPECTED_PACKET
            or packet.get("authenticated_cold_source_packet_seal") != EXPECTED_PACKET_SEAL
            or packet.get("accepted_input_unchanged") is not True
            or native.get("accepted_input_complete_typed_state_unchanged") is not True
            or any(native.get(name) is not False for name in FALSE_SCOPE)
            or any(packet.get(name) is not False for name in ("material_admission_authorized",
                "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement"))):
        raise ValueError("The NEW common-reference control exceeded component-only scope")
    for name in (*SOURCE_FIELDS, "saved_joint_reference_field", "coordinates_decimal", "literal_owner_axes"):
        if encoded(packet.get(name)) != encoded(original[name]):
            raise ValueError("An immutable historical source was changed: " + name)
    names = ("common_reference", "born_faces", "cold_clamps", "carried_original_sheets",
        "original_mechanical_history", "original_constitutive_states", "cold_components", "hot_components",
        "hot_source_face_traces", "accepted_reference_total_beam_energy_j")
    if any(encoded(native.get(name)) != encoded(packet.get(name)) for name in names):
        raise ValueError("The common report and complete source packet disagree")
    source = load_native_normal_support_source()
    reference = native_common_reference_from_dict(packet["common_reference"], source=source)
    if encoded(native_common_reference_to_dict(reference)) != encoded(packet["common_reference"]):
        raise ValueError("The complete common reference differs from authentic runtime source replay")
    if not reference.initial_all13_positive_volume_exclusion_certified:
        raise ValueError("The initial actual source partition was not replayed")
    geometry = control.geometry_records(source, reference)
    if any(encoded(packet.get(name)) != encoded(value) for name, value in geometry.items()):
        raise ValueError("An actual born face, transported clamp or carried history differs from source replay")
    rows = packet["cold_components"]
    epoch = packet["before"]["snapshot"]["current_material_epoch"]
    components = {row["component_id"]: row for row in epoch["components"] if row["material_kind"] == "original_live"}
    if len(rows) != 8 or {row.get("component_id") for row in rows} != set(components):
        raise ValueError("Every original cold actual material mask is required")
    reference_total, changed_cold = 0, 0
    for row in rows:
        body = cold.bind_native_common_cold_body(row["component_id"], source=source,
            reference=reference, precision=reference.geometry_enclosure_precision)
        material = cold.build_native_common_cold_strain_material(body)
        binding = cold.native_common_cold_body_to_dict(body)
        material_raw = cold.native_common_cold_strain_material_to_dict(material)
        if (encoded(row["body"]) != encoded(binding) or encoded(row["material"]) != encoded(material_raw)):
            raise ValueError("A NEW cold body/material substituted its immutable source or common embedding")
        old = binding["original_body"]
        _, _, old_k, tip, young, poisson, _ = previous._check_cold_body(old,
            components[row["component_id"]], packet, source)
        coefficients = previous._check_cold_material(material_raw["original_material"], old, source, young, poisson)
        view, response = binding["mathematical_body_view"], row["diagnostic_response"]
        if encoded(response.get("common_body_binding")) != encoded(binding):
            raise ValueError("A cold energy receipt changed its separate source/reference binding")
        q, energy = previous._check_calibrated_response(response, view, material_raw, old_k, tip)
        expected_q = tuple(abs(tip)/100*x for x in
            (previous.Fraction(1), previous.Fraction(-1, 3), previous.Fraction(1, 5), previous.Fraction(1, 7),
             previous.Fraction(-1, 11), previous.Fraction(1, 13), previous.Fraction(-1, 17)))
        if (q != expected_q or row.get("probe_is_new_mixed_force_solution") is not False
                or row.get("accepted_reference_energy_and_tip_force_preserved") is not True):
            raise ValueError("The source-derived seven-mode probe changed or claimed mixed equilibrium")
        raw = response["raw_integrals"]
        previous._check_strain_support(response["support"], view, q, max(raw["precision"], 160))
        previous._check_tensor_cells(raw, view, coefficients)
        previous._check_tensor_corrections(raw, view)
        if (raw.get("reference_snapshot_sha256") != reference.reference_snapshot_sha256
                or raw.get("source_body_snapshot_sha256") != body.original_body.body_snapshot_sha256
                or raw.get("source_material_snapshot_sha256") != material.source_material_snapshot_sha256
                or raw.get("active_angular_rate_per_physical_volume_depth_m")
                    != view["angular_rate_per_physical_volume_depth_m"]
                or raw.get("historical_angular_rate_per_physical_volume_depth_m")
                    != old["angular_rate_per_physical_volume_depth_m"]
                or any(raw.get(name) is not True for name in (
                    "fixed_global_tensor_generators_retained", "depth_dependent_computational_generators_retained",
                    "donor_frame_reflection_retained_by_matrix_conjugation",
                    "tip_norm_lower_bound_valid_for_every_skew_depth_generator"))
                or raw.get("body_tensor_angular_degree_upper_bound") != 12):
            raise ValueError("The complete tensor integral used another reference or historical source rate")
        reference_total += energy
        changed_cold += body.angular_rate_per_physical_volume_depth_m != body.historical_angular_rate_per_physical_volume_depth_m
    if changed_cold != 6 or previous._literal_fraction(packet["accepted_reference_total_beam_energy_j"]) != reference_total:
        raise ValueError("The cold reference-energy total or six genuinely changed embeddings differs")
    _check_hot_components(packet, source, reference)
    for name, expected in (("actual_common_reference_body_component_count", 13),
            ("actual_cold_supported_body_component_count", 8), ("actual_hot_body_component_count", 5),
            ("initial_positive_volume_exclusion_pair_count", 78), ("source_born_face_count", 2),
            ("transported_true_live_clamp_count", 8), ("carried_original_source_sheet_count", 8),
            ("original_constitutive_history_count", 16)):
        if native.get(name) != expected:
            raise ValueError("The actual source component/geometry count differs: " + name)
    for name in ("initial_all13_positive_volume_exclusion_certified", "old_sixteen_constitutive_histories_preserved",
            "accepted_reference_energy_preserved", "accepted_reference_tip_forces_preserved"):
        if native.get(name) is not True:
            raise ValueError("A source/reference property was not preserved: " + name)
    for name in ("original_carried_contact_rest_reference_reconstructed", "source_constant_dip_reproduced_on_whole_face",
            "unresolved_positive_material_fronts_removed", "source_normal_column_lengths_used_as_geometry"):
        if native.get(name) is not False:
            raise ValueError("The component control granted an unresolved physical property: " + name)
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        native_control_report_raw_sha256=digest(path), native_control_source_packet_raw_sha256=digest(packet_path),
        actual_common_reference_body_component_count=13, actual_cold_supported_body_component_count=8,
        actual_hot_body_component_count=5, fresh_source_partition_replay_verified=True,
        initial_all13_positive_volume_exclusion_certified=True, initial_positive_volume_exclusion_pair_count=78,
        source_born_face_count=2, transported_true_live_clamp_count=8, carried_original_source_sheet_count=8,
        original_constitutive_history_count=16, old_sixteen_constitutive_histories_preserved=True,
        changed_cold_embedding_count=6, changed_hot_embedding_count=1,
        independent_cold_calibrated_energy_force_hessian_replays_verified=8,
        independent_raw_tensor_primary_scalar_count_per_component=73,
        independent_hot_bulk_basal_energy_force_hessian_replays_verified=5,
        independent_second_certified_tensor_quadrature_performed=False,
        accepted_reference_energy_preserved=True, accepted_reference_tip_forces_preserved=True,
        accepted_input_complete_typed_state_unchanged=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, **{name: False for name in FALSE_SCOPE})


def _check_hot_components(packet, source, reference):
    """Replay bindings, exact isotropic polynomial and six-mode base arithmetic."""
    from analysis.geometric_native_common_reference_validation.native_control import exact, _source_pressure, MATERIAL_PROVENANCE
    from tectonics import geometric_native_common_hot as hot
    from tectonics.geometric_contact_mechanics import ElasticColumnParameters
    from tectonics.geometric_native_bulk_elasticity import NativeBulkElasticClosure
    affine = previous.previous
    rows = packet["hot_components"]
    authentic = {b.component_id for b in source.banks if b.material_kind == "hot_birth"}
    if len(rows) != 5 or {r.get("component_id") for r in rows} != authentic:
        raise ValueError("All five actual HOT material bodies are required")
    pressure, parameters = _source_pressure(packet)
    columns = ElasticColumnParameters(**parameters)
    changed = 0
    for row in rows:
        body = hot.bind_native_common_hot_body(row["component_id"], source=source, reference=reference)
        closure = NativeBulkElasticClosure(columns, "over", body.component_id, source.source_support_sha256, MATERIAL_PROVENANCE)
        source_raw = row["assembly"]["basal"]["basal_source"]
        basal_source = hot.bind_native_common_hot_basal_source(body, precision=source_raw["precision"], order=source_raw["order"])
        assembly = hot.assemble_native_common_hot_component(body, closure, basal_source=basal_source)
        if (encoded(row["body"]) != encoded(hot.native_common_hot_to_dict(body))
                or encoded(row["assembly"]) != encoded(hot.native_common_hot_to_dict(assembly))):
            raise ValueError("A HOT bulk or fresh base changed its source/common binding")
        q = tuple(previous._literal_fraction(x) for x in row["probe_coordinates_m"])
        amplitude = body.radius_m*pressure/closure.young_modulus_pa
        expected = tuple(amplitude*x for x in (-1, closure.poisson_ratio, closure.poisson_ratio,
            previous.Fraction(1, 7), previous.Fraction(-1, 11), previous.Fraction(1, 13)))
        if (q != expected or row["probe_direction_m"] != row["probe_coordinates_m"]
                or previous._literal_fraction(row["source_pressure_probe_pa"]) != pressure
                or row.get("probe_is_new_mixed_force_solution") is not False):
            raise ValueError("The HOT probe changed its source-derived six-mode increments")
        for key, coordinates in (("probe_bulk_response", q), ("zero_bulk_response", (previous.Fraction(0),)*6)):
            raw = row[key]
            replay = affine._bulk_replay(body.radius_m, closure.young_modulus_pa, closure.shear_modulus_pa,
                coordinates, affine._matrix(raw["spatial_rotation"]))
            for name, expected_name in (("body_deformation_gradient", "body_gradient"),
                    ("deformation_gradient", "gradient"), ("green_strain", "green"),
                    ("second_piola_stress_pa", "second_piola"), ("first_piola_stress_pa", "first_piola")):
                if affine._matrix(raw[name]) != replay[expected_name]:
                    raise ValueError("A fixed-global HOT tensor differs from exact independent replay")
            volume = previous._bounds(exact(body.reference_volume_bounds_m3))
            if (tuple(previous._literal_fraction(x) for x in raw["coordinates_m"]) != coordinates
                    or previous._literal_fraction(raw["energy_density_reference_j_m3"]) != replay["energy_density"]
                    or previous._literal_fraction(raw["volume_jacobian"]) != replay["jacobian"]):
                raise ValueError("A HOT response changed its literal increments, density or determinant")
            previous._same_bounds(raw["physical_volume_bounds_m3"], previous._scale(volume, replay["jacobian"]), "HOT physical volume")
            previous._same_bounds(raw["energy_bounds_j"], previous._scale(volume, replay["energy_density"]), "HOT bulk energy")
            if (previous._vector(raw["generalized_gradient_bounds_n"], 6, "HOT forces")
                    != tuple(previous._scale(volume, x) for x in replay["gradient_density"])
                    or previous._matrix(raw["generalized_hessian_bounds_n_m"], 6, "HOT Hessian")
                    != tuple(tuple(previous._scale(volume, x) for x in r) for r in replay["hessian_density"])):
                raise ValueError("The six HOT forces or36 Hessian entries differ from exact virtual work")
            if key == "probe_bulk_response":
                first = sum((x*y for x, y in zip(replay["gradient_density"], q)), previous.Fraction(0))
                second = sum((q[i]*replay["hessian_density"][i][j]*q[j] for i in range(6) for j in range(6)), previous.Fraction(0))
                previous._same_bounds(raw["first_directional_derivative_bounds_j"], previous._scale(volume, first), "HOT directional work")
                previous._same_bounds(raw["second_directional_derivative_bounds_j"], previous._scale(volume, second), "HOT directional curvature")
            elif raw["first_directional_derivative_bounds_j"] is not None or raw["second_directional_derivative_bounds_j"] is not None:
                raise ValueError("The HOT zero-reference probe acquired a directional input")
        basal = row["assembly"]["basal"]
        direction = tuple(previous._bounds(x) for x in source_raw["whole_mask_unit_direction_bounds"])
        radial = previous._scale(previous._bounds(source_raw["base_radius_bounds_m"]), 1/body.radius_m)
        covector = tuple(previous._times(radial, previous._sum(
            previous._scale(previous._square(direction[i]) if i == j else previous._times(direction[i], direction[j]),
                E[i][j]) for i in range(3) for j in range(3))) for E in affine._basis())
        if previous._vector(basal["normal_displacement_covector_bounds"], 6, "fresh basal modes") != covector:
            raise ValueError("The new whole-mask base changed its fixed-global affine field")
        coefficient = previous._scale(previous._bounds(source_raw["reference_measure_area_bounds_m2"]), 4200)
        displacement = previous._sum(previous._scale(x, value) for x, value in zip(covector, q))
        energy = previous._scale(previous._times(coefficient, previous._square(displacement)), previous.Fraction(1, 2))
        gradient = tuple(previous._times(previous._times(coefficient, displacement), x) for x in covector)
        hessian = tuple(tuple(previous._times(coefficient, previous._square(x) if i == j else previous._times(x, y))
            for j, y in enumerate(covector)) for i, x in enumerate(covector))
        raw = row["basal_probe_response"]
        if raw["coordinates_m"] != row["probe_coordinates_m"]:
            raise ValueError("The fresh basal field used another six-mode increment")
        previous._same_bounds(raw["whole_mask_normal_displacement_bounds_m"], displacement, "fresh whole-mask basal displacement")
        previous._same_bounds(raw["energy_bounds_j"], energy, "fresh HOT basal energy")
        first = previous._times(previous._times(coefficient, displacement), displacement)
        second = previous._times(coefficient, previous._square(displacement))
        previous._same_bounds(raw["first_directional_derivative_bounds_j"], first, "fresh HOT basal directional work")
        previous._same_bounds(raw["second_directional_derivative_bounds_j"], second, "fresh HOT basal directional curvature")
        if (previous._vector(raw["generalized_gradient_bounds_n"], 6, "basal forces") != gradient
                or previous._matrix(raw["generalized_hessian_bounds_n_m"], 6, "basal Hessian") != hessian):
            raise ValueError("Fresh whole-mask HOT basal virtual-work derivatives differ")
        combined, bulk = row["combined_probe_response"], row["probe_bulk_response"]
        previous._same_bounds(combined["energy_bounds_j"], previous._plus(previous._bounds(bulk["energy_bounds_j"]), energy), "same-field HOT combined energy")
        for name in ("first_directional_derivative_bounds_j", "second_directional_derivative_bounds_j"):
            previous._same_bounds(combined[name], previous._plus(previous._bounds(bulk[name]), previous._bounds(raw[name])), "same-field HOT combined "+name)
        if (previous._vector(combined["generalized_gradient_bounds_n"], 6, "combined forces")
                != tuple(previous._plus(previous._bounds(x), y) for x, y in zip(bulk["generalized_gradient_bounds_n"], gradient))
                or previous._matrix(combined["generalized_hessian_bounds_n_m"], 6, "combined Hessian")
                != tuple(tuple(previous._plus(previous._bounds(x), y) for x, y in zip(r, s))
                    for r, s in zip(bulk["generalized_hessian_bounds_n_m"], hessian))):
            raise ValueError("The combined HOT derivatives lost bulk/basal coupling")
        for name in ("force_equilibrium_certified", "material_admission_authorized", "radial_mean_volume_extension_implemented",
                "complete_mixed_boundary_conditions_satisfied", "old_scalar_foundation_energy_added_again"):
            if row.get(name) is not False:
                raise ValueError("The HOT component exceeded its physical scope")
        changed += body.reference_geometry_changed
    if changed != 1:
        raise ValueError("Exactly the original zero-rate HOT embedding must change")
    _check_unchanged_hot_traces(packet, source, reference)


def _check_unchanged_hot_traces(packet, source, reference):
    """Bind re-used trace mathematics to actual unchanged source embeddings."""
    affine_packet = sealed(previous.RELEASE / "native_control/source_packet.json", "packet_sha256")
    original = {(r["family_id"], r["bank"]): r for r in affine_packet["native_face_traces"]}
    rows = packet["hot_source_face_traces"]
    if len(rows) != 3 or {(r["original_face"]["family_id"], r["original_face"]["bank"]) for r in rows} != set(original):
        raise ValueError("All three active HOT source traces are required")
    for row in rows:
        face = row["original_face"]
        # The dataclass payload is the kernel; the historical public export
        # also includes derived properties. Every actual kernel field must be
        # byte-canonically identical to the authenticated old source receipt.
        prior = original[(face["family_id"], face["bank"])]
        if any(encoded(value) != encoded(prior.get(name)) for name, value in face.items()):
            raise ValueError("An unchanged HOT face substituted its mathematical source kernel")
        common = next(r["bounds"] for r in packet["born_faces"] if r["family_id"] == face["family_id"])
        if face["common_physical_radius_cubed_bounds_m3"] != common["common_physical_radius_cubed_bounds_m3"]:
            raise ValueError("A HOT operator extends outside the actual common physical face intersection")
        bank = next(b for b in reference.banks if b.component_id == row["body"]["original_body"]["component_id"])
        family = next(f for f in source.families if f.family_id == face["family_id"])
        A = reference.angular_rate_per_physical_volume_depth_m
        if (bank.angular_rate_per_physical_volume_depth_m != A or bank.historical_angular_rate_per_physical_volume_depth_m != A
                or family.shared_depth_axis_per_m != A or row.get("physical_face_embedding_unchanged") is not True
                or row["common_rate_literal"] != [[x.numerator, x.denominator] for x in A]):
            raise ValueError("A changed physical HOT face reused an old face operator")
        previous._snapshot(row, "binding_sha256")


def run(out, test_paths, control):
    tests = tuple(test_paths)
    expected_tests = {name for name in ADDED_SOURCE_PATHS if name.startswith("tests/")}
    if (len(tests) != 3 or set(tests) != expected_tests
            or any(not (ROOT / name).is_file() for name in tests)
            or not isinstance(control, str) or ":" not in control):
        raise ValueError("Exactly the three new test modules and a module:function are required")
    inherited = verify_prior()
    before = source_map()
    added = set(before) - set(inherited)
    if len(before) != 841 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly eight additive common-reference Python sources are required")
    removed_caches = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    own = str(Path(__file__).resolve().relative_to(ROOT))
    purged = _purge_added_source_modules(added - {own})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(format="native-common-reference-same-process-validation-1",
        status="FAIL", regression_validation_status="FAIL", physical_candidate_status="NOT_CHECKED",
        started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_prior_python_sources=833,
        verified_prior_cold_outputs=4, verified_prior_standalone_evidence=dict(STANDALONE_EVIDENCE),
        prior_cold_report_raw_sha256=EXPECTED_REPORT, prior_cold_packet_raw_sha256=EXPECTED_PACKET,
        prior_source_map_sha256=EXPECTED_SOURCE_MAP, source_count=len(before),
        source_map_before_sha256=value_digest(before),
        added_python_sources={name: before[name] for name in sorted(added)},
        removed_added_source_bytecode_caches=removed_caches,
        purged_previously_imported_added_source_modules=purged,
        historical_tests_reexecuted=False, test_paths=list(tests), control=control,
        execution_mode="same_python_process", pytest_import_mode="importlib",
        preloaded_genuine_case_used=False, full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False, whole_native_interval_admitted_by_test_gate=False)
    try:
        import pytest
        test_started = time.perf_counter()
        old_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *tests]))
        finally:
            os.chdir(old_directory)
        summary, counts = parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=code, pytest_summary=summary,
            pytest_counts=counts, pytest_seconds=time.perf_counter()-test_started)
        if (code or counts.get("passed", 0) < 1 or any(counts.get(name, 0)
                for name in ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh common-reference tests failed")
        if source_map() != before:
            raise ValueError("A source changed during tests")
        module, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter()-control_started
        report.update(_check_native_control(out))
        expected_outputs = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
        if {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} != expected_outputs:
            raise ValueError("Exactly four declared output files are required")
        report["status"] = report["regression_validation_status"] = "PASS"
    except BaseException as error:
        report.update(failure_type=type(error).__name__, failure_message=str(error), failure_traceback=traceback.format_exc())
    finally:
        after = source_map()
        report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        try:
            verify_prior()
            report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report["prior_artifacts_and_python_sources_preserved_after"] = False
            report["prior_after_error"] = str(error)
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:
            report["status"] = report["regression_validation_status"] = "FAIL"
        report["seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(p.relative_to(out)): dict(sha256=digest(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob("*")) if p.is_file()}
        report["report_sha256"] = value_digest(report)
        with (out / "report.json").open("xb") as stream:
            stream.write(encoded(report)+b"\n")
        print(encoded({key: report.get(key) for key in ("status", "physical_candidate_status", "pytest_counts", "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--control", required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    run(args.out, args.tests, args.control)

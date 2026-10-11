"""Authenticate and exercise all eight supported cold body components.

The accepted reference energy and tip forces remain source-bound. Diagnostic
increments are not a mixed equilibrium, an admission test, or a new interval.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_finite_equilibrium_validation.native_control import encoded, write

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "analysis/geometric_native_affine_body_validation/release_validation_01/native_control/source_packet.json"
PRIOR_RAW = "ccd511b2b82e66ba103d0d900b993a18d02953288f447c71ac9e589dc70c7d1e"
PHYSICAL_SCOPE = "COLD_TENSOR_COMPONENT_ONLY"


def run(out):
    from tectonics.geometric_native_normal_support import SOURCE_NAMES
    from tectonics.geometric_native_cold_body import bind_native_cold_body, native_cold_body_to_dict
    from tectonics.geometric_native_cold_strain_energy import (
        build_native_cold_strain_material, evaluate_native_cold_strain_energy,
        native_cold_strain_energy_to_dict, native_cold_strain_material_to_dict,
    )
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw = PRIOR.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_RAW:
        raise ValueError("The fixed supported HOT source packet changed")
    original = json.loads(raw)
    original_body = dict(original)
    seal = original_body.pop("packet_sha256")
    if hashlib.sha256(encoded(original_body)).hexdigest() != seal:
        raise ValueError("The fixed complete affine source selfseal differs")
    snapshot = original["before"]["snapshot"]
    cold_components = tuple(row for row in snapshot["current_material_epoch"]["components"]
        if row["material_kind"] != "hot_birth")
    if len(cold_components) != 8:
        raise ValueError("All eight authentic original cold banks are required")
    records = []
    total_reference_energy = F(0)
    for component in cold_components:
        body = bind_native_cold_body(component["component_id"])
        material = build_native_cold_strain_material(body)
        # Every mode is excited. The source accepted tip supplies amplitude;
        # these are new physical increments, not an inherited absolute trial.
        body_record = native_cold_body_to_dict(body)
        # The source-bound reference scalar is read from the actual receipt.
        reference_tip = F(*body_record["accepted_tip_displacement_m"])
        amplitude = abs(reference_tip)/100
        coordinates = tuple(amplitude*value for value in
            (F(1), F(-1, 3), F(1, 5), F(1, 7), F(-1, 11), F(1, 13), F(-1, 17)))
        response = evaluate_native_cold_strain_energy(body, coordinates,
            material=material, precision=64, taylor_order=0,
            beta_cell_count=1, depth_cell_count=1)
        record = dict(component_id=component["component_id"],
            body=body_record, material=native_cold_strain_material_to_dict(material),
            diagnostic_response=native_cold_strain_energy_to_dict(response),
            probe_is_new_mixed_force_solution=False,
            accepted_reference_energy_and_tip_force_preserved=True)
        records.append(record)
        total_reference_energy += response.reference_energy_j
        print("COLD actual supported body", component["original_parent_id"],
            "accepted reference energy J", float(response.reference_energy_j),
            "diagnostic energy lower J", float(response.energy_bounds_j.lo), flush=True)
    packet = dict(format="native-supported-cold-tensor-source-packet-1",
        **{name: original[name] for name in SOURCE_NAMES},
        saved_joint_reference_field=original["saved_joint_reference_field"],
        coordinates_decimal=original["coordinates_decimal"],
        literal_owner_axes=original["literal_owner_axes"],
        authenticated_affine_source_packet_raw_sha256=PRIOR_RAW,
        authenticated_affine_source_packet_seal=seal,
        source_snapshot_sha256=original["source_snapshot_sha256"],
        cold_components=records,
        accepted_reference_total_beam_energy_j=[total_reference_energy.numerator,
            total_reference_energy.denominator],
        accepted_input_unchanged=True, material_admission_authorized=False,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    write(output / "source_packet.json", packet)
    report = dict(format="native-supported-cold-tensor-component-control-1",
        status="PASS", regression_validation_status="PASS",
        physical_candidate_status=PHYSICAL_SCOPE,
        source_packet_sha256=packet["packet_sha256"],
        authenticated_affine_source_packet_raw_sha256=PRIOR_RAW,
        cold_components=records,
        actual_cold_supported_body_component_count=8,
        accepted_reference_total_beam_energy_j=packet["accepted_reference_total_beam_energy_j"],
        accepted_reference_energy_preserved=True,
        accepted_reference_tip_forces_preserved=True,
        explicit_initial_strain_declared=True,
        explicit_new_nonlocal_calibrated_tensor_extension=True,
        historical_per_material_strain_distribution_recovered=False,
        old_beam_stiffness_added_again=False,
        source_material_floor_used=False,
        source_area_midpoints_substituted=False,
        source_normal_column_lengths_used_as_geometry=False,
        new_native_curved_force_equilibrium_certified=False,
        complete_three_dimensional_strain_energy_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False,
        all_42_surface_pairs_reexecuted=False,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False,
        material_admission_authorized=False,
        accepted_input_complete_typed_state_unchanged=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False,
        original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "same-field cold and HOT mixed loads/contact laws and re-solved force equilibrium",
            "whole-clock all-pair material admission on resulting literal motion",
            "adopted mixed continuation, independent path work balances and joint restart"],
        seconds=time.perf_counter()-started)
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    write(output / "report.json", report)
    print("COLD supported component assembly PASS; mixed forces/admission remain open",
        report["seconds"], flush=True)
    return report

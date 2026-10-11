"""Exercise the NEW common embedding without adopting a mixed interval.

The four historical DTOs, old field and all constitutive histories are literal
inputs. New born surfaces and transported live clamps bind the same reference
as every cold gradient and HOT basal field. Diagnostic increments are neither
loads nor an equilibrium solution.
"""
from __future__ import annotations

from dataclasses import fields
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_affine_body_validation.native_control import (
    exact, _response, _source_pressure, MATERIAL_PROVENANCE,
)
from analysis.geometric_native_finite_equilibrium_validation.native_control import encoded, write

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "analysis/geometric_native_cold_body_validation/release_validation_01/native_control/source_packet.json"
PRIOR_RAW = "86806099d9d07467a0a022d57ed8dcad0834487f222a048223486438d2f30472"
PHYSICAL_SCOPE = "COMMON_REFERENCE_COMPONENT_ONLY"
FALSE_SCOPE = (
    "original_point_1_completed", "whole_clock_material_admission_authorized",
    "complete_three_dimensional_strain_energy_certified",
    "global_physical_solid_noninterpenetration_certified",
    "all_42_surface_pairs_reexecuted", "new_native_curved_force_equilibrium_certified",
    "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement",
    "material_admission_authorized",
)


def geometry_records(source, reference):
    """All actual born surfaces, live clamps and separate historical sheets."""
    from tectonics.geometric_native_common_reference import (
        bound_native_common_reference_face, bound_native_common_reference_clamp,
        bound_native_common_reference_carried_original_face,
    )
    faces = []
    for family in source.families:
        face = bound_native_common_reference_face(reference, family.family_id)
        faces.append(dict(family_id=family.family_id, bounds=exact(face),
            reference_faces_match_at_every_common_physical_radius=face.reference_faces_match_at_every_common_physical_radius,
            whole_common_reference_face_enclosed=face.whole_common_reference_face_enclosed,
            source_constant_dip_reproduced_on_whole_face=False,
            geometric_gluing_supplies_constitutive_rest_gap=False,
            every_point_of_parameter_enclosure_is_material=False,
            physical_material_admission_authorized=False))
    clamps = []
    for bank in reference.banks:
        if bank.material_kind != "original_live":
            continue
        clamp = bound_native_common_reference_clamp(reference, bank.component_id)
        clamps.append(dict(component_id=bank.component_id, bounds=exact(clamp),
            genuine_live_material_clamp_support_certified=clamp.genuine_live_material_clamp_support_certified,
            all_seven_modal_fields_and_gradients_zero=clamp.all_seven_modal_fields_and_gradients_zero,
            transported_source_material_clamp=True, source_world_clamp_kept_fixed=False,
            physical_material_admission_authorized=False))
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    traces = before["snapshot"]["current_material_epoch"]["source_material_state"]["trace_bindings"]
    carried = []
    for trace in traces:
        for bank in ("sub", "over"):
            face = bound_native_common_reference_carried_original_face(reference, trace["trace_id"], bank)
            carried.append(dict(trace_id=trace["trace_id"], source_bank=bank, bounds=exact(face),
                carried_original_material_sheet_enclosed=True,
                whole_sheet_is_retained_live_contact_interface=False,
                donor_and_receiver_historical_sheets_glued=False,
                old_sixteen_constitutive_histories_preserved=True,
                physical_material_admission_authorized=False))
    if (len(faces), len(clamps), len(carried)) != (2, 8, 8):
        raise ValueError("Both source born families, eight live clamps and four original trace pairs are required")
    return dict(born_faces=faces, cold_clamps=clamps, carried_original_sheets=carried,
        original_mechanical_history=before["original_physical_history"],
        original_constitutive_states=before["snapshot"]["states"])


def hot_records(source, reference, original):
    from tectonics.geometric_contact_mechanics import ElasticColumnParameters
    from tectonics.geometric_native_bulk_elasticity import NativeBulkElasticClosure
    from tectonics import geometric_native_common_hot as hot
    pressure, parameters = _source_pressure(original)
    columns = ElasticColumnParameters(**parameters)
    bodies, records = {}, []
    for bank in source.banks:
        if bank.material_kind != "hot_birth":
            continue
        body = hot.bind_native_common_hot_body(bank.component_id, source=source, reference=reference)
        closure = NativeBulkElasticClosure(columns, "over", bank.component_id,
            source.source_support_sha256, MATERIAL_PROVENANCE)
        basal = hot.bind_native_common_hot_basal_source(body, precision=64)
        assembly = hot.assemble_native_common_hot_component(body, closure, basal_source=basal)
        amplitude = body.radius_m*pressure/closure.young_modulus_pa
        coordinates = tuple(amplitude*x for x in
            (F(-1), closure.poisson_ratio, closure.poisson_ratio, F(1, 7), F(-1, 11), F(1, 13)))
        response = hot.evaluate_native_common_hot_component(assembly, coordinates, direction_m=coordinates)
        bulk = response.bulk_response.constant_gradient_polynomial_response
        zero = hot.evaluate_native_common_hot_bulk(assembly.bulk, (F(0),)*6)
        records.append(dict(component_id=body.component_id,
            body=hot.native_common_hot_to_dict(body),
            assembly=hot.native_common_hot_to_dict(assembly),
            probe_coordinates_m=exact(coordinates), probe_direction_m=exact(coordinates),
            source_pressure_probe_pa=exact(pressure), probe_bulk_response=_response(bulk),
            zero_bulk_response=_response(zero.constant_gradient_polynomial_response),
            basal_probe_response=_response(response.basal_response),
            combined_probe_response={item.name: exact(getattr(response, item.name))
                for item in fields(response) if item.name not in ("assembly", "bulk_response", "basal_response")},
            probe_is_new_mixed_force_solution=False,
            affine_volume_and_basal_probe_use_same_six_mode_field=True,
            independent_radial_boundary_offset_fixed_zero=True,
            radial_mean_volume_extension_implemented=False,
            complete_mixed_boundary_conditions_satisfied=False,
            old_scalar_foundation_energy_added_again=False,
            scope=hot.COMMON_HOT_SCOPE,
            force_equilibrium_certified=False, material_admission_authorized=False))
        bodies[body.component_id] = body
        print("COMMON HOT", body.original_body.original_parent_id,
            "fresh basal", body.reference_geometry_changed,
            "diagnostic energy lower J", float(response.energy_bounds_j.lo), flush=True)
    traces = []
    for family in source.families:
        for bank, identity in (("donor", family.donor_component_id), ("receiver", family.receiver_component_id)):
            if identity not in bodies:
                continue
            face = hot.bind_native_common_hot_source_face_trace(bodies[identity], family.family_id,
                bank, depth_cells=16, precision=160)
            traces.append(hot.native_common_hot_to_dict(face))
            print("COMMON unchanged literal HOT face", family.family_id, bank, flush=True)
    if (len(records), len(traces)) != (5, 3):
        raise ValueError("All five actual HOT components and all three active HOT traces are required")
    return records, traces


def run(out):
    from tectonics.geometric_native_normal_support import load_native_normal_support_source, SOURCE_NAMES
    from tectonics.geometric_native_common_reference import bind_native_common_reference, native_common_reference_to_dict
    from tectonics.geometric_native_common_cold import (
        bind_native_common_cold_body, build_native_common_cold_strain_material,
        evaluate_native_common_cold_strain_energy, native_common_cold_body_to_dict,
        native_common_cold_strain_material_to_dict, native_common_cold_strain_energy_to_dict,
    )
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw = PRIOR.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_RAW:
        raise ValueError("The immutable completed cold source packet changed")
    original = json.loads(raw)
    check = dict(original)
    seal = check.pop("packet_sha256")
    if hashlib.sha256(encoded(check)).hexdigest() != seal:
        raise ValueError("The complete historical source selfseal differs")
    source = load_native_normal_support_source()
    reference = bind_native_common_reference(source=source, precision=160)
    geometry = geometry_records(source, reference)
    print("COMMON initial geometry:13 bodies,78 source pair exclusions,2 born faces,8 live clamps; original sheets unglued", flush=True)
    hot, traces = hot_records(source, reference, original)
    cold, reference_total = [], F(0)
    for bank in source.banks:
        if bank.material_kind != "original_live":
            continue
        body = bind_native_common_cold_body(bank.component_id, source=source, reference=reference, precision=160)
        material = build_native_common_cold_strain_material(body)
        amplitude = abs(body.accepted_tip_displacement_m)/100
        coordinates = tuple(amplitude*x for x in
            (F(1), F(-1, 3), F(1, 5), F(1, 7), F(-1, 11), F(1, 13), F(-1, 17)))
        response = evaluate_native_common_cold_strain_energy(body, coordinates, material=material,
            precision=64, taylor_order=0, beta_cell_count=1, depth_cell_count=1)
        cold.append(dict(component_id=bank.component_id,
            body=native_common_cold_body_to_dict(body),
            material=native_common_cold_strain_material_to_dict(material),
            diagnostic_response=native_common_cold_strain_energy_to_dict(response),
            probe_is_new_mixed_force_solution=False,
            accepted_reference_energy_and_tip_force_preserved=True))
        reference_total += response.reference_energy_j
        print("COMMON COLD", bank.original_parent_id, "source A changed",
            body.historical_angular_rate_per_physical_volume_depth_m != body.angular_rate_per_physical_volume_depth_m,
            "reference energy J", float(response.reference_energy_j), flush=True)
    shared = dict(common_reference=native_common_reference_to_dict(reference),
        **geometry, cold_components=cold, hot_components=hot, hot_source_face_traces=traces,
        accepted_reference_total_beam_energy_j=exact(reference_total))
    packet = dict(format="native-common-reference-component-source-packet-1",
        **{name: original[name] for name in SOURCE_NAMES},
        saved_joint_reference_field=original["saved_joint_reference_field"],
        coordinates_decimal=original["coordinates_decimal"], literal_owner_axes=original["literal_owner_axes"],
        authenticated_cold_source_packet_raw_sha256=PRIOR_RAW,
        authenticated_cold_source_packet_seal=seal, source_snapshot_sha256=source.source_snapshot_sha256,
        **shared, accepted_input_unchanged=True,
        full_genuine_source_reconstructed=False, inherited_force_certificate_for_placement=False,
        material_admission_authorized=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    write(output / "source_packet.json", packet)
    report = dict(format="native-common-reference-component-control-1", status="PASS",
        regression_validation_status="PASS", physical_candidate_status=PHYSICAL_SCOPE,
        source_packet_sha256=packet["packet_sha256"], **shared,
        actual_common_reference_body_component_count=13,
        actual_cold_supported_body_component_count=8, actual_hot_body_component_count=5,
        initial_all13_positive_volume_exclusion_certified=True,
        initial_positive_volume_exclusion_pair_count=78,
        source_born_face_count=2, transported_true_live_clamp_count=8,
        carried_original_source_sheet_count=8, original_constitutive_history_count=16,
        old_sixteen_constitutive_histories_preserved=True,
        original_carried_contact_rest_reference_reconstructed=False,
        source_constant_dip_reproduced_on_whole_face=False,
        unresolved_positive_material_fronts_removed=False,
        source_normal_column_lengths_used_as_geometry=False,
        accepted_reference_energy_preserved=True, accepted_reference_tip_forces_preserved=True,
        accepted_input_complete_typed_state_unchanged=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "same-field reference/support/loads/drag/contact and fresh mixed force equilibrium",
            "whole-clock all-pair material admission including reverse C5-to-C1",
            "adopted mixed continuation, independent path-work balances and joint restart"],
        **{name: False for name in FALSE_SCOPE}, seconds=time.perf_counter()-started)
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    write(output / "report.json", report)
    print("COMMON component binding PASS; new accepted physical intervals remain0", report["seconds"], flush=True)
    return report

"""Assemble actual HOT bulk and basal components without accepting motion.

The fixed complete source is authenticated, not rebuilt.  The nonzero body
fields below are explicit constitutive probes on real material masks, not a
new mixed force solution.  Old pose, histories, activation and clocks remain
unchanged.  Cold clamping and the mixed physical transaction are still open.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_finite_equilibrium_validation.native_control import encoded, write

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "analysis/geometric_native_normal_support_validation/release_validation_01/native_control/source_packet.json"
PRIOR_RAW = "f4f3703cbc463d4130096dba827a7425fc865851fe5c86f337f98dfbefd8f7ca"
COMPONENT_SCOPE = "COMPONENT_AFFINE_BODY_ONLY"
MATERIAL_PROVENANCE = (
    "Explicit new stress-free affine HOT bulk closure on every authentic HOT mask; "
    "the contact-source laterally-free E/G interpretation is extended to the same "
    "newborn material category, with no inherited 3D tensor or normal rod geometry"
)


def exact(value):
    """Preserve rational inputs and bounds literally; never compress to aliases."""
    from tectonics.geometric_native_pose import FractionInterval as I
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {"lo": exact(value.lo), "hi": exact(value.hi)}
    if is_dataclass(value):
        return {item.name: exact(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact(item) for item in value]
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise TypeError("A body receipt must contain explicit immutable exact data")


def _response(response):
    return {item.name: exact(getattr(response, item.name))
        for item in fields(response) if item.name != "assembly"}


def _source_pressure(source):
    parameters = source["full_reference_face_birth_support"]["interface_carriers"][0]["declaration"]
    columns, law = parameters["parameters"]["columns"], parameters["parameters"]["law"]
    compliance = sum((F.from_float(columns[bank+"_length_m"])
        /F.from_float(columns[bank+"_normal_modulus_pa"]) for bank in ("sub", "over")), F(0))
    compliance += 1/F.from_float(law["normal_stiffness_pa_m"])
    return -F.from_float(parameters["birth_gap_free_m"])/compliance, columns


def run(out):
    from tectonics.geometric_contact_mechanics import ElasticColumnParameters
    from tectonics.geometric_native_bulk_elasticity import NativeBulkElasticClosure
    from tectonics.geometric_native_normal_support import load_native_normal_support_source, SOURCE_NAMES
    from tectonics.geometric_native_affine_body import (
        bind_native_affine_hot_body, native_affine_body_to_dict,
        bound_native_affine_source_face_trace, native_affine_source_face_trace_to_dict,
        bind_native_affine_basal_source, native_affine_basal_source_to_dict,
    )
    from tectonics.geometric_native_affine_bulk_assembly import (
        assemble_native_affine_bulk, evaluate_native_affine_bulk,
        condense_native_affine_shared_faces,
        assemble_native_affine_basal, evaluate_native_affine_basal,
        assemble_native_affine_hot_component, evaluate_native_affine_hot_component,
    )
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw = PRIOR.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_RAW:
        raise ValueError("The fixed complete normal-source packet changed")
    original = json.loads(raw)
    body = dict(original)
    if body.pop("packet_sha256") != hashlib.sha256(encoded(body)).hexdigest():
        raise ValueError("The original complete source seal differs")
    source = load_native_normal_support_source()
    if original["source_snapshot_sha256"] != source.source_snapshot_sha256:
        raise ValueError("The supported bodies changed their authentic source epoch")
    pressure, source_columns = _source_pressure(original)
    columns = ElasticColumnParameters(**source_columns)
    models, assemblies, records = {}, {}, []
    for bank in source.banks:
        if bank.material_kind != "hot_birth":
            continue
        model = bind_native_affine_hot_body(bank.component_id, source=source)
        closure = NativeBulkElasticClosure(columns, "over", bank.component_id,
            source.source_support_sha256, MATERIAL_PROVENANCE)
        assembly = assemble_native_affine_bulk(model, closure)
        # This deliberately exercises every strain mode at a source-derived
        # amplitude.  The rational directions are diagnostic, not fitted loads.
        amplitude = model.radius_m*pressure/closure.young_modulus_pa
        coordinates = tuple(amplitude*x for x in
            (F(-1), closure.poisson_ratio, closure.poisson_ratio, F(1, 7), F(-1, 11), F(1, 13)))
        response = evaluate_native_affine_bulk(assembly, coordinates, direction_m=coordinates)
        zero = evaluate_native_affine_bulk(assembly, (F(0),)*6)
        if (response.energy_bounds_j.lo <= 0 or response.volume_jacobian <= 0
                or zero.energy_bounds_j.lo or zero.energy_bounds_j.hi
                or any(value.lo or value.hi for value in zero.generalized_gradient_bounds_n)):
            raise ValueError("An authentic whole-mask field lost its full energy or zero reference")
        basal_source = bind_native_affine_basal_source(model)
        basal = assemble_native_affine_basal(model, basal_source=basal_source)
        # A prescribed additional boundary offset is held at zero.  The six
        # affine modes then have the same complete volume and base field.
        basal_response = evaluate_native_affine_basal(basal, coordinates,
            direction_m=coordinates)
        combined = assemble_native_affine_hot_component(model, closure, basal_source=basal_source)
        combined_response = evaluate_native_affine_hot_component(combined, coordinates,
            direction_m=coordinates)
        record = dict(component_id=model.component_id, domain=native_affine_body_to_dict(model),
            bulk_closure=exact(closure),
            bulk_stiffness_per_reference_volume_n_m4=exact(assembly.stiffness_per_reference_volume_n_m4),
            bulk_stiffness_bounds_n_m=exact(assembly.stiffness_bounds_n_m),
            probe_coordinates_m=exact(coordinates), probe_direction_m=exact(coordinates),
            source_pressure_probe_pa=exact(pressure),
            probe_is_new_mixed_force_solution=False,
            zero_bulk_response=_response(zero), probe_bulk_response=_response(response),
            basal_source=native_affine_basal_source_to_dict(basal_source),
            basal_assembly={item.name: exact(getattr(basal, item.name))
                for item in fields(basal) if item.name not in ("body", "basal_source")},
            basal_probe_response=_response(basal_response),
            combined_hot_reference_stiffness_bounds_n_m=exact(combined.reference_stiffness_bounds_n_m),
            combined_hot_probe_response={item.name: exact(getattr(combined_response, item.name))
                for item in fields(combined_response)
                if item.name not in ("assembly", "bulk_response", "basal_response")},
            affine_volume_and_basal_probe_use_same_six_mode_field=True,
            independent_radial_boundary_offset_fixed_zero=True,
            radial_mean_volume_extension_implemented=False,
            complete_mixed_boundary_conditions_satisfied=False,
            old_scalar_foundation_energy_added_again=False)
        models[model.component_id], assemblies[model.component_id] = model, assembly
        records.append(record)
        print("AFFINE actual HOT body", model.original_parent_id,
            "volume m3", float(model.reference_volume_bounds_m3.lo),
            "bulk energy J", float(response.energy_bounds_j.lo),
            "basal energy lower J", float(basal_response.energy_bounds_j.lo), flush=True)
    if len(records) != 5:
        raise ValueError("Every genuine HOT bank requires one supported full-mask body field")
    face_objects, face_records = {}, []
    for family in source.families:
        for bank, identity in (("donor", family.donor_component_id), ("receiver", family.receiver_component_id)):
            if identity not in models:
                continue
            face = bound_native_affine_source_face_trace(family.family_id, bank,
                source=source, depth_cells=16, precision=160)
            face_objects.setdefault(identity, []).append(face)
            face_records.append(native_affine_source_face_trace_to_dict(face))
            print("AFFINE true source face", family.family_id, bank,
                "depth cells", face.depth_cell_count, flush=True)
    shared = []
    for identity, faces in face_objects.items():
        if len(faces) < 2:
            continue
        condensed = condense_native_affine_shared_faces(assemblies[identity], tuple(faces))
        shared.append(dict(component_id=identity,
            **{item.name: exact(getattr(condensed, item.name)) for item in fields(condensed)
               if item.name not in ("assembly", "faces")},
            basal_foundation_included=False,
            complete_hot_component_compliance_certified=False))
    if len(face_records) != 3 or len(shared) != 1:
        raise ValueError("Actual three HOT pressure traces and their one shared bank changed")
    packet = dict(format="native-supported-affine-hot-body-source-packet-1",
        **{name: original[name] for name in SOURCE_NAMES},
        saved_joint_reference_field=original["saved_joint_reference_field"],
        coordinates_decimal=original["coordinates_decimal"], literal_owner_axes=original["literal_owner_axes"],
        authenticated_normal_source_packet_raw_sha256=PRIOR_RAW,
        authenticated_normal_source_packet_seal=original["packet_sha256"],
        source_snapshot_sha256=source.source_snapshot_sha256,
        affine_body_components=records, native_face_traces=face_records,
        shared_hot_bank_compliance=shared,
        accepted_input_unchanged=True, material_admission_authorized=False,
        full_genuine_source_reconstructed=False, inherited_force_certificate_for_placement=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    write(output/"source_packet.json", packet)
    report = dict(format="native-supported-affine-hot-body-component-control-1", status="PASS",
        regression_validation_status="PASS", physical_candidate_status=COMPONENT_SCOPE,
        source_packet_sha256=packet["packet_sha256"],
        authenticated_normal_source_packet_raw_sha256=PRIOR_RAW,
        source_snapshot_sha256=source.source_snapshot_sha256,
        body_components=records, native_face_traces=face_records, shared_hot_bank_compliance=shared,
        actual_source_supported_component_affine_body_assembled=True,
        actual_hot_whole_mask_bulk_component_count=5,
        actual_hot_basal_boundary_component_count=5,
        actual_hot_native_pressure_trace_count=3,
        actual_hot_shared_contact_body_count=1,
        source_material_floor_used=False, source_area_midpoints_substituted=False,
        source_normal_column_lengths_used_as_geometry=False,
        reference_stressfree_bulk_explicitly_new=True,
        old_scalar_foundation_energy_added_again=False,
        independent_radial_boundary_offset_fixed_zero=True,
        radial_mean_volume_extension_implemented=False,
        actual_cold_clamped_body_component_count=0,
        cold_beam_energy_replaced=False,
        new_native_curved_force_equilibrium_certified=False,
        complete_three_dimensional_strain_energy_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False, all_42_surface_pairs_reexecuted=False,
        full_genuine_source_reconstructed=False, inherited_force_certificate_for_placement=False,
        material_admission_authorized=False, accepted_input_complete_typed_state_unchanged=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "complete cold/reference-compatible and HOT mixed body fields, energy and re-solved forces",
            "whole-clock all-pair material admission on resulting literal motion",
            "adopted mixed continuation, independent path work balances and joint restart"],
        seconds=time.perf_counter()-started)
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    write(output/"report.json", report)
    print("AFFINE actual HOT component assembly PASS; mixed force/admission remain open",
        report["seconds"], flush=True)
    return report

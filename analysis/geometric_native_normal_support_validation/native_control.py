"""Reject unplaceable source rods and declare finite-body bulk constitutive data.

The successful regression outcome is a physical rejection.  No whole-mask
body field, new force solve, material admission, or accepted interval is
created by this controller.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_finite_equilibrium_validation.native_control import (
    encoded, outward, payload, write,
)

PHYSICAL_REJECTION = "REJECTED_INVALID_NORMAL_SUPPORT"
NEW_CLOSURE_PROVENANCE = (
    "New finite-body interpretation: supplied effective normal modulus is "
    "laterally free Young modulus, supplied shear modulus defines standard "
    "isotropy; historical normal rod lengths are not material geometry"
)


def _diagonal(values):
    return tuple(tuple(value if i == j else F(0) for j in range(3))
                 for i, value in enumerate(values))


def _bulk_declarations(source, original):
    from tectonics.geometric_contact_mechanics import ElasticColumnParameters
    from tectonics.geometric_native_bulk_elasticity import (
        NativeBulkElasticClosure, evaluate_native_bulk_elasticity,
    )
    closures, energy_checks = {}, []
    carriers = original["full_reference_face_birth_support"]["interface_carriers"]
    for carrier in carriers:
        declaration = carrier["declaration"]
        parameters = declaration["parameters"]
        columns = ElasticColumnParameters(**parameters["columns"])
        compliance = (F.from_float(columns.sub_length_m)/F.from_float(columns.sub_normal_modulus_pa)
            + F.from_float(columns.over_length_m)/F.from_float(columns.over_normal_modulus_pa)
            + 1/F.from_float(parameters["law"]["normal_stiffness_pa_m"]))
        pressure = -min(F.from_float(declaration["birth_gap_free_m"]), F(0))/compliance
        for side, source_bank in (("donor", "sub"), ("receiver", "over")):
            identity = carrier[side]["component_id"]
            closure = NativeBulkElasticClosure(columns, source_bank, identity,
                source.source_support_sha256, NEW_CLOSURE_PROVENANCE)
            row = dict(component_id=identity, source_support_sha256=closure.source_support_sha256,
                source_columns=payload(columns), lateral_closure=closure.lateral_closure,
                lateral_closure_provenance=closure.lateral_closure_provenance,
                young_modulus_pa=outward(closure.young_modulus_pa),
                shear_modulus_pa=outward(closure.shear_modulus_pa),
                poisson_ratio=outward(closure.poisson_ratio),
                lame_lambda_pa=outward(closure.lame_lambda_pa),
                new_lateral_closure_declared=True,
                previous_three_dimensional_tensor_inherited=False,
                source_column_lengths_are_actual_bank_geometry=False,
                actual_whole_mask_field_solved=False,
                complete_three_dimensional_strain_energy_certified=False)
            if identity in closures and closures[identity] != row:
                raise ValueError("A shared authentic bank received conflicting material declarations")
            closures[identity] = row
            # Source preload supplies this check's strain amplitude; no new
            # fitted strain, rod support, material reservoir or body field.
            strain = pressure/closure.young_modulus_pa
            free = evaluate_native_bulk_elasticity(closure,
                _diagonal((-strain, closure.poisson_ratio*strain, closure.poisson_ratio*strain)))
            old_density = closure.young_modulus_pa*strain*strain/2
            robin = evaluate_native_bulk_elasticity(closure,
                _diagonal((-strain, strain, F(0))))
            if (free.energy_density_j_m3 != old_density
                    or free.stress_pa[1][1] or free.stress_pa[2][2]
                    or robin.energy_density_j_m3/old_density != F(8, 5)):
                raise ValueError("The new tensor lost the free-column limit or the transverse energy")
            energy_checks.append(dict(family_id=declaration["family_id"], bank=side,
                component_id=identity, preload_pressure_pa=outward(pressure),
                source_derived_strain=outward(strain),
                old_one_dimensional_density_j_m3=outward(old_density),
                laterally_free_bulk_density_j_m3=outward(free.energy_density_j_m3),
                isochoric_robin_linear_density_j_m3=outward(robin.energy_density_j_m3),
                isochoric_to_one_dimensional_energy_ratio=outward(F(8, 5)),
                local_constitutive_check_only=True, actual_whole_mask_field_solved=False))
    return tuple(closures[key] for key in sorted(closures)), tuple(energy_checks)


def run(out):
    from tectonics.geometric_native_normal_support import (
        CURVED_PACKET, CURVED_PACKET_RAW_SHA256, SOURCE_NAMES,
        load_native_normal_support_source, preflight_native_normal_support,
        native_normal_support_to_dict,
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    original_raw = CURVED_PACKET.read_bytes()
    if hashlib.sha256(original_raw).hexdigest() != CURVED_PACKET_RAW_SHA256:
        raise ValueError("The complete independently authenticated curved source changed")
    original = json.loads(original_raw)
    source = load_native_normal_support_source(curved_packet_bytes=original_raw)
    certificates = preflight_native_normal_support(source)
    diagnostic = native_normal_support_to_dict(source, certificates)
    if len(certificates) != 2:
        raise ValueError("Both genuine new contacts require a source support check")
    for row in certificates:
        if (not row.full_source_column_support_rejected
                or not row.positive_measure_activation_strip_proved
                or row.outside_roof_margin_bounds_m.lo <= F(356)
                or row.column_lengths_m != (F(1000), F(1000))
                or row.source_preload_gap_m != F.from_float(-.01)
                or row.activation_time_fraction != 0):
            raise ValueError("A complete source rod lacks its positive-area roof exclusion")
        print("NORMAL source rod rejected", row.family_id,
            "whole angle", float(row.native_angle_end_bounds_radians.lo),
            "depth", float(row.interior_depth_m.lo), float(row.interior_depth_m.hi),
            "minimum excess above all same-owner roofs m",
            float(row.outside_roof_margin_bounds_m.lo), flush=True)
    closures, energy_checks = _bulk_declarations(source, original)
    source_packet = dict(format="native-normal-support-authenticated-source-packet-1",
        **{name: original[name] for name in SOURCE_NAMES},
        saved_joint_reference_field=original["saved_joint_reference_field"],
        coordinates_decimal=original["actual_curved_force_trial"]["coordinates_decimal"],
        literal_owner_axes=original["actual_curved_force_trial"]["literal_owner_axes"],
        authenticated_curved_packet_raw_sha256=CURVED_PACKET_RAW_SHA256,
        authenticated_curved_packet_seal=original["packet_sha256"],
        source_snapshot_sha256=source.source_snapshot_sha256,
        bulk_closures=list(closures), accepted_input_unchanged=True,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False,
        material_admission_authorized=False)
    source_packet["packet_sha256"] = hashlib.sha256(encoded(source_packet)).hexdigest()
    write(out/"source_packet.json", source_packet)
    report = dict(format="native-normal-support-physical-rejection-control-1",
        status=PHYSICAL_REJECTION, physical_candidate_status=PHYSICAL_REJECTION,
        regression_validation_status="PASS",
        source_packet_sha256=source_packet["packet_sha256"],
        authenticated_curved_packet_raw_sha256=CURVED_PACKET_RAW_SHA256,
        source_snapshot_sha256=source.source_snapshot_sha256,
        source_epoch_sha256=source.source_epoch_sha256,
        source_support_sha256=source.source_support_sha256,
        normal_support_witnesses=diagnostic["certificates"],
        fixed_roof_reference_stage="complete activation face at accepted before pose",
        normal_columns_clipped=False, source_preload_preserved=True,
        all_same_owner_bank_roofs_checked=True,
        alternative_finite_body_deformation_models_ruled_out=False,
        bulk_closures=list(closures), local_bulk_energy_checks=list(energy_checks),
        bulk_tensor_newly_declared=True, actual_body_stiffness_assembled=False,
        actual_whole_mask_field_solved=False,
        shared_bank_cross_contact_operator_implemented=True,
        complete_three_dimensional_strain_energy_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False,
        all_42_surface_pairs_reexecuted=False,
        previous_surface_pair_classification_inherited=False,
        new_native_curved_force_equilibrium_certified=False,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False,
        material_admission_authorized=False,
        accepted_input_complete_typed_state_unchanged=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "one supported displacement field per true bank, full bulk/interface energy and re-solved own forces",
            "whole-clock all-pair material admission on resulting literal motion",
            "adopted mixed continuation, path work balances and joint restart"],
        seconds=time.perf_counter()-started)
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    write(out/"report.json", report)
    print("NORMAL regression check PASS; physical placement", PHYSICAL_REJECTION,
        report["seconds"], flush=True)
    return report

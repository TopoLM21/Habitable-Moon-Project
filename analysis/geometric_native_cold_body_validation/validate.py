"""Fresh cold-body checks preserving the complete sealed 826-source epoch.

The native controller owns the limited cold-component claims.  This gate
replays the source masks and tensor polynomial evidence without admitting a
new physical motion, rebuilding the genuine case, or reusing old force scope.
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
from analysis.geometric_native_affine_body_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules
from analysis.geometric_native_spherical_solid_validation.validate import _verify_outputs

RELEASE = ROOT / "analysis/geometric_native_affine_body_validation/release_validation_01"
EXPECTED_REPORT = "a5ed86778a9388a56df2c89d758d17795c861c118bd4de66c6be0b5400b084db"
EXPECTED_REPORT_SEAL = "6e6f0e87f0cd823caea1ebb383ffcb716fa539560024cd24c859b4d35903cf7d"
EXPECTED_NATIVE_REPORT = "a0af761764d8c9283982fecffeeb3e12726538b647d76df08bbef704e72b72de"
EXPECTED_NATIVE_REPORT_SEAL = "bfcfb2a94109f6478a7672c6d38403f4a3f9b47fe7be276a30e3b23c8365d2de"
EXPECTED_PACKET = "ccd511b2b82e66ba103d0d900b993a18d02953288f447c71ac9e589dc70c7d1e"
EXPECTED_PACKET_SEAL = "afbc10eb17cdc173177a32507be37f77f936827fe83e64764165ae702f892697"
EXPECTED_SOURCE_MAP = "a0343749b87411fff19fb966a8ab5771d8115ec2a4e4477b774649a05b3d2d95"
PHYSICAL_COMPONENT_SCOPE = "COLD_TENSOR_COMPONENT_ONLY"
STANDALONE_EVIDENCE = {
    "independent_audit.json": "89ce99a0f12e4f2933465754eeca12bffdebb3d0e1d8c6fcfa5c7b5ded550c4a",
    "independent_audit_source.txt": "7f481c533996070db320b03bb1bcca71487e06361209c74859b3f29f0a2a556c",
}
SOURCE_FIELDS = previous.SOURCE_FIELDS
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_cold_body.py",
    "tectonics/geometric_native_cold_tensor_integration.py",
    "tectonics/geometric_native_cold_strain_energy.py",
    "tests/test_geometric_native_cold_body.py",
    "tests/test_geometric_native_cold_strain_energy.py",
    "analysis/geometric_native_cold_body_validation/validate.py",
    "analysis/geometric_native_cold_body_validation/native_control.py",
))
FALSE_SCOPE = (
    "original_point_1_completed", "whole_clock_material_admission_authorized",
    "complete_three_dimensional_strain_energy_certified",
    "global_physical_solid_noninterpenetration_certified",
    "all_42_surface_pairs_reexecuted", "new_native_curved_force_equilibrium_certified",
    "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement",
    "material_admission_authorized",
)

encoded = previous.encoded
digest = previous.digest
value_digest = previous.value_digest
read = previous.read
sealed = previous.sealed
source_map = previous.source_map
parse_counts = previous.parse_counts
_literal_fraction = previous._literal_fraction
_interval = previous._interval


def _bounds(value):
    return _interval(value)


def _point(value):
    value = Fraction(value)
    return value, value


def _plus(first, second):
    return first[0]+second[0], first[1]+second[1]


def _minus(first, second):
    return first[0]-second[1], first[1]-second[0]


def _times(first, second):
    values = tuple(a*b for a in first for b in second)
    return min(values), max(values)


def _scale(value, scalar):
    return _times(value, _point(scalar))


def _divide(first, second):
    if second[0] <= 0 <= second[1]:
        raise ValueError("A calibrated tensor divisor contains zero")
    return _times(first, (1/second[1], 1/second[0]))


def _sum(values):
    total = _point(0)
    for value in values:
        total = _plus(total, value)
    return total


def _square(value):
    lo, hi = value
    return (Fraction(0) if lo <= 0 <= hi else min(lo*lo, hi*hi), max(lo*lo, hi*hi))


def _rounded(value, precision):
    """Replay the declared relative binary export grid with rational integers."""
    if type(precision) is not int or precision < 1:
        raise ValueError("A tensor receipt requires its literal precision budget")
    maximum = max(abs(value[0]), abs(value[1]))
    if not maximum:
        return _point(0)
    exponent = maximum.numerator.bit_length()-maximum.denominator.bit_length()
    below = (maximum.numerator < maximum.denominator << exponent if exponent >= 0
        else maximum.numerator << -exponent < maximum.denominator)
    exponent -= int(below)
    shift = (3322*precision+999)//1000+8-exponent
    factor = Fraction(1 << shift) if shift >= 0 else Fraction(1, 1 << -shift)
    low, high = value[0]*factor, value[1]*factor
    return Fraction(low.numerator//low.denominator)/factor, Fraction(-((-high.numerator)//high.denominator))/factor


def _same_bounds(value, expected, label):
    if _bounds(value) != expected:
        raise ValueError("Independent interval replay differs: "+label)


def _vector(value, count, label):
    if type(value) is not list or len(value) != count:
        raise ValueError("A complete interval vector is required: "+label)
    return tuple(_bounds(item) for item in value)


def _matrix(value, count, label):
    if type(value) is not list or len(value) != count:
        raise ValueError("A complete interval matrix is required: "+label)
    return tuple(_vector(row, count, label) for row in value)


def _symmetric(values):
    if len(values) != 28:
        raise ValueError("All twenty-eight independent seven-mode coefficients are required")
    matrix = [[None]*7 for _ in range(7)]
    index = 0
    for i in range(7):
        for j in range(i, 7):
            matrix[i][j] = matrix[j][i] = values[index]
            index += 1
    return tuple(tuple(row) for row in matrix)


def _snapshot(value, name):
    body = dict(value)
    actual = body.pop(name)
    if not isinstance(actual, str) or value_digest(body) != actual:
        raise ValueError("An exported complete component snapshot differs: "+name)
    return actual


def _check_cold_body(body, component, packet, source):
    """Match every body source to immutable accepted DTOs, without building it."""
    from tectonics.geometric_intrinsic_geometry import rotation_increment_from_dict
    from tectonics.geometric_native_pose import FractionInterval, certify_rotation_history, interval_atan
    from tectonics.geometric_native_polygons import native_plane_from_dict, _plane_values

    _snapshot(body, "geometry_source_sha256")
    before = packet["before"]
    epoch = before["snapshot"]["current_material_epoch"]
    material = epoch["source_material_state"]
    mechanics = material["original_mechanics_payload"]
    identity, parent_id = component["component_id"], component["original_parent_id"]
    bank = next(row for row in source.banks if row.component_id == identity)
    parents = [row for row in material["journal"]["source_surface"]["fragments"] if row["fragment_id"] == parent_id]
    allocations = [row for row in material["parent_allocations"] if row["fragment_id"] == parent_id]
    bindings = [row for row in material["trace_bindings"] if parent_id in (row["donor_fragment_id"], row["receiver_fragment_id"])]
    if len(parents) != 1 or len(allocations) != 1 or len(bindings) != 1:
        raise ValueError("A cold body lacks one complete original material/trace source")
    binding = bindings[0]
    role = "sub" if binding["donor_fragment_id"] == parent_id else "over"
    traces = mechanics["fault_traces"]["traces"]
    trace_index = next(i for i, row in enumerate(traces) if row["trace_id"] == binding["trace_id"])
    trace = traces[trace_index]
    parameters = next(row[role] for row in mechanics["bending_parameters"] if row["trace_id"] == trace["trace_id"])
    tip_row = next(row for row in before["snapshot"]["transverse_displacement_m"] if row[0] == trace["trace_id"])
    radius = Fraction.from_float(material["journal"]["source_surface"]["radius_km"])*1000
    thickness = sum((Fraction.from_float(value) for value in parents[0]["parcel"]["specific_properties"][:2]), Fraction(0))*1000
    old_index = 3*len(mechanics["plate_ids"])+2*trace_index+int(role == "over")
    stiffness_row = before["original_physical_history"]["original_reference"]["stiffness"][old_index]
    old_k = Fraction.from_float(stiffness_row[old_index])/radius**2
    accepted_tip = Fraction.from_float(tip_row[1 if role == "sub" else 2])
    young, poisson, elastic_h, width = tuple(Fraction.from_float(parameters[name]) for name in
        ("young_modulus_pa", "poisson_ratio", "elastic_thickness_m", "width_m"))
    roof = bank.roof_radial_displacement_m
    elastic_depth = ((radius+roof)**3-(radius+roof-elastic_h)**3)/(3*radius**2)
    if (old_k <= 0 or any(Fraction.from_float(value) for i, value in enumerate(stiffness_row) if i != old_index)
            or roof != accepted_tip or not 0 < elastic_depth < thickness):
        raise ValueError("The original accepted cold elastic reference differs")
    for name, expected in (
            ("component_id", identity), ("material_id", component["material_id"]),
            ("original_parent_id", parent_id), ("owner", component["owner"]),
            ("trace_id", trace["trace_id"]), ("source_bank", role),
            ("old_absolute_coordinate_index", old_index),
            ("bank_alias", "C"+str(sorted(row.component_id for row in source.banks).index(identity))),
            ("source_snapshot_sha256", source.source_snapshot_sha256),
            ("source_support_sha256", source.source_support_sha256),
            ("source_epoch_sha256", source.source_epoch_sha256),
            ("source_packet_raw_sha256", source.curved_packet_raw_sha256),
            ("saved_field_declaration_precision", 128),
            ("coordinate_order", ["old_tip_increment", "xx", "yy", "zz", "xy", "xz", "yz"])):
        if body.get(name) != expected:
            raise ValueError("A cold body source identity differs: "+name)
    for name, expected in (("radius_m", radius), ("areal_material_thickness_m", thickness),
            ("roof_radial_displacement_m", roof), ("accepted_tip_displacement_m", accepted_tip),
            ("source_old_tip_stiffness_n_m", old_k), ("young_modulus_pa", young),
            ("poisson_ratio", poisson), ("elastic_thickness_m", elastic_h),
            ("width_m", width), ("beta_cut_radians", width/radius),
            ("elastic_layer_areal_depth_m", elastic_depth)):
        if _literal_fraction(body.get(name)) != expected:
            raise ValueError("A cold body substituted a literal source scalar: "+name)
    area = _bounds(component["area_bounds_sr"])
    _same_bounds(body["actual_area_bounds_sr"], area, "actual complete cold mask area")
    _same_bounds(body["reference_volume_bounds_m3"], _scale(area, radius**2*thickness), "actual complete cold mask volume")
    corrections = [row for row in epoch["paired_corrections"] if row["negative_component_id"] == identity]
    domains = [row for row in material["sweep_certificate"]["donor_domains"] if row["fragment_id"] == parent_id]
    if len(domains) > 1:
        raise ValueError("A complete cold source must retain a unique once-only archive")
    domain = domains[0] if domains else None
    support = before["temporal_admission"]["original_bank_support"]
    clamp = next(row for row in support["live_clamps"] if row["trace_id"] == trace["trace_id"] and row["bank"] == role)
    if (support["live_original_clamp_support_certified"] is not True
            or support["complete_original_bank_support_certified"] is not True
            or clamp["archived_material_supplies_clamp"] is not False
            or clamp["newborn_material_supplies_clamp"] is not False
            or _literal_fraction(clamp["width_m"]) != width or clamp["owner"] != component["owner"]):
        raise ValueError("The new field lost its authentic complete original clamp support")
    for name, value in (("complete_mask_source_json", component),
            ("complete_parent_source_json", parents[0]), ("complete_parent_allocation_json", allocations[0]),
            ("complete_source_trace_json", trace), ("complete_source_bending_parameters_json", parameters),
            ("complete_source_sweep_certificate_json", material["sweep_certificate"]),
            ("complete_source_clamp_support_json", support), ("source_clamp_row_json", clamp),
            ("saved_field_json", packet["saved_joint_reference_field"])):
        if type(body.get(name)) is not str or encoded(json.loads(body[name])) != encoded(value):
            raise ValueError("A complete cold source payload differs: "+name)
    if body["complete_donor_domain_json"] != ("" if domain is None else encoded(domain).decode()):
        raise ValueError("A cold field changed its exact endpoint archive domain")
    if body["paired_corrections_source_json"] != [encoded(row).decode() for row in corrections]:
        raise ValueError("The entire negative curved source correction was not retained")
    _same_bounds(body["archive_area_bounds_sr"], _bounds(allocations[0]["archived_area_bounds_sr"]), "source total archive")
    _same_bounds(body["endpoint_archive_area_bounds_sr"], _point(0) if domain is None else _bounds(domain["endpoint_area_bounds_sr"]), "source endpoint archive")
    if _vector(body["paired_curved_removed_area_bounds_sr"], len(corrections), "negative curved areas") != tuple(_bounds(row["area_bounds_sr"]) for row in corrections):
        raise ValueError("A negative curved correction was replaced by an area alias")
    start = tuple(Fraction.from_float(value) for value in trace["start"])
    end = tuple(Fraction.from_float(value) for value in trace["end"])
    cross = (start[1]*end[2]-start[2]*end[1], start[2]*end[0]-start[0]*end[2], start[0]*end[1]-start[1]*end[0])
    inward = tuple((-1 if role == "sub" else 1)*value for value in cross)
    for name, expected in (("original_quarter_start", start), ("original_quarter_end", end), ("original_quarter_inward", inward)):
        if tuple(_literal_fraction(value) for value in body[name]) != expected:
            raise ValueError("The supported cold field changed the source quarter frame")
    journal = material["journal"]
    histories = [dict(omega_rad_per_myr=dict(journal["omega_rad_per_myr"])[component["owner"]], dt_myr=journal["dt_myr"])]
    histories.extend(dict(epoch["additional_owner_histories"])[component["owner"]])
    if encoded(json.loads(body["source_owner_history_json"])) != encoded(histories):
        raise ValueError("The cold modal reference omitted the accepted owner history")
    geometry_precision = body["geometry_enclosure_precision"]
    if type(geometry_precision) is not int or not 64 <= geometry_precision <= 512:
        raise ValueError("The accepted cold reference requires its literal geometry precision")
    rotation = certify_rotation_history(tuple(rotation_increment_from_dict(row) for row in histories), geometry_precision).intervals
    expected_rotation = tuple(tuple((value.lo, value.hi) for value in row) for row in rotation)
    if _matrix(body["owner_rotation_matrix_bounds"], 3, "accepted owner rotation") != expected_rotation:
        raise ValueError("The source owner frame was not replayed from its literal history")
    for name, vector in (("current_quarter_start_bounds", start), ("current_quarter_tangent_bounds", end), ("current_quarter_inward_bounds", inward)):
        expected = tuple(_sum(_scale(entry, value) for entry, value in zip(row, vector)) for row in expected_rotation)
        if _vector(body[name], 3, name) != expected:
            raise ValueError("A cold modal frame differs from accepted source motion")
    groups = [row for row in packet["saved_joint_reference_field"]["groups"] if identity in row["component_ids"]]
    if len(groups) != 1:
        raise ValueError("The actual cold bank lacks one saved joint field group")
    axes = [row.shared_depth_axis_per_m for row in source.families if row.family_id in groups[0]["family_ids"]]
    axis = axes[0] if axes else (Fraction(0),)*3
    if any(value != axis for value in axes) or tuple(_literal_fraction(value) for value in body["angular_rate_per_physical_volume_depth_m"]) != axis:
        raise ValueError("The cold field substituted a saved128 angular alias midpoint")
    if any(not bounds[0] <= value <= bounds[1] for value, bounds in zip(axis, _vector(groups[0]["angular_rate_per_physical_volume_depth_m"], 3, "saved128 depth axis"))):
        raise ValueError("The literal cold depth axis is outside its saved128 declaration")
    if domain is None or domain["endpoint_domain"] is None:
        if body["endpoint_plane_source_json"] != "" or body["endpoint_plane_coefficients_in_quarter"] != []:
            raise ValueError("An unremoved cold bank invented an endpoint source plane")
    else:
        planes = [row for row in domain["endpoint_domain"]["supports"] if row["rotations"]]
        if len(planes) != 1 or encoded(json.loads(body["endpoint_plane_source_json"])) != encoded(planes[0]):
            raise ValueError("The source endpoint removed strip lost its unique foreign support")
        plane = _plane_values(native_plane_from_dict(planes[0]), max(geometry_precision, 256))[1]
        expected = tuple(_sum(_scale((value.lo, value.hi), scalar) for value, scalar in zip(plane, frame)) for frame in (start, end, inward))
        if _vector(body["endpoint_plane_coefficients_in_quarter"], 3, "endpoint source plane") != expected or expected[2][1] >= 0:
            raise ValueError("The endpoint strip coefficients differ from their authentic source plane")
    curved_cap, curved_beta = _point(0), _point(0)
    if corrections:
        sweep = material["sweep_certificate"]
        if (parent_id != sweep["extra_endpoint_fixed_parent_id"] or domain is None
                or domain["domain_case"] != "endpoint_plus_curved_swept_extra"
                or (start, end, inward) != ((Fraction(0), Fraction(-1), Fraction(0)),
                    (Fraction(0), Fraction(0), Fraction(1)), (Fraction(1), Fraction(0), Fraction(0)))
                or any(_bounds(sweep["proof_bounds"][name])[0] <= 0 for name in
                    ("c_positive", "y_over_u_positive", "b_derivative_numerator_positive"))):
            raise ValueError("The curved cold removal lost its authentic whole-time source branch")
        ratio = _bounds(sweep["envelope"]["maximum_ratio_bounds"])
        plane_coefficients = _vector(body["endpoint_plane_coefficients_in_quarter"], 3, "source terminal plane")
        terminal = _divide(_scale(plane_coefficients[1], -1), plane_coefficients[2])
        cap_value = interval_atan(FractionInterval(*ratio), max(geometry_precision, 256))
        beta_value = interval_atan(FractionInterval(*terminal), max(geometry_precision, 256))
        curved_cap, curved_beta = (cap_value.lo, cap_value.hi), (beta_value.lo, beta_value.hi)
    _same_bounds(body["paired_curved_cap_theta_width_bounds_radians"], curved_cap, "true whole-clock swept theta cap")
    _same_bounds(body["paired_curved_beta_upper_bounds_radians"], curved_beta, "true whole-clock swept beta extent")
    if not body["modal_reference_provenance"].strip() or not body["provenance"].strip():
        raise ValueError("A new cold modal reference needs explicit model provenance")
    return radius, thickness, old_k, accepted_tip, young, poisson, elastic_depth


def _check_cold_material(material, body, source, young, poisson):
    fields = ("component_id", "body_snapshot_sha256", "source_snapshot_sha256",
        "source_support_sha256", "upper_young_modulus_pa", "upper_poisson_ratio",
        "lower_young_modulus_pa", "lower_shear_modulus_pa", "source_lower_tensor_template_json",
        "material_model", "initial_strain_model", "lower_layer_extension_provenance", "calibration_provenance")
    if set(material) != set(fields) | {"material_snapshot_sha256"}:
        raise ValueError("The full explicit layered cold material declaration is required")
    if value_digest([material[name] for name in fields]) != material["material_snapshot_sha256"]:
        raise ValueError("The complete cold material snapshot differs")
    templates, coefficients = [], []
    for family in source.families:
        columns = json.loads(family.complete_carrier_source_json)["declaration"]["parameters"]["columns"]
        templates.append([family.family_id, columns])
        for role in ("sub", "over"):
            coefficients.append((Fraction.from_float(columns[role+"_normal_modulus_pa"]),
                Fraction.from_float(columns[role+"_shear_modulus_pa"])))
    if not coefficients or len(set(coefficients)) != 1:
        raise ValueError("The lower tensor template is not one authentic uniform source declaration")
    lower_young, lower_shear = coefficients[0]
    for name, expected in (("component_id", body["component_id"]),
            ("body_snapshot_sha256", body["geometry_source_sha256"]),
            ("source_snapshot_sha256", source.source_snapshot_sha256),
            ("source_support_sha256", source.source_support_sha256),
            ("material_model", "new_layered_tensor_with_positive_nonlocal_old_tip_calibration"),
            ("initial_strain_model", "new_initial_strain_w_ref_times_complete_symmetric_tip_gradient")):
        if material[name] != expected:
            raise ValueError("The explicit cold constitutive scope differs: "+name)
    for name, expected in (("upper_young_modulus_pa", young), ("upper_poisson_ratio", poisson),
            ("lower_young_modulus_pa", lower_young), ("lower_shear_modulus_pa", lower_shear)):
        if _literal_fraction(material[name]) != expected:
            raise ValueError("An authentic cold tensor modulus differs: "+name)
    if (encoded(json.loads(material["source_lower_tensor_template_json"])) != encoded(templates)
            or not material["lower_layer_extension_provenance"].strip()
            or not material["calibration_provenance"].strip()):
        raise ValueError("The lower tensor must retain its exact source and explicit new model provenance")
    lower_poisson = lower_young/(2*lower_shear)-1
    if (young <= 0 or lower_young <= 0 or lower_shear <= 0
            or not -1 < poisson < Fraction(1, 2) or not -1 < lower_poisson < Fraction(1, 2)):
        raise ValueError("Each declared full layer tensor must be positive definite")
    return {"upper": (young*poisson/((1+poisson)*(1-2*poisson)), young/(2*(1+poisson))),
        "lower": (2*lower_shear*lower_poisson/(1-2*lower_poisson), lower_shear)}


def _check_raw_tensor_projection(raw, body, material, coordinates):
    """Replay the 73 primary strain integrals without the production assembler."""
    _snapshot(raw, "integral_snapshot_sha256")
    if (raw["component_id"] != body["component_id"]
            or raw["body_snapshot_sha256"] != body["geometry_source_sha256"]
            or raw["material_snapshot_sha256"] != material["material_snapshot_sha256"]
            or tuple(_literal_fraction(value) for value in raw["coordinates_m"]) != coordinates
            or raw["primary_integrals_use_symbolic_tip_remainder"] is not True
            or raw["complete_mixed_force_equilibrium_certified"] is not False
            or raw["physical_material_admission_authorized"] is not False
            or type(raw["taylor_order"]) is not int or not 0 <= raw["taylor_order"] <= 12):
        raise ValueError("The raw supported tensor integral changed its field or physical scope")
    precision = raw["precision"]
    if type(precision) is not int or not 64 <= precision <= 512:
        raise ValueError("The raw cold integration precision is outside its literal budget")
    quarter = _vector(raw["quarter_integral_bounds"], 73, "quarter tensor integrals")
    removal = _vector(raw["removed_material_correction_bounds"], 73, "actual removed tensor integrals")
    actual = tuple(_rounded(_minus(a, b), precision) for a, b in zip(quarter, removal))
    if _vector(raw["actual_material_integral_bounds"], 73, "actual tensor integrals") != actual:
        raise ValueError("A complete actual cold mask integral differs from its signed corrections")
    _same_bounds(raw["reference_tip_strain_stiffness_unintersected_n_m"], actual[0], "unrefined actual tip norm")
    lower_proof = _check_tip_norm_lower_proof(raw["reference_tip_strain_stiffness_lower_bound_proof"], body, material, precision)
    if actual[0][1] < lower_proof:
        raise ValueError("The whole tensor integral contradicts its independent supported norm lower proof")
    k = max(actual[0][0], lower_proof), actual[0][1]
    prem, prem_first, second = actual[1], actual[2:9], _symmetric(actual[9:37])
    rem_norm, rem_first, rem_second = actual[37], actual[38:45], _symmetric(actual[45:73])
    if k[0] <= 0:
        raise ValueError("The actual supported tip tensor norm needs strict positive proof")
    q0 = coordinates[0]
    projection = _plus(_scale(k, q0), prem)
    projection_first = tuple(_plus(k if i == 0 else _point(0), prem_first[i]) for i in range(7))
    norm = _plus(_plus(_scale(k, q0*q0), _scale(prem, 2*q0)), rem_norm)
    first = tuple(_plus(_plus(_plus(_scale(k, q0), prem) if i == 0 else _point(0),
        _scale(prem_first[i], q0)), rem_first[i]) for i in range(7))
    hessian = tuple(tuple(_sum((k if i == j == 0 else _point(0),
        prem_first[j] if i == 0 else _point(0), prem_first[i] if j == 0 else _point(0),
        rem_second[i][j], _scale(second[i][j], q0))) for j in range(7)) for i in range(7))
    for name, expected in (("reference_tip_strain_stiffness_n_m", k), ("projection_n", projection),
            ("incremental_strain_inner_product_j", norm), ("projection_remainder_n", prem),
            ("remainder_strain_inner_product_j", rem_norm)):
        _same_bounds(raw[name], expected, name)
    for name, expected in (("projection_gradient_n_m", projection_first),
            ("incremental_local_gradient_n", first), ("projection_remainder_gradient_n_m", prem_first),
            ("remainder_local_gradient_n", rem_first)):
        if _vector(raw[name], 7, name) != expected:
            raise ValueError("A full raw tensor gradient differs: "+name)
    for name, expected in (("projection_hessian_n_m2", second),
            ("incremental_local_hessian_n_m", hessian), ("remainder_local_hessian_n_m", rem_second)):
        if _matrix(raw[name], 7, name) != expected:
            raise ValueError("A full raw tensor Hessian differs: "+name)
    return k, prem, prem_first, second, rem_norm, rem_first, rem_second, norm


def _check_tip_norm_lower_proof(proof, body, material, precision):
    """Derive the complete actual-mask tip norm coercivity from exact moments."""
    from tectonics.geometric_native_pose import FractionInterval, interval_pi
    from tectonics.geometric_native_spherical_solid import interval_cbrt

    beta, radius = _literal_fraction(body["beta_cut_radians"]), _literal_fraction(body["radius_m"])
    young, poisson = _literal_fraction(material["upper_young_modulus_pa"]), _literal_fraction(material["upper_poisson_ratio"])
    lower_young, lower_shear = _literal_fraction(material["lower_young_modulus_pa"]), _literal_fraction(material["lower_shear_modulus_pa"])
    upper_shear = young/(2*(1+poisson))
    upper_lame = young*poisson/((1+poisson)*(1-2*poisson))
    lower_poisson = lower_young/(2*lower_shear)-1
    lower_lame = 2*lower_shear*lower_poisson/(1-2*lower_poisson)
    if min(upper_lame, lower_lame) < 0 or not 0 < beta < 1:
        raise ValueError("The genuine tip coercivity proof requires nonnegative full layer Lamé coefficients")
    pi_value = interval_pi(precision)
    pi = pi_value.lo, pi_value.hi
    shape_moment, slope_moment = Fraction(33, 140)*beta, Fraction(6, 5)/beta
    maximum = 4+Fraction(9, 4)/(beta*beta)
    archive = _bounds(body["archive_area_bounds_sr"])
    curved = tuple(_bounds(value) for value in body["paired_curved_removed_area_bounds_sr"])
    removed = archive[1]+sum((value[1] for value in curved), Fraction(0))
    angular = pi[0]/2*(1-beta*beta/2)*(4*shape_moment+slope_moment)-maximum*removed
    top = radius+_literal_fraction(body["roof_radial_displacement_m"])
    thickness = _literal_fraction(body["areal_material_thickness_m"])
    bottom_value = interval_cbrt(FractionInterval(top**3-3*radius**2*thickness), precision)
    bottom = bottom_value.lo, bottom_value.hi
    upper_height = _literal_fraction(body["elastic_thickness_m"])
    lower_height = top-upper_height-bottom[1], top-upper_height-bottom[0]
    if angular <= 0 or lower_height[0] <= 0:
        raise ValueError("The authentic retained cold material has no positive tip norm coercivity")
    lower = angular*(upper_shear*upper_height+lower_shear*lower_height[0])
    for name, expected in (("beta_cut_radians", beta),
            ("transverse_shape_squared_integral_radians", shape_moment),
            ("transverse_slope_squared_integral_per_radian", slope_moment),
            ("maximum_removed_angular_strain_factor", maximum),
            ("angular_integral_lower_bound", angular), ("upper_layer_shear_pa", upper_shear),
            ("lower_layer_shear_pa", lower_shear), ("physical_upper_layer_height_m", upper_height),
            ("stiffness_lower_bound_n_m", lower)):
        if _literal_fraction(proof[name]) != expected:
            raise ValueError("An independent actual-mask tip coercivity coefficient differs: "+name)
    for name, expected in (("pi_bounds", pi), ("source_archive_area_bounds_sr", archive),
            ("physical_lower_layer_height_bounds_m", lower_height),
            ("physical_material_bottom_radius_bounds_m", bottom)):
        _same_bounds(proof[name], expected, name)
    if (_vector(proof["paired_curved_removed_area_bounds_sr"], len(curved), "coercivity curved removal") != curved
            or proof["positive_lame_coefficients_retained"] is not True
            or proof["strain_quadratic_form_lower_bound_includes_depth_shear"] is not True):
        raise ValueError("The analytic tip norm lower proof omitted genuine removed material or depth shear")
    return lower


def _check_calibrated_response(response, body, material, old_k, accepted_tip):
    """Differentiate the explicit positive rank-one calibration independently."""
    if (encoded(response.get("body")) != encoded(body)
            or encoded(response.get("material")) != encoded(material)):
        raise ValueError("The energy response substituted its complete field or material declaration")
    coordinates = tuple(_literal_fraction(value) for value in response["coordinates_m"])
    if len(coordinates) != 7 or not all(coordinates):
        raise ValueError("The cold diagnostic must exercise every supported physical mode")
    for name in ("nonlocal_constitutive_calibration_declared",
            "accepted_global_beam_reference_energy_preserved", "accepted_global_beam_reference_tip_force_preserved"):
        if response.get(name) is not True:
            raise ValueError("The explicitly new calibrated cold source reference was not preserved")
    for name in ("old_beam_stiffness_added_as_second_energy", "historical_full_three_dimensional_prestress_recovered",
            "historical_archived_strain_energy_density_recovered", "complete_three_dimensional_strain_energy_certified",
            "mixed_force_equilibrium_certified", "material_admission_authorized", "original_point_1_completed"):
        if response.get(name) is not False:
            raise ValueError("The cold component response exceeded its declared limited physical scope")
    raw = response["raw_integrals"]
    k, prem, prem_gradient, second, rem_norm, rem_gradient, rem_hessian, full_norm = _check_raw_tensor_projection(raw, body, material, coordinates)
    reference_energy, reference_force = old_k*accepted_tip*accepted_tip/2, old_k*accepted_tip
    if (_literal_fraction(response["reference_energy_j"]) != reference_energy
            or _literal_fraction(response["reference_tip_force_n"]) != reference_force):
        raise ValueError("The authentic old global beam energy or tip force was reset")
    projection_increment = _plus(_point(coordinates[0]), _divide(prem, k))
    total = _plus(_point(accepted_tip), projection_increment)
    derivatives = tuple(_plus(_point(int(i == 0)), _divide(prem_gradient[i], k)) for i in range(7))
    residual = _minus(_scale(rem_norm, Fraction(1, 2)), _divide(_square(prem), _scale(k, 2)))
    if residual[1] < 0:
        raise ValueError("The positive tensor orthogonal-complement proof is inconsistent")
    residual = max(Fraction(0), residual[0]), residual[1]
    energy = _plus(_scale(_square(total), old_k/2), residual)
    gradient = tuple(_minus(_plus(_scale(_times(total, derivatives[i]), old_k), rem_gradient[i]),
        _times(_divide(prem, k), prem_gradient[i])) for i in range(7))
    hessian = tuple(tuple(_plus(_minus(_plus(_scale(_times(derivatives[i], derivatives[j]), old_k), rem_hessian[i][j]),
        _divide(_times(prem_gradient[i], prem_gradient[j]), k)),
        _times(_divide(_minus(_scale(total, old_k), prem), k), second[i][j])) for j in range(7)) for i in range(7))
    _same_bounds(response["energy_bounds_j"], energy, "new calibrated cold energy")
    _same_bounds(response["raw_green_energy_bounds_j"], _scale(full_norm, Fraction(1, 2)), "complete raw Green energy")
    _same_bounds(response["projection_increment_m"], projection_increment, "correlated source tip increment")
    _same_bounds(response["total_old_tip_strain_projection_m"], total, "nonzero accepted tip strain")
    if (_vector(response["generalized_gradient_bounds_n"], 7, "calibrated seven-mode forces") != gradient
            or _matrix(response["generalized_hessian_bounds_n_m"], 7, "calibrated seven-mode Hessian") != hessian):
        raise ValueError("Independent full calibrated virtual-work derivatives differ")
    if hessian != tuple(zip(*hessian)):
        raise ValueError("The independent calibrated Hessian is not symmetric")
    if energy[0] < 0:
        raise ValueError("The explicitly positive calibrated cold potential lost its bound")
    return coordinates, reference_energy


def _check_strain_support(support, body, coordinates, precision):
    """Replay the local metric proof; grant no global embedding or admission."""
    from tectonics.geometric_native_pose import FractionInterval, interval_sqrt
    from tectonics.geometric_native_spherical_solid import interval_cbrt

    radius = _literal_fraction(body["radius_m"])
    top = radius+_literal_fraction(body["roof_radial_displacement_m"])
    thickness = _literal_fraction(body["areal_material_thickness_m"])
    cube = top**3-3*radius**2*thickness
    if cube <= 0:
        raise ValueError("The complete actual cold body reaches a nonpositive radius")
    lower = interval_cbrt(FractionInterval(cube), precision).lo
    axis = tuple(_literal_fraction(value) for value in body["angular_rate_per_physical_volume_depth_m"])
    rate = interval_sqrt(FractionInterval(sum((value*value for value in axis), Fraction(0))), precision).hi
    beta = _literal_fraction(body["beta_cut_radians"])
    if not 0 < beta <= Fraction(1, 4):
        raise ValueError("The whole clamped field needs its bounded authentic source angle")
    beta_gradient = 1/lower+top**2/radius**2*rate/(1-beta*beta/2)
    tip_norm = 1/lower+Fraction(3, 2)/beta*beta_gradient
    enrichment_norm = 1/radius+Fraction(3, 2)/beta*(top/radius)*beta_gradient
    initial_norm = 2*abs(_literal_fraction(body["accepted_tip_displacement_m"]))*tip_norm
    increment_norm = abs(coordinates[0])*tip_norm+sum((abs(value)*enrichment_norm for value in coordinates[1:]), Fraction(0))
    if initial_norm >= 1 or increment_norm >= 1:
        raise ValueError("The complete cold modal increment lost the certified metric domain")
    if (support["body_snapshot_sha256"] != body["geometry_source_sha256"]
            or tuple(_literal_fraction(value) for value in support["coordinates_m"]) != coordinates):
        raise ValueError("The local whole-mask support proof belongs to another field")
    for name, expected in (("reference_tip_gradient_norm_upper_per_m", tip_norm),
            ("enrichment_gradient_norm_upper_per_m", enrichment_norm),
            ("incremental_gradient_norm_upper", increment_norm)):
        if _literal_fraction(support[name]) != expected:
            raise ValueError("A complete cold gradient bound differs from its source formula")
    expected = {"reference_radius_bounds_m": (lower, top),
        "initial_metric_eigenvalue_bounds": (1-initial_norm, 1+initial_norm),
        "incremental_jacobian_bounds": ((1-increment_norm)**3, (1+increment_norm)**3),
        "total_elastic_metric_eigenvalue_bounds": ((1-increment_norm)**2-initial_norm, (1+increment_norm)**2+initial_norm)}
    for name, value in expected.items():
        _same_bounds(support[name], value, name)
        if value[0] <= 0:
            raise ValueError("The independent local cold metric/J bound is not positive")


def _check_tensor_cells(raw, body, coefficients):
    """Check complete cell coverage and signed receipt arithmetic, not re-integrate."""
    cells = raw["cell_receipts"]
    precision = raw["precision"]
    if type(cells) is not list or not cells or raw["integration_cell_count"] != len(cells):
        raise ValueError("The complete cold tensor cell proof was truncated")
    beta_end = _literal_fraction(body["beta_cut_radians"])
    elastic_depth = _literal_fraction(body["elastic_layer_areal_depth_m"])
    thickness = _literal_fraction(body["areal_material_thickness_m"])
    layer_ranges = {"upper": (Fraction(0), elastic_depth), "lower": (elastic_depth, thickness)}
    quarter, tails = [_point(0)]*73, [Fraction(0)]*73
    for layer, (z0, z1) in layer_ranges.items():
        selected = [cell for cell in cells if cell["layer"] == layer]
        if not selected:
            raise ValueError("Both physical cold tensor layers must be integrated")
        beta_intervals = sorted({_bounds(cell["beta_bounds_radians"]) for cell in selected})
        depth_intervals = sorted({_bounds(cell["areal_material_depth_bounds_m"]) for cell in selected})
        for intervals, lo, hi in ((beta_intervals, Fraction(0), beta_end), (depth_intervals, z0, z1)):
            if (intervals[0][0] != lo or intervals[-1][1] != hi
                    or any(a >= b for a, b in intervals)
                    or any(first[1] != second[0] for first, second in zip(intervals, intervals[1:]))):
                raise ValueError("The tensor cells do not cover the entire clamped quarter/depth")
        if (len(selected) != len(beta_intervals)*len(depth_intervals)
                or {(_bounds(cell["beta_bounds_radians"]), _bounds(cell["areal_material_depth_bounds_m"])) for cell in selected}
                != {(beta, depth) for beta in beta_intervals for depth in depth_intervals}):
            raise ValueError("The complete quarter tensor partition contains gaps or duplicates")
    for cell in cells:
        layer = cell["layer"]
        if layer not in coefficients or tuple(_literal_fraction(cell[name]) for name in ("lame_pa", "shear_pa")) != coefficients[layer]:
            raise ValueError("A cold tensor integration cell substituted its physical layer tensor")
        polynomial = _vector(cell["polynomial_integral_bounds"], 73, "whole cell polynomial integral")
        remainder = tuple(_literal_fraction(value) for value in cell["remainder_absolute_upper_bounds"])
        if len(remainder) != 73 or any(value < 0 for value in remainder):
            raise ValueError("All complete tensor derivative remainders must be nonnegative")
        bounds = tuple(_rounded(_plus(value, (-tail, tail)), precision) for value, tail in zip(polynomial, remainder))
        if _vector(cell["integral_bounds"], 73, "certified cell integral") != bounds:
            raise ValueError("A complete tensor cell lost its outward derivative tail")
        for index, value in enumerate(bounds):
            quarter[index] = _rounded(_plus(quarter[index], value), precision)
            tails[index] += remainder[index]
    if (tuple(quarter) != _vector(raw["quarter_integral_bounds"], 73, "summed quarter integral")
            or tuple(tails) != tuple(_literal_fraction(value) for value in raw["maximum_taylor_remainder_absolute_upper_bounds"])):
        raise ValueError("The complete quarter integral or derivative-tail sum differs")


def _check_tensor_corrections(raw, body):
    """Replay source endpoint-cap partition and every signed correction receipt."""
    precision = raw["precision"]
    radius = _literal_fraction(body["radius_m"])
    elastic = _literal_fraction(body["elastic_layer_areal_depth_m"])
    thickness = _literal_fraction(body["areal_material_thickness_m"])
    layer_ranges = {"upper": (Fraction(0), elastic), "lower": (elastic, thickness)}
    endpoint = _bounds(body["endpoint_archive_area_bounds_sr"])
    curved = tuple(_bounds(value) for value in body["paired_curved_removed_area_bounds_sr"])
    receipts = raw["correction_receipts"]
    expected_count = 2*(int(endpoint != _point(0))+sum(value != _point(0) for value in curved))
    if type(receipts) is not list or len(receipts) != expected_count:
        raise ValueError("Every nonzero endpoint and paired curved removal needs both layer receipts")
    parameters = None
    if endpoint != _point(0):
        a, b, c = _vector(body["endpoint_plane_coefficients_in_quarter"], 3, "true endpoint plane")
        if c[1] >= 0:
            raise ValueError("The genuine endpoint source plane lacks inward-negative proof")
        d = _rounded(_divide(_scale(a, -1), c), precision+16)
        e = _rounded(_divide(_scale(b, -1), c), precision+16)
        def cap_atan(value):
            if value[0] < 0 or value[1] > Fraction(1, 10):
                raise ValueError("A native removed cap exceeded its certified small-angle range")
            return _rounded((max(Fraction(0), value[0]-value[1]**3/3), value[1]), precision+16)
        if d[0] >= 0 and e[0] >= 0:
            case, cap, maximum = "whole_quarter", _point(0), d[1]+e[1]
        elif d[0] > 0 and e[1] < 0:
            case, cap, maximum = "quarter_minus_terminal_cap", cap_atan(_divide(_scale(e, -1), d)), d[1]
        elif d[1] < 0 and e[0] > 0:
            case, cap, maximum = "terminal_cap_only", cap_atan(_divide(e, _scale(d, -1))), e[1]
        else:
            raise ValueError("The endpoint source signs require a separately certified partition")
        if maximum > _literal_fraction(body["beta_cut_radians"]):
            raise ValueError("The complete endpoint removal reaches beyond the physical supported collar")
        parameters = d, e, case, cap, maximum
    total = [_point(0)]*73
    seen = {layer: [] for layer in layer_ranges}
    for receipt in receipts:
        layer = receipt["layer"]
        if layer not in layer_ranges or _bounds(receipt["areal_material_depth_bounds_m"]) != layer_ranges[layer]:
            raise ValueError("A removed material receipt changed its physical layer depth")
        kind = receipt["correction_kind"]
        area = _bounds(receipt["removed_area_bounds_sr"])
        scalar_bounds = _vector(receipt["complete_supported_quarter_scalar_bounds"], 73, "removed-mask scalar bounds")
        if kind == "endpoint_archive_exact_theta_strip":
            if parameters is None or area != endpoint:
                raise ValueError("An endpoint correction substituted another actual archive measure")
            d, e, case, cap, maximum = parameters
            for name, expected in (("plane_cos_coefficient_bounds", d), ("plane_sin_coefficient_bounds", e),
                    ("endpoint_cap_width_bounds_radians", cap)):
                _same_bounds(receipt[name], expected, name)
            if receipt["positive_theta_case"] != case or _literal_fraction(receipt["maximum_positive_delta_upper_bound"]) != maximum:
                raise ValueError("The source endpoint correction changed its true theta strip")
            polynomial = _vector(receipt["polynomial_integral_bounds"], 73, "removed-strip polynomial")
            depth_tails = tuple(_literal_fraction(value) for value in receipt["depth_remainder_absolute_upper_bounds"])
            transverse_tails = tuple(_literal_fraction(value) for value in receipt["transverse_and_atan_remainder_absolute_upper_bounds"])
            if (len(depth_tails) != 73 or len(transverse_tails) != 73
                    or any(value < 0 for value in depth_tails+transverse_tails)):
                raise ValueError("The removed strip must preserve both complete depth and transverse tails")
            expected = tuple(_rounded(_plus(_rounded(_plus(value, (-depth, depth)), precision), (-transverse, transverse)), precision)
                for value, depth, transverse in zip(polynomial, depth_tails, transverse_tails))
        elif kind == "paired_curved_removed":
            if area not in curved or area == _point(0):
                raise ValueError("A curved correction substituted an unauthenticated source measure")
            z0, z1 = layer_ranges[layer]
            _same_bounds(receipt["theta_terminal_cap_width_bounds_radians"],
                _bounds(body["paired_curved_cap_theta_width_bounds_radians"]), "authentic whole-sweep curved theta localization")
            beta = _bounds(body["paired_curved_beta_upper_bounds_radians"])
            _same_bounds(receipt["beta_bounds_radians"], (Fraction(0), beta[1]), "authentic whole-sweep curved beta localization")
            multiplier = _scale(area, radius**2*(z1-z0))
            expected = tuple(_rounded(_times(value, multiplier), precision) for value in scalar_bounds)
        else:
            raise ValueError("An unbound material removal law appeared in cold tensor evidence")
        seen[layer].append((kind, area))
        if _vector(receipt["correction_integral_bounds"], 73, "actual source removed integral") != expected:
            raise ValueError("A complete removed-mask tensor correction differs from its source receipt")
        for index, value in enumerate(expected):
            total[index] = _rounded(_plus(total[index], value), precision)
    expected_removals = ([] if endpoint == _point(0) else [("endpoint_archive_exact_theta_strip", endpoint)])+[
        ("paired_curved_removed", area) for area in curved if area != _point(0)]
    if any(sorted(values) != sorted(expected_removals) for values in seen.values()):
        raise ValueError("Endpoint and curved removed masks were omitted or counted twice")
    if tuple(total) != _vector(raw["removed_material_correction_bounds"], 73, "complete signed removal sum"):
        raise ValueError("The complete actual cold removed-material tensor sum differs")


def verify_prior():
    """Authenticate all 826 bodies and every historical output and audit."""
    inherited = previous.verify_prior()
    path = RELEASE / "report.json"
    if digest(path) != EXPECTED_REPORT:
        raise ValueError("Authoritative affine-body regression report changed")
    report = sealed(path, "report_sha256")
    if (read(path).get("report_sha256") != EXPECTED_REPORT_SEAL
            or report.get("status") != "PASS"
            or report.get("regression_validation_status") != "PASS"
            or report.get("physical_candidate_status") != previous.PHYSICAL_COMPONENT_SCOPE
            or report.get("source_count") != 826
            or report.get("pytest_counts") != {"passed": 39}
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or set(report.get("added_python_sources", {})) != previous.ADDED_SOURCE_PATHS
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("Affine-body source generation or component-only scope differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 826 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The original 826-source generation differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    _verify_outputs(RELEASE, report, 4)
    if {str(path.relative_to(RELEASE)) for path in RELEASE.rglob("*") if path.is_file()} != set(report["outputs"]) | {"report.json"}:
        raise ValueError("The sealed affine-body release gained or lost a file")
    native_path = RELEASE / "native_control/report.json"
    packet_path = RELEASE / "native_control/source_packet.json"
    if (digest(native_path) != EXPECTED_NATIVE_REPORT
            or digest(packet_path) != EXPECTED_PACKET
            or read(native_path).get("report_sha256") != EXPECTED_NATIVE_REPORT_SEAL
            or read(packet_path).get("packet_sha256") != EXPECTED_PACKET_SEAL):
        raise ValueError("The complete preserved affine-body evidence changed")
    native = sealed(native_path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    if (native.get("status") != "PASS"
            or native.get("physical_candidate_status") != previous.PHYSICAL_COMPONENT_SCOPE
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or native.get("actual_hot_whole_mask_bulk_component_count") != 5
            or native.get("actual_hot_basal_boundary_component_count") != 5
            or native.get("actual_hot_native_pressure_trace_count") != 3
            or native.get("actual_hot_shared_contact_body_count") != 1
            or native.get("actual_cold_clamped_body_component_count") != 0
            or packet.get("accepted_input_unchanged") is not True
            or packet.get("material_admission_authorized") is not False):
        raise ValueError("The preserved affine-body source or scope differs")
    for row in (report, native):
        if (row.get("accepted_original_native_intervals") != 1
                or row.get("accepted_new_native_intervals") != 0
                or any(row.get(name) is not False for name in FALSE_SCOPE)):
            raise ValueError("Historical affine-body evidence gained physical acceptance")
    for name, expected in STANDALONE_EVIDENCE.items():
        path = RELEASE.parent / name
        if digest(path) != expected:
            raise ValueError("A standalone affine-body audit changed: " + name)
        if path.suffix == ".json":
            row = sealed(path, "report_sha256")
            if (row.get("status") != "PASS" or row.get("fresh_tests") != 39
                    or row.get("final_python_sources") != 826
                    or row.get("source_map_canonical_sha256") != EXPECTED_SOURCE_MAP
                    or row.get("original_point_1_completed") is not False
                    or row.get("material_admission_authorized") is not False):
                raise ValueError("The independent affine-body audit scope differs")
    return inherited


def _check_cold_component_evidence(native, packet):
    """Bind whole source masks and independently replay tensor/projection data."""
    from tectonics.geometric_native_normal_support import load_native_normal_support_source

    rows = native.get("cold_components")
    if (type(rows) is not list or len(rows) != 8
            or encoded(rows) != encoded(packet.get("cold_components"))):
        raise ValueError("All eight identical complete cold proofs are required in both outputs")
    source = load_native_normal_support_source()
    epoch = packet["before"]["snapshot"]["current_material_epoch"]
    components = {row["component_id"]: row for row in epoch["components"] if row["material_kind"] == "original_live"}
    if len(components) != 8 or {row.get("component_id") for row in rows} != set(components):
        raise ValueError("The cold body proof substituted or omitted an authentic complete component")
    reference_total = Fraction(0)
    for row in rows:
        identity = row["component_id"]
        body, material, response = row["body"], row["material"], row["diagnostic_response"]
        radius, thickness, old_k, accepted_tip, young, poisson, elastic_depth = _check_cold_body(body, components[identity], packet, source)
        coefficients = _check_cold_material(material, body, source, young, poisson)
        coordinates, reference_energy = _check_calibrated_response(response, body, material, old_k, accepted_tip)
        raw = response["raw_integrals"]
        _check_strain_support(response["support"], body, coordinates, max(raw["precision"], 160))
        _check_tensor_cells(raw, body, coefficients)
        _check_tensor_corrections(raw, body)
        if (row.get("probe_is_new_mixed_force_solution") is not False
                or row.get("accepted_reference_energy_and_tip_force_preserved") is not True):
            raise ValueError("A cold diagnostic claimed a new physical force solution")
        reference_total += reference_energy
    for payload in (native, packet):
        if _literal_fraction(payload["accepted_reference_total_beam_energy_j"]) != reference_total:
            raise ValueError("The eight authentic accepted cold beam energies were not conserved")
    if (native.get("actual_cold_supported_body_component_count") != 8
            or native.get("accepted_reference_energy_preserved") is not True
            or native.get("accepted_reference_tip_forces_preserved") is not True
            or native.get("explicit_initial_strain_declared") is not True
            or native.get("explicit_new_nonlocal_calibrated_tensor_extension") is not True
            or any(native.get(name) is not False for name in (
                "historical_per_material_strain_distribution_recovered", "old_beam_stiffness_added_again",
                "source_material_floor_used", "source_area_midpoints_substituted", "source_normal_column_lengths_used_as_geometry"))):
        raise ValueError("The cold component reference, provenance or scope differs")
    return dict(actual_cold_supported_body_component_count=8,
        independent_cold_source_mask_reference_bindings_verified=8,
        independent_cold_calibrated_energy_force_hessian_replays_verified=8,
        independent_raw_tensor_primary_scalar_count_per_component=73,
        complete_tensor_cell_coverage_and_receipt_arithmetic_verified=True,
        independent_second_certified_tensor_quadrature_performed=False,
        accepted_reference_energy_preserved=True, accepted_reference_tip_forces_preserved=True,
        explicit_new_nonlocal_calibrated_tensor_extension=True,
        historical_per_material_strain_distribution_recovered=False,
        old_beam_stiffness_added_again=False,
        accepted_reference_total_beam_energy_j=[reference_total.numerator, reference_total.denominator])


def _check_native_control(out):
    path = out / "native_control/report.json"
    packet_path = out / "native_control/source_packet.json"
    native = sealed(path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    original = sealed(RELEASE / "native_control/source_packet.json", "packet_sha256")
    if (native.get("status") != "PASS"
            or native.get("physical_candidate_status") != PHYSICAL_COMPONENT_SCOPE
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("original_point_1_remaining_work_blocks") != 1
            or type(native.get("remaining_physical_conditions")) is not list
            or len(native["remaining_physical_conditions"]) != 3
            or native.get("accepted_input_complete_typed_state_unchanged") is not True
            or any(native.get(name) is not False for name in FALSE_SCOPE)
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("authenticated_affine_source_packet_raw_sha256") != EXPECTED_PACKET
            or packet.get("authenticated_affine_source_packet_seal") != EXPECTED_PACKET_SEAL
            or packet.get("accepted_input_unchanged") is not True
            or packet.get("material_admission_authorized") is not False
            or packet.get("inherited_force_certificate_for_placement") is not False
            or packet.get("full_genuine_source_reconstructed") is not False):
        raise ValueError("Native cold-body proof exceeded its component-only scope")
    for name in SOURCE_FIELDS:
        if encoded(packet.get(name)) != encoded(original[name]):
            raise ValueError("A complete authenticated source payload differs: " + name)
    for name in ("coordinates_decimal", "literal_owner_axes"):
        if encoded(packet.get(name)) != encoded(original[name]):
            raise ValueError("The canonical saved source pose differs: " + name)
    reference = packet["saved_joint_reference_field"]
    if reference["declaration_precision"] != 128 or reference["component_count"] != 13:
        raise ValueError("The literal saved source reference field differs")
    component_evidence = _check_cold_component_evidence(native, packet)
    return dict(**component_evidence,
        native_control_report_raw_sha256=digest(path),
        native_control_source_packet_raw_sha256=digest(packet_path),
        physical_candidate_status=PHYSICAL_COMPONENT_SCOPE,
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
        raise ValueError("The two new cold-body test modules and a module:function are required")
    inherited = verify_prior()
    before = source_map()
    added = set(before) - set(inherited)
    if len(before) != 833 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly seven additive cold-body Python sources are required")
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
        format="native-cold-body-same-process-validation-1",
        status="FAIL", regression_validation_status="FAIL",
        physical_candidate_status="NOT_CHECKED",
        started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_solid_placement_outputs=5,
        verified_radial_normalization_diagnosis=True, verified_prior_python_sources=826,
        verified_finite_equilibrium_outputs=4, verified_failed_release_outputs=3,
        verified_spherical_solid_outputs=4, verified_joint_spherical_outputs=4,
        verified_curved_equilibrium_outputs=4, verified_normal_support_outputs=4,
        verified_affine_body_outputs=4,
        verified_normal_support_standalone_raw_evidence=dict(previous.STANDALONE_EVIDENCE),
        verified_affine_body_standalone_raw_evidence=dict(STANDALONE_EVIDENCE),
        prior_affine_body_report_raw_sha256=EXPECTED_REPORT,
        prior_affine_body_native_report_raw_sha256=EXPECTED_NATIVE_REPORT,
        prior_affine_body_packet_raw_sha256=EXPECTED_PACKET,
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
            raise RuntimeError("Fresh source-bound cold-body regressions failed")
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
            raise ValueError("The cold-body gate requires exactly four declared output files")
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

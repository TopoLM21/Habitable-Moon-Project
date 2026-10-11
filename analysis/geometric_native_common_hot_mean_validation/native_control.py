"""Whole authentic HOT law and finite radial-base mean controls.

The inherited cold component is authenticated from its saved complete packet;
its integral, tests and solver are not reexecuted. All new states here are
diagnostic component queries, with the genuine source HOT30 remaining zero.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_SCOPE = "DECLARED_COMPLETE_HOT_FINITE_BASE_MEAN_ENDPOINT_ONLY"
PROVENANCE = ("Whole original five HOT supports with preserved birth/material/work histories; "
              "NEW fixed cold895 owner scales on authentic879 source-world HOT/depth reference, Green StVK E/G "
              "and objective finite radial Winkler base law with implicit actual-base mean")
OPTIONS = dict(precision=96, root_iterations=12)
AUTHORITIES = (
    ("analysis/geometric_native_common_material_surface_transport_validation/release_validation_02/native_control/source_packet.json",
     "f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64",
     "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"),
    ("analysis/geometric_native_common_carrier_source_work_validation/release_validation_02/native_control/source_packet.json",
     "b45d5d4d80c74c509bfac033c4a60c5b96c8327704e0c727fcf680406d82e57e",
     "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"),
    ("analysis/geometric_native_common_cold_continuum_energy_validation/release_validation_01/native_control/source_packet.json",
     "4a5c059176cef063bf12ff7bd98df830395504d6c12051f269a177fcbc375fec",
     "c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439"),
    ("analysis/geometric_native_common_reference_validation/release_validation_01/native_control/source_packet.json",
     "dc8f820dbd926c53223a5c141a10296d1c7135efc6bf1077763785866dda9275",
     "60259e6e56bb7335fa994988adfbbac5ac933b524ca164b0298c2e0a16a40446"),
    ("analysis/geometric_native_normal_support_validation/release_validation_01/native_control/source_packet.json",
     "f4f3703cbc463d4130096dba827a7425fc865851fe5c86f337f98dfbefd8f7ca",
     "e2db27aa164161482117993f4f23010293eae45100c91d3a52ff0508075688ee"),
    ("analysis/geometric_native_curved_equilibrium_validation/release_validation_01/native_control/candidate_packet.json",
     "0e3554a55a8c93eb8ecbdae8351f3280a2ae4bd8596254950afef7f77b98d2cb",
     "f192cb4a9fbdcfd83ab4bad69e1c5c12307bfe4df78626aa514deb3d4f84cccd"),
)
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _read_authority(authority):
    path, raw_pin, seal = authority
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_pin:
        raise ValueError("The complete immutable authority changed: " + path)
    packet = json.loads(raw)
    if (packet["packet_sha256"] != seal or
            digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != seal):
        raise ValueError("The complete authority selfseal changed: " + path)
    return packet


@lru_cache(maxsize=1)
def source_packets():
    """Full879/891/899/commonHOT/normal/conditional19, never old controls."""
    return tuple(_read_authority(authority) for authority in AUTHORITIES)


@lru_cache(maxsize=1)
def system():
    from tectonics.geometric_native_common_hot_mean import bind_native_common_hot_mean
    transport, work, cold, common, normal, curved = source_packets()
    return bind_native_common_hot_mean(transport, work, cold, common_hot_packet=common,
        normal_support_packet=normal, curved19_packet=curved,
        provenance=PROVENANCE, precision=OPTIONS["precision"])


@lru_cache(maxsize=8)
def response(coordinates=None):
    from tectonics.geometric_native_common_hot_mean import evaluate_native_common_hot_mean
    return evaluate_native_common_hot_mean(system(), coordinates, **OPTIONS)


def diagnostic_coordinates():
    """NEW literal q values; saved old Cartesian probes are not source states.

    Saved COMMON HOT source order is xx,yy,zz,xy,xz,yz. The trace-free
    amplitudes retain that saved probe and its exact shear order.
    q is explicitly chosen as its literal trace/3, not its old physical mean.
    """
    result = []
    for body in system().bodies:
        a = body.saved_nonzero_probe_old_cartesian_amplitudes_m
        if len(a) != 6:
            raise ValueError("A complete saved six-amplitude HOT diagnostic is required")
        result.extend(((a[0]+a[1]+a[2])/3, (a[0]-a[1])/2,
                       (a[0]+a[1]-2*a[2])/6, a[3], a[4], a[5]))
    if len(result) != 30 or any(x == 0 for x in result):
        raise ValueError("The declared nonzero diagnostic must exercise every HOT30 coordinate")
    return tuple(result)


def query_coordinates():
    probe = diagnostic_coordinates()
    isotropic = tuple(probe[i] if i % 6 == 0 else F(0) for i in range(30))
    return (("genuine_zero_hot_source", system().bound_coordinate_increments),
            ("new_full30_finite_mean_diagnostic", probe),
            ("new_reversed_full30_finite_mean_diagnostic", tuple(-x for x in probe)),
            ("new_all_five_isotropic_mean_diagnostic", isotropic))


def summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        complete_cold_continuum_operator_certified=True,
        complete_hot_material_constitutive_operator_certified=True,
        finite_hot_base_mean_chart_certified=True,
        whole_cold_material_support_geometry_certified=True,
        hot_constitutive_law_defined=True, finite_mean_chart_defined=True,
        finite_mean_equilibrium_certified=False,
        physical_source_force_prestress_work_handover_certified=False,
        historical_archived_three_dimensional_prestress_recovered=False,
        prepared_reference_full_system_work_certified=False,
        reference_change_full_system_work_certified=False,
        new_mixed_force_equilibrium_certified=False,
        physical_material_admission_authorized=False,
        whole_clock_all_pair_admission_authorized=False,
        full_system_temporal_path_work_certified=False,
        clocks_advanced=False, proposed_source_damage_adopted=False,
        source_geometry_elapsed_time_s=0, contact_geometry_elapsed_time_s=2000000,
        memory_elapsed_time_s=1000000, material_geometry_elapsed_time_s=1000000,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, original_point_1_completed=False,
        remaining_physical_conditions=list(REMAINING_CONDITIONS),
        historical_force_solve_reexecuted=False, previous_tensor_quadrature_reexecuted=False)


def _material_point_queries(bound, evaluated, api):
    """True original polygon-vertex labels at roof, interior and actual base.

    The authentic positive correction has its whole containing-chart range
    proof in the operator. A chart sample is never called correction material.
    """
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_joint_spherical_field import rotate_joint_physical_depth
    from tectonics.geometric_native_spherical_solid import interval_cbrt
    rows = []
    for body, encoded_response in zip(bound.bodies, evaluated.body_responses_json):
        bank = json.loads(body.complete_common_bank_json)
        mask = json.loads(bank["complete_mask_source_json"])
        polygon = mask["proxy_geometries"][0]
        ray = api._normalized_native_vertex_bounds(polygon, OPTIONS["precision"])[0]
        cert = json.loads(encoded_response)["finite_mean_root_certificate"]
        h = api._interval(cert["scalar_affine_amplitude_bounds_m"])
        index = bound.bodies.index(body)
        b = evaluated.coordinate_increments[6*index+1:6*index+6]
        T = api._deviator(b)
        Fmat = tuple(tuple(I(int(i == j))*(1+h/body.radius_m)+T[i][j]/body.radius_m
                         for j in range(3)) for i in range(3))
        eig = api._interval(cert["whole_bracket_deformation_eigenvalue_bounds"])
        J = I(body.fixed_owner_scale**3*eig.lo**3,
              body.fixed_owner_scale**3*eig.hi**3)
        for label, depth in (("original_roof", F(0)),
                             ("original_interior", body.areal_material_thickness_m/2),
                             ("original_true_base", body.areal_material_thickness_m)):
            radial = interval_cbrt(I(body.radius_m**3-3*body.radius_m**2*depth), OPTIONS["precision"])
            direction = rotate_joint_physical_depth(ray,
                body.angular_rate_per_physical_volume_depth_m, I(depth),
                precision=OPTIONS["precision"], order=32)
            prepared = tuple(body.fixed_owner_scale*radial*x for x in direction)
            current = api._mv(Fmat, prepared)
            rows.append(dict(component_id=body.component_id, label=label,
                original_polygon_source_sha256=digest(polygon), original_vertex_index=0,
                original_unit_ray_bounds=api._payload(ray), material_depth_m=api._payload(depth),
                original_radius_bounds_m=api._payload(radial),
                prepared_position_bounds_m=api._payload(prepared),
                deformed_position_bounds_m=api._payload(current),
                deformed_original_material_jacobian_bounds=api._payload(J),
                source_density_pushforward_bounds_kg_m3=api._payload(I(body.density_kg_m3)/J),
                actual_source_polygon_label=True, containing_chart_surrogate_used=False,
                physical_all_pair_clearance_certified=False))
    return rows


def run(output_dir):
    from tectonics import geometric_native_common_hot_mean as api
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bound = system()
    dto = api.native_common_hot_mean_to_dict(bound)
    authorities_before = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                          for p, raw, seal in AUTHORITIES}
    packet = dict(format="native-common-hot-finite-mean-packet-1", **summary(),
        source_authority_raw_sha256={p: raw for p, raw, seal in AUTHORITIES},
        source_authority_packet_sha256={p: seal for p, raw, seal in AUTHORITIES},
        hot_mean_system=dto, evaluation_queries=[],
        genuine_hot30_source_is_exactly_zero=True,
        saved_cartesian_probes_promoted_to_accepted_sources=False,
        conditional_old19_diagnostics_promoted_to_accepted_sources=False,
        diagnostic_coordinate_declaration=("NEW q=literal saved Cartesian trace/3; five "
            "trace-free coordinates derived from saved xx,yy,zz,xy,xz,yz. These q values "
            "are chosen diagnostic targets, not recovered old physical base means."))
    for label, coordinates in query_coordinates():
        evaluated = response(coordinates)
        if (len(evaluated.gradient_bounds_n) != 30 or len(evaluated.hessian_bounds_n_m) != 30
                or any(len(row) != 30 for row in evaluated.hessian_bounds_n_m)
                or len(evaluated.body_responses_json) != 5
                or not evaluated.complete_hot30_operator_certified
                or not evaluated.finite_actual_base_mean_chart_certified
                or evaluated.source_scalar_foundation_added_again
                or evaluated.accepted_new_native_intervals
                or evaluated.physical_material_admission_authorized
                or evaluated.finite_mean_equilibrium_certified):
            raise ValueError("The actual complete HOT30 component changed its physical scope")
        for body, raw_response in zip(bound.bodies, evaluated.body_responses_json):
            row = json.loads(raw_response)
            cert = row["finite_mean_root_certificate"]
            if (not cert["whole_mask_unique_root_certified"]
                    or not cert["true_area_normalization"]
                    or cert["positive_source_area_floor_used"]
                    or cert["literal_root_replaced_by_midpoint"]
                    or not cert["individual_complete_body_positive_jacobian_certified"]
                    or not cert["individual_complete_body_global_injectivity_certified"]
                    or not row["chart_force_curvature_term_included"]
                    or not row["full_same_field_basal_variance_retained"]
                    or len(cert["complete_mask_cells"]) != len(body.mask_cells)):
                raise ValueError("Every HOT response requires its authentic whole-mask mean and full twojet")
        packet["evaluation_queries"].append(dict(label=label,
            arguments=dict(coordinate_increments=api._payload(coordinates), **OPTIONS),
            response=api.native_common_hot_mean_response_to_dict(evaluated)))
    transport, work, cold, common, normal, curved = source_packets()
    restored = api.native_common_hot_mean_from_dict(dto, transport, work, cold,
        common_hot_packet=common, normal_support_packet=normal, curved19_packet=curved)
    if encoded(api.native_common_hot_mean_to_dict(restored)) != encoded(dto):
        raise ValueError("The complete HOT material/reference/birth/clock/work restart changed")
    for row, (label, coordinates) in zip(packet["evaluation_queries"], query_coordinates()):
        restarted = api.native_common_hot_mean_response_from_dict(row["response"], system=restored,
            expected_coordinate_increments=coordinates, **OPTIONS)
        if encoded(api.native_common_hot_mean_response_to_dict(restarted)) != encoded(row["response"]):
            raise ValueError("The full HOT30 operator and finite mean restart changed")
    packet["material_point_queries"] = _material_point_queries(bound, response(diagnostic_coordinates()), api)
    packet["complete_declared_hot_mean_restart_verified"] = True
    authorities_after = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p, raw, seal in AUTHORITIES}
    if authorities_before != authorities_after or authorities_after != packet["source_authority_raw_sha256"]:
        raise ValueError("An old full authority changed during the new HOT controls")
    packet["all_full_source_authorities_unchanged_before_after"] = True
    packet["packet_sha256"] = digest(packet)
    (output / "source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-hot-finite-mean-control-1", status="PASS", **summary(),
        complete_declared_hot_mean_restart_verified=True,
        complete_hot_coordinate_count=30, original_hot_material_count=5,
        complete_operator_query_count=len(packet["evaluation_queries"]),
        independent_material_label_query_count=len(packet["material_point_queries"]),
        positive_paired_curved_correction_count=1,
        genuine_hot30_source_is_exactly_zero=True,
        source_packet_sha256=packet["packet_sha256"],
        source_packet_raw_sha256=hashlib.sha256((output / "source_packet.json").read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started)
    report["report_sha256"] = digest(report)
    (output / "report.json").write_bytes(encoded(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report

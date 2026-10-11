"""Whole cold endpoint geometry, restored from complete authentic authorities.

No historical force solve or tensor quadrature is replayed.  The whole-volume
theorem concerns the declared smooth cold extension and fixed radial guards;
it does not adopt physical time, damage or a new mixed force equilibrium.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_SCOPE = "DECLARED_GLOBAL_CONTINUOUS_COLD_MATERIAL_CARRIER_ENDPOINT_ONLY"
PACKET_FORMAT = "native-common-cold-material-carrier-packet-1"
PROVENANCE = "Authentic56 COMMON cold modes; smooth original-quarter partition and fixed envelope radial support guards"
AUTHORITIES = (
    ("analysis/geometric_native_common_material_surface_transport_validation/release_validation_02/native_control/source_packet.json",
     "f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64",
     "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"),
    ("analysis/geometric_native_common_carrier_source_work_validation/release_validation_02/native_control/source_packet.json",
     "b45d5d4d80c74c509bfac033c4a60c5b96c8327704e0c727fcf680406d82e57e",
     "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"),
)
ARCHIVE_AUTHORITY = (
    "analysis/geometric_native_archive_material_carrier_validation/release_validation_02/native_control/source_packet.json",
    "6d34f8acd0a5ba66221117277e7acd496689f5858e26320d06fd705918e0c8a7",
    "6833539d33c2b29a7ab1d884d4536727849bbdd58d8c819f46fdf90ecf6c7063",
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
        raise ValueError("The complete immutable source changed: " + path)
    packet = json.loads(raw)
    if (packet["packet_sha256"] != seal or
            digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != seal):
        raise ValueError("The complete source authority selfseal changed: " + path)
    return packet


@lru_cache(maxsize=1)
def source_packets():
    """Complete879 and891 authorities, in the new binder's argument order."""
    return tuple(_read_authority(authority) for authority in AUTHORITIES)


@lru_cache(maxsize=1)
def archive_packet():
    return _read_authority(ARCHIVE_AUTHORITY)


@lru_cache(maxsize=1)
def archive_carrier():
    from analysis.geometric_native_common_carrier_source_work_validation import native_control as saved
    # This public DTO restoration validates complete source geometry/material.
    # saved.system(), response(), run() and previous force tests are not called.
    carrier = saved.archive_carrier()
    if carrier.model_sha256 != source_packets()[1]["carrier_source_work_system"]["archive_carrier_model_sha256"]:
        raise ValueError("The authentic archive carrier model changed")
    archive_packet()
    return carrier


@lru_cache(maxsize=1)
def system():
    from tectonics.geometric_native_common_cold_material_carrier import bind_native_common_cold_material_carrier
    return bind_native_common_cold_material_carrier(*source_packets(), archive_carrier(),
        provenance=PROVENANCE, precision=96)


def summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        whole_cold_material_support_geometry_certified=True,
        physical_source_force_prestress_work_handover_certified=False,
        hot_constitutive_law_defined=False, finite_mean_chart_defined=False,
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
        historical_force_solve_reexecuted=False, previous_tensor_quadrature_reexecuted=False,
        original_cartesian_roof_chart_derivative_retained=True,
        whole_original_quarter_seams_smoothly_glued=True,
        fixed_radial_support_guard_defined=True,
        reference_change_full_system_work_certified=False)


def control_queries():
    """Actual source labels, full parent depths, both seams and inactive poles.

    The samples verify implementations.  The whole-support conclusion comes
    from the uniform theorem, not from promoting these samples to a volume.
    """
    from tectonics.geometric_native_pose import FractionInterval as I
    rows = []
    transport = source_packets()[0]
    for index, query in enumerate(transport["original_material_queries"]):
        r = query["response"]
        if r["material_depth_bounds_m"] == {"lo": [0, 1], "hi": [0, 1]}:
            rows.append(("authentic_source_" + str(index), r["original_parent_id"],
                         tuple(F(*x) for x in r["original_direction_literal"]), I(0), None))
    if len(rows) != 32:
        raise ValueError("The full original32 material source labels are required")
    for bank in system().banks:
        n = bank.original_quarter_inward
        y = sum((x[1] for x in (bank.original_quarter_start, bank.original_quarter_end)), F(0))
        z = sum((x[2] for x in (bank.original_quarter_start, bank.original_quarter_end)), F(0))
        ray = (n[0], y, z)
        for label, depth in (("top", I(0)), ("middle", I(bank.areal_material_thickness_m/2)),
                             ("bottom", I(bank.areal_material_thickness_m)),
                             ("depth_box", I(2000, 2001))):
            rows.append((bank.original_parent_id + "_" + label, bank.original_parent_id, ray, depth, None))
    seams = (("receiver_y_seam", ("octant:1", "octant:3"), (F(-1), F(0), F(100))),
             ("donor_y_seam", ("octant:5", "octant:7"), (F(1), F(0), F(100))),
             ("receiver_z_seam", ("octant:2", "octant:3"), (F(-1), F(100), F(0))),
             ("donor_z_seam", ("octant:6", "octant:7"), (F(1), F(100), F(0))))
    for label, parents, ray in seams:
        for parent in parents:
            rows.append((label + "_" + parent, parent, ray, I(5000), None))
    for label, parent, ray in (
            ("receiver_y_blend_positive", "octant:3", (F(-1), F(1), F(10000))),
            ("receiver_y_blend_negative", "octant:1", (F(-1), F(-1), F(10000))),
            ("donor_z_blend_positive", "octant:7", (F(1), F(10000), F(1))),
            ("donor_z_blend_negative", "octant:6", (F(1), F(10000), F(-1))),
            ("receiver_inactive_pole", "octant:3", (F(-1), F(0), F(0))),
            ("donor_inactive_pole", "octant:7", (F(1), F(0), F(0)))):
        rows.append((label, parent, ray, I(5000), None))
    # The reference guard is fixed across the entire coordinate box; it is
    # never adjusted from a queried displacement or interval midpoint.
    rows.append(("zero_cold_coordinates", "octant:3", (F(-1), F(1), F(10000)), I(5000), (F(0),)*56))
    rows.append(("reversed_cold_coordinates", "octant:3", (F(-1), F(1), F(10000)), I(5000),
                 tuple(-x for x in system().bound_coordinate_increments)))
    return tuple(rows)


def run(output_dir):
    from tectonics import geometric_native_common_cold_material_carrier as api
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bound = system()
    c = bound.whole_domain_certificate
    if not (c.original_parent_count == 8 and c.nonempty_archive_cohort_count == 3
            and c.within_owner_parent_corner_gluing_certified
            and c.whole_owner_material_injectivity_certified
            and c.whole_cold_cross_owner_support_separation_certified
            and c.complete_live_and_archive_material_retained
            and c.cold_owner_radial_gap_lower_bound_m > 0):
        raise ValueError("The actual uniform whole cold certificate is incomplete")
    system_dto = api.native_common_cold_material_carrier_to_dict(bound)
    packet = dict(format=PACKET_FORMAT, **summary(),
        source_authority_raw_sha256={p: raw for p, raw, seal in AUTHORITIES+(ARCHIVE_AUTHORITY,)},
        source_authority_packet_sha256={p: seal for p, raw, seal in AUTHORITIES+(ARCHIVE_AUTHORITY,)},
        cold_material_carrier_system=system_dto,
        archive_carrier_system=archive_packet()["carrier_system"],
        evaluation_queries=[])
    allocations = set()
    cohorts = set()
    for label, parent, ray, depth, coordinates in control_queries():
        coefficients = bound.bound_coordinate_increments if coordinates is None else coordinates
        point = api.evaluate_native_common_cold_material_point(bound, parent, ray, depth,
            coordinate_increments=coefficients, precision=96)
        if (point.volume_jacobian_bounds.lo <= 0 or len(point.mode_displacement_columns) != 56
                or len(point.mode_original_gradient_columns_per_m) != 56
                or point.physical_material_admission_authorized or point.accepted_new_native_intervals):
            raise ValueError("A whole cold local query changes physical scope or loses its full derivative")
        if not sum((w.lo for p, w in point.seam_weights), F(0)) <= 1 <= sum((w.hi for p, w in point.seam_weights), F(0)):
            raise ValueError("The actual seamless owner partition does not enclose unity")
        allocations.add(point.allocation)
        if point.allocation == "archived":
            cohorts.add(point.cohort_id)
        packet["evaluation_queries"].append(dict(label=label, arguments=dict(
            original_parent_id=parent, original_ray=api._payload(ray),
            material_depth_bounds_m=api._payload(depth), time_fraction=[1, 1],
            coordinate_increments=api._payload(coefficients), precision=96),
            response=api.native_common_cold_material_point_to_dict(point)))
    restored = api.native_common_cold_material_carrier_from_dict(system_dto,
        material_transport_packet=source_packets()[0], source_work_packet=source_packets()[1],
        archive_carrier=archive_carrier())
    if encoded(api.native_common_cold_material_carrier_to_dict(restored)) != encoded(system_dto):
        raise ValueError("The complete declared cold system restart changed")
    row = next(r for r in packet["evaluation_queries"] if r["label"] == "receiver_y_blend_positive")
    a = row["arguments"]
    from tectonics.geometric_native_pose import FractionInterval as I
    restarted = api.native_common_cold_material_point_from_dict(row["response"], system=restored,
        expected_original_parent_id=a["original_parent_id"],
        expected_original_ray=tuple(F(*x) for x in a["original_ray"]),
        expected_material_depth_bounds_m=I(F(*a["material_depth_bounds_m"]["lo"]), F(*a["material_depth_bounds_m"]["hi"])),
        expected_coordinate_increments=tuple(F(*x) for x in a["coordinate_increments"]), precision=96)
    if encoded(api.native_common_cold_material_point_to_dict(restarted)) != encoded(row["response"]):
        raise ValueError("The full56-column seam point joint restart changed")
    packet["complete_declared_cold_material_carrier_restart_verified"] = True
    packet["complete_cold_query_count"] = len(packet["evaluation_queries"])
    packet["actual_material_allocation_names"] = sorted(allocations)
    packet["packet_sha256"] = digest(packet)
    (output / "source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-cold-material-carrier-control-1", status="PASS", **summary(),
        complete_declared_cold_material_carrier_restart_verified=True,
        complete_cold_query_count=len(packet["evaluation_queries"]), complete_original_material_label_count=32,
        whole_original_parent_count=8, whole_nonempty_archive_cohort_count=3,
        complete_cold_coordinate_count=56, actual_material_allocation_names=sorted(allocations),
        sampled_archived_cohort_count=len(cohorts),
        whole_support_proof_is_uniform_not_sampled=True,
        whole_domain_certificate=api._payload(c),
        source_packet_sha256=packet["packet_sha256"],
        source_packet_raw_sha256=hashlib.sha256((output / "source_packet.json").read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started)
    report["report_sha256"] = digest(report)
    (output / "report.json").write_bytes(encoded(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report

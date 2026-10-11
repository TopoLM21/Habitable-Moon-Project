"""Full cold continuum controls on the authentic prepared895 material chart.

Saved full authorities are authenticated once. Historical force solves and
tensor quadrature controls are not replayed. This is a declared endpoint law,
not recovered archive prestress, physical time admission or mixed equilibrium.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_SCOPE = "DECLARED_GLOBAL_COLD_CONTINUUM_ENERGY_ENDPOINT_ONLY"
PROVENANCE = "Full original live/archive cold continuum on fixed895 prepared reference; coupled eight-tip constitutive projection with literal beam ledgers once"
QUADRATURE_OPTIONS = dict(precision=96, taylor_order=0, beta_cell_count=8,
    depth_cell_count=2, theta_cell_count=4, integration_method="whole_interval",
    max_cells=4096)
AUTHORITIES = (
    ("analysis/geometric_native_common_material_surface_transport_validation/release_validation_02/native_control/source_packet.json",
     "f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64",
     "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"),
    ("analysis/geometric_native_common_carrier_source_work_validation/release_validation_02/native_control/source_packet.json",
     "b45d5d4d80c74c509bfac033c4a60c5b96c8327704e0c727fcf680406d82e57e",
     "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"),
    ("analysis/geometric_native_common_cold_material_carrier_validation/release_validation_02/native_control/source_packet.json",
     "fbafd092a4eb98472e60f79665e1e0fe8e9e1c2c2b9d3679344a32e59593cc89",
     "27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70"),
)
MATERIAL_AUTHORITY = (
    "analysis/geometric_native_cold_body_validation/release_validation_01/native_control/source_packet.json",
    "86806099d9d07467a0a022d57ed8dcad0834487f222a048223486438d2f30472",
    "8d1744eec08c28b3c3e15593688b812c4aca50677322c3d10db6caf381920a13",
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
    """Complete879/891/895 packets, in the continuum binder's argument order."""
    return tuple(_read_authority(authority) for authority in AUTHORITIES)


@lru_cache(maxsize=1)
def material_packet():
    return _read_authority(MATERIAL_AUTHORITY)


@lru_cache(maxsize=1)
def archive_carrier():
    from analysis.geometric_native_common_carrier_source_work_validation import native_control as saved
    return saved.archive_carrier()


@lru_cache(maxsize=1)
def cold_carrier():
    from tectonics.geometric_native_common_cold_material_carrier import native_common_cold_material_carrier_from_dict
    transport, work, carrier = source_packets()
    return native_common_cold_material_carrier_from_dict(carrier["cold_material_carrier_system"],
        material_transport_packet=transport, source_work_packet=work,
        archive_carrier=archive_carrier())


@lru_cache(maxsize=1)
def system():
    from tectonics.geometric_native_common_cold_continuum_energy import bind_native_common_cold_continuum_energy
    return bind_native_common_cold_continuum_energy(*source_packets(),
        cold_carrier=cold_carrier(),
        cold_tensor_packet=material_packet(), provenance=PROVENANCE, precision=96)


def summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        whole_cold_material_support_geometry_certified=True,
        physical_source_force_prestress_work_handover_certified=False,
        historical_archived_three_dimensional_prestress_recovered=False,
        prepared_reference_full_system_work_certified=False,
        reference_change_full_system_work_certified=False,
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
        historical_force_solve_reexecuted=False, previous_tensor_quadrature_reexecuted=False)


@lru_cache(maxsize=3)
def response(coordinates=None):
    from tectonics.geometric_native_common_cold_continuum_energy import evaluate_native_common_cold_continuum_energy
    return evaluate_native_common_cold_continuum_energy(system(), coordinates,
        **QUADRATURE_OPTIONS)


def _strain_point_queries(bound, api):
    """Authentic label controls for an independent Cartesian derivative audit."""
    from tectonics.geometric_native_pose import FractionInterval as I
    original = {bank.original_parent_id: bank for bank in cold_carrier().banks}
    rows = []
    for bank in bound.banks:
        old = original[bank.original_parent_id]
        def ray(normal, start, end):
            return tuple(normal*n+start*a+end*b for n, a, b in zip(
                old.original_quarter_inward, old.original_quarter_start,
                old.original_quarter_end))
        upper = bank.elastic_layer_areal_depth_m/2
        lower = (bank.elastic_layer_areal_depth_m+bank.areal_material_thickness_m)/2
        queries = [("interior_upper", ray(F(1), F(100), F(100)), upper),
                   ("seam_lower", ray(F(1), F(100), F(0)), lower),
                   ("inactive_upper", ray(F(100), F(1), F(1)), upper)]
        if bank.owner == cold_carrier().whole_domain_certificate.donor.owner:
            queries.append(("activation_collar_lower", ray(F(1, 100000000), F(1), F(1)), lower))
        for label, original_ray, depth in queries:
            point = api.bound_native_common_cold_continuum_point(bound,
                bank.original_parent_id, original_ray, I(depth), precision=96)
            rows.append(dict(label=bank.original_parent_id+":"+label,
                arguments=dict(original_parent_id=bank.original_parent_id,
                    original_ray=api._payload(original_ray),
                    material_depth_bounds_m=api._payload(I(depth)), precision=96),
                strain_point=api.native_common_cold_continuum_strain_point_to_dict(point)))
    return rows


def run(output_dir):
    from tectonics import geometric_native_common_cold_continuum_energy as api
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bound = system()
    system_dto = api.native_common_cold_continuum_energy_to_dict(bound)
    packet = dict(format="native-common-cold-continuum-energy-packet-1", **summary(),
        source_authority_raw_sha256={p: raw for p, raw, seal in AUTHORITIES+(MATERIAL_AUTHORITY,)},
        source_authority_packet_sha256={p: seal for p, raw, seal in AUTHORITIES+(MATERIAL_AUTHORITY,)},
        cold_continuum_energy_system=system_dto, evaluation_queries=[])
    for label, coordinates in (("registered_cold_endpoint", bound.bound_coordinate_increments),
                               ("prepared_zero_cold_reference", (F(0),)*56)):
        evaluated = response(None if coordinates == bound.bound_coordinate_increments else coordinates)
        if (len(evaluated.gradient_n) != 56 or len(evaluated.hessian_n_m) != 56
                or any(len(row) != 56 for row in evaluated.hessian_n_m)
                or not evaluated.gram_inverse_certificate.positive_definite_certified
                or evaluated.accepted_new_native_intervals):
            raise ValueError("The actual56-coordinate cold continuum is incomplete or changes physical scope")
        certificate = evaluated.integrals.quadrature_certificate
        if (not certificate.whole_domain_enclosure_certified
                or not certificate.authentic_original_volume_included_once
                or not certificate.whole_interval_box_range_certified
                or not certificate.directed_ieee_basic_arithmetic_certified
                or certificate.gauss_difference_used_as_error_certificate
                or tuple(certificate.complete_original_parent_ids) != tuple(
                    bank.original_parent_id for bank in bound.banks)
                or len(certificate.nonempty_archive_domain_ids) != 3):
            raise ValueError("The actual operator requires the complete certified original material domain")
        packet["evaluation_queries"].append(dict(label=label, arguments=dict(
            coordinate_increments=api._payload(coordinates), **QUADRATURE_OPTIONS),
            response=api.native_common_cold_continuum_energy_response_to_dict(evaluated)))
    restored = api.native_common_cold_continuum_energy_from_dict(system_dto,
        material_transport_packet=source_packets()[0], source_work_packet=source_packets()[1],
        material_carrier_packet=source_packets()[2], cold_carrier=cold_carrier(),
        cold_tensor_packet=material_packet())
    if encoded(api.native_common_cold_continuum_energy_to_dict(restored)) != encoded(system_dto):
        raise ValueError("The full material/reference/beam/clock continuum restart changed")
    row = packet["evaluation_queries"][0]
    restarted = api.native_common_cold_continuum_energy_response_from_dict(row["response"], system=restored,
        expected_coordinate_increments=bound.bound_coordinate_increments,
        **QUADRATURE_OPTIONS)
    if encoded(api.native_common_cold_continuum_energy_response_to_dict(restarted)) != encoded(row["response"]):
        raise ValueError("The joint56-coordinate continuum operator restart changed")
    packet["strain_point_queries"] = _strain_point_queries(bound, api)
    packet["complete_cold_continuum_operator_certified"] = True
    packet["complete_declared_cold_continuum_restart_verified"] = True
    packet["packet_sha256"] = digest(packet)
    (output / "source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-cold-continuum-energy-control-1", status="PASS", **summary(),
        complete_cold_continuum_operator_certified=True,
        complete_declared_cold_continuum_restart_verified=True,
        complete_cold_coordinate_count=56, source_beam_count=8,
        complete_operator_query_count=len(packet["evaluation_queries"]),
        independent_cartesian_label_query_count=len(packet["strain_point_queries"]),
        full_source_material_parent_count=8, full_nonempty_archive_cohort_count=3,
        source_packet_sha256=packet["packet_sha256"],
        source_packet_raw_sha256=hashlib.sha256((output / "source_packet.json").read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started)
    report["report_sha256"] = digest(report)
    (output / "report.json").write_bytes(encoded(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report

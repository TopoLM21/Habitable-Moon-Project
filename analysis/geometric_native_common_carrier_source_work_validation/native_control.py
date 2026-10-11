"""Fresh registered COMMON handover and genuine cohesive preparation controls.

Saved source DTOs are authenticated once. Historical force solves, tests and
tensor quadrature are not run. The paired placement and proposed fracture
history remain a candidate until whole-system admission and equilibrium.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_SCOPE = "DECLARED_ARCHIVE_CARRIER_REGISTERED_COMMON_COHESIVE_PREPARATION_ONLY"
PACKET_FORMAT = "native-common-carrier-source-work-packet-1"
PROVENANCE = "Registered original COMMON fields attached to declared paired archive carrier; authentic cohesive series preparation"
AUTHORITIES = (
    ("analysis/geometric_native_common_local_source_work_validation/release_validation_02/native_control/source_packet.json",
     "791c01ce8b0b78562be8df3b55f56dde5afcc5b26c95de227eaff79b53bffa33",
     "faca2daf3a40dc39e6ae7aaca7fcd91c887756fb50eed3c37b09b8183d920469"),
    ("analysis/geometric_native_common_material_surface_transport_validation/release_validation_02/native_control/source_packet.json",
     "f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64",
     "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"),
    ("analysis/geometric_native_common_source_carrier_validation/release_validation_02/native_control/source_packet.json",
     "265206430b4c864db32bce1be0a9a75b71fa180158f3d81e4f52131bebdf320c",
     "a46fe79aeabef70ec01672ad4c33c737bfc06d5323a112edcfeb2616f7b148a3"),
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
        raise ValueError("An immutable full source authority changed: " + path)
    value = json.loads(raw)
    if (value["packet_sha256"] != seal or
            digest({k: v for k, v in value.items() if k != "packet_sha256"}) != seal):
        raise ValueError("A full source authority selfseal changed: " + path)
    return value


@lru_cache(maxsize=1)
def source_packets():
    """The complete sealed883/879/867 authorities, in binder argument order."""
    return tuple(_read_authority(authority) for authority in AUTHORITIES)


@lru_cache(maxsize=1)
def archive_packet():
    return _read_authority(ARCHIVE_AUTHORITY)


@lru_cache(maxsize=1)
def archive_carrier():
    from analysis.geometric_native_archive_material_carrier_validation import native_control as saved
    from tectonics.geometric_native_archive_material_carrier import native_archive_material_carrier_from_dict
    packet = archive_packet()
    if encoded(packet["source_material_state"]) != encoded(saved.source_documents()["material_state.json"]):
        raise ValueError("The paired carrier lost its complete original material authority")
    # Authentic public material/geometry restoration; no historical force or
    # old tensor/contact quadrature control is executed.
    return native_archive_material_carrier_from_dict(packet["carrier_system"],
        expected_source_material_state=saved.source_state())


@lru_cache(maxsize=1)
def system():
    from tectonics.geometric_native_common_carrier_source_work import bind_native_common_carrier_source_work
    return bind_native_common_carrier_source_work(*source_packets(), archive_carrier(),
        provenance=PROVENANCE, precision=96)


def summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        registered_common_cold_coordinates_attached_to_source_labels=True,
        physical_source_force_prestress_work_handover_certified=False,
        source_cohesive_fracture_history_proposed=True,
        original_normal_fracture_history_available=False,
        new_virgin_normal_fracture_seed_declared=True,
        proposed_source_damage_adopted=False,
        whole_support_nonzero_common_cold_admission_certified=False,
        current_deformed_follower_normal_contact_law_defined=False,
        hot_constitutive_law_defined=False, finite_mean_chart_defined=False,
        new_mixed_force_equilibrium_certified=False,
        whole_clock_all_pair_admission_authorized=False,
        physical_material_admission_authorized=False,
        full_system_temporal_path_work_certified=False,
        historical_force_solve_reexecuted=False,
        previous_tensor_quadrature_reexecuted=False,
        source_geometry_elapsed_time_s=0,
        contact_geometry_elapsed_time_s=2000000,
        memory_elapsed_time_s=1000000,
        material_geometry_elapsed_time_s=1000000,
        clocks_advanced=False, accepted_original_native_intervals=1,
        accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, original_point_1_completed=False,
        remaining_physical_conditions=list(REMAINING_CONDITIONS))


@lru_cache(maxsize=8)
def response(preparation_fraction):
    from tectonics.geometric_native_common_carrier_source_work import evaluate_native_common_carrier_source_work
    return evaluate_native_common_carrier_source_work(system(), time_fraction=F(1),
        preparation_fraction=preparation_fraction, precision=96)


@lru_cache(maxsize=1)
def preparation_probes():
    """Choose actual strict branch boxes from all16 enclosed affine drivers.

    These are sampling points on a declared preparation parameter, not
    estimates of material-entry times or a physical time admission.
    """
    start = response(F(0))
    upper_elastic = min((a.before_loading_response.onset_free_gap_m-a.before_free_gap_bounds_m.hi)
        /a.preparation_normal_driver_bounds_m.hi for a in start.source_atom_responses)
    lower_soft = max((a.before_loading_response.onset_free_gap_m-a.before_free_gap_bounds_m.lo)
        /a.preparation_normal_driver_bounds_m.lo for a in start.source_atom_responses)
    upper_soft = min((a.before_loading_response.failure_free_gap_m-a.before_free_gap_bounds_m.hi)
        /a.preparation_normal_driver_bounds_m.hi for a in start.source_atom_responses)
    if not (0 < upper_elastic <= lower_soft < upper_soft < 1):
        raise ValueError("No joint strict elastic/softening probe exists for the authentic16 atoms")
    return (("original_registered_reference", F(0)),
            ("intact_preparation", upper_elastic/2),
            ("softening_preparation", (lower_soft+upper_soft)/2),
            ("declared_paired_endpoint", F(1)))


def _zeros(value):
    from tectonics.geometric_native_pose import FractionInterval as I
    if type(value) is I:
        if not value.lo <= 0 <= value.hi:
            raise ValueError("A physical work, moment or material-chart residual misses zero")
    else:
        for item in value:
            _zeros(item)


def run(output_dir):
    from tectonics import geometric_native_common_carrier_source_work as api
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bound = system()
    system_dto = api.native_common_carrier_source_work_system_to_dict(bound)
    packet = dict(format=PACKET_FORMAT, **summary(),
        source_authority_raw_sha256={p: raw for p, raw, seal in AUTHORITIES+(ARCHIVE_AUTHORITY,)},
        source_authority_packet_sha256={p: seal for p, raw, seal in AUTHORITIES+(ARCHIVE_AUTHORITY,)},
        carrier_source_work_system=system_dto,
        archive_carrier_system=archive_packet()["carrier_system"],
        evaluation_queries=[])
    branches = set()
    atoms_checked = 0
    for label, lam in preparation_probes():
        evaluated = response(lam)
        if (len(evaluated.source_atom_responses) != 16 or len(evaluated.source_beam_responses) != 8 or
                len(evaluated.potential_gradient_bounds) != 100 or len(evaluated.potential_hessian_bounds) != 100):
            raise ValueError("The genuine16+8+8 source assembly or100-coordinate jet is incomplete")
        sides = [side for atom in evaluated.source_atom_responses for side in (atom.sub, atom.over)]
        if len({(s.point_id, s.side) for s in sides}) != 32:
            raise ValueError("The complete32 original material labels must attach once")
        for atom in evaluated.source_atom_responses:
            for residual in (atom.action_reaction_residual_bounds_n, atom.world_moment_residual_bounds_nm,
                    atom.virtual_work_residual_bounds, atom.preparation_energy_balance_residual_bounds_j):
                _zeros(residual)
            _zeros(atom.sub.frame_pullback_identity_residual_bounds)
            _zeros(atom.over.frame_pullback_identity_residual_bounds)
            if atom.preparation_normal_driver_bounds_m.lo <= 0:
                raise ValueError("The genuine normal preparation is not monotone over its whole affine path")
            branches.add(atom.loading_response.branch)
            atoms_checked += 1
        _zeros(evaluated.preparation_energy_balance_residual_bounds_j)
        if evaluated.source_state_or_work_advanced or evaluated.accepted_new_native_intervals:
            raise ValueError("A preparation component must not silently adopt damage or physical time")
        packet["evaluation_queries"].append(dict(label=label,
            arguments=dict(time_fraction=[1, 1], preparation_fraction=[lam.numerator, lam.denominator],
                           coordinate_increments=api._payload(bound.bound_coordinate_increments), precision=96),
            response=api.native_common_carrier_source_work_response_to_dict(evaluated)))
    if branches != {"intact_or_compressed", "monotone_softening", "fully_failed_open"}:
        raise ValueError("The actual control must exercise intact, softening and fully failed branches")
    endpoint = response(F(1))
    if any(a.loading_response.branch != "fully_failed_open" for a in endpoint.source_atom_responses):
        raise ValueError("The paired endpoint has unresolved normal cohesive contacts")
    packet["complete_declared_preparation_endpoint_atom_count"] = 16
    packet["actual_branch_names"] = sorted(branches)
    packet["complete_source_query_count"] = len(packet["evaluation_queries"])
    packet["source_atom_query_count"] = atoms_checked
    restored = api.native_common_carrier_source_work_system_from_dict(system_dto,
        local_source_packet=source_packets()[0], material_transport_packet=source_packets()[1],
        source_carrier_packet=source_packets()[2], archive_carrier=archive_carrier())
    if encoded(api.native_common_carrier_source_work_system_to_dict(restored)) != encoded(system_dto):
        raise ValueError("The complete system preparation restart changed")
    row = packet["evaluation_queries"][2]
    restarted = api.native_common_carrier_source_work_response_from_dict(row["response"], system=restored,
        expected_time_fraction=F(1), expected_preparation_fraction=F(*row["arguments"]["preparation_fraction"]),
        expected_coordinate_increments=restored.bound_coordinate_increments, precision=96)
    if encoded(api.native_common_carrier_source_work_response_to_dict(restarted)) != encoded(row["response"]):
        raise ValueError("The joint100-coordinate cohesive preparation restart changed")
    packet["complete_declared_preparation_restart_verified"] = True
    packet["packet_sha256"] = digest(packet)
    (output/"source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-carrier-source-work-control-1", status="PASS", **summary(),
        complete_source_query_count=len(packet["evaluation_queries"]), source_atom_query_count=atoms_checked,
        complete_original_material_label_count=32, original_archived_label_count=8,
        original_source_atom_count=16, original_source_beam_count=8, registered_bulk_remainder_count=8,
        complete_coordinate_count=100, cold_coordinate_count=56, hot_coordinate_count=30,
        actual_branch_names=sorted(branches), source_packet_sha256=packet["packet_sha256"],
        source_packet_raw_sha256=hashlib.sha256((output/"source_packet.json").read_bytes()).hexdigest(),
        complete_declared_preparation_restart_verified=True, elapsed_seconds=time.monotonic()-started)
    report["report_sha256"] = digest(report)
    (output/"report.json").write_bytes(encoded(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report

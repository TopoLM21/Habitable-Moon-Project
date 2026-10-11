"""Original spring memory and work remain tied to the genuine refused force.

The shared case reconstructs the original checkpoint and actual force, then
replays the saved whole first-entry material state.  Exact-time observations
are compared with an independent Decimal return map; no new material point
receives a nodal state or a fraction of the spring's work.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from tectonics.geometric_native_first_entry_material import (
    native_first_entry_material_from_dict,
)
from tectonics.geometric_native_mantle_loading import native_spatial_mantle_load_from_dict
from tectonics.geometric_native_physical_history import (
    _candidate_payload,
    build_native_physical_history_binding,
    native_original_pair_history_from_dict,
    native_original_pair_history_to_dict,
    native_physical_history_binding_from_dict,
    native_physical_history_binding_to_dict,
    query_native_original_pair_history,
    validate_native_physical_history_binding,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_underthrust_motion import (
    UnresolvedNativeUnderthrustMotion, advance_native_underthrust_motion,
)
from tectonics.geometric_native_underthrust_motion_io import load_native_underthrust_motion_checkpoint
from tectonics.mesh import build_icosphere


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_first_entry_validation/control_release_02"
F = Fraction


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


@lru_cache(maxsize=1)
def get_physical_case():
    """Shared original source case; no mocked force or invented admission law."""
    paths = tuple(CONTROL/name for name in ("material_state.json", "point_certificates.json",
        "original_force/initial.json", "original_force/after_refusal.json",
        "original_force/native_source.json", "original_force/force_candidate.json"))
    original_files = {path: path.read_bytes() for path in paths}
    mesh = build_icosphere(0)
    origin, before, _ = load_native_underthrust_motion_checkpoint(
        CONTROL/"original_force/initial.json", mesh=mesh)
    current_load = native_spatial_mantle_load_from_dict(json.loads(
        original_files[CONTROL/"original_force/native_source.json"]), before.current_surface, mesh=mesh)
    saved_candidate = json.loads(original_files[CONTROL/"original_force/force_candidate.json"])
    try:
        advance_native_underthrust_motion(mesh, origin, before, current_load,
            saved_candidate["dt_s"], provenance=saved_candidate["provenance"])
    except UnresolvedNativeUnderthrustMotion as refusal:
        candidate = refusal.candidate
        assert encoded(_candidate_payload(candidate, refusal.overlaps)) == encoded(saved_candidate)
    else:
        raise AssertionError("The unchanged original physical candidate must remain refused")
    material_state = native_first_entry_material_from_dict(json.loads(
        original_files[CONTROL/"material_state.json"]))
    checkpoint_sha256 = hashlib.sha256(original_files[CONTROL/"original_force/initial.json"]).hexdigest()
    binding = build_native_physical_history_binding(mesh, origin, before, candidate,
        material_state, current_load=current_load, original_checkpoint_sha256=checkpoint_sha256)
    point_certificates = json.loads(original_files[CONTROL/"point_certificates.json"])
    return dict(mesh=mesh, origin=origin, before=before, candidate=candidate,
        current_load=current_load, material_state=material_state,
        checkpoint_sha256=checkpoint_sha256, binding=binding, history=binding,
        original_files=original_files, point_certificates=point_certificates)


@pytest.fixture(scope="module")
def actual_history():
    return get_physical_case()


def decimals(value):
    return Decimal(value.numerator)/Decimal(value.denominator)


def sine_cosine(angle):
    sine, cosine = angle, Decimal(1)
    sinterm, costerm = angle, Decimal(1)
    for index in range(1, 300):
        sinterm *= -angle*angle/Decimal((2*index)*(2*index+1))
        costerm *= -angle*angle/Decimal((2*index-1)*(2*index))
        sine += sinterm
        cosine += costerm
        if max(abs(sinterm), abs(costerm)) < Decimal("1e-560"):
            return sine, cosine
    raise AssertionError("Independent material-coordinate sine/cosine failed")


def field(polynomial, angle):
    sine, cosine = sine_cosine(angle)
    basis = (Decimal(1), cosine, sine, 2*cosine*cosine-1, 2*sine*cosine)
    return sum((decimals(F.from_float(coefficient))*value
                for coefficient, value in zip(polynomial.coefficients, basis)), Decimal(0))


def independent_dense_pair(binding, node, time):
    with localcontext() as context:
        context.prec = 600
        theta, fraction = decimals(node.computational_angle), decimals(time)
        old = next(row for row in binding.previous_free_anchor_fields if row[0] == node.trace_id)
        new = next(row for row in binding.unaccepted_candidate_free_anchor_fields if row[0] == node.trace_id)
        gap0, jump0, gap1, jump1 = (field(polynomial, theta)
            for polynomial in (*old[1:], *new[1:]))
        gap, jump = gap0+fraction*(gap1-gap0), jump0+fraction*(jump1-jump0)
        memory = next(row for row in binding.previous_memory.traces if row.trace_id == node.trace_id)
        piece = next(piece for piece in memory.pieces
            if F.from_float(piece.low_radians) <= node.computational_angle <= F.from_float(piece.high_radians))
        old_plastic, old_cumulative = field(piece.plastic_slip, theta), field(piece.cumulative_slip, theta)
        trace = node.original_material
        kn = decimals(F.from_float(1/(1/trace.columns.normal_stiffness_pa_m
            + 1/trace.law.normal_stiffness_pa_m)))
        kt = decimals(F.from_float(1/(1/trace.columns.shear_stiffness_pa_m
            + 1/trace.law.tangential_stiffness_pa_m)))
        pressure = -kn*gap
        mu = decimals(F.from_float(trace.law.friction_dry+
            (trace.law.friction_wet-trace.law.friction_dry)*trace.water))
        cohesion = decimals(F.from_float(trace.law.cohesion_pa*
            (1-(1-trace.law.wet_cohesion_fraction)*trace.water)))
        strength, trial = cohesion+mu*pressure, kt*(jump-old_plastic)
        correction = max(Decimal(0), trial-strength)-max(Decimal(0), -trial-strength)
        increment = correction/kt
        plastic = old_plastic+increment
        return dict(free_gap_bounds_m=gap, free_jump_bounds_m=jump,
            previous_plastic_slip_bounds_m=old_plastic,
            plastic_increment_bounds_m=increment, plastic_slip_bounds_m=plastic,
            cumulative_slip_bounds_m=old_cumulative+abs(increment),
            normal_pressure_bounds_pa=pressure, shear_traction_bounds_pa=kt*(jump-plastic))


def contained(bounds, value):
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-F(value)) < F(1, 10**500)*max(1, abs(bounds.lo))
    else:
        assert bounds.lo <= F(value) <= bounds.hi


def test_original_quadrature_measures_preload_and_all_work_ledgers_survive(actual_history):
    binding, before, candidate = (actual_history[key] for key in ("binding", "before", "candidate"))
    nodes = binding.quadrature_histories
    assert len(nodes) == 16
    assert tuple(node.point_id for node in nodes) == tuple(identity for identity, _ in before.states)
    assert binding.initial_mechanics == before.initial_mechanics
    assert binding.previous_memory == before.spatial_memory
    assert binding.unaccepted_candidate_memory == candidate.spatial_step.memory
    assert tuple((identity, gap.coefficients, jump.coefficients)
        for identity, gap, jump in binding.previous_free_anchor_fields) == before.free_anchor_fields
    assert binding.unaccepted_candidate_free_anchor_fields == candidate.spatial_step.anchor_fields
    for node in nodes:
        assert node.prepared_state == node.previous_state == dict(before.initial_mechanics.states)[node.point_id]
        assert node.unaccepted_candidate_state == dict(candidate.states)[node.point_id]
        assert node.original_area_m2 == node.previous_state.area_m2 == node.unaccepted_candidate_state.area_m2
        assert node.previous_state.gap_free_m == -.1
        assert node.previous_state.traction_normal_pa < 0
        assert node.previous_state.contact_energy_j > 0
        assert node.previous_state.cumulative_external_work_j > 0
        assert node.unaccepted_candidate_state.plastic_slip_m < 0
    assert dict(binding.before_work_ledgers)["preparation_work_j"] > 0
    assert dict(binding.before_work_ledgers)["cumulative_source_work_j"] == 0
    assert dict(binding.unaccepted_candidate_work_ledgers)["source_work_j"] > 8e16


@pytest.mark.parametrize("time", [F(0), F(2, 7), F(1), F(1, 10**400)])
def test_singleton_constitutive_history_matches_independent_original_field_return_map(actual_history, time):
    binding = actual_history["binding"]
    for node in binding.quadrature_histories:
        query = query_native_original_pair_history(binding, node.point_id, I(time))
        independent = independent_dense_pair(binding, node, time)
        for name, expected in independent.items():
            contained(getattr(query, name), expected)
        assert query.original_quadrature_history == node
        assert query.time_fraction_bounds == I(time)
        assert query.elapsed_time_s_bounds == I(F.from_float(node.previous_state.elapsed_time_s)
            + F.from_float(binding.literal_dt_s)*time)
        assert query.native_elapsed_time_s_bounds == I(F.from_float(binding.accepted_before.elapsed_time_s)
            + F.from_float(binding.literal_dt_s)*time)
        assert query.physical_time_myr_bounds == I(F.from_float(float(binding.origin_surface.time_myr))
            + F.from_float(binding.literal_dt_myr)*time)
        assert query.tip_maps.reference_coordinates == tuple(F.from_float(x) for x in node.original_point.position)
        assert not query.incoming_pair_handover_authorized
        assert not query.candidate_mechanics_adopted
        assert not query.nodal_work_allocated_to_material_volume


def test_exact_tiny_time_advances_both_saved_clocks_without_adopting_candidate_work(actual_history):
    binding = actual_history["binding"]
    time = F(1, 10**400)
    query = query_native_original_pair_history(binding, binding.quadrature_histories[0].point_id, I(time))
    assert query.elapsed_time_s_bounds.lo > F.from_float(query.original_quadrature_history.previous_state.elapsed_time_s)
    assert query.physical_time_myr_bounds.lo > 0
    assert query.plastic_increment_bounds_m.hi < 0
    assert float(query.physical_time_myr_bounds.lo) == 0
    assert binding.accepted_before.cumulative_source_work_j == 0
    assert binding.accepted_before.receipts == ()


def test_finite_SO3_geometry_does_not_replace_integrated_mechanical_anchors(actual_history):
    binding = actual_history["binding"]
    different = False
    for node in binding.quadrature_histories:
        query = query_native_original_pair_history(binding, node.point_id, I(F(1)))
        different |= (query.tip_maps.gap_m.hi < query.free_gap_bounds_m.lo
            or query.free_gap_bounds_m.hi < query.tip_maps.gap_m.lo)
        assert not query.tip_maps.finite_pose_change_replaces_free_anchors
        assert tuple((identity, gap.coefficients, jump.coefficients)
            for identity, gap, jump in binding.contact_time_path.previous_anchor_fields) == binding.accepted_before.free_anchor_fields
    assert different, "Prepared preload and finite tip separation must remain separate physical quantities"


def test_history_observations_do_not_change_any_original_surface_archive_clock_memory_or_work(actual_history):
    binding = actual_history["binding"]
    before = native_physical_history_binding_to_dict(binding)
    for node in binding.quadrature_histories:
        query_native_original_pair_history(binding, node.point_id, I(F(1, 3), F(2, 3)))
    assert native_physical_history_binding_to_dict(binding) == before
    assert binding.accepted_before.current_inventory.records == ()
    assert binding.accepted_before.elapsed_time_s == 0
    assert all(path.read_bytes() == original for path, original in actual_history["original_files"].items())


@pytest.mark.parametrize("point_id", ["new-incoming-bulk-material", "octant:6", 0, True])
def test_incoming_or_aliased_point_cannot_receive_an_original_nodal_history(actual_history, point_id):
    with pytest.raises(ValueError):
        query_native_original_pair_history(actual_history["binding"], point_id, I(F(1, 2)))


def test_original_history_query_roundtrip_keeps_saved_endpoint_states_and_nonadoption(actual_history):
    binding = actual_history["binding"]
    query = query_native_original_pair_history(binding, binding.quadrature_histories[5].point_id,
        I(F(1, 3), F(2, 3)))
    raw = native_original_pair_history_to_dict(query, binding=binding)
    assert native_original_pair_history_from_dict(deepcopy(raw), binding=binding) == query
    assert raw["intermediate_work_computed"] is False
    assert raw["nodal_work_allocated_to_material_volume"] is False
    assert raw["original_quadrature_history"]["previous_state"]["contact_energy_j"] > 0


@pytest.mark.parametrize("field", ["binding_sha256", "point_id", "plastic_slip_bounds_m",
    "normal_pressure_bounds_pa", "candidate_mechanics_adopted",
    "nodal_work_allocated_to_material_volume", "intermediate_work_computed"])
def test_resealed_query_cannot_forge_identity_memory_pressure_or_work(actual_history, field):
    binding = actual_history["binding"]
    query = query_native_original_pair_history(binding, binding.quadrature_histories[0].point_id, I(F(1, 2)))
    raw = native_original_pair_history_to_dict(query, binding=binding)
    if field == "binding_sha256":
        raw[field] = "f"*64
    elif field == "point_id":
        raw[field] = binding.quadrature_histories[1].point_id
    elif field.endswith("bounds_m") or field.endswith("bounds_pa"):
        raw[field] = dict(lo=[0, 1], hi=[0, 1])
    else:
        raw[field] = True
    raw["query_sha256"] = hashlib.sha256(encoded({key: value for key, value in raw.items()
        if key != "query_sha256"})).hexdigest()
    with pytest.raises(ValueError):
        native_original_pair_history_from_dict(raw, binding=binding)


@pytest.mark.parametrize("changes", [dict(time_fraction_bounds=I(F(-1, 10))),
    dict(time_fraction_bounds=I(F(11, 10))), dict(precision=True), dict(precision=8)])
def test_query_rejects_invalid_clock_and_precision_aliases(actual_history, changes):
    parameters = dict(time_fraction_bounds=I(F(1, 2)), precision=128)
    parameters.update(changes)
    with pytest.raises(ValueError):
        query_native_original_pair_history(actual_history["binding"],
            actual_history["binding"].quadrature_histories[0].point_id, **parameters)


@pytest.mark.parametrize("field", ["literal_dt_s", "literal_dt_myr", "before_work_ledgers",
    "unaccepted_candidate_work_ledgers", "source_before_payload_json", "quadrature_histories"])
def test_typed_history_cache_cannot_authorize_a_changed_clock_work_or_node(field, actual_history):
    binding = actual_history["binding"]
    if field.startswith("literal_dt"):
        value = getattr(binding, field)*2
    elif field.endswith("work_ledgers"):
        rows = getattr(binding, field)
        value = ((rows[0][0], rows[0][1]*2 if rows[0][1] else 1.), *rows[1:])
    elif field == "source_before_payload_json":
        value = binding.source_before_payload_json+b" "
    else:
        nodes = binding.quadrature_histories
        changed = replace(nodes[0], previous_state=replace(nodes[0].previous_state,
            cumulative_external_work_j=0.))
        value = (changed, *nodes[1:])
    with pytest.raises(ValueError):
        validate_native_physical_history_binding(replace(binding, **{field: value}))


def test_complete_history_dto_replays_actual_external_force_sources(actual_history):
    case = actual_history
    raw = native_physical_history_binding_to_dict(case["binding"])
    repeated = native_physical_history_binding_from_dict(deepcopy(raw), case["mesh"],
        case["origin"], case["before"], case["candidate"], case["material_state"],
        current_load=case["current_load"], original_checkpoint_sha256=case["checkpoint_sha256"])
    assert repeated == case["binding"]
    assert repeated.original_pair_history_authenticated
    assert not repeated.incoming_pair_handover_authorized
    assert not repeated.candidate_mechanics_adopted


def test_complete_force_replay_rejects_a_relabelled_owner_velocity_candidate(actual_history):
    case = actual_history
    candidate = case["candidate"]
    omega = list(candidate.omega_rad_per_myr)
    first = omega[0]
    omega[0] = (first[0], (first[1][0]*2 if first[1][0] else 1e-18, *first[1][1:]))
    forged = replace(candidate, omega_rad_per_myr=tuple(omega))
    with pytest.raises(ValueError, match="genuine force replay"):
        build_native_physical_history_binding(case["mesh"], case["origin"], case["before"],
            forged, case["material_state"], current_load=case["current_load"],
            original_checkpoint_sha256=case["checkpoint_sha256"])


@pytest.mark.parametrize("alias", ["tuple_rational", "numpy_precision", "integer_boolean"])
def test_query_dto_requires_literal_json_types_and_rejects_equivalent_aliases(actual_history, alias):
    binding = actual_history["binding"]
    query = query_native_original_pair_history(binding, binding.quadrature_histories[0].point_id, I(F(1, 2)))
    raw = native_original_pair_history_to_dict(query, binding=binding)
    if alias == "tuple_rational":
        raw["time_fraction_bounds"]["lo"] = tuple(raw["time_fraction_bounds"]["lo"])
    elif alias == "numpy_precision":
        raw["precision"] = np.int64(raw["precision"])
    else:
        raw["original_pair_history_authenticated"] = 1
    with pytest.raises(ValueError):
        native_original_pair_history_from_dict(raw, binding=binding)


def test_actual_engine_scalar_identity_cannot_be_replaced_by_a_python_float_alias(actual_history):
    binding = actual_history["binding"]
    candidate = binding.actual_candidate
    diagnostics = list(candidate.candidate_diagnostics)
    index = next(index for index, (_, value) in enumerate(diagnostics) if isinstance(value, np.generic))
    name, value = diagnostics[index]
    diagnostics[index] = (name, float(value))
    aliased = replace(candidate, candidate_diagnostics=tuple(diagnostics))
    with pytest.raises(ValueError):
        validate_native_physical_history_binding(replace(binding, actual_candidate=aliased))

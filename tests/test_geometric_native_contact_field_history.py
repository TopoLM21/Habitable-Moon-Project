"""Continuous source memory and elastic energy without nodal interpolation."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.geometric_native_contact_field_history import (
    native_contact_field_history_from_dict,
    native_contact_field_history_to_dict,
    query_native_contact_field_history,
    validate_native_contact_field_history,
)
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_physical_history import get_physical_case


F = Fraction


@pytest.fixture(scope="module")
def field_case():
    case = get_physical_case()
    return dict(case=case, history=case["binding"],
        source=native_physical_history_binding_to_dict(case["binding"]))


def decimal(value):
    return Decimal(value.numerator)/Decimal(value.denominator)


def sine_cosine(angle):
    sine, cosine = angle, Decimal(1)
    sinterm, costerm = angle, Decimal(1)
    for index in range(1, 1000):
        sinterm *= -angle*angle/Decimal((2*index)*(2*index+1))
        costerm *= -angle*angle/Decimal((2*index-1)*(2*index))
        sine += sinterm
        cosine += costerm
        if max(abs(sinterm), abs(costerm)) < Decimal("1e-560"):
            return sine, cosine
    raise AssertionError("Independent source-field trigonometry failed")


def polynomial(value, theta):
    sine, cosine = sine_cosine(theta)
    basis = (Decimal(1), cosine, sine, cosine*cosine-sine*sine, 2*sine*cosine)
    return sum((decimal(F.from_float(coefficient))*item
        for coefficient, item in zip(value.coefficients, basis)), Decimal(0))


def independent_field(history, trace_id, theta, time, *, piece_index=None):
    """Evaluate one retained constitutive expression directly at 600 digits."""
    with localcontext() as context:
        context.prec = 600
        angle, fraction = decimal(theta), decimal(time)
        path = history.contact_time_path
        index = next(index for index, trace in enumerate(path.initial_mechanics.fault_traces.traces)
            if trace.trace_id == trace_id)
        trace = path.initial_mechanics.fault_traces.traces[index]
        previous = path.step.previous_memory.traces[index]
        if piece_index is None:
            piece_index = next(index for index, piece in enumerate(previous.pieces)
                if F.from_float(piece.low_radians) <= theta <= F.from_float(piece.high_radians))
        piece = previous.pieces[piece_index]
        _, gap0, jump0 = path.previous_anchor_fields[index]
        _, gap1, jump1 = path.step.anchor_fields[index]
        g0, j0, g1, j1 = (polynomial(value, angle) for value in (gap0, jump0, gap1, jump1))
        gap, jump = g0+fraction*(g1-g0), j0+fraction*(j1-j0)
        previous_slip = polynomial(piece.plastic_slip, angle)
        previous_cumulative = polynomial(piece.cumulative_slip, angle)
        kncol, ktcol = (decimal(F.from_float(value)) for value in
            (trace.columns.normal_stiffness_pa_m, trace.columns.shear_stiffness_pa_m))
        kncontact, ktcontact = (decimal(F.from_float(value)) for value in
            (trace.law.normal_stiffness_pa_m, trace.law.tangential_stiffness_pa_m))
        kn, kt = (decimal(F.from_float(1/(1/column+1/contact))) for column, contact in
            ((trace.columns.normal_stiffness_pa_m, trace.law.normal_stiffness_pa_m),
             (trace.columns.shear_stiffness_pa_m, trace.law.tangential_stiffness_pa_m)))
        pressure = -kn*gap
        mu = decimal(F.from_float(trace.law.friction_dry+
            (trace.law.friction_wet-trace.law.friction_dry)*trace.water))
        cohesion = decimal(F.from_float(trace.law.cohesion_pa*
            (1-(1-trace.law.wet_cohesion_fraction)*trace.water)))
        strength = cohesion+mu*pressure
        trial = kt*(jump-previous_slip)
        increment = (max(Decimal(0), trial-strength)-max(Decimal(0), -trial-strength))/kt
        slip, cumulative = previous_slip+increment, previous_cumulative+abs(increment)
        # Use the authenticated down branch's exact traction identity after
        # independently checking its return-map sign. This avoids a Decimal
        # cancellation residue in the actual zero-friction source law.
        assert trial <= -strength
        shear = -strength
        contact_gap = -pressure/kncontact
        elastic_shear = shear/ktcontact
        contact_jump = slip+elastic_shear
        bulk_normal, bulk_shear = contact_gap-gap, contact_jump-jump
        bulk_energy = (kncol*bulk_normal*bulk_normal+ktcol*bulk_shear*bulk_shear)/2
        contact_energy = (kncontact*contact_gap*contact_gap+ktcontact*elastic_shear*elastic_shear)/2
        return dict(free_gap_bounds_m=gap, free_jump_bounds_m=jump,
            previous_plastic_slip_bounds_m=previous_slip,
            previous_cumulative_slip_bounds_m=previous_cumulative,
            plastic_increment_bounds_m=increment, plastic_slip_bounds_m=slip,
            cumulative_slip_bounds_m=cumulative, normal_pressure_bounds_pa=pressure,
            shear_traction_bounds_pa=shear, contact_gap_bounds_m=contact_gap,
            contact_jump_bounds_m=contact_jump, bulk_normal_strain_bounds_m=bulk_normal,
            bulk_shear_strain_bounds_m=bulk_shear,
            contact_elastic_shear_strain_bounds_m=elastic_shear,
            bulk_energy_density_bounds_j_m2=bulk_energy,
            contact_energy_density_bounds_j_m2=contact_energy,
            total_energy_density_bounds_j_m2=bulk_energy+contact_energy)


def enclosed(bounds, value):
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-F(value)) <= F(1, 10**540)*max(1, abs(bounds.lo))
    else:
        assert bounds.lo <= F(value) <= bounds.hi


def query(case, trace_id, angle, time, *, context=None, precision=256):
    return query_native_contact_field_history(case["history"], trace_id, angle, time,
        expected_source_history_dto=case["source"], source_expression_context=context,
        precision=precision)


@pytest.mark.parametrize("theta,time", [(F(0), F(0)), (F(2, 9), F(1, 7)),
    (F(5, 8), F(3, 8)), (F(7, 5), F(1)),
    (F.from_float(1.5707963267948966)-F(1, 2**900), F(4, 11)),
    (F(1, 10**400), F(1, 10**400))])
def test_arbitrary_material_coordinates_match_independent_return_map_and_energy(field_case, theta, time):
    history = field_case["history"]
    for trace in history.initial_mechanics.fault_traces.traces:
        value = query(field_case, trace.trace_id, I(theta), I(time))
        for name, expected in independent_field(history, trace.trace_id, theta, time).items():
            enclosed(getattr(value, name), expected)
        assert value.computational_angle_bounds == I(theta)
        assert value.time_fraction_bounds == I(time)
        assert value.finite_point_area_m2 is value.finite_point_work_j is None
        assert value.cumulative_work_density_j_m2 is None


def test_entire_material_clock_rectangle_contains_all_sampled_original_fields(field_case):
    history = field_case["history"]
    angle, time = I(F(1, 9), F(7, 5)), I(F(1, 10**400), F(1))
    context = {"kind": "implicit_root_expression", "root": "external_geometry_authority",
        "time_window": {"lo": [1, 10**400], "hi": [1, 1]}}
    for trace in history.initial_mechanics.fault_traces.traces:
        value = query(field_case, trace.trace_id, angle, time, context=context)
        assert value.source_expression_context == context
        for theta in (angle.lo, F(4, 7), angle.hi):
            for fraction in (time.lo, F(3, 8), time.hi):
                for name, expected in independent_field(history, trace.trace_id, theta, fraction).items():
                    enclosed(getattr(value, name), expected)
        assert value.source_previous_memory is history.previous_memory
        assert value.source_unaccepted_candidate_memory is history.unaccepted_candidate_memory
        assert value.source_contact_time_path is history.contact_time_path


def test_all_closed_piece_intersections_survive_without_selecting_a_piece_midpoint(field_case):
    history = field_case["history"]
    for before, candidate in zip(history.previous_memory.traces, history.unaccepted_candidate_memory.traces):
        extent = F.from_float(before.extent_radians)
        value = query(field_case, before.trace_id, I(F(0), extent), I(F(0), F(1)))
        assert tuple(row.original_piece for row in value.previous_memory_intersections) == before.pieces
        assert tuple(row.original_piece for row in value.unaccepted_candidate_memory_intersections) == candidate.pieces
        for piece_index, piece in enumerate(before.pieces):
            for endpoint in (F.from_float(piece.low_radians), F.from_float(piece.high_radians)):
                boundary = query(field_case, before.trace_id, I(endpoint), I(F(2, 7)))
                expected_indices = tuple(index for index, old in enumerate(before.pieces)
                    if F.from_float(old.low_radians) <= endpoint <= F.from_float(old.high_radians))
                assert tuple(row.piece_index for row in boundary.previous_memory_intersections) == expected_indices
                for name, expected in independent_field(history, before.trace_id, endpoint,
                        F(2, 7), piece_index=piece_index).items():
                    enclosed(getattr(boundary, name), expected)


def test_exact_zero_shear_and_energy_floor_survive_sub_float_angle_and_clock(field_case):
    history = field_case["history"]
    tiny = F(1, 10**400)
    trace = history.initial_mechanics.fault_traces.traces[0]
    value = query(field_case, trace.trace_id, I(tiny), I(tiny))
    assert value.shear_traction_bounds_pa == I(F(0))
    assert value.contact_elastic_shear_strain_bounds_m == I(F(0))
    assert value.bulk_energy_density_bounds_j_m2.lo > 0
    assert value.contact_energy_density_bounds_j_m2.lo > 0
    assert value.plastic_increment_bounds_m.hi < 0
    assert value.native_elapsed_time_s_bounds.lo > 0
    assert value.physical_time_myr_bounds.lo > 0
    assert float(value.physical_time_myr_bounds.lo) == 0
    assert history.accepted_before.elapsed_time_s == 0
    assert history.accepted_before.cumulative_source_work_j == 0
    assert history.accepted_before.receipts == ()


def test_saved_nodal_endpoint_ledgers_are_preserved_as_records_without_becoming_density(field_case):
    history = field_case["history"]
    before = native_physical_history_binding_to_dict(history)
    for trace in history.initial_mechanics.fault_traces.traces:
        value = query(field_case, trace.trace_id, I(F(1, 2)), I(F(1)))
        raw = native_contact_field_history_to_dict(value, history=history,
            expected_source_history_dto=field_case["source"])
        assert raw["finite_point_area_m2"] is None
        assert raw["finite_point_work_j"] is None
        assert raw["cumulative_work_density_j_m2"] is None
        assert raw["source_history"] == before
        assert not raw["candidate_mechanics_adopted"]
        assert not raw["incoming_pair_handover_authorized"]
    assert native_physical_history_binding_to_dict(history) == before


def sample_query(case):
    trace = case["history"].initial_mechanics.fault_traces.traces[0]
    angle, time = I(F(1, 3), F(2, 3)), I(F(2, 7), F(3, 7))
    context = {"kind": "external_implicit_root", "original_root_id": "root:sample"}
    return query(case, trace.trace_id, angle, time, context=context), dict(history=case["history"],
        trace_id=trace.trace_id, computational_angle_bounds=angle, time_fraction_bounds=time,
        expected_source_history_dto=case["source"], source_expression_context=context, precision=256)


def test_full_field_roundtrip_requires_independent_trace_angle_clock_and_root_context(field_case):
    value, arguments = sample_query(field_case)
    raw = native_contact_field_history_to_dict(value, history=field_case["history"],
        expected_source_history_dto=field_case["source"])
    rebuilt = native_contact_field_history_from_dict(deepcopy(raw), **arguments)
    assert native_contact_field_history_to_dict(rebuilt, history=field_case["history"],
        expected_source_history_dto=field_case["source"]) == raw


def reseal(raw):
    raw["query_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in raw.items()
        if key != "query_sha256"}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return raw


@pytest.mark.parametrize("mutation", ["memory", "source_work", "source_area", "angle", "time", "root",
                                      "energy", "pieces", "finite_area", "finite_work"])
def test_rehashed_dto_cannot_forge_complete_source_or_implicit_expression(field_case, mutation):
    value, arguments = sample_query(field_case)
    raw = deepcopy(native_contact_field_history_to_dict(value, history=field_case["history"],
        expected_source_history_dto=field_case["source"]))
    if mutation in ("source_work", "source_area"):
        raw["source_history"]["actual_candidate"]["states"][0][1][
            "area_m2" if mutation == "source_area" else "cumulative_external_work_j"] += 1
    elif mutation == "memory":
        raw["previous_plastic_slip_bounds_m"] = {"lo": [1, 1], "hi": [1, 1]}
    elif mutation == "angle":
        raw["computational_angle_bounds"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "time":
        raw["time_fraction_bounds"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "root":
        raw["source_expression_context"]["original_root_id"] = "different-root"
    elif mutation == "energy":
        raw["total_energy_density_bounds_j_m2"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "pieces":
        raw["previous_memory_intersections"] = []
    else:
        raw["finite_point_area_m2" if mutation == "finite_area" else "finite_point_work_j"] = [1, 1]
    with pytest.raises(ValueError):
        native_contact_field_history_from_dict(reseal(raw), **arguments)


@pytest.mark.parametrize("mutation", ["bool_coefficient", "missing_full_source", "wrong_trace", "tuple_context"])
def test_source_and_argument_aliases_are_refused(field_case, mutation):
    trace = field_case["history"].initial_mechanics.fault_traces.traces[0]
    arguments = dict(expected_source_history_dto=deepcopy(field_case["source"]),
        source_expression_context=None, precision=128)
    trace_id = trace.trace_id
    if mutation == "bool_coefficient":
        # JSON boolean True must not authenticate a literal numeric coefficient.
        arguments["expected_source_history_dto"]["accepted_before"]["elapsed_time_s"] = False
    elif mutation == "missing_full_source":
        arguments["expected_source_history_dto"] = {"binding_sha256": field_case["history"].binding_sha256}
    elif mutation == "wrong_trace":
        trace_id = "octant:6"
    else:
        arguments["source_expression_context"] = {"root": ("tuple",)}
    with pytest.raises(ValueError):
        query_native_contact_field_history(field_case["history"], trace_id, I(F(1, 2)), I(F(1, 3)), **arguments)


def test_frozen_query_cannot_replace_a_derived_energy_or_drop_an_original_piece(field_case):
    value, _ = sample_query(field_case)
    for changed in (replace(value, total_energy_density_bounds_j_m2=I(F(0))),
                    replace(value, previous_memory_intersections=())):
        with pytest.raises(ValueError):
            validate_native_contact_field_history(changed, history=field_case["history"],
                expected_source_history_dto=field_case["source"])

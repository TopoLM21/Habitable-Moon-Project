"""Actual candidate handover and independent objective virtual-power controls."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL,
)
from tectonics.geometric_native_material_port_handover import (
    certify_native_candidate_material_port_handover,
    native_candidate_material_port_handover_from_dict,
    native_candidate_material_port_handover_to_dict,
)
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tests.test_geometric_native_physical_history import get_physical_case
from tests.test_geometric_native_physical_pair_time import decimal, rotate


F = Fraction


@pytest.fixture(scope="module")
def actual_handover():
    case = get_physical_case()
    history = case["binding"]
    source = native_physical_history_binding_to_dict(history)
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Explicit reduced-model receiver-attached complete-configuration law")
    handover = certify_native_candidate_material_port_handover(history, law=law,
        expected_source_history_dto=source, precision=128)
    return dict(case=case, history=history, source=source, law=law, handover=handover)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Decimal(0))


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def scale(value, vector):
    return tuple(value*x for x in vector)


def atan_small(value):
    result = term = value
    for index in range(1, 2000):
        term *= -value*value
        addition = term/Decimal(2*index+1)
        result += addition
        if abs(addition) < Decimal("1e-450"):
            return result
    raise AssertionError("Independent source-bank angle failed to converge")


def enclosed(bounds, value):
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-F(value)) < F(1, 10**420)*max(1, abs(bounds.lo))
    else:
        assert bounds.lo <= F(value) <= bounds.hi


def test_real_incoming_attachments_use_source_bank_interior_and_keep_loaded_history(actual_handover):
    history, value = actual_handover["history"], actual_handover["handover"]
    nodes = {node.point_id: node for node in history.quadrature_histories}
    assert len(value.ports) == len(value.retained_point_ids) == 8
    transferred_ids = {port.point_id for port in value.ports}
    assert transferred_ids.isdisjoint(value.retained_point_ids)
    assert transferred_ids | set(value.retained_point_ids) == set(nodes)
    assert all(port.target_reference_coordinates[0].lo > 0 for port in value.ports)
    radial = {row[0]: row[1:] for row in history.actual_candidate.transverse_displacement_m}
    for port in value.ports:
        node, local = nodes[port.point_id], port.transport_state.local_state
        assert local is node.unaccepted_candidate_state
        assert local.plastic_slip_m < 0
        assert local.cumulative_slip_m > 0
        assert local.contact_energy_j > 0
        assert local.cumulative_external_work_j > 0
        assert local.area_m2 == node.original_area_m2
        assert all(margin.lo >= 0 for margin in port.target_material_margins)
        assert 0 < port.target_chart.x_over_width.lo <= port.target_chart.x_over_width.hi < 1
        assert port.target_chart.shape.hi < 1
        own_tip = F.from_float(radial[port.trace_id][0])
        assert port.target_chart.tip_displacement_m.lo == own_tip
        assert not (port.target_chart.displacement_m.lo <= own_tip <= port.target_chart.displacement_m.hi)
    assert native_physical_history_binding_to_dict(history) == actual_handover["source"]
    assert history.accepted_before.elapsed_time_s == 0
    assert history.accepted_before.cumulative_source_work_j == 0
    assert history.accepted_before.receipts == ()
    assert not value.candidate_mechanics_adopted
    assert not value.physical_pair_admission_authorized


@pytest.mark.parametrize("component", ["normal", "down"])
def test_objective_operator_matches_independent_full_carrier_virtual_power(actual_handover, component):
    history, value = actual_handover["history"], actual_handover["handover"]
    nodes = {node.point_id: node for node in history.quadrature_histories}
    bindings = {binding.trace_id: binding for binding in history.material_state.trace_bindings}
    parameters = {row.trace_id: row for row in history.initial_mechanics.bending_parameters}
    radial = {row[0]: row[1:] for row in history.actual_candidate.transverse_displacement_m}
    virtual_sub = (F(2, 7), F(-3, 11), F(5, 13))
    virtual_over = (F(-7, 17), F(11, 19), F(-13, 23))
    radial_sub, radial_over = F(17, 29), F(-19, 31)
    with localcontext() as context:
        context.prec = 500
        radius = decimal(F.from_float(history.origin_surface.radius_km)*1000)
        axes = {owner: tuple(decimal(x)*decimal(history.dt_myr) for x in axis)
            for owner, axis in history.omega_records}
        vs, vo = tuple(map(decimal, virtual_sub)), tuple(map(decimal, virtual_over))
        for port in value.ports:
            node, binding = nodes[port.point_id], bindings[port.trace_id]
            q = tuple(decimal(F.from_float(x)) for x in node.original_point.position)
            e_ref = tuple(decimal(F.from_float(x)) for x in (
                node.original_point.normal if component == "normal" else node.original_point.down_dip))
            qs = rotate(rotate(q, axes[binding.overriding_plate]),
                tuple(-x for x in axes[binding.subducting_plate]))
            start = tuple(decimal(F.from_float(x)) for x in node.original_material.start)
            end = tuple(decimal(F.from_float(x)) for x in node.original_material.end)
            inward = tuple(-x for x in cross(start, end))
            transverse = dot(qs, inward)
            plane_norm = (dot(qs, start)**2+dot(qs, end)**2).sqrt()
            distance = radius*atan_small(transverse/plane_norm)
            width = decimal(F.from_float(parameters[port.trace_id].sub.width_m))
            coordinate = distance/width
            phi = 1-Decimal("1.5")*coordinate+coordinate**3/2
            ws, wo = (decimal(F.from_float(x)) for x in radial[port.trace_id])
            world = rotate(q, axes[binding.overriding_plate])
            frame = rotate(e_ref, axes[binding.overriding_plate])
            rs, ro = scale(radius+phi*ws, world), scale(radius+wo, world)
            separation = tuple(b-a for a, b in zip(rs, ro))
            sub_velocity = add(cross(vs, rs), scale(phi*decimal(radial_sub), world))
            over_velocity = add(cross(vo, ro), scale(decimal(radial_over), world))
            relative_velocity = tuple(b-a for a, b in zip(sub_velocity, over_velocity))
            ordinary_power = dot(frame, relative_velocity)
            carrier_power = dot(cross(vo, frame), separation)
            expected = ordinary_power+carrier_power
            row = port.normal_operator if component == "normal" else port.down_operator
            bounds = (sum((x*y for x, y in zip(row[0], virtual_sub)), F(0))
                + sum((x*y for x, y in zip(row[1], virtual_over)), F(0))
                + row[2]*radial_sub+row[3]*radial_over)
            enclosed(bounds, expected)
            enclosed(port.target_chart.shape, phi)
            assert abs(carrier_power) > Decimal("1e-10")
            # This deliberately excludes the carrier term and therefore fails
            # the complete objective power identity for these physical ports.
            assert not (bounds.lo <= F(ordinary_power) <= bounds.hi)


def test_full_candidate_handover_roundtrip_and_nonadoption_scope(actual_handover):
    value = actual_handover["handover"]
    raw = native_candidate_material_port_handover_to_dict(value)
    rebuilt = native_candidate_material_port_handover_from_dict(deepcopy(raw),
        expected_history=actual_handover["history"],
        expected_source_history_dto=actual_handover["source"],
        expected_law=actual_handover["law"])
    assert native_candidate_material_port_handover_to_dict(rebuilt) == raw
    assert raw["source_epoch"] == "unaccepted_candidate"
    assert raw["original_interface_measure_preserved"]
    assert raw["unchanged_global_beam_state"]
    assert not raw["transported_candidate_is_force_equilibrium"]
    assert not raw["native_full_bank_and_clamp_support_certified"]
    assert not raw["finite_measure_assigned_to_first_entry_point"]
    assert raw["accepted_loaded_native_intervals"] == raw["accepted_source_work_j"] == 0


@pytest.mark.parametrize("mutation", ["duplicate_port", "label", "operator", "context", "source_area", "source_work"])
def test_resealed_candidate_dto_cannot_forge_material_or_full_source(actual_handover, mutation):
    raw = deepcopy(native_candidate_material_port_handover_to_dict(actual_handover["handover"]))
    if mutation == "duplicate_port":
        raw["ports"].append(deepcopy(raw["ports"][0]))
    elif mutation == "label":
        raw["ports"][0]["target_reference_coordinates"][0] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "operator":
        raw["ports"][0]["normal_operator"]["receiver_frame_couple_included"] = False
    elif mutation == "context":
        raw["ports"][0]["target_chart"]["source"]["tip_displacement_m"] = {"lo": [1, 1], "hi": [1, 1]}
    else:
        raw["source_history"]["actual_candidate"]["states"][0][1][
            "area_m2" if mutation == "source_area" else "cumulative_external_work_j"] += 1
    raw["handover_sha256"] = hashlib.sha256(encoded({key: item for key, item in raw.items()
        if key != "handover_sha256"})).hexdigest()
    with pytest.raises(ValueError):
        native_candidate_material_port_handover_from_dict(raw,
            expected_history=actual_handover["history"], expected_source_history_dto=actual_handover["source"],
            expected_law=actual_handover["law"])


def test_full_source_and_literal_dto_are_required(actual_handover):
    with pytest.raises(ValueError):
        certify_native_candidate_material_port_handover(actual_handover["history"],
            law=actual_handover["law"], expected_source_history_dto={"binding_sha256": "0"*64})
    raw = native_candidate_material_port_handover_to_dict(actual_handover["handover"])
    raw["retained_point_ids"] = tuple(raw["retained_point_ids"])
    with pytest.raises(ValueError):
        native_candidate_material_port_handover_from_dict(raw,
            expected_history=actual_handover["history"], expected_source_history_dto=actual_handover["source"],
            expected_law=actual_handover["law"])

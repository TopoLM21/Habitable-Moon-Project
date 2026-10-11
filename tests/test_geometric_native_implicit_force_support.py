"""Temporal original banks stay complete and actual continuous clamps stay live."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json

import numpy as np
import pytest
from scipy.linalg import expm

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_native_implicit_force_support import (
    UnresolvedNativeImplicitForceSupport, _entry_depth_upper, _normal_majorant,
    certify_native_implicit_force_support, native_implicit_force_support_from_dict,
    native_implicit_force_support_to_dict, validate_native_implicit_force_support,
)
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_physical_history import get_physical_case


@pytest.fixture(scope="module")
def implicit_support_case():
    original = get_physical_case()
    history = original["binding"]
    source = native_physical_history_binding_to_dict(history)
    value = certify_native_implicit_force_support(history, expected_source_history_dto=source)
    return dict(original=original, history=history, source=source, value=value,
        raw=native_implicit_force_support_to_dict(value))


def test_support_retains_all_original_domains_and_four_symbolic_reservoirs(implicit_support_case):
    history, value = implicit_support_case["history"], implicit_support_case["value"]
    assert len(value.parent_supports) == 8
    assert {bank.original_fragment.fragment_id for bank in value.parent_supports} == {
        fragment.fragment_id for fragment in history.material_state.journal.source_surface.fragments}
    allocations = {item.fragment_id: item for item in history.material_state.parent_allocations}
    for bank in value.parent_supports:
        fragment, allocation = bank.original_fragment, bank.parent_allocation
        assert allocation is allocations[fragment.fragment_id]
        assert tuple(row.reservoir for row in allocation.reservoirs) == EXTENSIVE_FIELDS
        for row in allocation.reservoirs:
            assert row.parent_amount == F.from_float(getattr(fragment.parcel, row.reservoir))
            assert row.measure_reference == allocation.measure_reference
            assert row.live_expression == "original_parent_amount - archived_amount"
            assert row.conservation_identity == "live_amount + archived_amount = original_parent_amount"
        assert allocation.material_age_at_endpoint_myr == allocation.original_age_myr+history.dt_myr
    curved, = (bank for bank in value.parent_supports
        if bank.archive_case == "endpoint_plus_curved_swept_extra")
    assert curved.original_fragment.fragment_id == "octant:5"
    assert curved.parent_allocation.archived_area_bounds_sr.lo > 0
    for raw in implicit_support_case["raw"]["parent_supports"]:
        assert not raw["curved_archive_replaced_by_polygon"]
        assert raw["coverage_identity"] == "original_parent = live_original disjoint_union once_only_archive"
        if raw["archive_case"] == "unarchived_receiver":
            assert raw["archived_area_bounds_sr"] == dict(lo=[0, 1], hi=[0, 1])
            assert raw["archive_mask"].startswith("empty:")


def test_actual_full_axes_prove_every_continuous_original_clamp_live(implicit_support_case):
    history, value = implicit_support_case["history"], implicit_support_case["value"]
    assert value.omega_records == history.omega_records
    assert all(component != 0 for _, omega in value.omega_records for component in omega)
    expected = history.dt_myr*sum((abs(x) for _, omega in history.omega_records for x in omega), F(0))
    assert value.rotation_path_l1_bound == expected
    assert value.normal_component_error_bound == expected/(1-expected)
    assert len(value.live_clamps) == 8
    assert {(clamp.trace_id, clamp.bank) for clamp in value.live_clamps} == {
        (trace.trace_id, bank) for trace in history.initial_mechanics.fault_traces.traces
        for bank in ("sub", "over")}
    for clamp in value.live_clamps:
        assert clamp.width_m == clamp.exact_width_radians*value.radius_m
        assert clamp.original_live_parent_ids
        if clamp.bank == "sub":
            assert len(clamp.vertex_entry_depth_upper_bounds) == 4
            assert max(clamp.vertex_entry_depth_upper_bounds) < 0
            assert clamp.vertex_entry_depth_upper_bounds == tuple(
                _entry_depth_upper(ray.reference, value.normal_component_error_bound)
                for ray in clamp.enclosing_geometry.vertices)
            # Positive conic combinations prove the interior as well as edges.
            point = tuple(sum((ray.reference[i] for ray in clamp.enclosing_geometry.vertices), F(0))
                for i in range(3))
            assert _entry_depth_upper(point, value.normal_component_error_bound) < 0
        else:
            assert not clamp.vertex_entry_depth_upper_bounds
            assert "receiver owner" in clamp.proof


def test_support_is_bound_to_original_source_radius_time_and_full_physics(implicit_support_case):
    history, value = implicit_support_case["history"], implicit_support_case["value"]
    assert value.radius_m == F.from_float(history.origin_surface.radius_km)*1000
    assert value.time_myr == I(F.from_float(history.material_state.journal.source_surface.time_myr),
        F.from_float(history.material_state.journal.source_surface.time_myr)+history.dt_myr)
    assert value.material_state_sha256 == history.material_state.state_sha256
    assert value.original_checkpoint_sha256 == history.original_checkpoint_sha256
    assert value.trace_sha256 == history.initial_mechanics.fault_traces.trace_sha256
    assert value.bending_source_sha256 == history.initial_mechanics.bending_source_sha256
    assert json.loads(value.source_history_payload_json) == implicit_support_case["source"]
    assert value.complete_original_bank_support_certified
    assert value.live_original_clamp_support_certified
    assert not value.physical_pair_admission_authorized
    assert not value.candidate_mechanics_adopted
    assert value.native_force_intervals_accepted == 0
    original = implicit_support_case["original"]
    assert not original["before"].receipts
    assert not original["before"].current_inventory.records
    assert original["before"].elapsed_time_s == 0
    assert original["before"].cumulative_source_work_j == 0
    assert all(path.read_bytes() == data for path, data in original["original_files"].items())


def test_support_strict_restart_recomputes_from_complete_independent_source(implicit_support_case):
    case = implicit_support_case
    replay = native_implicit_force_support_from_dict(deepcopy(case["raw"]),
        expected_history=case["history"], expected_source_history_dto=deepcopy(case["source"]))
    assert native_implicit_force_support_to_dict(replay) == case["raw"]


def _reseal(raw):
    raw["support_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in raw.items()
        if key != "support_sha256"}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize("mutation", (
    "radius", "clock", "omega", "source", "width", "frame", "normal_bound",
    "clamp_depth", "clamp_parent", "parent_missing", "reservoir", "curved_mask",
    "archive_identity", "material_sha", "acceptance", "boolean_radius"))
def test_resealed_support_substitution_refuses(implicit_support_case, mutation):
    case = implicit_support_case
    raw = deepcopy(case["raw"])
    if mutation == "radius":
        raw["radius_m"][0] += 1
    elif mutation == "clock":
        raw["time_myr"]["hi"][0] += 1
    elif mutation == "omega":
        raw["omega_records"][0][1][0][0] += 1
    elif mutation == "source":
        raw["source_history"]["original_checkpoint_sha256"] = "0"*64
    elif mutation == "width":
        raw["live_clamps"][0]["width_m"][0] += 1
    elif mutation == "frame":
        raw["live_clamps"][0]["enclosing_geometry"]["supports"] = []
    elif mutation == "normal_bound":
        raw["normal_component_error_bound"] = [0, 1]
    elif mutation == "clamp_depth":
        raw["live_clamps"][0]["vertex_entry_depth_upper_bounds"][0] = [0, 1]
    elif mutation == "clamp_parent":
        raw["live_clamps"][0]["original_live_parent_ids"] = []
    elif mutation == "parent_missing":
        raw["parent_supports"].pop()
    elif mutation == "reservoir":
        raw["parent_supports"][0]["reservoirs"][0]["parent_amount"][0] += 1
    elif mutation == "curved_mask":
        raw["parent_supports"][5]["curved_archive_replaced_by_polygon"] = True
    elif mutation == "archive_identity":
        raw["parent_supports"][0]["live_domain_id"] = raw["parent_supports"][0]["archive_domain_id"]
    elif mutation == "material_sha":
        raw["material_state_sha256"] = "0"*64
    elif mutation == "acceptance":
        raw["physical_pair_admission_authorized"] = True
    elif mutation == "boolean_radius":
        raw["radius_m"][1] = True
    _reseal(raw)
    with pytest.raises(ValueError, match="independent full source"):
        native_implicit_force_support_from_dict(raw, expected_history=case["history"],
            expected_source_history_dto=case["source"])


@pytest.mark.parametrize("field,new_value", (
    ("radius_m", F(100001)), ("normal_component_error_bound", F(0)),
    ("time_myr", I(0, 1)), ("live_clamps", ()), ("parent_supports", ()),
    ("material_state_sha256", "0"*64)))
def test_typed_support_changes_cannot_reuse_a_supplied_seal(implicit_support_case, field, new_value):
    value = replace(implicit_support_case["value"], **{field: new_value})
    with pytest.raises(ValueError, match="full original material and clamp replay"):
        validate_native_implicit_force_support(value)


def test_exact_normal_majorant_contains_independent_two_axis_matrix_products():
    axes = ((3, (F(1, 50), F(-1, 100), F(1, 125))),
            (9, (F(-1, 80), F(1, 90), F(-1, 150))))
    rho, error = _normal_majorant(axes, F(1))
    assert rho == sum((abs(x) for _, omega in axes for x in omega), F(0))
    matrices = []
    for _, omega in axes:
        x, y, z = map(float, omega)
        matrices.append(np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]]))
    normal = np.array([-1., 0., 0.])
    for time in (0., .125, .5, .875, 1.):
        moved = expm(-time*matrices[0])@expm(time*matrices[1])@normal
        assert np.max(np.abs(moved-normal)) < float(error)


@pytest.mark.parametrize("axes,duration", (
    (((0, (F(0), F(0), F(0))), (1, (F(0), F(0), F(0)))), F(1)),
    (((0, (F(1, 4), F(0), F(0))), (1, (F(0), F(0), F(0)))), F(1))))
def test_zero_or_out_of_domain_relative_majorant_refuses(axes, duration):
    with pytest.raises(UnresolvedNativeImplicitForceSupport):
        _normal_majorant(axes, duration)


def test_depth_gate_retains_a_positive_unresolved_clamp_without_an_epsilon():
    assert _entry_depth_upper((F(1, 10**60), F(1), F(0)), F(1, 10**59)) > 0
    assert _entry_depth_upper((F(1), F(40), F(0)), F(1, 10**6)) < 0


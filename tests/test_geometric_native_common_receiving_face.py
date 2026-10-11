"""Authentic receiving-face coverage, work and immutable-source controls."""
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import copy
import json

import pytest

from tectonics import geometric_native_common_receiving_face as api
from analysis.geometric_native_common_receiving_face_validation.native_control import context

I = api.I


def contains(interval, exact):
    assert interval.lo <= exact <= interval.hi


def overlap(a, b):
    assert a.lo <= b.hi and b.lo <= a.hi


def cubes():
    reference = context()[0]
    return I(reference.radius_m**3-3*reference.radius_m**2*F(1000))


@lru_cache(maxsize=2)
def point(query_index=0):
    ctx = context()
    return api.evaluate_native_receiving_face(*ctx,
        ctx[4]['evaluation_queries'][query_index]['label'], I(F(1, 3)), cubes(),
        cold_coordinates=(F(0),)*56)


@lru_cache(maxsize=1)
def whole_face():
    ctx = context()
    return api.integrate_native_receiving_face(*ctx,
        ctx[4]['evaluation_queries'][0]['label'], cold_coordinates=(F(0),)*56)


def test_new_virgin_law_binds_actual_face_and_fixed_reference_measure():
    reference, cut, law, _, _ = context()
    assert law.cut_snapshot_sha256 == cut.cut_snapshot_sha256
    assert law.fixed_rest_gap_m == 0 and law.reference_clock_fraction == 1
    assert law.permanently_unbonded and law.frictionless
    assert not law.old_columns_charged_again
    assert not law.old_memory_or_minus_001_gap_inherited
    assert not law.preparation_or_activation_work_paid
    assert 'PHYSICAL_VOLUME_DEPTH' in law.measure_model
    for stiffness in (True, 10**7, F(0), F(-1)):
        with pytest.raises(ValueError):
            api.declare_native_receiving_face(reference, cut,
                normal_stiffness_pa_m=stiffness, provenance='explicit fixture')
    with pytest.raises(ValueError):
        api.declare_native_receiving_face(reference, cut, provenance=' ')


def test_normalized_chart_preserves_named_start_entire_root_band_and_depth():
    reference, cut, *_ = context()
    start, du = api._chart(reference, cut, I(0), cubes(), 96)
    end, _ = api._chart(reference, cut, I(1), cubes(), 96)
    full, _ = api._chart(reference, cut, I(0, 1), None, 96)
    root = cut.maximum_ratio_time_fraction_bounds
    assert start.parameter_expression == 'source_start'
    assert end.parameter_expression == 'source_end'
    assert du == I(1-root.hi, 1-root.lo)
    assert full.parameter_bounds.lo == root.lo and full.parameter_bounds.hi == 1
    assert full.common_physical_radius_cubed_bounds_m3 == cut.common_physical_radius_cubed_bounds_m3
    assert full.receiver_material_depth_bounds_m.hi > full.receiver_material_depth_bounds_m.lo
    assert start.genuine_query_support_certified and end.genuine_query_support_certified
    assert not full.genuine_query_support_certified
    assert not full.every_point_of_parameter_enclosure_is_material


def test_actual_prepared_gap_and_roof_aware_frame_bridge_are_retained():
    q = point()
    cold = json.loads(q.cold_field_json)
    law = context()[2]
    delta = tuple(q.hot_spatial_values[i]-api._interval(cold['position_bounds_m'][i])
                  for i in range(3))
    geometric_gap = sum((a*b for a, b in zip(delta, q.current_receiver_unit_normal_bounds)), I(0))
    overlap(q.gap_bounds_m, geometric_gap)
    assert law.fixed_rest_gap_m == 0
    assert q.gap_bounds_m != I(0)
    assert any(x != I(0) for x in delta)
    bank = next(b for b in context()[3].banks
                if b.original_parent_id == context()[1].donor_original_parent_id)
    difference = tuple(q.cold_parent_to_hot_world_input_bridge_bounds[i][j]
                       - bank.material_owner_pose_bounds[i][j]
                       for i in range(3) for j in range(3))
    assert any(x.hi < 0 or x.lo > 0 for x in difference)
    product = api._mm(q.cold_parent_to_hot_world_input_bridge_bounds, q.bridge_inverse_bounds)
    for i in range(3):
        for j in range(3):
            contains(product[i][j], F(i == j))


def test_unilateral_work_has_compression_opening_and_semismooth_closure():
    def gap(value):
        return api._Jet(value, (I(3),), ((I(5),),), 96)
    energy, stress, tangent, branch = api._unilateral(gap(I(-2)), F(7), 96)
    assert branch == 'compression' and stress == I(-14) and tangent == I(7)
    assert energy.v == I(14) and energy.d == (I(-42),) and energy.h == ((I(-7),),)
    energy, stress, tangent, branch = api._unilateral(gap(I(2)), F(7), 96)
    assert branch == 'opening' and stress == tangent == energy.v == I(0)
    assert energy.d == (I(0),) and energy.h == ((I(0),),)
    energy, stress, tangent, branch = api._unilateral(gap(I(-1, 1)), F(7), 96)
    assert branch == 'closure_crossing_semismooth_hull'
    assert tangent == I(0, 7) and stress == I(-7, 0)
    contains(energy.h[0][0], F(-35)); contains(energy.h[0][0], F(63))
    # At exact closure the classical second derivative has two one-sided values.
    _, _, tangent, branch = api._unilateral(gap(I(0)), F(7), 96)
    assert tangent == I(0, 7) and branch == 'closure_crossing_semismooth_hull'


def test_transporting_normal_curvature_matches_independent_work_derivatives():
    # A simple shear A=[[1,x,0],[0,1,0],[0,0,1]] transports m=e_x to
    # n=(1,-x,0)/sqrt(1+x^2). With separation=-e_x and kn=2,
    # independent physical work is E(x)=1/(1+x^2).
    const = lambda value: api._Jet.constant(value, 1, 96)
    x = api._Jet(I(F(1, 2)), (I(1),), ((I(0),),), 96)
    A = ((const(1), x, const(0)), (const(0), const(1), const(0)),
         (const(0), const(0), const(1)))
    cofactor = (api._cross(A[1], A[2]), api._cross(A[2], A[0]), api._cross(A[0], A[1]))
    raw = tuple(sum((row[j]*(F(1) if j == 0 else F(0)) for j in range(3)), 0)
                for row in cofactor)
    magnitude = sum((v*v for v in raw), 0).sqrt().inverse()
    normal = tuple(v*magnitude for v in raw)
    energy, _, _, branch = api._unilateral(-normal[0], F(2), 96)
    assert branch == 'compression'
    contains(energy.v, F(4, 5))
    contains(energy.d[0], F(-16, 25))
    contains(energy.h[0][0], F(-32, 125))
    # Exact rational finite differences are independent of the jet algebra.
    value = lambda t: 1/(1+t*t)
    a = F(1, 2); step = F(1, 10**12)
    first = (value(a+step)-value(a-step))/(2*step)
    second = (value(a+step)-2*value(a)+value(a-step))/(step*step)
    assert abs(first-F(-16, 25)) < F(1, 10**22)
    assert abs(second-F(-32, 125)) < F(1, 10**22)
    # Freezing the unit normal would instead produce zero derivatives here.
    assert energy.d[0].hi < 0 and energy.h[0][0].hi < 0


def test_joint_owner_rotation_preserves_true_face_work_and_scope():
    q = point(1)
    indices = q.active_coordinate_indices
    for axis in range(3):
        contains(q.gradient_density_bounds[indices.index(axis)]
                 + q.gradient_density_bounds[indices.index(3+axis)], F(0))
    assert q.physical_receiving_traction_law_defined
    assert q.complete_correlated_face_parameterization_defined
    assert not q.rectangular_geometry_enclosure_is_wholly_material
    assert not q.implicit_root_midpoint_used
    assert not q.physical_source_work_handover_certified
    assert not q.preparation_or_activation_work_paid
    assert not q.physical_material_admission_authorized
    assert not q.new_mixed_equilibrium_certified
    assert not q.clocks_advanced and q.accepted_new_native_intervals == 0


def test_typed_declaration_and_point_mutations_cannot_be_resealed():
    law = context()[2]; q = point()
    for forged in (replace(law, fixed_rest_gap_m=F(-1, 100)),
                   replace(law, preparation_or_activation_work_paid=True)):
        forged = replace(forged, binding_sha256=api._sha(api._payload(api._body(forged))))
        with pytest.raises(ValueError):
            api.native_receiving_face_declaration_to_dict(forged)
    for forged in (replace(q, gap_bounds_m=I(0)),
                   replace(q, hot_spatial_values=(I(0),)*12),
                   replace(q, clocks_advanced=True)):
        forged = replace(forged, binding_sha256=api._sha(api._payload(api._body(forged))))
        with pytest.raises(ValueError):
            api.native_receiving_face_point_to_dict(forged)


def test_complete_source_packet_and_query_kind_are_authenticated():
    ctx = context(); hot = copy.deepcopy(ctx[4]); label = hot['evaluation_queries'][0]['label']
    hot['clocks_advanced'] = True
    hot['packet_sha256'] = api._sha({k: v for k, v in hot.items() if k != 'packet_sha256'})
    with pytest.raises(ValueError, match='immutable HOT903'):
        api.evaluate_native_receiving_face(*ctx[:4], hot, label, I(F(1, 3)), cubes())
    for lam in (F(0), True, I(-1), I(0, 2)):
        with pytest.raises(ValueError):
            api.evaluate_native_receiving_face(*ctx, label, lam, cubes())
    with pytest.raises(ValueError):
        api.evaluate_native_receiving_face(*ctx, label, I(0), cubes(), precision=True)


def test_restart_requires_exact_complete_material_field_and_declared_law():
    ctx = context(); reference, cut, law, cold, hot = ctx
    raw_law = api.native_receiving_face_declaration_to_dict(law)
    restored_law = api.native_receiving_face_declaration_from_dict(raw_law, reference=reference, cut=cut)
    assert api.native_receiving_face_declaration_to_dict(restored_law) == raw_law
    raw = api.native_receiving_face_point_to_dict(point())
    kwargs = dict(reference=reference, cut=cut, declaration=law, cold_system=cold, hot_packet=hot)
    restored = api.native_receiving_face_point_from_dict(raw, **kwargs)
    assert api.native_receiving_face_point_to_dict(restored) == raw
    altered = copy.deepcopy(raw); altered['gap_bounds_m'] = api._payload(I(0))
    with pytest.raises(ValueError, match='restart changed'):
        api.native_receiving_face_point_from_dict(altered, **kwargs)
    altered = copy.deepcopy(raw_law); altered['old_memory_or_minus_001_gap_inherited'] = True
    with pytest.raises(ValueError, match='restart changed'):
        api.native_receiving_face_declaration_from_dict(altered, reference=reference, cut=cut)


def test_whole_face_operator_covers_root_and_depth_once_without_admission():
    result = whole_face()
    assert result['complete_curved_face_covered'] and result['positive_correction_boundary_used_once']
    assert not result['source_root_band_omitted'] and not result['implicit_root_midpoint_used']
    assert not result['extra_elastic_column_energy_added'] and not result['old_source_atoms_added_here']
    assert result['semismooth_tangent_hull_retained']
    assert len(result['gradient_bounds']) == len(result['hessian_bounds']) == 100
    assert all(len(row) == 100 for row in result['hessian_bounds'])
    assert not result['force_precision_sufficient_for_mixed_solve']
    assert not result['physical_material_admission_authorized']
    assert not result['new_mixed_equilibrium_certified']
    assert not result['clocks_advanced'] and result['accepted_new_native_intervals'] == 0

"""Physical support, cut-independent quadrature and retained interval history."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.geometric_initiation_validation.admission_case import fixture
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_boundary import surface_digest
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_fault_traces import (
    PhysicalFaultTrace, TraceQuadrature, UnresolvedFaultTrace,
    bind_fault_trace_history, declare_fault_trace_set, fault_trace_points,
    fault_trace_set_from_dict, fault_trace_set_to_dict, trace_history_from_dict,
    trace_history_to_dict, validate_fault_trace_history, validate_fault_trace_set,
)
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import GeometricFragment, split_geometry
from tectonics.spherical_polygons import arc_length


def setup():
    _, surface = fixture()
    columns = ElasticColumnParameters(5e4, 5e10, 2e10, 5e4, 5e10, 2e10)
    law = ContactLawParameters(normal_stiffness_pa_m=1e7, tangential_stiffness_pa_m=1e7,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=1e4,
        friction_dry=.05, friction_wet=.025, wet_cohesion_fraction=.5, viscosity_pa_s_m=1e15)
    traces = []
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        start, end = contact.start, contact.end
        horizontal = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 0 else -1.)
        if np.dot(np.cross(start, end), horizontal) < 0.:
            start, end = end, start
        traces.append(PhysicalFaultTrace(f"physical-quarter:{index}", start, end, 1, 0,
            math.pi/6., columns, law, 1e3, 0., 0., -20., 0.,
            "explicit synthetic dipping plane with original preparation anchors"))
    trace_set = declare_fault_trace_set(surface, traces, TraceQuadrature(2, 4))
    return surface, trace_set


def subdivide_one_contact(surface):
    contact = extract_contacts(surface.fragments, surface.radius_km)[0]
    target = next(f for f in surface.fragments if f.fragment_id == contact.fragment_a)
    polygon = np.asarray(target.polygon)
    zero = np.flatnonzero(polygon[:, 0] == 0.)
    apex = polygon[next(i for i in range(3) if i not in zero)]
    start, end = polygon[zero]
    midpoint = start+end
    midpoint /= np.linalg.norm(midpoint)
    children = split_geometry(target, (np.array([apex,start,midpoint]), np.array([apex,midpoint,end])), label="pure-subdivision")
    result = replace(surface, fragments=tuple(f for f in surface.fragments if f.fragment_id != target.fragment_id)+children)
    return result, contact


def test_declared_traces_cover_actual_arcs_without_contact_ids_in_physical_records():
    surface, trace_set = setup()
    coverage = validate_fault_trace_set(surface, trace_set)
    assert len(coverage.overlaps) == 4
    assert all(len(trace_ids) == 1 for _, trace_ids in coverage.contact_trace_ids)
    assert all("contact_id" not in asdict(t) for t in trace_set.traces)
    assert all(np.dot(np.cross(t.start,t.end),t.normal_sub_to_over) > 0 for t in trace_set.traces)


def test_quadrature_has_stable_point_ids_and_physical_area_and_frame():
    surface, trace_set = setup()
    points = fault_trace_points(surface, trace_set)
    assert len(points) == 4*2*4
    assert len({p.point_id for p in points}) == len(points)
    for trace in trace_set.traces:
        subset = [p for p in points if p.trace_id == trace.trace_id]
        expected_length = arc_length(trace.start,trace.end)*surface.radius_km*1000.
        assert math.fsum(p.line_weight_m for p in subset) == pytest.approx(expected_length,rel=2e-15)
        assert math.fsum(p.area_m2 for p in subset) == pytest.approx(expected_length*trace.contact_height_m/math.sin(trace.dip_radians),rel=2e-15)
        for p in subset:
            assert np.linalg.norm(p.position) == pytest.approx(1.,abs=2e-15)
            assert np.linalg.norm(p.horizontal) == pytest.approx(1.,abs=2e-15)
            assert np.dot(p.position,p.horizontal) == pytest.approx(0.,abs=2e-15)
            assert np.dot(p.normal,p.down_dip) == pytest.approx(0.,abs=2e-15)


def test_one_physical_arc_spans_new_material_cuts_with_identical_nodes_and_signature():
    surface, trace_set = setup()
    changed, _ = subdivide_one_contact(surface)
    rebound = declare_fault_trace_set(changed, trace_set.traces, trace_set.quadrature)
    assert len(extract_contacts(changed.fragments,changed.radius_km)) == 5
    assert rebound.trace_sha256 == trace_set.trace_sha256
    assert rebound.surface_sha256 != trace_set.surface_sha256
    assert fault_trace_points(changed,rebound) == fault_trace_points(surface,trace_set)
    coverage = validate_fault_trace_set(changed,rebound)
    assert any(len(contact_ids)==2 for _,contact_ids in coverage.trace_contact_ids)
    with pytest.raises(ValueError,match="another exact surface"):
        validate_fault_trace_set(changed,trace_set)


def test_physical_trace_bounds_need_not_coincide_with_material_contact_endpoints():
    surface, original = setup()
    reference = original.traces[0]
    traces=[]
    # The four clockwise arcs lie on x=0 with h=-x. A non-grid origin crosses
    # each original material boundary while covering precisely the same circle.
    for index in range(4):
        first=.17-index*math.pi/2.
        last=first-math.pi/2.
        traces.append(replace(reference,trace_id=f"shifted:{index}",
            start=(0.,math.cos(first),math.sin(first)),end=(0.,math.cos(last),math.sin(last))))
    changed=declare_fault_trace_set(surface,traces,TraceQuadrature(3,5))
    coverage=validate_fault_trace_set(surface,changed)
    assert len(coverage.overlaps)==8
    assert all(len(values)==2 for _,values in coverage.contact_trace_ids)


def test_zero_measure_rounded_endpoint_touch_does_not_create_extra_support():
    surface, original=setup()
    ref=original.traces[0]
    traces=[]
    for index in range(4):
        a=-index*math.pi/2.;b=a-math.pi/2.
        traces.append(replace(ref,trace_id=f"clockwise:{index}",
            start=(0.,math.cos(a),math.sin(a)),end=(0.,math.cos(b),math.sin(b))))
    result=declare_fault_trace_set(surface,traces,TraceQuadrature(1,4))
    assert len(validate_fault_trace_set(surface,result).overlaps)==4


def test_rounded_vertex_touch_does_not_inherit_another_contact_bank_pair():
    surface, original = setup()
    # Eight distinct octant owners give different bank pairs on adjacent
    # coplanar quarter arcs. One ulp of endpoint arithmetic is only a touch.
    surface = replace(surface, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=index))
        for index, fragment in enumerate(surface.fragments)))
    traces = []
    reference = original.traces[0]
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        start, end = np.asarray(contact.start), np.asarray(contact.end)
        horizontal = np.asarray(contact.normal_a_to_b)
        if np.dot(np.cross(start, end), horizontal) < 0.:
            start, end = end, start
        tangent = np.cross(horizontal, start)
        tangent /= np.linalg.norm(tangent)
        angle = math.nextafter(arc_length(start, end), math.inf)
        rounded_end = start*math.cos(angle)+tangent*math.sin(angle)
        traces.append(replace(reference, trace_id=f"vertex:{index}",
            start=tuple(start), end=tuple(rounded_end),
            subducting_plate=contact.plate_a, overriding_plate=contact.plate_b))
    result = declare_fault_trace_set(surface, traces, TraceQuadrature(1, 4))
    assert len(validate_fault_trace_set(surface, result).overlaps) == 12


def test_positive_missing_support_and_duplicate_opposite_planes_are_rejected():
    surface, trace_set=setup()
    with pytest.raises(UnresolvedFaultTrace,match="Missing"):
        declare_fault_trace_set(surface,trace_set.traces[:-1],trace_set.quadrature)
    original=trace_set.traces[0]
    horizontal=np.asarray(original.normal_sub_to_over)
    tangent=np.cross(horizontal,original.start)
    point=np.asarray(original.start)*math.cos(1e-8)+tangent*math.sin(1e-8)
    shortened=replace(original,start=tuple(point))
    with pytest.raises(UnresolvedFaultTrace,match="Missing"):
        declare_fault_trace_set(surface,(shortened,*trace_set.traces[1:]),trace_set.quadrature)
    alternate=replace(original,trace_id="opposed",start=original.end,end=original.start,
                      subducting_plate=0,overriding_plate=1)
    with pytest.raises(UnresolvedFaultTrace,match="Multiple"):
        declare_fault_trace_set(surface,(*trace_set.traces,alternate),trace_set.quadrature)
    backwards=replace(original,start=original.end,end=original.start)
    with pytest.raises(UnresolvedFaultTrace,match="oppose"):
        declare_fault_trace_set(surface,(backwards,*trace_set.traces[1:]),trace_set.quadrature)


def test_history_retains_original_contact_evidence_after_explicit_subdivision_rebind():
    surface,trace_set=setup()
    changed,contact=subdivide_one_contact(surface)
    origin=PolarityEvidence("old-material",contact.contact_id,1,0,"old attached material direction")
    physical=bind_fault_trace_history(surface,trace_set,(origin,))
    validate_fault_trace_history(trace_set,physical)
    assert len(physical)==1
    assert physical[0].origin==origin
    assert physical[0].origin_surface_sha256==surface_digest(surface)
    assert origin.contact_id not in {c.contact_id for c in extract_contacts(changed.fragments,changed.radius_km)}
    rebound=declare_fault_trace_set(changed,trace_set.traces,trace_set.quadrature)
    assert validate_fault_trace_history(rebound,physical)==physical
    restored=trace_history_from_dict(json.loads(json.dumps(trace_history_to_dict(physical))),rebound)
    assert restored==physical


def test_conflicting_and_opposed_history_are_retained_and_rejected_without_selection():
    surface,trace_set=setup()
    contact=extract_contacts(surface.fragments,surface.radius_km)[0]
    raw=tuple(PolarityEvidence(f"old:{sub}",contact.contact_id,sub,1-sub,"old attached material") for sub in (0,1))
    physical=bind_fault_trace_history(surface,trace_set,raw)
    assert {p.origin for p in physical}==set(raw)
    with pytest.raises(UnresolvedFaultTrace,match="Conflicting"):
        validate_fault_trace_history(trace_set,physical)
    with pytest.raises(UnresolvedFaultTrace,match="Conflicting"):
        validate_fault_trace_set(surface,trace_set,raw)
    opposed=bind_fault_trace_history(surface,trace_set,(raw[0],))
    with pytest.raises(UnresolvedFaultTrace,match="opposes"):
        validate_fault_trace_history(trace_set,opposed)


def test_passive_rotation_transforms_physical_points_and_preserves_ids_and_area():
    surface,trace_set=setup()
    matrix=Rotation.from_rotvec([.2,-.1,.3]).as_matrix()
    rotated=replace(surface,fragments=tuple(GeometricFragment(f.fragment_id,
        tuple(map(tuple,np.asarray(f.polygon)@matrix.T)),f.parcel,f.parent_fragment_id) for f in surface.fragments))
    traces=tuple(replace(t,start=tuple(matrix@t.start),end=tuple(matrix@t.end)) for t in trace_set.traces)
    rebound=declare_fault_trace_set(rotated,traces,trace_set.quadrature)
    for original,new in zip(fault_trace_points(surface,trace_set),fault_trace_points(rotated,rebound)):
        assert original.point_id==new.point_id
        assert new.area_m2==pytest.approx(original.area_m2,rel=2e-15)
        for name in ("position","horizontal","normal","down_dip"):
            np.testing.assert_allclose(getattr(new,name),matrix@np.asarray(getattr(original,name)),rtol=0.,atol=3e-15)


def test_trace_checkpoint_strict_roundtrip_and_bindings():
    surface,trace_set=setup()
    raw=json.loads(json.dumps(fault_trace_set_to_dict(surface,trace_set)))
    assert fault_trace_set_from_dict(raw,surface)==trace_set
    with pytest.raises(ValueError,match="another exact surface"):
        fault_trace_set_from_dict(raw,replace(surface,time_myr=1.))
    for key in ("trace_sha256","surface_sha256"):
        bad=dict(raw);bad[key]="0"*64
        with pytest.raises(ValueError):fault_trace_set_from_dict(bad,surface)
    raw["traces"][0]["law"].pop("viscosity_pa_s_m")
    with pytest.raises(ValueError,match="every existing"):
        fault_trace_set_from_dict(raw,surface)


@pytest.mark.parametrize("name,value",[("pore_pressure_pa",1.),("water",True),("water",1.1),
    ("contact_height_m",0.),("initial_gap_m",.1),("dip_radians",0.),("dip_radians",math.pi/2.),
    ("subducting_plate",True),("provenance","")])
def test_invalid_or_unsupported_physical_inputs_are_explicitly_rejected(name,value):
    _,trace_set=setup()
    with pytest.raises(ValueError):replace(trace_set.traces[0],**{name:value})


@pytest.mark.parametrize("panel,order",[(0,4),(1,1),(True,4),(1,True),(1,2.5)])
def test_quadrature_has_no_implicit_or_invalid_configuration(panel,order):
    with pytest.raises(ValueError):TraceQuadrature(panel,order)
    with pytest.raises(TypeError):TraceQuadrature()


def test_physical_history_io_rejects_origin_and_support_corruption():
    surface,trace_set=setup()
    contact=extract_contacts(surface.fragments,surface.radius_km)[0]
    history=bind_fault_trace_history(surface,trace_set,(PolarityEvidence("old",contact.contact_id,1,0,"origin"),))
    raw=json.loads(json.dumps(trace_history_to_dict(history)))
    raw[0]["origin"].pop("origin_evidence_id")
    with pytest.raises(ValueError,match="every polarity"):
        trace_history_from_dict(raw,trace_set)
    raw=json.loads(json.dumps(trace_history_to_dict(history)))
    raw[0]["start"],raw[0]["end"]=raw[0]["end"],raw[0]["start"]
    with pytest.raises(ValueError,match="orientation"):
        trace_history_from_dict(raw,trace_set)

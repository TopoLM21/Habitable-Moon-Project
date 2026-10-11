"""True-mask/contact-cut tests and independent literal-SO3 curve derivatives."""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import copy
import json

import pytest

from tectonics import geometric_native_common_contact_cuts as api
from tectonics.geometric_native_common_reference import bind_native_common_reference
from tectonics.geometric_native_pose import FractionInterval as I


@lru_cache(maxsize=1)
def context():
    ref=bind_native_common_reference()
    return ref,api.bind_common_contact_cuts(ref)


def cut(suffix):
    return next(c for c in context()[1].cuts if c.cut_id.endswith(suffix))


def probe(c, fraction=F(1,3)):
    p=c.parameter_guaranteed_bounds
    return I(p.lo+fraction*(p.hi-p.lo))


def cube(tau=F(1000)):
    R=context()[0].radius_m
    return I(R**3-3*R**2*tau)


def dec(x):
    x=F(x)
    return D(x.numerator)/D(x.denominator)


def dot(a,b):
    return sum((x*y for x,y in zip(a,b)),D(0))


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def scale(a,x):
    return tuple(a*y for y in x)


def add(a,b):
    return tuple(x+y for x,y in zip(a,b))


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def rotate(axis,vector):
    """Independent Decimal literal exponential action, no stored source series."""
    term=total=tuple(vector)
    magnitude=dot(vector,vector).sqrt()
    if not magnitude:
        return total
    # The true root derivative is about1e-43. Absolute truncation would lose
    # forty-three relative digits before normalizing the resulting tiny face.
    threshold=D("1e-170")*magnitude
    for n in range(1,130):
        term=scale(D(1)/n,cross(axis,term))
        total=add(total,term)
        if max(map(abs,term))<threshold:
            return total
    raise AssertionError("Independent literal exponential failed to converge")


def independent_curve(ref,u,tau):
    """Differentiate the two literal SO3 owner maps before forming a,b,k,h."""
    before=json.loads(dict(ref.complete_source_dto_json)["before"])
    journal=before["snapshot"]["current_material_epoch"]["source_material_state"]["journal"]
    dt=D.from_float(journal["dt_myr"])
    owners=dict(journal["omega_rad_per_myr"])
    donor=tuple(D.from_float(x)*dt for x in owners[1])
    receiver=tuple(D.from_float(x)*dt for x in owners[0])
    right=rotate(scale(u,receiver),(D(-1),D(0),D(0)))
    n=rotate(scale(-u,donor),right)
    first=rotate(scale(-u,donor),cross(receiver,right))
    second=rotate(scale(-u,donor),cross(receiver,cross(receiver,right)))
    dn=add(scale(D(-1),cross(donor,n)),first)
    ddn=add(sub(cross(donor,cross(donor,n)),scale(D(2),cross(donor,first))),second)
    C,Y,Z=-n[0],n[1],n[2]
    Cp,Yp,Zp=-dn[0],dn[1],dn[2]
    Cpp,Ypp,Zpp=-ddn[0],ddn[1],ddn[2]
    A1,B1=Yp*C-Y*Cp,Zp*C-Z*Cp
    A1p,B1p=Ypp*C-Y*Cpp,Zpp*C-Z*Cpp
    a,b=Y/C,Z/C
    k=B1/A1
    v=-(B1p*A1-B1*A1p)/(A1*A1)
    h=b-a*k
    q=(h,-k,D(1))
    qp=(v*a,v,D(0))
    s=dot(q,q).sqrt()
    U0=scale(D(1)/s,q)
    Up0=sub(scale(D(1)/s,qp),scale(dot(q,qp)/(s*s*s),q))
    N0=scale(D(1)/(D(1)+a*a+b*b).sqrt(),(D(-1),a,b))
    U,Up,N=(rotate(donor,x) for x in (U0,Up0,N0))
    speed=dot(Up,Up).sqrt()
    R=dec(ref.radius_m)
    r=R
    radius_cube=R**3-3*R**2*tau
    for _ in range(24):
        r=(2*r+radius_cube/(r*r))/3
    A=tuple(dec(x) for x in ref.angular_rate_per_physical_volume_depth_m)
    Q=lambda x:rotate(scale(tau,A),x)
    X=scale(r,Q(U))
    Xp=scale(r,Q(Up))
    Xt=Q(add(scale(-R*R/(r*r),U),scale(r,cross(A,U))))
    raw=cross(Xp,Xt)
    J=dot(raw,raw).sqrt()
    normal=scale(D(1)/J,raw)
    tangent=scale(D(1)/dot(Xp,Xp).sqrt(),Xp)
    down_raw=sub(Xt,scale(dot(Xt,tangent),tangent))
    down=scale(D(1)/dot(down_raw,down_raw).sqrt(),down_raw)
    return dict(U=U,Up=Up,qd=U0,X=X,Xp=Xp,Xt=Xt,J=J,normal=normal,down=down,speed=speed)


def contains(value, interval, error=D("1e-145")):
    assert dec(interval.lo)-error<=value<=dec(interval.hi)+error


def test_all_five_actual_cuts_and_true_curve_have_source_lineage_not_atoms():
    ref,collection=context()
    assert len(collection.cuts)==6
    assert collection.complete_source_dto_json==ref.complete_source_dto_json
    assert collection.complete_historical_saved_field_json==ref.saved_field_json
    pairs={(c.donor_original_parent_id,c.receiver_original_parent_id) for c in collection.cuts if c.receiving_trace_id}
    assert pairs=={("octant:5","octant:1"),("octant:5","octant:3"),("octant:6","octant:2"),
                   ("octant:7","octant:2"),("octant:7","octant:3")}
    assert all(not c.old_mechanical_atom_ids_assigned and not c.constitutive_rest_contact_reference_recovered
               and not c.physical_material_admission_authorized for c in collection.cuts)
    hot=cut("hot-curved-extra")
    assert hot.receiving_trace_id is None
    assert hot.receiver_material_kind=="hot_birth"
    assert hot.genuine_native_arc_json is None
    correction=json.loads(hot.complete_paired_correction_source_json)
    assert F(*correction["area_bounds_sr"]["lo"])>0
    assert hot.receiver_component_id==correction["positive_component_id"]
    assert "B1/A1" in hot.source_parameter_expression
    c63=cut("octant:5:octant:3")
    c72=cut("octant:7:octant:2")
    assert c63.receiving_trace_id==cut("octant:7:octant:3").receiving_trace_id
    assert c72.receiving_trace_id==cut("octant:6:octant:2").receiving_trace_id
    assert c63.parameter_endpoint_bounds.lo>0 and c63.parameter_endpoint_bounds.hi<F(1,10**30)
    assert c72.parameter_endpoint_bounds.lo>0 and c72.parameter_endpoint_bounds.hi<F(1,10**30)


def test_c6_terminal_trim_keeps_exact_expression_and_refuses_archived_proxy_end():
    ref,_=context()
    c=cut("octant:5:octant:1")
    assert c.kind=="implicit_terminal_trim"
    assert c.exact_terminal_slope_expression=="B1(1)/A1(1)"
    assert c.source_live_cut_endpoint_margins[0].lo>0
    assert c.source_live_cut_endpoint_margins[1].hi<0
    assert 0<c.parameter_endpoint_bounds.lo<=c.parameter_endpoint_bounds.hi<c.native_untrimmed_angle_endpoint_bounds.lo
    with pytest.raises(ValueError,match="parameter enclosure"):
        api.evaluate_common_contact_cut(ref,c,c.native_untrimmed_angle_endpoint_bounds,cube())
    full=api.evaluate_common_contact_cut(ref,c,physical_radius_cubed_bounds_m3=cube(),precision=128)
    assert not full.genuine_query_support_certified
    assert not full.every_point_of_parameter_enclosure_is_material
    endpoint=api.evaluate_common_contact_cut(ref,c,"end",cube(),precision=128)
    assert endpoint.genuine_query_support_certified
    assert not endpoint.every_point_of_parameter_enclosure_is_material
    assert endpoint.parameter_expression=="source_end"


@pytest.mark.parametrize("suffix",("octant:5:octant:1","octant:5:octant:3","octant:6:octant:2",
                                    "octant:7:octant:2","octant:7:octant:3","hot-curved-extra"))
def test_each_actual_cut_uses_common_radius_and_real_distinct_depths(suffix):
    ref,_=context()
    c=cut(suffix)
    g=api.evaluate_common_contact_cut(ref,c,probe(c),precision=128)
    assert g.genuine_query_support_certified and g.every_point_of_parameter_enclosure_is_material
    assert g.reference_faces_match_at_every_common_physical_radius
    assert g.area_density_bounds_m2_per_parameter_per_material_m.lo>0
    assert g.donor_material_depth_bounds_m.lo>=0
    assert g.receiver_material_depth_bounds_m.lo>=0
    db=next(b for b in ref.banks if b.component_id==c.donor_component_id)
    rb=next(b for b in ref.banks if b.component_id==c.receiver_component_id)
    assert g.donor_material_depth_bounds_m.hi<=db.areal_material_thickness_m
    assert g.receiver_material_depth_bounds_m.hi<=rb.areal_material_thickness_m
    R=ref.radius_m
    offset=((R+rb.roof_radial_displacement_m)**3-(R+db.roof_radial_displacement_m)**3)/(3*R**2)
    assert g.receiver_material_depth_bounds_m.lo-g.donor_material_depth_bounds_m.lo==offset
    assert g.receiver_material_depth_bounds_m.hi-g.donor_material_depth_bounds_m.hi==offset
    if c.receiving_trace_id:
        assert g.receiver_original_unit_direction_bounds is not None
        assert g.receiving_native_angle_bounds_radians is not None
        assert g.receiving_computational_angle_bounds_radians is not None
    else:
        assert g.receiver_original_unit_direction_bounds is None
        assert g.receiving_native_angle_bounds_radians is None
    with pytest.raises(ValueError,match="physical material depths"):
        api.evaluate_common_contact_cut(ref,c,probe(c),I(c.common_physical_radius_cubed_bounds_m3.hi+1))


def test_unknown_curve_turning_band_is_kept_regular_without_membership_inference():
    ref,_=context()
    c=cut("hot-curved-extra")
    band=api.evaluate_common_contact_cut(ref,c,c.maximum_ratio_time_fraction_bounds,cube(),precision=128)
    assert band.source_angular_speed_bounds.lo>0
    assert band.area_density_bounds_m2_per_parameter_per_material_m.lo>0
    assert not band.genuine_query_support_certified
    assert not band.original_quarter_constraints_refined_from_true_cut_proof
    start=api.evaluate_common_contact_cut(ref,c,"start",cube(),precision=128)
    assert start.genuine_query_support_certified
    assert not start.every_point_of_parameter_enclosure_is_material
    assert start.parameter_bounds==c.maximum_ratio_time_fraction_bounds
    with pytest.raises(ValueError,match="parameter enclosure"):
        api.evaluate_common_contact_cut(ref,c,I(c.maximum_ratio_time_fraction_bounds.lo/2),cube())


@pytest.mark.parametrize("parameter",("near_root",F(1,4),F(1,2),F(1)))
def test_true_curve_physical_differential_and_normal_match_independent_so3(parameter):
    ref,_=context()
    c=cut("hot-curved-extra")
    near_root=parameter=="near_root"
    if near_root:
        parameter=c.maximum_ratio_time_fraction_bounds.hi+F(1,10**30)
    g=api.evaluate_common_contact_cut(ref,c,I(parameter),cube(),precision=128,order=64)
    with localcontext() as ctx:
        ctx.prec=180
        oracle=independent_curve(ref,dec(parameter),D(1000))
        for name,values in (("U",g.source_unit_world_direction_bounds),
                            ("Up",g.source_unit_world_direction_parameter_derivative_bounds),
                            ("qd",g.donor_original_unit_direction_bounds),
                            ("X",g.world_position_bounds_m),("Xp",g.parameter_tangent_bounds_m),
                            ("Xt",g.material_depth_tangent_bounds),("normal",g.receiver_inward_unit_normal_bounds),
                            ("down",g.inward_unit_down_direction_bounds)):
            for value,bounds in zip(oracle[name],values):
                contains(value,bounds)
        contains(oracle["J"],g.area_density_bounds_m2_per_parameter_per_material_m,error=D("1e-150"))
        contains(oracle["speed"],g.source_angular_speed_bounds,error=D("1e-150"))
        assert g.area_density_bounds_m2_per_parameter_per_material_m.hi<F(1,10**20)
        # Relative checks ensure a constant point/zero derivative cannot pass.
        h=D("1e-33") if near_root else D("1e-24")
        plus=independent_curve(ref,dec(parameter)+h,D(1000))
        minus=independent_curve(ref,dec(parameter)-h,D(1000))
        fd=scale(D(1)/(2*h),sub(plus["X"],minus["X"]))
        error=dot(sub(fd,oracle["Xp"]),sub(fd,oracle["Xp"])).sqrt()/dot(oracle["Xp"],oracle["Xp"]).sqrt()
        assert error<D("1e-40")


def test_terminal_join_is_same_physical_point_and_receiving_normal_on_both_curves():
    ref,_=context()
    a=api.evaluate_common_contact_cut(ref,cut("octant:5:octant:1"),"end",cube(),precision=128)
    b=api.evaluate_common_contact_cut(ref,cut("hot-curved-extra"),"end",cube(),precision=128)
    for x,y in zip(a.world_position_bounds_m,b.world_position_bounds_m):
        assert x.lo<=y.hi and y.lo<=x.hi
    for x,y in zip(a.receiver_inward_unit_normal_bounds,b.receiver_inward_unit_normal_bounds):
        assert x.lo<=y.hi and y.lo<=x.hi
    assert a.cut.receiving_trace_id is not None
    assert b.cut.receiving_trace_id is None


def test_serialized_receipts_replay_and_reject_resealed_source_mask_history_and_rate():
    ref,collection=context()
    raw=api.common_contact_cuts_to_dict(collection)
    assert api.common_contact_cuts_from_dict(raw,reference=ref)==collection
    c=cut("hot-curved-extra")
    single=api.common_contact_cut_to_dict(c)
    assert api.common_contact_cut_from_dict(single,reference=ref)==c
    for name,value in (("complete_donor_mask_source_json","{}"),("complete_source_history_json","{}"),
                       ("reference_snapshot_sha256","0"*64),("receiving_trace_id","invented"),
                       ("terminal_slope_bounds",I(0))):
        forged=replace(c,**{name:value},cut_snapshot_sha256="")
        forged=replace(forged,cut_snapshot_sha256=api._seal(forged,"cut_snapshot_sha256"))
        with pytest.raises(ValueError):
            api.validate_common_contact_cut(forged,reference=ref)
    for path in ("complete_source_history_json","complete_receiver_mask_source_json","cut_snapshot_sha256"):
        altered=copy.deepcopy(raw)
        altered["cuts"][0][path]="changed"
        with pytest.raises(ValueError,match="independently rebuilt"):
            api.common_contact_cuts_from_dict(altered,reference=ref)
    wrong=replace(ref,angular_rate_per_physical_volume_depth_m=(F(0),F(0),F(0)))
    with pytest.raises(ValueError):
        api.evaluate_common_contact_cut(wrong,c,I(F(1,2)),cube())
    g=api.evaluate_common_contact_cut(ref,c,I(F(1,2)),cube(),precision=128)
    evaluated=api.common_contact_cut_bounds_to_dict(g)
    assert api.common_contact_cut_bounds_from_dict(evaluated,reference=ref)==g
    forged=replace(g,genuine_query_support_certified=False,bounds_snapshot_sha256="")
    forged=replace(forged,bounds_snapshot_sha256=api._seal(forged,"bounds_snapshot_sha256"))
    with pytest.raises(ValueError,match="authenticated parameter/source expression"):
        api.validate_common_contact_cut_bounds(forged,reference=ref)
    evaluated["world_position_bounds_m"][0]={"lo":[0,1],"hi":[0,1]}
    with pytest.raises(ValueError,match="independently rebuilt"):
        api.common_contact_cut_bounds_from_dict(evaluated,reference=ref)

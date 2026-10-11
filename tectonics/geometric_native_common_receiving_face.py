"""Two-sided authentic curved material traces and a NEW virgin HOT contact.

The fixed genuine common-face measure is charged once. Elastic columns are
already in the separate bulk laws and are not added here. Source-contact
history, reference preparation work, activation payment, physical time and
equilibrium are deliberately separate from this declared endpoint operator.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_joint_spherical_field import rotate_joint_physical_depth
from .geometric_native_common_contact_cuts import (
    validate_common_contact_cut, evaluate_common_contact_cut,
    common_contact_cut_to_dict, common_contact_cut_bounds_to_dict,
)
from .geometric_native_common_hot_receiving import (
    SOURCE_SEAL, encoded, _sha, _precision, _field_twojet, _cross, _mv, _mm, _axis,
)
from .geometric_native_common_carrier_source_work import (
    _payload, _interval, _fraction, _round, _typed_guard, _inverse,
)

SCOPE = 'NEW_VIRGIN_COMPLETE_CURVED_FACE_ENDPOINT_CONTACT_OPERATOR_ONLY'
MODEL = 'FRICTIONLESS_UNBONDED_BARE_PENALTY_ON_FIXED_GENUINE_COMMON_FACE'
_DECLARATIONS, _POINTS = {}, {}


def _body(value):
    return {f.name:getattr(value,f.name) for f in fields(value) if f.name!='binding_sha256'}


def _remember(value, registry):
    value=replace(value,binding_sha256=_sha(_payload(_body(value))))
    registry[value.binding_sha256]=_typed_guard(value)
    return value


def _validate(value, kind, registry):
    if (type(value) is not kind or _sha(_payload(_body(value)))!=value.binding_sha256
            or registry.get(value.binding_sha256)!=_typed_guard(value)):
        raise ValueError('Complete typed receiving-face source binding changed')
    return value


def _hot(packet, query_label, component):
    if (type(packet) is not dict or packet.get('packet_sha256')!=SOURCE_SEAL
            or _sha({k:v for k,v in packet.items() if k!='packet_sha256'})!=SOURCE_SEAL):
        raise ValueError('The complete immutable HOT903 source is required')
    queries=[q for q in packet['evaluation_queries'] if q['label']==query_label]
    bodies=packet['hot_mean_system']['bodies']
    indices=[i for i,b in enumerate(bodies) if b['component_id']==component]
    if type(query_label) is not str or len(queries)!=1 or len(indices)!=1:
        raise ValueError('Exactly one authentic saved HOT chart and actual receiving body are required')
    index=indices[0];body=bodies[index];response=queries[0]['response']
    chart=json.loads(response['body_responses_json'][index]);cert=chart['finite_mean_root_certificate']
    if (chart['component_id']!=component or cert['component_id']!=component
            or cert['whole_mask_unique_root_certified'] is not True
            or cert['source_polygon_pose_applied_once'] is not True
            or cert['additional_owner_rotation_applied'] is not False
            or _fraction(body['source_birth_radial_displacement_m'])!=0):
        raise ValueError('The actual finite-mean chart or source roof changed')
    return index,body,response,chart


@dataclass(frozen=True,slots=True)
class NativeReceivingFaceDeclaration:
    cut_snapshot_sha256: str
    complete_cut_json: str
    normal_stiffness_pa_m: F
    fixed_rest_gap_m: F
    reference_clock_fraction: F
    measure_model: str
    provenance: str
    binding_sha256: str
    scope: str = SCOPE
    permanently_unbonded: bool = True
    frictionless: bool = True
    old_columns_charged_again: bool = False
    old_memory_or_minus_001_gap_inherited: bool = False
    preparation_or_activation_work_paid: bool = False
    physical_material_admission_authorized: bool = False
    clocks_advanced: bool = False


def declare_native_receiving_face(reference, cut, *, normal_stiffness_pa_m=F(10**7),
        provenance):
    cut=validate_common_contact_cut(cut,reference=reference)
    if (cut.kind!='true_curved_envelope' or cut.receiving_trace_id is not None
            or cut.receiver_material_kind!='hot_birth' or not cut.complete_paired_correction_source_json):
        raise ValueError('This NEW law applies only to the complete authentic C6/HOT C1 curved face')
    if type(normal_stiffness_pa_m) is not F or normal_stiffness_pa_m<=0:
        raise ValueError('The bare interface stiffness is an explicit positive Fraction')
    if type(provenance) is not str or not provenance.strip():
        raise ValueError('A NEW physical law declaration needs explicit provenance')
    return _remember(NativeReceivingFaceDeclaration(cut.cut_snapshot_sha256,
        encoded(common_contact_cut_to_dict(cut)).decode(),normal_stiffness_pa_m,F(0),F(1),
        'FIXED_AUTHENTIC_COMMON_FACE_AREA_J_DU_D_PHYSICAL_VOLUME_DEPTH',provenance,''),_DECLARATIONS)


def native_receiving_face_declaration_to_dict(declaration):
    _validate(declaration,NativeReceivingFaceDeclaration,_DECLARATIONS)
    return dict(format='native-curved-receiving-face-declaration-1',**_payload(declaration))


def native_receiving_face_declaration_from_dict(raw, *, reference, cut):
    from .geometric_native_common_cold_material_carrier import _saved_json_kinds
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get('format')!='native-curved-receiving-face-declaration-1':
        raise ValueError('A complete literal receiving declaration DTO is required')
    value=declare_native_receiving_face(reference,cut,
        normal_stiffness_pa_m=_fraction(raw['normal_stiffness_pa_m']),provenance=raw['provenance'])
    if encoded(native_receiving_face_declaration_to_dict(value))!=encoded(raw):
        raise ValueError('Receiving declaration restart changed its law, measure or source')
    return value


def _chart(reference,cut,lam,cubes,precision):
    if type(lam) is not I or not 0<=lam.lo<=lam.hi<=1:
        raise ValueError('The complete correlated face chart uses an exact interval lambda in [0,1]')
    root=cut.maximum_ratio_time_fraction_bounds
    # The exact implicit root is retained. These monotone bounds enclose its
    # correlated expression, including the entire uncertain root band.
    u=I(root.lo+(1-root.lo)*lam.lo,root.hi+(1-root.hi)*lam.hi)
    parameter='start' if lam==I(0) else 'end' if lam==I(1) else u
    bounds=evaluate_common_contact_cut(reference,cut,parameter,cubes,precision=precision,order=32)
    return bounds,I(1-root.hi,1-root.lo)


class _Jet:
    """Outward dense second-order jet; material labels are fixed inputs."""
    __slots__=('v','d','h','p')
    def __init__(self,v,d,h,p):self.v,self.d,self.h,self.p=v,d,h,p
    @classmethod
    def constant(cls,v,n,p):
        return cls(v if type(v) is I else I(v),(I(0),)*n,((I(0),)*n,)*n,p)
    def other(self,x):return x if type(x) is _Jet else self.constant(x,len(self.d),self.p)
    def __add__(self,x):
        x=self.other(x);r=lambda v:_round(v,self.p)
        return _Jet(r(self.v+x.v),tuple(r(a+b) for a,b in zip(self.d,x.d)),
            tuple(tuple(r(a+b) for a,b in zip(A,B)) for A,B in zip(self.h,x.h)),self.p)
    __radd__=__add__
    def __neg__(self):return _Jet(-self.v,tuple(-a for a in self.d),tuple(tuple(-a for a in row) for row in self.h),self.p)
    def __sub__(self,x):return self+-self.other(x)
    def __rsub__(self,x):return self.other(x)+-self
    def __mul__(self,x):
        x=self.other(x);r=lambda v:_round(v,self.p);n=len(self.d)
        return _Jet(r(self.v*x.v),tuple(r(self.d[i]*x.v+self.v*x.d[i]) for i in range(n)),
            tuple(tuple(r(self.h[i][j]*x.v+self.v*x.h[i][j]+self.d[i]*x.d[j]+self.d[j]*x.d[i])
                for j in range(n)) for i in range(n)),self.p)
    __rmul__=__mul__
    def compose(self,value,first,second):
        r=lambda v:_round(v,self.p);n=len(self.d)
        return _Jet(r(value),tuple(r(first*d) for d in self.d),
            tuple(tuple(r(first*self.h[i][j]+second*self.d[i]*self.d[j]) for j in range(n)) for i in range(n)),self.p)
    def inverse(self):
        if self.v.lo<=0<=self.v.hi:raise ValueError('A jet reciprocal crosses zero')
        return self.compose(1/self.v,-1/(self.v*self.v),2/(self.v*self.v*self.v))
    def sqrt(self):
        value=interval_sqrt(self.v,self.p+16)
        if value.lo<=0:raise ValueError('The transported receiving face lost positive normal magnitude')
        return self.compose(value,1/(2*value),-1/(4*value*value*value))


def _embed(values,first,second,localindices,indices,precision):
    where={v:i for i,v in enumerate(indices)};n=len(indices);result=[]
    for c,value in enumerate(values):
        d=[I(0)]*n;h=[[I(0)]*n for _ in range(n)]
        for a,globalindex in enumerate(localindices):
            d[where[globalindex]]=first[c][a]
            for b,other in enumerate(localindices):h[where[globalindex]][where[other]]=second[c][a][b]
        result.append(_Jet(value,tuple(d),tuple(tuple(row) for row in h),precision))
    return tuple(result)


def _cold_twojet(field,indices,precision):
    values=field.position_bounds_m;owner=field.owner
    modes=field.mode_displacement_columns
    selected=tuple(i for i in range(56) if 36+i in indices)
    local=tuple(range(3*owner,3*owner+3))+tuple(36+i for i in selected)
    n=len(local);first=[];second=[]
    for component in range(3):
        first.append(tuple(_cross(_axis(a),values)[component] for a in range(3))
            +tuple(modes[i][component] for i in selected))
        matrix=[]
        for a in range(n):
            row=[]
            for b in range(n):
                if a<3 and b<3:
                    v=(_cross(_axis(a),_cross(_axis(b),values))[component]
                        +_cross(_axis(b),_cross(_axis(a),values))[component])/2
                elif a<3 or b<3:
                    v=_cross(_axis(a if a<3 else b),modes[selected[(b if a<3 else a)-3]])[component]
                else:v=I(0)
                row.append(_round(v,precision))
            matrix.append(tuple(row))
        second.append(tuple(matrix))
    return _embed(values,tuple(first),tuple(second),local,indices,precision)


def _unilateral(gap,stiffness,precision):
    negative=I(min(gap.v.lo,F(0)),min(gap.v.hi,F(0)))
    energy=I(stiffness*negative.hi**2/2,stiffness*negative.lo**2/2)
    stress=stiffness*negative
    tangent=I(stiffness) if gap.v.hi<0 else I(0) if gap.v.lo>0 else I(0,stiffness)
    response=gap.compose(energy,stress,tangent)
    branch='compression' if gap.v.hi<0 else 'opening' if gap.v.lo>0 else 'closure_crossing_semismooth_hull'
    return response,stress,tangent,branch


@dataclass(frozen=True,slots=True)
class NativeReceivingFacePoint:
    declaration_binding_sha256: str
    hot_source_packet_sha256: str
    hot_query_label: str
    cold_coordinates_m: tuple
    lambda_bounds: I
    implicit_root_bounds: I
    source_parameter_expression: str
    cut_bounds_json: str
    cold_field_json: str
    hot_body_json: str
    hot_chart_json: str
    cold_parent_to_hot_world_input_bridge_bounds: tuple
    bridge_inverse_bounds: tuple
    cold_gradient_in_hot_world_input_bounds: tuple
    hot_prepared_position_bounds_m: tuple
    hot_prepared_original_world_gradient_bounds: tuple
    original_world_material_normal_bounds: tuple
    active_coordinate_indices: tuple
    hot_spatial_values: tuple
    hot_spatial_first: tuple
    hot_spatial_second: tuple
    current_receiver_unit_normal_bounds: tuple
    gap_bounds_m: I
    gap_first: tuple
    gap_second: tuple
    normal_traction_bounds_pa: I
    normal_tangent_bounds_pa_m: I
    branch: str
    reference_area_density_bounds: I
    energy_density_bounds_j_per_lambda_per_m: I
    gradient_density_bounds: tuple
    hessian_density_bounds: tuple
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    complete_correlated_face_parameterization_defined: bool = True
    rectangular_geometry_enclosure_is_wholly_material: bool = False
    implicit_root_midpoint_used: bool = False
    physical_receiving_traction_law_defined: bool = True
    physical_source_work_handover_certified: bool = False
    preparation_or_activation_work_paid: bool = False
    physical_material_admission_authorized: bool = False
    new_mixed_equilibrium_certified: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0


def evaluate_native_receiving_face(reference,cut,declaration,cold_system,hot_packet,hot_query_label,
        lambda_bounds,physical_radius_cubed_bounds_m3=None,*,cold_coordinates=None,precision=96):
    from .geometric_native_common_cold_cut_fields import (
        evaluate_native_common_cold_cut_field,native_common_cold_cut_field_to_dict,
    )
    _precision(precision);cut=validate_common_contact_cut(cut,reference=reference)
    _validate(declaration,NativeReceivingFaceDeclaration,_DECLARATIONS)
    if declaration.cut_snapshot_sha256!=cut.cut_snapshot_sha256:
        raise ValueError('The declared law belongs to another complete face')
    bodyindex,body,response,chart=_hot(hot_packet,hot_query_label,cut.receiver_component_id)
    bounds,du=_chart(reference,cut,lambda_bounds,physical_radius_cubed_bounds_m3,precision)
    cold=evaluate_native_common_cold_cut_field(cold_system,bounds,
        coordinate_increments=cold_coordinates,precision=precision)
    R=_fraction(body['radius_m']);s=_fraction(body['fixed_owner_scale'])
    omega=tuple(_fraction(x) for x in body['angular_rate_per_physical_volume_depth_m'])
    if R!=reference.radius_m or omega!=reference.angular_rate_per_physical_volume_depth_m:
        raise ValueError('The HOT receiving reference does not match the genuine source cut')
    depth=bounds.receiver_material_depth_bounds_m;U=bounds.source_unit_world_direction_bounds
    if not 0<=depth.lo<=depth.hi<=_fraction(body['areal_material_thickness_m']):
        raise ValueError('The paired HOT label lies outside its actual whole depth')
    r=interval_cbrt(I(R**3)-3*R**2*depth,precision)
    cols=tuple(rotate_joint_physical_depth(_axis(j),omega,depth,precision=precision,order=32) for j in range(3))
    Q=tuple(tuple(cols[j][i] for j in range(3)) for i in range(3))
    X=tuple(_round(s*r*x,precision) for x in _mv(Q,U))
    axial=_cross(tuple(I(x) for x in omega),U);coefficient=(I(R**3)-3*R**2*depth)/R**2
    original=tuple(tuple(I(int(i==j))-coefficient*axial[i]*U[j] for j in range(3)) for i in range(3))
    A0=tuple(tuple(_round(s*v,precision) for v in row) for row in _mm(Q,original))
    cert=chart['finite_mean_root_certificate'];h=_interval(cert['scalar_affine_amplitude_bounds_m'])
    coordinates=tuple(_fraction(x) for x in response['coordinate_increments'][6*bodyindex:6*bodyindex+6])
    J=tuple(tuple(_interval(v) for v in row) for row in chart['implicit_physical_coordinate_jacobian'])
    hs=tuple(tuple(_interval(v) for v in row) for row in chart['implicit_scalar_coordinate_hessian'])
    values,first,second=_field_twojet(X,A0,R,h,coordinates[1:],J,hs,precision)
    hotindices=tuple(range(3*body['owner'],3*body['owner']+3))+tuple(range(6+6*bodyindex,12+6*bodyindex))
    coldindices=tuple(36+i for i,mode in enumerate(cold.mode_displacement_columns) if any(v!=I(0) for v in mode))
    indices=tuple(sorted(set(range(6))|set(hotindices)|set(coldindices)))
    hotjets=_embed(values,first,second,hotindices,indices,precision);coldjets=_cold_twojet(cold,indices,precision)
    # Pull the genuine common reference normal back into HOT's original-world
    # input. Its scale cancels on unit normalization; no cold input is substituted.
    material=_mv(tuple(zip(*A0)),bounds.receiver_inward_unit_normal_bounds)
    norm=interval_sqrt(sum((v*v for v in material),I(0)),precision+16)
    if norm.lo<=0:raise ValueError('The original receiving material normal lost regularity')
    material=tuple(_round(v/norm,precision) for v in material)
    A=tuple(tuple(hotjets[3+3*i+j] for j in range(3)) for i in range(3))
    cofactor=(_cross(A[1],A[2]),_cross(A[2],A[0]),_cross(A[0],A[1]))
    raw=tuple(sum((row[j]*material[j] for j in range(3)),0) for row in cofactor)
    magnitude=sum((v*v for v in raw),0).sqrt().inverse()
    normal=tuple(v*magnitude for v in raw)
    gap=sum(((hotjets[i]-coldjets[i])*normal[i] for i in range(3)),0)-declaration.fixed_rest_gap_m
    potential,traction,tangent,branch=_unilateral(gap,declaration.normal_stiffness_pa_m,precision)
    density=_round(bounds.area_density_bounds_m2_per_parameter_per_material_m*du,precision)
    potential=potential*density
    # Roof-aware differential between the two original material Cartesian inputs.
    bank=next(b for b in cold_system.banks if b.original_parent_id==cut.donor_original_parent_id)
    q=cold.original_unit_direction_bounds;r0=cold.original_reference_radius_bounds_m
    tangential=r/r0;radial=r0*r0/(r*r)
    C=tuple(tuple(tangential*int(i==j)+(radial-tangential)*q[i]*q[j] for j in range(3)) for i in range(3))
    bridge=tuple(tuple(_round(v,precision) for v in row) for row in _mm(bank.material_owner_pose_bounds,C))
    inverse=_inverse(bridge,precision)
    worldA=_mm(cold.original_cartesian_gradient_bounds,inverse)
    result=NativeReceivingFacePoint(declaration.binding_sha256,SOURCE_SEAL,hot_query_label,
        cold.bound_coordinate_increments,lambda_bounds,cut.maximum_ratio_time_fraction_bounds,
        'u=u_m+(1-u_m)*lambda; u_m is the exact unique source implicit root',
        encoded(common_contact_cut_bounds_to_dict(bounds)).decode(),
        encoded(native_common_cold_cut_field_to_dict(cold)).decode(),encoded(body).decode(),encoded(chart).decode(),
        bridge,inverse,worldA,X,A0,material,indices,values,first,second,tuple(v.v for v in normal),
        gap.v,gap.d,gap.h,traction,tangent,branch,density,potential.v,potential.d,potential.h,precision,'')
    return _remember(result,_POINTS)


def native_receiving_face_point_to_dict(point):
    _validate(point,NativeReceivingFacePoint,_POINTS)
    return dict(format='native-curved-receiving-face-material-operator-1',**_payload(point))


def native_receiving_face_point_from_dict(raw,*,reference,cut,declaration,cold_system,hot_packet):
    from .geometric_native_common_cold_material_carrier import _saved_json_kinds
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get('format')!='native-curved-receiving-face-material-operator-1':
        raise ValueError('A complete literal receiving material/operator DTO is required')
    point=evaluate_native_receiving_face(reference,cut,declaration,cold_system,hot_packet,raw['hot_query_label'],
        _interval(raw['lambda_bounds']),_interval(json.loads(raw['cut_bounds_json'])['common_physical_radius_cubed_bounds_m3']),
        cold_coordinates=tuple(_fraction(v) for v in raw['cold_coordinates_m']),precision=raw['precision'])
    if encoded(native_receiving_face_point_to_dict(point))!=encoded(raw):
        raise ValueError('Receiving face restart changed its complete material/operator binding')
    return point


def integrate_native_receiving_face(reference,cut,declaration,cold_system,hot_packet,hot_query_label,
        *,cold_coordinates=None,precision=96):
    """Whole-face range integral including the implicit start and full depth.

    This conservative enclosure is not a tight-force certificate. A closure
    crossing returns a semismooth tangent hull, never a unique classical Hessian.
    """
    point=evaluate_native_receiving_face(reference,cut,declaration,cold_system,hot_packet,hot_query_label,
        I(0,1),cold_coordinates=cold_coordinates,precision=precision)
    bounds=json.loads(point.cut_bounds_json);depth=_interval(bounds['receiver_material_depth_bounds_m'])
    width=depth.hi-depth.lo
    if width<=0:raise ValueError('A complete receiving face needs positive common material depth')
    indices=point.active_coordinate_indices
    gradient=[I(0)]*100;hessian=[[I(0)]*100 for _ in range(100)]
    for a,i in enumerate(indices):
        gradient[i]=_round(point.gradient_density_bounds[a]*width,precision)
        for b,j in enumerate(indices):hessian[i][j]=_round(point.hessian_density_bounds[a][b]*width,precision)
    return dict(scope=SCOPE,point=native_receiving_face_point_to_dict(point),
        energy_bounds_j=_payload(_round(point.energy_density_bounds_j_per_lambda_per_m*width,precision)),
        gradient_bounds=_payload(tuple(gradient)),hessian_bounds=_payload(tuple(tuple(row) for row in hessian)),
        integration_method='WHOLE_CORRELATED_LAMBDA_AND_DEPTH_OUTWARD_RANGE',
        positive_correction_boundary_used_once=True,complete_curved_face_covered=True,
        source_root_band_omitted=False,implicit_root_midpoint_used=False,
        extra_elastic_column_energy_added=False,old_source_atoms_added_here=False,
        classical_hessian_unique=point.branch!='closure_crossing_semismooth_hull',
        semismooth_tangent_hull_retained=True,force_precision_sufficient_for_mixed_solve=False,
        physical_receiving_traction_law_defined=True,preparation_or_activation_work_paid=False,
        physical_source_work_handover_certified=False,physical_material_admission_authorized=False,
        new_mixed_equilibrium_certified=False,clocks_advanced=False,accepted_new_native_intervals=0)

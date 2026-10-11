"""Authentic HOT receiving-field jets and work-conjugate local pullback.

These ports retain the finite-mean chart curvature and material-gradient
variations needed by a subsequent paired COLD--HOT interface operator.
Saved polygon labels are not the complete curved receiving face. No new
contact identity, traction law, elapsed interval or physical admission is
inferred from a point port.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I
from .geometric_native_joint_spherical_field import rotate_joint_physical_depth
from .geometric_native_common_carrier_source_work import _interval, _fraction, _payload, _round, _typed_guard

SOURCE_SEAL = 'c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c'
SCOPE = 'AUTHENTIC_HOT_MATERIAL_POINT_TWOJETS_AND_LOCAL_WORK_PULLBACK_ONLY'
GENERATORS = (
    ((1,0,0),(0,1,0),(0,0,1)),
    ((1,0,0),(0,-1,0),(0,0,0)),
    ((1,0,0),(0,1,0),(0,0,-2)),
    ((0,1,0),(1,0,0),(0,0,0)),
    ((0,0,1),(0,0,0),(1,0,0)),
    ((0,0,0),(0,0,1),(0,1,0)),
)
_GUARDS = {}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _body(value):
    return {f.name:getattr(value,f.name) for f in fields(value) if f.name != 'binding_sha256'}


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError('Receiving-field precision is a literal integer in [96,512]')


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _mv(A,x):
    return tuple(sum((a*v for a,v in zip(row,x)),I(0)) for row in A)


def _mm(A,B):
    return tuple(tuple(sum((A[i][k]*B[k][j] for k in range(3)),I(0)) for j in range(3)) for i in range(3))


def _det(A):
    return sum((A[0][i]*_cross(A[1],A[2])[i] for i in range(3)),I(0))


def _axis(i):
    return tuple(I(int(i==j)) for j in range(3))


def _field_twojet(X,A0,R,h,b,J,hs,precision):
    """Twelve spatial channels against owner3 + finite-mean body6.

    A0 takes ORIGINAL_SOURCE_WORLD_CARTESIAN input. Native polygon pose
    is already in X/A0; source owner history is not another rotation.
    """
    deformation=tuple(tuple(I(int(i==j))+sum((GENERATORS[k][i][j]*a/R
        for k,a in enumerate((h,)+b)),I(0)) for j in range(3)) for i in range(3))
    valueX=_mv(deformation,X);valueA=_mm(deformation,A0)
    matrices=tuple(tuple(tuple(sum((GENERATORS[k][i][j]*J[k][a]/R
        for k in range(6)),I(0)) for j in range(3)) for i in range(3)) for a in range(6))
    firstX=tuple(_mv(M,X) for M in matrices);firstA=tuple(_mm(M,A0) for M in matrices)
    values=valueX+tuple(v for row in valueA for v in row)
    first=[];second=[]
    for channel in range(12):
        vector=valueX if channel<3 else tuple(valueA[i][(channel-3)%3] for i in range(3))
        component=channel if channel<3 else (channel-3)//3
        bodyvectors=firstX if channel<3 else tuple(tuple(M[i][(channel-3)%3] for i in range(3)) for M in firstA)
        reference=X[component] if channel<3 else A0[component][(channel-3)%3]
        owners=tuple(_cross(_axis(a),vector)[component] for a in range(3))
        first.append(owners+tuple(v[component] for v in bodyvectors))
        matrix=[]
        for a in range(9):
            row=[]
            for c in range(9):
                if a<3 and c<3:
                    result=(_cross(_axis(a),_cross(_axis(c),vector))[component]
                        +_cross(_axis(c),_cross(_axis(a),vector))[component])/2
                elif a<3 or c<3:
                    owner=a if a<3 else c;body=(c if a<3 else a)-3
                    result=_cross(_axis(owner),bodyvectors[body])[component]
                else:
                    result=reference*hs[a-3][c-3]/R
                row.append(_round(result,precision))
            matrix.append(tuple(row))
        second.append(tuple(matrix))
    return (tuple(_round(v,precision) for v in values),
        tuple(tuple(_round(v,precision) for v in row) for row in first),tuple(second))


@dataclass(frozen=True,slots=True)
class NativeHotReceivingPoint:
    source_packet_sha256: str
    source_operator_response_binding_sha256: str
    source_point_descriptor_json: str
    source_body_json: str
    source_chart_json: str
    query_label: str
    component_id: str
    original_parent_id: str
    material_id: str
    owner: int
    active_coordinate_indices: tuple
    coordinate_units: tuple
    body_coordinate_increments_m: tuple
    prepared_position_bounds_m: tuple
    prepared_original_world_gradient_bounds: tuple
    spatial_channels: tuple
    spatial_channel_units: tuple
    spatial_values: tuple
    spatial_first: tuple
    spatial_second: tuple
    positive_material_jacobian_bounds: I
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    source_polygon_label_authenticated: bool = True
    finite_mean_chart_curvature_retained: bool = True
    complete_owner_body_cross_variations_retained: bool = True
    source_polygon_pose_applied_once: bool = True
    original_input_frame: str = 'ORIGINAL_SOURCE_WORLD_CARTESIAN'
    complete_curved_receiving_face_covered: bool = False
    physical_receiving_traction_law_defined: bool = False
    physical_source_work_handover_certified: bool = False
    physical_material_admission_authorized: bool = False
    new_mixed_equilibrium_certified: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0


def evaluate_native_hot_receiving_point(packet,query_label,point_index,*,precision=96):
    _precision(precision)
    if type(packet) is not dict or packet.get('packet_sha256')!=SOURCE_SEAL or _sha({k:v for k,v in packet.items() if k!='packet_sha256'})!=SOURCE_SEAL:
        raise ValueError('The complete immutable HOT903 material/chart source is required')
    if type(query_label) is not str or type(point_index) is not int or not 0<=point_index<15:
        raise ValueError('An exact saved query label and genuine source point index are required')
    queries=[q for q in packet['evaluation_queries'] if q['label']==query_label]
    if len(queries)!=1:
        raise ValueError('Receiving fields require exactly one authentic saved finite-mean chart')
    descriptor=packet['material_point_queries'][point_index]
    bodies=packet['hot_mean_system']['bodies']
    index=next(i for i,b in enumerate(bodies) if b['component_id']==descriptor['component_id'])
    body=bodies[index];response=queries[0]['response']
    chart=json.loads(response['body_responses_json'][index])
    cert=chart['finite_mean_root_certificate']
    if (chart['component_id']!=body['component_id'] or cert['component_id']!=body['component_id']
            or descriptor['actual_source_polygon_label'] is not True
            or descriptor['containing_chart_surrogate_used'] is not False
            or cert['whole_mask_unique_root_certified'] is not True
            or cert['source_polygon_pose_applied_once'] is not True
            or cert['additional_owner_rotation_applied'] is not False):
        raise ValueError('The true original label/chart/body correspondence changed')
    R=_fraction(body['radius_m']);scale=_fraction(body['fixed_owner_scale'])
    depth=_fraction(descriptor['material_depth_m'])
    ray=tuple(_interval(x) for x in descriptor['original_unit_ray_bounds'])
    X=tuple(_interval(x) for x in descriptor['prepared_position_bounds_m'])
    omega=tuple(_fraction(x) for x in body['angular_rate_per_physical_volume_depth_m'])
    cols=tuple(rotate_joint_physical_depth(_axis(j),omega,I(depth),precision=precision,order=32) for j in range(3))
    rotation=tuple(tuple(cols[j][i] for j in range(3)) for i in range(3))
    axial=_cross(tuple(I(x) for x in omega),ray)
    coefficient=(R**3-3*R**2*depth)/R**2
    original=tuple(tuple(I(int(i==j))-coefficient*axial[i]*ray[j] for j in range(3)) for i in range(3))
    A0=tuple(tuple(_round(scale*x,precision) for x in row) for row in _mm(rotation,original))
    h=_interval(cert['scalar_affine_amplitude_bounds_m'])
    coordinates=tuple(_fraction(x) for x in response['coordinate_increments'][6*index:6*index+6])
    J=tuple(tuple(_interval(x) for x in row) for row in chart['implicit_physical_coordinate_jacobian'])
    hs=tuple(tuple(_interval(x) for x in row) for row in chart['implicit_scalar_coordinate_hessian'])
    values,first,second=_field_twojet(X,A0,R,h,coordinates[1:],J,hs,precision)
    eig=_interval(cert['whole_bracket_deformation_eigenvalue_bounds'])
    jac=I(scale**3*eig.lo**3,scale**3*eig.hi**3)
    if jac.lo<=0:raise ValueError('A HOT receiving field requires the full positive material Jacobian theorem')
    reference_det=_det(A0)
    current_det=_det(tuple(tuple(values[3+3*i+j] for j in range(3)) for i in range(3)))
    if not reference_det.lo<=scale**3<=reference_det.hi or current_det.hi<jac.lo or current_det.lo>jac.hi:
        raise ValueError('The actual original-world derivative contradicts its inherited material Jacobian theorem')
    owner=body['owner']
    result=NativeHotReceivingPoint(SOURCE_SEAL,response['binding_sha256'],encoded(descriptor).decode(),
        encoded(body).decode(),encoded(chart).decode(),query_label,body['component_id'],body['original_parent_id'],
        body['material_id'],owner,tuple(range(3*owner,3*owner+3))+tuple(range(6+6*index,12+6*index)),
        ('rad',)*3+('m',)*6,coordinates,X,A0,
        ('x','y','z')+tuple('A'+str(i)+str(j) for i in range(3) for j in range(3)),
        ('m',)*3+('1',)*9,values,first,second,jac,precision,'')
    result=replace(result,binding_sha256=_sha(_payload(_body(result))))
    _GUARDS[result.binding_sha256]=_typed_guard(result)
    return result


def native_hot_receiving_point_to_dict(point):
    if (type(point) is not NativeHotReceivingPoint or _sha(_payload(_body(point)))!=point.binding_sha256
            or _GUARDS.get(point.binding_sha256)!=_typed_guard(point)):
        raise ValueError('The receiving point changed its complete authentic typed binding')
    return dict(format='native-hot-receiving-material-point-1',**_payload(point))


def native_hot_receiving_point_from_dict(raw,packet,*,query_label,point_index,precision=96):
    point=evaluate_native_hot_receiving_point(packet,query_label,point_index,precision=precision)
    if encoded(native_hot_receiving_point_to_dict(point))!=encoded(raw):
        raise ValueError('The receiving restart changed its full field, query or source lineage')
    return point


def pullback_native_hot_receiving_work(point,spatial_gradient,spatial_hessian,*,precision=96):
    """Pull back a supplied12-channel potential jet, never infer its law.

    Spatial gradients conjugate to x are N; those conjugate to A are N*m.
    The caller's material normal/orientation potential must supply its
    complete position/gradient tangent, including cross blocks.
    """
    native_hot_receiving_point_to_dict(point);_precision(precision)
    if (type(spatial_gradient) is not tuple or len(spatial_gradient)!=12
            or any(type(v) is not I for v in spatial_gradient)
            or type(spatial_hessian) is not tuple or len(spatial_hessian)!=12
            or any(type(row) is not tuple or len(row)!=12 or any(type(v) is not I for v in row) for row in spatial_hessian)):
        raise ValueError('The full typed12-channel supplied potential force and tangent are required')
    if any(spatial_hessian[i][j]!=spatial_hessian[j][i] for i in range(12) for j in range(12)):
        raise ValueError('The supplied potential tangent must retain symmetric mixed partials')
    G=tuple(_round(sum((spatial_gradient[i]*point.spatial_first[i][a] for i in range(12)),I(0)),precision) for a in range(9))
    H=tuple(tuple(_round(sum((spatial_gradient[i]*point.spatial_second[i][a][b] for i in range(12)),I(0))
        +sum((point.spatial_first[i][a]*spatial_hessian[i][j]*point.spatial_first[j][b]
            for i in range(12) for j in range(12)),I(0)),precision) for b in range(9)) for a in range(9))
    return dict(active_coordinate_indices=point.active_coordinate_indices,
        gradient_units=('N*m',)*3+('N',)*6,gradient_bounds=G,hessian_bounds=H,
        force_times_chart_curvature_included=True,orientation_hyperstress_work_included=True,
        physical_receiving_traction_law_inferred=False,scope=SCOPE)

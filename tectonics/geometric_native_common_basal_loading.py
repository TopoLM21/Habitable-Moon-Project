"""State-dependent spatial mantle forcing on authenticated material trace lifts.

This is a NEW declared reference-source-cell discrete basal operator.  Every
velocity column is supplied by the registered material trace port.  It is not
a continuum base quadrature certificate, a mixed equilibrium, or a time step.
The prescribed angular field and the independent isotropic vector drag retain
their distinct meanings; no conservative spatial loading potential is used.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_common_source_carrier import _round
from .geometric_native_common_carrier_source_work import _payload


FORMAT = 'native-common-source-cell-basal-loading-system-1'
RESPONSE_FORMAT = 'native-common-source-cell-basal-loading-response-1'
SCOPE = 'NEW_DECLARED_REFERENCE_SOURCE_CELL_MATERIAL_TRACE_LIFT'
FIELD = 'FIXED_GLOBAL_ANGULAR_SPATIAL_SOURCE_WITH_FIXED_SOURCE_RADIUS'
DRAG = 'INDEPENDENT_ISOTROPIC_FULL_VECTOR_LINEAR_DRAG'
_SYSTEMS, _RESPONSES, _CONTEXTS, _EVALUATIONS = {}, {}, {}, {}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _body(value):
    return {field.name: getattr(value, field.name) for field in fields(value)
            if field.name != 'binding_sha256'}


def _seal(value):
    return hashlib.sha256(_json(_payload(_body(value))).encode()).hexdigest()


def _remember(value, registry):
    value = replace(value, binding_sha256=_seal(value))
    previous = registry.get(value.binding_sha256)
    if previous is not None:
        return _guard(previous, type(value), registry)
    registry[value.binding_sha256] = value
    return value


def _guard(value, cls, registry):
    if (type(value) is not cls or registry.get(value.binding_sha256) is not value
            or _seal(value) != value.binding_sha256):
        raise ValueError('An altered, resealed or unregistered basal loading is not authorized')
    return value


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError('Basal precision must be a literal integer from96 to512')
    return value


def _rates(value, size):
    if (type(value) is not tuple or len(value) != size
            or any(type(x) is not F for x in value)):
        raise ValueError('Every physical coordinate rate, including lambda, must be a literal Fraction')
    return value


def _vector(value, size=3):
    if (type(value) is not tuple or len(value) != size
            or any(type(x) is not I for x in value)):
        raise ValueError('Complete immutable outward interval vector required')
    return value


def _matrix(value, rows, cols):
    if type(value) is not tuple or len(value) != rows:
        raise ValueError('Complete immutable outward interval matrix required')
    return tuple(_vector(row, cols) for row in value)


def _square(value):
    return I(0 if value.lo <= 0 <= value.hi else min(value.lo**2,value.hi**2),
             max(value.lo**2,value.hi**2))


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)
                if (x.lo != 0 or x.hi != 0) and (y.lo != 0 or y.hi != 0)), I(0))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _map(a,x):
    return tuple(_dot(row,x) for row in a)


def _intersection(a,b):
    if a.hi < b.lo or b.hi < a.lo:
        raise ValueError('Independent exact spatial and generalized power enclosures disagree')
    return I(max(a.lo,b.lo),min(a.hi,b.hi))


def _source(source):
    """Pure algebra uses literal source coefficients, never binary64 values."""
    if type(source) is not dict or set(source) != {
            'radius_m','basal_drag_pa_s_m','angular_velocity_rad_s',
            'amplitude_constant','amplitude_gradient'}:
        raise ValueError('Complete literal spatial source coefficients required')
    if (type(source['radius_m']) is not F or source['radius_m'] <= 0
            or type(source['basal_drag_pa_s_m']) is not F
            or source['basal_drag_pa_s_m'] <= 0
            or type(source['amplitude_constant']) is not F):
        raise ValueError('The fixed source radius and independent drag must be positive literal Fractions')
    for key in ('angular_velocity_rad_s','amplitude_gradient'):
        if (type(source[key]) is not tuple or len(source[key]) != 3
                or any(type(x) is not F for x in source[key])):
            raise ValueError('All literal source vector components are required')
    # This guarantees positive driving amplitude over the entire unit sphere.
    a0=source['amplitude_constant']; b=source['amplitude_gradient']
    if a0 <= 0 or a0*a0 <= sum((x*x for x in b),F(0)):
        raise ValueError('The authentic prescribed amplitude must remain positive on the full sphere')
    return source


def _spatial_source_twojet(source,position,*,precision=96):
    """Driving and its nonsymmetric Cartesian derivative, with fixed source R."""
    _source(source); _precision(precision); x=_vector(position)
    r=interval_sqrt(sum((_square(v) for v in x),I(0)),precision+16)
    if r.lo <= 0:
        raise ValueError('A material basal trace must be separated from the Cartesian origin')
    q=tuple(_round(v/r,precision) for v in x)
    omega=tuple(I(v) for v in source['angular_velocity_rad_s'])
    b=tuple(I(v) for v in source['amplitude_gradient'])
    a=I(source['amplitude_constant'])+_dot(b,q)
    wq=_cross(omega,q)
    W=tuple(tuple(_cross(omega,tuple(I(int(k==j)) for k in range(3)))[i]
                  for j in range(3)) for i in range(3))
    scale=source['basal_drag_pa_s_m']*source['radius_m']
    traction=tuple(_round(scale*a*v,precision) for v in wq)
    P=tuple(tuple(I(int(i==j))-q[i]*q[j] for j in range(3)) for i in range(3))
    derivative=tuple(tuple(_round(scale/r*sum(
        ((wq[i]*b[l]+a*W[i][l])*P[l][j] for l in range(3)),I(0)),precision)
        for j in range(3)) for i in range(3))
    return traction,derivative,dict(position_bounds_m=_payload(x),
        current_radius_bounds_m=_payload(r),current_unit_direction_bounds=_payload(q),
        amplitude_bounds=_payload(a),prescribed_driving_traction_bounds_pa=_payload(traction),
        prescribed_spatial_derivative_bounds_pa_m=_payload(derivative),
        fixed_source_radius_m=_payload(source['radius_m']),
        amplitude_multiplies_drag=False,spatial_derivative_assumed_symmetric=False)


def _record(record,size):
    if type(record) is not dict or not {'position_bounds_m','global_coordinate_indices',
            'position_first_bounds','position_second_bounds','source_weight_bounds_m2'} <= set(record):
        raise ValueError('A complete immutable material trace record is required')
    x=_vector(record['position_bounds_m']); ids=record['global_coordinate_indices']
    if (type(ids) is not tuple or not ids or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0<=i<size for i in ids)):
        raise ValueError('All distinct global material trace coordinate indices are required')
    first=_matrix(record['position_first_bounds'],3,len(ids))
    rawsecond=record['position_second_bounds']
    if type(rawsecond) is not tuple or len(rawsecond) != 3:
        raise ValueError('Complete material position curvature is required')
    second=tuple(_matrix(row,len(ids),len(ids)) for row in rawsecond)
    if any(row[i][j] != row[j][i] for row in second for i in range(len(ids)) for j in range(i)):
        raise ValueError('Material position curvature must retain its symmetric coordinate twojet')
    weight=record['source_weight_bounds_m2']
    if type(weight) is not I or weight.lo <= 0:
        raise ValueError('Each actual source material trace requires a strictly positive fixed measure enclosure')
    return x,ids,first,second,weight


def _trace_algebra(source,records,rates,*,size=101,precision=96):
    """Generic diagnostic algebra; only the registered public port authorizes it.

    The state tangent is accumulated directly at the explicitly supplied
    physical rates. No cubic drag-state tensor is built or saved.
    """
    _source(source);_precision(precision);_rates(rates,size)
    if type(size) is not int or size <= 0 or type(records) is not tuple or not records:
        raise ValueError('A complete nonempty discrete trace collection is required')
    L=[I(0) for _ in range(size)];D=[[I(0) for _ in range(size)] for _ in range(size)]
    Q=[I(0) for _ in range(size)];K=[[I(0) for _ in range(size)] for _ in range(size)]
    directsource=I(0);directdiss=I(0);directnet=I(0);receipts=[]
    beta=source['basal_drag_pa_s_m']
    for record in records:
        x,ids,J,H,weight=_record(record,size)
        traction,dtraction,receipt=_spatial_source_twojet(source,x,precision=precision)
        cols=tuple(tuple(J[a][i] for a in range(3)) for i in range(len(ids)))
        active=tuple(i for i in range(len(ids)) if any(v.lo or v.hi for v in cols[i])
            or any(H[a][i][k].lo or H[a][i][k].hi for a in range(3) for k in range(len(ids))))
        # An omitted exact zero column has an exact zero curvature row too.
        # This sparsity is proved by the authenticated field, not assumed for
        # any missing load. The full original index list remains in its receipt.
        u=tuple(_round(sum((J[a][j]*rates[index] for j,index in enumerate(ids) if rates[index]),I(0)),precision)
                for a in range(3))
        slip=tuple(_round(t-beta*v,precision) for t,v in zip(traction,u))
        localL=tuple(_round(weight*_dot(col,traction),precision) for col in cols)
        localQ=tuple(_round(weight*_dot(col,slip),precision) for col in cols)
        # These depend on k, not on the projected force row i. Computing
        # their contractions once avoids a hidden cubic loop in H*rate.
        forcechanges={}
        for k in active:
            duk=tuple(_round(sum((H[a][j][k]*rates[indexj]
                for j,indexj in enumerate(ids) if rates[indexj]),I(0)),precision) for a in range(3))
            dt=_map(dtraction,cols[k])
            forcechanges[k]=tuple(d-beta*du for d,du in zip(dt,duk))
        for i in active:
            index=ids[i]
            L[index]=_round(L[index]+localL[i],precision)
            Q[index]=_round(Q[index]+localQ[i],precision)
            for j in active:
                other=ids[j]
                if j>=i:
                    value=_round(beta*weight*_dot(cols[i],cols[j]),precision)
                    D[index][other]=_round(D[index][other]+value,precision)
                    if other != index:
                        D[other][index]=D[index][other]
            for k in active:
                other=ids[k]
                hik=tuple(H[a][i][k] for a in range(3))
                forcechange=forcechanges[k]
                value=_round(weight*(_dot(hik,slip)+_dot(cols[i],forcechange)),precision)
                K[index][other]=_round(K[index][other]+value,precision)
        ps=_round(weight*_dot(traction,u),precision)
        pd=_round(beta*weight*sum((_square(v) for v in u),I(0)),precision)
        pn=_round(weight*_dot(slip,u),precision)
        directsource=_round(directsource+ps,precision)
        directdiss=_round(directdiss+pd,precision)
        directnet=_round(directnet+pn,precision)
        receipt.update(trace_id=record.get('trace_id'),global_coordinate_indices=ids,
            trace_record_sha256=hashlib.sha256(_json(_payload(record)).encode()).hexdigest(),
            source_material_cell_index=record.get('source_material_cell_index'),
            source_binding_sha256=record.get('source_binding_sha256'),
            source_weight_bounds_m2=_payload(weight),actual_material_velocity_bounds_m_s=_payload(u),
            net_basal_traction_bounds_pa=_payload(slip),source_power_bounds_w=_payload(ps),
            drag_dissipation_bounds_w=_payload(pd),net_power_bounds_w=_payload(pn),
            fixed_source_measure_used_once=True,source_area_has_state_derivatives=False,
            gram_drag_positive_semidefinite_by_positive_measure=True)
        receipts.append(_json(_payload(receipt)))
    generalizedsource=_round(sum((L[i]*rates[i] for i in range(size) if rates[i]),I(0)),precision)
    generalizeddiss=_round(sum((rates[i]*D[i][j]*rates[j]
        for i in range(size) if rates[i] for j in range(size) if rates[j]),I(0)),precision)
    generalizednet=_round(sum((Q[i]*rates[i] for i in range(size) if rates[i]),I(0)),precision)
    psource=_intersection(directsource,generalizedsource)
    pdiss=_intersection(directdiss,generalizeddiss)
    pnet=_intersection(_intersection(directnet,generalizednet),psource-pdiss)
    if pdiss.lo < 0:
        raise ValueError('The independent positive-measure vector drag failed its dissipation proof')
    # A second, independently contracted net force must enclose each directly
    # projected slip force. This is a check, not a replacement of its enclosure.
    contractedQ=tuple(_round(L[i]-sum((D[i][j]*rates[j] for j in range(size) if rates[j]),I(0)),precision)
                      for i in range(size))
    residual=tuple(_intersection(Q[i],contractedQ[i]) for i in range(size))
    return dict(prescribed_generalized_force_bounds=tuple(L),drag_bounds=tuple(tuple(row) for row in D),
        residual_bounds=residual,state_tangent_bounds=tuple(tuple(row) for row in K),
        velocity_tangent_bounds=tuple(tuple(-v for v in row) for row in D),
        source_power_bounds_w=psource,drag_dissipation_bounds_w=pdiss,net_power_bounds_w=pnet,
        independently_projected_source_power_bounds_w=directsource,
        independently_projected_drag_dissipation_bounds_w=directdiss,
        independently_projected_net_power_bounds_w=directnet,
        generalized_source_power_bounds_w=generalizedsource,
        generalized_drag_dissipation_bounds_w=generalizeddiss,
        generalized_net_power_bounds_w=generalizednet,
        per_trace_receipts_json=tuple(receipts))


@dataclass(frozen=True,slots=True)
class NativeCommonBasalLoadingSystem:
    trace_system_binding_sha256: str
    coordinate_order: tuple
    coordinate_units: tuple
    coordinate_rate_units: tuple
    generalized_force_units: tuple
    source_radius_m: F
    basal_drag_pa_s_m: F
    mantle_angular_velocity_rad_s: tuple
    prescribed_amplitude_constant: F
    prescribed_amplitude_gradient: tuple
    complete903_source_binding_sha256: str
    authenticated_internal_contact_arm_only_coordinate_indices: tuple
    binding_sha256: str
    scope: str = SCOPE
    spatial_source_model: str = FIELD
    drag_model: str = DRAG
    discrete_basal_coordinate_count: int = 101
    complete_declared_discrete_basal_operator_defined: bool = True
    continuum_whole_base_quadrature_certified: bool = False
    physical_current_area_traction_law_certified: bool = False
    general_state_full_physics_operator_certified: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    physical_material_admission_authorized: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    full_system_temporal_path_work_certified: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0
    owner_query_zero_current_derivatives_only: bool = True
    general_finite_owner_state_chart_defined: bool = False


@dataclass(frozen=True,slots=True)
class NativeCommonBasalLoadingResponse:
    system_binding_sha256: str
    trace_evaluation_binding_sha256: str
    coordinate_values: tuple
    coordinate_rates: tuple
    coordinate_order: tuple
    prescribed_generalized_force_bounds: tuple
    drag_bounds: tuple
    residual_bounds: tuple
    state_tangent_bounds: tuple
    velocity_tangent_bounds: tuple
    source_power_bounds_w: I
    drag_dissipation_bounds_w: I
    net_power_bounds_w: I
    independently_projected_source_power_bounds_w: I
    independently_projected_drag_dissipation_bounds_w: I
    independently_projected_net_power_bounds_w: I
    generalized_source_power_bounds_w: I
    generalized_drag_dissipation_bounds_w: I
    generalized_net_power_bounds_w: I
    per_trace_receipts_json: tuple
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    complete_declared_discrete_basal_operator_computed: bool = True
    discrete_basal_coordinate_count: int = 101
    drag_positive_semidefinite_by_same_field_gram_identity: bool = True
    spatial_and_generalized_power_enclosures_overlap: bool = True
    state_tangent_assumed_symmetric: bool = False
    conservative_spatial_loading_potential_used: bool = False
    physical_coordinate_rates_including_lambda_explicit: bool = True
    physical_time_schedule_inferred_from_calibration: bool = False
    internal_contact_arm_only_basal_columns_exact_zero: bool = True
    continuum_whole_base_quadrature_certified: bool = False
    general_state_full_physics_operator_certified: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    physical_material_admission_authorized: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    full_system_temporal_path_work_certified: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0
    owner_query_zero_current_derivatives_only: bool = True
    general_finite_owner_state_chart_defined: bool = False


def _trace_api():
    from . import geometric_native_common_basal_trace as trace
    return trace


def bind_native_common_basal_loading(trace_system):
    """Bind only the sibling port's authenticated complete source/catalogue."""
    trace=_trace_api()
    context=trace.native_common_basal_trace_loading_context(trace_system)
    source=_source(context['source'])
    order=context['coordinate_order'];units=context['coordinate_units']
    if (type(order) is not tuple or len(order)!=101 or len(set(order))!=101
            or type(units) is not tuple or len(units)!=101
            or units!=('rad',)*6+('m',)*94+('1',)):
        raise ValueError('The entire authenticated original100 catalogue plus lambda is required')
    internal=context['internal_contact_arm_only_coordinate_indices']
    if internal!=tuple(range(92,100)):
        raise ValueError('The source contact-arm-only internal catalogue must be authenticated')
    rates=('rad/s',)*6+('m/s',)*94+('1/s',)
    forces=('N*m',)*6+('N',)*94+('J',)
    system=_remember(NativeCommonBasalLoadingSystem(trace_system.binding_sha256,order,units,rates,forces,
        source['radius_m'],source['basal_drag_pa_s_m'],source['angular_velocity_rad_s'],
        source['amplitude_constant'],source['amplitude_gradient'],
        context['complete903_source_binding_sha256'],internal,''),_SYSTEMS)
    _CONTEXTS[system.binding_sha256]=trace_system
    return system


def validate_native_common_basal_loading(system):
    system=_guard(system,NativeCommonBasalLoadingSystem,_SYSTEMS)
    _trace_api().native_common_basal_trace_loading_context(_CONTEXTS[system.binding_sha256])
    return system


def native_common_basal_loading_to_dict(system):
    system=validate_native_common_basal_loading(system)
    return dict(format=FORMAT,**_payload(system))


def native_common_basal_loading_from_dict(raw,trace_system):
    if type(raw) is not dict:
        raise ValueError('Complete basal loading system DTO required')
    expected=bind_native_common_basal_loading(trace_system)
    if raw!=native_common_basal_loading_to_dict(expected):
        raise ValueError('The complete basal loading restart changed its source, scope or catalogue')
    return expected


def evaluate_native_common_basal_loading(system,trace_evaluation,*,coordinate_rates,precision=96):
    system=validate_native_common_basal_loading(system);_precision(precision);_rates(coordinate_rates,101)
    trace_system=_CONTEXTS[system.binding_sha256]
    context=_trace_api().native_common_basal_trace_loading_context(trace_system,trace_evaluation)
    if context['coordinate_order'] != system.coordinate_order:
        raise ValueError('The trace changed its authenticated coordinate catalogue')
    key=(system.binding_sha256,trace_evaluation.binding_sha256,coordinate_rates,precision)
    if key in _EVALUATIONS:
        return _guard(_EVALUATIONS[key],NativeCommonBasalLoadingResponse,_RESPONSES)
    records=context['trace_records']
    for record in records:
        _,ids,_,_,_=_record(record,101)
        if set(ids)&set(system.authenticated_internal_contact_arm_only_coordinate_indices):
            raise ValueError('Contact-arm-only residuals cannot invent an original-body basal field')
    source=dict(radius_m=system.source_radius_m,basal_drag_pa_s_m=system.basal_drag_pa_s_m,
        angular_velocity_rad_s=system.mantle_angular_velocity_rad_s,
        amplitude_constant=system.prescribed_amplitude_constant,
        amplitude_gradient=system.prescribed_amplitude_gradient)
    result=_trace_algebra(source,records,coordinate_rates,size=101,precision=precision)
    response=_remember(NativeCommonBasalLoadingResponse(system.binding_sha256,trace_evaluation.binding_sha256,
        context['coordinate_values'],coordinate_rates,system.coordinate_order,**result,precision=precision,
        binding_sha256=''),_RESPONSES)
    _EVALUATIONS[key]=response
    return response


def native_common_basal_loading_response_to_dict(response):
    response=_guard(response,NativeCommonBasalLoadingResponse,_RESPONSES)
    return dict(format=RESPONSE_FORMAT,**_payload(response))


def native_common_basal_loading_response_from_dict(raw,*,system,trace_evaluation,
        expected_coordinate_rates,precision=96):
    if type(raw) is not dict:
        raise ValueError('Complete basal loading response DTO required')
    expected=evaluate_native_common_basal_loading(system,trace_evaluation,
        coordinate_rates=expected_coordinate_rates,precision=precision)
    if raw!=native_common_basal_loading_response_to_dict(expected):
        raise ValueError('The complete basal response differs from the exact expected query and rates')
    return expected

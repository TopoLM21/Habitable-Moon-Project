"""Rational source clipping enclosures and independent native-geometry replay.

Literal binary64 X+P and A-P are source conventions, not approximation credits.
All subsequent clipping arithmetic is replayed with integer nearest-even binary
rounding.  True sphere roots use integer-isqrt rational enclosures.  This bounds
errors of the represented chart; it does not bound an angular/operator integral.
No planetary, clipping, material, or operator producer is imported here.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np


FORMAT = 'genesis-source-geometry-operands-0.1'
SCOPE = 'source_geometry_error_enclosures_no_operator_angular_or_physical_certificate'
PRECISION_BITS = int(np.finfo(np.longdouble).nmant) + 1
MIN_SUBNORMAL_EXPONENT = int(np.finfo(np.longdouble).minexp) - PRECISION_BITS + 1
MAX_NORMAL_EXPONENT = int(np.finfo(np.longdouble).maxexp) - 1
EPSILON = Fraction(2) ** (1-PRECISION_BITS)


class GeometryBoundRefusal(ValueError):
    """A source event or native arithmetic convention remains unresolved."""


def require(condition, message):
    if not condition:
        raise GeometryBoundRefusal(message)


def rational(value):
    require(not isinstance(value, (bool, np.bool_)), 'Boolean source operand')
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        try:
            return Fraction(*value.as_integer_ratio())
        except (ValueError, OverflowError) as error:
            raise GeometryBoundRefusal('Nonfinite source operand') from error
    require(isinstance(value, str) and len(value) <= 4096, 'Invalid source scalar')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise GeometryBoundRefusal('Invalid rational source scalar') from error


def _pow2(exponent):
    return Fraction(2) ** exponent


def _floor_log2(value):
    require(value > 0, 'Positive operand required')
    estimate = value.numerator.bit_length() - value.denominator.bit_length()
    return estimate if value >= _pow2(estimate) else estimate-1


def _nearest_integer(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    comparison = 2*remainder-value.denominator
    return quotient + int(comparison > 0 or (comparison == 0 and quotient % 2))


def round_binary(value, *, bits=PRECISION_BITS, minimum_exponent=MIN_SUBNORMAL_EXPONENT,
                 maximum_exponent=MAX_NORMAL_EXPONENT):
    """IEEE nearest-even rounding by exact integer comparisons, no float math."""
    value = rational(value)
    require(type(bits) is int and 2 <= bits <= 256, 'Invalid significand precision')
    if not value:
        return Fraction()
    sign = 1 if value > 0 else -1
    magnitude = abs(value)
    exponent = _floor_log2(magnitude)
    require(exponent <= maximum_exponent, 'Native overflow unresolved')
    quantum = _pow2(max(exponent-bits+1, minimum_exponent))
    rounded = _nearest_integer(magnitude/quantum)*quantum
    require(not rounded or _floor_log2(rounded) <= maximum_exponent, 'Native overflow unresolved')
    return sign*rounded


def sqrt_enclosure(value, *, binary_places=256):
    """Enclose sqrt(q) with dyadic endpoints using integer sqrt(floor(q*2^k))."""
    value = rational(value)
    require(value >= 0, 'Negative rational square root')
    require(type(binary_places) is int and 32 <= binary_places <= 4096, 'Invalid root enclosure precision')
    if not value:
        return Fraction(), Fraction()
    # Relative dyadic scale also preserves arbitrarily small positive slivers.
    root_exponent = _floor_log2(value)//2
    quantum = _pow2(root_exponent-binary_places+1)
    scaled = value/(quantum*quantum)
    floor_root = math.isqrt(scaled.numerator//scaled.denominator)
    low = floor_root*quantum
    high = low if low*low == value else (floor_root+1)*quantum
    require(low*low <= value <= high*high, 'Integer-root containment failed')
    return low, high


def round_sqrt_binary(value):
    value = rational(value)
    require(value >= 0, 'Negative native square root')
    if not value:
        return Fraction()
    exponent = _floor_log2(value)//2
    quantum = _pow2(max(exponent-PRECISION_BITS+1, MIN_SUBNORMAL_EXPONENT))
    scaled = value/(quantum*quantum)
    floor_root = math.isqrt(scaled.numerator//scaled.denominator)
    midpoint_square = Fraction((2*floor_root+1)**2, 4)
    upper = scaled > midpoint_square or (scaled == midpoint_square and floor_root % 2)
    return (floor_root+int(upper))*quantum


def _promote_exact(value):
    """Replay the frozen source's guarded integer-to-native conversion exactly."""
    value = rational(value)
    if not value:
        return value
    numerator, denominator = abs(value.numerator), value.denominator
    keep = PRECISION_BITS+7  # frozen np.finfo(LD).nmant+8
    numerator_shift = max(numerator.bit_length()-keep, 0)
    denominator_shift = max(denominator.bit_length()-keep, 0)
    numerator_native = round_binary(Fraction(numerator >> numerator_shift))
    denominator_native = round_binary(Fraction(denominator >> denominator_shift))
    result = round_binary(numerator_native/denominator_native)
    result = round_binary(result*_pow2(numerator_shift-denominator_shift))
    require(result != 0, 'Positive exact source quantity lost to native underflow')
    return result if value > 0 else -result


def _dot(left, right, *, native=False):
    if not native:
        return sum((x*y for x,y in zip(left,right)), Fraction())
    result = Fraction()
    for x,y in zip(left,right):
        result = round_binary(result+round_binary(x*y))
    return result


def _array64(value, shape, name):
    raw = np.asarray(value)
    require(raw.shape == shape and raw.dtype.kind in 'ifu', name+' shape/type differs')
    result = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(result).all(), name+' is nonfinite')
    return result


def source_line(raw, N):
    require(type(raw) is dict, 'Source must be an object')
    section = raw.get('section', raw.get('geometry_source', raw))
    require(type(section) is dict, 'Source section must be an object')
    required = {'triangle_m','displacement_A_m','slope_P_m','normal_xyz','offset_m','lambda_bounds'}
    require(required <= section.keys(), 'Missing source geometry operands')
    require({'seed_center_world_m','seed_radius_m'} <= raw.keys(), 'Missing original sphere operands')
    X,A,P = (_array64(section[key], (3,3),key) for key in
             ('triangle_m','displacement_A_m','slope_P_m'))
    C,D = X+P,A-P  # mandatory original binary64 source arithmetic
    require(np.isfinite(C).all() and np.isfinite(D).all(), 'Source binary64 birth chart overflow')
    center_raw=np.asarray(raw['seed_center_world_m'])
    require(center_raw.shape==(3,) and center_raw.dtype.kind in 'ifu' and np.isfinite(center_raw).all(),
            'Sphere center shape/type differs')
    center = tuple(map(rational, center_raw))
    require(all(round_binary(q)==q for q in center), 'Sphere center not a literal native binary operand')
    normal = tuple(map(rational, _array64(section['normal_xyz'],(3,), 'Literal Gamma normal')))
    offset, radius = rational(section['offset_m']),rational(raw['seed_radius_m'])
    bounds = tuple(map(rational,_array64(section['lambda_bounds'],(2,), 'Original bounds')))
    nraw = np.asarray(N)
    require(nraw.shape == (3,), 'Original material N shape differs')
    n = tuple(map(rational,nraw))
    require(all(round_binary(q)==q for q in n), 'Original N is not a literal native binary operand')
    require(min(n)>=0 and abs(sum(n)-1) <= 128*rational(np.finfo(float).eps),
            'Original material N leaves source simplex')
    require(radius>0 and 0<bounds[0]<bounds[1], 'Invalid source sphere or bounds')
    cr = [tuple(map(rational,C[:,j])) for j in range(3)]
    dr = [tuple(map(rational,D[:,j])) for j in range(3)]
    u = tuple(_dot(n,row) for row in cr)
    d = tuple(_dot(n,row) for row in dr)
    v = tuple(x-y for x,y in zip(d,center))
    uu,uv,vv = _dot(u,u),_dot(u,v),_dot(v,v)
    require(uu>0, 'Collapsed source birth radial line')
    margin = radius*radius-vv+uv*uv/uu
    return dict(n=n, C=cr,D=dr,center=center,normal=normal,offset=offset,radius=radius,
                bounds=bounds,u=u,d=d,v=v,uu=uu,uv=uv,vv=vv,
                margin=margin,closest=-uv/uu,
                slope=_dot(normal,u),intercept=_dot(normal,d)-offset)


def _native_clip(line):
    n,center=line['n'],line['center']
    u=tuple(_dot(n,row,native=True) for row in line['C'])
    d=tuple(_dot(n,row,native=True) for row in line['D'])
    v=tuple(round_binary(x-y) for x,y in zip(d,center))
    uu,uv=_dot(u,u,native=True),_dot(u,v,native=True)
    require(uu>0, 'Native direction collapsed')
    closest=round_binary(-uv/uu)
    p=tuple(round_binary(vj+round_binary(closest*uj)) for vj,uj in zip(v,u))
    radius2=round_binary(line['radius']**2)
    p2=_dot(p,p,native=True)
    margin=round_binary(radius2-p2)
    line_scale=tuple(round_binary(_dot(tuple(abs(x) for x in n),tuple(abs(x) for x in row),native=True)+abs(c))
                     for row,c in zip(line['D'],center))
    u_scale=tuple(_dot(tuple(abs(x) for x in n),tuple(abs(x) for x in row),native=True) for row in line['C'])
    p_error=tuple(round_binary(32*EPSILON*round_binary(round_binary(a+round_binary(abs(closest)*b))+abs(c)))
                  for a,b,c in zip(line_scale,u_scale,p))
    margin_error=round_binary(round_binary(round_binary(8*EPSILON*round_binary(radius2+p2))+
                    round_binary(2*_dot(tuple(abs(x) for x in p),p_error,native=True)))+
                    _dot(p_error,p_error,native=True))
    fallback=abs(margin)<=margin_error
    if fallback:
        if line['margin']<=0:
            return None
        margin=_promote_exact(line['margin'])
        closest=_promote_exact(line['closest']) if line['closest'] else Fraction()
        p=tuple(_promote_exact(line['v'][j]+line['closest']*line['u'][j])
                if line['v'][j]+line['closest']*line['u'][j] else Fraction() for j in range(3))
    elif margin<0:
        return None
    half_sphere=round_sqrt_binary(round_binary(margin/uu))
    require(half_sphere>0, 'Native positive sphere halfwidth vanished')
    lower,upper=line['bounds']
    left=max(-half_sphere,round_binary(lower-closest))
    right=min(half_sphere,round_binary(upper-closest))
    if right<=left:
        return None
    half=round_binary(round_binary(right-left)/2)
    mid=round_binary(closest+round_binary(round_binary(left+right)/2))
    native_bounds=(round_binary(closest+left),round_binary(closest+right))
    relative=((tuple(_promote_exact(line['v'][j]+mid*line['u'][j])
                     if line['v'][j]+mid*line['u'][j] else Fraction() for j in range(3))),
              tuple(_promote_exact(half*uj) if half*uj else Fraction() for uj in line['u']))
    ep0=line['slope']*mid+line['intercept'];ep1=line['slope']*half
    phi=tuple(_promote_exact(v) if v else Fraction() for v in (ep0,ep1))
    expected=dict(lambda_mid=mid,lambda_halfwidth=half,lambda_bounds=native_bounds,
                  birth_relative_coefficients=relative,literal_phi_coefficients=phi,
                  relative_lambda_mid_to_original_lower=round_binary(mid-lower),
                  relative_lambda_mid_to_original_upper=round_binary(upper-mid),
                  gamma_root_arithmetic_estimate=Fraction())
    root=side=distance=left_width=right_width=None
    if phi[1]==0:
        side=Fraction(1,2) if phi[0]>0 else Fraction(-1,2) if phi[0]<0 else Fraction()
    else:
        eroot=-ep0/ep1
        root=_promote_exact(eroot) if eroot else Fraction()
        if eroot>=1:
            side=Fraction(-1,2) if phi[1]>0 else Fraction(1,2)
        elif eroot<=-1:
            side=Fraction(1,2) if phi[1]>0 else Fraction(-1,2)
        else:
            distance=_promote_exact(1-abs(eroot))
            left_width=_promote_exact(1+eroot)
            right_width=_promote_exact(1-eroot)
        expected['gamma_root_arithmetic_estimate']=_promote_exact(abs(eroot-root)) if eroot!=root else Fraction()
    expected.update(gamma_root_xi=root,constant_side=side,gamma_endpoint_distance_xi=distance,
                    gamma_left_width_xi=left_width,gamma_right_width_xi=right_width)
    return dict(expected=expected,exact_phi=(ep0,ep1),exact_gamma_xi=None if ep1==0 else -ep0/ep1,
                exact_sign_fallback_used=fallback,closest=closest,sphere_halfwidth=half_sphere,
                sphere_margin=margin,sphere_margin_estimate=margin_error)


def _source_interval(line, binary_places):
    if line['margin']<0:
        return None
    require(line['margin']!=0, 'Exact sphere tangency: positive-support certificate refused')
    lower,upper=line['bounds'];closest=line['closest']
    clamped=min(max(closest,lower),upper)
    minimum=line['uu']*clamped**2+2*line['uv']*clamped+line['vv']-line['radius']**2
    if minimum>0:
        return None
    require(minimum!=0, 'Exact original-bound sphere tangency: positive-support certificate refused')
    h0,h1=sqrt_enclosure(line['margin']/line['uu'],binary_places=binary_places)
    lo=(max(lower,closest-h1),max(lower,closest-h0))
    hi=(min(upper,closest+h0),min(upper,closest+h1))
    require(lo[1]<hi[0], 'Positive source sphere support unresolved at rational root precision')
    mid=((lo[0]+hi[0])/2,(lo[1]+hi[1])/2)
    half=((hi[0]-lo[1])/2,(hi[1]-lo[0])/2)
    return dict(low=lo,high=hi,mid=mid,half=half,sqrt=(h0,h1))


def _flat(value):
    if isinstance(value,(list,tuple)):
        return sum((_flat(v) for v in value), [])
    return [value]


def _native(witness,label):
    records=witness.get('native_exact_binary_ratio',{})
    require(type(records) is dict and label in records,'Missing native geometry witness: '+label)
    record=records[label]
    require(type(record) is dict and type(record.get('shape')) is list,'Native shape differs')
    shape=record['shape'];pairs=record.get('numerator_denominator')
    require(len(shape)<=3 and all(type(v) is int and 0<=v<=3 for v in shape),'Native geometry shape differs')
    require(type(pairs) is list and len(pairs)==math.prod(shape),'Native geometry operand count differs')
    values=[]
    for pair in pairs:
        require(type(pair) is list and len(pair)==2 and all(type(x) is str and len(x)<=4096 for x in pair),
                'Native geometry ratio differs')
        try:
            a,b=map(int,pair)
        except ValueError as error:
            raise GeometryBoundRefusal('Invalid native ratio') from error
        require(b>0,'Native denominator nonpositive')
        q=Fraction(a,b)
        require(round_binary(q)==q,'Native operand is not representable in declared significand precision')
        values.append(q)
    return values,shape


def _q(value):
    return str(value.numerator)+'/'+str(value.denominator)


def _enclosure(bounds):
    return dict(lower=_q(bounds[0]),upper=_q(bounds[1]))


def _error(native,truth):
    difference=(native-truth[1],native-truth[0])
    absolute_upper=max(abs(difference[0]),abs(difference[1]))
    absolute_lower=Fraction() if difference[0]<=0<=difference[1] else min(abs(difference[0]),abs(difference[1]))
    return dict(signed_error=_enclosure(difference),absolute_error_bound=_q(absolute_upper),
                absolute_error_lower=_q(absolute_lower),source_value_enclosure=_enclosure(truth),
                native_value=_q(native),bound_derived_from_same_component_source_operands=True,
                positive_scale_floor_used=False,other_component_error_borrowed=False)


def _affine_interval(constant,slope,bounds):
    ends=(constant+slope*bounds[0],constant+slope*bounds[1])
    return min(ends),max(ends)


def _divide_interval(numerator,denominator):
    require(denominator[0]>0 or denominator[1]<0,'Interval denominator includes zero')
    ends=[x/y for x in numerator for y in denominator]
    return min(ends),max(ends)


def audit_geometry_operands(raw,N,witness,*,binary_places=256):
    """Authenticate native chart and bound its error relative to source clipping.

    A shifted/resealed witness is refused unless it matches independently
    replayed original operations bit for bit.  Error bounds follow from exact
    source operands and integer-root containment, never from a requested pass
    tolerance.  Error propagation into operator components remains a next lane.
    """
    line=source_line(raw,N)
    true=_source_interval(line,binary_places)
    represented=_native_clip(line)
    interval=witness.get('interval')
    base=dict(format=FORMAT,scope=SCOPE,precision_bits=PRECISION_BITS,
              sqrt_enclosure_binary_places=binary_places,
              literal_binary64_birth_source_arithmetic_retained=True,
              source_history_reinitialized=False,source_geometry_accuracy_certified=False,
              operator_geometry_error_propagated=False,angular_accuracy_certified=False,
              physical_creation_admitted=False,physical_release_admitted=False,
              physical_contraction_certified=False,available_for_creation_funding=False,
              automated_goal_enabled=False,positive_scale_floor_used=False)
    if true is None or represented is None or interval is None:
        require(true is None and represented is None and interval is None,
                'Source/native positive support classification differs')
        return dict(base,status='PASS',empty=True,source_geometry_error_bounds_certified=True,
                    native_operation_replay_matches=True,support_classification_certified=True,
                    exact_source_margin_m2=_q(line['margin']))
    require(type(interval) is dict,'Saved geometry interval is not an object')
    comparisons={};failed=[]
    expected=represented['expected']
    required_shapes={'lambda_mid':[],'lambda_halfwidth':[],'lambda_bounds':[2],
        'birth_relative_coefficients':[2,3],'literal_phi_coefficients':[2],
        'relative_lambda_mid_to_original_lower':[],'relative_lambda_mid_to_original_upper':[],
        'gamma_root_arithmetic_estimate':[],'gamma_endpoint_distance_xi':[],
        'gamma_left_width_xi':[],'gamma_right_width_xi':[]}
    for name,prediction in expected.items():
        if name in ('gamma_root_xi','constant_side'):
            continue
        if prediction is None:
            require(interval.get(name) is None,'Unexpected native Gamma bank width: '+name)
            continue
        values,shape=_native(witness,'interval.'+name)
        require(shape==required_shapes[name],'Native geometry operand shape differs: '+name)
        predicted=_flat(prediction)
        public=np.asarray(interval.get(name),dtype=object)
        require(list(public.shape)==shape and
                [rational(q) for q in public.reshape(-1)]==[round_binary(q,bits=53,minimum_exponent=-1074,maximum_exponent=1023) for q in values],
                'Public/native geometry operand differs: '+name)
        passed=values==predicted
        comparisons[name]=dict(passed=passed,native=[_q(q) for q in values],
                              independently_replayed=[_q(q) for q in predicted])
        if not passed:
            failed.append(name)
    side_values,side_shape=_native(witness,'interval.side')
    require(side_shape==[1],'Native Gamma event shape differs')
    side_prediction=expected['gamma_root_xi'] if expected['gamma_root_xi'] is not None else expected['constant_side']
    comparisons['Gamma_event']=dict(passed=side_values==[side_prediction],
                native=[_q(q) for q in side_values],independently_replayed=[_q(side_prediction)])
    public_side=interval.get('gamma_root_xi') if expected['gamma_root_xi'] is not None else interval.get('constant_side')
    require(public_side is not None and rational(public_side)==round_binary(side_values[0],bits=53,minimum_exponent=-1074,maximum_exponent=1023),
            'Public/native Gamma event differs')
    if side_values!=[side_prediction]:
        failed.append('Gamma_event')
    require(not failed,'Native geometry operation replay differs: '+', '.join(failed))
    nvals,nshape=_native(witness,'interval.barycentric')
    require(nshape==[3] and tuple(nvals)==line['n'],'Native original barycentric source differs')
    mid,half=expected['lambda_mid'],expected['lambda_halfwidth']
    require(half>0,'Native chart has nonpositive width')
    errors=dict(lambda_mid=_error(mid,true['mid']),lambda_halfwidth=_error(half,true['half']),
                represented_lower_endpoint=_error(mid-half,true['low']),
                represented_upper_endpoint=_error(mid+half,true['high']))
    errors['archived_lower_endpoint']=_error(expected['lambda_bounds'][0],true['low'])
    errors['archived_upper_endpoint']=_error(expected['lambda_bounds'][1],true['high'])
    errors['birth_relative_coefficients']=[]
    for row,(bounds,multiplier) in enumerate(((true['mid'],False),(true['half'],True))):
        for j in range(3):
            source_bounds=_affine_interval(0 if multiplier else line['v'][j],line['u'][j],bounds)
            errors['birth_relative_coefficients'].append(dict(index=[row,j],
                **_error(expected['birth_relative_coefficients'][row][j],source_bounds)))
    errors['literal_phi_coefficients']=[
        _error(expected['literal_phi_coefficients'][0],_affine_interval(line['intercept'],line['slope'],true['mid'])),
        _error(expected['literal_phi_coefficients'][1],_affine_interval(0,line['slope'],true['half']))]
    errors['relative_lambda_mid_to_original_lower']=_error(expected['relative_lambda_mid_to_original_lower'],
                                      (true['mid'][0]-line['bounds'][0],true['mid'][1]-line['bounds'][0]))
    errors['relative_lambda_mid_to_original_upper']=_error(expected['relative_lambda_mid_to_original_upper'],
                                      (line['bounds'][1]-true['mid'][1],line['bounds'][1]-true['mid'][0]))
    slope,intercept=line['slope'],line['intercept']
    require(slope!=0 or intercept!=0,'Entire positive source line lies on literal Gamma')
    root=None if slope==0 else -intercept/slope
    crosses=False
    source_side=None
    if root is None:
        source_side=Fraction(1,2) if intercept>0 else Fraction(-1,2)
    elif true['low'][1]<root<true['high'][0]:
        crosses=True
    elif root<=true['low'][0] or root>=true['high'][1]:
        midpoint=(true['mid'][0]+true['mid'][1])/2
        phi=slope*midpoint+intercept
        require(phi!=0,'Source Gamma constant-side classification unresolved')
        source_side=Fraction(1,2) if phi>0 else Fraction(-1,2)
    else:
        require(False,'Source Gamma/sphere endpoint event unresolved at rational enclosure precision')
    native_crosses=expected['constant_side'] is None
    require(crosses==native_crosses,'Source/native Gamma crossing classification differs')
    if not crosses:
        require(expected['constant_side']==source_side,'Source/native Gamma bank sign differs')
    else:
        true_root_xi=_divide_interval((root-true['mid'][1],root-true['mid'][0]),true['half'])
        errors['gamma_root_xi']=_error(expected['gamma_root_xi'],true_root_xi)
        errors['gamma_left_width_xi']=_error(expected['gamma_left_width_xi'],
                                         (1+true_root_xi[0],1+true_root_xi[1]))
        errors['gamma_right_width_xi']=_error(expected['gamma_right_width_xi'],
                                         (1-true_root_xi[1],1-true_root_xi[0]))
        native_exact=represented['exact_gamma_xi']
        require(-1<native_exact<1 and expected['gamma_left_width_xi']>0 and expected['gamma_right_width_xi']>0,
                'Native Gamma positive bank sliver vanished')
    # Bounds deliberately include the λ center shift even when absolute λ is
    # much larger than the clipped span.  No accuracy admission follows until
    # these errors have been propagated into each physical field/operator.
    return dict(base,status='PASS',empty=False,source_geometry_error_bounds_certified=True,
                native_operation_replay_matches=True,support_classification_certified=True,
                native_replay_checks=comparisons,component_error_enclosures=errors,
                source_interval={key:_enclosure(value) for key,value in true.items()},
                original_source_lambda_bounds=[_q(q) for q in line['bounds']],
                represented_source_ball_endpoint_quadratic_m2=[_q(line['uu']*q*q+2*line['uv']*q+
                    line['vv']-line['radius']**2) for q in (mid-half,mid+half)],
                source_Gamma_crosses=crosses,source_Gamma_bank_sign=None if crosses else _q(source_side),
                source_Gamma_root_lambda=None if root is None else _q(root),
                exact_source_margin_m2=_q(line['margin']),
                represented_chart_endpoint_outside_source_ball_possible=
                    (mid-half<true['low'][0] or mid+half>true['high'][1]),
                lambda_absolute_rounding_error_retained_in_own_geometry_operands=True,
                represented_and_archived_lambda_endpoints_distinguished=True,
                native_sign_fallback_used=represented['exact_sign_fallback_used'])

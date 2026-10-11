"""Independent high-precision fixed-N source-line checks.

The sphere discriminant and Gamma signs are evaluated as exact rational
expressions in the literal input bits.  Direct field differentiation and
positive high-precision Gauss quadrature use no clipping, polynomial, operator,
memory factory, or planetary producer.  The retained float64 vertex convention
and the mathematical inverse-X derivative are separate, explicitly labelled
lanes.  Checking a finite collection of lines cannot certify angular coverage
or the accuracy of a physical contraction integral.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math

import mpmath
import numpy as np


BLOCKS = ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n")
TERMS = ("cross_parent_work_j", "cross_added_work_j", "self_work_j",
         "stiffness_cross_bilinear_j", "stiffness_self_bilinear_j",
         "mass_cross_bilinear_kg_m2", "mass_self_bilinear_kg_m2")
SCOPE = "independent_fixed_N_high_precision_checks_no_angular_or_physical_certificate"


class IndependentLineRefusal(ValueError):
    def __init__(self,message,*,diagnostics=None):
        super().__init__(message)
        self.diagnostics=diagnostics


def require(condition, message):
    if not condition:
        raise IndependentLineRefusal(message)


def _ctx(precision):
    require(type(precision) is int and 40 <= precision <= 200, "Precision must be an integer in 40..200")
    context = mpmath.mp.clone()
    context.dps = precision
    return context


def _fraction(value):
    require(not isinstance(value, (bool, np.bool_)), "Boolean arithmetic operand")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        try:
            return Fraction(*value.as_integer_ratio())
        except (ValueError, OverflowError) as error:
            raise IndependentLineRefusal("Nonfinite arithmetic operand") from error
    require(isinstance(value, str), "Unsupported arithmetic operand")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise IndependentLineRefusal("Invalid decimal/rational arithmetic operand") from error


def _mp(context, value):
    if hasattr(value, "_mpf_"):
        result=context.mpf(value)
        require(context.isfinite(result),"Nonfinite high-precision arithmetic operand")
        return result
    fraction = _fraction(value)
    return context.mpf(fraction.numerator) / fraction.denominator


def _array(value, shape, name):
    raw = np.asarray(value)
    require(raw.dtype.kind in "ifu" and raw.shape == shape, name + " shape/type differs")
    result = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(result).all(), name + " is nonfinite")
    return result.copy()


def _matrix(context, array):
    value = np.asarray(array)
    if value.ndim == 1:
        return context.matrix([_mp(context, x) for x in value])
    return context.matrix([[_mp(context, x) for x in row] for row in value])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _n(value):
    if isinstance(value,(list,tuple)):
        require(all(not isinstance(x,(bool,np.bool_)) for x in value),"Boolean material N operand")
    raw = np.asarray(value)
    require(raw.shape == (3,), "Material N shape differs")
    require(all(isinstance(x,(int,np.integer,float,np.floating,str,Fraction))
                and not isinstance(x,(bool,np.bool_)) for x in raw), "Material N scalar type differs")
    result = tuple(_fraction(x) for x in raw)
    require(min(result) >= 0, "Material N leaves the original triangle")
    epsilon = np.finfo(raw.dtype).eps if raw.dtype.kind == "f" else np.finfo(float).eps
    tolerance = Fraction(*float(128*epsilon).as_integer_ratio())
    # A literal floating barycentric sum is retained; it is never normalized.
    require(abs(sum(result)-1) <= tolerance, "Material N sum differs")
    return result


def _geometry(raw, precision):
    context = _ctx(precision)
    require(type(raw) is dict, "Line source must be an object")
    section = raw.get("section", raw.get("geometry_source", raw))
    require(type(section) is dict, "Literal source section must be an object")
    required = {"triangle_m", "displacement_A_m", "slope_P_m", "lambda_bounds", "normal_xyz", "offset_m"}
    require(required <= set(section) and {"seed_center_world_m", "seed_radius_m"} <= set(raw), "Missing literal line geometry operand")
    X, A, P = (_array(section[key], (3, 3), key) for key in
               ("triangle_m", "displacement_A_m", "slope_P_m"))
    bounds = _array(section["lambda_bounds"], (2,), "Original lambda bounds")
    normal = _array(section["normal_xyz"], (3,), "Literal Gamma normal")
    center = _array(raw["seed_center_world_m"], (3,), "Sphere center")
    _fraction(raw["seed_radius_m"])
    _fraction(section["offset_m"])
    radius, offset = float(raw["seed_radius_m"]), float(section["offset_m"])
    require(math.isfinite(radius) and radius > 0 and math.isfinite(offset), "Sphere radius/Gamma offset differs")
    require(0 < bounds[0] < bounds[1], "Original lambda bounds differ")
    require(abs(float(normal @ normal)-1) <= 128*np.finfo(float).eps, "Literal Gamma normal is not unit length")
    rows = [[_fraction(v) for v in row] for row in X]
    determinant_exact = sum(rows[0][j]*(rows[1][(j+1)%3]*rows[2][(j+2)%3]
                             -rows[1][(j+2)%3]*rows[2][(j+1)%3]) for j in range(3))
    require(determinant_exact > 0, "Original reference chart has nonpositive orientation")
    determinant = _mp(context,determinant_exact)
    # Perform exactly the old source binary64 sums BEFORE high precision.
    C, D = X+P, A-P
    return dict(context=context, section=section, X=X, A=A, P=P, C=C, D=D,
                bounds=bounds, normal=normal, center=center, radius=radius, offset=offset,
                determinant=determinant)


def _radial_geometry(geometry, N):
    context = geometry["context"]
    rational_n = _n(N)
    C, D = geometry["C"], geometry["D"]
    direction = tuple(sum(rational_n[i]*_fraction(C[i, j]) for i in range(3)) for j in range(3))
    origin = tuple(sum(rational_n[i]*_fraction(D[i, j]) for i in range(3))
                   - _fraction(geometry["center"][j]) for j in range(3))
    norm2, projection, origin2 = _dot(direction, direction), _dot(origin, direction), _dot(origin, origin)
    require(norm2 > 0, "Collapsed birth radial line")
    radius = _fraction(geometry["radius"])
    discriminant = radius*radius*norm2-origin2*norm2+projection*projection
    result = dict(intervals=[], discriminant=discriminant, direction=direction, origin=origin)
    if discriminant <= 0:
        return result
    exact_mid = -projection/norm2
    original_lo, original_hi = (_fraction(v) for v in geometry["bounds"])
    closest = min(max(exact_mid, original_lo), original_hi)
    minimum_quadratic = norm2*closest*closest+2*projection*closest+origin2-radius*radius
    if minimum_quadratic >= 0:
        return result
    mid = _mp(context, exact_mid)
    half = context.sqrt(_mp(context, discriminant))/_mp(context, norm2)
    require(mid-half < mid+half, "Unresolved positive sphere support at selected precision")
    lo = max(_mp(context, geometry["bounds"][0]), mid-half)
    hi = min(_mp(context, geometry["bounds"][1]), mid+half)
    require(hi > lo, "Unresolved positive original-cohort support at selected precision")
    normal = tuple(_fraction(v) for v in geometry["normal"])
    slope = _dot(normal, direction)
    intercept = _dot(normal, tuple(origin[j]+_fraction(geometry["center"][j]) for j in range(3)))
    intercept -= _fraction(geometry["offset"])
    require(slope != 0 or intercept != 0, "Positive-length material line is coplanar with literal Gamma")
    root = None if slope == 0 else _mp(context, -intercept/slope)
    cuts = [lo] + ([root] if root is not None and lo < root < hi else []) + [hi]
    intervals = []
    for left, right in zip(cuts, cuts[1:]):
        midpoint = (left+right)/2
        phi = _mp(context, slope)*midpoint+_mp(context, intercept)
        require(phi != 0, "Gamma midpoint sign unresolved")
        intervals.append(dict(lambda_low=left, lambda_high=right, side=context.mpf(".5") if phi > 0 else context.mpf("-.5")))
    result.update(intervals=intervals, lambda_mid=(lo+hi)/2, lambda_halfwidth=(hi-lo)/2,
                  gamma_root_xi=None if root is None else (root-(lo+hi)/2)/((hi-lo)/2),
                  slope=slope, intercept=intercept)
    return result


def independent_line_geometry(raw, N, *, precision=80):
    """Exact rational classification, followed by high-precision root evaluation."""
    geometry = _geometry(raw, precision)
    result = _radial_geometry(geometry, N)
    context = geometry["context"]
    output = dict(scope=SCOPE, precision_decimal_digits=precision,
        barycentric=[context.nstr(_mp(context, x), precision) for x in _n(N)],
        intervals=[dict(lambda_low=context.nstr(v["lambda_low"], precision),
                        lambda_high=context.nstr(v["lambda_high"], precision), side=float(v["side"]))
                   for v in result["intervals"]],
        sphere_margin_times_direction_squared=context.nstr(_mp(context, result["discriminant"]), precision),
        literal_float64_source_sums_retained=True, source_history_reinitialized=False,
        angular_coverage_certified=False, physical_contraction_certified=False,
        physical_creation_admitted=False, physical_release_admitted=False)
    for key in ("lambda_mid", "lambda_halfwidth", "gamma_root_xi"):
        output[key] = None if result.get(key) is None else context.nstr(result[key], precision)
    return output


def _sym(context, vector, gradient):
    return context.matrix((vector[0]*gradient[0], vector[1]*gradient[1], vector[2]*gradient[2],
        vector[0]*gradient[1]+vector[1]*gradient[0], vector[0]*gradient[2]+vector[2]*gradient[0],
        vector[1]*gradient[2]+vector[2]*gradient[1]))


def _literal_vertices(X, Q):
    """Independently execute the declared old binary64 plane-coordinate convention.

    This is not the direct inverse-X diagnostic, and its bits are never replaced
    by a higher-precision reinitialization.  No producer function is called.
    """
    local = X @ Q
    h = float(np.mean(local[:, 2]))
    require(h > 0 and np.allclose(local[:, 2], h, rtol=2e-12, atol=2e-9), "Original source-frame plane differs")
    design = np.column_stack((np.ones(3), local[:, :2]))
    require(np.linalg.det(design) > 0, "Original top triangle orientation differs")
    gradients = np.linalg.inv(design)[1:]
    result0, result1 = np.zeros((3, 6, 18)), np.zeros((3, 6, 18))
    for point in range(3):
        for vertex in range(3):
            gradient = np.r_[gradients[:, vertex], -local[point, :2] @ gradients[:, vertex]/h]
            normal_gradient = np.array((0., 0., float(point == vertex)/h))
            for axis in range(3):
                vector = Q[axis]
                def sym(g):
                    return np.array((vector[0]*g[0], vector[1]*g[1], vector[2]*g[2],
                        vector[0]*g[1]+vector[1]*g[0], vector[0]*g[2]+vector[2]*g[0],
                        vector[1]*g[2]+vector[2]*g[1]))
                column = sym(gradient)
                result1[point, :, 6*vertex+axis] = column
                result0[point, :, 6*vertex+3+axis] = column+sym(normal_gradient)
                result1[point, :, 6*vertex+3+axis] = -column
    return result0, result1


@dataclass(frozen=True)
class IndependentLineSource:
    geometry: dict
    elasticity: object
    basis: object
    memory0: object
    memory1: object
    literal_vertex_B0: object
    literal_vertex_B1: object
    angular_sample_inverse: object
    inverse_XT: object
    density: object


def read_line_source(raw, *, precision=80):
    geometry = _geometry(raw, precision)
    context = geometry["context"]
    def alias(long, short):
        require(long in raw or short in raw, "Missing retained source operand: "+long)
        return raw[long] if long in raw else raw[short]
    C = _array(alias("elasticity_pa", "C"), (6, 6), "Source elasticity")
    Q = _array(alias("basis_xyz", "Q"), (3, 3), "Source proper basis")
    e0, e1 = (_array(alias(long, short), (3, 6), long) for long, short in
              (("memory0", "e0"), ("memory1", "e1")))
    require(np.allclose(Q.T@Q, np.eye(3), rtol=0, atol=2e-12) and np.linalg.det(Q) > 0, "Source basis differs")
    require(np.allclose(C, C.T, rtol=0, atol=64*np.finfo(float).eps*np.max(np.abs(C)))
            and np.linalg.eigvalsh(C).min() > 0, "Source elasticity is not positive symmetric")
    B0, B1 = _literal_vertices(geometry["X"], Q)
    sample = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    mass, volume = geometry["section"].get("mass_kg"), geometry["section"].get("volume_m3")
    density = None
    if mass is not None and volume is not None:
        _fraction(mass)
        _fraction(volume)
        require(math.isfinite(float(mass)) and math.isfinite(float(volume)) and mass > 0 and volume > 0, "Retained mass/volume differs")
        density = _mp(context, mass)/_mp(context, volume)
    return IndependentLineSource(geometry, _matrix(context, C), _matrix(context, Q),
        _matrix(context, e0), _matrix(context, e1), tuple(_matrix(context, v) for v in B0),
        tuple(_matrix(context, v) for v in B1), _matrix(context, np.linalg.inv(sample)),
        _matrix(context, geometry["X"]).T**-1, density)


def _exact_scalar_fraction(value):
    if hasattr(value,"_mpf_"):
        sign,mantissa,exponent,bits=value._mpf_
        require(bits>=0,"Nonfinite high-precision arithmetic operand")
        return Fraction((-1 if sign else 1)*mantissa)*(Fraction(2)**exponent)
    return _fraction(value)


def _fields(source, N, lam, side):
    geometry, context = source.geometry, source.geometry["context"]
    exact_N=_n(N)
    exact_lam=_exact_scalar_fraction(lam)
    N = [_mp(context, v) for v in exact_N]
    lam, side = _mp(context, exact_lam), _mp(context, side)
    lam_minus_one=_mp(context,exact_lam-1)
    require(lam > 0 and side in (context.mpf("-.5"), context.mpf(".5")), "Lambda/side field operands differ")
    exact_b,exact_a=(_fraction(v) for v in geometry["bounds"])
    b,a=_mp(context,exact_b),_mp(context,exact_a)
    require(exact_b <= exact_lam <= exact_a, "Field leaves original radial support")
    Cbirth, Dbirth = _matrix(context, geometry["C"]), _matrix(context, geometry["D"])
    # Evaluate the declared affine birth chart in exact input rationals before
    # the one MP rounding.  Rounding an absolute lambda near1 before taking
    # world or original-bound differences discards the small local distance.
    exact_C=[[ _fraction(v) for v in row] for row in geometry["C"]]
    exact_D=[[ _fraction(v) for v in row] for row in geometry["D"]]
    offset=context.matrix([_mp(context,sum(exact_N[i]*(exact_lam*exact_C[i][j]+exact_D[i][j])
        for i in range(3))-_fraction(geometry["center"][j])) for j in range(3)])
    radius2 = _mp(context, geometry["radius"])**2
    r2 = _dot(offset, offset)/radius2
    require(r2 <= 1+256*context.eps, "Field lies outside the literal sphere")
    t = _mp(context,(exact_lam-exact_b)/(exact_a-exact_b))
    sphere, angular, radial = (1-r2)**2, 27*N[0]*N[1]*N[2], 4*t*(1-t)
    bubble = sphere*angular*radial
    derivative_birth = [context.matrix([_mp(context,exact_lam*(exact_C[v][j]-exact_C[0][j])
                        +exact_D[v][j]-exact_D[0][j]) for j in range(3)]) for v in (1,2)]
    derivative_birth.append(context.matrix([_mp(context,sum(exact_N[i]*exact_C[i][j]
                        for i in range(3))) for j in range(3)]))
    sphere_gradient = [-4*(1-r2)*_dot(v, offset)/radius2 for v in derivative_birth]
    angular_gradient = (27*N[2]*(N[0]-N[1]), 27*N[1]*(N[0]-N[2]), context.mpf(0))
    radial_gradient = (context.mpf(0), context.mpf(0), 4*(1-2*t)/(a-b))
    parameter = [sphere*radial*angular_gradient[i]+angular*radial*sphere_gradient[i]
                 +angular*sphere*radial_gradient[i] for i in range(3)]
    inverse = source.inverse_XT
    grad_lambda = context.matrix([sum(inverse[i,j] for i in range(3)) for j in range(3)])
    grad_N = [context.matrix([(inverse[i,j]-N[i]*grad_lambda[j])/lam for j in range(3)]) for i in range(3)]
    gradient = parameter[0]*grad_N[1]+parameter[1]*grad_N[2]+parameter[2]*grad_lambda
    B = sum((N[i]*source.literal_vertex_B0[i]+N[i]*source.literal_vertex_B1[i]/lam for i in range(3)), context.zeros(6,18))
    T, direct = context.zeros(3,18), context.zeros(6,18)
    for vertex in range(3):
        gradient_P = lam_minus_one*grad_N[vertex]+N[vertex]*grad_lambda
        for axis in range(3):
            index = 6*vertex+axis
            T[axis,index], T[axis,index+3] = N[vertex], lam_minus_one*N[vertex]
            vector = source.basis.T[:,axis]
            direct[:,index] = _sym(context, vector, source.basis.T*grad_N[vertex])
            direct[:,index+3] = _sym(context, vector, source.basis.T*gradient_P)
    local_T, local_gradient = source.basis.T*T, source.basis.T*gradient
    extra = context.zeros(6,18)
    for column in range(18):
        extra[:,column] = _sym(context, local_T[:,column], local_gradient)
    without_H = bubble*B+extra
    interpolation = context.matrix([N])*source.angular_sample_inverse
    elastic = (interpolation*source.memory0).T+(interpolation*source.memory1).T/lam
    return dict(w=bubble, L=lam*gradient, Bhat=lam*B, Ghat_without_H=lam*without_H,
        T=T, E_without_H=bubble*T, memoryhat=lam*elastic, B=B, G=side*without_H,
        E=side*bubble*T, memory=elastic, direct_derivative_B=direct,
        literal_vs_direct_derivative_difference=B-direct,
        literal_source_vertex_convention="old_binary64_plane_B0_B1_then_high_precision_N_interpolation")


def evaluate_source_fields(raw, N, lambda_value, side, *, precision=80):
    source = raw if isinstance(raw, IndependentLineSource) else read_line_source(raw, precision=precision)
    return _fields(source, N, lambda_value, side)


def _abs(context, matrix):
    return context.matrix([[abs(matrix[i,j]) for j in range(matrix.cols)] for i in range(matrix.rows)])


def _blocks(source, fields, parent, added):
    context, C, density = source.geometry["context"], source.elasticity, source.density
    B, G, T, E = fields["B"]*parent, fields["G"]*added, fields["T"]*parent, fields["E"]*added
    ownC = _abs(context,C)
    values = dict(stiffness_cross=B.T*C*G, stiffness_enriched=G.T*C*G,
        mass_cross=density*T.T*E, mass_enriched=density*E.T*E,
        enrichment_force_n=G.T*C*fields["memory"])
    own = dict(stiffness_cross=_abs(context,B).T*ownC*_abs(context,G),
        stiffness_enriched=_abs(context,G).T*ownC*_abs(context,G),
        mass_cross=density*_abs(context,T).T*_abs(context,E),
        mass_enriched=density*_abs(context,E).T*_abs(context,E),
        enrichment_force_n=_abs(context,G).T*ownC*_abs(context,fields["memory"]))
    return values, own


def _basis(context, value, name):
    array = np.asarray(value)
    require(array.ndim==2 and array.shape[0]==18 and 1<=array.shape[1]<=18 and array.dtype.kind in "ifu", name+" shape differs")
    require(np.isfinite(array).all(), name+" is nonfinite")
    return _matrix(context,array)


def _probe(source, fields, N, lam, probe):
    context, C = source.geometry["context"], source.elasticity
    def vector(key):return _matrix(context,_array(probe[key],(18,),key))
    q,a = vector("parent_increment_m"),vector("enrichment_increment_m")
    z = _matrix(context,_array(probe.get("mean_z_m",np.zeros(18)),(18,),"mean_z_m"))
    e0 = source.memory0 if "mean_e0" not in probe else _matrix(context,_array(probe["mean_e0"],(3,6),"mean_e0"))
    e1 = source.memory1 if "mean_e1" not in probe else _matrix(context,_array(probe["mean_e1"],(3,6),"mean_e1"))
    angular = context.matrix([[_mp(context,v) for v in _n(N)]])*source.angular_sample_inverse
    memory = (angular*e0).T+(angular*e1).T/lam
    Bq,Ga,Gz,Tq,Ea = fields["B"]*q,fields["G"]*a,fields["G"]*z,fields["T"]*q,fields["E"]*a
    def inner(left,middle,right):
        return (left.T*middle*right)[0],(_abs(context,left).T*_abs(context,middle)*_abs(context,right))[0]
    value,own = {},{}
    rows = ((TERMS[0],Bq,C,Gz),(TERMS[1],Ga,C,memory),(TERMS[2],Ga,C,Gz),
            (TERMS[3],Bq,C,Ga),(TERMS[4],Ga,C,Ga),
            (TERMS[5],Tq,context.eye(3)*source.density,Ea),
            (TERMS[6],Ea,context.eye(3)*source.density,Ea))
    for name,left,middle,right in rows:
        value[name],own[name] = inner(left,middle,right)
    for name,left,right in (("enrichment_force_n",fields["G"],memory+Gz),
                            ("added_parent_force_n",fields["B"],Gz)):
        value[name] = left.T*C*right
        own[name] = _abs(context,left).T*_abs(context,C)*_abs(context,right)
    return value,own


def _flatten(prefix, values):
    names,numbers = [],[]
    for key,value in values.items():
        if hasattr(value,"rows"):
            for i in range(value.rows):
                for j in range(value.cols):
                    suffix = f"[{i}]" if key in ("enrichment_force_n","added_parent_force_n") else f"[{i},{j}]"
                    names.append(prefix+key+suffix);numbers.append(value[i,j])
        else:
            names.append(prefix+key);numbers.append(value)
    return names,numbers


@lru_cache(maxsize=24)
def _gauss(precision, order):
    context = _ctx(precision)
    nodes,weights = context.gauss_quadrature(order,"legendre")
    return tuple(nodes),tuple(weights)


def integrate_source_line(raw, N, *, added_basis=None, parent_basis=None, probes=(),
                          precision=80, quadrature_orders=(12,20)):
    """Direct positive quadrature; agreement is confined to this fixed-N line."""
    source = raw if isinstance(raw,IndependentLineSource) else read_line_source(raw,precision=precision)
    context = source.geometry["context"]
    require(source.density is not None, "Direct integral needs retained mass_kg and volume_m3")
    require(len(quadrature_orders)==2 and all(type(v) is int and 8<=v<=48 for v in quadrature_orders)
            and quadrature_orders[0]<quadrature_orders[1], "Use two increasing quadrature orders in 8..48")
    require(len(probes)<=16 and len({p.get("name") for p in probes})==len(probes), "Probe names/count differ")
    require(all(isinstance(p.get("name"),str) and p["name"] for p in probes), "Probe names must be nonempty strings")
    for probe in probes:
        require({"parent_increment_m","enrichment_increment_m"}<=set(probe),"Missing declared probe coordinate")
        for key in ("parent_increment_m","enrichment_increment_m","mean_z_m"):
            if key in probe:_array(probe[key],(18,),key)
        for key in ("mean_e0","mean_e1"):
            if key in probe:_array(probe[key],(3,6),key)
    parent = context.eye(18) if parent_basis is None else _basis(context,parent_basis,"Parent basis")
    added = None if added_basis is None else _basis(context,added_basis,"Added basis")
    geometry = _radial_geometry(source.geometry,N)
    runs=[]
    for order in quadrature_orders:
        total=own_total=None;names=None
        nodes,weights=_gauss(context.dps,order)
        # An empty line still exposes the full requested component schema.
        intervals = geometry["intervals"]
        visits = intervals if intervals else [dict(lambda_low=_mp(context,source.geometry["bounds"][0]),
            lambda_high=_mp(context,source.geometry["bounds"][0]),side=context.mpf(".5"))]
        for interval in visits:
            mid=(interval["lambda_low"]+interval["lambda_high"])/2
            half=(interval["lambda_high"]-interval["lambda_low"])/2
            for node,weight in zip(nodes,weights):
                exact_lam=_exact_scalar_fraction(mid)+_exact_scalar_fraction(half)*_exact_scalar_fraction(node)
                lam=_mp(context,exact_lam)
                if half==0:
                    # Avoid manufacturing a field inside an absent sphere.
                    full={key:context.zeros(18,1 if key=="enrichment_force_n" else 18) for key in BLOCKS}
                    flattened_names,values=_flatten("full.",full);operands=list(values)
                    if added is not None:
                        projected={key:context.zeros(added.cols if key in ("stiffness_enriched","mass_enriched","enrichment_force_n") else parent.cols,
                            1 if key=="enrichment_force_n" else added.cols) for key in BLOCKS}
                        n,v=_flatten("projected.",projected);flattened_names+=n;values+=v;operands+=v
                    for probe in probes:
                        row={key:context.mpf(0) for key in TERMS}
                        row.update(enrichment_force_n=context.zeros(18,1),added_parent_force_n=context.zeros(18,1))
                        n,v=_flatten("probe."+probe["name"]+".",row);flattened_names+=n;values+=v;operands+=v
                else:
                    fields=_fields(source,N,exact_lam,interval["side"])
                    full,own=_blocks(source,fields,context.eye(18),context.eye(18))
                    flattened_names,values=_flatten("full.",full);_,operands=_flatten("full.",own)
                    if added is not None:
                        projected,own=_blocks(source,fields,parent,added)
                        n,v=_flatten("projected.",projected);_,o=_flatten("projected.",own)
                        flattened_names+=n;values+=v;operands+=o
                    for probe in probes:
                        row,own=_probe(source,fields,N,lam,probe)
                        n,v=_flatten("probe."+probe["name"]+".",row);_,o=_flatten("probe."+probe["name"]+".",own)
                        flattened_names+=n;values+=v;operands+=o
                if total is None:
                    names=tuple(flattened_names);total=[context.mpf(0) for _ in values];own_total=list(total)
                require(tuple(flattened_names)==names,"Direct component schema changed")
                measure=half*weight*source.geometry["determinant"]*lam*lam
                for i,(value,own) in enumerate(zip(values,operands)):
                    total[i]+=measure*value;own_total[i]+=measure*own
                if half==0:break
        runs.append((names,total,own_total))
    require(runs[0][0]==runs[1][0],"High precision quadrature schema changed")
    differences=[abs(a-b) for a,b in zip(runs[0][1],runs[1][1])]
    own=[max(a,b) for a,b in zip(runs[0][2],runs[1][2])]
    arithmetic=[context.mpf(262144)*context.eps*v for v in own]
    failed=[i for i,(error,budget) in enumerate(zip(differences,arithmetic)) if error>budget]
    if failed:
        checks={runs[1][0][i]:dict(
            coarse_observed_value=context.nstr(runs[0][1][i],precision),
            fine_observed_value=context.nstr(runs[1][1][i],precision),
            absolute_difference=context.nstr(differences[i],precision),
            arithmetic_bound=context.nstr(arithmetic[i],precision),
            own_operand_scale=context.nstr(own[i],precision),
            positive_scale_floor_used=False,
            ratio=None if arithmetic[i]==0 else context.nstr(differences[i]/arithmetic[i],30)) for i in failed}
        diagnostics=dict(format="independent-source-fixed-N-line-refusal-0.1",scope=SCOPE,
            status="BOUNDED_HIGH_PRECISION_DIRECT_QUADRATURE_REFUSED",
            refusal_message="Bounded high precision direct line quadrature did not agree",
            oracle_input_binding_sha256=_input_binding(raw,N,added_basis,parent_basis,probes,context.dps),
            component_names=list(runs[1][0]),coarse_observed_values=tuple(runs[0][1]),
            fine_observed_values=tuple(runs[1][1]),coarse_absolute_operand_scales=tuple(runs[0][2]),
            fine_absolute_operand_scales=tuple(runs[1][2]),failed_component_checks=checks,
            checked_component_count=len(runs[1][0]),failed_component_count=len(failed),
            quadrature_orders=tuple(quadrature_orders),precision_decimal_digits=context.dps,
            trusted_oracle_available=False,observed_fine_values_certified=False,
            geometry=independent_line_geometry(raw if not isinstance(raw,IndependentLineSource) else
                dict(section=source.geometry["section"],seed_center_world_m=source.geometry["center"],seed_radius_m=source.geometry["radius"]),N,precision=context.dps),
            source_geometry_accuracy_certified=False,angular_accuracy_certified=False,
            general_contraction_accuracy_certified=False,physical_contraction_certified=False,
            physical_creation_admitted=False,physical_release_admitted=False,available_for_creation_funding=False)
        raise IndependentLineRefusal(diagnostics["refusal_message"],diagnostics=diagnostics)
    return dict(scope=SCOPE,status="HIGH_PRECISION_FIXED_LINE_QUADRATURE_AGREEMENT",
        trusted_oracle_available=True,numerical_quadrature_agreement_checked=True,
        oracle_input_binding_sha256=_input_binding(raw,N,added_basis,parent_basis,probes,context.dps),
        component_names=runs[1][0],flattened_values=tuple(runs[1][1]),
        absolute_operand_scales=tuple(own),numerical_error_bounds=tuple(e+b for e,b in zip(differences,arithmetic)),
        quadrature_orders=tuple(quadrature_orders),precision_decimal_digits=context.dps,
        geometry=independent_line_geometry(raw if not isinstance(raw,IndependentLineSource) else
            dict(section=source.geometry["section"],seed_center_world_m=source.geometry["center"],seed_radius_m=source.geometry["radius"]),N,precision=context.dps),
        reference_side_volumes_m3={label:sum(source.geometry["determinant"]*(v["lambda_high"]**3-v["lambda_low"]**3)/3
            for v in geometry["intervals"] if (v["side"]>0)==positive)
            for label,positive in (("reference_volume_negative_m3",False),("reference_volume_positive_m3",True))},
        reference_side_volume_scope="true_source_fixed_N_lines_no_angular_volume_certificate",
        projection_before_Gram=True,parent_work_scope="whole_original_cohort_parent_work_is_not_a_clipped_line_term",
        angular_coverage_certified=False,general_contraction_accuracy_certified=False,
        physical_contraction_certified=False,physical_creation_admitted=False,
        physical_release_admitted=False,available_for_creation_funding=False)


def archive_line_result(result):
    """Decimal archive of compact selected-line evidence, never a physical seal."""
    precision=result["precision_decimal_digits"]
    context=_ctx(precision)
    def plain(value):
        if hasattr(value,"_mpf_"):return context.nstr(value,precision)
        if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [plain(v) for v in value]
        return value
    output=plain(result)
    output.setdefault("format","independent-source-fixed-N-line-quadrature-0.1")
    output["sha256"]=hashlib.sha256(json.dumps(output,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return output


def _input_binding(raw,N,added,parent,probes,precision):
    if isinstance(raw,IndependentLineSource):
        source=raw
        raw=dict(section=source.geometry["section"],seed_center_world_m=source.geometry["center"],
                 seed_radius_m=source.geometry["radius"],C=source.elasticity.tolist(),Q=source.basis.tolist(),
                 e0=source.memory0.tolist(),e1=source.memory1.tolist())
    def plain(value):
        if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple,np.ndarray)):return [plain(v) for v in value]
        if isinstance(value,(np.integer,np.floating,Fraction)):
            f=_fraction(value);return str(f.numerator)+"/"+str(f.denominator)
        if hasattr(value,"_mpf_"):
            sign,mantissa,exponent,_=value._mpf_
            f=Fraction((-1 if sign else 1)*mantissa)*(Fraction(2)**exponent)
            return str(f.numerator)+"/"+str(f.denominator)
        return value
    body=dict(source=plain(raw),N=[str(v.numerator)+"/"+str(v.denominator) for v in _n(N)],
              added_basis=plain(added),parent_basis=plain(parent),probes=plain(probes),precision=precision)
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def decode_native_array(witness,label):
    """Decode exact logical ratios; no longdouble padding or float64 truncation."""
    records=witness.get("native_exact_binary_ratio",{})
    require(type(records) is dict and label in records,"Missing native logical witness: "+label)
    record=records[label]
    require(type(record) is dict and type(record.get("shape")) is list,"Native shape differs")
    shape=record["shape"]
    require(len(shape)<=4 and all(type(n) is int and 0<=n<=10000 for n in shape),"Native shape is unbounded")
    pairs=record.get("numerator_denominator")
    require(type(pairs) is list and len(pairs)==math.prod(shape),"Native logical value count differs")
    result=[]
    for pair in pairs:
        require(type(pair) is list and len(pair)==2 and all(type(v) is str and len(v)<=2048 for v in pair),"Native logical ratio differs")
        try:
            numerator,denominator=map(int,pair)
        except ValueError as error:
            raise IndependentLineRefusal("Native ratio must contain integer strings") from error
        require(denominator>0,"Native denominator is nonpositive")
        result.append(Fraction(numerator,denominator))
    return np.asarray(result,dtype=object).reshape(shape)


def _native_or_public(witness,label,public):
    if label in witness.get("native_exact_binary_ratio",{}):
        return decode_native_array(witness,label),True
    return np.asarray(public,dtype=object),False


def _mp_array(context,value):
    return np.asarray([_mp(context,v) for v in np.asarray(value,dtype=object).ravel()],dtype=object).reshape(np.shape(value))


def _scalar_check(context,actual,expected,budget,own,*,native):
    actual,expected,budget,own=(_mp(context,v) for v in (actual,expected,budget,own))
    require(context.isfinite(actual) and context.isfinite(expected) and context.isfinite(budget)
            and context.isfinite(own) and budget>=0 and own>=0,"Invalid line component/budget")
    # Conversion allowance applies only when the saved terminal number is
    # binary64.  It is neither geometric nor angular integration credit.
    conversion=context.mpf(0) if native else _mp(context,np.finfo(float).eps)*abs(actual)
    bound=budget+conversion
    error=abs(actual-expected)
    return bool(error<=bound),dict(actual=context.nstr(actual,24),expected=context.nstr(expected,24),
        absolute_difference=context.nstr(error,24),arithmetic_bound=context.nstr(bound,24),
        own_operand_scale=context.nstr(own,24),binary64_terminal_conversion_used=not native,
        positive_scale_floor_used=False)


def _geometry_witness_check(raw,N,witness,precision):
    context=_ctx(precision)
    true=independent_line_geometry(raw,N,precision=precision)
    interval=witness.get("interval")
    if interval is None:
        return (not true["intervals"],dict(support_classification_matches=not true["intervals"],empty=True))
    mid,_=_native_or_public(witness,"interval.lambda_mid",interval["lambda_mid"])
    half,_=_native_or_public(witness,"interval.lambda_halfwidth",interval["lambda_halfwidth"])
    mid,half=_mp(context,np.asarray(mid).item()),_mp(context,np.asarray(half).item())
    require(half>0,"Saved positive line halfwidth differs")
    if not true["intervals"]:
        return False,dict(support_classification_matches=False,empty=False)
    lo,hi=context.mpf(true["intervals"][0]["lambda_low"]),context.mpf(true["intervals"][-1]["lambda_high"])
    # Endpoint errors are reported independently.  Passing this comparison
    # means a source arithmetic observation, not certified clipping accuracy.
    discrepancy=max(abs(mid-half-lo),abs(mid+half-hi))
    true_half=(hi-lo)/2
    allowance=_mp(context,256*np.finfo(np.longdouble).eps)*true_half
    halfwidth_error=abs(half-true_half)
    halfwidth_allowance=_mp(context,256*np.finfo(np.longdouble).eps)*true_half
    width_ok=bool(halfwidth_error<=halfwidth_allowance)
    radial=_radial_geometry(_geometry(raw,precision),N)
    true_root=None if radial["slope"]==0 else _mp(context,-radial["intercept"]/radial["slope"])
    true_xi=None if true_root is None else (true_root-mid)/half
    saved_root=interval.get("gamma_root_xi")
    if saved_root is not None:
        if "interval.side" in witness.get("native_exact_binary_ratio",{}):
            saved_root=decode_native_array(witness,"interval.side").reshape(-1)[0]
        else:
            saved_root,_=_native_or_public(witness,"interval.gamma_root_xi",saved_root)
        saved_root=_mp(context,np.asarray(saved_root).item())
    true_crosses=true_xi is not None and -1<true_xi<1
    gamma_error=None
    if true_crosses:
        gamma_ok=saved_root is not None
        if gamma_ok:
            gamma_error=abs(saved_root-true_xi)
            gamma_ok=bool(gamma_error<=_mp(context,256*np.finfo(np.longdouble).eps)*(1+abs(true_xi)))
    else:
        phi=_mp(context,radial["slope"])*mid+_mp(context,radial["intercept"])
        true_side=context.mpf(".5") if phi>0 else context.mpf("-.5")
        saved_side=interval.get("constant_side")
        gamma_ok=saved_root is None and (saved_side is None or _mp(context,saved_side)==true_side)
    bank_checks={}
    if true_crosses:
        for key,true_width in (("gamma_left_width_xi",1+true_xi),("gamma_right_width_xi",1-true_xi)):
            if interval.get(key) is not None:
                saved_width,_=_native_or_public(witness,"interval."+key,interval[key])
                saved_width=_mp(context,np.asarray(saved_width).item())
                bank_error=abs(saved_width-true_width)
                bank_allowance=_mp(context,256*np.finfo(np.longdouble).eps)*true_width
                bank_checks[key]=dict(absolute_difference=context.nstr(bank_error,30),
                    own_bank_width_allowance=context.nstr(bank_allowance,30),
                    passed=bool(saved_width>0 and bank_error<=bank_allowance))
        gamma_ok=gamma_ok and all(c["passed"] for c in bank_checks.values())
    checked=bool(discrepancy<=allowance and width_ok and gamma_ok)
    return checked,dict(support_classification_matches=True,empty=False,
        maximum_lambda_endpoint_difference=context.nstr(discrepancy,30),
        halfwidth_matches_own_scale=width_ok,
        halfwidth_absolute_difference=context.nstr(halfwidth_error,30),
        halfwidth_own_scale_allowance=context.nstr(halfwidth_allowance,30),
        arithmetic_observation_allowance=context.nstr(allowance,30),
        endpoint_comparison_uses_own_support_span=True,
        absolute_lambda_midpoint_scale_borrowed=False,
        saved_gamma_root_xi=None if saved_root is None else context.nstr(saved_root,30),
        true_gamma_root_xi=None if true_xi is None else context.nstr(true_xi,30),
        Gamma_split_matches_literal_source=gamma_ok,
        Gamma_bank_width_own_scale_checks=bank_checks,
        gamma_root_difference=None if gamma_error is None else context.nstr(gamma_error,30),
        sphere_and_Gamma_geometry_accuracy_certified=False)


def audit_line_witness(raw,N,witness,*,added_basis=None,parent_basis=None,probes=(),precision=80,
                       oracle=None,raise_on_mismatch=True):
    """Check every saved line component against the independent direct route.

    Saved arithmetic budgets remain separate from true-source geometry and
    from angular convergence.  A cached oracle is reusable only for exactly
    the same source, native N, declared bases, probes and precision.
    """
    require(type(witness) is dict,"Line witness must be an object")
    expected_binding=_input_binding(raw,N,added_basis,parent_basis,probes,precision)
    if oracle is None:
        oracle=integrate_source_line(raw,N,added_basis=added_basis,parent_basis=parent_basis,
                                     probes=probes,precision=precision)
    require(oracle.get("oracle_input_binding_sha256")==expected_binding,"Cached oracle source/N/W/probe binding differs")
    require(oracle.get("status")=="HIGH_PRECISION_FIXED_LINE_QUADRATURE_AGREEMENT"
            and oracle.get("trusted_oracle_available",True) is True
            and oracle.get("observed_fine_values_certified",True) is True,
            "Cached oracle does not contain successful high precision quadrature agreement")
    context=_ctx(precision)
    expected=dict(zip(oracle["component_names"],oracle["flattened_values"]))
    own=dict(zip(oracle["component_names"],oracle["absolute_operand_scales"]))
    error=dict(zip(oracle["component_names"],oracle["numerical_error_bounds"]))
    names=witness.get("component_names")
    require(type(names) is list and len(names)==len(set(names)) and set(names)==set(expected),"Saved line component schema differs")
    values,native=_native_or_public(witness,"values",witness.get("values"))
    budgets,_=_native_or_public(witness,"budgets",witness.get("roundoff_component_bounds"))
    require(values.shape==budgets.shape==(len(names),),"Saved line values/budgets shape differs")
    checks={};failed=[];numeric_failed=[];budget_unresolved=[]
    for name,value,budget in zip(names,values,budgets):
        require(_mp(context,budget)>=0,"Saved line arithmetic budget is negative")
        passed,check=_scalar_check(context,value,expected[name],_mp(context,budget)+error[name],own[name],native=native)
        # Inflating an arithmetic witness is not a substitute for agreement.
        cap=_mp(context,65536*np.finfo(np.longdouble).eps)*own[name]
        honest=bool(_mp(context,budget)<=cap)
        check.update(passed=passed and honest,
            numeric_component_agreement_with_declared_budget=passed,
            saved_arithmetic_budget_within_independent_operands=honest,
            coefficient_construction_operand_accuracy_certified=False)
        if not passed:numeric_failed.append(name)
        if not honest:budget_unresolved.append(name)
        checks[name]=check
        if not check["passed"]:failed.append(name)
    geometry_ok,geometry_check=_geometry_witness_check(raw,N,witness,precision)
    status="PASS" if not failed and geometry_ok else "OPEN"
    if raise_on_mismatch:
        require(status=="PASS","Independent source line component/geometry differs: "+", ".join(failed[:4]))
    represented=None
    if witness.get("interval") is not None and isinstance(witness.get("coefficients"),dict):
        represented=audit_represented_primitive(raw,N,witness,added_basis=added_basis,
            parent_basis=parent_basis,probes=probes,precision=precision,raise_on_mismatch=False)
    return dict(scope=SCOPE,source_direct_quadrature_status=status,
        fixed_line_numeric_agreement_status="PASS" if not numeric_failed else "OPEN",
        saved_arithmetic_operand_scope_status="UNRESOLVED",
        saved_arithmetic_own_field_scale_observation_status="PASS" if not budget_unresolved else "OPEN",
        saved_arithmetic_operand_authentication_certified=False,
        arithmetic_budget_scope="own_field_scale_observation_does_not_authenticate_coefficient_construction_operands",
        numerically_mismatching_components=numeric_failed,
        unresolved_budget_scope_components=budget_unresolved,
        represented_polynomial_primitive_audit=represented,
        all_component_checks=checks,mismatching_components=failed,
        geometry_status="PASS" if geometry_ok else "OPEN",geometry_checks=geometry_check,
        checked_component_count=len(names),native_logical_values_used=native,
        cached_oracle_source_binding_verified=True,angular_accuracy_certified=False,
        general_contraction_accuracy_certified=False,physical_contraction_certified=False,
        physical_creation_admitted=False,physical_release_admitted=False,available_for_creation_funding=False)


def _polynomial_value(context,coefficients,xi):
    coefficients=_mp_array(context,coefficients)
    require(coefficients.ndim>=1 and 1<=len(coefficients)<=24,"Coefficient axis differs")
    result=np.asarray(coefficients[-1],dtype=object).copy()
    for coefficient in coefficients[-2::-1]:result=result*xi+coefficient
    return np.asarray(result,dtype=object)


def audit_coefficient_fields(raw,N,witness,*,precision=80,raise_on_mismatch=True):
    """Source fields against represented centered coefficients at fixed probes."""
    require(type(witness) is dict and type(witness.get("coefficients")) is dict,"Missing coefficient witness")
    source=read_line_source(raw,precision=precision);context=source.geometry["context"]
    interval=witness.get("interval")
    require(type(interval) is dict,"Coefficient audit needs a positive line interval")
    mid,_=_native_or_public(witness,"interval.lambda_mid",interval["lambda_mid"])
    half,_=_native_or_public(witness,"interval.lambda_halfwidth",interval["lambda_halfwidth"])
    mid,half=_mp(context,np.asarray(mid).item()),_mp(context,np.asarray(half).item())
    require(half>0,"Coefficient line halfwidth differs")
    coefficient_bounds=witness.get("coefficient_roundoff_bounds",{})
    keys=("lam","w","L","Bhat","Ghat_without_H","T","E_without_H","memoryhat")
    require(set(keys)<=set(witness["coefficients"]),"Coefficient source fields missing")
    checks={};failed=[];numeric_failed=[];budget_unresolved=[]
    for xi in (context.mpf(0),context.mpf(3)/7):
        lam=mid+half*xi
        fields=_fields(source,N,lam,context.mpf(".5"))
        expected={k:(lam if k=="lam" else fields[k]) for k in keys}
        for key in keys:
            coefficient,native=_native_or_public(witness,"coefficient."+key,witness["coefficients"][key])
            require(key in coefficient_bounds,"Coefficient arithmetic bounds missing")
            cbound,_=_native_or_public(witness,"coefficient_bound."+key,coefficient_bounds[key])
            require(coefficient.shape==cbound.shape,"Coefficient/bound shape differs")
            mpbounds=_mp_array(context,cbound)
            require(all(context.isfinite(v) and v>=0 for v in mpbounds.ravel()),
                    "Coefficient arithmetic bound is negative/nonfinite")
            value=_polynomial_value(context,coefficient,xi)
            bound=_polynomial_value(context,mpbounds,abs(xi))
            own=_polynomial_value(context,np.abs(_mp_array(context,coefficient)),abs(xi))
            exp=expected[key]
            if hasattr(exp,"rows"):
                exp=np.asarray([[exp[i,j] for j in range(exp.cols)] for i in range(exp.rows)],dtype=object)
                if value.ndim==1 and exp.shape[1]==1:exp=exp[:,0]
            else:exp=np.asarray(exp,dtype=object)
            require(value.shape==exp.shape,"Coefficient source field shape differs: "+key)
            for index in np.ndindex(value.shape):
                name=f"xi={context.nstr(xi,8)}.{key}"+str(index)
                passed,check=_scalar_check(context,value[index],exp[index],bound[index],own[index],native=native)
                cap=_mp(context,65536*np.finfo(np.longdouble).eps)*(own[index]+abs(exp[index]))
                budget_scope_resolved=bool(bound[index]<=cap)
                # This necessary own-scale observation cannot certify the
                # full construction operands of a cancelled source coefficient.
                check.update(passed=passed and budget_scope_resolved,
                    independent_coefficient_budget_scope_resolved=budget_scope_resolved,
                    coefficient_construction_operand_accuracy_certified=False)
                if not passed:numeric_failed.append(name)
                if not budget_scope_resolved:budget_unresolved.append(name)
                check["numeric_source_field_agreement_with_declared_budget"]=passed
                passed=check["passed"];checks[name]=check
                if not passed:failed.append(name)
    status="PASS" if not failed else "OPEN"
    if raise_on_mismatch:require(status=="PASS","Independent coefficient source field differs: "+", ".join(failed[:4]))
    return dict(scope=SCOPE,coefficient_source_status=status,checks=checks,mismatching_components=failed,
        source_field_numeric_agreement_status="PASS" if not numeric_failed else "OPEN",
        saved_coefficient_arithmetic_operand_scope_status="UNRESOLVED",
        saved_coefficient_own_field_scale_observation_status="PASS" if not budget_unresolved else "OPEN",
        saved_coefficient_operand_authentication_certified=False,
        arithmetic_budget_scope="own_field_scale_observation_does_not_authenticate_coefficient_construction_operands",
        numerically_mismatching_components=numeric_failed,
        unresolved_budget_scope_components=budget_unresolved,
        checked_component_count=len(checks),source_geometry_accuracy_certified=False,
        high_precision_direct_derivative_is_diagnostic_not_operative_substitution=True,
        angular_accuracy_certified=False,general_contraction_accuracy_certified=False,
        physical_contraction_certified=False,physical_creation_admitted=False,physical_release_admitted=False)


def _represented_intervals(context,witness):
    interval=witness.get("interval")
    require(type(interval) is dict,"Represented primitive needs a positive interval")
    def scalar(key):
        value,_=_native_or_public(witness,"interval."+key,interval[key])
        return _mp(context,np.asarray(value).item())
    mid,half=scalar("lambda_mid"),scalar("lambda_halfwidth")
    require(half>0 and mid-half>0,"Represented radial bounds differ")
    root=interval.get("gamma_root_xi")
    if root is None:
        side=_mp(context,interval.get("constant_side"))
        require(side in (context.mpf("-.5"),context.mpf(".5")),"Represented constant side differs")
        visits=[(context.mpf(-1),context.mpf(1),side)]
    else:
        phi,_=_native_or_public(witness,"interval.literal_phi_coefficients",interval["literal_phi_coefficients"])
        phi=_mp_array(context,phi)
        require(phi.shape==(2,) and phi[1]!=0,"Represented Gamma slope differs")
        orientation=context.mpf(".5") if phi[1]>0 else context.mpf("-.5")
        if "gamma_left_width_xi" in interval and interval["gamma_left_width_xi"] is not None:
            left,right=scalar("gamma_left_width_xi"),scalar("gamma_right_width_xi")
            require(left>0 and right>0,"Represented Gamma bank widths differ")
            require(abs(left+right-2)<=_mp(context,32*np.finfo(np.longdouble).eps),"Represented Gamma bank coverage differs")
            # The smaller endpoint distance retains a thin bank even when
            # its large complement rounded to exactly2 in native precision.
            cut=-1+left if left<=right else 1-right
            require(-1<cut<1,"Unresolved positive represented Gamma bank")
            visits=[(context.mpf(-1),cut,-orientation),(cut,context.mpf(1),orientation)]
        else:
            if "interval.side" in witness.get("native_exact_binary_ratio",{}):
                root=decode_native_array(witness,"interval.side").reshape(-1)[0]
            root=_mp(context,root)
            require(-1<root<1,"Represented Gamma root differs")
            visits=[(context.mpf(-1),root,-orientation),(root,context.mpf(1),orientation)]
    return mid,half,visits


def independent_polynomial_integral(witness,coefficients,*,side_power=0,precision=80):
    """MP antiderivative of a represented centered polynomial, not source geometry.

    Input coefficients are literal exact ratios or numeric scalars, with degree
    on the first axis.  Gamma bank widths are retained independently so a tiny
    positive side cannot disappear through a rounded root near +/-1.
    """
    require(side_power in (0,1,2) and type(side_power) is int,"Polynomial side power differs")
    context=_ctx(precision)
    _,half,visits=_represented_intervals(context,witness)
    coefficients=_mp_array(context,coefficients)
    require(1<=coefficients.ndim<=4 and 1<=len(coefficients)<=24,"Polynomial degree/shape differs")
    result=np.asarray(coefficients[0]*0,dtype=object)
    own=np.asarray(result.copy(),dtype=object)
    for degree,coefficient in enumerate(coefficients):
        for lo,hi,side in visits:
            moment=(hi**(degree+1)-lo**(degree+1))/(degree+1)
            unsigned=(abs(hi)**(degree+1)+abs(lo)**(degree+1))/(degree+1)
            result=result+half*side**side_power*moment*coefficient
            own=own+half*abs(side)**side_power*unsigned*np.abs(coefficient)
    return dict(value=np.asarray(result,dtype=object),absolute_operand_scale=np.asarray(own,dtype=object),
        high_precision_arithmetic_bound=np.asarray(own*context.eps*262144,dtype=object),
        scope="represented_polynomial_antiderivative_no_source_geometry_or_angular_certificate",
        precision_decimal_digits=precision,source_geometry_accuracy_certified=False,
        angular_accuracy_certified=False,physical_contraction_certified=False)


def _represented_fields(context,coefficients,xi,lam,side):
    fields={key:_polynomial_value(context,value,xi) for key,value in coefficients.items()}
    def matrix(key):
        value=fields[key]
        return context.matrix([[v for v in row] for row in value]) if value.ndim==2 else context.matrix(list(value))
    return dict(B=matrix("Bhat")/lam,G=side*matrix("Ghat_without_H")/lam,
        T=matrix("T"),E=side*matrix("E_without_H"),memory=matrix("memoryhat")/lam)


def audit_represented_primitive(raw,N,witness,*,added_basis=None,parent_basis=None,probes=(),precision=80,
                                raise_on_mismatch=True):
    """Independent positive quadrature of represented coefficient contractions.

    Order12 integrates every declared degree<=16 exactly in exact arithmetic.
    This lane uses the archived interval and coefficients, therefore it can
    diagnose primitive/contraction arithmetic while the true-source geometry
    or field-coefficient comparison remains OPEN.
    """
    source=read_line_source(raw,precision=precision);context=source.geometry["context"]
    require(source.density is not None,"Represented primitive needs retained mass/volume")
    mid,half,visits=_represented_intervals(context,witness)
    keys=("Bhat","Ghat_without_H","T","E_without_H","memoryhat")
    limits=dict(Bhat=2,Ghat_without_H=8,T=2,E_without_H=8,memoryhat=2)
    coefficients={}
    for key in keys:
        value,_=_native_or_public(witness,"coefficient."+key,witness["coefficients"][key])
        require(1<=len(value)<=limits[key],"Represented field degree exceeds declared polynomial: "+key)
        coefficients[key]=value
    parent=context.eye(18) if parent_basis is None else _basis(context,parent_basis,"Parent basis")
    added=None if added_basis is None else _basis(context,added_basis,"Added basis")
    total=own_total=None;names=None
    nodes,weights=_gauss(precision,12)
    represented_det,_=_native_or_public(witness,"interval.reference_determinant_m3",witness["interval"]["reference_determinant_m3"])
    represented_det=_mp(context,np.asarray(represented_det).item())
    require(represented_det>0,"Represented reference determinant differs")
    for lo,hi,side in visits:
        midpoint=(lo+hi)/2;width=(hi-lo)/2
        for node,weight in zip(nodes,weights):
            xi=midpoint+width*node;lam=mid+half*xi
            fields=_represented_fields(context,coefficients,xi,lam,side)
            full,own=_blocks(source,fields,context.eye(18),context.eye(18))
            n,values=_flatten("full.",full);_,operands=_flatten("full.",own)
            if added is not None:
                projected,own=_blocks(source,fields,parent,added)
                n1,v=_flatten("projected.",projected);_,o=_flatten("projected.",own)
                n+=n1;values+=v;operands+=o
            for probe in probes:
                row,own=_probe(source,fields,N,lam,probe)
                n1,v=_flatten("probe."+probe["name"]+".",row);_,o=_flatten("probe."+probe["name"]+".",own)
                n+=n1;values+=v;operands+=o
            if total is None:
                names=tuple(n);total=[context.mpf(0) for _ in values];own_total=list(total)
            require(tuple(n)==names,"Represented primitive schema changed")
            measure=half*width*weight*represented_det*lam*lam
            for i,(value,operand) in enumerate(zip(values,operands)):
                total[i]+=measure*value;own_total[i]+=measure*operand
    expected=dict(zip(names,total));own=dict(zip(names,own_total))
    savednames=witness.get("component_names")
    require(type(savednames) is list and len(savednames)==len(set(savednames)) and set(savednames)==set(expected),"Represented primitive component schema differs")
    values,native=_native_or_public(witness,"values",witness.get("values"))
    budgets,_=_native_or_public(witness,"budgets",witness.get("roundoff_component_bounds"))
    require(values.shape==budgets.shape==(len(savednames),),"Represented primitive values/budgets differ")
    checks={};failed=[]
    for name,value,budget in zip(savednames,values,budgets):
        require(_mp(context,budget)>=0,"Saved primitive arithmetic budget is negative")
        # The saved arithmetic enclosure includes construction and contraction
        # operands; this lane reports agreement within that declared enclosure.
        # It does not authenticate the producer's construction operand bound.
        passed,check=_scalar_check(context,value,expected[name],_mp(context,budget)+context.eps*262144*own[name],own[name],native=native)
        check["passed"]=passed;checks[name]=check
        if not passed:failed.append(name)
    status="PASS" if not failed else "OPEN"
    if raise_on_mismatch:require(status=="PASS","Represented polynomial primitive differs: "+", ".join(failed[:4]))
    return dict(scope="independent_represented_polynomial_contraction_primitive_fixed_geometry",
        represented_polynomial_primitive_status=status,checked_component_count=len(checks),
        component_checks=checks,mismatching_components=failed,quadrature_order=12,
        declared_contraction_maximum_degree=16,arithmetic_budget_operand_authentication_certified=False,
        source_geometry_accuracy_certified=False,angular_accuracy_certified=False,
        general_contraction_accuracy_certified=False,physical_contraction_certified=False,
        physical_creation_admitted=False,physical_release_admitted=False,available_for_creation_funding=False)


def audit_geometry_witness(raw,N,witness,*,precision=80):
    """Source-bound geometry observation independent of quadrature success."""
    passed,checks=_geometry_witness_check(raw,N,witness,precision)
    return dict(scope="independent_fixed_N_geometry_arithmetic_observation",
        geometry_status="PASS" if passed else "OPEN",geometry_checks=checks,
        source_geometry_accuracy_certified=False,angular_accuracy_certified=False,
        general_contraction_accuracy_certified=False,physical_contraction_certified=False,
        physical_creation_admitted=False,physical_release_admitted=False,available_for_creation_funding=False)

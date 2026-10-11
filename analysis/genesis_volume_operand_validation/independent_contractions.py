"""Exact selected-line contraction witnesses, without producer arithmetic.

The certificate describes a particular captured line.  It neither supplies a
forward error theorem for unseen lines nor authenticates the historical
producer's advertised operand bounds.  Its error numbers are exact rational
differences, so their verification does not accept arbitrary inflated bounds.
Source endpoint displacement is deliberately outside this selected-interval
certificate.  No planetary, clipping or mechanical producer is imported.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math

import numpy as np


BLOCKS = ("stiffness_cross", "stiffness_enriched", "mass_cross",
          "mass_enriched", "enrichment_force_n")
FORMAT = "genesis-selected-line-exact-contraction-certificate-0.1"
ZERO = Fraction(0)
ONE = Fraction(1)


class ContractionProofRefusal(ValueError):
    """A certificate or literal arithmetic operand is inconsistent."""


def _require(condition, message):
    if not condition:
        raise ContractionProofRefusal(message)


def _fraction(value):
    _require(not isinstance(value, (bool, np.bool_)), "Boolean arithmetic operand")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        try:
            return Fraction(*value.as_integer_ratio())
        except (ValueError, OverflowError) as error:
            raise ContractionProofRefusal("Nonfinite arithmetic operand") from error
    _require(isinstance(value, str), "Unsupported arithmetic operand")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ContractionProofRefusal("Invalid rational arithmetic operand") from error


def _array(value):
    raw = np.asarray(value, dtype=object)
    return np.asarray([_fraction(v) for v in raw.ravel()], dtype=object).reshape(raw.shape)


def _zeros(shape):
    return np.full(shape, ZERO, dtype=object)


def _identity(size):
    result = _zeros((size, size))
    for i in range(size):
        result[i, i] = ONE
    return result


def decode_native_array(witness, label):
    """Read only canonical exact binary ratios, never public float64 values."""
    record = witness.get("native_exact_binary_ratio", {}).get(label)
    _require(type(record) is dict and type(record.get("shape")) is list,
             "Missing native arithmetic operand: " + label)
    shape = record["shape"]
    _require(len(shape) <= 4 and all(type(n) is int and 0 <= n <= 10000 for n in shape),
             "Native shape is invalid")
    pairs = record.get("numerator_denominator")
    _require(type(pairs) is list and len(pairs) == math.prod(shape) <= 100000,
             "Native logical operand count differs")
    values = []
    for pair in pairs:
        _require(type(pair) is list and len(pair) == 2 and
                 all(type(v) is str and len(v) <= 4096 for v in pair),
                 "Native logical ratio is invalid")
        try:
            numerator, denominator = map(int, pair)
        except ValueError as error:
            raise ContractionProofRefusal("Native ratio needs integer strings") from error
        _require(denominator > 0 and denominator & (denominator - 1) == 0,
                 "Native denominator is not a binary power")
        value = Fraction(numerator, denominator)
        _require(str(value.numerator) == pair[0] and str(value.denominator) == pair[1],
                 "Native ratio is not canonical")
        values.append(value)
    return np.asarray(values, dtype=object).reshape(shape)


def _scalar(witness, key):
    value = decode_native_array(witness, "interval." + key)
    _require(value.shape == (), "Interval scalar shape differs: " + key)
    return value.item()


def polynomial_sum(left, right):
    left, right = _array(left), _array(right)
    shape = np.broadcast_shapes(left.shape[1:], right.shape[1:])
    result = _zeros((max(len(left), len(right)),) + shape)
    for i, coefficient in enumerate(left):
        result[i] += coefficient
    for i, coefficient in enumerate(right):
        result[i] += coefficient
    return result


def polynomial_product(left, right):
    left, right = _array(left), _array(right)
    shape = np.broadcast_shapes(left.shape[1:], right.shape[1:])
    result = _zeros((len(left) + len(right) - 1,) + shape)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i+j] += a*b
    return result


def polynomial_project(polynomial, basis):
    polynomial, basis = _array(polynomial), _array(basis)
    _require(polynomial.ndim == 3 and basis.ndim == 2 and
             polynomial.shape[-1] == basis.shape[0], "Polynomial projection shape differs")
    return np.asarray([coefficient @ basis for coefficient in polynomial], dtype=object)


def polynomial_bilinear(left, middle, right):
    left, middle, right = _array(left), _array(middle), _array(right)
    _require(left.ndim == right.ndim == 3 and middle.shape ==
             (left.shape[1], right.shape[1]), "Polynomial bilinear shape differs")
    result = _zeros((len(left)+len(right)-1, left.shape[-1], right.shape[-1]))
    transformed = [coefficient.T @ middle for coefficient in left]
    for i, a in enumerate(transformed):
        for j, b in enumerate(right):
            result[i+j] += a @ b
    return result


def _literal_source_gamma(raw, N, midpoint, halfwidth):
    section = raw["section"]
    # These two represented binary64 sums are the frozen source convention.
    C = np.asarray(section["triangle_m"], float) + np.asarray(section["slope_P_m"], float)
    D = np.asarray(section["displacement_A_m"], float) - np.asarray(section["slope_P_m"], float)
    C, D = _array(C), _array(D)
    N, normal = _array(N), _array(section["normal_xyz"])
    u, origin = N @ C, N @ D
    slope = sum(normal*u, ZERO)
    intercept = sum(normal*origin, ZERO)-_fraction(section["offset_m"])
    return slope*midpoint+intercept, slope*halfwidth


def exact_moments(witness, degree, side_power=0, *, gamma_mode="native_formula", raw=None, N=None):
    """Exact ordinary, H and H² moments for a declared captured chart.

    ``native_formula`` evaluates the frozen paired moment expression in exact
    rational arithmetic, retaining both the root and endpoint-distance bits.
    ``literal_source`` integrates one coherent exact source Gamma cut on the
    represented interval.  It never silently renormalizes the material N.
    """
    _require(type(degree) is int and 0 <= degree <= 16 and
             type(side_power) is int and side_power in (0, 1, 2),
             "Moment degree/side power differs")
    half = _scalar(witness, "lambda_halfwidth")
    _require(half > 0, "Selected radial interval is not positive")
    moments = [2*half/Fraction(k+1) if k % 2 == 0 else ZERO for k in range(degree+1)]
    if side_power == 0:
        return tuple(moments)
    if side_power == 2:
        return tuple(v/4 for v in moments)
    _require(gamma_mode in ("native_formula", "literal_source"), "Gamma moment mode differs")
    if gamma_mode == "literal_source":
        _require(raw is not None and N is not None, "Literal Gamma needs source operands")
        phi0, phi1 = _literal_source_gamma(raw, N, _scalar(witness, "lambda_mid"), half)
        _require(phi0 != 0 or phi1 != 0, "Positive interval lies in literal Gamma")
        if phi1 == 0 or abs(phi0) >= abs(phi1):
            side = Fraction(1, 2) if phi0 > 0 else Fraction(-1, 2)
            return tuple(side*v for v in moments)
        root, orientation = -phi0/phi1, ONE if phi1 > 0 else -ONE
        return tuple(orientation*half*((ONE if k % 2 else ZERO)-root**(k+1))/Fraction(k+1)
                     for k in range(degree+1))
    interval = witness.get("interval")
    _require(type(interval) is dict, "Missing interval descriptor")
    if interval.get("gamma_root_xi") is None:
        side_packet = decode_native_array(witness,"interval.side")
        _require(side_packet.shape == (1,), "Native constant bank shape differs")
        side = side_packet[0]
        _require(side in (Fraction(-1, 2), Fraction(1, 2)), "Constant bank side differs")
        _require(side == _fraction(interval.get("constant_side")),
                 "Public/native constant bank side differs")
        return tuple(side*v for v in moments)
    root_packet = decode_native_array(witness, "interval.side")
    _require(root_packet.shape == (1,), "Native Gamma root shape differs")
    root = root_packet[0]
    phi = decode_native_array(witness, "interval.literal_phi_coefficients")
    _require(phi.shape == (2,) and phi[1] != 0 and abs(root) <= 1,
             "Native Gamma orientation/root differs")
    orientation = ONE if phi[1] > 0 else -ONE
    distance = _scalar(witness, "gamma_endpoint_distance_xi")
    _require(0 <= distance <= 1, "Native Gamma endpoint distance differs")
    return tuple(-orientation*half*root**(k+1)/Fraction(k+1) if k % 2 == 0 else
                 orientation*half*distance*sum((abs(root)**j for j in range(k+1)), ZERO)/Fraction(k+1)
                 for k in range(degree+1))


def exact_polynomial_integral(polynomial, moments):
    polynomial = _array(polynomial)
    _require(1 <= len(polynomial) <= 17 and len(moments) >= len(polynomial),
             "Polynomial integral degree differs")
    result = _zeros(polynomial.shape[1:])
    for coefficient, moment in zip(polynomial, moments):
        result += coefficient*_fraction(moment)
    return result


def _alias(raw, primary, alternate):
    _require(primary in raw or alternate in raw, "Missing source operand: " + primary)
    return raw[primary] if primary in raw else raw[alternate]


def _memory_coefficients(raw, N, lam, e0, e1):
    # The angular inverse is independently duplicated in the accepted binary64
    # convention, then treated as an exact literal construction operand.
    sample = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    angular = _array(N) @ _array(np.linalg.inv(sample))
    first, second = angular @ _array(e0), angular @ _array(e1)
    return np.asarray((lam[0]*first+second, lam[1]*first), dtype=object)


def _weights(raw, witness, mode):
    section = raw["section"]
    mass, volume = section["mass_kg"], section["volume_m3"]
    _require(_fraction(mass) > 0 and _fraction(volume) > 0, "Original density differs")
    if mode == "native_expression":
        det = _scalar(witness, "reference_determinant_m3")
        # Reproduce only the two frozen scalar factor operations.  Their exact
        # output bits are inputs to the exact represented-polynomial lane.
        det_native = np.longdouble(det.numerator)/np.longdouble(det.denominator)
        density_native = np.longdouble(mass)/np.longdouble(volume)
        mass_factor = _fraction(det_native*density_native)
    elif mode == "literal_source":
        X = _array(section["triangle_m"])
        det = sum((X[0,j]*(X[1,(j+1)%3]*X[2,(j+2)%3]-X[1,(j+2)%3]*X[2,(j+1)%3])
                   for j in range(3)), ZERO)
        mass_factor = det*_fraction(mass)/_fraction(volume)
    else:
        raise ContractionProofRefusal("Contraction weight mode differs")
    _require(det > 0, "Reference determinant differs")
    return det, mass_factor


def _projector(value, name):
    value = _array(value)
    _require(value.ndim == 2 and value.shape[0] == 18 and 1 <= value.shape[1] <= 18,
             name+" shape differs")
    return value


def _probe_vector(probe, primary, alternate, default=None):
    value = probe.get(primary, probe.get(alternate, default))
    _require(value is not None, "Missing probe operand: "+primary)
    value = _array(value)
    _require(value.shape == (18,), "Probe vector shape differs")
    return value[:, None]


def exact_line_contractions(raw, N, witness, *, coefficients=None, added_basis=None,
                            parent_basis=None, probes=(), gamma_mode="native_formula",
                            weight_mode="native_expression"):
    """Return exact Fraction values for the complete declared component schema.

    Coefficients supplied by the independent source builder are accepted only
    as arithmetic operands here; certificate construction below rebuilds them.
    All 18 columns are retained before products and declared projections.
    """
    _require(type(raw) is dict and np.shape(N) == (3,), "Source/N shape differs")
    N = _array(N)
    _require(min(N) >= 0 and abs(sum(N, ZERO)-ONE) <= _fraction(128*np.finfo(float).eps),
             "Material N leaves the original chart")
    required = ("lam", "Bhat", "Ghat_without_H", "T", "E_without_H", "memoryhat")
    if coefficients is None:
        f = {key: decode_native_array(witness, "coefficient."+key) for key in required}
    else:
        _require(set(required) <= set(coefficients), "Source coefficient families missing")
        f = {key: _array(coefficients[key]) for key in required}
    _require(f["lam"].shape == (2,) and all(f[key].shape[1:] == shape for key, shape in
        (("Bhat", (6,18)), ("Ghat_without_H", (6,18)), ("T", (3,18)),
         ("E_without_H", (3,18)), ("memoryhat", (6,)))), "Coefficient family shapes differ")
    C = _array(_alias(raw, "elasticity_pa", "C"))
    _require(C.shape == (6,6), "Elasticity shape differs")
    det, mass_factor = _weights(raw, witness, weight_mode)
    moments = {side: exact_moments(witness, 16, side, gamma_mode=gamma_mode, raw=raw, N=N)
               for side in (0,1,2)}
    lam2 = polynomial_product(f["lam"], f["lam"])
    lam2 = lam2.reshape((len(lam2),1,1))
    def integral(polynomial, side, factor):
        return factor*exact_polynomial_integral(polynomial, moments[side])
    def blocks(V, W):
        B, G = polynomial_project(f["Bhat"], V), polynomial_project(f["Ghat_without_H"], W)
        T, E = polynomial_project(f["T"], V), polynomial_project(f["E_without_H"], W)
        memory = f["memoryhat"][:,:,None]
        return {
            "stiffness_cross": integral(polynomial_bilinear(B,C,G),1,det),
            "stiffness_enriched": integral(polynomial_bilinear(G,C,G),2,det),
            "mass_cross": integral(polynomial_product(lam2,polynomial_bilinear(T,_identity(3),E)),1,mass_factor),
            "mass_enriched": integral(polynomial_product(lam2,polynomial_bilinear(E,_identity(3),E)),2,mass_factor),
            "enrichment_force_n": integral(polynomial_bilinear(G,C,memory),1,det)[:,0],
        }
    result = {}
    def append(prefix, values):
        for key, value in values.items():
            value = np.asarray(value,dtype=object)
            for index in np.ndindex(value.shape):
                suffix = "["+",".join(map(str,index))+"]" if index else ""
                result[prefix+"."+key+suffix] = value[index]
    append("full", blocks(_identity(18), _identity(18)))
    if added_basis is not None:
        V = _identity(18) if parent_basis is None else _projector(parent_basis,"Parent basis")
        append("projected", blocks(V,_projector(added_basis,"Added basis")))
    _require(len(probes) <= 16 and len({p.get("name") for p in probes}) == len(probes),
             "Declared probe names/count differ")
    for probe in probes:
        _require(isinstance(probe.get("name"),str) and probe["name"], "Probe name differs")
        q = _probe_vector(probe,"parent_increment_m","q")
        a = _probe_vector(probe,"enrichment_increment_m","a")
        z = _probe_vector(probe,"mean_z_m","mean_z",np.zeros(18))
        Bq, Ga, Gz = [polynomial_project(f[key],basis) for key,basis in
                      (("Bhat",q),("Ghat_without_H",a),("Ghat_without_H",z))]
        Tq, Ea = polynomial_project(f["T"],q), polynomial_project(f["E_without_H"],a)
        e0 = probe.get("mean_e0",_alias(raw,"memory0","e0"))
        e1 = probe.get("mean_e1",_alias(raw,"memory1","e1"))
        mean = _memory_coefficients(raw,N,f["lam"],e0,e1)[:,:,None]
        definitions = (
            ("cross_parent_work_j",Bq,C,Gz,1,det,False),
            ("cross_added_work_j",Ga,C,mean,1,det,False),
            ("self_work_j",Ga,C,Gz,2,det,False),
            ("stiffness_cross_bilinear_j",Bq,C,Ga,1,det,False),
            ("stiffness_self_bilinear_j",Ga,C,Ga,2,det,False),
            ("mass_cross_bilinear_kg_m2",Tq,_identity(3),Ea,1,mass_factor,True),
            ("mass_self_bilinear_kg_m2",Ea,_identity(3),Ea,2,mass_factor,True),
        )
        values = {}
        for name,left,middle,right,side,factor,with_lam2 in definitions:
            p = polynomial_bilinear(left,middle,right)
            if with_lam2:
                p = polynomial_product(lam2,p)
            values[name] = integral(p,side,factor)[0,0]
        values["added_parent_force_n"] = integral(polynomial_bilinear(f["Bhat"],C,Gz),1,det)[:,0]
        values["enrichment_force_n"] = (
            integral(polynomial_bilinear(f["Ghat_without_H"],C,mean),1,det)[:,0]+
            integral(polynomial_bilinear(f["Ghat_without_H"],C,Gz),2,det)[:,0])
        append("probe."+probe["name"],values)
    return result


def _ratio(value):
    value = _fraction(value)
    return [str(value.numerator),str(value.denominator)]


def _plain(value):
    if isinstance(value, Fraction):
        return {"fraction":_ratio(value)}
    if isinstance(value,np.ndarray):
        return _plain(value.tolist())
    if isinstance(value,(np.integer,np.floating)):
        return {"fraction":_ratio(value)}
    if isinstance(value,dict):
        return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()


def build_contraction_certificate(raw,N,witness,*,added_basis=None,parent_basis=None,probes=()):
    """Build unique exact residuals from source bits and one saved line packet."""
    from .independent_coefficients import exact_source_coefficients
    native_N = decode_native_array(witness,"interval.barycentric")
    _require(native_N.shape == (3,) and np.array_equal(native_N,_array(N)),
             "Captured native material N differs")
    names = witness.get("component_names")
    values = decode_native_array(witness,"values")
    budgets = decode_native_array(witness,"budgets")
    _require(type(names) is list and len(names) == len(set(names)) and values.shape == budgets.shape == (len(names),),
             "Saved contraction component schema differs")
    source_coefficients = exact_source_coefficients(raw,N,_scalar(witness,"lambda_mid"),
                                                  _scalar(witness,"lambda_halfwidth"))
    options = dict(added_basis=added_basis,parent_basis=parent_basis,probes=probes)
    represented = exact_line_contractions(raw,N,witness,**options)
    source_native = exact_line_contractions(raw,N,witness,coefficients=source_coefficients,**options)
    source_literal = exact_line_contractions(raw,N,witness,coefficients=source_coefficients,
                         gamma_mode="literal_source",weight_mode="literal_source",**options)
    _require(set(names) == set(represented) == set(source_native) == set(source_literal),
             "Reconstructed contraction schema differs")
    components = {}
    budget_failures = []
    selected_interval_budget_failures = []
    for name,value,budget in zip(names,values,budgets):
        _require(budget >= 0,"Saved arithmetic bound is negative")
        primitive = abs(value-represented[name])
        coefficient = abs(represented[name]-source_native[name])
        gamma_weights = abs(source_native[name]-source_literal[name])
        total = abs(value-source_literal[name])
        enclosing = primitive+coefficient+gamma_weights
        _require(total <= enclosing,"Exact triangle enclosure is inconsistent")
        if primitive > budget:
            budget_failures.append(name)
        if enclosing > budget:
            selected_interval_budget_failures.append(name)
        components[name] = dict(
            saved_native_value=_ratio(value),
            exact_represented_native_expression=_ratio(represented[name]),
            exact_source_coefficients_native_expression=_ratio(source_native[name]),
            exact_source_gamma_weights_on_represented_interval=_ratio(source_literal[name]),
            represented_polynomial_arithmetic_residual=_ratio(primitive),
            source_coefficient_contraction_difference=_ratio(coefficient),
            source_gamma_and_weights_contraction_difference=_ratio(gamma_weights),
            exact_selected_interval_residual=_ratio(total),
            exact_selected_interval_triangle_enclosure=_ratio(enclosing),
            historical_saved_budget_sufficient_for_represented_arithmetic=bool(primitive<=budget),
            historical_saved_budget_sufficient_for_selected_interval_enclosure=bool(enclosing<=budget))
    body = dict(format=FORMAT,status="PASS",checked_component_count=len(names),
        input_binding_sha256=_digest(dict(raw=raw,N=N,witness=witness,**options)),
        component_names=names,components=components,
        canonical_exact_error_certificate=True,
        arbitrary_error_bound_inflation_accepted=False,
        exact_captured_selected_interval_error_certified=True,
        historical_saved_bound_operand_authentication_certified=False,
        historical_saved_bound_represented_sufficiency_status="PASS" if not budget_failures else "OPEN",
        historical_saved_bound_represented_sufficiency_failures=budget_failures,
        historical_saved_bound_selected_interval_sufficiency_status="PASS" if not selected_interval_budget_failures else "OPEN",
        historical_saved_bound_selected_interval_sufficiency_failures=selected_interval_budget_failures,
        source_endpoint_displacement_accuracy_certified=False,
        selected_line_uniform_forward_bound_certified=False,
        angular_node_uniform_error_certified=False,
        angular_accuracy_certified=False,general_contraction_accuracy_certified=False,
        physical_contraction_certified=False,physical_creation_admitted=False,
        physical_release_admitted=False,available_for_creation_funding=False,
        source_history_reinitialized=False)
    body["certificate_sha256"] = _digest(body)
    return body


def verify_contraction_certificate(raw,N,witness,certificate,*,added_basis=None,parent_basis=None,probes=()):
    """Recompute the exact certificate; even consistently resealed inflation fails."""
    _require(type(certificate) is dict,"Contraction certificate must be an object")
    recomputed = build_contraction_certificate(raw,N,witness,added_basis=added_basis,
                                             parent_basis=parent_basis,probes=probes)
    _require(certificate == recomputed,"Canonical exact contraction certificate differs")
    return dict(status="PASS",checked_component_count=recomputed["checked_component_count"],
                certificate_sha256=recomputed["certificate_sha256"],
                source_endpoint_displacement_accuracy_certified=False,
                angular_accuracy_certified=False,physical_contraction_certified=False,
                available_for_creation_funding=False)

"""Complete declared COMMON101 state/rate force with finite owner charts.

The actual cold56/preparation energy, HOT30 implicit mean, sixteen original
material contacts and the fixed-reference68 basal rule share one literal
state. This is an endpoint operator, not a recovered physical prestress/work
handover, an equilibrium certificate or an accepted temporal continuation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
from math import factorial
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_common_carrier_source_work import (
    _payload, _digest, _typed_guard, _fraction, _interval, _vector, _matrix,
    _packet, _round, _inverse, evaluate_native_cohesive_series_loading,
)
from .geometric_native_common_mixed_force import _mf_original_source_gradient
from .geometric_native_common_preparation_contacts import _source_law
from .geometric_native_archive_material_carrier import _sin_cos
from .geometric_native_common_hot_mean import (
    validate_native_common_hot_mean, evaluate_native_common_hot_mean,
    native_common_hot_mean_response_to_dict,
)
from . import geometric_native_common_basal_trace as trace_api
from .geometric_native_common_basal_loading import _trace_algebra

FORMAT = 'native-common-general-state-force-system-1'
RESPONSE_FORMAT = 'native-common-general-state-force-response-1'
SCOPE = 'NEW_DECLARED_COMMON101_FINITE_OWNER_STATE_RATE_OPERATOR'
SOURCE_PINS = (
    ('907', '5e5555712e57fd69156ee475fe5feb161baef3cb96dd911a443228d160f53005',
     'bba7f1f4913d6bc47f5f41beff0a279be3c203993ddd0b0b06979c719abeda1a'),
    ('895', 'fbafd092a4eb98472e60f79665e1e0fe8e9e1c2c2b9d3679344a32e59593cc89',
     '27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70'),
    ('903', '0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035',
     'c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c'),
)
REMAINING_CONDITIONS = (
    'Physical material/force/prestress/work handover, full preparation and load work, general-state mixed operator and NEW equilibrium',
    'Whole-clock all-pair admission, including reverse C5 to C1',
    'Accepted continuation with independent temporal path work, balances and restart',
)
_SYSTEMS, _CONTEXTS, _RESPONSES, _EVALUATIONS, _STATE_EVALUATIONS = {}, {}, {}, {}, {}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _body(value):
    return {f.name: getattr(value, f.name) for f in fields(value)
            if f.name != 'binding_sha256'}


def _remember(value, registry):
    value = replace(value, binding_sha256=_digest(_body(value)))
    registry[value.binding_sha256] = _typed_guard(value)
    return value


def _guard(value, cls, registry):
    if (type(value) is not cls or _digest(_body(value)) != value.binding_sha256
            or registry.get(value.binding_sha256) != _typed_guard(value)):
        raise ValueError('An altered or resealed general state lacks its registered complete source')
    return value


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError('General force precision must be a literal integer in [96,512]')


def _pair(i, j):
    return (i, j) if i <= j else (j, i)


def _square(value):
    return I(0 if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


class _Jet:
    """Sparse complete twojet on the authentic101 coordinate indices."""
    __slots__ = ('value', 'gradient', 'hessian', 'precision')

    def __init__(self, value, gradient=None, hessian=None, precision=96):
        _precision(precision)
        self.precision = precision
        self.value = _round(value if type(value) is I else I(value), precision)
        self.gradient, self.hessian = {}, {}
        for index, item in (gradient or {}).items():
            if type(index) is not int or not 0 <= index < 101:
                raise ValueError('A general derivative must name a real COMMON101 coordinate')
            item = _round(item if type(item) is I else I(item), precision)
            if item != I(0):
                self.gradient[index] = item
        for pair, item in (hessian or {}).items():
            if (type(pair) is not tuple or len(pair) != 2 or pair[0] > pair[1]
                    or any(type(i) is not int or not 0 <= i < 101 for i in pair)):
                raise ValueError('A general Hessian needs an ordered actual coordinate pair')
            item = _round(item if type(item) is I else I(item), precision)
            if item != I(0):
                self.hessian[pair] = item

    @classmethod
    def variable(cls, value, index, precision=96):
        return cls(value, {index: I(1)}, precision=precision)

    def _coerce(self, other):
        if type(other) is _Jet:
            if other.precision != self.precision:
                raise ValueError('General twojets cannot mix precisions')
            return other
        return _Jet(other, precision=self.precision)

    def __add__(self, other):
        other = self._coerce(other)
        return _Jet(self.value+other.value,
            {i: self.gradient.get(i, I(0))+other.gradient.get(i, I(0))
                for i in self.gradient.keys() | other.gradient.keys()},
            {p: self.hessian.get(p, I(0))+other.hessian.get(p, I(0))
                for p in self.hessian.keys() | other.hessian.keys()}, self.precision)

    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.value, {i: -v for i, v in self.gradient.items()},
                    {p: -v for p, v in self.hessian.items()}, self.precision)

    def __sub__(self, other):
        return self+(-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other)+(-self)

    def __mul__(self, other):
        other = self._coerce(other)
        if ((self.value == I(0) and not self.gradient and not self.hessian)
                or (other.value == I(0) and not other.gradient and not other.hessian)):
            return _Jet(0, precision=self.precision)
        indices = self.gradient.keys() | other.gradient.keys()
        pairs = set(self.hessian) | set(other.hessian)
        pairs.update(_pair(i, j) for i in self.gradient for j in other.gradient)
        return _Jet(self.value*other.value,
            {i: self.gradient.get(i, I(0))*other.value+self.value*other.gradient.get(i, I(0))
                for i in indices},
            {p: self.hessian.get(p, I(0))*other.value+self.value*other.hessian.get(p, I(0))
                +self.gradient.get(p[0], I(0))*other.gradient.get(p[1], I(0))
                +self.gradient.get(p[1], I(0))*other.gradient.get(p[0], I(0))
                for p in pairs}, self.precision)

    __rmul__ = __mul__

    def compose(self, value, first, second):
        ids = sorted(self.gradient)
        pairs = set(self.hessian) | {(i, j) for k, i in enumerate(ids) for j in ids[k:]}
        return _Jet(value, {i: first*v for i, v in self.gradient.items()},
            {p: first*self.hessian.get(p, I(0))
                +second*self.gradient.get(p[0], I(0))*self.gradient.get(p[1], I(0))
                for p in pairs}, self.precision)

    def square(self):
        return self.compose(_square(self.value), 2*self.value, I(2))

    def sqrt(self):
        if self.value.lo <= 0:
            raise ValueError('A general norm must stay strictly separated from zero')
        root = interval_sqrt(self.value, self.precision+16)
        return self.compose(root, 1/(2*root), -1/(4*root*root*root))

    def inverse(self):
        if self.value.lo <= 0 <= self.value.hi:
            raise ValueError('A general reciprocal cannot cross zero')
        r = 1/self.value
        return self.compose(r, -r*r, 2*r*r*r)

    def __truediv__(self, other):
        return self*self._coerce(other).inverse()

    def __rtruediv__(self, other):
        return self._coerce(other)*self.inverse()


def _export(jet):
    return dict(value_bounds=_payload(jet.value),
        first_derivative_bounds=[[i, _payload(v)] for i, v in sorted(jet.gradient.items())],
        second_derivative_bounds=[[i, j, _payload(v)] for (i, j), v in sorted(jet.hessian.items())])


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), _Jet(0, precision=a[0].precision))


def _mv(a, b):
    return tuple(_dot(row, b) for row in a)


def _mm(a, b):
    return tuple(tuple(_dot(a[i], tuple(b[k][j] for k in range(3)))
                       for j in range(3)) for i in range(3))


def _constant_matrix(a, precision):
    return tuple(tuple(_Jet(v, precision=precision) for v in row) for row in a)


def _identity(precision):
    return _constant_matrix(tuple(tuple(int(i == j) for j in range(3)) for i in range(3)), precision)


def _skew(x):
    zero = _Jet(0, precision=x[0].precision)
    return ((zero, -x[2], x[1]), (x[2], zero, -x[0]), (-x[1], x[0], zero))


def _owner_rotation(theta, owner, precision):
    """Finite exp(skew(theta)) and its true chart derivatives, also at zero.

    An induced-infinity-norm series tail encloses the value and the first
    two derivatives separately. No division by the vanishing angle occurs.
    """
    if (type(theta) is not tuple or len(theta) != 3 or any(type(v) is not F for v in theta)
            or owner not in (0, 1)):
        raise ValueError('Finite owner charts need three literal radians and an authentic owner')
    ids = tuple(3*owner+i for i in range(3))
    K = _skew(tuple(_Jet.variable(v, i, precision) for v, i in zip(theta, ids)))
    b = max(sum((abs(v) for k, v in enumerate(theta) if k != i), F(0)) for i in range(3))
    if b > F(1, 4):
        raise ValueError('The declared finite owner chart exceeds its exact quarter-radian norm budget')
    term = total = _identity(precision)
    if b == 0:
        total = tuple(tuple(total[i][j]+K[i][j] for j in range(3)) for i in range(3))
        K2 = _mm(K, K)
        total = tuple(tuple(total[i][j]+K2[i][j]/2 for j in range(3)) for i in range(3))
        n, tails = 2, (F(0), F(0), F(0))
    else:
        target = F(1, 10**(precision+8))
        for n in range(1, precision+65):
            product = _mm(term, K)
            term = tuple(tuple(v/n for v in row) for row in product)
            total = tuple(tuple(total[i][j]+term[i][j] for j in range(3)) for i in range(3))
            if n < 2:
                continue
            tails = tuple(b**(n+1-r)/factorial(n+1-r)/(1-b/F(n+2-r)) for r in range(3))
            if max(tails) <= target:
                break
        else:
            raise ValueError('Finite owner exponential exhausted its declared derivative-tail budget')
        total = tuple(tuple(_Jet(v.value+I(-tails[0], tails[0]),
            {i: v.gradient.get(i, I(0))+I(-tails[1], tails[1]) for i in ids},
            {_pair(i, j): v.hessian.get(_pair(i, j), I(0))+I(-tails[2], tails[2])
                for k, i in enumerate(ids) for j in ids[k:]}, precision)
            for v in row) for row in total)
    receipt = dict(owner=owner, coordinate_indices=ids, coordinates_radians=_payload(theta),
        method='EXACT_LINEAR_GENERATOR_INTERVAL_TAYLOR_WITH_VALUE_AND_TWO_DERIVATIVE_TAILS',
        generator_infinity_norm_upper=_payload(b), taylor_term_count=n,
        value_first_second_tail_upper=_payload(tails), zero_chart_exact_second_order_identity=b == 0,
        tail_theorem='b^(N+1-r)/(N+1-r)!/(1-b/(N+2-r)); r=0,1,2',
        rotation_twojet=[[_export(v) for v in row] for row in total])
    return total, receipt


def _signed_sin_cos(value, precision):
    if value.lo < -2 or value.hi > 2:
        raise ValueError('The declared twist must keep its angle inside [-2,2] radians')
    if value.lo >= 0:
        return _sin_cos(value, precision)
    if value.hi <= 0:
        s, c = _sin_cos(-value, precision)
        return -s, c
    s, c = _sin_cos(I(0, max(-value.lo, value.hi)), precision)
    return I(-s.hi, s.hi), c


def _fixed_rotation(alpha, omega, precision):
    """Rodrigues as a scalar analytic twojet for a fixed genuine generator."""
    norm2 = sum((v*v for v in omega), F(0))
    if norm2 == 0:
        return _identity(precision)
    norm = interval_sqrt(I(norm2), precision+16)
    angle = alpha*norm
    sinv, cosv = _signed_sin_cos(angle.value, precision+16)
    sine = angle.compose(sinv, cosv, -sinv)
    cosine = angle.compose(cosv, -sinv, -cosv)
    W = _skew(tuple(_Jet(v, precision=precision) for v in omega))
    W2 = _mm(W, W)
    return tuple(tuple(_Jet(int(i == j), precision=precision)
        +sine*W[i][j]/norm+(1-cosine)*W2[i][j]/norm2 for j in range(3)) for i in range(3))


def _prepared_fields(x, A, preparation, lam, precision):
    """Full original-gradient twojet; its body curvature includes D³T."""
    d = preparation.declaration
    radius = _dot(x, x).sqrt()
    tau = (d.radius_m**3-radius*radius*radius/d.fixed_radial_scale**3)/(3*d.radius_m**2)
    gradient_tau = tuple(-radius*v/(d.fixed_radial_scale**3*d.radius_m**2) for v in x)
    Q = _fixed_rotation(lam*tau, d.angular_rate_per_physical_volume_depth_m, precision)
    W = _skew(tuple(_Jet(v, precision=precision) for v in d.angular_rate_per_physical_volume_depth_m))
    wx = _mv(W, x)
    Z = tuple(tuple(_Jet(int(i == j), precision=precision)+lam*wx[i]*gradient_tau[j]
                    for j in range(3)) for i in range(3))
    DT = _mm(Q, Z)
    position, original = _mv(Q, x), _mm(DT, A)
    return position, original, DT, dict(
        physical_volume_depth_twojet=_export(tau),
        physical_volume_depth_gradient_twojet=[_export(v) for v in gradient_tau],
        preparation_rotation_twojet=[[_export(v) for v in row] for row in Q],
        preparation_spatial_gradient_twojet=[[_export(v) for v in row] for row in DT],
        original_cartesian_gradient_twojet=[[_export(v) for v in row] for row in original],
        body_body_original_gradient_curvature_included=True,
        mixed_body_preparation_original_gradient_curvature_included=True,
        physical_depth_evaluated_at_actual_body_position=True,
        map_jacobian_exact=_payload(F(1)))


def _cofactor(A):
    result = []
    for i in range(3):
        row = []
        for j in range(3):
            rr, cc = [k for k in range(3) if k != i], [k for k in range(3) if k != j]
            minor = A[rr[0]][cc[0]]*A[rr[1]][cc[1]]-A[rr[0]][cc[1]]*A[rr[1]][cc[0]]
            row.append(minor if (i+j) % 2 == 0 else -minor)
        result.append(tuple(row))
    return tuple(result)


def _normal(A, material_normal, precision):
    N = tuple(_Jet(v, precision=precision) for v in material_normal)
    covector = _mv(_cofactor(A), N)
    norm = _dot(covector, covector).sqrt()
    normal = tuple(v/norm for v in covector)
    return normal, dict(material_normal_bounds=_payload(material_normal),
        cofactor_normal_twojet=[_export(v) for v in covector],
        cofactor_normal_norm_twojet=_export(norm), unit_normal_twojet=[_export(v) for v in normal],
        whole_point_normal_norm_strictly_positive=True,
        full_original_gradient_body_and_preparation_curvature_included=True,
        current_area_energy_multiplier_added=False)


def _cold_api():
    from . import geometric_native_common_coupled_cold_energy as api
    return api


@dataclass(frozen=True, slots=True)
class NativeCommonGeneralForceSystem:
    coordinate_order: tuple
    coordinate_units: tuple
    coordinate_rate_units: tuple
    generalized_force_units: tuple
    cold_energy_system_binding_sha256: str
    cold_material_carrier_binding_sha256: str
    hot_mean_system_binding_sha256: str
    basal_trace_system_binding_sha256: str
    preparation_binding_sha256s: tuple
    contact_zero_authority_binding_sha256: str
    complete_source_raw_sha256s: tuple
    complete_source_packet_sha256s: tuple
    source_clocks_s: tuple
    provenance: str
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    complete_declared_general_state_operator_defined: bool = True
    general_finite_owner_state_chart_defined: bool = True
    general_state_full_physics_operator_certified: bool = False
    physical_source_prestress_work_handover_certified: bool = False
    reference_preparation_work_paid: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    physical_material_admission_authorized: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    continuum_whole_base_quadrature_certified: bool = False
    full_system_temporal_path_work_certified: bool = False
    accepted_old_native_intervals: int = 1
    accepted_new_native_intervals: int = 0
    clocks_advanced: bool = False
    remaining_condition_count: int = 3
    remaining_conditions: tuple = REMAINING_CONDITIONS


def bind_native_common_general_force(source907, source895, source903, *, cold_carrier,
        hot_system, preparations, basal_trace_system, cold_energy_system, provenance, precision=96):
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError('The new complete operator requires explicit provenance')
    packets = []
    for packet, (_, rawpin, seal) in zip((source907, source895, source903), SOURCE_PINS):
        _packet(packet, seal)
        encoded = _json(packet)
        if hashlib.sha256((encoded+'\n').encode()).hexdigest() != rawpin:
            raise ValueError('A complete immutable general-force source packet changed')
        packets.append(encoded)
    cold_api = _cold_api()
    cold_api.validate_native_common_coupled_cold_energy(cold_energy_system)
    cold_context = cold_api.native_common_coupled_cold_energy_context(cold_energy_system)
    validate_native_common_hot_mean(hot_system)
    trace_api.validate_native_common_basal_trace(basal_trace_system)
    trace_context = trace_api.native_common_basal_trace_loading_context(basal_trace_system)
    order = tuple(tuple(v) for v in source907['mixed_force_system']['coordinate_order'])
    expected = order+(('preparation', 'cold_depth_twist', 'lambda'),)
    if (len(order) != 100 or len(set(expected)) != 101 or basal_trace_system.coordinate_order != expected
            or cold_energy_system.coordinate_order != tuple(order[36:92])+(expected[100],)
            or cold_energy_system.cold_carrier_binding_sha256 != cold_carrier.binding_sha256
            or cold_context['cold_carrier'].binding_sha256 != cold_carrier.binding_sha256
            or tuple(p.binding_sha256 for p in preparations) != cold_energy_system.preparation_binding_sha256s
            or basal_trace_system.cold_carrier_binding_sha256 != cold_carrier.binding_sha256
            or basal_trace_system.hot_system_binding_sha256 != hot_system.binding_sha256):
        raise ValueError('All genuine cold/HOT/preparation/basal ports must share the same complete catalogue')
    authority = [row['authority'] for row in source907['contact_field_authorities']
                 if row['label'] == 'prepared_zero_cold_reference']
    if len(authority) != 1:
        raise ValueError('Exactly one complete zero material contact authority is required')
    authority = authority[0]
    if (_digest({k: v for k, v in authority.items() if k != 'binding_sha256'}) != authority['binding_sha256']
            or authority['cold_coordinate_increments'] != [[0, 1]]*56
            or authority['cold_carrier_binding_sha256'] != cold_carrier.binding_sha256
            or len(authority['point_receipts']) != 32 or len(authority['original_atom_authorities']) != 16):
        raise ValueError('The authentic source16/contact32 affine material authority changed')
    for point in authority['point_receipts']:
        cur = point['current']
        if (cur != point['prepared_zero'] or cur['bound_coordinate_increments'] != [[0, 1]]*56
                or cur['system_binding_sha256'] != cold_carrier.binding_sha256
                or _digest({k: v for k, v in cur.items() if k not in ('format', 'binding_sha256')}) != cur['binding_sha256']
                or len(cur['mode_displacement_columns']) != 56 or len(cur['mode_original_gradient_columns_per_m']) != 56
                or cur['material_membership_authenticated'] is not True):
            raise ValueError('A complete original material affine field changed its source seal or columns')
    clocks = tuple(F(v) for v in (0, 2000000, 1000000, 1000000))
    if cold_energy_system.source_clocks_s != clocks or any(p.source_clocks_s != clocks for p in preparations):
        raise ValueError('The complete endpoint cannot change original source clocks')
    units = tuple(basal_trace_system.coordinate_units)
    system = NativeCommonGeneralForceSystem(expected, units,
        tuple('rad/s' if i < 6 else '1/s' if i == 100 else 'm/s' for i in range(101)),
        tuple('N*m' if i < 6 else 'J' if i == 100 else 'N' for i in range(101)),
        cold_energy_system.binding_sha256, cold_carrier.binding_sha256, hot_system.binding_sha256,
        basal_trace_system.binding_sha256, tuple(p.binding_sha256 for p in preparations),
        authority['binding_sha256'], tuple(v[1] for v in SOURCE_PINS), tuple(v[2] for v in SOURCE_PINS),
        clocks, provenance, precision, '')
    system = _remember(system, _SYSTEMS)
    _CONTEXTS[system.binding_sha256] = dict(packets=tuple(packets), authority_json=_json(authority),
        cold=cold_carrier, hot=hot_system, preparations=preparations, trace=basal_trace_system,
        cold_energy=cold_energy_system, source=trace_context['source'],
        internal_beams=tuple(json.loads(_json(source907['mixed_force_system']['internal_beams']))),
        system_binding_sha256=system.binding_sha256)
    return system


def validate_native_common_general_force(system):
    _guard(system, NativeCommonGeneralForceSystem, _SYSTEMS)
    ctx = _CONTEXTS.get(system.binding_sha256)
    if ctx is None:
        raise ValueError('The general force lacks its original complete source context')
    _cold_api().validate_native_common_coupled_cold_energy(ctx['cold_energy'])
    validate_native_common_hot_mean(ctx['hot'])
    trace_api.validate_native_common_basal_trace(ctx['trace'])
    return system


def native_common_general_force_to_dict(system):
    return dict(format=FORMAT, **_payload(validate_native_common_general_force(system)))


def native_common_general_force_from_dict(raw, source907, source895, source903, **ports):
    if type(raw) is not dict or raw.get('format') != FORMAT:
        raise ValueError('A complete general-force system DTO is required')
    value = bind_native_common_general_force(source907, source895, source903,
        provenance=raw.get('provenance'), precision=raw.get('precision'), **ports)
    if _json(native_common_general_force_to_dict(value)) != _json(raw):
        raise ValueError('A resealed general system changed its complete source or chart')
    return value


def _coordinates(system, values, rates):
    if (type(values) is not tuple or len(values) != 101 or any(type(v) is not F for v in values)
            or type(rates) is not tuple or len(rates) != 101 or any(type(v) is not F for v in rates)):
        raise ValueError('The general endpoint requires all101 literal Fraction coordinates and physical rates')
    ctx = _CONTEXTS[system.binding_sha256]
    if not 0 <= values[100] <= 1:
        raise ValueError('Physical preparation must remain in [0,1]')
    if any(abs(v) > 100 for v in values[6:36]):
        raise ValueError('The authentic HOT finite mean chart retains its100 metre bound')
    if any(abs(v) > b for v, b in zip(values[36:92], ctx['cold'].coordinate_envelope_abs_m)):
        raise ValueError('The actual cold material field exceeds its whole895 envelope')
    authority = json.loads(ctx['authority_json'])
    for point in authority['point_receipts']:
        field = point['original_saved_attachment']['field']
        bank = next(b for b in ctx['cold'].banks if b.original_parent_id == field['original_parent_id'])
        if bank.radius_m+values[field['residual_coordinate_index']] <= 0:
            raise ValueError('An original internal arm must retain a positive source radius')
    return values, rates


def _side(point, cold, preparations, values, rotations, precision):
    cur, item = point['current'], point['original_saved_attachment']
    field = item['field']
    x0, A0 = _vector(cur['position_bounds_m']), _matrix(cur['original_cartesian_gradient_bounds'])
    modes = tuple(_vector(v) for v in cur['mode_displacement_columns'])
    gradients = tuple(_matrix(v) for v in cur['mode_original_gradient_columns_per_m'])
    x = tuple(_Jet(x0[i]+sum((values[36+k]*m[i] for k, m in enumerate(modes)), I(0)),
        {36+k: m[i] for k, m in enumerate(modes)}, precision=precision) for i in range(3))
    A = tuple(tuple(_Jet(A0[i][j]+sum((values[36+k]*G[i][j] for k, G in enumerate(gradients)), I(0)),
        {36+k: G[i][j] for k, G in enumerate(gradients)}, precision=precision)
        for j in range(3)) for i in range(3))
    lam = _Jet.variable(values[100], 100, precision)
    tx, preparedA, DT, recipe = _prepared_fields(x, A, preparations[cur['owner']], lam, precision)
    Aold, bridge = _mf_original_source_gradient(item, cold, precision=precision)
    invold = _inverse(Aold, precision)
    l0 = tuple(a-b for a, b in zip(_vector(field['source_reference_position_bounds_m']),
                                  _vector(field['material_reference_position_bounds_m'])))
    virtual = _vector(field['source_virtual_radial_direction_bounds'])
    internal = _Jet.variable(values[field['residual_coordinate_index']], field['residual_coordinate_index'], precision)
    arm = tuple(_Jet(a, precision=precision)+internal*v for a, v in zip(l0, virtual))
    transport = _mm(preparedA, _constant_matrix(invold, precision))
    y = tuple(a+b for a, b in zip(tx, _mv(transport, arm)))
    world = _mv(rotations[cur['owner']], y)
    receipt = dict(point_id=point['point_id'], side=point['side'], owner=cur['owner'],
        allocation=cur['allocation'], archive_debits=cur['archive_debits'],
        original_material_id=cur['original_material_id'], original_parent_id=cur['original_parent_id'],
        source_point_binding_sha256=cur['binding_sha256'], complete_source_affine_field_json=_json(cur),
        complete_original_source_input_frame_bridge=bridge, old_original_cartesian_gradient_bounds=_payload(Aold),
        old_original_cartesian_inverse_bounds=_payload(invold), original_internal_arm_bounds_m=_payload(l0),
        original_virtual_radial_direction_bounds=_payload(virtual),
        internal_radial_coordinate_index=field['residual_coordinate_index'],
        actual_cold_position_twojet=[_export(v) for v in x],
        actual_cold_original_gradient_twojet=[[_export(v) for v in row] for row in A],
        prepared_point_position_twojet=[_export(v) for v in tx],
        material_transport_twojet=[[_export(v) for v in row] for row in transport],
        material_attachment_position_twojet=[_export(v) for v in y],
        world_attachment_position_twojet=[_export(v) for v in world],
        full_actual_cold56_columns_retained=True, full_body_internal_preparation_cross_curvature_included=True,
        physical_source_historical_prestress_recovered=False, **recipe)
    return world, preparedA, receipt


def _contacts(ctx, values, rotations, precision):
    authority = json.loads(ctx['authority_json'])
    pointmap = {(p['point_id'], p['side']): p for p in authority['point_receipts']}
    sides = {key: _side(p, ctx['cold'], ctx['preparations'], values, rotations, precision)
             for key, p in pointmap.items()}
    total = _Jet(0, precision=precision)
    recoverable, fracture = I(0), I(0)
    atoms = []
    exact_zero = not any(values[:6]+values[36:])
    for original in authority['original_atom_authorities']:
        atom, aid = original['original_atom'], original['point_id']
        sy, _, sr = sides[aid, 'sub']
        oy, overA, ore = sides[aid, 'over']
        overpoint = pointmap[aid, 'over']
        normal, nr = _normal(overA, _vector(original['original_receiver_material_normal_bounds']), precision)
        owner = overpoint['current']['owner']
        worldnormal = _mv(rotations[owner], normal)
        # The authentic zero calibration is a fixed source expression, not
        # a new body/preparation value or an inferred historical prestress.
        baseline = ctx.get('contact_baselines', {}).get((aid, precision))
        if baseline is None:
            # Both original-source affine attachment positions and the
            # Nanson normal at zero are evaluated from the SAME bound fields.
            zero = (F(0),)*101
            identity = (_identity(precision), _identity(precision))
            zs, _, _ = _side(pointmap[aid, 'sub'], ctx['cold'], ctx['preparations'], zero, identity, precision)
            zo, zA, _ = _side(pointmap[aid, 'over'], ctx['cold'], ctx['preparations'], zero, identity, precision)
            zn, _ = _normal(zA, _vector(original['original_receiver_material_normal_bounds']), precision)
            baseline = _dot(zn, tuple(b-a for a, b in zip(zs, zo))).value
            ctx.setdefault('contact_baselines', {})[aid, precision] = baseline
        scalar = _dot(worldnormal, tuple(b-a for a, b in zip(sy, oy)))-baseline
        if exact_zero:
            scalar = _Jet(0, scalar.gradient, scalar.hessian, precision)
        _, kn, C, T, gamma, K, anchor = _source_law(atom)
        lawargs = dict(contact_stiffness_pa_m=kn, column_compliance_m_pa=C,
            tensile_strength_pa=T, fracture_energy_j_m2=gamma, precision=precision)
        loading = evaluate_native_cohesive_series_loading(I(anchor)+scalar.value, **lawargs)
        initial = evaluate_native_cohesive_series_loading(I(anchor), **lawargs)
        area = _fraction(atom['original_area_m2'])
        literal = sum((_fraction(v) for v in atom['stored_source_energy_components_j']), F(0))
        offset = I(literal)-area*initial.loading_potential_density_bounds_j_m2
        energy = offset+area*loading.loading_potential_density_bounds_j_m2
        stored = offset+area*loading.recoverable_energy_density_bounds_j_m2
        dissipated = area*loading.fracture_work_density_bounds_j_m2
        if exact_zero:
            energy = stored = I(literal)
            dissipated = I(0)
        ej = scalar.compose(energy, area*loading.traction_bounds_pa, area*loading.tangent_bounds_pa_m)
        total += ej
        recoverable += stored
        fracture += dissipated
        atoms.append(_json(dict(point_id=aid, complete_original_atom_json=_json(atom),
            original_area_m2=_payload(area), literal_source_energy_j=_payload(literal),
            fixed_prepared_zero_scalar_reference_bounds_m=_payload(baseline),
            scalar_increment_twojet=_export(scalar), energy_twojet=_export(ej),
            loading=_payload(loading), energy_bounds_j=_payload(energy),
            recoverable_energy_bounds_j=_payload(stored), fracture_work_bounds_j=_payload(dissipated),
            receiver_nanson_normal=nr, world_receiver_normal_twojet=[_export(v) for v in worldnormal],
            source_original_area_included_once=True, current_area_energy_multiplier_added=False,
            finite_owner_chart_first_and_second_derivatives_retained=True,
            proposed_normal_damage_or_fracture_work_applied=False,
            source_zero_traction_calibration_is_physical_prestress_handover=False)))
    return total, recoverable, fracture, tuple(atoms), tuple(_json(v[2]) for v in sides.values())


def _finite_basal(ctx, values, rates, rotations, precision, root_iterations):
    zero_owner_values = (F(0),)*6+values[6:]
    trace = trace_api.evaluate_native_common_basal_trace(ctx['trace'], zero_owner_values,
        precision=precision, root_iterations=root_iterations)
    port = trace_api.native_common_basal_trace_loading_context(ctx['trace'], trace)
    records = []
    for old in port['trace_records']:
        ids = old['global_coordinate_indices']
        x = tuple(_Jet(old['position_bounds_m'][axis],
            {index: old['position_first_bounds'][axis][j] for j, index in enumerate(ids) if index >= 6},
            {_pair(i, j): old['position_second_bounds'][axis][a][b]
                for a, i in enumerate(ids) if i >= 6
                for b, j in enumerate(ids) if j >= i and j >= 6}, precision) for axis in range(3))
        y = _mv(rotations[old['owner']], x)
        newids = tuple(sorted(set().union(*(v.gradient.keys() for v in y),
            *(set(p) for v in y for p in v.hessian)) | set(ids)))
        record = dict(old)
        record.update(position_bounds_m=tuple(v.value for v in y), global_coordinate_indices=newids,
            position_first_bounds=tuple(tuple(v.gradient.get(i, I(0)) for i in newids) for v in y),
            position_second_bounds=tuple(tuple(tuple(v.hessian.get(_pair(i, j), I(0))
                for j in newids) for i in newids) for v in y),
            owner_rotation_query_is_zero_slice=False, finite_owner_chart_first_and_second_derivatives_retained=True,
            source_zero_owner_trace_response_binding_sha256=trace.binding_sha256,
            source_zero_owner_material_point_binding_sha256=old['actual_point_binding_sha256'],
            actual_point_binding_sha256=_digest(dict(general_system=ctx['system_binding_sha256'],
                source_zero_owner_material_point=old['actual_point_binding_sha256'],
                coordinates101=values, position_twojet=tuple(_export(v) for v in y), precision=precision)),
            new_registered_general_state_port=True)
        records.append(record)
    # Source/catalogue validation occurs before this generic algebra call.
    if (len(records) != 68 or tuple(port['coordinate_order']) != tuple(ctx['trace'].coordinate_order)
            or port['source'] != ctx['source']):
        raise ValueError('The finite material trace lost an authentic source or catalogue association')
    result = _trace_algebra(ctx['source'], tuple(records), rates, size=101, precision=precision)
    return result, tuple(_json(_payload(r)) for r in records)


@dataclass(frozen=True, slots=True)
class NativeCommonGeneralForceResponse:
    system_binding_sha256: str
    coordinate_order: tuple
    coordinate_values: tuple
    coordinate_rates: tuple
    energy_bounds_j: I
    conservative_gradient_bounds: tuple
    conservative_hessian_bounds: tuple
    residual_bounds: tuple
    state_tangent_bounds: tuple
    velocity_tangent_bounds: tuple
    energy_partitions_json: str
    cold_response_json: str
    hot_response_json: str
    contact_atoms_json: tuple
    contact_side_receipts_json: tuple
    basal_trace_records_json: tuple
    basal_response_json: str
    owner_rotation_receipts_json: tuple
    internal_beam_receipts_json: tuple
    precision: int
    root_iterations: int
    theta_cell_count: int
    beta_cell_count: int
    depth_cell_count: int
    max_cells: int
    source_clocks_s: tuple
    binding_sha256: str
    scope: str = SCOPE
    complete_declared_general_state_operator_computed: bool = True
    general_finite_owner_state_chart_defined: bool = True
    full_actual_cold56_and_hot30_fields_retained: bool = True
    actual_original_contact_atom_count: int = 16
    actual_original_contact_side_count: int = 32
    actual_fixed_reference_basal_trace_count: int = 68
    frozen_owner_load_potential_used: bool = False
    conservative_spatial_loading_potential_used: bool = False
    state_tangent_assumed_symmetric: bool = False
    physical_coordinate_rates_including_lambda_explicit: bool = True
    general_state_full_physics_operator_certified: bool = False
    physical_source_prestress_work_handover_certified: bool = False
    reference_preparation_work_paid: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    physical_material_admission_authorized: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    continuum_whole_base_quadrature_certified: bool = False
    full_system_temporal_path_work_certified: bool = False
    proposed_normal_damage_or_fracture_work_applied: bool = False
    accepted_old_native_intervals: int = 1
    accepted_new_native_intervals: int = 0
    clocks_advanced: bool = False
    remaining_condition_count: int = 3
    remaining_conditions: tuple = REMAINING_CONDITIONS


def evaluate_native_common_general_force(system, coordinates101=None, *, coordinate_rates,
        precision=96, root_iterations=12, theta_cell_count=4, beta_cell_count=1,
        depth_cell_count=1, max_cells=4096):
    validate_native_common_general_force(system)
    _precision(precision)
    values = (F(0),)*101 if coordinates101 is None else coordinates101
    values, rates = _coordinates(system, values, coordinate_rates)
    if type(root_iterations) is not int or not 0 <= root_iterations <= 64:
        raise ValueError('The finite mean root budget must be a literal integer')
    key = (system.binding_sha256, values, rates, precision, root_iterations,
           theta_cell_count, beta_cell_count, depth_cell_count, max_cells)
    if key in _EVALUATIONS:
        return _guard(_EVALUATIONS[key], NativeCommonGeneralForceResponse, _RESPONSES)
    ctx = _CONTEXTS[system.binding_sha256]
    state_key = (system.binding_sha256, values, precision, root_iterations,
                 theta_cell_count, beta_cell_count, depth_cell_count, max_cells)
    if state_key in _STATE_EVALUATIONS:
        prior, rotations = _STATE_EVALUATIONS[state_key]
        _guard(prior, NativeCommonGeneralForceResponse, _RESPONSES)
        basal, records = _finite_basal(ctx, values, rates, rotations, precision, root_iterations)
        R = tuple(_round(prior.conservative_gradient_bounds[i]-basal['residual_bounds'][i], precision)
                  for i in range(101))
        Kq = tuple(tuple(_round(prior.conservative_hessian_bounds[i][j]-basal['state_tangent_bounds'][i][j], precision)
                         for j in range(101)) for i in range(101))
        result = replace(prior, coordinate_rates=rates, residual_bounds=R,
            state_tangent_bounds=Kq, velocity_tangent_bounds=basal['drag_bounds'],
            basal_trace_records_json=records, basal_response_json=_json(_payload(basal)), binding_sha256='')
        result = _remember(result, _RESPONSES)
        _EVALUATIONS[key] = result
        return result
    owner = tuple(_owner_rotation(values[3*i:3*i+3], i, precision) for i in range(2))
    rotations = tuple(v[0] for v in owner)
    cold_api = _cold_api()
    cold = cold_api.evaluate_native_common_coupled_cold_energy(ctx['cold_energy'], values[36:92], values[100],
        precision=precision, theta_cell_count=theta_cell_count, beta_cell_count=beta_cell_count,
        depth_cell_count=depth_cell_count, max_cells=max_cells)
    cold_ids = tuple(range(36, 92))+(100,)
    energy = _Jet(cold.energy_j, dict(zip(cold_ids, cold.gradient57)),
        {(i, j): cold.hessian57[a][b] for a, i in enumerate(cold_ids)
            for b, j in enumerate(cold_ids) if j >= i}, precision)
    hot = evaluate_native_common_hot_mean(ctx['hot'], values[6:36], precision=precision,
                                         root_iterations=root_iterations)
    hotjet = _Jet(hot.energy_bounds_j, {6+i: v for i, v in enumerate(hot.gradient_bounds_n)},
        {(6+i, 6+j): hot.hessian_bounds_n_m[i][j] for i in range(30) for j in range(i, 30)}, precision)
    energy += hotjet
    correction = _Jet(0, precision=precision)
    beam_receipts = []
    # Bank ordering comes from the same authenticcold continuum source port.
    cold_context = cold_api.native_common_coupled_cold_energy_context(ctx['cold_energy'])
    banks = cold_context['source899']['cold_continuum_energy_system']['banks']
    for k, bank in enumerate(banks):
        # The authentic mixed source retains the original physical-parent
        # beam/internal catalogue. Its globalindices must not follow bank k.
        beam = next(b for b in ctx['internal_beams'] if b['original_parent_id'] == bank['original_parent_id'])
        ci = beam['global_coordinate_index']
        p = _Jet(cold.tip_projection_m[k], dict(zip(cold_ids, cold.tip_projection_first[k])),
            {(i, j): cold.tip_projection_second[k][a][b]
                for a, i in enumerate(cold_ids) for b, j in enumerate(cold_ids) if j >= i}, precision)
        r = _Jet.variable(values[ci], ci, precision)
        f, K = cold.literal_reference_forces_n[k], cold.tip_stiffnesses_n_m[k]
        shift = f*r+K*p*r+K*r.square()/2
        correction += shift
        beam_receipts.append(_json(dict(original_parent_id=bank['original_parent_id'],
            global_internal_coordinate_index=ci, tip_projection_twojet=_export(p),
            literal_reference_force_n=_payload(f), tip_stiffness_n_m=_payload(K),
            exact_internal_shift_correction_twojet=_export(shift), original_beam_energy_added_again=False)))
    energy += correction
    contact, recoverable, fracture, atoms, sides = _contacts(ctx, values, rotations, precision)
    energy += contact
    basal, records = _finite_basal(ctx, values, rates, rotations, precision, root_iterations)
    gradient = tuple(energy.gradient.get(i, I(0)) for i in range(101))
    H = tuple(tuple(energy.hessian.get(_pair(i, j), I(0)) for j in range(101)) for i in range(101))
    R = tuple(_round(gradient[i]-basal['residual_bounds'][i], precision) for i in range(101))
    Kq = tuple(tuple(_round(H[i][j]-basal['state_tangent_bounds'][i][j], precision)
                     for j in range(101)) for i in range(101))
    partitions = dict(cold_whole_material_and_eight_beams_bounds_j=_payload(cold.energy_j),
        hot_whole_material_and_finite_radial_base_bounds_j=_payload(hot.energy_bounds_j),
        eight_internal_beam_driver_correction_bounds_j=_payload(correction.value),
        original_source_atom_loading_potential_bounds_j=_payload(contact.value),
        original_source_atom_recoverable_energy_bounds_j=_payload(recoverable),
        proposed_normal_fracture_work_bounds_j=_payload(fracture),
        frozen_owner_load_drag_potential_added=False, spatial_basal_force_has_conservative_potential=False,
        total_bounds_j=_payload(energy.value))
    result = NativeCommonGeneralForceResponse(system.binding_sha256, system.coordinate_order, values, rates,
        energy.value, gradient, H, R, Kq, basal['drag_bounds'], _json(partitions),
        _json(cold_api.native_common_coupled_cold_energy_response_to_dict(cold)),
        _json(native_common_hot_mean_response_to_dict(hot)), atoms, sides, records,
        _json(_payload(basal)), tuple(_json(v[1]) for v in owner), tuple(beam_receipts),
        precision, root_iterations, theta_cell_count, beta_cell_count, depth_cell_count, max_cells,
        system.source_clocks_s, '')
    result = _remember(result, _RESPONSES)
    _EVALUATIONS[key] = result
    _STATE_EVALUATIONS[state_key] = (result, rotations)
    return result


def native_common_general_force_response_to_dict(response):
    return dict(format=RESPONSE_FORMAT, **_payload(_guard(response, NativeCommonGeneralForceResponse, _RESPONSES)))


def native_common_general_force_response_from_dict(raw, *, system, expected_coordinates101,
        expected_coordinate_rates, precision=96, root_iterations=12, theta_cell_count=4,
        beta_cell_count=1, depth_cell_count=1, max_cells=4096):
    if type(raw) is not dict or raw.get('format') != RESPONSE_FORMAT:
        raise ValueError('A complete general endpoint response DTO is required')
    value = evaluate_native_common_general_force(system, expected_coordinates101,
        coordinate_rates=expected_coordinate_rates, precision=precision, root_iterations=root_iterations,
        theta_cell_count=theta_cell_count, beta_cell_count=beta_cell_count,
        depth_cell_count=depth_cell_count, max_cells=max_cells)
    if _json(native_common_general_force_response_to_dict(value)) != _json(raw):
        raise ValueError('A resealed general response changed its literal query or full operator')
    return value

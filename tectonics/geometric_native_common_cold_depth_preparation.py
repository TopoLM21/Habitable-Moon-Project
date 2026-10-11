"""A physical, smooth depth-twist preparation of the complete cold map.

T_a(x) = exp(a [Omega] tau(x)) x,
tau(x) = (R**3 - (|x|/s)**3)/(3 R**2).

This composes with the current895 material map; it changes positions and
original gradients. It is not a change of material reference and does not
declare the resulting state stress free. The path is a radius-preserving
global bijection with determinant one, separately for each authentic owner.
Neither preparation work, force equilibrium nor new admission is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F

from .geometric_native_pose import (
    FractionInterval as I, interval_sqrt, _rounded_interval,
)
from .geometric_native_joint_spherical_field import rotate_joint_physical_depth
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_common_cold_material_carrier import _registered, _saved_json_kinds
from .geometric_native_common_carrier_source_work import (
    _encoded, _payload, _digest, _typed_guard,
)


FORMAT = "native-common-cold-depth-physical-preparation-1"
SCOPE = "DECLARED_PHYSICAL_COLD_DEPTH_TWIST_PATH_WITH_UNPAID_PREPARATION_WORK"
_ZERO = (I(0),)*3
_IDENTITY = tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))
_REGISTRY = {}


def _vector(value):
    if type(value) is not tuple or len(value) != 3 or any(type(x) is not I for x in value):
        raise ValueError("Physical preparation requires immutable three-interval vectors")
    return value


def _matrix(value):
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("Physical preparation requires an immutable three-by-three interval matrix")
    return tuple(_vector(row) for row in value)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _map(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _multiply(a, b):
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(3)), I(0))
                       for j in range(3)) for i in range(3))


def _square(x):
    return I(0 if x.lo <= 0 <= x.hi else min(x.lo*x.lo, x.hi*x.hi),
             max(x.lo*x.lo, x.hi*x.hi))


def _body(value):
    return {f.name: getattr(value, f.name) for f in fields(value) if f.name != "binding_sha256"}


def _fraction_parameter(value):
    if type(value) is F:
        value = I(value)
    if type(value) is not I or not 0 <= value.lo <= value.hi <= 1:
        raise ValueError("Preparation fraction must be a literal Fraction or interval in [0,1]")
    return value


@dataclass(frozen=True, slots=True)
class DeclaredColdDepthPreparation:
    """Pure exact map declaration; it carries no material-mask authority."""
    radius_m: F
    fixed_radial_scale: F
    angular_rate_per_physical_volume_depth_m: tuple
    provenance: str

    def __post_init__(self):
        if (type(self.radius_m) is not F or self.radius_m <= 0
                or type(self.fixed_radial_scale) is not F or self.fixed_radial_scale <= 0):
            raise ValueError("Physical preparation radius and scale require positive literal Fractions")
        omega = self.angular_rate_per_physical_volume_depth_m
        if type(omega) is not tuple or len(omega) != 3 or any(type(x) is not F for x in omega):
            raise ValueError("Physical preparation rate requires three literal Fractions")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("Physical preparation requires its declared provenance")


@dataclass(frozen=True, slots=True)
class ColdDepthPreparationField:
    declaration: DeclaredColdDepthPreparation
    preparation_fraction_bounds: I
    inverse: bool
    original_position_bounds_m: tuple
    radius_bounds_m: I
    physical_volume_depth_bounds_m: I
    physical_volume_depth_gradient_bounds: tuple
    physical_volume_depth_hessian_bounds_per_m: tuple
    rotation_bounds: tuple
    position_bounds_m: tuple
    spatial_gradient_bounds: tuple
    inverse_spatial_gradient_bounds: tuple
    spatial_hessian_bounds_per_m: tuple
    relative_green_strain_linear_coefficients: tuple
    relative_green_strain_quadratic_coefficients: tuple
    relative_green_strain_bounds: tuple
    preparation_velocity_bounds_m: tuple
    preparation_acceleration_bounds_m: tuple
    preparation_spatial_gradient_rate_bounds: tuple
    original_cartesian_gradient_bounds: tuple | None
    preparation_original_gradient_rate_bounds: tuple | None
    original_green_strain_linear_coefficients: tuple | None
    original_green_strain_quadratic_coefficients: tuple | None
    mode_displacement_columns: tuple
    mode_original_gradient_columns_per_m: tuple
    precision: int
    order: int
    map_volume_jacobian_bounds: I = I(1)
    radius_preservation_identity: bool = True
    global_inverse_identity: bool = True
    preparation_work_paid: bool = False
    source_reference_or_rest_metric_reset: bool = False
    physical_material_admission_authorized: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    physical_elapsed_time_added: bool = False
    accepted_new_native_intervals: int = 0


def evaluate_declared_cold_depth_preparation(declaration, position_bounds_m, *,
        preparation_fraction, original_gradient_bounds=None,
        mode_displacement_columns=(), mode_original_gradient_columns_per_m=(),
        inverse=False, precision=96, order=28):
    """Outward geometry, full spatial two-jet and physical path derivatives.

    Bounds enclose the input correlated expression, not every material point
    of a rectangle. Mode gradient transformation includes D²T[v, A]; merely
    rotating the old gradients would lose this contribution. The fraction is
    a preparation parameter, not an advancement of a physical source clock.
    The Cartesian two-jet exists at zero too, but origin-crossing rectangles
    are refused to avoid dividing their enclosure by an unresolved norm.
    Relative Green strain is exactly quadratic in the preparation fraction:
    E_rel = fraction sym(d) + fraction² d.T d/2, d=(Omega×x) outer grad(tau).
    The original-gradient coefficients A.T E_rel A are INCREMENTS to the
    old Green strain. They do not reset old strain, prestress or an energy.
    """
    if type(declaration) is not DeclaredColdDepthPreparation:
        raise ValueError("A literal physical preparation declaration is required")
    declaration.__post_init__()
    x = _vector(position_bounds_m)
    fraction = _fraction_parameter(preparation_fraction)
    if type(inverse) is not bool:
        raise ValueError("Inverse selection requires a literal boolean")
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Physical preparation precision must be an integer in [96,512]")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Physical preparation order must be an integer in [8,64]")
    A = None if original_gradient_bounds is None else _matrix(original_gradient_bounds)
    if type(mode_displacement_columns) is not tuple or type(mode_original_gradient_columns_per_m) is not tuple:
        raise ValueError("Physical preparation mode columns require immutable tuples")
    modes = tuple(_vector(v) for v in mode_displacement_columns)
    gradients = tuple(_matrix(g) for g in mode_original_gradient_columns_per_m)
    if gradients and (A is None or len(gradients) != len(modes)):
        raise ValueError("Mode original gradients require the original matrix and matching displacement columns")
    rnd = lambda v: _rounded_interval(v, precision)
    radius = interval_sqrt(sum((_square(v) for v in x), I(0)), precision+16)
    if radius.lo <= 0:
        raise ValueError("Split physical preparation inputs away from an unresolved origin")
    R, s = declaration.radius_m, declaration.fixed_radial_scale
    omega = tuple(I(v) for v in declaration.angular_rate_per_physical_volume_depth_m)
    denominator = s**3*R**2
    tau = rnd((I(R**3)-radius*radius*radius/s**3)/(3*R**2))
    dtau = tuple(rnd(-radius*v/denominator) for v in x)
    ddtau = tuple(tuple(rnd(-(radius*int(i == j)+x[i]*x[j]/radius)/denominator)
                           for j in range(3)) for i in range(3))
    sign = -1 if inverse else 1
    alpha = sign*fraction
    columns = tuple(rotate_joint_physical_depth(tuple(I(int(i == j)) for i in range(3)),
        declaration.angular_rate_per_physical_volume_depth_m, rnd(alpha*tau),
        precision=precision, order=order) for j in range(3))
    Q = _transpose(columns)
    wx, wwx = _cross(omega, x), _cross(omega, _cross(omega, x))
    W = tuple(tuple(_cross(omega, tuple(I(int(i == j)) for i in range(3)))[k]
                    for j in range(3)) for k in range(3))
    V = tuple(tuple(rnd(I(int(i == j))+alpha*wx[i]*dtau[j])
                    for j in range(3)) for i in range(3))
    DT = tuple(tuple(rnd(v) for v in row) for row in _multiply(Q, V))
    Vminus = tuple(tuple(rnd(I(int(i == j))-alpha*wx[i]*dtau[j])
                         for j in range(3)) for i in range(3))
    inverseDT = tuple(tuple(rnd(v) for v in row) for row in _multiply(Vminus, _transpose(Q)))
    d = tuple(tuple(rnd(wx[i]*dtau[j]) for j in range(3)) for i in range(3))
    green_linear = tuple(tuple(rnd(sign*(d[i][j]+d[j][i])/2)
                               for j in range(3)) for i in range(3))
    dtd = _multiply(_transpose(d), d)
    green_quadratic = tuple(tuple(rnd(v/2) for v in row) for row in dtd)
    green = tuple(tuple(rnd(fraction*green_linear[i][j]
                            +fraction*fraction*green_quadratic[i][j])
                        for j in range(3)) for i in range(3))
    # H[output_axis][input_axis][input_axis]. Both input-axis terms matter.
    H = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            before = tuple(rnd(alpha*(W[k][i]*dtau[j]+W[k][j]*dtau[i]+wx[k]*ddtau[i][j])
                               +alpha*alpha*wwx[k]*dtau[i]*dtau[j]) for k in range(3))
            H[i][j] = tuple(rnd(v) for v in _map(Q, before))
    hessian = tuple(tuple(tuple(H[i][j][k] for j in range(3))
                          for i in range(3)) for k in range(3))
    velocity = tuple(rnd(sign*tau*v) for v in _map(Q, wx))
    acceleration = tuple(rnd(tau*tau*v) for v in _map(Q, wwx))
    rate_before = tuple(tuple(rnd(sign*(tau*W[i][j]+wx[i]*dtau[j]
                                +alpha*tau*wwx[i]*dtau[j])) for j in range(3)) for i in range(3))
    DTrate = tuple(tuple(rnd(v) for v in row) for row in _multiply(Q, rate_before))
    pushed_modes = tuple(tuple(rnd(v) for v in _map(DT, mode)) for mode in modes)
    pushed_gradients = []
    for mode, gradient in zip(modes, gradients):
        linear = _multiply(DT, gradient)
        pushed_gradients.append(tuple(tuple(rnd(linear[k][j]+sum(
            (hessian[k][i][l]*mode[i]*A[l][j] for i in range(3) for l in range(3)), I(0)))
            for j in range(3)) for k in range(3)))
    original_green = lambda coefficient: None if A is None else tuple(tuple(rnd(v) for v in row)
        for row in _multiply(_multiply(_transpose(A), coefficient), A))
    return ColdDepthPreparationField(declaration, fraction, inverse, x, radius, tau, dtau, ddtau,
        Q, tuple(rnd(v) for v in _map(Q, x)), DT, inverseDT, hessian,
        green_linear, green_quadratic, green, velocity, acceleration,
        DTrate, None if A is None else tuple(tuple(rnd(v) for v in row) for row in _multiply(DT, A)),
        None if A is None else tuple(tuple(rnd(v) for v in row) for row in _multiply(DTrate, A)),
        original_green(green_linear), original_green(green_quadratic),
        pushed_modes, tuple(pushed_gradients), precision, order)


@dataclass(frozen=True, slots=True)
class NativeCommonColdDepthPreparation:
    declaration: DeclaredColdDepthPreparation
    reference_snapshot_sha256: str
    system_binding_sha256: str
    archive_carrier_model_sha256: str
    owner: int
    original_parent_ids: tuple
    source_clocks_s: tuple
    binding_sha256: str
    scope: str = SCOPE
    preparation_work_paid: bool = False
    physical_material_admission_authorized: bool = False
    source_state_or_work_advanced: bool = False
    accepted_new_native_intervals: int = 0


def bind_native_common_cold_depth_preparation(reference, system, owner, *, provenance):
    """Bind the literal common rate and one complete registered cold owner."""
    reference = validate_native_common_reference(reference)
    carrier = _registered(system)
    if type(owner) is not int or owner not in (carrier.donor_owner, carrier.receiver_owner):
        raise ValueError("Physical depth preparation requires one authentic cold owner")
    certificate = (system.whole_domain_certificate.donor if owner == carrier.donor_owner
                   else system.whole_domain_certificate.receiver)
    banks = tuple(b for b in system.banks if b.owner == owner)
    if (not certificate.global_injectivity_certified or len(banks) != 4
            or tuple(b.original_parent_id for b in banks) != certificate.parent_ids
            or any(b.radius_m != reference.radius_m or
                b.angular_rate_per_physical_volume_depth_m != reference.angular_rate_per_physical_volume_depth_m
                for b in banks)):
        raise ValueError("Physical depth preparation differs from its complete registered common owner")
    for bank in banks:
        matches = tuple(b for b in reference.banks if b.component_id == bank.component_id)
        if len(matches) != 1 or matches[0].original_parent_id != bank.original_parent_id or matches[0].owner != owner:
            raise ValueError("Physical depth preparation changed an authentic source material identity")
    declaration = DeclaredColdDepthPreparation(reference.radius_m, certificate.fixed_radial_scale,
        reference.angular_rate_per_physical_volume_depth_m, provenance)
    value = NativeCommonColdDepthPreparation(declaration, reference.reference_snapshot_sha256,
        system.binding_sha256, carrier.model_sha256, owner, certificate.parent_ids,
        (system.source_geometry_elapsed_time_s, system.contact_geometry_elapsed_time_s,
         system.memory_elapsed_time_s, system.material_geometry_elapsed_time_s), "")
    value = replace(value, binding_sha256=_digest(_body(value)))
    _REGISTRY[value.binding_sha256] = _typed_guard(value)
    return value


def _validate_bound(value):
    if (type(value) is not NativeCommonColdDepthPreparation
            or _digest(_body(value)) != value.binding_sha256
            or _REGISTRY.get(value.binding_sha256) != _typed_guard(value)):
        raise ValueError("An altered or resealed physical depth preparation lacks its complete source binding")
    return value


def evaluate_native_common_cold_depth_preparation(preparation, position_bounds_m, **kwargs):
    """Evaluate a declared path; source-point membership must come from895.

    Binding proves the owner parameters, not membership of a supplied box.
    Complete cold fields can be composed with this same owner-wide map.
    """
    preparation = _validate_bound(preparation)
    return evaluate_declared_cold_depth_preparation(preparation.declaration, position_bounds_m, **kwargs)


def native_common_cold_depth_preparation_to_dict(value):
    return dict(format=FORMAT, **_payload(_validate_bound(value)))


def native_common_cold_depth_preparation_from_dict(raw, *, reference, system, owner, provenance):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("A complete physical depth preparation DTO is required")
    rebuilt = bind_native_common_cold_depth_preparation(reference, system, owner, provenance=provenance)
    if _encoded(raw) != _encoded(native_common_cold_depth_preparation_to_dict(rebuilt)):
        raise ValueError("Physical depth preparation restart changed its complete expected source or declaration")
    return rebuilt


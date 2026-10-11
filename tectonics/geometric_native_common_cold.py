"""Cold tensor mechanics on the explicitly NEW shared radial reference.

The immutable source body and its historical saved128 rate are retained as
separate inputs.  The active common-reference bank supplies the new embedding,
including the transported three-dimensional clamp.  It never replaces the
historical source DTO.  The six enrichment tensors remain fixed in the global
frame; an orthogonal computational frame only simplifies their contractions.

These are component strain/support receipts, not mixed equilibrium, temporal
admission, or a recovery of historical three-dimensional prestress.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_pi, interval_sqrt, _rounded_interval
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_curved_certified_integration import TaylorAlgebra, TaylorJet, validated_integrate_rectangle
from .geometric_native_cold_tensor_integration import (
    QuarterAngularAlgebra, _transpose, _matmul, _sym, _sym_gram,
    _tensor_inner, _quarter_tensor_inner, _response_values, _symmetric_rows,
    _endpoint_strip_parameters, _weighted_endpoint_moments,
    _reference_tip_stiffness_lower_bound, _encoded, _snapshot_sha,
    RawColdTensorIntegral, ColdTensorCellReceipt, ColdTensorCorrectionReceipt,
    ColdTensorEndpointStripReceipt,
)
from .geometric_native_cold_strain_energy import (
    NativeColdStrainSupport, NativeColdStrainEnergyResponse, _square, _payload,
    build_native_cold_strain_material, validate_native_cold_strain_material,
)


_BODIES = {}
_MATERIALS = {}
_INTEGRALS = set()
_INTEGRAL_CACHE = {}


def _coordinates(value):
    if type(value) is not tuple or len(value) != 7 or any(type(x) is not F for x in value):
        raise ValueError("Common cold increments require seven literal metre Fractions")
    return value


@dataclass(frozen=True, slots=True)
class NativeCommonColdBody:
    original_body: object
    embedding_bank: object
    body_snapshot_sha256: str

    def __getattr__(self, name):
        # Only immutable physical/source scalars are delegated.  The active
        # rate and geometry identity below are explicitly NEW properties.
        return getattr(self.original_body, name)

    @property
    def geometry_source_sha256(self):
        return self.body_snapshot_sha256

    @property
    def reference_snapshot_sha256(self):
        return self.embedding_bank.reference_snapshot_sha256

    @property
    def angular_rate_per_physical_volume_depth_m(self):
        return self.embedding_bank.angular_rate_per_physical_volume_depth_m

    @property
    def historical_angular_rate_per_physical_volume_depth_m(self):
        return self.original_body.angular_rate_per_physical_volume_depth_m

    @property
    def reference_model(self):
        return "NEW_shared_radial_volume_reference_with_transported_source_masks_and_clamps"


def _body_binding(value):
    return (value.original_body.body_snapshot_sha256,
            value.embedding_bank.bank_snapshot_sha256,
            value.embedding_bank.reference_snapshot_sha256,
            value.original_body.component_id,
            "fixed_global_symmetric_generators_in_orthogonal_computational_frame")


def bind_native_common_cold_body(component_id, *, reference=None, source=None, precision=160):
    from .geometric_native_cold_body import bind_native_cold_body
    from .geometric_native_common_reference import (
        bind_native_common_reference, bind_native_common_reference_bank,
    )
    if reference is None:
        reference = bind_native_common_reference(source=source, precision=precision)
    original = bind_native_cold_body(component_id, source=source, precision=precision)
    bank = bind_native_common_reference_bank(reference, component_id)
    if (bank.source_snapshot_sha256 != original.source_snapshot_sha256
            or bank.radius_m != original.radius_m
            or bank.areal_material_thickness_m != original.areal_material_thickness_m
            or bank.roof_radial_displacement_m != original.roof_radial_displacement_m
            or bank.historical_angular_rate_per_physical_volume_depth_m != original.angular_rate_per_physical_volume_depth_m):
        raise ValueError("The NEW reference bank does not bind the original cold source geometry and historical rate")
    value = NativeCommonColdBody(original, bank, "")
    digest = _snapshot_sha(_body_binding(value))
    value = replace(value, body_snapshot_sha256=digest)
    _BODIES[digest] = _body_binding(value)
    return validate_native_common_cold_body(value)


def validate_native_common_cold_body(body):
    from .geometric_native_cold_body import validate_native_cold_body
    from .geometric_native_common_reference import validate_native_common_reference_bank
    if type(body) is not NativeCommonColdBody:
        raise ValueError("A separately bound NEW common-reference cold body is required")
    validate_native_cold_body(body.original_body)
    validate_native_common_reference_bank(body.embedding_bank)
    binding = _body_binding(body)
    if (_BODIES.get(body.body_snapshot_sha256) != binding
            or body.body_snapshot_sha256 != _snapshot_sha(binding)
            or body.original_body.component_id != body.embedding_bank.component_id):
        raise ValueError("An altered common cold binding cannot replace its source or active embedding")
    return body


@dataclass(frozen=True, slots=True)
class NativeCommonColdMaterial:
    original_material: object
    body_snapshot_sha256: str
    reference_snapshot_sha256: str
    material_snapshot_sha256: str

    def __getattr__(self, name):
        return getattr(self.original_material, name)

    @property
    def source_material_snapshot_sha256(self):
        return self.original_material.material_snapshot_sha256

    def lame_coefficients(self, layer):
        return self.original_material.lame_coefficients(layer)


def _material_binding(value):
    return (value.original_material.material_snapshot_sha256,
            value.body_snapshot_sha256, value.reference_snapshot_sha256,
            "same_declared_layered_tensor_calibrated_on_NEW_common_reference")


def build_native_common_cold_strain_material(body):
    body = validate_native_common_cold_body(body)
    source_material = build_native_cold_strain_material(body.original_body)
    value = NativeCommonColdMaterial(source_material, body.body_snapshot_sha256,
                                    body.reference_snapshot_sha256, "")
    digest = _snapshot_sha(_material_binding(value))
    value = replace(value, material_snapshot_sha256=digest)
    _MATERIALS[digest] = _material_binding(value)
    return value


def validate_native_common_cold_strain_material(material, *, body=None):
    if type(material) is not NativeCommonColdMaterial:
        raise ValueError("The common reference needs its own material binding")
    binding = _material_binding(material)
    if (_MATERIALS.get(material.material_snapshot_sha256) != binding
            or material.material_snapshot_sha256 != _snapshot_sha(binding)):
        raise ValueError("An altered common-reference material cannot replace its source declaration")
    if body is not None:
        body = validate_native_common_cold_body(body)
        validate_native_cold_strain_material(material.original_material, body=body.original_body)
        if (material.body_snapshot_sha256 != body.body_snapshot_sha256
                or material.reference_snapshot_sha256 != body.reference_snapshot_sha256):
            raise ValueError("The common material belongs to another embedding")
    else:
        validate_native_cold_strain_material(material.original_material)
    return material


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), 0)


def _map(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _cross_matrix(vector):
    x, y, z = vector
    return ((0, -z, y), (z, 0, -x), (-y, x, 0))


@lru_cache(maxsize=32)
def _frame_constants(body_sha):
    # The full source binder already certifies O in SO(3).  B may be an
    # improper donor quarter frame: matrix conjugation, rather than a local
    # cross-product shortcut, retains its reflection sign exactly.
    body = _BODIES_OBJECTS[body_sha]
    source = body.original_body
    B = tuple(tuple(axis[i] for axis in (source.original_quarter_start,
              source.original_quarter_end, source.original_quarter_inward)) for i in range(3))
    P = _matmul(source.owner_rotation_matrix_bounds, B)
    K_full = _matmul(_transpose(P), _matmul(_cross_matrix(body.angular_rate_per_physical_volume_depth_m), P))
    K = tuple(tuple(I(0) if i == j else K_full[i][j] if i < j else -K_full[j][i]
                    for j in range(3)) for i in range(3))
    norm_squared = sum((x*x for x in body.angular_rate_per_physical_volume_depth_m), F(0))
    return B, P, K, norm_squared


_BODIES_OBJECTS = {}


def _modal_local_fields(body, sine, cosine, beta, depth, algebra):
    """Correct gradients transformed by C=QOB, with global Ei unchanged.

    M_i=C.T Ei C depends on depth.  Rodrigues sine/cosine and radius are
    evaluated in the same Taylor algebra as every later strain contraction;
    their full derivative tails are therefore included in rectangle proofs.
    Q cancels from the radial-tip gradient.  Nothing differentiates a local
    displacement while discarding the moving-frame connection.
    """
    _BODIES_OBJECTS[body.body_snapshot_sha256] = body
    B, P0, K0, norm_squared = _frame_constants(body.body_snapshot_sha256)
    scalar = algebra.scalar if type(algebra) is QuarterAngularAlgebra else algebra
    C = algebra.constant
    Sc = scalar.constant
    R, b = C(body.radius_m), C(body.beta_cut_radians)
    cube = C((body.radius_m+body.roof_radial_displacement_m)**3)-3*R*R*depth
    r = algebra.cbrt(cube)
    tau = (R**3-cube)/(3*R*R)
    sb, cb = algebra.sin(beta), algebra.cos(beta)
    v = (cosine*cb, sine*cb, sb)
    tangent = (-cosine*sb, -sine*sb, cb)
    K = tuple(tuple(C(x) for x in row) for row in K0)
    rate = K[2][0]*cosine+K[2][1]*sine
    grad = tuple(t/r+(r*r/(R*R))*rate*x for t, x in zip(tangent, v))
    scaled = beta/b
    chi = (C(1)-scaled)**2*(scaled+C(2))/2
    slope = -C(F(3, 2))*(C(1)-scaled)*(C(1)+scaled)/b
    # All frame coefficients are theta-independent.  Keeping them in the
    # scalar algebra avoids repeatedly multiplying constant theta polynomials.
    P = tuple(tuple(Sc(x) for x in row) for row in P0)
    if norm_squared:
        scalar_tau = tau._constant_scalar() if type(algebra) is QuarterAngularAlgebra else tau
        rate_norm = scalar.sqrt(Sc(norm_squared))
        ss, cc = scalar.sin(rate_norm*scalar_tau), scalar.cos(rate_norm*scalar_tau)
        KS = tuple(tuple(Sc(x) for x in row) for row in K0)
        K2 = _matmul(KS, KS)
        rotation = tuple(tuple(Sc(int(i == j))+ss/rate_norm*KS[i][j]
            +(Sc(1)-cc)/norm_squared*K2[i][j] for j in range(3)) for i in range(3))
        frame_scalar = _matmul(P, rotation)
    else:
        frame_scalar = P
    # The global xx/yy/zz/xy/xz/yz tensors are rank-one/symmetric rank-two.
    # This is C.T Ei C, including the genuinely depth-dependent Ei rotation.
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    matrices = tuple(tuple(tuple(C(frame_scalar[a][i]*frame_scalar[b0][j]
        if a == b0 else frame_scalar[a][i]*frame_scalar[b0][j]+frame_scalar[b0][i]*frame_scalar[a][j])
        for j in range(3)) for i in range(3)) for a, b0 in pairs)
    displacements = [tuple(chi*x for x in v)]
    gradients = [tuple(tuple(chi/r*(C(int(i == j))-v[i]*v[j])+slope*v[i]*grad[j]
                           for j in range(3)) for i in range(3))]
    for matrix in matrices:
        direction = tuple(r/R*x for x in _map(matrix, v))
        displacements.append(tuple(chi*x for x in direction))
        gradients.append(tuple(tuple(chi*matrix[i][j]/R+slope*direction[i]*grad[j]
                              for j in range(3)) for i in range(3)))
    frame = tuple(tuple(C(x) for x in row) for row in frame_scalar)
    return tuple(displacements), tuple(gradients), frame, v, grad, r, tau, chi, slope, K, B, P0


@dataclass(frozen=True, slots=True)
class NativeCommonColdModalGeometry:
    world_point_m: tuple
    radial_direction: tuple
    material_beta_gradient_per_m: tuple
    reference_flow_gradient_after_owner_rotation: tuple
    reference_gradient_from_original_cartesian: tuple
    mode_displacement_per_m: tuple
    mode_gradient_m_inv: tuple
    computational_mode_gradient_m_inv: tuple
    computational_frame: tuple
    radius: object
    physical_volume_depth: object
    source_shape: object
    source_shape_beta_derivative: object
    body_snapshot_sha256: str
    reference_snapshot_sha256: str

    @property
    def exact_reference_jacobian_determinant(self):
        return F(1)

    @property
    def material_membership_authenticated(self):
        return False


def evaluate_native_common_cold_modal_geometry(body, sine_theta, cosine_theta, beta, z, algebra):
    """Analytic interior formula for 0<=beta<=b; membership is separate.

    Use bound_native_common_cold_point for material-coordinate point queries:
    it implements the exact zero extension and clamp partition explicitly.
    """
    body = validate_native_common_cold_body(body)
    modes, gradients, frame, v, grad, r, tau, chi, slope, K, B, P = _modal_local_fields(
        body, sine_theta, cosine_theta, beta, z, algebra)
    e = _map(frame, v)
    point = tuple(r*x for x in e)
    world_modes = tuple(_map(frame, value) for value in modes)
    world_gradients = tuple(_matmul(_matmul(frame, value), _transpose(frame)) for value in gradients)
    local_reference = tuple(tuple(algebra.constant(int(i == j))
        -r**3/body.radius_m**2*_map(K, v)[i]*v[j] for j in range(3)) for i in range(3))
    reference_original = _matmul(_matmul(frame, local_reference),
                                tuple(tuple(algebra.constant(x) for x in row) for row in _transpose(B)))
    reference_after_owner = _matmul(_matmul(frame, local_reference),
                                   tuple(tuple(algebra.constant(x) for x in row) for row in _transpose(P)))
    return NativeCommonColdModalGeometry(point, e, _map(frame, grad), reference_after_owner,
        reference_original, world_modes, world_gradients, gradients, frame, r, tau, chi, slope,
        body.body_snapshot_sha256, body.reference_snapshot_sha256)


@dataclass(frozen=True, slots=True)
class NativeCommonColdPointBounds:
    body: NativeCommonColdBody
    reference_point_bounds_m: tuple
    displaced_point_bounds_m: tuple
    modal_displacement_bounds: tuple
    modal_gradient_bounds_m_inv: tuple
    incremental_gradient_bounds: tuple
    incremental_determinant_bounds: I
    field_active_inside_clamp: bool

    @property
    def material_membership_authenticated(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_common_cold_point(body, theta_bounds, beta_bounds, material_depth_bounds_m,
                                  amplitudes_m=(F(0),)*7, *, precision=160):
    """Physical common-reference point with the exact source clamp extension.

    A query straddling the material clamp is split by the caller.  At and
    beyond beta=b, the seven increments and gradients are identically zero;
    the full authentic material and its NEW reference point remain present.
    """
    body = validate_native_common_cold_body(body)
    coordinates = _coordinates(amplitudes_m)
    if (any(type(x) is not I for x in (theta_bounds, beta_bounds, material_depth_bounds_m))
            or not 0 <= theta_bounds.lo <= theta_bounds.hi <= interval_pi(precision).hi/2
            or not 0 <= beta_bounds.lo <= beta_bounds.hi <= interval_pi(precision).hi/2
            or not 0 <= material_depth_bounds_m.lo <= material_depth_bounds_m.hi <= body.areal_material_thickness_m):
        raise ValueError("Common cold points need valid complete material-coordinate bounds")
    if beta_bounds.lo < body.beta_cut_radians < beta_bounds.hi:
        raise ValueError("Split a common cold point query at the exact material clamp")
    algebra = TaylorAlgebra(0, precision)
    geometry = evaluate_native_common_cold_modal_geometry(body,
        algebra.sin(algebra.constant(theta_bounds)), algebra.cos(algebra.constant(theta_bounds)),
        algebra.constant(beta_bounds), algebra.constant(material_depth_bounds_m), algebra)
    point = tuple(x.constant_bounds for x in geometry.world_point_m)
    active = beta_bounds.lo < body.beta_cut_radians
    zero_vector, zero_matrix = (I(0),)*3, ((I(0),)*3,)*3
    modes = tuple(tuple(x.constant_bounds for x in vector) for vector in geometry.mode_displacement_per_m) if active else (zero_vector,)*7
    gradients = tuple(tuple(tuple(x.constant_bounds for x in row) for row in matrix)
                      for matrix in geometry.mode_gradient_m_inv) if active else (zero_matrix,)*7
    displaced = tuple(point[i]+sum((coordinates[a]*modes[a][i] for a in range(7)), I(0)) for i in range(3))
    gradient = tuple(tuple(I(int(i == j))+sum((coordinates[a]*gradients[a][i][j]
                    for a in range(7)), I(0)) for j in range(3)) for i in range(3))
    cross = (gradient[1][1]*gradient[2][2]-gradient[1][2]*gradient[2][1],
             gradient[1][2]*gradient[2][0]-gradient[1][0]*gradient[2][2],
             gradient[1][0]*gradient[2][1]-gradient[1][1]*gradient[2][0])
    return NativeCommonColdPointBounds(body, point, displaced, modes, gradients,
                                      gradient, _dot(gradient[0], cross), active)


@dataclass(frozen=True, slots=True)
class NativeCommonColdStrainSupport(NativeColdStrainSupport):
    reference_snapshot_sha256: str = ""
    active_angular_rate_per_physical_volume_depth_m: tuple = ()
    historical_angular_rate_per_physical_volume_depth_m: tuple = ()
    new_transported_three_dimensional_clamp_used: bool = True


def certify_native_common_cold_strain_support(body, coordinates_m, *, precision=160):
    """Rebuild the whole-mask local metric/J proof using literal NEW A*."""
    body = validate_native_common_cold_body(body)
    coordinates = _coordinates(coordinates_m)
    R, top = body.radius_m, body.radius_m+body.roof_radial_displacement_m
    cube = top**3-3*R**2*body.areal_material_thickness_m
    if cube <= 0:
        raise ValueError("The complete common cold material reaches a nonpositive radius")
    radius = I(interval_cbrt(I(cube), precision).lo, top)
    norm_A = interval_sqrt(I(sum((x*x for x in body.angular_rate_per_physical_volume_depth_m), F(0))), precision).hi
    b = body.beta_cut_radians
    if not 0 < b <= F(1, 4):
        raise ValueError("The source clamp angle exceeds the common gradient proof domain")
    beta_norm = 1/radius.lo+(top**2/R**2)*norm_A/(1-b*b/2)
    tip_norm = 1/radius.lo+F(3, 2)/b*beta_norm
    enrichment_norm = 1/R+F(3, 2)/b*(top/R)*beta_norm
    initial_norm = 2*abs(body.accepted_tip_displacement_m)*tip_norm
    increment_norm = abs(coordinates[0])*tip_norm+sum((abs(x)*enrichment_norm for x in coordinates[1:]), F(0))
    total_metric = I((1-increment_norm)**2-initial_norm, (1+increment_norm)**2+initial_norm)
    if initial_norm >= 1 or increment_norm >= 1 or total_metric.lo <= 0:
        raise ValueError("The actual NEW common strain does not certify the positive local metric/J domain")
    return NativeCommonColdStrainSupport(body.body_snapshot_sha256, coordinates, radius,
        tip_norm, enrichment_norm, I(1-initial_norm, 1+initial_norm), increment_norm,
        I((1-increment_norm)**3, (1+increment_norm)**3), total_metric,
        body.reference_snapshot_sha256, body.angular_rate_per_physical_volume_depth_m,
        body.historical_angular_rate_per_physical_volume_depth_m)


def _common_tensor_fields(body, coordinates, beta, depth, scalar_algebra, *,
                          theta_sine_bounds=None, theta_cosine_bounds=None):
    algebra = QuarterAngularAlgebra(scalar_algebra)
    sine = algebra.sine_theta() if theta_sine_bounds is None else algebra.constant(theta_sine_bounds)
    cosine = algebra.cosine_theta() if theta_cosine_bounds is None else algebra.constant(theta_cosine_bounds)
    _, gradients, *_ = _modal_local_fields(body, sine, cosine, algebra.constant(beta), algebra.constant(depth), algebra)
    active = tuple(a for a in range(7) if coordinates[a])
    L = tuple(tuple(sum((coordinates[a]*gradients[a][i][j] for a in active), algebra.constant(0))
                    for j in range(3)) for i in range(3))
    H0, LTL = _sym(gradients[0]), _sym_gram(L, L)
    linear = tuple(_sym(value) for value in gradients)
    nonlinear = tuple(_sym_gram(L, value) for value in gradients)
    green = tuple(tuple(sum((coordinates[a]*linear[a][i][j] for a in range(1, 7)), 0)+LTL[i][j]/2
                        for j in range(3)) for i in range(3))
    first = tuple(tuple(tuple((linear[a][i][j] if a else algebra.constant(0))+nonlinear[a][i][j]
                             for j in range(3)) for i in range(3)) for a in range(7))
    second = tuple(_sym_gram(gradients[a], gradients[b]) for a in range(7) for b in range(a, 7))
    if any(i+j > 6 for matrix in (H0, green)+first+second
           for row in matrix for value in row for i, j in value.coefficients):
        raise ValueError("The NEW common strain generators exceeded degree six before contraction")
    return H0, green, first, second


def common_cold_tensor_angular_kernel(body, coordinates_m, beta, depth, scalar_algebra, *, lame_pa, shear_pa):
    body = validate_native_common_cold_body(body)
    fields = _common_tensor_fields(body, _coordinates(coordinates_m), beta, depth, scalar_algebra)
    values = _response_values(fields, lambda x, y: _quarter_tensor_inner(x, y, lame_pa, shear_pa))
    factor = scalar_algebra.cos(beta)*body.radius_m**2
    return tuple(value*factor for value in values)


@dataclass(frozen=True, slots=True)
class RawCommonColdTensorIntegral(RawColdTensorIntegral):
    reference_snapshot_sha256: str = ""
    source_body_snapshot_sha256: str = ""
    source_material_snapshot_sha256: str = ""
    active_angular_rate_per_physical_volume_depth_m: tuple = ()
    historical_angular_rate_per_physical_volume_depth_m: tuple = ()
    fixed_global_tensor_generators_retained: bool = True
    depth_dependent_computational_generators_retained: bool = True
    donor_frame_reflection_retained_by_matrix_conjugation: bool = True
    body_tensor_angular_degree_upper_bound: int = 12
    tip_norm_lower_bound_valid_for_every_skew_depth_generator: bool = True


def _common_raw_receipt(body, material, coordinates, quarter, correction, cell_count,
                        tails, order, precision, cell_receipts, correction_receipts):
    actual = tuple(_rounded_interval(a-b, precision) for a, b in zip(quarter, correction))
    # This geometric coercivity inequality is derived from NEW H0 as well:
    # two tangential diagonals chi/r and transverse shear chi_beta/(2r)
    # survive every skew A*.  The added radial diagonal is nonnegative in
    # the Frobenius norm, while lambda*(trace H0)^2 remains nonnegative.
    # Reusing its exact moment arithmetic does not inherit an A=0 proof.
    proof = _reference_tip_stiffness_lower_bound(body, material, precision)
    if actual[0].hi < proof.stiffness_lower_bound_n_m:
        raise ValueError("The NEW supported tensor norm contradicts its independent geometric coercivity")
    k = I(max(actual[0].lo, proof.stiffness_lower_bound_n_m), actual[0].hi)
    prem, prem_first, second = actual[1], actual[2:9], _symmetric_rows(actual[9:37])
    rem_norm, rem_first, rem_second = actual[37], actual[38:45], _symmetric_rows(actual[45:73])
    q0 = coordinates[0]
    projection = q0*k+prem
    projection_first = tuple((k if i == 0 else I(0))+prem_first[i] for i in range(7))
    norm = q0*q0*k+2*q0*prem+rem_norm
    first = tuple((q0*k+prem if i == 0 else I(0))+q0*prem_first[i]+rem_first[i] for i in range(7))
    hessian = tuple(tuple((k if i == j == 0 else I(0))+
        (prem_first[j] if i == 0 else I(0))+(prem_first[i] if j == 0 else I(0))+
        rem_second[i][j]+q0*second[i][j] for j in range(7)) for i in range(7))
    values = (body.component_id, body.body_snapshot_sha256, material.material_snapshot_sha256, coordinates,
        k, actual[0], proof, projection, projection_first, second, norm, first, hessian,
        prem, prem_first, rem_norm, rem_first, rem_second, quarter, correction, actual,
        cell_count, tails, order, precision, cell_receipts, correction_receipts)
    extras = dict(reference_snapshot_sha256=body.reference_snapshot_sha256,
        source_body_snapshot_sha256=body.original_body.body_snapshot_sha256,
        source_material_snapshot_sha256=material.source_material_snapshot_sha256,
        active_angular_rate_per_physical_volume_depth_m=body.angular_rate_per_physical_volume_depth_m,
        historical_angular_rate_per_physical_volume_depth_m=body.historical_angular_rate_per_physical_volume_depth_m)
    provisional = RawCommonColdTensorIntegral(*values, "", **extras)
    receipt = replace(provisional, integral_snapshot_sha256=_snapshot_sha(provisional))
    _INTEGRALS.add(receipt.integral_snapshot_sha256)
    return receipt


def validate_native_common_cold_tensor_integral(value, *, body=None, material=None):
    if (type(value) is not RawCommonColdTensorIntegral
            or value.integral_snapshot_sha256 != _snapshot_sha(value)
            or value.integral_snapshot_sha256 not in _INTEGRALS):
        raise ValueError("A genuine immutable NEW common cold tensor integral is required")
    if body is not None:
        body = validate_native_common_cold_body(body)
        if (value.component_id != body.component_id
                or value.body_snapshot_sha256 != body.body_snapshot_sha256
                or value.reference_snapshot_sha256 != body.reference_snapshot_sha256
                or value.source_body_snapshot_sha256 != body.original_body.body_snapshot_sha256
                or value.active_angular_rate_per_physical_volume_depth_m != body.angular_rate_per_physical_volume_depth_m):
            raise ValueError("The tensor integral belongs to a different common/source binding")
    if material is not None:
        validate_native_common_cold_strain_material(material, body=body)
        if (value.material_snapshot_sha256 != material.material_snapshot_sha256
                or value.source_material_snapshot_sha256 != material.source_material_snapshot_sha256):
            raise ValueError("The common tensor integral belongs to a different material")
    if value.reference_tip_strain_stiffness_n_m.lo <= 0:
        raise ValueError("The NEW supported reference tip tensor norm lacks a positive proof")
    return value



def _strip_depth_kernel(body, coordinates, depth, scalar, *, lam, shear, parameters):
    d, e, case, cap, _ = parameters
    inner = TaylorAlgebra(scalar.order+1, scalar.precision, (1, scalar.order))
    beta = inner.variable(I(0), coordinate=0, scale=F(1))
    depth_inner = TaylorJet(inner, {(0, j): value for (i, j), value in depth.coefficients.items() if i == 0})
    fields = _common_tensor_fields(body, coordinates, beta, depth_inner, inner)
    first_moments = _weighted_endpoint_moments(d, e, case, cap, 1, scalar.precision)
    second_moments = _weighted_endpoint_moments(d, e, case, cap, 2, scalar.precision)
    first_values = _response_values(fields, lambda x, y: _quarter_tensor_inner(x, y, lam, shear, first_moments))
    second_values = _response_values(fields, lambda x, y: _quarter_tensor_inner(x, y, lam, shear, second_moments))
    factor = inner.cos(beta)*body.radius_m**2
    answer = []
    for first, second in zip(first_values, second_values):
        first, second = first*factor, second*factor
        coefficients = {(0, j): first.coefficient(0, j)+second.coefficient(1, j)/2
                        for j in range(scalar.order+1)}
        answer.append(TaylorJet(scalar, coefficients))
    return tuple(answer)


def _bounded_primary_values(body, coordinates, beta_box, depth_box, *, precision, lam, shear,
                            sine_bounds=I(0, 1), cosine_bounds=I(0, 1), order=2,
                            include_angular_measure=True):
    algebra = TaylorAlgebra(order, precision, (order, 0))
    beta = algebra.variable(beta_box, coordinate=0, scale=F(1))
    depth = algebra.constant(depth_box)
    fields = _common_tensor_fields(body, coordinates, beta, depth, algebra,
        theta_sine_bounds=sine_bounds, theta_cosine_bounds=cosine_bounds)
    values = _response_values(fields, lambda x, y: _tensor_inner(x, y, lam, shear))
    factor = algebra.cos(beta)*body.radius_m**2 if include_angular_measure else algebra.constant(1)
    return tuple(value._constant_scalar()*factor for value in values)


def bound_native_common_cold_tensor_density(body, material, coordinates_m, theta_bounds,
        beta_bounds, material_depth_bounds_m, *, precision=96, layer):
    """Bound all73 correlated density terms at physical material coordinates.

    No area, thickness or angular measure multiplies these local tensor
    densities.  This point oracle does not issue an authenticated whole-mask
    integral receipt or establish material membership.
    """
    body = validate_native_common_cold_body(body)
    material = validate_native_common_cold_strain_material(material, body=body)
    coordinates = _coordinates(coordinates_m)
    # This also checks the coordinate domain and exact clamp partition.
    point = bound_native_common_cold_point(body, theta_bounds, beta_bounds,
                                          material_depth_bounds_m, coordinates, precision=precision)
    lam, shear = material.lame_coefficients(layer)
    lower, upper = ((F(0), body.elastic_layer_areal_depth_m) if layer == "upper"
                    else (body.elastic_layer_areal_depth_m, body.areal_material_thickness_m))
    if not lower <= material_depth_bounds_m.lo <= material_depth_bounds_m.hi <= upper:
        raise ValueError("A tensor density query straddles its authentic physical material layer")
    if not point.field_active_inside_clamp:
        return (I(0),)*73
    algebra = TaylorAlgebra(0, precision)
    values = _bounded_primary_values(body, coordinates, beta_bounds, material_depth_bounds_m,
        precision=precision, lam=lam, shear=shear,
        sine_bounds=algebra.sin(algebra.constant(theta_bounds)).constant_bounds,
        cosine_bounds=algebra.cos(algebra.constant(theta_bounds)).constant_bounds,
        order=0, include_angular_measure=False)
    return tuple(value.constant_bounds for value in values)


def _integrate_endpoint_strip(body, coordinates, layer, z0, z1, lam, shear, precision, order):
    parameters = _endpoint_strip_parameters(body, precision)
    d, e, case, cap, D = parameters
    if D > body.beta_cut_radians:
        raise ValueError("The native endpoint strip extends beyond the supported collar")
    integral = validated_integrate_rectangle(lambda unused, depth, scalar: _strip_depth_kernel(
        body, coordinates, depth, scalar, lam=lam, shear=shear, parameters=parameters),
        I(0, 1), I(z0, z1), order=order, precision=precision)
    if case == "terminal_cap_only":
        sine = I(max(F(0), 1-cap.hi**2/2), 1)
        cosine = I(0, cap.hi)
        angle_length = cap.hi
    else:
        sine = cosine = I(0, 1)
        angle_length = interval_pi(precision).hi/2
    bounded = _bounded_primary_values(body, coordinates, I(0, D), I(z0, z1),
        precision=precision, lam=lam, shear=shear, sine_bounds=sine, cosine_bounds=cosine)
    tails = tuple((max(abs(value.coefficient(0, 0).lo), abs(value.coefficient(0, 0).hi))*D**3/3+
                   max(abs(value.coefficient(1, 0).lo), abs(value.coefficient(1, 0).hi))*D**4/3+
                   max(abs(value.coefficient(2, 0).lo), abs(value.coefficient(2, 0).hi))*D**3/3)
                  *angle_length*(z1-z0) for value in bounded)
    corrected = tuple(_rounded_interval(value+I(-tail, tail), precision)
                      for value, tail in zip(integral.integral_bounds, tails))
    receipt = ColdTensorEndpointStripReceipt("endpoint_archive_exact_theta_strip", layer,
        body.endpoint_archive_area_bounds_sr, I(z0, z1), d, e, case, cap, D,
        integral.polynomial_integral_bounds, integral.remainder_absolute_upper_bounds, tails,
        tuple(value.constant_bounds for value in bounded), corrected)
    return receipt


def integrate_native_common_cold_tensor_response(body, material, coordinates_m, *, precision=96,
        taylor_order=0, beta_cell_count=1, depth_cell_count=1):
    """Integrate one actual cold supported field and retain every tensor term."""
    body = validate_native_common_cold_body(body)
    material = validate_native_common_cold_strain_material(material, body=body)
    if type(coordinates_m) is not tuple or len(coordinates_m) != 7 or any(type(q) is not F for q in coordinates_m):
        raise ValueError("Seven literal metre coordinates are required")
    if type(precision) is not int or not 64 <= precision <= 512:
        raise ValueError("Cold tensor precision must lie in [64,512]")
    if type(taylor_order) is not int or not 0 <= taylor_order <= 12:
        raise ValueError("Cold tensor Taylor order must lie in [0,12]")
    if any(type(n) is not int or n <= 0 for n in (beta_cell_count, depth_cell_count)):
        raise ValueError("Cold tensor cell counts must be positive literal integers")
    cache_key = (body.body_snapshot_sha256, material.material_snapshot_sha256,
        coordinates_m, precision, taylor_order, beta_cell_count, depth_cell_count)
    if cache_key in _INTEGRAL_CACHE:
        return validate_native_common_cold_tensor_integral(_INTEGRAL_CACHE[cache_key], body=body, material=material)
    quarter, correction, tails = [I(0)]*73, [I(0)]*73, [F(0)]*73
    cells, removals = [], []
    beta_cut = body.beta_cut_radians
    layers = (("upper", F(0), body.elastic_layer_areal_depth_m),
              ("lower", body.elastic_layer_areal_depth_m, body.areal_material_thickness_m))
    for layer, z0, z1 in layers:
        lam, shear = material.lame_coefficients(layer)
        for bi in range(beta_cell_count):
            beta_box = I(beta_cut*bi/beta_cell_count, beta_cut*(bi+1)/beta_cell_count)
            for zi in range(depth_cell_count):
                depth_box = I(z0+(z1-z0)*zi/depth_cell_count, z0+(z1-z0)*(zi+1)/depth_cell_count)
                evaluator = lambda beta, depth, algebra: common_cold_tensor_angular_kernel(
                    body, coordinates_m, beta, depth, algebra, lame_pa=lam, shear_pa=shear)
                integral = validated_integrate_rectangle(evaluator, beta_box, depth_box,
                                                        order=taylor_order, precision=precision)
                cells.append(ColdTensorCellReceipt(beta_box, depth_box, layer, lam, shear,
                    integral.polynomial_integral_bounds, integral.remainder_absolute_upper_bounds,
                    integral.integral_bounds))
                for index, value in enumerate(integral.integral_bounds):
                    quarter[index] = _rounded_interval(quarter[index]+value, precision)
                    tails[index] += integral.remainder_absolute_upper_bounds[index]
        if body.endpoint_archive_area_bounds_sr != I(0):
            strip = _integrate_endpoint_strip(body, coordinates_m, layer, z0, z1, lam, shear,
                                             precision, taylor_order)
            removals.append(strip)
            for index, value in enumerate(strip.correction_integral_bounds):
                correction[index] = _rounded_interval(correction[index]+value, precision)
        removed_measures = tuple(("paired_curved_removed", area)
                                 for area in body.paired_curved_removed_area_bounds_sr)
        for kind, area in removed_measures:
            if area == I(0):
                continue
            theta_width = body.paired_curved_cap_theta_width_bounds_radians
            beta_upper = body.paired_curved_beta_upper_bounds_radians
            if type(theta_width) is not I or type(beta_upper) is not I:
                raise ValueError("The curved correction requires its true swept narrow-cap source")
            sine = I(max(F(0), 1-theta_width.hi**2/2), 1)
            cosine = I(0, theta_width.hi)
            values = _bounded_primary_values(body, coordinates_m, I(0, beta_upper.hi), I(z0, z1),
                precision=precision, lam=lam, shear=shear, sine_bounds=sine, cosine_bounds=cosine,
                order=0, include_angular_measure=False)
            bounds = tuple(value.constant_bounds for value in values)
            multiplier = area*body.radius_m**2*(z1-z0)
            corrected = tuple(_rounded_interval(value*multiplier, precision) for value in bounds)
            removals.append(ColdTensorCorrectionReceipt(kind, layer, area, I(z0, z1), bounds, corrected,
                theta_width, I(0, beta_upper.hi)))
            for index, value in enumerate(corrected):
                correction[index] = _rounded_interval(correction[index]+value, precision)
    result = validate_native_common_cold_tensor_integral(_common_raw_receipt(body, material, coordinates_m,
        tuple(quarter), tuple(correction), len(cells), tuple(tails), taylor_order, precision,
        tuple(cells), tuple(removals)), body=body, material=material)
    _INTEGRAL_CACHE[cache_key] = result
    return result


@dataclass(frozen=True, slots=True)
class NativeCommonColdStrainEnergyResponse(NativeColdStrainEnergyResponse):
    reference_snapshot_sha256: str = ""
    historical_reference_embedding_inherited: bool = False
    new_transported_three_dimensional_clamp_used: bool = True


def evaluate_native_common_cold_strain_energy(body, coordinates_m, *, material=None,
        integrals=None, precision=96, taylor_order=0, beta_cell_count=1,
        depth_cell_count=1):
    """Objective full seven-mode U/g/H, calibrated to authentic old w/k.

    The initial strain is explicitly NEW w_ref*H0 on the common reference.
    The correlated Green-q0*H0 remainder preserves accepted source energy
    and tip force exactly at zero increment without an interval cancellation.
    """
    body = validate_native_common_cold_body(body)
    coordinates = _coordinates(coordinates_m)
    if material is None:
        material = build_native_common_cold_strain_material(body)
    material = validate_native_common_cold_strain_material(material, body=body)
    support = certify_native_common_cold_strain_support(body, coordinates, precision=max(precision, 160))
    if integrals is None:
        integrals = integrate_native_common_cold_tensor_response(body, material, coordinates,
            precision=precision, taylor_order=taylor_order,
            beta_cell_count=beta_cell_count, depth_cell_count=depth_cell_count)
    raw = validate_native_common_cold_tensor_integral(integrals, body=body, material=material)
    if raw.coordinates_m != coordinates:
        raise ValueError("Common tensor integrals describe different physical increments")
    k, prem = raw.reference_tip_strain_stiffness_n_m, raw.projection_remainder_n
    prem_gradient = raw.projection_remainder_gradient_n_m
    rem_norm, rem_gradient = raw.remainder_strain_inner_product_j, raw.remainder_local_gradient_n
    rem_hessian, second = raw.remainder_local_hessian_n_m, raw.projection_hessian_n_m2
    old_k, wref = body.source_old_tip_stiffness_n_m, body.accepted_tip_displacement_m
    reference_energy, reference_force = old_k*wref*wref/2, old_k*wref
    increment = I(coordinates[0])+prem/k
    total = I(wref)+increment
    derivatives = tuple(I(int(i == 0))+prem_gradient[i]/k for i in range(7))
    residual = rem_norm/2-_square(prem)/(2*k)
    if residual.hi < 0:
        raise ValueError("The complete strain integral contradicts its positive orthogonal complement")
    residual = I(max(F(0), residual.lo), residual.hi)
    energy = old_k*_square(total)/2+residual
    gradient = tuple(old_k*total*derivatives[i]+rem_gradient[i]-(prem/k)*prem_gradient[i] for i in range(7))
    hessian = tuple(tuple(old_k*derivatives[i]*derivatives[j]+rem_hessian[i][j]
        -prem_gradient[i]*prem_gradient[j]/k+((old_k*total-prem)/k)*second[i][j]
        for j in range(7)) for i in range(7))
    if not any(coordinates):
        energy = I(reference_energy)
        gradient = (I(reference_force),)+gradient[1:]
        first_row = tuple((I(old_k) if j == 0 else old_k*prem_gradient[j]/k)
            +(reference_force/k)*second[0][j] for j in range(7))
        hessian = (first_row,)+tuple((first_row[i],)+hessian[i][1:] for i in range(1, 7))
    return NativeCommonColdStrainEnergyResponse(body, material, raw, support, coordinates,
        reference_energy, reference_force, energy, gradient, hessian,
        raw.incremental_strain_inner_product_j/2, increment, total,
        body.reference_snapshot_sha256)


def native_common_cold_mathematical_body_view(body):
    """Explicit computational replay view, never an authenticated source DTO."""
    from .geometric_native_cold_body import native_cold_body_to_dict
    body = validate_native_common_cold_body(body)
    result = native_cold_body_to_dict(body.original_body)
    result.update(geometry_source_sha256=body.body_snapshot_sha256,
        body_snapshot_sha256=body.body_snapshot_sha256,
        source_body_snapshot_sha256=body.original_body.body_snapshot_sha256,
        reference_snapshot_sha256=body.reference_snapshot_sha256,
        embedding_bank_snapshot_sha256=body.embedding_bank.bank_snapshot_sha256,
        historical_angular_rate_per_physical_volume_depth_m=_encoded(body.historical_angular_rate_per_physical_volume_depth_m),
        angular_rate_per_physical_volume_depth_m=_encoded(body.angular_rate_per_physical_volume_depth_m),
        angular_rate_role="NEW_computational_embedding_not_original_saved128_source",
        reference_model=body.reference_model,
        historical_three_dimensional_clamp_inherited=False,
        new_transported_three_dimensional_clamp_used=True)
    return result


def native_common_cold_body_to_dict(body):
    from .geometric_native_cold_body import native_cold_body_to_dict
    body = validate_native_common_cold_body(body)
    return dict(original_body=native_cold_body_to_dict(body.original_body),
        embedding_bank=_encoded(body.embedding_bank),
        body_snapshot_sha256=body.body_snapshot_sha256,
        mathematical_body_view=native_common_cold_mathematical_body_view(body))


def native_common_cold_strain_material_to_dict(material):
    from .geometric_native_cold_strain_energy import native_cold_strain_material_to_dict
    material = validate_native_common_cold_strain_material(material)
    original = native_cold_strain_material_to_dict(material.original_material)
    result = dict(original)
    result.update(original_material=original,
        source_material_snapshot_sha256=material.source_material_snapshot_sha256,
        source_body_snapshot_sha256=material.original_material.body_snapshot_sha256,
        body_snapshot_sha256=material.body_snapshot_sha256,
        reference_snapshot_sha256=material.reference_snapshot_sha256,
        material_snapshot_sha256=material.material_snapshot_sha256,
        material_binding_role="NEW_common_reference_binding_of_immutable_source_tensor_declaration")
    return result


def native_common_cold_strain_energy_to_dict(response):
    if type(response) is not NativeCommonColdStrainEnergyResponse:
        raise ValueError("A genuine common-reference cold strain response is required")
    body = validate_native_common_cold_body(response.body)
    validate_native_common_cold_strain_material(response.material, body=body)
    validate_native_common_cold_tensor_integral(response.raw_integrals, body=body, material=response.material)
    result = _payload(response)
    result["body"] = native_common_cold_mathematical_body_view(body)
    result["material"] = native_common_cold_strain_material_to_dict(response.material)
    result["common_body_binding"] = native_common_cold_body_to_dict(body)
    for name in ("nonlocal_constitutive_calibration_declared", "accepted_global_beam_reference_energy_preserved",
            "accepted_global_beam_reference_tip_force_preserved", "old_beam_stiffness_added_as_second_energy",
            "historical_full_three_dimensional_prestress_recovered", "historical_archived_strain_energy_density_recovered",
            "complete_three_dimensional_strain_energy_certified", "mixed_force_equilibrium_certified",
            "material_admission_authorized", "original_point_1_completed"):
        result[name] = getattr(response, name)
    return result

"""One bank-wide nonlinear reduction of the retained source into COMMON cold fields.

Authentic whole-bank strain integrals supply eight projections of seven-mode
fields.  The source16 and its eight beams are composed with that one map; the
orthogonal bulk remainder is added once.  This is a new nonlocal reduced law,
not pointwise COMMON-face traction equivalence or archived 3D body recovery.
Evaluation requires registered integrals at the complete literal coordinates
and never runs a solver or quadrature or chooses an interval midpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction as F
import hashlib
from math import isqrt

from .geometric_native_pose import FractionInterval as I
from .geometric_native_common_contact_ports import _encoded, _payload
from .geometric_native_common_source_carrier import (
    _precision, _round, _dot, _source_jet, _world_reactions, _read_fraction, BRANCHES)
from .geometric_native_common_material_transfer import (
    validate_native_common_material_transfer_system, _DEPENDENCIES as _TRANSFER_DEPENDENCIES)
from .geometric_native_common_cold import (
    validate_native_common_cold_body, validate_native_common_cold_strain_material,
    validate_native_common_cold_tensor_integral)

FORMAT = "native-common-bankwise-strain-projection-system-1"
RESPONSE_FORMAT = "native-common-bankwise-strain-projection-response-1"
SCOPE = "COMMON_BANKWISE_COLD_STRAIN_PROJECTION_REDUCTION_COMPONENT_ONLY"
MODEL = "ONE_SOURCE16_BEAM8_POTENTIAL_COMPOSED_WITH_WHOLE_COLD_STRAIN_PROJECTION_PLUS_ORTHOGONAL_BULK_ONCE"
FORCE_BASIS = "archival_old_reference"
ENERGY_OWNERSHIP = "SOURCE_BEAM_PROJECTION_AND_COMMON_ORTHOGONAL_BULK_ONCE_NO_SECOND_CALIBRATED_BEAM"
OWNER_COVARIANCE = "FRESH_LEFT_RIGID_OWNER_ROTATION_SUPERPOSED_AFTER_FROZEN_COMMON_REFERENCE_AND_BODY_DEFORMATION_WITH_COROTATED_FIXED_REFERENCE_GENERATORS"
_FLAGS = dict(
    source_internal_eight_reduced_into_cold_body_fields=True,
    all_fifty_six_cold_coordinates_in_projection_catalogue=True,
    source_hot_thirty_coordinate_coupling_exact_zero=True,
    full_gradient_hessian_chainrule_including_prestress_curvature=True,
    source_literal_reference_energy_work_ids_and_clocks_preserved=True,
    orthogonal_bulk_remainder_energized_once=True,
    midpoint_projection_or_per_node_pseudoinverse_used=False,
    pointwise_common_face_traction_equivalence_certified=False,
    physical_archived_three_dimensional_material_handover_certified=False,
    hot_receiving_contact_or_new_mean_law_certified=False,
    complete_physical_source_material_handover_certified=False,
    new_mixed_force_equilibrium_certified=False,
    whole_clock_all_pair_admission_authorized=False,
    source_state_or_work_advanced=False,
    original_point_1_completed=False,
    finite_owner_increments_supported=False,
    old_total_finite_path_work_equal_to_new_bulk_work=False,
)
_REGISTRY = {}
_DEPENDENCIES = {}
_BIND_CACHE = {}
_QUERY_CACHE = {}
_CONTEXT_TYPES = {}


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _content_types(value):
    """Pin recursive Python kinds as well as the complete canonical values.

    Canonical DTO encoding deliberately represents tuples/lists and Fraction
    pairs alike. Typed replay must additionally reject replacing a registered
    dataclass or a literal Fraction with a payload-equivalent untyped value.
    This tree retains no mutable object or large source string.
    """
    kind = type(value)
    if is_dataclass(value):
        return kind, tuple((f.name, _content_types(getattr(value, f.name))) for f in fields(value))
    if kind in (tuple, list):
        return kind, tuple(_content_types(v) for v in value)
    if kind is dict:
        return kind, tuple((type(k), k, _content_types(v)) for k, v in
            sorted(value.items(), key=lambda row: (type(row[0]).__name__, repr(row[0]))))
    return kind


def _seal(value):
    return _digest({f.name: getattr(value, f.name) for f in fields(value) if f.name != "binding_sha256"})


def _remember(value, dependencies):
    value = replace(value, binding_sha256=_seal(value))
    key = type(value), value.binding_sha256
    _REGISTRY[key], _DEPENDENCIES[key] = _encoded(value), dependencies
    return value


def _registered(value, kind):
    if (type(value) is not kind or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((kind, value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or resealed bankwise projection lacks complete authentic replay")
    return _DEPENDENCIES[kind, value.binding_sha256]


def _square(value):
    return I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


def _contains_zero(value):
    return value.lo <= 0 <= value.hi


def _projection_cauchy_cap(k, N, precision):
    """Upward dyadic bound for |p/k| from the SAME tensor inner product.

    Here k=<H0,H0> and the ORIGINAL channel37 N=<E_rem,E_rem>, not
    its half-energy. Cauchy-Schwarz gives p**2 <= k*N, hence
    |p/k| <= sqrt(N.hi/k.lo). The cap contracts an enclosure of the
    unchanged quotient; it is not a constitutive clip or a differentiated
    function. All rounding uses integers, never an interval centre.
    """
    if (k.lo <= 0 or N.hi < 0 or type(precision) is not int or precision <= 0):
        raise ValueError("A Cauchy projection requires its positive norm and original nonnegative remainder norm")
    ratio = N.hi/k.lo
    bits = (3322*(precision+16)+999)//1000+8
    scale = 1 << bits
    upward_square = -((-ratio.numerator*scale*scale)//ratio.denominator)
    root = isqrt(upward_square)
    if root*root < upward_square:
        root += 1
    cap = F(root, scale)
    if cap*cap < ratio:
        raise ValueError("The integer Cauchy square-root cap lacks its upward-square certificate")
    return cap


@dataclass(frozen=True, slots=True)
class NativeCommonStrainProjectionBlock:
    component_id: str
    trace_id: str
    side: str
    source_coordinate_index: int
    global_coordinate_indices: tuple
    coordinates_m: tuple
    body_snapshot_sha256: str
    material_snapshot_sha256: str
    raw_integral_sha256: str
    reference_tip_strain_stiffness_n_m: I
    projection_remainder_n: I
    projection_remainder_gradient_n_m: tuple
    projection_remainder_hessian_n_m2: tuple
    remainder_strain_inner_product_j: I
    remainder_local_gradient_n: tuple
    remainder_local_hessian_n_m: tuple
    projection_increment_bounds_m: I
    projection_gradient_bounds: tuple
    projection_hessian_bounds: tuple
    orthogonal_energy_bounds_j: I
    orthogonal_gradient_bounds_n: tuple
    orthogonal_hessian_bounds_n_m: tuple
    reconstructed_origin_projection_remainder_bounds_n: I
    reconstructed_origin_tip_projection_derivative_bounds: I
    projection_quotient_bounds_m: I
    projection_quotient_cauchy_absolute_upper_bound_m: F
    reference_projection_tip_derivative_exact: F = F(1)
    source_kind: str = "AUTHENTIC_WHOLE_COMMON_COLD_STRAIN_INTEGRAL"
    pointwise_source_endpoint_motion_equivalence_certified: bool = False


@dataclass(frozen=True, slots=True)
class NativeCommonSourceInverseTwoJet:
    anchor: str
    global_coordinate_order: tuple
    source_coordinate_order: tuple
    source_anchor_coordinate_bounds: tuple
    global_anchor_coordinate_increments: tuple
    source_to_global_first_derivative_bounds: tuple
    source_to_global_second_derivative_blocks: tuple
    inverse_tip_first_derivative_bounds: tuple
    inverse_tip_second_derivative_bounds: tuple
    first_composition_identity_residual_bounds: tuple
    second_composition_identity_residual_bounds: tuple
    source_only_hessian_curvature_cancellation_bounds: tuple
    positive_tip_derivative_certified: bool
    correlated_inverse_identity_defined_symbolically: bool = True
    source_only_operator_certificate: bool = True
    zero_bulk_quadrature_invented: bool = False
    total_bulk_potential_hessian_identified_with_old_source: bool = False


@dataclass(frozen=True, slots=True)
class NativeCommonStrainProjectionSystem:
    source_transfer_binding_sha256: str
    source_snapshot_sha256: str
    reference_snapshot_sha256: str
    source_force_basis: str
    coordinate_order: tuple
    source_coordinate_order: tuple
    bound_coordinate_increments: tuple
    projection_blocks: tuple
    coupled_cold_coordinate_indices: tuple
    hot_zero_coordinate_indices: tuple
    coordinate_to_source_index: tuple
    coordinate_source_projection_derivative_bounds: tuple
    projection_jacobian_bounds: tuple
    reference_source_injection: tuple
    reference_inverse_curvature_blocks: tuple
    reference_origin_inverse_2jet: NativeCommonSourceInverseTwoJet
    cold_context_binding_sha256: str
    complete_original_source_material_state_json: str
    complete_current_source_states_json: str
    complete_current_source_memory_json: str
    complete_before_work_ledgers_json: str
    complete_unaccepted_candidate_work_ledgers_json: str
    original_source_work_lineages: tuple
    source_serial_reference_energy_j: F
    source_beam_literal_reference_energy_j: F
    source_beam_energy_rounding_residual_j: F
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    source_memory_elapsed_time_s: F
    source_geometry_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    provenance: str
    precision: int
    binding_sha256: str
    energetic_owner: str = ENERGY_OWNERSHIP
    source_state_or_work_advanced: bool = False
    finite_owner_increments_supported: bool = False
    source_internal_coordinate_count: int = 0
    source_cold_coordinate_count: int = 56
    source_hot_zero_coordinate_count: int = 30
    physical_source_material_handover_certified: bool = False
    owner_body_covariance_law: str = OWNER_COVARIANCE
    bank_strain_projection_invariant_under_superposed_owner_rotation: bool = True

    @property
    def scope(self):
        return SCOPE


def _projection_algebra(k, p, dp, hp, N, dN, hN, coordinates_m, precision):
    """Pure whole-bank strain algebra; gN/HN differentiate N/2, not N.

    The same helper is used for every authenticated production block. Scalar
    inputs are rigorous intervals; no fake registered context is required for
    an independent exact-polynomial mathematical check of this operation.
    """
    if k.lo <= 0 or len(dp) != 7 or len(hp) != 7 or any(len(row) != 7 for row in hp):
        raise ValueError("A bankwide projection needs its proved positive tensor norm and complete second variations")
    cap = _projection_cauchy_cap(k, N, precision)
    raw_quotient = p/k
    lower, upper = max(raw_quotient.lo, -cap), min(raw_quotient.hi, cap)
    if lower > upper:
        raise ValueError("The registered projection contradicts its same-inner-product Cauchy certificate")
    quotient = I(lower, upper)
    s = _round(I(coordinates_m[0])+quotient, precision)
    ds = tuple(_round(I(int(i == 0))+dp[i]/k, precision) for i in range(7))
    hs = tuple(tuple(_round(hp[i][j]/k, precision) for j in range(7)) for i in range(7))
    R = _round(N/2-_square(p)/(2*k), precision)
    if R.hi < 0:
        raise ValueError("The whole strain integral contradicts its nonnegative orthogonal complement")
    R = I(max(F(0), R.lo), R.hi)
    gR = tuple(_round(dN[i]-quotient*dp[i], precision) for i in range(7))
    hR = tuple(tuple(_round(hN[i][j]-dp[i]*dp[j]/k-quotient*hp[i][j], precision)
        for j in range(7)) for i in range(7))
    # prem is exactly quadratic because Green-q_tip H0 contains no linear
    # tip term. These are consistency bounds for the structural origin,
    # not a fabricated origin integral or an extrapolated bulk Hessian.
    q = coordinates_m
    origin_p = _round(p-sum((dp[i]*q[i] for i in range(7)), I(0))
        +sum((q[i]*hp[i][j]*q[j]/2 for i in range(7) for j in range(7)), I(0)), precision)
    origin_tip_dp = _round(dp[0]-sum((hp[0][j]*q[j] for j in range(7)), I(0)), precision)
    if not _contains_zero(origin_p) or not _contains_zero(origin_tip_dp) or ds[0].lo <= 0:
        raise ValueError("The quadratic source Green projection or its monotone bankwise tip chart is unresolved")
    return dict(projection_increment_bounds_m=s, projection_gradient_bounds=ds,
        projection_hessian_bounds=hs, orthogonal_energy_bounds_j=R,
        orthogonal_gradient_bounds_n=gR, orthogonal_hessian_bounds_n_m=hR,
        reconstructed_origin_projection_remainder_bounds_n=origin_p,
        reconstructed_origin_tip_projection_derivative_bounds=origin_tip_dp,
        projection_quotient_bounds_m=quotient,
        projection_quotient_cauchy_absolute_upper_bound_m=cap)


def _projection_block(body, material, raw, transfer, precision):
    validate_native_common_cold_body(body)
    validate_native_common_cold_strain_material(material, body=body)
    validate_native_common_cold_tensor_integral(raw, body=body, material=material)
    original = body.original_body
    order = transfer.coordinate_order[:92]
    indices = tuple(order.index(("body", body.component_id, name)) for name in body.coordinate_order)
    source_index = transfer.source_coordinate_order.index(("radial", original.trace_id, original.source_bank))
    if (body.reference_snapshot_sha256 != transfer.reference_snapshot_sha256
            or len(indices) != 7 or len(raw.coordinates_m) != 7
            or any(type(x) is not F for x in raw.coordinates_m)
            or body.coordinate_order[0] != "old_tip_increment"):
        raise ValueError("A source projection requires its complete authentic COMMON seven-mode cold field")
    beam = next(b for b in transfer.beams if b.source_coordinate_index == source_index)
    if (beam.component_id != body.component_id or beam.trace_id != original.trace_id
            or beam.side != original.source_bank
            or beam.stiffness_exact_n_m != original.source_old_tip_stiffness_n_m
            or beam.reference_displacement_m != original.accepted_tip_displacement_m):
        raise ValueError("The bankwide strain projection cannot substitute an original global beam")
    k, p = raw.reference_tip_strain_stiffness_n_m, raw.projection_remainder_n
    dp, hp = raw.projection_remainder_gradient_n_m, raw.projection_hessian_n_m2
    N, dN, hN = raw.remainder_strain_inner_product_j, raw.remainder_local_gradient_n, raw.remainder_local_hessian_n_m
    values = _projection_algebra(k, p, dp, hp, N, dN, hN, raw.coordinates_m, precision)
    return NativeCommonStrainProjectionBlock(body.component_id, original.trace_id, original.source_bank,
        source_index, indices, raw.coordinates_m, body.body_snapshot_sha256, material.material_snapshot_sha256,
        raw.integral_snapshot_sha256, k, p, dp, hp, N, dN, hN,
        values["projection_increment_bounds_m"], values["projection_gradient_bounds"],
        values["projection_hessian_bounds"], values["orthogonal_energy_bounds_j"],
        values["orthogonal_gradient_bounds_n"], values["orthogonal_hessian_bounds_n_m"],
        values["reconstructed_origin_projection_remainder_bounds_n"],
        values["reconstructed_origin_tip_projection_derivative_bounds"],
        values["projection_quotient_bounds_m"],
        values["projection_quotient_cauchy_absolute_upper_bound_m"])


def _inverse_twojet(order, source_order, blocks, source_anchor, global_anchor, *, reference_origin, precision):
    matrix = [[I(0) for _ in range(14)] for _ in range(92)]
    for i in range(6):
        matrix[i][i] = I(1)
    second, firsts, seconds, first_residual, second_residual = [], [], [], [], []
    for b in blocks:
        derivative = I(1) if reference_origin else b.projection_gradient_bounds[0]
        curvature = b.projection_hessian_bounds[0][0]
        if derivative.lo <= 0:
            raise ValueError("The single global inverse2jet needs eight proved monotone tip derivatives")
        alpha = I(1) if reference_origin else _round(1/derivative, precision)
        gamma = _round(-curvature if reference_origin else -curvature/(derivative*derivative*derivative), precision)
        matrix[b.global_coordinate_indices[0]][b.source_coordinate_index] = alpha
        second.append((b.source_coordinate_index, b.global_coordinate_indices[0], gamma))
        firsts.append(alpha); seconds.append(gamma)
        first_residual.append(I(0) if reference_origin else _round(derivative*alpha-1, precision))
        second_residual.append(_round(curvature*alpha*alpha+derivative*gamma, precision))
    if any(not _contains_zero(v) for v in first_residual+second_residual):
        raise ValueError("A bankwise inverse2jet fails the complete first/second composite identity")
    # The source-only Hessian cancellation is correlated algebra.  The bulk
    # remainder, whose finite response is separately retained, is not removed.
    return NativeCommonSourceInverseTwoJet(
        "STRUCTURAL_GREEN_ZERO_REFERENCE_NO_ZERO_QUADRATURE" if reference_origin else "ACTUAL_NONZERO_REGISTERED_COLD_PROBE",
        order, source_order, source_anchor, global_anchor, tuple(tuple(r) for r in matrix),
        tuple(second), tuple(firsts), tuple(seconds), tuple(first_residual), tuple(second_residual),
        tuple(I(0) for _ in blocks), True)


def bind_native_common_strain_projection(transfer, *, cold_contexts, provenance, precision=96):
    transfer = validate_native_common_material_transfer_system(transfer)
    precision = _precision(precision)
    if (type(cold_contexts) is not tuple or len(cold_contexts) != 8
            or any(type(row) is not tuple or len(row) != 3 for row in cold_contexts)
            or type(provenance) is not str or not provenance.strip()):
        raise ValueError("Declare all eight authentic body/material/integral contexts and the new bankwide reduced law")
    # A cache key authenticates the COMPLETE body/material/raw content, not
    # just declared old binding IDs. An altered field obtains another key and
    # must pass all original validators below. Authentic replay need not run
    # the eight large old source validators again before checking this key.
    context_hash = _digest(cold_contexts)
    key = transfer.binding_sha256, context_hash, provenance, precision
    if key in _BIND_CACHE:
        return validate_native_common_strain_projection_system(
            _BIND_CACHE[key], transfer=transfer, cold_contexts=cold_contexts)
    blocks = tuple(_projection_block(*row, transfer, precision) for row in cold_contexts)
    if len({b.component_id for b in blocks}) != 8 or {b.source_coordinate_index for b in blocks} != set(range(6, 14)):
        raise ValueError("One global reduction must account for every source radial beam exactly once")
    blocks = tuple(sorted(blocks, key=lambda b: b.source_coordinate_index))
    order = transfer.coordinate_order[:92]
    coordinates, maps, derivatives = [F(0)]*92, [None]*92, [I(0)]*92
    J = [[I(0) for _ in range(92)] for _ in range(14)]
    for i in range(6):
        maps[i], derivatives[i], J[i][i] = i, I(1), I(1)
    cold_indices = []
    T = [[F(0) for _ in range(14)] for _ in range(92)]
    for i in range(6):
        T[i][i] = F(1)
    for b in blocks:
        T[b.global_coordinate_indices[0]][b.source_coordinate_index] = F(1)
        for j, index in enumerate(b.global_coordinate_indices):
            coordinates[index], maps[index], derivatives[index] = b.coordinates_m[j], b.source_coordinate_index, b.projection_gradient_bounds[j]
            J[b.source_coordinate_index][index] = b.projection_gradient_bounds[j]
            cold_indices.append(index)
    hot = tuple(i for i in range(6, 92) if maps[i] is None)
    if len(hot) != 30 or len(cold_indices) != 56:
        raise ValueError("The declared92 catalogue must contain56 cold and30 HOT field coordinates")
    reference_inverse = _inverse_twojet(order, transfer.source_coordinate_order, blocks,
        (I(0),)*14, (F(0),)*92, reference_origin=True, precision=precision)
    value = NativeCommonStrainProjectionSystem(transfer.binding_sha256, transfer.source_snapshot_sha256,
        transfer.reference_snapshot_sha256, FORCE_BASIS, order, transfer.source_coordinate_order,
        tuple(coordinates), blocks, tuple(sorted(cold_indices)), hot, tuple(maps), tuple(derivatives),
        tuple(tuple(row) for row in J), tuple(tuple(row) for row in T),
        reference_inverse.source_to_global_second_derivative_blocks, reference_inverse,
        context_hash, transfer.complete_original_material_state_json,
        transfer.complete_current_source_states_json, transfer.complete_current_source_memory_json,
        transfer.complete_before_work_ledgers_json, transfer.complete_unaccepted_candidate_work_ledgers_json,
        transfer.original_source_work_lineages, transfer.source_serial_reference_energy_j,
        transfer.beam_reference_energy_j, transfer.beam_energy_rounding_residual_j,
        transfer.source_contact_elapsed_time_s, transfer.source_contact_step_count, transfer.source_memory_elapsed_time_s,
        transfer.source_geometry_elapsed_time_s, transfer.material_geometry_elapsed_time_s, provenance, precision, "")
    result = _remember(value, (transfer, cold_contexts))
    _CONTEXT_TYPES[type(result), result.binding_sha256] = _content_types(cold_contexts)
    _BIND_CACHE[key] = result
    return result


def validate_native_common_strain_projection_system(value, *, transfer=None, cold_contexts=None):
    actual_transfer, actual_contexts = _registered(value, NativeCommonStrainProjectionSystem)
    validate_native_common_material_transfer_system(actual_transfer)
    types = _CONTEXT_TYPES[type(value), value.binding_sha256]
    if (_content_types(actual_contexts) != types
            or _digest(actual_contexts) != value.cold_context_binding_sha256):
        raise ValueError("An authentic body/material/integral context changed after complete projection binding")
    if transfer is not None and (type(transfer) is not type(actual_transfer) or _encoded(transfer) != _encoded(actual_transfer)):
        raise ValueError("The nonlinear reduction belongs to another complete source transfer")
    if (cold_contexts is not None and cold_contexts is not actual_contexts
            and (_content_types(cold_contexts) != types
                 or _digest(cold_contexts) != value.cold_context_binding_sha256)):
        raise ValueError("The nonlinear reduction belongs to another complete registered strain context")
    return value


def _energy_assembly_flags(*,
        separate_source_beam_energized=False, full_common_calibrated_cold_energy_energized=False,
        continuous_source_measure_energized=False):
    values = (separate_source_beam_energized, full_common_calibrated_cold_energy_energized,
              continuous_source_measure_energized)
    if any(type(v) is not bool or v for v in values):
        raise ValueError("The projected source beams, orthogonal bulk and literal atoms cannot be energized twice")
    return ENERGY_OWNERSHIP


def validate_native_common_strain_projection_energy_assembly(system, *,
        separate_source_beam_energized=False, full_common_calibrated_cold_energy_energized=False,
        continuous_source_measure_energized=False):
    validate_native_common_strain_projection_system(system)
    return _energy_assembly_flags(separate_source_beam_energized=separate_source_beam_energized,
        full_common_calibrated_cold_energy_energized=full_common_calibrated_cold_energy_energized,
        continuous_source_measure_energized=continuous_source_measure_energized)


@dataclass(frozen=True, slots=True)
class NativeCommonProjectedSourceAtom:
    point_id: str
    trace_id: str
    source_accounting_bucket: str
    original_area_m2: F
    source_force_basis: str
    sub_position_bounds_m: tuple
    over_position_bounds_m: tuple
    receiver_director_bounds: tuple
    scalar_increment_bounds_m: tuple
    scalar_free_anchor_bounds_m: tuple
    scalar_first_derivative_bounds: tuple
    scalar_second_derivative_bounds: tuple
    traction_bounds_pa: tuple
    effective_tangent_stiffness_pa_m: tuple
    potential_bounds_j: I
    potential_gradient_bounds: tuple
    potential_hessian_bounds: tuple
    world_reactions: tuple
    stored_source_energy_components_j: tuple
    original_source_state_json: str
    source_work_lineage_id: tuple
    archival_literal_world_reactions_json: str
    archival_mathematical_minus_literal_reaction_bounds: tuple
    proposed_plastic_increment_bounds_m: I
    actual_common_face_force_distribution_certified: bool = False


@dataclass(frozen=True, slots=True)
class NativeCommonProjectedSourceBeam:
    trace_id: str
    side: str
    source_coordinate_index: int
    projected_coordinate_increment_bounds_m: I
    potential_increment_bounds_j: I
    gradient_bounds_n: I
    hessian_exact_n_m: F
    reference_force_exact_n: F
    reference_energy_exact_j: F


def _source_kernel(transfer, increments, branch, precision):
    source = _TRANSFER_DEPENDENCIES[type(transfer), transfer.binding_sha256][0]
    order = transfer.source_coordinate_order
    atoms = []
    for c in source.carriers:
        b = c.archival_old_reference
        ds = increments[order.index(("radial", c.trace_id, "sub"))]
        do = increments[order.index(("radial", c.trace_id, "over"))]
        if (I(source.source_radius_m)+ds).lo <= 0 or (I(source.source_radius_m)+do).lo <= 0:
            raise ValueError("The whole interval source projection must keep positive virtual tip radii")
        vs, vo = b.sub_direction_bounds, b.over_direction_bounds
        ps = tuple(_round(p+ds*v, precision) for p, v in zip(b.sub_position_bounds_m, vs))
        po = tuple(_round(p+do*v, precision) for p, v in zip(b.over_position_bounds_m, vo))
        frames = b.receiver_director_bounds
        _, B, H = _source_jet(order, c.sub_owner, c.over_owner, c.trace_id, (ps, po), (vs, vo), frames, precision)
        eta = tuple(_round(_dot(e, tuple(do*o-ds*s for s, o in zip(vs, vo))), precision) for e in frames)
        K = c.series_stiffness_exact_pa_m[0], F(0) if branch == BRANCHES[0] else c.series_stiffness_exact_pa_m[1]
        traction = tuple(_round(I(t)+k*x, precision) for t, k, x in zip(c.literal_source_tractions_pa, K, eta))
        if traction[0].hi >= 0:
            raise ValueError("The authenticated interval reduced source leaves its compressed local branch")
        A = c.original_area_m2
        energy = _round(I(sum(c.stored_source_energy_components_j, F(0)))
            +A*sum((I(t)*x+k*x*x/2 for t, k, x in zip(c.literal_source_tractions_pa, K, eta)), I(0)), precision)
        gradient = tuple(_round(A*sum((traction[a]*B[a][i] for a in range(2)), I(0)), precision) for i in range(14))
        hessian = tuple(tuple(_round(A*sum((K[a]*B[a][i]*B[a][j]+traction[a]*H[a][i][j]
            for a in range(2)), I(0)), precision) for j in range(14)) for i in range(14))
        world = _world_reactions(A, traction, (ps, po), (vs, vo), frames, precision)
        atoms.append(NativeCommonProjectedSourceAtom(c.point_id, c.trace_id, c.source_accounting_bucket,
            A, FORCE_BASIS, ps, po, frames, eta,
            tuple(_round(I(f)+x, precision) for f, x in zip(c.stored_free_anchors_m, eta)), B, H,
            traction, K, energy, gradient, hessian, world, c.stored_source_energy_components_j,
            c.source_local_state_json, c.original_source_work_lineage_id, c.archival_literal_world_reactions_json,
            c.archival_mathematical_minus_literal_reaction_bounds, eta[1] if branch == BRANCHES[0] else I(0)))
    beams = tuple(NativeCommonProjectedSourceBeam(b.trace_id, b.side, b.source_coordinate_index,
        increments[b.source_coordinate_index], _round(b.stiffness_exact_n_m*(b.reference_displacement_m
        *increments[b.source_coordinate_index]+_square(increments[b.source_coordinate_index])/2), precision),
        _round(b.stiffness_exact_n_m*(I(b.reference_displacement_m)+increments[b.source_coordinate_index]), precision),
        b.stiffness_exact_n_m, b.reference_force_exact_n, b.reference_energy_exact_j) for b in transfer.beams)
    E = _round(I(transfer.beam_reference_energy_j)+sum((a.potential_bounds_j for a in atoms), I(0))
        +sum((b.potential_increment_bounds_j for b in beams), I(0)), precision)
    G = tuple(_round(sum((a.potential_gradient_bounds[i] for a in atoms), I(0))
        +sum((b.gradient_bounds_n for b in beams if b.source_coordinate_index == i), I(0)), precision) for i in range(14))
    H = tuple(tuple(_round(sum((a.potential_hessian_bounds[i][j] for a in atoms), I(0))
        +sum((I(b.hessian_exact_n_m) for b in beams if b.source_coordinate_index == i == j), I(0)), precision)
        for j in range(14)) for i in range(14))
    return tuple(atoms), beams, E, G, H


@dataclass(frozen=True, slots=True)
class NativeCommonStrainProjectionResponse:
    system_binding_sha256: str
    source_transfer_binding_sha256: str
    source_force_basis: str
    coordinate_order: tuple
    coordinate_increments: tuple
    branch: str
    source_coordinate_increment_bounds: tuple
    source_atom_responses: tuple
    source_beam_responses: tuple
    source_potential_bounds_j: I
    source_gradient_bounds: tuple
    source_hessian_bounds: tuple
    potential_bounds_j: I
    potential_gradient_bounds: tuple
    potential_hessian_bounds: tuple
    chainrule_curvature_hessian_bounds: tuple
    orthogonal_energy_bounds_j: I
    orthogonal_gradient_bounds: tuple
    orthogonal_hessian_bounds: tuple
    current_source_inverse_2jet: NativeCommonSourceInverseTwoJet
    source_only_inverse_pullback_gradient_bounds: tuple
    source_only_inverse_pullback_hessian_bounds: tuple
    total_inverse_pullback_gradient_bounds: tuple
    total_inverse_pullback_hessian_bounds: tuple
    source_only_inverse_first_variation_residual_bounds: tuple
    source_only_inverse_second_variation_residual_bounds: tuple
    hot_zero_coordinate_indices: tuple
    original_source_work_lineages: tuple
    complete_unaccepted_candidate_work_ledgers_json: str
    source_contact_elapsed_time_s: F
    source_geometry_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    precision: int
    binding_sha256: str
    energetic_owner: str = ENERGY_OWNERSHIP
    source_state_or_work_advanced: bool = False
    source_internal_coordinate_count: int = 0
    hot_thirty_source_coupling_exact_zero: bool = True
    source_only_inverse_hessian_certificate_includes_embedding_curvature: bool = True
    total_finite_work_identified_with_old_source_work: bool = False
    physical_source_material_handover_certified: bool = False
    owner_body_covariance_law: str = OWNER_COVARIANCE
    bank_strain_projection_invariant_under_superposed_owner_rotation: bool = True

    @property
    def scope(self):
        return SCOPE


def _global_chain(system, sourceG, sourceH, precision):
    maps, ds = system.coordinate_to_source_index, system.coordinate_source_projection_derivative_bounds
    G_R, H_R, curvature = [I(0)]*92, [[I(0) for _ in range(92)] for _ in range(92)], [[I(0) for _ in range(92)] for _ in range(92)]
    for b in system.projection_blocks:
        for i, gi in enumerate(b.global_coordinate_indices):
            G_R[gi] = b.orthogonal_gradient_bounds_n[i]
            for j, gj in enumerate(b.global_coordinate_indices):
                H_R[gi][gj] = b.orthogonal_hessian_bounds_n_m[i][j]
                curvature[gi][gj] = _round(sourceG[b.source_coordinate_index]*b.projection_hessian_bounds[i][j], precision)
    G = tuple(_round(sourceG[maps[i]]*ds[i]+G_R[i], precision) if maps[i] is not None else I(0) for i in range(92))
    H = tuple(tuple(_round(ds[i]*sourceH[maps[i]][maps[j]]*ds[j]+curvature[i][j]+H_R[i][j], precision)
        if maps[i] is not None and maps[j] is not None else I(0) for j in range(92)) for i in range(92))
    return G, H, tuple(tuple(row) for row in curvature), tuple(G_R), tuple(tuple(row) for row in H_R)


def _inverse_pullback(system, inverse, G, H, precision):
    indices, factors, second = list(range(6))+[0]*8, [I(1)]*6+[I(0)]*8, [I(0)]*14
    for b, alpha, gamma in zip(system.projection_blocks, inverse.inverse_tip_first_derivative_bounds,
                               inverse.inverse_tip_second_derivative_bounds):
        r = b.source_coordinate_index
        indices[r], factors[r], second[r] = b.global_coordinate_indices[0], alpha, gamma
    gradient = tuple(_round(factors[i]*G[indices[i]], precision) for i in range(14))
    hessian = tuple(tuple(_round(factors[i]*H[indices[i]][indices[j]]*factors[j]
        +(G[indices[i]]*second[i] if i == j else I(0)), precision) for j in range(14)) for i in range(14))
    return gradient, hessian


def evaluate_native_common_strain_projection(system, *, coordinate_increments=None,
        branch=BRANCHES[0], precision=96, separate_source_beam_energized=False,
        full_common_calibrated_cold_energy_energized=False, continuous_source_measure_energized=False):
    system = validate_native_common_strain_projection_system(system)
    _energy_assembly_flags(
        separate_source_beam_energized=separate_source_beam_energized,
        full_common_calibrated_cold_energy_energized=full_common_calibrated_cold_energy_energized,
        continuous_source_measure_energized=continuous_source_measure_energized)
    precision = _precision(precision)
    q = system.bound_coordinate_increments if coordinate_increments is None else coordinate_increments
    if type(q) is not tuple or len(q) != 92 or any(type(x) is not F for x in q) or q != system.bound_coordinate_increments or branch not in BRANCHES:
        raise ValueError("A92 query must match every complete literal registered cold integral coordinate and the declared source shear branch")
    key = system.binding_sha256, q, branch, precision
    if key in _QUERY_CACHE:
        cached = _QUERY_CACHE[key]
        actual, = _registered(cached, NativeCommonStrainProjectionResponse)
        if _encoded(actual) != _encoded(system):
            raise ValueError("The cached92 response belongs to another complete projection context")
        return cached
    transfer, _ = _DEPENDENCIES[type(system), system.binding_sha256]
    increments = [I(0)]*14
    for b in system.projection_blocks:
        increments[b.source_coordinate_index] = b.projection_increment_bounds_m
    increments = tuple(increments)
    atoms, beams, sourceE, sourceG, sourceH = _source_kernel(transfer, increments, branch, precision)
    G, H, curvature, G_R, H_R = _global_chain(system, sourceG, sourceH, precision)
    R = _round(sum((b.orthogonal_energy_bounds_j for b in system.projection_blocks), I(0)), precision)
    inverse = _inverse_twojet(system.coordinate_order, system.source_coordinate_order, system.projection_blocks,
        increments, q, reference_origin=False, precision=precision)
    # Compute the source-only part separately, retaining the same nonlinear
    # coordinate Hessian; dropping G D2s would spoil this inverse identity.
    source_globalG = tuple(_round(G[i]-G_R[i], precision) for i in range(92))
    source_globalH = tuple(tuple(_round(H[i][j]-H_R[i][j], precision) for j in range(92)) for i in range(92))
    inverseG, inverseH = _inverse_pullback(system, inverse, source_globalG, source_globalH, precision)
    totalG, totalH = _inverse_pullback(system, inverse, G, H, precision)
    first_res = tuple(_round(a-b, precision) for a, b in zip(inverseG, sourceG))
    second_res = tuple(tuple(_round(inverseH[i][j]-sourceH[i][j], precision) for j in range(14)) for i in range(14))
    if any(not _contains_zero(v) for v in first_res) or any(not _contains_zero(v) for row in second_res for v in row):
        raise ValueError("The single global inverse2jet fails source-only force and prestressed Hessian preservation")
    response = NativeCommonStrainProjectionResponse(system.binding_sha256, system.source_transfer_binding_sha256,
        FORCE_BASIS, system.coordinate_order, q, branch, increments, atoms, beams, sourceE, sourceG, sourceH,
        _round(sourceE+R, precision), G, H, curvature, R, G_R, H_R, inverse,
        inverseG, inverseH, totalG, totalH, first_res, second_res, system.hot_zero_coordinate_indices,
        system.original_source_work_lineages, system.complete_unaccepted_candidate_work_ledgers_json,
        system.source_contact_elapsed_time_s, system.source_geometry_elapsed_time_s,
        system.material_geometry_elapsed_time_s, precision, "")
    result = _remember(response, (system,))
    _QUERY_CACHE[key] = result
    return result


def validate_native_common_strain_projection_response(value, *, system=None):
    actual, = _registered(value, NativeCommonStrainProjectionResponse)
    validate_native_common_strain_projection_system(actual)
    if system is not None and (type(system) is not type(actual) or _encoded(system) != _encoded(actual)):
        raise ValueError("The92 response belongs to another authentic full strain projection context")
    return value


def native_common_strain_projection_system_to_dict(value):
    value = validate_native_common_strain_projection_system(value)
    result = _payload(value)
    result.update(format=FORMAT, scope=SCOPE, model=MODEL, **_FLAGS)
    return result


def native_common_strain_projection_system_from_dict(raw, *, transfer, cold_contexts):
    if type(raw) is not dict:
        raise ValueError("The bankwise nonlinear reduction requires its literal complete DTO")
    result = bind_native_common_strain_projection(transfer, cold_contexts=cold_contexts,
        provenance=raw.get("provenance"), precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_strain_projection_system_to_dict(result)):
        raise ValueError("Projection DTO differs from registered geometry, coordinates, source work or complete nonlinear operator replay")
    return result


native_common_strain_projection_to_dict = native_common_strain_projection_system_to_dict
native_common_strain_projection_from_dict = native_common_strain_projection_system_from_dict


def native_common_strain_projection_response_to_dict(value):
    value = validate_native_common_strain_projection_response(value)
    result = _payload(value)
    result.update(format=RESPONSE_FORMAT, scope=SCOPE, model=MODEL, **_FLAGS)
    return result


def native_common_strain_projection_response_from_dict(raw, *, system):
    if type(raw) is not dict or type(raw.get("coordinate_increments")) is not list:
        raise ValueError("The92 response requires its full literal query coordinates")
    result = evaluate_native_common_strain_projection(system,
        coordinate_increments=tuple(_read_fraction(v) for v in raw["coordinate_increments"]),
        branch=raw.get("branch"), precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_strain_projection_response_to_dict(result)):
        raise ValueError("The92 response differs from source, curvature, bulk and inverse2jet replay")
    return result

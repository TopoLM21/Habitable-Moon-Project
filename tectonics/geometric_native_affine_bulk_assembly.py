"""Actual-mask HOT affine bulk energy and shared pressure-face compliance.

Six symmetric Cartesian Ritz modes act on a source-bound complete HOT bank,
with amplitudes in metres and gradient generators E/R.  The volume integral
is exact for the constant deformation gradient: its only geometric enclosure
is the authenticated whole-mask reference-volume interval.  No midpoint area,
normal-column length, arbitrary supplied stiffness, or numerical floor enters.

This is a new stress-free HOT bulk component, not a completed mixed model.
Cold cantilever compatibility, the prestressed interface law, the coupled
force solve, all-pair clock admission and adoption remain separate obligations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F

from .geometric_native_pose import FractionInterval as I
from .geometric_native_bulk_elasticity import (
    NativeBulkElasticClosure, evaluate_native_bulk_elasticity,
    validate_native_bulk_elastic_closure, _matrix, _transpose, _multiply,
    _contraction, _determinant3, _positive_definite_factor, _factor_solve,
)


DOF_NAMES = ("xx", "yy", "zz", "xy", "xz", "yz")


def _geometry():
    # Lazy import permits independent source registration before construction.
    from . import geometric_native_affine_body as geometry
    return geometry


def _coordinates(value, name):
    if type(value) is not tuple or len(value) != 6 or any(type(x) is not F for x in value):
        raise ValueError(name + " requires six immutable exact Fraction amplitudes in metres")
    return value


def _identity():
    return tuple(tuple(F(int(i == j)) for j in range(3)) for i in range(3))


def _add(first, second):
    return tuple(tuple(a+b for a, b in zip(row, other)) for row, other in zip(first, second))


def _scale(matrix, scalar):
    return tuple(tuple(x*scalar for x in row) for row in matrix)


def _rotation(value):
    if value is None:
        return _identity()
    result = _matrix(value, "Explicit passive spatial SO(3)", size=3, width=3)
    if _multiply(_transpose(result), result) != _identity() or _determinant3(result) != 1:
        raise ValueError("Explicit spatial rotation must be an exact proper SO(3) matrix")
    return result


def _interval(value, name):
    if type(value) is not I:
        raise ValueError(name + " requires a literal outward FractionInterval")
    return value


def _square(value):
    return I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


def _trace(matrix):
    return sum((matrix[i][i] for i in range(3)), F(0))


def _tensor_action(closure, strain):
    return tuple(tuple(2*closure.shear_modulus_pa*strain[i][j]
        +(closure.lame_lambda_pa*_trace(strain) if i == j else F(0))
        for j in range(3)) for i in range(3))


@dataclass(frozen=True, slots=True)
class NativeAffineBulkAssembly:
    body: object
    closure: NativeBulkElasticClosure
    generators: tuple
    dof_names: tuple
    reference_volume_bounds_m3: I
    stiffness_per_reference_volume_n_m4: tuple
    stiffness_bounds_n_m: tuple

    @property
    def actual_whole_mask_bulk_component_integrated(self):
        return True

    @property
    def material_tensor_is_explicit_new_hot_closure(self):
        return True

    @property
    def previous_complete_three_dimensional_tensor_inherited(self):
        return False

    @property
    def arbitrary_discrete_stiffness_supplied(self):
        return False

    @property
    def source_normal_column_lengths_used_as_geometry(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def mixed_boundary_conditions_satisfied(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def assemble_native_affine_bulk(body, closure):
    """Integrate the actual whole HOT mask's six-mode reference stiffness.

    The supplied material declaration is explicitly new.  Its support and
    bank ID must match the authenticated geometric body; a source-contact
    tensor extended to an otherwise isolated HOT bank remains a new closure.
    """
    geometry = _geometry()
    body = geometry.validate_native_affine_hot_body(body)
    closure = validate_native_bulk_elastic_closure(closure)
    if (closure.bank_source_id != body.component_id
            or closure.source_support_sha256 != body.source_support_sha256):
        raise ValueError("The new bulk material declaration differs from its authentic HOT bank")
    generators = geometry.AFFINE_SYMMETRIC_GENERATORS
    if type(generators) is not tuple or len(generators) != 6:
        raise ValueError("The complete six-mode Cartesian symmetric basis is required")
    generators = tuple(_matrix(x, "Symmetric Cartesian generator", size=3, width=3)
                       for x in generators)
    if any(x != _transpose(x) for x in generators):
        raise ValueError("Affine bulk reference generators must retain symmetric strains")
    volume = _interval(body.reference_volume_bounds_m3, "Actual whole-mask volume")
    if volume.lo <= 0 or body.radius_m <= 0:
        raise ValueError("Actual whole-mask volume and source radius must be positive without floors")
    stiffness = tuple(tuple((2*closure.shear_modulus_pa*_contraction(first, second)
        +closure.lame_lambda_pa*_trace(first)*_trace(second))/body.radius_m**2
        for second in generators) for first in generators)
    _positive_definite_factor(stiffness)
    enclosed = tuple(tuple(volume*x for x in row) for row in stiffness)
    return NativeAffineBulkAssembly(body, closure, generators, DOF_NAMES, volume, stiffness, enclosed)


def validate_native_affine_bulk_assembly(assembly):
    if type(assembly) is not NativeAffineBulkAssembly:
        raise ValueError("A genuine NativeAffineBulkAssembly is required")
    replay = assemble_native_affine_bulk(assembly.body, assembly.closure)
    if assembly != replay:
        raise ValueError("Cached affine body stiffness differs from actual-mask integration")
    return assembly


@dataclass(frozen=True, slots=True)
class NativeAffineBulkResponse:
    assembly: NativeAffineBulkAssembly
    coordinates_m: tuple
    spatial_rotation: tuple
    body_deformation_gradient: tuple
    deformation_gradient: tuple
    green_strain: tuple
    second_piola_stress_pa: tuple
    first_piola_stress_pa: tuple
    energy_density_reference_j_m3: F
    volume_jacobian: F
    physical_volume_bounds_m3: I
    energy_bounds_j: I
    generalized_gradient_bounds_n: tuple
    generalized_hessian_bounds_n_m: tuple
    first_directional_derivative_bounds_j: I | None
    second_directional_derivative_bounds_j: I | None

    @property
    def actual_whole_mask_bulk_component_integrated(self):
        return True

    @property
    def whole_bank_affine_embedding_injective(self):
        # A nonsingular global affine map is injective for every point of Ω.
        return True

    @property
    def global_energy_convexity_certified(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def evaluate_native_affine_bulk(assembly, coordinates_m, *, direction_m=None,
        spatial_rotation=None):
    """Integrate full StVK energy, exact virtual forces and the exact Hessian.

    A supplied exact proper spatial rotation changes Piola representation,
    preserving energy and every modal force.  A positive affine determinant
    proves individual embedding only, not separation from another bank.
    """
    assembly = validate_native_affine_bulk_assembly(assembly)
    coordinates = _coordinates(coordinates_m, "Affine body coordinates")
    rotation = _rotation(spatial_rotation)
    radius = assembly.body.radius_m
    generators = tuple(_scale(generator, 1/radius) for generator in assembly.generators)
    body_gradient = _identity()
    for value, generator in zip(coordinates, generators):
        body_gradient = _add(body_gradient, _scale(generator, value))
    gradient = _multiply(rotation, body_gradient)
    jacobian = _determinant3(gradient)
    if jacobian <= 0:
        raise ValueError("The complete affine HOT body requires a strictly positive Jacobian")
    right = _multiply(_transpose(gradient), gradient)
    strain = tuple(tuple((right[i][j]-F(int(i == j)))/2 for j in range(3)) for i in range(3))
    elastic = evaluate_native_bulk_elasticity(assembly.closure, strain)
    piola = _multiply(gradient, elastic.stress_pa)
    variations = tuple(_multiply(rotation, generator) for generator in generators)
    gradient_density = tuple(_contraction(piola, generator) for generator in variations)
    piola_variations = []
    for variation in variations:
        strain_variation = _scale(_add(_multiply(_transpose(variation), gradient),
                                      _multiply(_transpose(gradient), variation)), F(1, 2))
        stress_variation = _tensor_action(assembly.closure, strain_variation)
        piola_variations.append(_add(_multiply(variation, elastic.stress_pa),
                                     _multiply(gradient, stress_variation)))
    hessian_density = tuple(tuple(_contraction(variation, piola_variation)
        for piola_variation in piola_variations) for variation in variations)
    if any(hessian_density[i][j] != hessian_density[j][i] for i in range(6) for j in range(i)):
        raise ValueError("Exact StVK virtual-work Hessian lost energetic symmetry")
    volume = assembly.reference_volume_bounds_m3
    first = second = None
    if direction_m is not None:
        direction = _coordinates(direction_m, "Affine directional variation")
        first = volume*sum((x*y for x, y in zip(gradient_density, direction)), F(0))
        second = volume*sum((direction[i]*hessian_density[i][j]*direction[j]
                             for i in range(6) for j in range(6)), F(0))
    return NativeAffineBulkResponse(assembly, coordinates, rotation, body_gradient, gradient,
        strain, elastic.stress_pa, piola, elastic.energy_density_j_m3, jacobian,
        volume*jacobian, volume*elastic.energy_density_j_m3,
        tuple(volume*x for x in gradient_density),
        tuple(tuple(volume*x for x in row) for row in hessian_density), first, second)


@dataclass(frozen=True, slots=True)
class NativeAffineSharedFaceCondensation:
    assembly: NativeAffineBulkAssembly
    faces: tuple
    face_ids: tuple
    face_operators_bounds_m2: tuple
    stiffness_inverse_per_reference_volume_m4_n: tuple
    unit_pressure_body_response_bounds_m_pa: tuple
    pressure_compliance_bounds_m3_pa: tuple

    @property
    def actual_whole_mask_stiffness_used(self):
        return True

    @property
    def shared_bank_cross_contact_terms_retained(self):
        return True

    @property
    def basal_foundation_included(self):
        return False

    @property
    def complete_hot_component_compliance_certified(self):
        return False

    @property
    def exact_face_operators_substituted_for_intervals(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def condense_native_affine_shared_faces(assembly, faces):
    """Bound actual source pressure-face B K^-1 B^T, including cross terms.

    B integrates normal virtual displacement against a unit face pressure.
    Thus B has units m² and the pressure compliance has units m³/Pa.  It is
    not a fabricated exact average-gap operator.  The one common physical
    volume is retained as an exact symbolic scalar interval multiplier.
    """
    assembly = validate_native_affine_bulk_assembly(assembly)
    if type(faces) is not tuple or not faces:
        raise ValueError("Every actual pressure face of the shared bank must be an immutable tuple")
    geometry = _geometry()
    faces = tuple(geometry.validate_native_affine_source_face_trace(face,
        body=assembly.body) for face in faces)
    ids = tuple(face.family_id for face in faces)
    if len(set(ids)) != len(ids):
        raise ValueError("A shared bank pressure face may only be registered once")
    operators = tuple(face.generalized_normal_force_bounds_n_per_pa for face in faces)
    if any(type(row) is not tuple or len(row) != 6 or any(type(x) is not I for x in row)
           for row in operators):
        raise ValueError("Actual source-face virtual-work operators require all six outward intervals")
    lower, diagonal = _positive_definite_factor(assembly.stiffness_per_reference_volume_n_m4)
    inverse_columns = tuple(_factor_solve(lower, diagonal,
        tuple(F(int(i == j)) for i in range(6))) for j in range(6))
    inverse = _transpose(inverse_columns)
    reciprocal_volume = I(F(1))/assembly.reference_volume_bounds_m3
    responses = tuple(tuple(sum((I(inverse[i][j])*row[j] for j in range(6)), I(0))
                           *reciprocal_volume for i in range(6)) for row in operators)
    count = len(faces)
    compliance = [[None]*count for _ in range(count)]
    for i in range(count):
        for j in range(i, count):
            coefficient = sum((operators[i][a]*I(inverse[a][b])*operators[j][b]
                               for a in range(6) for b in range(6)), I(0))*reciprocal_volume
            # The exact diagonal is a positive quadratic form.  Intersection
            # with its mathematical range is a proof, not a stiffness floor.
            if i == j:
                coefficient = I(max(F(0), coefficient.lo), coefficient.hi)
            compliance[i][j] = compliance[j][i] = coefficient
    return NativeAffineSharedFaceCondensation(assembly, faces, ids, operators, inverse,
        responses, tuple(tuple(row) for row in compliance))


def validate_native_affine_shared_face_condensation(value):
    if type(value) is not NativeAffineSharedFaceCondensation:
        raise ValueError("A source-derived affine shared-face condensation is required")
    replay = condense_native_affine_shared_faces(value.assembly, value.faces)
    if replay != value:
        raise ValueError("Cached pressure compliance differs from the source body and true face traces")
    return value


@dataclass(frozen=True, slots=True)
class NativeAffineBasalAssembly:
    body: object
    basal_source: object
    dof_names: tuple
    normal_displacement_covector_bounds: tuple
    reference_measure_area_bounds_m2: I
    foundation_stiffness_density_pa_m: F
    stiffness_bounds_n_m: tuple

    @property
    def whole_authentic_hot_basal_mask_integrated(self):
        return True

    @property
    def original_density_and_gravity_preserved(self):
        return True

    @property
    def all_six_affine_boundary_modes_retained(self):
        return True

    @property
    def independent_radial_mean_body_mode_introduced(self):
        return False

    @property
    def reference_area_winkler_extension_explicit(self):
        return True

    @property
    def physical_current_area_winkler_law_inherited(self):
        return False

    @property
    def old_scalar_foundation_energy_added_again(self):
        return False

    @property
    def full_mixed_boundary_conditions_satisfied(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def _interval_generator_quadratic(generator, direction):
    # Symmetric generators are evaluated with correlated squares on diagonals.
    return sum((I(generator[i][i])*_square(direction[i]) for i in range(3)), I(0)) + sum((
        I(2*generator[i][j])*direction[i]*direction[j]
        for i in range(3) for j in range(i)), I(0))


def assemble_native_affine_basal(body, *, basal_source=None):
    """Integrate a NEW source-coefficient Lagrangian Winkler extension.

    The actual complete material footprint supplies dΩ.  The constitutive
    measure is explicitly R²dΩ, preserving the saved density×gravity.  It is
    NOT silently replaced by
    the different current spherical base measure r_b²dΩ.  At the actual
    base, v is the unit world direction and each affine basis has normal
    displacement (r_b/R) v^T Eα v.  All six modes and their cross terms are
    retained, using exactly the same physical field as the bulk energy.

    The resulting six-mode boundary energy replaces, rather than adds to,
    the old scalar foundation component.  No independent seventh radial
    mean mode is introduced: such a nonuniform spatial displacement would
    need its own volume strain, absent from this affine component.
    """
    geometry = _geometry()
    body = geometry.validate_native_affine_hot_body(body)
    if basal_source is None:
        basal_source = geometry.bind_native_affine_basal_source(body)
    source = geometry.validate_native_affine_basal_source(basal_source, body=body)
    direction = source.whole_mask_unit_direction_bounds
    if type(direction) is not tuple or len(direction) != 3 or any(type(x) is not I for x in direction):
        raise ValueError("The actual entire basal mask requires three outward unit-direction bounds")
    base_radius = _interval(source.base_radius_bounds_m, "Actual source base radius")
    if base_radius.lo <= 0:
        raise ValueError("The actual HOT base radius must be positive")
    generators = geometry.AFFINE_SYMMETRIC_GENERATORS
    covector = tuple(base_radius/body.radius_m
        *_interval_generator_quadratic(generator, direction) for generator in generators)
    area = _interval(source.reference_measure_area_bounds_m2, "Source reference basal area")
    if area.lo <= 0 or area != source.actual_area_bounds_sr*body.radius_m**2:
        raise ValueError("The Lagrangian basal measure must retain the whole authentic mask at R²")
    density, gravity = source.density_kg_m3, source.gravity_m_s2
    if type(density) is not F or type(gravity) is not F or density <= 0 or gravity <= 0:
        raise ValueError("The original HOT density and gravity require literal positive source Fractions")
    coefficient = density*gravity
    stiffness = tuple(tuple(coefficient*area*(_square(first) if i == j else first*second)
        for j, second in enumerate(covector)) for i, first in enumerate(covector))
    return NativeAffineBasalAssembly(body, source, DOF_NAMES,
        covector, area, coefficient, stiffness)


def validate_native_affine_basal_assembly(assembly):
    if type(assembly) is not NativeAffineBasalAssembly:
        raise ValueError("A source-derived NativeAffineBasalAssembly is required")
    replay = assemble_native_affine_basal(assembly.body, basal_source=assembly.basal_source)
    if assembly != replay:
        raise ValueError("Cached basal stiffness differs from the whole mask's original coefficients")
    return assembly


@dataclass(frozen=True, slots=True)
class NativeAffineBasalResponse:
    assembly: NativeAffineBasalAssembly
    coordinates_m: tuple
    external_offset_m: F
    whole_mask_normal_displacement_bounds_m: I
    energy_bounds_j: I
    generalized_gradient_bounds_n: tuple
    generalized_hessian_bounds_n_m: tuple
    first_directional_derivative_bounds_j: I | None
    second_directional_derivative_bounds_j: I | None

    @property
    def whole_authentic_hot_basal_mask_integrated(self):
        return True

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def evaluate_native_affine_basal(assembly, coordinates_m, *, direction_m=None,
        external_offset_m=F(0)):
    """Bound the same affine field's entire basal energy and six forces.

    An optional offset is an explicitly prescribed boundary displacement,
    not an additional physical body coordinate or an uncomputed bulk mode.
    The actual HOT component uses the default zero offset.
    """
    assembly = validate_native_affine_basal_assembly(assembly)
    coordinates = _coordinates(coordinates_m, "Basal coordinates")
    if type(external_offset_m) is not F:
        raise ValueError("A prescribed external basal offset requires an exact Fraction")
    covector = assembly.normal_displacement_covector_bounds
    displacement = sum((x*y for x, y in zip(covector, coordinates)), I(external_offset_m))
    coefficient = assembly.foundation_stiffness_density_pa_m*assembly.reference_measure_area_bounds_m2
    energy = coefficient*_square(displacement)/2
    forces = tuple(coefficient*displacement*x for x in covector)
    first = second = None
    if direction_m is not None:
        direction = _coordinates(direction_m, "Basal directional variation")
        displacement_variation = sum((x*y for x, y in zip(covector, direction)), I(0))
        first = coefficient*displacement*displacement_variation
        second = coefficient*_square(displacement_variation)
    return NativeAffineBasalResponse(assembly, coordinates, external_offset_m, displacement, energy, forces,
        assembly.stiffness_bounds_n_m, first, second)


@dataclass(frozen=True, slots=True)
class NativeAffineHotComponentAssembly:
    bulk: NativeAffineBulkAssembly
    basal: NativeAffineBasalAssembly
    dof_names: tuple
    reference_stiffness_bounds_n_m: tuple

    @property
    def same_affine_field_bulk_and_basal(self):
        return True

    @property
    def actual_whole_mask_volume_and_boundary_integrated(self):
        return True

    @property
    def independent_radial_mean_body_mode_introduced(self):
        return False

    @property
    def old_scalar_foundation_energy_added_again(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def mixed_boundary_conditions_satisfied(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def assemble_native_affine_hot_component(body, closure, *, basal_source=None):
    """Assemble actual HOT bulk and its source basal law for ONE six-mode field."""
    bulk = assemble_native_affine_bulk(body, closure)
    basal = assemble_native_affine_basal(bulk.body, basal_source=basal_source)
    stiffness = tuple(tuple(a+b for a, b in zip(row, other))
        for row, other in zip(bulk.stiffness_bounds_n_m, basal.stiffness_bounds_n_m))
    return NativeAffineHotComponentAssembly(bulk, basal, DOF_NAMES, stiffness)


def validate_native_affine_hot_component_assembly(value):
    if type(value) is not NativeAffineHotComponentAssembly:
        raise ValueError("A genuine same-field HOT volume/basal assembly is required")
    validate_native_affine_bulk_assembly(value.bulk)
    validate_native_affine_basal_assembly(value.basal)
    if value.bulk.body != value.basal.body:
        raise ValueError("Volume and basal components must use one identical authentic HOT body")
    replay = assemble_native_affine_hot_component(value.bulk.body, value.bulk.closure,
                                                  basal_source=value.basal.basal_source)
    if replay != value:
        raise ValueError("Cached combined HOT stiffness differs from the same-field actual integration")
    return value


@dataclass(frozen=True, slots=True)
class NativeAffineHotComponentResponse:
    assembly: NativeAffineHotComponentAssembly
    bulk_response: NativeAffineBulkResponse
    basal_response: NativeAffineBasalResponse
    energy_bounds_j: I
    generalized_gradient_bounds_n: tuple
    generalized_hessian_bounds_n_m: tuple
    first_directional_derivative_bounds_j: I | None
    second_directional_derivative_bounds_j: I | None

    @property
    def same_affine_field_bulk_and_basal(self):
        return True

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def evaluate_native_affine_hot_component(assembly, coordinates_m, *, direction_m=None):
    """Integrate same-field HOT bulk+basal energy and every modal force/Hessian."""
    assembly = validate_native_affine_hot_component_assembly(assembly)
    bulk = evaluate_native_affine_bulk(assembly.bulk, coordinates_m, direction_m=direction_m)
    basal = evaluate_native_affine_basal(assembly.basal, coordinates_m, direction_m=direction_m)
    gradient = tuple(a+b for a, b in zip(bulk.generalized_gradient_bounds_n,
                                      basal.generalized_gradient_bounds_n))
    hessian = tuple(tuple(a+b for a, b in zip(row, other)) for row, other in zip(
        bulk.generalized_hessian_bounds_n_m, basal.generalized_hessian_bounds_n_m))
    first = second = None
    if direction_m is not None:
        first = bulk.first_directional_derivative_bounds_j+basal.first_directional_derivative_bounds_j
        second = bulk.second_directional_derivative_bounds_j+basal.second_directional_derivative_bounds_j
    return NativeAffineHotComponentResponse(assembly, bulk, basal, bulk.energy_bounds_j+basal.energy_bounds_j,
        gradient, hessian, first, second)

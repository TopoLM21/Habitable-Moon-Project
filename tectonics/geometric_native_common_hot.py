"""NEW common-reference binding of the authentic five six-mode HOT bodies.

Historical affine receipts are retained as immutable inputs.  A constant
GLOBAL Cartesian affine gradient has the same whole-volume StVK polynomial
after any volume-preserving reference transport.  Reusing that mathematical
polynomial does not reuse the changed reference geometry.  Basal directions
are freshly transported at the true material base, including all genuine
positive correction charts; the measure remains the declared R² dΩ.

These receipts establish components, not loads, mixed equilibrium, evolving
contact laws, whole-clock separation or adoption of a physical interval.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction as F
import hashlib
import json
from math import isfinite

from .geometric_native_pose import FractionInterval as I
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_joint_spherical_field import (
    bound_joint_layer_point, rotate_joint_physical_depth,
)
from .geometric_native_normal_support import (
    _encoded, load_native_normal_support_source,
    validate_native_normal_support_source,
)
from . import geometric_native_affine_body as historical
from . import geometric_native_affine_bulk_assembly as polynomial


COMMON_HOT_SCOPE = "COMMON_REFERENCE_HOT_COMPONENT_ONLY"
COMMON_HOT_PROVENANCE = (
    "NEW stress-free HOT reference transported by the authenticated common "
    "volume-preserving map; fixed GLOBAL six Cartesian affine modes; "
    "historical body and saved field remain separate immutable sources"
)
_REGISTRY = {}
_SOURCES = {}


def _payload(value):
    # The historical ElasticColumnParameters are literal source floats.
    # Preserve them as JSON numbers, rather than silently changing their type
    # or claiming that they were originally exact Fraction declarations.
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {"lo": _payload(value.lo), "hi": _payload(value.hi)}
    if is_dataclass(value):
        return {item.name: _payload(getattr(value, item.name)) for item in fields(value)}
    if type(value) is tuple:
        return [_payload(item) for item in value]
    if type(value) is float and isfinite(value):
        return value
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("Common HOT receipts require immutable exact data and preserved finite source floats")


def _geometry():
    from . import geometric_native_common_reference as geometry
    return geometry


def _seal(value):
    return hashlib.sha256(_encoded({item.name: _payload(getattr(value, item.name))
        for item in fields(value) if item.name != "binding_sha256"})).hexdigest()


def _remember(value):
    result = replace(value, binding_sha256=_seal(value))
    _REGISTRY[(type(result), result.binding_sha256)] = result
    return result


def _registered(value, kind):
    if (type(value) is not kind
            or _REGISTRY.get((kind, value.binding_sha256)) != value
            or value.binding_sha256 != _seal(value)):
        raise ValueError("An altered common HOT receipt lacks authenticated source replay")
    return value


class _ComponentScope:
    @property
    def scope(self):
        return COMMON_HOT_SCOPE

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False

    @property
    def material_admission_authorized(self):
        return False

    @property
    def whole_clock_all_pair_admission_certified(self):
        return False

    @property
    def independent_radial_mean_body_mode_introduced(self):
        return False

    @property
    def old_scalar_foundation_energy_added_again(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCommonHotBody(_ComponentScope):
    original_body: historical.NativeAffineHotBody
    common_bank: object
    provenance: str
    binding_sha256: str

    @property
    def component_id(self):
        return self.original_body.component_id

    @property
    def radius_m(self):
        return self.original_body.radius_m

    @property
    def reference_volume_bounds_m3(self):
        return self.common_bank.reference_volume_bounds_m3

    @property
    def reference_geometry_changed(self):
        return (self.original_body.angular_rate_per_physical_volume_depth_m
                != self.common_bank.angular_rate_per_physical_volume_depth_m)

    @property
    def reference_snapshot_sha256(self):
        return self.common_bank.reference_snapshot_sha256

    @property
    def fixed_global_symmetric_generators(self):
        return self.original_body.symmetric_generators

    @property
    def coordinate_order(self):
        return self.original_body.coordinate_order


def _check_body_sources(original, bank):
    historical.validate_native_affine_hot_body(original)
    _geometry().validate_native_common_reference_bank(bank)
    names = ("component_id", "material_id", "original_parent_id", "owner",
        "material_kind", "radius_m", "areal_material_thickness_m",
        "roof_radial_displacement_m", "actual_area_bounds_sr",
        "reference_volume_bounds_m3", "complete_mask_source_json",
        "saved_field_json", "source_snapshot_sha256", "source_support_sha256",
        "source_epoch_sha256", "source_packet_raw_sha256",
        "saved_field_declaration_precision", "paired_corrections_source_json")
    for name in names:
        if getattr(original, name) != getattr(bank, name):
            raise ValueError("Common HOT binding changed authentic source " + name)
    if (original.material_kind != "hot_birth"
            or original.angular_rate_per_physical_volume_depth_m
                != bank.historical_angular_rate_per_physical_volume_depth_m
            or original.complete_birth_source_json != bank.complete_parent_allocation_or_birth_json):
        raise ValueError("Common HOT binding changed the immutable historical body")


def bind_native_common_hot_body(component_id, *, reference=None, source=None,
        provenance=COMMON_HOT_PROVENANCE):
    source = load_native_normal_support_source() if source is None else validate_native_normal_support_source(source)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Common HOT binding requires explicit NEW reference provenance")
    if reference is None:
        reference = _geometry().bind_native_common_reference(source=source)
    bank = _geometry().bind_native_common_reference_bank(reference, component_id)
    original = historical.bind_native_affine_hot_body(component_id, source=source)
    _check_body_sources(original, bank)
    body = _remember(NativeCommonHotBody(original, bank, provenance, ""))
    _SOURCES[body.binding_sha256] = source
    return body


def validate_native_common_hot_body(body):
    body = _registered(body, NativeCommonHotBody)
    _check_body_sources(body.original_body, body.common_bank)
    return body


@dataclass(frozen=True, slots=True)
class NativeCommonHotPointBounds(_ComponentScope):
    body: NativeCommonHotBody
    reference_point_bounds_m: tuple
    affine_point_bounds_m: tuple
    incremental_gradient: tuple
    incremental_determinant: F
    physical_volume_bounds_m3: I
    translation_m: tuple

    @property
    def material_membership_authenticated(self):
        return False


def bound_native_common_hot_body_point(body, direction_bounds, material_depth_bounds_m,
        coordinates_m, *, translation_m=(F(0),)*3, precision=160, order=28):
    body = validate_native_common_hot_body(body)
    historical._budget(precision, order)
    historical._interval_vector(direction_bounds)
    translation = historical._fraction_tuple(translation_m, 3, "Common HOT translation")
    gradient = historical.native_affine_body_gradient(body.original_body, coordinates_m)
    reference = bound_joint_layer_point(body.common_bank.layer, direction_bounds,
        material_depth_bounds_m, precision=precision, order=order)
    image = tuple(x+y for x, y in zip(historical._matvec(gradient, reference), translation))
    determinant = historical._determinant(gradient)
    return NativeCommonHotPointBounds(body, reference, image, gradient, determinant,
        body.reference_volume_bounds_m3*determinant, translation)


@dataclass(frozen=True, slots=True)
class NativeCommonHotBulkAssembly(_ComponentScope):
    body: NativeCommonHotBody
    coefficient_kernel: polynomial.NativeAffineBulkAssembly
    binding_sha256: str

    @property
    def closure(self):
        return self.coefficient_kernel.closure

    @property
    def stiffness_bounds_n_m(self):
        return self.coefficient_kernel.stiffness_bounds_n_m

    @property
    def stiffness_per_reference_volume_n_m4(self):
        return self.coefficient_kernel.stiffness_per_reference_volume_n_m4

    @property
    def fixed_global_bulk_polynomial_exactly_invariant(self):
        return True


def assemble_native_common_hot_bulk(body, closure):
    body = validate_native_common_hot_body(body)
    kernel = polynomial.assemble_native_affine_bulk(body.original_body, closure)
    # The common binding transports the existing explicit HOT declaration;
    # it does not authorize another modulus under the same support hash.
    declared = _encoded(_payload(kernel.closure.source_columns))
    source = _SOURCES[body.binding_sha256]
    authentic = tuple(_encoded(json.loads(family.complete_carrier_source_json)
        ["declaration"]["parameters"]["columns"]) for family in source.families)
    if declared not in authentic:
        raise ValueError("Common HOT bulk changed the preserved literal source columns")
    if kernel.reference_volume_bounds_m3 != body.reference_volume_bounds_m3:
        raise ValueError("Common transport must preserve the authentic whole HOT volume")
    return _remember(NativeCommonHotBulkAssembly(body, kernel, ""))


def validate_native_common_hot_bulk_assembly(assembly):
    assembly = _registered(assembly, NativeCommonHotBulkAssembly)
    validate_native_common_hot_body(assembly.body)
    polynomial.validate_native_affine_bulk_assembly(assembly.coefficient_kernel)
    if assembly.coefficient_kernel.body != assembly.body.original_body:
        raise ValueError("Common bulk polynomial substituted another authentic body")
    return assembly


@dataclass(frozen=True, slots=True)
class NativeCommonHotBulkResponse(_ComponentScope):
    assembly: NativeCommonHotBulkAssembly
    constant_gradient_polynomial_response: polynomial.NativeAffineBulkResponse

    @property
    def energy_bounds_j(self):
        return self.constant_gradient_polynomial_response.energy_bounds_j

    @property
    def generalized_gradient_bounds_n(self):
        return self.constant_gradient_polynomial_response.generalized_gradient_bounds_n

    @property
    def generalized_hessian_bounds_n_m(self):
        return self.constant_gradient_polynomial_response.generalized_hessian_bounds_n_m

    @property
    def first_directional_derivative_bounds_j(self):
        return self.constant_gradient_polynomial_response.first_directional_derivative_bounds_j

    @property
    def second_directional_derivative_bounds_j(self):
        return self.constant_gradient_polynomial_response.second_directional_derivative_bounds_j


def evaluate_native_common_hot_bulk(assembly, coordinates_m, *, direction_m=None):
    assembly = validate_native_common_hot_bulk_assembly(assembly)
    response = polynomial.evaluate_native_affine_bulk(assembly.coefficient_kernel,
        coordinates_m, direction_m=direction_m)
    return NativeCommonHotBulkResponse(assembly, response)


@dataclass(frozen=True, slots=True)
class NativeCommonHotBasalSource(_ComponentScope):
    body: NativeCommonHotBody
    whole_mask_unit_direction_bounds: tuple
    containing_chart_unit_direction_bounds: tuple
    base_radius_bounds_m: I
    physical_volume_depth_bounds_m: I
    density_kg_m3: F
    gravity_m_s2: F
    actual_area_bounds_sr: I
    reference_measure_area_bounds_m2: I
    source_foundation_json: str
    positive_correction_containing_chart_count: int
    precision: int
    order: int
    binding_sha256: str

    @property
    def containing_charts_assigned_as_material(self):
        return False

    @property
    def complete_actual_mask_contained(self):
        return True

    @property
    def source_reference_measure_is_R_squared_dOmega(self):
        return True


def bind_native_common_hot_basal_source(body, *, precision=160, order=28):
    body = validate_native_common_hot_body(body)
    historical._budget(precision, order)
    original = body.original_body
    source = _SOURCES[body.binding_sha256]
    support = json.loads(dict(source.complete_source_dto_json)["full_reference_face_birth_support"])
    foundations = tuple(item for item in support["foundations"]
        if item["declaration"]["component_id"] == body.component_id)
    if len(foundations) != 1:
        raise ValueError("Common HOT base lost its original foundation source")
    foundation = foundations[0]
    parameters = foundation["declaration"]["parameters"]
    rho, gravity = (F.from_float(parameters[key]) for key in ("density_kg_m3", "gravity_m_s2"))
    if rho <= 0 or gravity <= 0:
        raise ValueError("Authentic source density and gravity must remain positive")
    charts = list(json.loads(original.complete_mask_source_json)["proxy_geometries"])
    positive = 0
    for encoded in original.paired_corrections_source_json:
        correction = json.loads(encoded)
        if correction["positive_component_id"] == body.component_id:
            charts.append(correction["containing_chart"])
            positive += 1
    if not charts:
        raise ValueError("The authentic complete HOT footprint has no containing chart")
    R = body.radius_m
    cube = (R+original.roof_radial_displacement_m)**3-3*R**2*original.areal_material_thickness_m
    radius = interval_cbrt(I(cube), precision)
    tau = I((R**3-cube)/(3*R**2))
    cones = []
    for chart in charts:
        vertices = historical._normalized_native_vertex_bounds(chart, precision)
        rotated = tuple(rotate_joint_physical_depth(vertex,
            body.common_bank.angular_rate_per_physical_volume_depth_m, tau,
            precision=precision, order=order) for vertex in vertices)
        cones.append(historical._unit_cone_bounds(rotated, precision))
    return _remember(NativeCommonHotBasalSource(body, historical._hull(cones), tuple(cones),
        radius, tau, rho, gravity, original.actual_area_bounds_sr,
        original.actual_area_bounds_sr*R**2, _encoded(foundation).decode(),
        positive, precision, order, ""))


def validate_native_common_hot_basal_source(value, *, body=None):
    value = _registered(value, NativeCommonHotBasalSource)
    validate_native_common_hot_body(value.body)
    if body is not None and validate_native_common_hot_body(body) != value.body:
        raise ValueError("Common basal source substituted another authentic HOT body")
    return value


@dataclass(frozen=True, slots=True)
class NativeCommonHotBasalAssembly(_ComponentScope):
    body: NativeCommonHotBody
    basal_source: NativeCommonHotBasalSource
    normal_displacement_covector_bounds: tuple
    reference_measure_area_bounds_m2: I
    foundation_stiffness_density_pa_m: F
    stiffness_bounds_n_m: tuple
    binding_sha256: str


def assemble_native_common_hot_basal(body, *, basal_source=None):
    body = validate_native_common_hot_body(body)
    basal_source = bind_native_common_hot_basal_source(body) if basal_source is None else basal_source
    source = validate_native_common_hot_basal_source(basal_source, body=body)
    covector = tuple(source.base_radius_bounds_m/body.radius_m
        *polynomial._interval_generator_quadratic(generator, source.whole_mask_unit_direction_bounds)
        for generator in body.fixed_global_symmetric_generators)
    area = source.reference_measure_area_bounds_m2
    coefficient = source.density_kg_m3*source.gravity_m_s2
    stiffness = tuple(tuple(coefficient*area*(polynomial._square(first) if i == j else first*second)
        for j, second in enumerate(covector)) for i, first in enumerate(covector))
    return _remember(NativeCommonHotBasalAssembly(body, source, covector, area,
        coefficient, stiffness, ""))


def validate_native_common_hot_basal_assembly(assembly):
    assembly = _registered(assembly, NativeCommonHotBasalAssembly)
    validate_native_common_hot_basal_source(assembly.basal_source, body=assembly.body)
    return assembly


@dataclass(frozen=True, slots=True)
class NativeCommonHotBasalResponse(_ComponentScope):
    assembly: NativeCommonHotBasalAssembly
    coordinates_m: tuple
    whole_mask_normal_displacement_bounds_m: I
    energy_bounds_j: I
    generalized_gradient_bounds_n: tuple
    generalized_hessian_bounds_n_m: tuple
    first_directional_derivative_bounds_j: I | None
    second_directional_derivative_bounds_j: I | None


def evaluate_native_common_hot_basal(assembly, coordinates_m, *, direction_m=None):
    assembly = validate_native_common_hot_basal_assembly(assembly)
    coordinates = polynomial._coordinates(coordinates_m, "Common HOT basal coordinates")
    covector = assembly.normal_displacement_covector_bounds
    displacement = sum((x*y for x, y in zip(covector, coordinates)), I(0))
    coefficient = assembly.foundation_stiffness_density_pa_m*assembly.reference_measure_area_bounds_m2
    energy = coefficient*polynomial._square(displacement)/2
    forces = tuple(coefficient*displacement*x for x in covector)
    first = second = None
    if direction_m is not None:
        direction = polynomial._coordinates(direction_m, "Common HOT basal variation")
        variation = sum((x*y for x, y in zip(covector, direction)), I(0))
        first, second = coefficient*displacement*variation, coefficient*polynomial._square(variation)
    return NativeCommonHotBasalResponse(assembly, coordinates, displacement, energy,
        forces, assembly.stiffness_bounds_n_m, first, second)


@dataclass(frozen=True, slots=True)
class NativeCommonHotComponentAssembly(_ComponentScope):
    bulk: NativeCommonHotBulkAssembly
    basal: NativeCommonHotBasalAssembly
    reference_stiffness_bounds_n_m: tuple
    binding_sha256: str

    @property
    def same_affine_field_bulk_and_basal(self):
        return True


def assemble_native_common_hot_component(body, closure, *, basal_source=None):
    bulk = assemble_native_common_hot_bulk(body, closure)
    basal = assemble_native_common_hot_basal(body, basal_source=basal_source)
    stiffness = tuple(tuple(x+y for x, y in zip(row, other))
        for row, other in zip(bulk.stiffness_bounds_n_m, basal.stiffness_bounds_n_m))
    return _remember(NativeCommonHotComponentAssembly(bulk, basal, stiffness, ""))


def validate_native_common_hot_component_assembly(assembly):
    assembly = _registered(assembly, NativeCommonHotComponentAssembly)
    validate_native_common_hot_bulk_assembly(assembly.bulk)
    validate_native_common_hot_basal_assembly(assembly.basal)
    if assembly.bulk.body != assembly.basal.body:
        raise ValueError("Common bulk and basal must retain one identical affine field")
    return assembly


@dataclass(frozen=True, slots=True)
class NativeCommonHotComponentResponse(_ComponentScope):
    assembly: NativeCommonHotComponentAssembly
    bulk_response: NativeCommonHotBulkResponse
    basal_response: NativeCommonHotBasalResponse
    energy_bounds_j: I
    generalized_gradient_bounds_n: tuple
    generalized_hessian_bounds_n_m: tuple
    first_directional_derivative_bounds_j: I | None
    second_directional_derivative_bounds_j: I | None


def evaluate_native_common_hot_component(assembly, coordinates_m, *, direction_m=None):
    assembly = validate_native_common_hot_component_assembly(assembly)
    bulk = evaluate_native_common_hot_bulk(assembly.bulk, coordinates_m, direction_m=direction_m)
    basal = evaluate_native_common_hot_basal(assembly.basal, coordinates_m, direction_m=direction_m)
    gradient = tuple(x+y for x, y in zip(bulk.generalized_gradient_bounds_n, basal.generalized_gradient_bounds_n))
    hessian = tuple(tuple(x+y for x, y in zip(row, other))
        for row, other in zip(bulk.generalized_hessian_bounds_n_m, basal.generalized_hessian_bounds_n_m))
    first = second = None
    if direction_m is not None:
        first = bulk.first_directional_derivative_bounds_j+basal.first_directional_derivative_bounds_j
        second = bulk.second_directional_derivative_bounds_j+basal.second_directional_derivative_bounds_j
    return NativeCommonHotComponentResponse(assembly, bulk, basal,
        bulk.energy_bounds_j+basal.energy_bounds_j, gradient, hessian, first, second)


@dataclass(frozen=True, slots=True)
class NativeCommonHotSourceFaceTrace(_ComponentScope):
    body: NativeCommonHotBody
    original_face: historical.NativeAffineSourceFaceTrace
    common_rate_literal: tuple
    physical_face_embedding_unchanged: bool
    binding_sha256: str

    @property
    def generalized_normal_force_bounds_n_per_pa(self):
        return self.original_face.generalized_normal_force_bounds_n_per_pa


def bind_native_common_hot_source_face_trace(body, family_id, bank, *,
        original_face=None, depth_cells=64, precision=160, order=28):
    body = validate_native_common_hot_body(body)
    source = _SOURCES[body.binding_sha256]
    family = tuple(item for item in source.families if item.family_id == family_id)
    if len(family) != 1:
        raise ValueError("Common HOT face needs its authentic complete source family")
    A = body.common_bank.angular_rate_per_physical_volume_depth_m
    if (family[0].shared_depth_axis_per_m != A
            or body.original_body.angular_rate_per_physical_volume_depth_m != A):
        raise ValueError("A changed historical face requires fresh common-reference integration")
    if original_face is None:
        original_face = historical.bound_native_affine_source_face_trace(family_id, bank,
            source=source, depth_cells=depth_cells, precision=precision, order=order)
    face = historical.validate_native_affine_source_face_trace(original_face,
        body=body.original_body, source=source)
    if face.family_id != family_id or face.bank != bank:
        raise ValueError("Common HOT face substituted another bank or source carrier")
    return _remember(NativeCommonHotSourceFaceTrace(body, face, A, True, ""))


def validate_native_common_hot_source_face_trace(value, *, body=None):
    value = _registered(value, NativeCommonHotSourceFaceTrace)
    validate_native_common_hot_body(value.body)
    historical.validate_native_affine_source_face_trace(value.original_face,
        body=value.body.original_body)
    if body is not None and validate_native_common_hot_body(body) != value.body:
        raise ValueError("Common HOT trace changed its actual bank")
    return value


def native_common_hot_to_dict(value):
    kinds = (NativeCommonHotBody, NativeCommonHotBulkAssembly, NativeCommonHotBasalSource,
        NativeCommonHotBasalAssembly, NativeCommonHotComponentAssembly, NativeCommonHotSourceFaceTrace)
    if type(value) not in kinds:
        raise ValueError("Only an authenticated common HOT receipt can be exported")
    _registered(value, type(value))
    return _payload(value)

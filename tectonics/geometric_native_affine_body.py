"""Source-bound affine Ritz fields on complete actual HOT material bodies.

The domain remains the source implicit mask, including curved corrections.
Its actual volume, rather than a normal-column length or a proxy area, is
integrated.  The saved joint placement is explicitly chosen as a new
stress-free bulk reference.  This module certifies geometry of individual
affine images and source activation traces, not boundary conditions, a
continuum minimizer, force equilibrium or all-pair material admission.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from math import factorial

from .geometric_native_pose import FractionInterval as I, _rounded_interval, interval_sqrt
from .geometric_native_spherical_solid import _cross, interval_cbrt
from .geometric_native_joint_spherical_field import (
    JointSphericalLayerMap, bound_joint_layer_point, rotate_joint_physical_depth,
)
from .geometric_native_normal_support import (
    NativeNormalSupportSource, _encoded, _payload, _read_interval,
    load_native_normal_support_source, validate_native_normal_support_source,
)


AFFINE_SYMMETRIC_GENERATORS = (
    ((F(1),F(0),F(0)),(F(0),F(0),F(0)),(F(0),F(0),F(0))),
    ((F(0),F(0),F(0)),(F(0),F(1),F(0)),(F(0),F(0),F(0))),
    ((F(0),F(0),F(0)),(F(0),F(0),F(0)),(F(0),F(0),F(1))),
    ((F(0),F(1),F(0)),(F(1),F(0),F(0)),(F(0),F(0),F(0))),
    ((F(0),F(0),F(1)),(F(0),F(0),F(0)),(F(1),F(0),F(0))),
    ((F(0),F(0),F(0)),(F(0),F(0),F(1)),(F(0),F(1),F(0))),
)
AFFINE_COORDINATE_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")
STRESS_FREE_REFERENCE_PROVENANCE = (
    "New explicit HOT bulk reference: the authenticated saved128 joint placement "
    "is stress free for this affine Ritz increment; no original bulk prestress is inferred"
)
_BODY_AUTHENTICATED = {}
_FACE_AUTHENTICATED = {}
_BASAL_AUTHENTICATED = {}
_BODY_SOURCES = {}


def _seal(value):
    body = {f.name:_payload(getattr(value,f.name)) for f in fields(value)
            if f.name != "geometry_source_sha256"}
    return hashlib.sha256(_encoded(body)).hexdigest()


def _remember(value, registry):
    result = replace(value, geometry_source_sha256=_seal(value))
    registry[result.geometry_source_sha256] = tuple(getattr(result,f.name) for f in fields(result))
    return result


def _validate_registered(value, kind, registry):
    if type(value) is not kind:
        raise ValueError("A typed source-bound affine geometry receipt is required")
    key = getattr(value, "geometry_source_sha256")
    if (registry.get(key) != tuple(getattr(value,f.name) for f in fields(value))
            or key != _seal(value)):
        raise ValueError("An altered or re-sealed affine receipt lacks authenticated source replay")
    return value


def _fraction_tuple(value, count, name):
    if type(value) is not tuple or len(value) != count or any(type(x) is not F for x in value):
        raise ValueError(name + " requires immutable literal Fraction coordinates")
    return value


def _interval_vector(value):
    if type(value) is not tuple or len(value) != 3 or any(type(x) is not I for x in value):
        raise ValueError("A geometric direction requires three literal FractionIntervals")
    if not _dot(value,value).lo <= 1 <= _dot(value,value).hi:
        raise ValueError("The direction enclosure cannot contain a unit material ray")
    return value


def _budget(precision, order):
    if type(precision) is not int or not 64 <= precision <= 512:
        raise ValueError("Affine geometry precision must be a literal integer from64 to512")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Affine geometry requires an explicit Taylor order from8 to64")


def _matvec(matrix, vector):
    return tuple(sum((x*y for x,y in zip(row,vector)),F(0)) for row in matrix)


def _dot(first, second):
    return sum((x*y for x,y in zip(first,second)),F(0))


def _determinant(matrix):
    return _dot(matrix[0], _cross(matrix[1],matrix[2]))


def _hull(vectors):
    return tuple(I(min(v[i].lo for v in vectors),max(v[i].hi for v in vectors)) for i in range(3))


@dataclass(frozen=True, slots=True)
class NativeAffineHotBody:
    component_id: str
    material_id: str
    original_parent_id: str
    owner: int
    material_kind: str
    radius_m: F
    areal_material_thickness_m: F
    roof_radial_displacement_m: F
    angular_rate_per_physical_volume_depth_m: tuple
    actual_area_bounds_sr: I
    reference_volume_bounds_m3: I
    complete_mask_source_json: str
    complete_birth_source_json: str
    paired_corrections_source_json: tuple
    saved_field_json: str
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_packet_raw_sha256: str
    saved_field_declaration_precision: int
    reference_origin_m: tuple
    symmetric_generators: tuple
    coordinate_order: tuple
    stressfree_reference_provenance: str
    provenance: str
    geometry_source_sha256: str

    @property
    def layer(self):
        return JointSphericalLayerMap(self.radius_m,self.roof_radial_displacement_m,
            self.areal_material_thickness_m,self.angular_rate_per_physical_volume_depth_m,
            self.stressfree_reference_provenance)

    @property
    def complete_implicit_material_mask_retained(self):
        return True

    @property
    def all_proxy_polygons_promoted_to_material(self):
        return False

    @property
    def whole_affine_field_defined_on_actual_body(self):
        return True

    @property
    def cold_clamp_boundary_conditions_certified(self):
        return False

    @property
    def complete_combined_strain_energy_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bind_native_affine_hot_body(component_id, *, source=None,
        provenance="Explicit affine HOT Ritz field on the complete source implicit mask"):
    source = load_native_normal_support_source() if source is None else validate_native_normal_support_source(source)
    if type(component_id) is not str or type(provenance) is not str or not provenance.strip():
        raise ValueError("An affine bank requires literal source identity and explicit provenance")
    matches = tuple(b for b in source.banks if b.component_id == component_id)
    if len(matches) != 1 or matches[0].material_kind != "hot_birth":
        raise ValueError("Whole affine HOT fields cannot replace a cold bank's continuous source clamp")
    bank = matches[0]
    dto = json.loads(dict(source.complete_source_dto_json)["before"])
    epoch = dto["snapshot"]["current_material_epoch"]
    component = json.loads(bank.complete_mask_source_json)
    material = epoch["source_material_state"]
    births = tuple(v for v in material["endpoint_vacancies"] if v["newborn"] is not None
        and v["newborn"]["vacancy_domain_id"] == component_id)
    if len(births) != 1:
        raise ValueError("The affine HOT body lost its actual source birth")
    birth = births[0]
    if (birth["newborn"]["material_id"] != bank.material_id
            or birth["newborn"]["owner"] != bank.owner
            or birth["fixed_original_parent_id"] != bank.original_parent_id):
        raise ValueError("The affine HOT body changed its original material allocation")
    area = _read_interval(component["area_bounds_sr"])
    if area.lo <= 0 or area != _read_interval(birth["area_bounds_sr"]):
        raise ValueError("The complete actual HOT mask must preserve its strictly positive source measure")
    reference = json.loads(source.saved_field_json)
    groups = tuple(g for g in reference["groups"] if component_id in g["component_ids"])
    if len(groups) != 1:
        raise ValueError("An actual HOT bank lacks its unique saved joint group")
    group = groups[0]
    axes = tuple(f.shared_depth_axis_per_m for f in source.families if f.family_id in group["family_ids"])
    if axes:
        A = axes[0]
        if any(a != A for a in axes):
            raise ValueError("One actual bank cannot acquire independent contact depth fields")
    else:
        aliases = tuple(_read_interval(x) for x in group["angular_rate_per_physical_volume_depth_m"])
        if any(x.width or x.lo for x in aliases):
            raise ValueError("A nonzero unbound saved angular declaration cannot be reconstructed by its midpoint")
        A = (F(0),)*3
    if any(not _read_interval(alias).lo <= x <= _read_interval(alias).hi
           for x,alias in zip(A,group["angular_rate_per_physical_volume_depth_m"])):
        raise ValueError("The actual angular field differs from its saved128 declaration")
    corrections = tuple(_encoded(c).decode() for c in epoch["paired_corrections"]
        if component_id in (c["negative_component_id"],c["positive_component_id"]))
    body = NativeAffineHotBody(component_id,bank.material_id,bank.original_parent_id,bank.owner,
        bank.material_kind,source.radius_m,bank.areal_material_thickness_m,
        bank.roof_radial_displacement_m,A,area,area*source.radius_m**2*bank.areal_material_thickness_m,
        bank.complete_mask_source_json,_encoded(birth).decode(),corrections,source.saved_field_json,
        source.source_snapshot_sha256,source.source_support_sha256,source.source_epoch_sha256,
        source.curved_packet_raw_sha256,source.saved_field_declaration_precision,(F(0),)*3,
        AFFINE_SYMMETRIC_GENERATORS,AFFINE_COORDINATE_ORDER,STRESS_FREE_REFERENCE_PROVENANCE,provenance,"")
    body.layer
    body = _remember(body,_BODY_AUTHENTICATED)
    _BODY_SOURCES[body.geometry_source_sha256] = source
    return body


def validate_native_affine_hot_body(body):
    return _validate_registered(body,NativeAffineHotBody,_BODY_AUTHENTICATED)


def native_affine_body_gradient(body, amplitudes_m):
    body = validate_native_affine_hot_body(body)
    amplitudes = _fraction_tuple(amplitudes_m,6,"Affine amplitudes in metres")
    gradient = tuple(tuple(F(i == j)+sum((a*E[i][j]/body.radius_m
        for a,E in zip(amplitudes,body.symmetric_generators)),F(0)) for j in range(3)) for i in range(3))
    if _determinant(gradient) <= 0:
        raise ValueError("An actual affine body image requires a positive exact determinant")
    return gradient


@dataclass(frozen=True, slots=True)
class NativeAffineBodyPointBounds:
    body: NativeAffineHotBody
    reference_point_bounds_m: tuple
    affine_point_bounds_m: tuple
    incremental_gradient: tuple
    incremental_determinant: F
    reference_volume_bounds_m3: I
    physical_volume_bounds_m3: I
    translation_m: tuple

    @property
    def material_membership_authenticated(self):
        return False

    @property
    def individual_affine_body_injectivity_certified(self):
        return True

    @property
    def density_pushforward_factor(self):
        return 1/self.incremental_determinant

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_affine_body_point(body,direction_bounds,material_depth_bounds_m,amplitudes_m,
        *,translation_m=(F(0),F(0),F(0)),precision=160,order=28):
    body = validate_native_affine_hot_body(body)
    _budget(precision,order)
    _interval_vector(direction_bounds)
    translation = _fraction_tuple(translation_m,3,"Literal affine translation")
    gradient = native_affine_body_gradient(body,amplitudes_m)
    reference = bound_joint_layer_point(body.layer,direction_bounds,material_depth_bounds_m,
        precision=precision,order=order)
    image = tuple(_rounded_interval(v+t,precision) for v,t in zip(_matvec(gradient,reference),translation))
    determinant = _determinant(gradient)
    return NativeAffineBodyPointBounds(body,reference,image,gradient,determinant,
        body.reference_volume_bounds_m3,body.reference_volume_bounds_m3*determinant,translation)


@dataclass(frozen=True, slots=True)
class NativeAffineReferenceGradientBounds:
    body: NativeAffineHotBody
    reference_gradient_bounds: tuple
    rotation_matrix_bounds: tuple
    radius_bounds_m: I
    physical_volume_depth_bounds_m: I
    determinant_exact: F
    signed_volume_jacobian_exact: F

    @property
    def F0(self):
        return self.reference_gradient_bounds

    @property
    def material_membership_authenticated(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_affine_reference_gradient(body,direction_bounds,material_depth_bounds_m,
        *,precision=160,order=28):
    body = validate_native_affine_hot_body(body)
    _budget(precision,order)
    u = _interval_vector(direction_bounds)
    z = material_depth_bounds_m
    if type(z) is not I or not 0 <= z.lo <= z.hi <= body.areal_material_thickness_m:
        raise ValueError("The reference differential must retain an actual source depth interval")
    R = body.radius_m
    cubes = I((R+body.roof_radial_displacement_m)**3)-3*R**2*z
    r = interval_cbrt(cubes,precision)
    tau = (I(R**3)-cubes)/(3*R**2)
    A = body.angular_rate_per_physical_volume_depth_m
    columns = tuple(rotate_joint_physical_depth(tuple(I(int(i == j)) for i in range(3)),
        A,tau,precision=precision,order=order) for j in range(3))
    Q = tuple(tuple(columns[j][i] for j in range(3)) for i in range(3))
    shear = _cross(A,u)
    C = tuple(tuple(I(int(i == j))-cubes/R**2*shear[i]*u[j] for j in range(3)) for i in range(3))
    gradient = tuple(tuple(_rounded_interval(sum((Q[i][k]*C[k][j] for k in range(3)),I(0)),precision)
        for j in range(3)) for i in range(3))
    # S is SO(3), and u dot(A cross u)=0, so the rank-one determinant is1.
    return NativeAffineReferenceGradientBounds(body,gradient,Q,r,tau,F(1),-R**2)


@dataclass(frozen=True, slots=True)
class NativeAffineSourceFaceTrace:
    family_id: str
    component_id: str
    bank: str
    gap_sign: int
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_packet_raw_sha256: str
    saved_field_declaration_precision: int
    full_angle_bounds_radians: I
    native_angle_end_bounds_radians: I
    known_angle_core_bounds_radians: I
    uncertain_angle_end_bounds_radians: I
    common_depth_below_shared_top_bounds_m: I
    common_reference_top_radius_m: F
    common_physical_radius_cubed_bounds_m3: I
    source_preload_gap_m: F
    coordinate_order: tuple
    generalized_normal_force_bounds_n_per_pa: tuple
    depth_cell_count: int
    precision: int
    order: int
    geometry_source_sha256: str

    @property
    def whole_source_activation_face_integrated(self):
        return True

    @property
    def uncertain_native_endpoint_retained(self):
        return True

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


@lru_cache(maxsize=32)
def _integrate_source_face_trace(source,family_id,depth_cells,precision,order):
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    family = next(f for f in source.families if f.family_id == family_id)
    banks = {b.component_id:b for b in source.banks}
    donor,receiver = banks[family.donor_component_id],banks[family.receiver_component_id]
    R = source.radius_m
    top = min(R+donor.roof_radial_displacement_m,R+receiver.roof_radial_displacement_m)
    bottom_cube = max((R+b.roof_radial_displacement_m)**3-3*R**2*b.areal_material_thickness_m
                      for b in (donor,receiver))
    depth_max = (top**3-bottom_cube)/(3*R**2)
    if depth_max <= 0:
        raise ValueError("The source activation banks lack a positive common radial face")
    end = family.native_angle_end_bounds_radians
    A = family.shared_depth_axis_per_m
    tau_top = (R**3-top**3)/(3*R**2)
    maximum_tau = max(abs(tau_top),abs(tau_top+depth_max))
    coefficients = tuple(_conjugation_coefficients(E,A,precision,order)
                         for E in AFFINE_SYMMETRIC_GENERATORS)
    remainder = _conjugation_remainder_bound(A,maximum_tau,precision,order)
    panel_moments = []
    for cell in range(depth_cells):
        low = tau_top+depth_max*cell/depth_cells
        high = tau_top+depth_max*(cell+1)/depth_cells
        exact = tuple((high**(k+1)-low**(k+1))/(k+1) for k in range(order+2))
        # First compute the correlated moments exactly, then enclose their
        # arithmetic once. Reusing these bounds avoids enormous inherited
        # source denominators in every Cartesian coefficient multiplication.
        moments = tuple((_rounded_interval(I(exact[k]),precision+32),
            _rounded_interval(I(R**3*exact[k]-3*R**2*exact[k+1]),precision+32))
            for k in range(order+1))
        width = high-low
        panel_moments.append((moments,remainder*width,
            remainder*(R**3-3*R**2*low)*width))
    # The source rotation is conjugated into the fixed generator. Exact depth
    # moments integrate every panel; this avoids interval SO(3) sampling of
    # each Cartesian vector and retains all correlated radial dependence.
    integrated = []
    for series in coefficients:
        zeroth = [[I(0) for _ in range(3)] for _ in range(3)]
        radial = [[I(0) for _ in range(3)] for _ in range(3)]
        for moments,error0,error3 in panel_moments:
            for k,coefficient in enumerate(series):
                first,third = moments[k]
                for i in range(3):
                    for j in range(3):
                        zeroth[i][j] = _rounded_interval(zeroth[i][j]+coefficient[i][j]*first,precision)
                        radial[i][j] = _rounded_interval(radial[i][j]+coefficient[i][j]*third,precision)
            for i in range(3):
                for j in range(3):
                    zeroth[i][j] = _rounded_interval(zeroth[i][j]+I(-error0,error0),precision)
                    radial[i][j] = _rounded_interval(radial[i][j]+I(-error3,error3),precision)
        integrated.append((tuple(tuple(row) for row in zeroth),tuple(tuple(row) for row in radial)))
    sums = [I(0) for _ in AFFINE_SYMMETRIC_GENERATORS]
    for theta,uncertain in ((I(0,end.lo),False),(I(end.lo,end.hi),True)):
        if not theta.width:
            continue
        sine,cosine = _trig_ranges_signed(theta.lo,theta.hi,precision)
        U = tuple(a*cosine+b*sine for a,b in zip(family.start_unit_bounds,family.tangent_unit_bounds))
        T = tuple(-a*sine+b*cosine for a,b in zip(family.start_unit_bounds,family.tangent_unit_bounds))
        projection = _dot(A,T)
        for i,(zeroth,radial) in enumerate(integrated):
            value = family.receiving_normal_orientation*(R*_dot(family.normal_unit_bounds,_matvec(zeroth,U))
                -projection/R*_dot(U,_matvec(radial,U)))
            integral = value*theta.width
            if uncertain:
                integral = I(min(F(0),integral.lo),max(F(0),integral.hi))
            sums[i] = _rounded_interval(sums[i]+integral,precision)
    return (tuple(sums),top,depth_max,I(bottom_cube,top**3))


def _conjugation_coefficients(generator,angular_rate,precision,order):
    """Enclose coefficients of exp(-C tau) E exp(C tau) about tau=0.

    L(E)=E C-C E, with C=[A]x. All arithmetic compression is outward;
    the exact source A remains the declared physical field throughout.
    """
    # These are outward arithmetic enclosures of the preserved literal A,
    # not midpoint declarations or replacements of its saved source field.
    # Keeping its enormous exact source denominators in every recurrence is
    # unnecessary once that exact value has been enclosed here.
    x,y,z = tuple(_rounded_interval(I(value),precision+32) for value in angular_rate)
    C = ((I(0),-z,y),(z,I(0),-x),(-y,x,I(0)))
    coefficient = tuple(tuple(I(value) for value in row) for row in generator)
    result = [coefficient]
    for k in range(order):
        coefficient = tuple(tuple(_rounded_interval(sum((coefficient[i][m]*C[m][j]
            -C[i][m]*coefficient[m][j] for m in range(3)),I(0))/(k+1),precision+16)
            for j in range(3)) for i in range(3))
        result.append(coefficient)
    return tuple(result)


def _conjugation_remainder_bound(angular_rate,maximum_abs_tau,precision,order):
    """Uniform matrix1-norm bound on the full exponential conjugation tail.

    Each symmetric generator has matrix1-norm1. The commutator has induced
    norm <=2|A|1. For q=2|A|1 max|tau|, the exponential tail starting at
    n=order+1 is <=q**n/n!/(1-q/(n+1)). No material/force scale floor enters.
    """
    rate = sum(map(abs,angular_rate),F(0))
    if not rate or not maximum_abs_tau:
        return F(0)
    q = _rounded_interval(I(2*rate*maximum_abs_tau),precision+16).hi
    ratio = q/(order+2)
    if ratio >= 1:
        raise ValueError("The source angular-depth field exceeds the validated commutator-series domain")
    return q**(order+1)/factorial(order+1)/(1-ratio)


def bound_native_affine_source_face_trace(family_id,bank,*,source=None,depth_cells=64,precision=160,order=28):
    source = load_native_normal_support_source() if source is None else validate_native_normal_support_source(source)
    _budget(precision,order)
    if type(bank) is not str or bank not in ("donor","receiver"):
        raise ValueError("An actual source face trace requires donor or receiver bank identity")
    if type(depth_cells) is not int or not 1 <= depth_cells <= 1024:
        raise ValueError("Whole-face integration requires a literal positive depth-cell count")
    families = tuple(f for f in source.families if f.family_id == family_id)
    if type(family_id) is not str or len(families) != 1:
        raise ValueError("The activation trace needs one complete authenticated native carrier")
    family = families[0]
    B,top,depth_max,cubes = _integrate_source_face_trace(source,family_id,depth_cells,precision,order)
    sign = 1 if bank == "receiver" else -1
    component_id = family.receiver_component_id if sign == 1 else family.donor_component_id
    end = family.native_angle_end_bounds_radians
    result = NativeAffineSourceFaceTrace(family_id,component_id,bank,sign,
        source.source_snapshot_sha256,source.source_support_sha256,source.source_epoch_sha256,
        source.curved_packet_raw_sha256,source.saved_field_declaration_precision,I(0,end.hi),end,
        I(0,end.lo),I(end.lo,end.hi),I(0,depth_max),top,cubes,family.source_preload_gap_m,
        AFFINE_COORDINATE_ORDER,tuple(sign*x for x in B),depth_cells,precision,order,"")
    return _remember(result,_FACE_AUTHENTICATED)


def validate_native_affine_source_face_trace(face,*,body=None,source=None):
    face = _validate_registered(face,NativeAffineSourceFaceTrace,_FACE_AUTHENTICATED)
    if body is not None:
        body = validate_native_affine_hot_body(body)
        if (face.component_id != body.component_id or face.source_snapshot_sha256 != body.source_snapshot_sha256):
            raise ValueError("A contact trace cannot substitute another physical bank or source epoch")
    if source is not None:
        source = validate_native_normal_support_source(source)
        if face.source_snapshot_sha256 != source.source_snapshot_sha256:
            raise ValueError("The affine face belongs to another complete source snapshot")
    return face


def _normalized_native_vertex_bounds(raw,precision):
    from .geometric_native_polygons import native_polygon_from_dict, _ray_values
    polygon = native_polygon_from_dict(raw)
    vertices = tuple(_ray_values(ray.rotated(polygon.pose),precision)[1] for ray in polygon.vertices)
    return tuple(tuple(_rounded_interval(x/interval_sqrt(_dot(v,v),precision+16),precision)
                       for x in v) for v in vertices)


def _unit_cone_bounds(vertices,precision):
    # Positive combinations of a convex native polygon's normalized rays give
    # q in their convex hull. A unit n0 with n0 dot(vertex)>=c proves |q|>=c;
    # |q|<=1 follows from convexity. No chart is allocated as material here.
    centre = tuple(sum((v[i].midpoint for v in vertices),F(0))/len(vertices) for i in range(3))
    norm = interval_sqrt(I(_dot(centre,centre)),precision+16)
    if norm.lo <= 0:
        raise ValueError("The complete containing chart needs subdivision into a positive cone")
    n0 = tuple(x/norm.hi for x in centre)
    c = min(_dot(n0,v).lo for v in vertices)
    if c <= 0:
        raise ValueError("The complete source chart lacks a certified containing hemisphere")
    bounds = _hull(vertices)
    scale = I(c,F(1))
    result = tuple(_rounded_interval(v/scale,precision) for v in bounds)
    return tuple(I(max(F(-1),v.lo),min(F(1),v.hi)) for v in result)


@dataclass(frozen=True, slots=True)
class NativeAffineBasalSource:
    body_id: str
    body_geometry_source_sha256: str
    whole_mask_unit_direction_bounds: tuple
    containing_chart_unit_direction_bounds: tuple
    base_radius_bounds_m: I
    density_kg_m3: F
    gravity_m_s2: F
    actual_area_bounds_sr: I
    reference_measure_area_bounds_m2: I
    source_foundation_json: str
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    saved_field_declaration_precision: int
    positive_correction_containing_chart_count: int
    precision: int
    order: int
    geometry_source_sha256: str

    @property
    def source_reference_measure_is_R_squared_dOmega(self):
        return True

    @property
    def complete_actual_mask_contained(self):
        return True

    @property
    def containing_charts_assigned_as_material(self):
        return False

    @property
    def inherited_float_foundation_credit(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bind_native_affine_basal_source(body,*,precision=160,order=28):
    body = validate_native_affine_hot_body(body)
    _budget(precision,order)
    source = _BODY_SOURCES[body.geometry_source_sha256]
    support = json.loads(dict(source.complete_source_dto_json)["full_reference_face_birth_support"])
    foundations = tuple(x for x in support["foundations"] if x["declaration"]["component_id"] == body.component_id)
    if len(foundations) != 1:
        raise ValueError("An actual HOT body lost its unique source basal declaration")
    foundation = foundations[0]
    parameters = foundation["declaration"]["parameters"]
    rho,gravity = F.from_float(parameters["density_kg_m3"]),F.from_float(parameters["gravity_m_s2"])
    if rho <= 0 or gravity <= 0:
        raise ValueError("The source hydrostatic Winkler parameters must remain positive")
    component = json.loads(body.complete_mask_source_json)
    charts = list(component["proxy_geometries"])
    positive = 0
    for encoded in body.paired_corrections_source_json:
        correction = json.loads(encoded)
        if correction["positive_component_id"] == body.component_id:
            charts.append(correction["containing_chart"])
            positive += 1
    if not charts:
        raise ValueError("The whole actual HOT mask lacks authenticated containing charts")
    z = body.areal_material_thickness_m
    R = body.radius_m
    cube = (R+body.roof_radial_displacement_m)**3-3*R**2*z
    radius = interval_cbrt(I(cube),precision)
    tau = I((R**3-cube)/(3*R**2))
    enclosures = []
    for chart in charts:
        vertices = _normalized_native_vertex_bounds(chart,precision)
        rotated = tuple(rotate_joint_physical_depth(v,body.angular_rate_per_physical_volume_depth_m,
            tau,precision=precision,order=order) for v in vertices)
        enclosures.append(_unit_cone_bounds(rotated,precision))
    result = NativeAffineBasalSource(body.component_id,body.geometry_source_sha256,
        _hull(enclosures),tuple(enclosures),radius,rho,gravity,body.actual_area_bounds_sr,
        body.actual_area_bounds_sr*R**2,_encoded(foundation).decode(),body.source_snapshot_sha256,
        body.source_support_sha256,body.source_epoch_sha256,body.saved_field_declaration_precision,
        positive,precision,order,"")
    return _remember(result,_BASAL_AUTHENTICATED)


def validate_native_affine_basal_source(value,*,body=None):
    value = _validate_registered(value,NativeAffineBasalSource,_BASAL_AUTHENTICATED)
    if body is not None:
        body = validate_native_affine_hot_body(body)
        if value.body_geometry_source_sha256 != body.geometry_source_sha256:
            raise ValueError("The affine basal receipt cannot substitute another actual body")
    return value


def native_affine_body_to_dict(body):
    return _payload(validate_native_affine_hot_body(body))


def native_affine_source_face_trace_to_dict(face):
    return _payload(validate_native_affine_source_face_trace(face))


def native_affine_basal_source_to_dict(value):
    return _payload(validate_native_affine_basal_source(value))

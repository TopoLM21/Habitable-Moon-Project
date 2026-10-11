"""Seven supported modal fields on each complete authenticated cold bank.

The old cantilever's cubic radial tip shape is given an explicit new 3D lift.
Its previous full 3D displacement field is not inferred. The six enrichments
and that lift vanish together with their gradients on the source clamp; the
complete material beyond it remains present with zero increment. Actual
archive domains and curved removed measures remain source data, not proxies.
Energy calibration, force balance and material admission are separate work.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import json

from .geometric_intrinsic_geometry import rotation_increment_from_dict
from .geometric_native_pose import FractionInterval as I, interval_pi, interval_atan, certify_rotation_history
from .geometric_native_normal_support import (
    _encoded, _payload, _read_interval, load_native_normal_support_source,
    validate_native_normal_support_source,
)
from .geometric_native_affine_body import (
    AFFINE_SYMMETRIC_GENERATORS, AFFINE_COORDINATE_ORDER,
    _remember, _validate_registered,
)

_AUTHENTICATED = {}
_BODY_SOURCES = {}


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),F(0))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _map(matrix,vector):
    return tuple(_dot(row,vector) for row in matrix)


def _det(matrix):
    return _dot(matrix[0],_cross(matrix[1],matrix[2]))


@dataclass(frozen=True, slots=True)
class NativeColdBody:
    component_id: str
    bank_alias: str
    material_id: str
    original_parent_id: str
    owner: int
    trace_id: str
    source_bank: str
    old_absolute_coordinate_index: int
    radius_m: F
    areal_material_thickness_m: F
    roof_radial_displacement_m: F
    accepted_tip_displacement_m: F
    source_old_tip_stiffness_n_m: F
    young_modulus_pa: F
    poisson_ratio: F
    elastic_thickness_m: F
    width_m: F
    beta_cut_radians: F
    elastic_layer_areal_depth_m: F
    original_quarter_start: tuple
    original_quarter_end: tuple
    original_quarter_inward: tuple
    current_quarter_start_bounds: tuple
    current_quarter_tangent_bounds: tuple
    current_quarter_inward_bounds: tuple
    owner_rotation_matrix_bounds: tuple
    angular_rate_per_physical_volume_depth_m: tuple
    actual_area_bounds_sr: I
    reference_volume_bounds_m3: I
    archive_area_bounds_sr: I
    endpoint_archive_area_bounds_sr: I
    paired_curved_removed_area_bounds_sr: tuple
    paired_curved_cap_theta_width_bounds_radians: I
    paired_curved_beta_upper_bounds_radians: I
    endpoint_plane_coefficients_in_quarter: tuple
    endpoint_plane_source_json: str
    complete_mask_source_json: str
    complete_parent_source_json: str
    complete_parent_allocation_json: str
    complete_donor_domain_json: str
    complete_source_sweep_certificate_json: str
    paired_corrections_source_json: tuple
    complete_source_trace_json: str
    complete_source_bending_parameters_json: str
    complete_source_clamp_support_json: str
    source_clamp_row_json: str
    source_owner_history_json: str
    saved_field_json: str
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_packet_raw_sha256: str
    saved_field_declaration_precision: int
    geometry_enclosure_precision: int
    coordinate_order: tuple
    modal_reference_provenance: str
    provenance: str
    geometry_source_sha256: str

    @property
    def body_snapshot_sha256(self):
        return self.geometry_source_sha256

    @property
    def physical_elastic_thickness_m(self):
        return self.elastic_thickness_m

    @property
    def complete_implicit_material_mask_retained(self):
        return True

    @property
    def archived_material_used_as_live_support(self):
        return False

    @property
    def proxy_polygons_promoted_to_material(self):
        return False

    @property
    def previous_full_three_dimensional_tip_field_recovered(self):
        return False

    @property
    def original_source_clamp_material_support_retained(self):
        return True

    @property
    def prepared_tip_displacement_reset(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bind_native_cold_body(component_id,*,source=None,precision=160,
        provenance="Explicit supported seven-mode cold field with the complete actual source mask"):
    source = load_native_normal_support_source() if source is None else validate_native_normal_support_source(source)
    if type(precision) is not int or not 64 <= precision <= 512:
        raise ValueError("Cold source geometry precision must be an integer from64 to512")
    if type(component_id) is not str or type(provenance) is not str or not provenance.strip():
        raise ValueError("An actual cold body requires literal identity and explicit provenance")
    matches = tuple(b for b in source.banks if b.component_id == component_id)
    if len(matches) != 1 or matches[0].material_kind != "original_live":
        raise ValueError("A source cold body must retain one actual original live bank")
    bank = matches[0]
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    epoch = before["snapshot"]["current_material_epoch"]
    material = epoch["source_material_state"]
    mechanics = material["original_mechanics_payload"]
    bindings = tuple(b for b in material["trace_bindings"]
        if bank.original_parent_id in (b["donor_fragment_id"],b["receiver_fragment_id"]))
    if len(bindings) != 1:
        raise ValueError("The complete cold bank lacks one original physical tip/clamp source")
    binding = bindings[0]
    role = "sub" if binding["donor_fragment_id"] == bank.original_parent_id else "over"
    trace_id = binding["trace_id"]
    traces = mechanics["fault_traces"]["traces"]
    indices = tuple(i for i,t in enumerate(traces) if t["trace_id"] == trace_id)
    if len(indices) != 1:
        raise ValueError("The cold bank lost its unique original physical trace")
    trace = traces[indices[0]]
    parameters = tuple(p for p in mechanics["bending_parameters"] if p["trace_id"] == trace_id)
    if len(parameters) != 1:
        raise ValueError("The cold bank lost its independently declared original bending parameters")
    parameters = parameters[0][role]
    R,H = source.radius_m,bank.areal_material_thickness_m
    E,nu,h,L = tuple(F.from_float(parameters[name]) for name in
        ("young_modulus_pa","poisson_ratio","elastic_thickness_m","width_m"))
    if E <= 0 or not -1 < nu < F(1,2) or not 0 < L/R <= F(1,4) or h <= 0:
        raise ValueError("The original cold material and clamp parameters must remain physical")
    tips = tuple(t for t in before["snapshot"]["transverse_displacement_m"] if t[0] == trace_id)
    if len(tips) != 1:
        raise ValueError("The current accepted cold bank lost its physical tip reference")
    wref = F.from_float(tips[0][1 if role == "sub" else 2])
    if wref != bank.roof_radial_displacement_m:
        raise ValueError("The saved128 source roof differs from the accepted physical tip source")
    qindex = 3*len(mechanics["plate_ids"])+2*indices[0]+int(role == "over")
    stiffness = before["original_physical_history"]["original_reference"]["stiffness"]
    kold = F.from_float(stiffness[qindex][qindex])/R**2
    if kold <= 0 or any(F.from_float(value) for j,value in enumerate(stiffness[qindex]) if j != qindex):
        raise ValueError("The source physical cold tip must retain its actual positive independent beam stiffness")
    top = R+bank.roof_radial_displacement_m
    ze = (top**3-(top-h)**3)/(3*R**2)
    if top-h <= 0 or not 0 < ze < H:
        raise ValueError("The source physical elastic thickness must fit inside the entire actual material layer")
    parents = tuple(p for p in material["journal"]["source_surface"]["fragments"]
        if p["fragment_id"] == bank.original_parent_id)
    allocations = tuple(p for p in material["parent_allocations"] if p["fragment_id"] == bank.original_parent_id)
    if len(parents) != 1 or len(allocations) != 1:
        raise ValueError("The original cold body lost its complete original parent and material allocation")
    parent,allocation = parents[0],allocations[0]
    if (allocation["live_domain_id"] != component_id or allocation["material_id"] != bank.material_id
            or allocation["owner"] != bank.owner):
        raise ValueError("A cold body cannot substitute another retained material reservoir")
    start,end = tuple(F.from_float(x) for x in trace["start"]),tuple(F.from_float(x) for x in trace["end"])
    inward = tuple((-1 if role == "sub" else 1)*x for x in _cross(start,end))
    if (_dot(start,start) != 1 or _dot(end,end) != 1 or _dot(start,end)
            or _dot(inward,inward) != 1):
        raise ValueError("The source cold clamp requires the literal orthonormal quarter frame")
    journal = material["journal"]
    motions = dict(journal["omega_rad_per_myr"])
    history_raw = [dict(omega_rad_per_myr=motions[bank.owner],dt_myr=journal["dt_myr"])]
    extras = dict(epoch["additional_owner_histories"])[bank.owner]
    history_raw.extend(extras)
    history = tuple(rotation_increment_from_dict(x) for x in history_raw)
    O = certify_rotation_history(history,precision).intervals
    current_start,current_end,current_inward = tuple(_map(O,v) for v in (start,end,inward))
    reference = json.loads(source.saved_field_json)
    groups = tuple(g for g in reference["groups"] if component_id in g["component_ids"])
    if len(groups) != 1:
        raise ValueError("The complete cold body lacks its unique saved joint depth field")
    axes = tuple(f.shared_depth_axis_per_m for f in source.families if f.family_id in groups[0]["family_ids"])
    if axes:
        A = axes[0]
        if any(a != A for a in axes):
            raise ValueError("A cold bank cannot acquire contact-specific conflicting bulk reference fields")
    else:
        aliases = tuple(_read_interval(x) for x in groups[0]["angular_rate_per_physical_volume_depth_m"])
        if any(x.width or x.lo for x in aliases):
            raise ValueError("A nonzero saved cold angular rate cannot be replaced with an alias midpoint")
        A = (F(0),)*3
    if any(not _read_interval(alias).lo <= x <= _read_interval(alias).hi for x,alias in
        zip(A,groups[0]["angular_rate_per_physical_volume_depth_m"])):
        raise ValueError("The actual cold depth flow differs from its saved128 declaration")
    domains = tuple(d for d in material["sweep_certificate"]["donor_domains"]
        if d["fragment_id"] == bank.original_parent_id)
    if len(domains) > 1:
        raise ValueError("The cold bank's once-only archive must remain unique")
    domain = domains[0] if domains else None
    archive = _read_interval(allocation["archived_area_bounds_sr"])
    endpoint = I(0) if domain is None else _read_interval(domain["endpoint_area_bounds_sr"])
    if domain is None and archive != I(0):
        raise ValueError("An unarchived receiver cannot lose actual material")
    corrections = tuple(c for c in epoch["paired_corrections"] if c["negative_component_id"] == component_id)
    curved = tuple(_read_interval(c["area_bounds_sr"]) for c in corrections)
    sweep = material["sweep_certificate"]
    curved_theta_cap,curved_beta_upper = I(0),I(0)
    if corrections:
        # This actual first-entry branch is in original (x,-y,z) coordinates:
        # tan(beta)<-a(t)cos(theta)+b(t)sin(theta), a=Y/C>0.
        # Its complete time envelope gives cot(theta)<=max b/a.  Source
        # b'>0 gives tan(beta)<=b(1).  The cap contains removed material;
        # it supplies no area or new material to the integration domain.
        if (bank.original_parent_id != sweep["extra_endpoint_fixed_parent_id"]
                or domain is None or domain["domain_case"] != "endpoint_plus_curved_swept_extra"
                or (start,end,inward) != ((F(0),F(-1),F(0)),(F(0),F(0),F(1)),(F(1),F(0),F(0)))):
            raise ValueError("The curved cold archive requires its authentic whole-time source branch")
        if any(_read_interval(sweep["proof_bounds"][key]).lo <= 0 for key in
                ("c_positive","y_over_u_positive","b_derivative_numerator_positive")):
            raise ValueError("The whole swept cap lost its source positive/monotone proof")
        ratio = _read_interval(sweep["envelope"]["maximum_ratio_bounds"])
        curved_theta_cap = interval_atan(ratio,max(precision,256))
    plane_json,plane_coefficients = "",()
    if domain is not None and domain["endpoint_domain"] is not None:
        from .geometric_native_polygons import native_plane_from_dict, _plane_values
        planes = tuple(p for p in domain["endpoint_domain"]["supports"] if p["rotations"])
        if len(planes) != 1:
            raise ValueError("The source endpoint archive lacks its unique actual foreign rotated support plane")
        raw_plane = planes[0]
        v = _plane_values(native_plane_from_dict(raw_plane),max(precision,256))[1]
        plane_coefficients = tuple(_dot(v,axis) for axis in (start,end,inward))
        if plane_coefficients[2].hi >= 0:
            raise ValueError("The endpoint archive strip requires its certified inward-negative source plane")
        plane_json = _encoded(raw_plane).decode()
        if corrections:
            # Here -v·E/(v·n) is b(1)=Z(1)/C(1), distinct from the
            # envelope's terminal_height=b(1)-a(1)*k(1).
            terminal_beta_ratio = -plane_coefficients[1]/plane_coefficients[2]
            if terminal_beta_ratio.lo <= 0:
                raise ValueError("The whole swept cold cap requires its positive terminal transverse bound")
            curved_beta_upper = interval_atan(terminal_beta_ratio,max(precision,256))
    clamp_support = before["temporal_admission"]["original_bank_support"]
    clamps = tuple(c for c in clamp_support["live_clamps"] if c["trace_id"] == trace_id and c["bank"] == role)
    if (len(clamps) != 1 or not clamp_support["live_original_clamp_support_certified"]
            or not clamp_support["complete_original_bank_support_certified"]):
        raise ValueError("The whole original source clamp must retain its independent live material support")
    clamp = clamps[0]
    if F(*clamp["width_m"]) != L or clamp["owner"] != bank.owner:
        raise ValueError("The cold body cannot substitute another clamp width or material owner")
    area = _read_interval(json.loads(bank.complete_mask_source_json)["area_bounds_sr"])
    if area.lo <= 0:
        raise ValueError("Every complete actual cold bank must retain positive source volume")
    sorted_components = tuple(sorted(b.component_id for b in source.banks))
    alias = "C"+str(sorted_components.index(component_id))
    body = NativeColdBody(component_id,alias,bank.material_id,bank.original_parent_id,bank.owner,
        trace_id,role,qindex,R,H,bank.roof_radial_displacement_m,wref,kold,E,nu,h,L,L/R,ze,
        start,end,inward,current_start,current_end,current_inward,O,A,area,area*R**2*H,
        archive,endpoint,curved,curved_theta_cap,curved_beta_upper,
        plane_coefficients,plane_json,bank.complete_mask_source_json,
        _encoded(parent).decode(),_encoded(allocation).decode(),"" if domain is None else _encoded(domain).decode(),
        _encoded(material["sweep_certificate"]).decode(),
        tuple(_encoded(c).decode() for c in corrections),_encoded(trace).decode(),_encoded(parameters).decode(),
        _encoded(clamp_support).decode(),_encoded(clamp).decode(),_encoded(history_raw).decode(),source.saved_field_json,
        source.source_snapshot_sha256,source.source_support_sha256,source.source_epoch_sha256,
        source.curved_packet_raw_sha256,source.saved_field_declaration_precision,precision,
        ("old_tip_increment",)+AFFINE_COORDINATE_ORDER,
        "New explicit radial3D lift of the unchanged source cubic tip shape; complete original3D displacement not recovered",
        provenance,"")
    body = _remember(body,_AUTHENTICATED)
    _BODY_SOURCES[body.geometry_source_sha256] = source
    return body


def validate_native_cold_body(body):
    return _validate_registered(body,NativeColdBody,_AUTHENTICATED)


@dataclass(frozen=True, slots=True)
class NativeColdModalGeometry:
    world_point_m: tuple
    radial_direction: tuple
    material_beta_gradient_per_m: tuple
    reference_flow_gradient_after_owner_rotation: tuple
    reference_gradient_from_original_cartesian: tuple
    mode_displacement_per_m: tuple
    mode_gradient_m_inv: tuple
    radius: object
    physical_volume_depth: object
    source_shape: object
    source_shape_beta_derivative: object

    @property
    def material_membership_authenticated(self):
        return False

    @property
    def exact_reference_jacobian_determinant(self):
        # SO(3), the rank-one determinant lemma, and p·(A×p)=0.
        return F(1)


def evaluate_native_cold_modal_geometry(body,sine_theta,cosine_theta,beta,z,algebra):
    """Analytic interior geometry, with theta trig scalars supplied by caller.

    An integration adapter may carry trig polynomials whose coefficients are
    beta/depth Taylor jets. This analytic expression requires0<=beta<=L/R;
    bound_native_cold_point implements the zero extension beyond the clamp.
    Domain membership is independent of either evaluator.
    """
    body = validate_native_cold_body(body)
    C = algebra.constant
    R,b = C(body.radius_m),C(body.beta_cut_radians)
    c_beta,s_beta = algebra.cos(beta),algebra.sin(beta)
    S,E,n = tuple(tuple(C(x) for x in axis) for axis in
        (body.current_quarter_start_bounds,body.current_quarter_tangent_bounds,body.current_quarter_inward_bounds))
    U = tuple(a*cosine_theta+t*sine_theta for a,t in zip(S,E))
    p = tuple(u*c_beta+v*s_beta for u,v in zip(U,n))
    tangent = tuple(v*c_beta-u*s_beta for u,v in zip(U,n))
    cube = C((body.radius_m+body.roof_radial_displacement_m)**3)-3*R*R*z
    r = algebra.cbrt(cube)
    tau = (R**3-cube)/(3*R*R)
    A = tuple(C(x) for x in body.angular_rate_per_physical_volume_depth_m)
    if any(body.angular_rate_per_physical_volume_depth_m):
        norm_squared = _dot(body.angular_rate_per_physical_volume_depth_m,
                            body.angular_rate_per_physical_volume_depth_m)
        norm = algebra.sqrt(C(norm_squared))
        x,y,v = A
        K = ((C(0),-v,y),(v,C(0),-x),(-y,x,C(0)))
        K2 = tuple(tuple(sum((K[i][k]*K[k][j] for k in range(3)),C(0)) for j in range(3)) for i in range(3))
        sine,cosine = algebra.sin(norm*tau),algebra.cos(norm*tau)
        Q = tuple(tuple(C(int(i == j))+sine/norm*K[i][j]
            +(C(1)-cosine)/C(norm_squared)*K2[i][j] for j in range(3)) for i in range(3))
        e,t = tuple(sum((Q[i][j]*p[j] for j in range(3)),C(0)) for i in range(3)),tuple(
            sum((Q[i][j]*tangent[j] for j in range(3)),C(0)) for i in range(3))
    else:
        Q = tuple(tuple(C(int(i == j)) for j in range(3)) for i in range(3))
        e,t = p,tangent
    point = tuple(r*x for x in e)
    rate = sum((n[i]*_cross(A,U)[i] for i in range(3)),C(0))
    grad_beta = tuple(ti/r+(r*r/(R*R))*rate*ei for ti,ei in zip(t,e))
    shear = _cross(A,p)
    rank_one = tuple(tuple(C(int(i == j))-(r**3/(R*R))*shear[i]*p[j]
        for j in range(3)) for i in range(3))
    reference_flow = tuple(tuple(sum((Q[i][k]*rank_one[k][j] for k in range(3)),C(0))
        for j in range(3)) for i in range(3))
    reference_gradient = tuple(tuple(sum((reference_flow[i][k]*C(body.owner_rotation_matrix_bounds[k][j])
        for k in range(3)),C(0)) for j in range(3)) for i in range(3))
    scaled = beta/b
    chi = (C(1)-scaled)**2*(scaled+C(2))/2
    chi_prime = -C(F(3,2))*(C(1)-scaled)*(C(1)+scaled)/b
    modes = [tuple(chi*x for x in e)]
    gradients = [tuple(tuple(chi/r*(C(int(i == j))-e[i]*e[j])
        +chi_prime*e[i]*grad_beta[j] for j in range(3)) for i in range(3))]
    for generator in AFFINE_SYMMETRIC_GENERATORS:
        direction = tuple(sum((C(generator[i][j])*point[j]/R for j in range(3)),C(0)) for i in range(3))
        modes.append(tuple(chi*x for x in direction))
        gradients.append(tuple(tuple(chi*C(generator[i][j])/R
            +chi_prime*direction[i]*grad_beta[j] for j in range(3)) for i in range(3)))
    return NativeColdModalGeometry(point,e,grad_beta,reference_flow,reference_gradient,
        tuple(modes),tuple(gradients),r,tau,chi,chi_prime)


@dataclass(frozen=True, slots=True)
class NativeColdPointBounds:
    body: NativeColdBody
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
    def local_positive_jacobian_certified(self):
        return self.incremental_determinant_bounds.lo > 0

    @property
    def whole_body_injectivity_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_cold_point(body,theta_bounds,beta_bounds,material_depth_bounds_m,
        amplitudes_m=(F(0),)*7,*,precision=160):
    from .geometric_native_curved_certified_integration import TaylorAlgebra
    body = validate_native_cold_body(body)
    if (type(theta_bounds) is not I or type(beta_bounds) is not I or type(material_depth_bounds_m) is not I
            or not 0 <= theta_bounds.lo <= theta_bounds.hi <= interval_pi(precision).hi/2
            or not 0 <= beta_bounds.lo <= beta_bounds.hi <= interval_pi(precision).hi/2
            or not 0 <= material_depth_bounds_m.lo <= material_depth_bounds_m.hi <= body.areal_material_thickness_m):
        raise ValueError("Cold geometric queries require a complete valid quarter/depth enclosure")
    if (type(amplitudes_m) is not tuple or len(amplitudes_m) != 7
            or any(type(x) is not F for x in amplitudes_m)):
        raise ValueError("Cold modal amplitudes must remain seven literal Fraction metres")
    algebra = TaylorAlgebra(0,precision)
    theta,beta,z = tuple(algebra.constant(x) for x in (theta_bounds,beta_bounds,material_depth_bounds_m))
    geometry = evaluate_native_cold_modal_geometry(body,algebra.sin(theta),algebra.cos(theta),beta,z,algebra)
    point = tuple(x.constant_bounds for x in geometry.world_point_m)
    active = beta_bounds.lo < body.beta_cut_radians
    if beta_bounds.lo >= body.beta_cut_radians:
        modes = tuple((I(0),)*3 for _ in range(7))
        gradients = tuple(tuple((I(0),)*3 for _ in range(3)) for _ in range(7))
    elif beta_bounds.hi <= body.beta_cut_radians:
        modes = tuple(tuple(x.constant_bounds for x in mode) for mode in geometry.mode_displacement_per_m)
        gradients = tuple(tuple(tuple(x.constant_bounds for x in row) for row in mode)
            for mode in geometry.mode_gradient_m_inv)
    else:
        inside = bound_native_cold_point(body,theta_bounds,I(beta_bounds.lo,body.beta_cut_radians),
            material_depth_bounds_m,amplitudes_m,precision=precision)
        modes = tuple(tuple(I(min(F(0),v.lo),max(F(0),v.hi)) for v in mode)
            for mode in inside.modal_displacement_bounds)
        gradients = tuple(tuple(tuple(I(min(F(0),v.lo),max(F(0),v.hi)) for v in row) for row in mode)
            for mode in inside.modal_gradient_bounds_m_inv)
    displacement = tuple(sum((a*mode[i] for a,mode in zip(amplitudes_m,modes)),I(0)) for i in range(3))
    image = tuple(x+d for x,d in zip(point,displacement))
    gradient = tuple(tuple(I(int(i == j))+sum((a*mode[i][j]
        for a,mode in zip(amplitudes_m,gradients)),I(0)) for j in range(3)) for i in range(3))
    return NativeColdPointBounds(body,point,image,modes,gradients,gradient,_det(gradient),active)


def native_cold_body_to_dict(body):
    return _payload(validate_native_cold_body(body))


@dataclass(frozen=True, slots=True)
class NativeColdBodySupportBounds:
    """Whole source-quarter gradient enclosure including its actual live mask.

    The integration domain remains the authenticated mask.  The larger
    quarter is used only to bound the field, never to allocate material.
    Outside the source clamp the increment and its gradient are exactly zero.
    """
    body: NativeColdBody
    amplitudes_m: tuple
    whole_clamped_quarter_modal_gradient_bounds_m_inv: tuple
    maximum_symmetric_old_tip_gradient_norm_m_inv: F
    prepared_metric_eigenvalue_lower_bound: F
    whole_clamped_quarter_incremental_gradient_bounds: tuple
    whole_clamped_quarter_incremental_determinant_bounds: I

    @property
    def complete_actual_material_mask_retained(self):
        return True

    @property
    def original_live_clamp_support_certified(self):
        return True

    @property
    def source_clamp_displacement_and_gradient_zero_certified(self):
        return True

    @property
    def outside_source_clamp_displacement_and_gradient_zero_certified(self):
        return True

    @property
    def prepared_initial_metric_positive_definite_certified(self):
        return self.prepared_metric_eigenvalue_lower_bound > 0

    @property
    def whole_body_local_positive_jacobian_certified(self):
        # Outside the clamped support the incremental determinant is exactly1.
        return self.whole_clamped_quarter_incremental_determinant_bounds.lo > 0

    @property
    def whole_body_injectivity_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_cold_body_support(body,amplitudes_m=(F(0),)*7,*,precision=160):
    """Bound all seven gradients over all source-supported live material.

    The symmetrized old-tip gradient row-sum bounds its spectral norm.
    Weyl's inequality then bounds I+2*w_ref*sym(G0), retaining the declared
    initial strain.  The local incremental Jacobian interval is reported
    separately from global embedding and physical pair admission.
    """
    body = validate_native_cold_body(body)
    point = bound_native_cold_point(body,I(0,interval_pi(precision).hi/2),
        I(0,body.beta_cut_radians),I(0,body.areal_material_thickness_m),
        amplitudes_m,precision=precision)
    G0 = point.modal_gradient_bounds_m_inv[0]
    symmetric = tuple(tuple((G0[i][j]+G0[j][i])/2 for j in range(3)) for i in range(3))
    norm = max(sum((max(abs(x.lo),abs(x.hi)) for x in row),F(0)) for row in symmetric)
    lower = 1-2*abs(body.accepted_tip_displacement_m)*norm
    return NativeColdBodySupportBounds(body,amplitudes_m,point.modal_gradient_bounds_m_inv,
        norm,lower,point.incremental_gradient_bounds,point.incremental_determinant_bounds)

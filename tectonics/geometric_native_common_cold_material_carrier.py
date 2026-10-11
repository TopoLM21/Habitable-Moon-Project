"""A declared smooth whole-owner cold extension on the authentic archive carrier.

The old bank fields are material-label functions, extended through a quintic
partition at their quarter seams.  Fixed radial guards separate the two cold
owners.  This is a geometry/support proposal, not a new bulk law, force solve,
HOT law, all-pair admission, or an accepted time interval.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_sqrt, interval_atan
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_joint_spherical_field import rotate_joint_physical_depth
from .geometric_native_archive_material_carrier import (
    NativeArchiveMaterialCarrier, native_archive_material_carrier_to_dict,
    evaluate_native_archive_material_point, native_archive_material_point_to_dict,
    _sin_cos,
)
from .geometric_native_affine_body import AFFINE_SYMMETRIC_GENERATORS
from .geometric_native_common_source_carrier import _round, _dot, _cross, _map
from .geometric_native_common_carrier_source_work import (
    _payload, _encoded, _digest, _typed_guard, _fraction, _interval, _vector,
    _matrix, _multiply, _transpose, _determinant, _inverse, _outer, _packet,
)

FORMAT = "native-common-whole-cold-material-carrier-1"
RESPONSE_FORMAT = "native-common-whole-cold-material-point-1"
SCOPE = "DECLARED_SMOOTH_WHOLE_OWNER_COLD_MATERIAL_GEOMETRY_AND_SUPPORT_ONLY"
EXTENSION = "QUINTIC_ORIGINAL_QUARTER_SEAM_PARTITION_WITH_FIXED_RADIAL_OWNER_GUARDS"
TRANSPORT_SEAL = "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"
SOURCE_WORK_SEAL = "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"
_ZERO = (I(0),)*3
_ZM = (_ZERO,)*3
_ID = tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))
_REGISTRY, _POINT_GUARDS, _POINT_CACHE = {}, {}, {}


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError("Cold material precision is an integer from 96 to 512")
    return value


def _body(value, seal="binding_sha256"):
    return {f.name:getattr(value,f.name) for f in fields(value) if f.name != seal}


def _up(value, precision):
    return _round(I(value),precision).hi


def _square(value):
    return I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2,value.hi**2),
             max(value.lo**2,value.hi**2))


def _signed_atan(value, precision):
    if value.lo >= 0:
        return interval_atan(value,precision)
    if value.hi <= 0:
        return -interval_atan(-value,precision)
    return I(-interval_atan(I(-value.lo),precision).hi,
             interval_atan(I(value.hi),precision).hi)


def _quintic_partition(angle_bounds, half_width):
    """Positive-sign partition and its derivative with respect to angle."""
    if (type(angle_bounds) is not I or type(half_width) is not F
            or not 0 < half_width < F(1,4)):
        raise ValueError("A seam partition requires exact bounds and a small exact width")
    if angle_bounds.hi <= -half_width:
        return I(0),I(0)
    if angle_bounds.lo >= half_width:
        return I(1),I(0)
    if angle_bounds.lo < -half_width or angle_bounds.hi > half_width:
        raise ValueError("Split a seam angle interval at its saturation boundaries")
    t = (angle_bounds+half_width)/(2*half_width)
    def value(t):
        return t**3*(10+t*(-15+6*t))
    weight = I(value(t.lo),value(t.hi))
    # S'=30t²(1−t)² is nonnegative. Its maximum is15/8 at t=1/2.
    candidates = (t.lo,t.hi)+((F(1,2),) if t.lo <= F(1,2) <= t.hi else ())
    derivatives = tuple(30*x*x*(1-x)*(1-x)/(2*half_width) for x in candidates)
    return weight,I(min(derivatives[:2]),max(derivatives))


def _roof_chart_correction(q, original_radius, old_radius, precision=96):
    """D(r_old q)/D(r0 q), retaining the nonzero authenticated bank roof."""
    _precision(precision)
    if (type(q) is not tuple or len(q) != 3 or any(type(x) is not I for x in q)
            or type(original_radius) is not I or type(old_radius) is not I
            or min(original_radius.lo,old_radius.lo) <= 0):
        raise ValueError("The roof chart correction requires a complete exact Cartesian chart")
    tangential,radial = old_radius/original_radius,original_radius*original_radius/(old_radius*old_radius)
    return tuple(tuple(_round(tangential*int(i==j)+(radial-tangential)*q[i]*q[j],precision)
                       for j in range(3)) for i in range(3))


@dataclass(frozen=True, slots=True)
class NativeWholeColdBank:
    original_parent_id: str
    component_id: str
    original_material_id: str
    owner: int
    radius_m: F
    areal_material_thickness_m: F
    roof_radial_displacement_m: F
    beta_cut_radians: F
    original_quarter_start: tuple
    original_quarter_end: tuple
    original_quarter_inward: tuple
    material_owner_pose_bounds: tuple
    angular_rate_per_physical_volume_depth_m: tuple
    coordinate_order: tuple
    coordinates_m: tuple
    source_context_json: str


@dataclass(frozen=True, slots=True)
class NativeWholeColdOwnerCertificate:
    owner: int
    parent_ids: tuple
    maximum_displacement_m: F
    maximum_bank_original_gradient_norm: F
    maximum_blended_original_gradient_norm: F
    archive_carrier_gradient_deviation_bound: F
    fixed_radial_scale: F
    total_gradient_deviation_bound: F
    local_minimum_singular_value: F
    global_colipschitz_margin: F
    volume_jacobian_bounds: I
    global_injectivity_certified: bool


@dataclass(frozen=True, slots=True)
class NativeWholeColdMaterialCertificate:
    original_parent_count: int
    nonempty_archive_cohort_count: int
    original_parent_ids: tuple
    complete_archive_domain_ids: tuple
    seam_half_width_radians: F
    original_radius_lower_bound_m: F
    donor: NativeWholeColdOwnerCertificate
    receiver: NativeWholeColdOwnerCertificate
    donor_radial_upper_bound_m: F
    receiver_radial_lower_bound_m: F
    cold_owner_radial_gap_lower_bound_m: F
    within_owner_parent_corner_gluing_certified: bool
    whole_owner_material_injectivity_certified: bool
    whole_cold_cross_owner_support_separation_certified: bool
    complete_live_and_archive_material_retained: bool
    whole_clock_all_pair_admission_authorized: bool = False
    physical_material_admission_authorized: bool = False
    original_point_1_completed: bool = False


@dataclass(frozen=True, slots=True)
class NativeCommonColdMaterialCarrier:
    source_packet_seals: tuple
    archive_carrier_model_sha256: str
    banks: tuple
    coordinate_order: tuple
    bound_coordinate_increments: tuple
    coordinate_envelope_abs_m: tuple
    whole_domain_certificate: NativeWholeColdMaterialCertificate
    source_geometry_elapsed_time_s: F
    contact_geometry_elapsed_time_s: F
    memory_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    provenance: str
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    extension_law: str = EXTENSION
    radial_support_guard: bool = True
    candidate_only: bool = True
    physical_source_force_prestress_work_handover_certified: bool = False
    hot_constitutive_law_defined: bool = False
    finite_mean_chart_defined: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    full_system_temporal_path_work_certified: bool = False
    source_state_or_work_advanced: bool = False
    accepted_original_native_intervals: int = 1
    accepted_new_native_intervals: int = 0


@dataclass(frozen=True, slots=True)
class NativeWholeColdBankField:
    original_parent_id: str
    component_id: str
    original_radius_bounds_m: I
    old_radius_bounds_m: I
    physical_volume_depth_bounds_m: I
    material_beta_bounds_radians: I
    source_shape_bounds: I
    source_shape_beta_derivative_bounds: I
    field_active_inside_material_clamp: bool
    reference_position_bounds_m: tuple
    reference_original_gradient_bounds: tuple
    old_common_gradient_bounds: tuple
    old_material_owner_pose_bounds: tuple
    roof_chart_correction_bounds: tuple
    mode_displacement_columns: tuple
    mode_spatial_gradient_columns_per_m: tuple
    mode_original_gradient_columns_per_m: tuple
    displacement_bounds_m: tuple
    displacement_original_gradient_bounds: tuple


@dataclass(frozen=True, slots=True)
class NativeCommonColdMaterialPoint:
    system_binding_sha256: str
    original_parent_id: str
    original_material_id: str
    owner: int
    original_ray: tuple
    material_depth_bounds_m: I
    time_fraction: F
    allocation: str
    cohort_id: str | None
    archive_debits: int
    original_reference_radius_bounds_m: I
    archive_carrier_position_bounds_m: tuple
    fixed_radial_scale: F
    displacement_bounds_m: tuple
    position_bounds_m: tuple
    original_cartesian_gradient_bounds: tuple
    volume_jacobian_bounds: I
    density_pullback_multiplier_bounds: I
    seam_weights: tuple
    seam_weight_original_gradients: tuple
    active_bank_fields: tuple
    coordinate_order: tuple
    bound_coordinate_increments: tuple
    mode_displacement_columns: tuple
    mode_original_gradient_columns_per_m: tuple
    source_archive_point_json: str
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    material_membership_authenticated: bool = True
    physical_material_admission_authorized: bool = False
    accepted_new_native_intervals: int = 0


def _certificate(banks, carrier, delta, guarded, precision):
    R=carrier.radius_m
    heights=dict(carrier.whole_domain_certificate.original_parent_thicknesses_m)
    r0lo=interval_cbrt(I(R**3-3*R**2*max(heights.values())),precision).lo
    if r0lo <= 0:
        raise ValueError("The complete original cold shell has no positive radial chart")
    owners=[]
    gamma=carrier.whole_domain_certificate.gamma
    bmax=gamma/24
    h=interval_sqrt(I(1-4*gamma*gamma),precision).lo
    k=carrier.dip_tangent_bounds.hi
    me=(1+2*gamma)/h+2*gamma/(h*h)
    donor_D=_up((2*k+1+me)*bmax+bmax*bmax/2+interval_sqrt(I(1+k*k),precision).hi/6,precision)
    ratio=R/r0lo
    receiver_D=_up(max(ratio-1,1-1/(ratio*ratio)),precision)
    for owner,D in ((carrier.donor_owner,donor_D),(carrier.receiver_owner,receiver_D)):
        rows=tuple(b for b in banks if b.owner==owner)
        if len(rows)!=4:
            raise ValueError("Each authentic owner retains all four original parent fields")
        Us,Ls=[],[]
        for bank in rows:
            top=R+bank.roof_radial_displacement_m
            roldlo=interval_cbrt(I(top**3-3*R**2*bank.areal_material_thickness_m),precision).lo
            if roldlo<=0 or not 0<bank.beta_cut_radians<=F(1,4):
                raise ValueError("The entire cold bank must retain its positive shell and source clamp")
            a0=2*abs(bank.coordinates_m[0]); S=2*sum(map(abs,bank.coordinates_m[1:]),F(0))
            A=sum(map(abs,bank.angular_rate_per_physical_volume_depth_m),F(0))
            U=_up(a0+top/R*S,precision)
            C=max(top/r0lo,(R/roldlo)**2)
            B=1+top**3*A/R**2
            L=_up(a0*(1/r0lo+A)+S/R*B*C+3*U/(2*bank.beta_cut_radians*r0lo),precision)
            Us.append(U); Ls.append(L)
        U,L=max(Us),max(Ls)
        b=max(b.beta_cut_radians for b in rows)
        cosine=_sin_cos(I(b),precision)[1].lo
        blended=_up(L+2*U*F(15,16)/(delta*r0lo*cosine),precision)
        scale=1+(-1 if owner==carrier.donor_owner else 1)*2*U/R if guarded else F(1)
        if scale<=0:
            raise ValueError("The fixed whole-owner radial guard has a nonpositive scale")
        total=_up(abs(scale-1)+scale*D+blended,precision)
        sigma=1-total
        margin=1-2*total
        if sigma<=0 or margin<=0:
            raise ValueError("The actual complete cold fields exceed the proved whole-owner injectivity budget")
        owners.append(NativeWholeColdOwnerCertificate(owner,tuple(sorted(b.original_parent_id for b in rows)),
            U,L,blended,D,scale,total,sigma,margin,
            I(_round(I(sigma**3),precision).lo,_round(I((1+total)**3),precision).hi),True))
    donor,receiver=owners
    upper=donor.fixed_radial_scale*R+donor.maximum_displacement_m
    lower=receiver.fixed_radial_scale*R-receiver.maximum_displacement_m
    gap=lower-upper
    return NativeWholeColdMaterialCertificate(8,3,tuple(sorted(heights)),
        carrier.whole_domain_certificate.complete_archive_domain_ids,delta,r0lo,donor,receiver,
        upper,lower,gap,True,True,guarded and gap>0,True)


def bind_native_common_cold_material_carrier(material_transport_packet, source_work_packet,
        archive_carrier, *, provenance, precision=96,
        seam_half_width_radians=F(1,100), radial_support_guard=True):
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Declare the new whole-owner seam extension and fixed radial guards")
    if (type(seam_half_width_radians) is not F or seam_half_width_radians!=F(1,100)
            or type(radial_support_guard) is not bool):
        raise ValueError("The proved seam law retains its exact declared width and literal guard flag")
    if type(archive_carrier) is not NativeArchiveMaterialCarrier:
        raise ValueError("A whole cold field requires the actual authenticated archive carrier")
    native_archive_material_carrier_to_dict(archive_carrier)
    _packet(material_transport_packet,TRANSPORT_SEAL); _packet(source_work_packet,SOURCE_WORK_SEAL)
    oldsys=source_work_packet['carrier_source_work_system']
    seed=json.loads(oldsys['seed_json'])
    state=json.loads(archive_carrier.source_material_state_json)
    other=json.loads(material_transport_packet['material_transfer']['complete_original_material_state_json'])
    oldstate=json.loads(seed['local']['complete_original_source_material_state_json'])
    canonical=lambda x: json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    if (canonical(state)!=canonical(other) or canonical(state)!=canonical(oldstate)
            or oldsys['archive_carrier_model_sha256']!=archive_carrier.model_sha256):
        raise ValueError("The complete 879/891 source state and actual archive carrier differ")
    bodymap={c['mathematical_body_view']['component_id']:c for c in material_transport_packet['original_cold_body_contexts']}
    if len(bodymap)!=8:
        raise ValueError("All eight authentic whole cold body contexts must occur exactly once")
    oldorder=tuple(tuple(row) for row in oldsys['coordinate_order'])
    oldcoordinates=tuple(_fraction(v) for v in oldsys['bound_coordinate_increments'])
    if len(oldorder)!=100 or len(oldcoordinates)!=100:
        raise ValueError("The prior source preparation retains its complete hundred-coordinate context")
    coldrows=tuple((row,value) for row,value in zip(oldorder,oldcoordinates)
                   if row[0]=='body' and row[1] in bodymap)
    order=tuple(row for row,_ in coldrows); values=tuple(value for _,value in coldrows)
    if len(order)!=56 or any(value==0 for value in values):
        raise ValueError("This declared continuation retains all 56 nonzero registered cold amplitudes")
    queries={}
    for item in material_transport_packet['original_material_queries']:
        r=item['response']; cid=r['geometry_field_component_id']
        if cid in queries and queries[cid]!=r['coordinates_m']:
            raise ValueError("A cold bank changes its registered coefficients across original source labels")
        queries[cid]=r['coordinates_m']
    if set(queries)!=set(bodymap):
        raise ValueError("Every complete cold bank needs its authentic saved nonzero field query")
    banks=[]
    for cid,c in bodymap.items():
        b=c['mathematical_body_view']; eb=c['embedding_bank']; orig=c['original_body']
        for n in ('component_id','original_parent_id','owner','radius_m','areal_material_thickness_m','roof_radial_displacement_m'):
            if b[n]!=eb[n] or b[n]!=orig[n]:
                raise ValueError("A complete bank context changes an authentic parent/material scalar")
        if b['angular_rate_per_physical_volume_depth_m']!=eb['angular_rate_per_physical_volume_depth_m']:
            raise ValueError("Use the actual COMMON depth flow, never the historical bank rate")
        coords=tuple(v for row,v in coldrows if row[1]==cid)
        if tuple(_fraction(v) for v in queries[cid])!=coords or len(coords)!=7:
            raise ValueError("The complete whole bank differs from the exact 891 registered cold values")
        banks.append(NativeWholeColdBank(b['original_parent_id'],cid,b['material_id'],b['owner'],
            *(_fraction(b[n]) for n in ('radius_m','areal_material_thickness_m','roof_radial_displacement_m','beta_cut_radians')),
            *(tuple(_fraction(v) for v in b[n]) for n in ('original_quarter_start','original_quarter_end','original_quarter_inward')),
            _matrix(b['owner_rotation_matrix_bounds']),
            tuple(_fraction(v) for v in eb['angular_rate_per_physical_volume_depth_m']),
            tuple(b['coordinate_order']),coords,canonical(c)))
    banks=tuple(sorted(banks,key=lambda b:b.original_parent_id))
    if tuple(b.original_parent_id for b in banks)!=tuple('octant:'+str(i) for i in range(8)):
        raise ValueError("The whole cold continuation retains all eight original parent identities")
    heights=dict(archive_carrier.whole_domain_certificate.original_parent_thicknesses_m)
    for bank in banks:
        if bank.radius_m!=archive_carrier.radius_m or bank.areal_material_thickness_m!=heights[bank.original_parent_id]:
            raise ValueError("The entire material shell must match the authentic full archive carrier")
    for owner in (archive_carrier.donor_owner,archive_carrier.receiver_owner):
        rows=tuple(b for b in banks if b.owner==owner)
        if (len({b.original_quarter_inward for b in rows})!=1
                or len({b.beta_cut_radians for b in rows})!=1
                or len({b.material_owner_pose_bounds for b in rows})!=1
                or len({b.angular_rate_per_physical_volume_depth_m for b in rows})!=1
                or len({b.areal_material_thickness_m for b in rows})!=1):
            raise ValueError("The smooth owner field requires the authentic common inward axis, pose, depth flow and clamp")
        if rows[0].original_quarter_inward != ((F(1),F(0),F(0)) if owner==archive_carrier.donor_owner else (F(-1),F(0),F(0))):
            raise ValueError("The full owner hemisphere has another original material normal")
    certificate=_certificate(banks,archive_carrier,seam_half_width_radians,radial_support_guard,precision)
    system=NativeCommonColdMaterialCarrier((TRANSPORT_SEAL,SOURCE_WORK_SEAL),archive_carrier.model_sha256,banks,order,values,tuple(2*abs(v) for v in values),
        certificate,*(_fraction(seed['local'][n]) for n in ('source_geometry_elapsed_time_s','source_contact_elapsed_time_s',
        'source_memory_elapsed_time_s','material_geometry_elapsed_time_s')),provenance,precision,'',radial_support_guard=radial_support_guard)
    system=replace(system,binding_sha256=_digest(_body(system)))
    _REGISTRY[system.binding_sha256]=(_typed_guard(system),archive_carrier)
    return system


def _registered(system):
    if type(system) is not NativeCommonColdMaterialCarrier:
        raise ValueError("A complete typed whole cold carrier is required")
    stored=_REGISTRY.get(system.binding_sha256)
    if stored is None or stored[0]!=_typed_guard(system) or _digest(_body(system))!=system.binding_sha256:
        raise ValueError("An altered, resealed or Python-kind-aliased whole cold context lacks authority")
    native_archive_material_carrier_to_dict(stored[1])
    return stored[1]


def _normalize(ray, precision):
    if type(ray) is not tuple or len(ray)!=3 or any(type(x) is not F for x in ray) or not any(ray):
        raise ValueError("An original cold ray retains three literal Fractions and positive homogeneous norm")
    norm=interval_sqrt(I(sum((x*x for x in ray),F(0))),precision+16)
    return tuple(_round(I(x)/norm,precision) for x in ray)


def _bank_field(system, parent_id, ray, depth, precision=96, *, coordinates_m=None):
    """Pure material-label formula, including the complete original roof chart."""
    _precision(precision)
    bank=next((b for b in system.banks if b.original_parent_id==parent_id),None)
    if bank is None or type(depth) is not I or not 0<=depth.lo<=depth.hi<=bank.areal_material_thickness_m:
        raise ValueError("A bank extension needs its complete authentic full areal-depth range")
    coordinates=bank.coordinates_m if coordinates_m is None else coordinates_m
    if (type(coordinates) is not tuple or len(coordinates)!=7 or any(type(a) is not F for a in coordinates)
            or any(abs(a)>2*abs(b) for a,b in zip(coordinates,bank.coordinates_m))):
        raise ValueError("The whole bank coordinate query lies outside its fixed declared envelope")
    q=_normalize(ray,precision)
    R=bank.radius_m; top=R+bank.roof_radial_displacement_m
    r0=interval_cbrt(I(R**3)-3*R**2*depth,precision)
    rold=interval_cbrt(I(top**3)-3*R**2*depth,precision)
    tau=depth+(R**3-top**3)/(3*R**2)
    Q=bank.material_owner_pose_bounds; v=_map(Q,q)
    e=rotate_joint_physical_depth(v,bank.angular_rate_per_physical_volume_depth_m,tau,precision=precision,order=28)
    cols=tuple(rotate_joint_physical_depth(tuple(I(int(i==j)) for i in range(3)),
        bank.angular_rate_per_physical_volume_depth_m,tau,precision=precision,order=28) for j in range(3))
    depthpose=_transpose(cols)
    beta_label=I(sum((a*x for a,x in zip(bank.original_quarter_inward,ray)),F(0)))/interval_sqrt(
        I(sum((x*x for x in ray),F(0))),precision+16)
    if beta_label.lo<0:
        raise ValueError("An owner field is undefined across the original opposite hemisphere")
    # Avoid the singular atan chart beyond the exact original clamp.
    b=bank.beta_cut_radians; sinb=_sin_cos(I(b),precision)[0]
    inactive=beta_label.lo>=sinb.hi
    h=interval_sqrt(I(sum((ray[i]*ray[i] for i in (1,2)),F(0)))/I(sum((x*x for x in ray),F(0))),precision+16)
    if inactive or h.hi==0:
        beta=I(b,F(2)) if h.hi!=0 else I(F(1),F(2))
        chi=slope=I(0); active=False
    else:
        if h.lo<=0:
            raise ValueError("Split an active original beta chart away from its pole")
        beta=interval_atan(beta_label/h,precision)
        if beta.lo<b<beta.hi:
            raise ValueError("Split a full cold query at its authentic original material clamp")
        active=beta.lo<b
        t=beta/b
        chi=_round((I(1)-t)*(I(1)-t)*(t+2)/2,precision) if active else I(0)
        slope=_round(-F(3,2)*(I(1)-t)*(I(1)+t)/b,precision) if active else I(0)
    X=tuple(_round(rold*x,precision) for x in e)
    axise=_cross(tuple(I(x) for x in bank.angular_rate_per_physical_volume_depth_m),e)
    dq=tuple(tuple((I(int(i==j))-q[i]*q[j])/r0 for j in range(3)) for i in range(3))
    de=_multiply(_multiply(depthpose,Q),dq)
    dtau=tuple(-r0*r0/R**2*x for x in q)
    de=tuple(tuple(_round(de[i][j]+axise[i]*dtau[j],precision) for j in range(3)) for i in range(3))
    drold=tuple(r0*r0/(rold*rold)*x for x in q)
    original=tuple(tuple(_round(e[i]*drold[j]+rold*de[i][j],precision) for j in range(3)) for i in range(3))
    correction=_roof_chart_correction(q,r0,rold,precision)
    av=_cross(tuple(I(x) for x in bank.angular_rate_per_physical_volume_depth_m),v)
    B=_multiply(depthpose,tuple(tuple(I(int(i==j))-rold*rold*rold/R**2*av[i]*v[j]
                                      for j in range(3)) for i in range(3)))
    B=tuple(tuple(_round(x,precision) for x in row) for row in B)
    if active:
        gb=tuple(_round((I(bank.original_quarter_inward[i])-beta_label*q[i])/(r0*h),precision) for i in range(3))
        modes=[tuple(_round(chi*x,precision) for x in e)]
        grads=[tuple(tuple(_round(chi*de[i][j]+slope*e[i]*gb[j],precision) for j in range(3)) for i in range(3))]
        for E in AFFINE_SYMMETRIC_GENERATORS:
            EX=_map(E,X)
            modes.append(tuple(_round(chi*x/R,precision) for x in EX))
            ED=_multiply(tuple(tuple(I(x) for x in row) for row in E),original)
            grads.append(tuple(tuple(_round(chi*ED[i][j]/R+slope*EX[i]/R*gb[j],precision)
                                     for j in range(3)) for i in range(3)))
        inverse=_inverse(_multiply(_multiply(B,Q),correction),precision)
        spatial=tuple(tuple(tuple(_round(x,precision) for x in row) for row in _multiply(g,inverse)) for g in grads)
    else:
        modes=[_ZERO]*7; grads=[_ZM]*7; spatial=(_ZM,)*7
    u=tuple(_round(sum((a*m[i] for a,m in zip(coordinates,modes)),I(0)),precision) for i in range(3))
    du=tuple(tuple(_round(sum((a*g[i][j] for a,g in zip(coordinates,grads)),I(0)),precision)
                   for j in range(3)) for i in range(3))
    return NativeWholeColdBankField(parent_id,bank.component_id,r0,rold,tau,beta,chi,slope,active,X,original,
        B,Q,correction,tuple(modes),spatial,tuple(grads),u,du)


def _seam_axis(ray, numerator, denominator, radius, delta, precision):
    a,b=ray[numerator],abs(ray[denominator])
    if b==0:
        if a==0:
            return I(F(1,2)),_ZERO
        return I(int(a>0)),_ZERO
    angle=_signed_atan(I(a/b),precision)
    weight,derivative=_quintic_partition(angle,delta)
    if derivative.hi==0:
        return weight,_ZERO
    norm=interval_sqrt(I(sum((x*x for x in ray),F(0))),precision+16)
    q=tuple(I(x)/norm for x in ray)
    denom=q[1]*q[1]+q[2]*q[2]
    d=[I(0)]*3
    d[numerator]=I(abs(ray[denominator]))/norm/(radius*denom)
    d[denominator]=-I((1 if ray[denominator]>0 else -1)*ray[numerator])/norm/(radius*denom)
    return weight,tuple(_round(derivative*x,precision) for x in d)


def _seam_partition(system, owner, ray, radius, precision):
    delta=system.whole_domain_certificate.seam_half_width_radians
    wy,gy=_seam_axis(ray,1,2,radius,delta,precision)
    wz,gz=_seam_axis(ray,2,1,radius,delta,precision)
    result=[]
    for bank in system.banks:
        if bank.owner!=owner:
            continue
        signs=tuple(sum((axis[i] for axis in (bank.original_quarter_start,bank.original_quarter_end)),F(0)) for i in (1,2))
        fy=wy if signs[0]>0 else I(1)-wy; fz=wz if signs[1]>0 else I(1)-wz
        dy=gy if signs[0]>0 else tuple(-x for x in gy); dz=gz if signs[1]>0 else tuple(-x for x in gz)
        weight=_round(fy*fz,precision)
        gradient=tuple(_round(dy[i]*fz+fy*dz[i],precision) for i in range(3))
        result.append((bank.original_parent_id,weight,gradient))
    return tuple(result)


def evaluate_native_common_cold_material_point(system, original_parent_id, original_ray,
        material_depth_bounds_m, *, time_fraction=F(1), coordinate_increments=None, precision=96):
    carrier=_registered(system); _precision(precision)
    if type(time_fraction) is not F or time_fraction!=1:
        raise ValueError("This whole cold support theorem covers the declared endpoint only")
    coordinates=system.bound_coordinate_increments if coordinate_increments is None else coordinate_increments
    if (type(coordinates) is not tuple or len(coordinates)!=56
            or any(type(x) is not F for x in coordinates)
            or any(abs(x)>bound for x,bound in zip(coordinates,system.coordinate_envelope_abs_m))):
        raise ValueError("The cold query lies outside its fixed 56-coordinate support envelope")
    bank=next((b for b in system.banks if b.original_parent_id==original_parent_id),None)
    if bank is None:
        raise ValueError("The original material parent does not belong to this whole cold carrier")
    _normalize(original_ray,precision)
    if any(sum((a*x for a,x in zip(axis,original_ray)),F(0))<0 for axis in (
            bank.original_quarter_start,bank.original_quarter_end,bank.original_quarter_inward)):
        raise ValueError("The complete ray lies outside its authentic original parent support")
    if type(material_depth_bounds_m) is not I or not 0<=material_depth_bounds_m.lo<=material_depth_bounds_m.hi<=bank.areal_material_thickness_m:
        raise ValueError("The complete exact query depth lies outside the entire authentic material shell")
    key=(system.binding_sha256,original_parent_id,original_ray,material_depth_bounds_m,coordinates,precision)
    if key in _POINT_CACHE:
        result=_POINT_CACHE[key]; native_common_cold_material_point_to_dict(result); return result
    point=evaluate_native_archive_material_point(carrier,original_parent_id,original_ray,material_depth_bounds_m,F(1),precision=precision)
    if point.material_id!=bank.original_material_id or point.owner!=bank.owner:
        raise ValueError("The actual archive carrier changes an authentic original material identity")
    certificate=system.whole_domain_certificate.donor if bank.owner==carrier.donor_owner else system.whole_domain_certificate.receiver
    # All cold fields vanish at the original owner pole, so a singular seam
    # chart is never required there; the archive map and complete material remain.
    if original_ray[1]==0 and original_ray[2]==0:
        partitions=tuple((b.original_parent_id,I(F(1,4)),_ZERO) for b in system.banks if b.owner==bank.owner)
    else:
        partitions=_seam_partition(system,bank.owner,original_ray,point.reference_radius_bounds_m,precision)
    fields_by_parent={}
    modes=[_ZERO]*56; gradients=[_ZM]*56
    index={cid:tuple(i for i,row in enumerate(system.coordinate_order) if row[1]==cid) for cid in {b.component_id for b in system.banks}}
    for parent,weight,dweight in partitions:
        if weight.hi==0 and all(x.hi==x.lo==0 for x in dweight):
            continue
        component=next(b.component_id for b in system.banks if b.original_parent_id==parent)
        field=_bank_field(system,parent,original_ray,material_depth_bounds_m,precision,
            coordinates_m=tuple(coordinates[i] for i in index[component]))
        fields_by_parent[parent]=field
        for k,globalindex in enumerate(index[field.component_id]):
            psi=field.mode_displacement_columns[k]; G=field.mode_original_gradient_columns_per_m[k]
            modes[globalindex]=tuple(_round(weight*x,precision) for x in psi)
            gradients[globalindex]=tuple(tuple(_round(weight*G[i][j]+psi[i]*dweight[j],precision)
                                                  for j in range(3)) for i in range(3))
    u=tuple(_round(sum((a*v[i] for a,v in zip(coordinates,modes)),I(0)),precision) for i in range(3))
    du=tuple(tuple(_round(sum((a*g[i][j] for a,g in zip(coordinates,gradients)),I(0)),precision)
                   for j in range(3)) for i in range(3))
    s=certificate.fixed_radial_scale
    position=tuple(_round(s*x+y,precision) for x,y in zip(point.position_bounds_m,u))
    gradient=tuple(tuple(_round(s*point.deformation_gradient_bounds[i][j]+du[i][j],precision)
                        for j in range(3)) for i in range(3))
    direct=_round(_determinant(gradient),precision); proved=certificate.volume_jacobian_bounds
    if direct.hi<proved.lo or direct.lo>proved.hi:
        raise ValueError("The actual local derivative contradicts the independent whole-owner Jacobian theorem")
    jac=I(max(direct.lo,proved.lo),min(direct.hi,proved.hi))
    if jac.lo<=0:
        raise ValueError("The complete actual cold original Cartesian Jacobian is not positive")
    result=NativeCommonColdMaterialPoint(system.binding_sha256,original_parent_id,bank.original_material_id,
        bank.owner,original_ray,material_depth_bounds_m,F(1),point.allocation,point.cohort_id,point.archive_debits,
        point.reference_radius_bounds_m,point.position_bounds_m,s,u,position,gradient,jac,_round(1/jac,precision),
        tuple((p,w) for p,w,_ in partitions),tuple((p,g) for p,_,g in partitions),
        tuple(fields_by_parent[p] for p in sorted(fields_by_parent)),system.coordinate_order,coordinates,
        tuple(modes),tuple(gradients),json.dumps(native_archive_material_point_to_dict(point,system=carrier),
            sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False),precision,'')
    result=replace(result,binding_sha256=_digest(_body(result)))
    _POINT_GUARDS[result.binding_sha256]=_typed_guard(result); _POINT_CACHE[key]=result
    return result


def native_common_cold_material_carrier_to_dict(system):
    _registered(system)
    return dict(format=FORMAT,**_payload(system))


def _saved_json_kinds(value):
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("A saved DTO retains literal string keys")
        for item in value.values():
            _saved_json_kinds(item)
    elif type(value) is list:
        for item in value:
            _saved_json_kinds(item)
    elif type(value) not in (str,int,bool) and value is not None:
        raise ValueError("A saved cold DTO retains its exact JSON scalar and container kinds")


def native_common_cold_material_carrier_from_dict(raw, *, material_transport_packet,
        source_work_packet, archive_carrier):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get('format')!=FORMAT:
        raise ValueError("A complete declared whole cold carrier DTO is required")
    rebuilt=bind_native_common_cold_material_carrier(material_transport_packet,source_work_packet,archive_carrier,
        provenance=raw.get('provenance'),precision=raw.get('precision'),
        seam_half_width_radians=_fraction(raw['whole_domain_certificate']['seam_half_width_radians']),
        radial_support_guard=raw.get('radial_support_guard'))
    if _encoded(raw)!=_encoded(native_common_cold_material_carrier_to_dict(rebuilt)):
        raise ValueError("The whole cold restart changes a complete authentic field or query context")
    return rebuilt


def native_common_cold_material_point_to_dict(point):
    if (type(point) is not NativeCommonColdMaterialPoint
            or _digest(_body(point))!=point.binding_sha256
            or _POINT_GUARDS.get(point.binding_sha256)!=_typed_guard(point)):
        raise ValueError("An altered, resealed or typed-aliased cold material point lacks authority")
    return dict(format=RESPONSE_FORMAT,**_payload(point))


def native_common_cold_material_point_from_dict(raw, *, system, expected_original_parent_id,
        expected_original_ray, expected_material_depth_bounds_m, expected_time_fraction=F(1),
        expected_coordinate_increments=None, precision=96):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get('format')!=RESPONSE_FORMAT:
        raise ValueError("A complete whole cold material point DTO is required")
    rebuilt=evaluate_native_common_cold_material_point(system,expected_original_parent_id,
        expected_original_ray,expected_material_depth_bounds_m,time_fraction=expected_time_fraction,
        coordinate_increments=expected_coordinate_increments,precision=precision)
    if _encoded(raw)!=_encoded(native_common_cold_material_point_to_dict(rebuilt)):
        raise ValueError("A complete cold point restart was altered or transplanted")
    return rebuilt

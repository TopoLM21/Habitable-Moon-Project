"""Whole original-material cold continuum on the prepared 895 reference.

The reference is ``s*chi_887`` at zero cold increment.  All derivatives are
with respect to the same 56 material amplitudes used by 895.  The eight old
beams calibrate one global strain projection, once each.  Their initial
strain span is a NEW constitutive declaration; archived prestress and the
work needed to prepare this reference are not recovered by this law.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json
import math

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_common_carrier_source_work import (
    _payload, _encoded, _digest, _typed_guard, _fraction, _matrix, _inverse,
    _multiply, _transpose, _round, _packet,
)
from .geometric_native_common_cold_material_carrier import (
    NativeCommonColdMaterialCarrier, native_common_cold_material_carrier_to_dict,
    _registered as _registered_cold, _bank_field, _seam_partition, _normalize,
    _saved_json_kinds, TRANSPORT_SEAL, SOURCE_WORK_SEAL,
)
from .geometric_native_archive_material_carrier import (
    _donor_geometry, _receiving_geometry, _rotation,
)
from .geometric_native_affine_body import AFFINE_SYMMETRIC_GENERATORS

FORMAT = "native-common-cold-continuum-energy-system-1"
RESPONSE_FORMAT = "native-common-cold-continuum-energy-response-1"
POINT_FORMAT = "native-common-cold-continuum-strain-point-1"
CARRIER_PACKET_SEAL = "27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70"
COLD_TENSOR_PACKET_SEAL = "8d1744eec08c28b3c3e15593688b812c4aca50677322c3d10db6caf381920a13"
MODEL = "NEW_LAYERED_WHOLE_MATERIAL_GLOBAL_EIGHT_TIP_GRAM_CALIBRATION"
MEASURE = "AUTHENTIC_ORIGINAL_AREAL_VOLUME_R_SQUARED_DOMEGA_DZ_ONCE"
INITIAL_STRAIN_MODEL = "NEW_DECLARED_SUM_WREF_J_HJ_IN_GLOBAL_EIGHT_TIP_SPAN"
SCOPE = "WHOLE_COLD_CONTINUUM_CONSTITUTIVE_ENDPOINT_COMPONENT_ONLY"
_SYSTEMS, _RESPONSES, _POINTS, _INTEGRALS, _INTEGRAL_CACHE = {}, {}, {}, {}, {}
_ZERO = (I(0),)*3
_ZM = (_ZERO,)*3


def _body(value):
    return {f.name:getattr(value,f.name) for f in fields(value) if f.name != "binding_sha256"}


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError("Continuum precision is a literal integer from 96 to 512")
    return value


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def _coordinates(system, value):
    value = system.bound_coordinate_increments if value is None else value
    if (type(value) is not tuple or len(value)!=56 or any(type(x) is not F for x in value)
            or any(abs(x)>bound for x,bound in zip(value,system.coordinate_envelope_abs_m))):
        raise ValueError("The continuum requires all 56 literal metre amplitudes inside the fixed 895 box")
    return value


def _sym(matrix):
    return tuple(tuple((matrix[i][j]+matrix[j][i])/2 for j in range(3)) for i in range(3))


def _mm(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(len(second))),0)
                       for j in range(len(second[0]))) for i in range(len(first)))


def _tr(matrix):
    return tuple(zip(*matrix))


def _mv(matrix, vector):
    return tuple(sum((a*b for a,b in zip(row,vector)),0) for row in matrix)


def _square(value):
    if type(value) is not I:
        return value*value
    return I(F(0) if value.lo<=0<=value.hi else min(value.lo**2,value.hi**2),
             max(value.lo**2,value.hi**2))


def _green_twojet(relative_gradient, modal_columns):
    """Exact Green strain and its full amplitude twojet, in any scalar algebra.

    ``relative_gradient`` is Du*(s*A_887)^-1, not the total F.  The modal
    columns use the same pullback.  The second derivative is independent of
    the amplitudes and is retained in the prestress Hessian contraction.
    """
    if (type(relative_gradient) is not tuple or len(relative_gradient)!=3
            or any(type(row) is not tuple or len(row)!=3 for row in relative_gradient)
            or type(modal_columns) is not tuple
            or any(type(g) is not tuple or len(g)!=3 or any(type(row) is not tuple or len(row)!=3 for row in g)
                   for g in modal_columns)):
        raise ValueError("Green mechanics requires immutable three-dimensional matrix columns")
    D=relative_gradient
    quadratic=_mm(_tr(D),D)
    symmetric=_sym(D)
    strain=tuple(tuple(symmetric[i][j]+quadratic[i][j]/2 for j in range(3)) for i in range(3))
    first=[]
    for column in modal_columns:
        linear=_sym(column); product=_sym(_mm(_tr(D),column))
        first.append(tuple(tuple(linear[i][j]+product[i][j] for j in range(3)) for i in range(3)))
    second=tuple(tuple(_sym(_mm(_tr(a),b)) for b in modal_columns) for a in modal_columns)
    return strain,tuple(first),second


def _tensor_inner(first, second, lame_pa, shear_pa):
    trace=sum((first[i][i] for i in range(3)),0)
    othertrace=trace if first is second else sum((second[i][i] for i in range(3)),0)
    if first is second and type(trace) is _BoxScalar:
        return lame_pa*(trace**2)+2*shear_pa*(sum((first[i][i]**2 for i in range(3)),0)
            +2*sum((first[i][j]**2 for i in range(3) for j in range(i+1,3)),0))
    return (lame_pa*trace*othertrace
            +2*shear_pa*(sum((first[i][i]*second[i][i] for i in range(3)),0)
                +2*sum((first[i][j]*second[i][j] for i in range(3) for j in range(i+1,3)),0)))


@dataclass(frozen=True,slots=True)
class NativeContinuumColdBank:
    original_parent_id: str
    component_id: str
    original_material_id: str
    owner: int
    radius_m: F
    areal_material_thickness_m: F
    elastic_layer_areal_depth_m: F
    upper_young_modulus_pa: F
    upper_poisson_ratio: F
    lower_young_modulus_pa: F
    lower_shear_modulus_pa: F
    source_old_tip_stiffness_n_m: F
    accepted_tip_displacement_m: F
    source_reference_energy_j: F
    source_reference_force_n: F
    tip_coordinate_index: int
    coordinate_indices: tuple
    complete_original_body_context_json: str
    complete_source_beam_json: str
    complete_original_cold_strain_material_json: str
    lower_tensor_extension_provenance: str

    def lame_coefficients(self,layer):
        if layer=="upper":
            E,nu=self.upper_young_modulus_pa,self.upper_poisson_ratio
            return E*nu/((1+nu)*(1-2*nu)),E/(2*(1+nu))
        if layer=="lower":
            E,G=self.lower_young_modulus_pa,self.lower_shear_modulus_pa
            nu=E/(2*G)-1
            return 2*G*nu/(1-2*nu),G
        raise ValueError("The continuum tensor requires an authentic upper or declared lower layer")


@dataclass(frozen=True,slots=True)
class NativeCommonColdContinuumEnergySystem:
    source_packet_seals: tuple
    cold_material_carrier_binding_sha256: str
    archive_carrier_model_sha256: str
    normal_support_source_snapshot_sha256: str
    normal_support_source_support_sha256: str
    complete_original_material_state_json: str
    complete_source_clock_work_history_json: str
    complete_source_carrier_certificate_json: str
    original_cold_tensor_material_snapshot_sha256s: tuple
    complete_lower_tensor_templates_json: str
    whole_owner_elastic_metric_certificates: tuple
    banks: tuple
    coordinate_order: tuple
    bound_coordinate_increments: tuple
    coordinate_envelope_abs_m: tuple
    source_geometry_elapsed_time_s: F
    contact_geometry_elapsed_time_s: F
    memory_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    provenance: str
    precision: int
    binding_sha256: str
    material_model: str = MODEL
    material_reference_measure: str = MEASURE
    initial_strain_model: str = INITIAL_STRAIN_MODEL
    scope: str = SCOPE
    original_parent_count: int = 8
    nonempty_archive_cohort_count: int = 3
    source_beam_count: int = 8
    source_beams_included_once: bool = True
    registered_old_bulk_remainders_used_as_continuum_integral: bool = False
    historical_full_three_dimensional_prestress_recovered: bool = False
    historical_archived_strain_energy_density_recovered: bool = False
    new_reference_preparation_work_certified: bool = False
    physical_source_force_prestress_work_handover_certified: bool = False
    hot_constitutive_law_defined: bool = False
    finite_mean_equilibrium_certified: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    physical_material_admission_authorized: bool = False
    accepted_new_native_intervals: int = 0


def bind_native_common_cold_continuum_energy(material_transport_packet, source_work_packet,
        material_carrier_packet, *, cold_carrier, cold_tensor_packet, provenance, precision=96):
    """Bind all constitutive scalars to complete authenticated 879/891/895 data."""
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Declare the whole-volume lower extension and prepared-reference initial strain")
    _packet(material_transport_packet,TRANSPORT_SEAL)
    _packet(source_work_packet,SOURCE_WORK_SEAL)
    _packet(material_carrier_packet,CARRIER_PACKET_SEAL)
    _packet(cold_tensor_packet,COLD_TENSOR_PACKET_SEAL)
    if cold_tensor_packet.get('format')!='native-supported-cold-tensor-source-packet-1':
        raise ValueError("The declared original full cold materials require their authenticated 833 packet")
    if type(cold_carrier) is not NativeCommonColdMaterialCarrier:
        raise ValueError("The continuum requires the authentic typed 895 complete cold carrier")
    carrier_dto=native_common_cold_material_carrier_to_dict(cold_carrier)
    if _json(carrier_dto)!=_json(material_carrier_packet['cold_material_carrier_system']):
        raise ValueError("The supplied 895 whole field differs from the complete authenticated packet")
    archive=_registered_cold(cold_carrier)
    seed=json.loads(source_work_packet['carrier_source_work_system']['seed_json'])
    beams=material_transport_packet['material_transfer']['beams']
    if _json(beams)!=_json(seed['beams']) or len(beams)!=8:
        raise ValueError("Every original source beam must occur exactly once, unchanged from 879 to 891")
    if source_work_packet['carrier_source_work_system']['archive_carrier_model_sha256']!=archive.model_sha256:
        raise ValueError("The source preparation and whole field require the same complete archive carrier")
    oldmaterials={row['component_id']:row for row in cold_tensor_packet['cold_components']}
    if len(oldmaterials)!=8 or len(cold_tensor_packet['cold_components'])!=8:
        raise ValueError("Every complete original 14-field cold material must be authenticated once")
    material_fields=('component_id','body_snapshot_sha256','source_snapshot_sha256','source_support_sha256',
        'upper_young_modulus_pa','upper_poisson_ratio','lower_young_modulus_pa','lower_shear_modulus_pa',
        'source_lower_tensor_template_json','material_model','initial_strain_model',
        'lower_layer_extension_provenance','calibration_provenance','material_snapshot_sha256')
    templates=[]; lower=[]; snapshots=[]
    for row in oldmaterials.values():
        m=row['material']
        if (set(m)!=set(material_fields) or m['component_id']!=row['component_id']
                or m['body_snapshot_sha256']!=row['body']['geometry_source_sha256']
                or hashlib.sha256(_json([m[name] for name in material_fields[:-1]]).encode()).hexdigest()!=m['material_snapshot_sha256']):
            raise ValueError("The complete original cold material seal or exact field topology changed")
        templates.append((m['component_id'],json.loads(m['source_lower_tensor_template_json'])))
        lower.append((_fraction(m['lower_young_modulus_pa']),_fraction(m['lower_shear_modulus_pa'])))
        snapshots.append((m['source_snapshot_sha256'],m['source_support_sha256']))
    if not lower or len(set(lower))!=1:
        raise ValueError("The source lower E/G must define one authentic declared tensor extension")
    lower_E,lower_G=lower[0]; lower_nu=lower_E/(2*lower_G)-1
    if len(set(snapshots))!=1:
        raise ValueError("The original full cold material declarations require one complete source snapshot")
    normal_snapshot,normal_support=snapshots[0]
    if lower_E<=0 or lower_G<=0 or not -1<lower_nu<F(1,2):
        raise ValueError("The declared lower source E/G tensor must be positive definite")
    contexts=material_transport_packet['original_cold_body_contexts']
    contextmap={c['mathematical_body_view']['component_id']:c for c in contexts}
    beammap={b['component_id']:b for b in beams}
    if len(contexts)!=8 or len(contextmap)!=8 or len(beammap)!=8:
        raise ValueError("The authentic full source retains eight distinct original cold parents and beams")
    banks=[]
    for bank in cold_carrier.banks:
        context=contextmap[bank.component_id]; body=context['mathematical_body_view']; beam=beammap[bank.component_id]
        oldmaterial=oldmaterials[bank.component_id]['material']
        if (body['original_parent_id']!=bank.original_parent_id or body['owner']!=bank.owner
                or body['source_snapshot_sha256']!=normal_snapshot
                or body['source_support_sha256']!=normal_support):
            raise ValueError("The complete source material and the actual normal E/G authority differ")
        E,nu=_fraction(body['young_modulus_pa']),_fraction(body['poisson_ratio'])
        if E!=_fraction(oldmaterial['upper_young_modulus_pa']) or nu!=_fraction(oldmaterial['upper_poisson_ratio']):
            raise ValueError("The whole 879 upper material differs from its complete authenticated 833 declaration")
        cut=_fraction(body['elastic_layer_areal_depth_m'])
        K,w=_fraction(beam['stiffness_exact_n_m']),_fraction(beam['reference_displacement_m'])
        if (E<=0 or not -1<nu<F(1,2) or not 0<cut<bank.areal_material_thickness_m or K<=0
                or K!=_fraction(body['source_old_tip_stiffness_n_m'])
                or w!=_fraction(body['accepted_tip_displacement_m'])
                or _fraction(beam['reference_energy_exact_j'])!=K*w*w/2
                or _fraction(beam['reference_force_exact_n'])!=K*w):
            raise ValueError("An authentic beam or exact real layer depth was changed")
        indices=tuple(i for i,row in enumerate(cold_carrier.coordinate_order) if row[1]==bank.component_id)
        if len(indices)!=7:
            raise ValueError("Each authentic bank retains its seven actual cold coordinates")
        banks.append(NativeContinuumColdBank(bank.original_parent_id,bank.component_id,bank.original_material_id,
            bank.owner,bank.radius_m,bank.areal_material_thickness_m,cut,E,nu,lower_E,lower_G,K,w,
            _fraction(beam['literal_reduced_beam_binary_energy_evaluation_j']),
            _fraction(beam['literal_reduced_beam_binary_force_evaluation_n']),
            indices[0],indices,_json(context),_json(beam),_json(oldmaterial),
            "NEW isotropic lower tensor from authenticated contact-source laterally free E/G; full original areal-depth cut"))
    banks=tuple(sorted(banks,key=lambda b:b.original_parent_id))
    metric=[]
    for owner,cert in ((archive.donor_owner,cold_carrier.whole_domain_certificate.donor),
                       (archive.receiver_owner,cold_carrier.whole_domain_certificate.receiver)):
        rows=tuple(b for b in cold_carrier.banks if b.owner==owner)
        rlo=cold_carrier.whole_domain_certificate.original_radius_lower_bound_m
        inversecap=1/(cert.fixed_radial_scale*(1-cert.archive_carrier_gradient_deviation_bound))
        wmax=max(abs(b.accepted_tip_displacement_m) for b in banks if b.owner==owner)
        max_tip=max(1/rlo+sum(map(abs,b.angular_rate_per_physical_volume_depth_m),F(0))
                    +F(3,2)/(b.beta_cut_radians*rlo) for b in rows)
        cosine=min(_sc_trig(I(b.beta_cut_radians),precision)[1].lo for b in rows)
        seam=8*F(15,16)/(cold_carrier.whole_domain_certificate.seam_half_width_radians*rlo*cosine)
        initialcap=_round(I(wmax*(max_tip+seam)*inversecap),precision).hi
        Dcap=_round(I(cert.maximum_blended_original_gradient_norm*inversecap),precision).hi
        initiallower=_round(I(1-2*initialcap),precision).lo
        fulllower=_round(I((1-Dcap)**2-2*initialcap),precision).lo
        if Dcap>=1 or initiallower<=0 or fulllower<=0:
            raise ValueError("The declared NEW initial strain or whole 56-box elastic metric lacks positivity")
        metric.append(dict(owner=owner,initial_strain_operator_norm_upper=initialcap,
            relative_increment_gradient_operator_norm_upper=Dcap,
            prepared_initial_elastic_metric_eigenvalue_lower=initiallower,
            complete_coordinate_box_elastic_metric_eigenvalue_lower=fulllower,
            whole_original_owner_and_depth_covered=True,historical_prestress_recovered=False))
    system=NativeCommonColdContinuumEnergySystem((TRANSPORT_SEAL,SOURCE_WORK_SEAL,CARRIER_PACKET_SEAL,COLD_TENSOR_PACKET_SEAL),
        cold_carrier.binding_sha256,archive.model_sha256,normal_snapshot,normal_support,
        material_transport_packet['material_transfer']['complete_original_material_state_json'],
        _json({k:v for k,v in seed.items() if 'clock' in k or 'history' in k or 'work' in k or 'ledger' in k}),
        _encoded(cold_carrier.whole_domain_certificate).decode(),
        tuple((b.original_parent_id,oldmaterials[b.component_id]['material']['material_snapshot_sha256']) for b in banks),
        _json(templates),tuple(metric),banks,cold_carrier.coordinate_order,
        cold_carrier.bound_coordinate_increments,cold_carrier.coordinate_envelope_abs_m,
        cold_carrier.source_geometry_elapsed_time_s,cold_carrier.contact_geometry_elapsed_time_s,
        cold_carrier.memory_elapsed_time_s,cold_carrier.material_geometry_elapsed_time_s,provenance,precision,'')
    system=replace(system,binding_sha256=_digest(_body(system)))
    _SYSTEMS[system.binding_sha256]=(_typed_guard(system),cold_carrier)
    return system


def _registered(system):
    if type(system) is not NativeCommonColdContinuumEnergySystem:
        raise ValueError("A complete typed source-bound continuum system is required")
    stored=_SYSTEMS.get(system.binding_sha256)
    if stored is None or stored[0]!=_typed_guard(system) or _digest(_body(system))!=system.binding_sha256:
        raise ValueError("An altered, resealed, or typed-aliased continuum system lacks source authority")
    native_common_cold_material_carrier_to_dict(stored[1])
    return stored[1],_registered_cold(stored[1])


@dataclass(frozen=True,slots=True)
class NativeCommonColdContinuumStrainPoint:
    system_binding_sha256: str
    original_parent_id: str
    original_ray: tuple
    material_depth_bounds_m: I
    coordinate_increments: tuple
    fixed_radial_scale: F
    prepared_reference_gradient_bounds: tuple
    prepared_reference_inverse_bounds: tuple
    mode_relative_gradient_columns_per_m: tuple
    relative_gradient_bounds: tuple
    green_strain_bounds: tuple
    green_strain_gradient_per_m: tuple
    green_strain_hessian_per_m2: tuple
    tip_strain_columns_per_m: tuple
    remainder_green_strain_bounds: tuple
    remainder_green_strain_gradient_per_m: tuple
    lame_coefficient_pa: F
    shear_modulus_pa: F
    precision: int
    binding_sha256: str


def bound_native_common_cold_continuum_point(system, original_parent_id, original_ray,
        material_depth_bounds_m, *, coordinate_increments=None, precision=96):
    """Pure label-field receipt; no force solve or material-clock mutation."""
    cold,archive=_registered(system); _precision(precision)
    coords=_coordinates(system,coordinate_increments)
    bank=next((b for b in system.banks if b.original_parent_id==original_parent_id),None)
    oldbank=next((b for b in cold.banks if b.original_parent_id==original_parent_id),None)
    if bank is None or type(material_depth_bounds_m) is not I or not 0<=material_depth_bounds_m.lo<=material_depth_bounds_m.hi<=bank.areal_material_thickness_m:
        raise ValueError("A continuum point retains its authentic entire parent depth")
    q=_normalize(original_ray,precision)
    if any(sum((a*x for a,x in zip(axis,original_ray)),F(0))<0 for axis in (
            oldbank.original_quarter_start,oldbank.original_quarter_end,oldbank.original_quarter_inward)):
        raise ValueError("The original continuum ray lies outside its complete original parent quarter")
    depth=material_depth_bounds_m
    if depth.lo<bank.elastic_layer_areal_depth_m<depth.hi:
        raise ValueError("Split a material interval at its authentic real layer cut")
    if bank.owner==archive.donor_owner:
        _,_,_,A,_,_,r0=_donor_geometry(archive,original_ray,q,depth,F(1),precision,())
        cert=cold.whole_domain_certificate.donor
    else:
        _,_,A,_,_,r0=_receiving_geometry(archive,q,depth,bank.areal_material_thickness_m,F(1),precision)
        cert=cold.whole_domain_certificate.receiver
    base=tuple(tuple(_round(cert.fixed_radial_scale*x,precision) for x in row) for row in A)
    inverse=_inverse(base,precision)
    columns=[_ZM]*56
    partitions=_seam_partition(cold,bank.owner,original_ray,r0,precision) if original_ray[1] or original_ray[2] else ()
    for parent,weight,dweight in partitions:
        if weight.hi==0 and all(x.lo==x.hi==0 for x in dweight):
            continue
        active=next(b for b in system.banks if b.original_parent_id==parent)
        field=_bank_field(cold,parent,original_ray,depth,precision,
            coordinates_m=tuple(coords[i] for i in active.coordinate_indices))
        for k,index in enumerate(active.coordinate_indices):
            psi=field.mode_displacement_columns[k]; G=field.mode_original_gradient_columns_per_m[k]
            gradient=tuple(tuple(_round(weight*G[i][j]+psi[i]*dweight[j],precision) for j in range(3)) for i in range(3))
            columns[index]=tuple(tuple(_round(x,precision) for x in row) for row in _multiply(gradient,inverse))
    columns=tuple(columns)
    D=tuple(tuple(_round(sum((a*g[i][j] for a,g in zip(coords,columns)),I(0)),precision) for j in range(3)) for i in range(3))
    E,Ei,Eij=_green_twojet(D,columns)
    tips=tuple(_sym(columns[b.tip_coordinate_index]) for b in system.banks)
    reference=tuple(tuple(sum((coords[b.tip_coordinate_index]*H[i][j] for b,H in zip(system.banks,tips)),I(0)) for j in range(3)) for i in range(3))
    remainder=tuple(tuple(E[i][j]-reference[i][j] for j in range(3)) for i in range(3))
    tipmap={b.tip_coordinate_index:H for b,H in zip(system.banks,tips)}
    remainder_first=tuple(tuple(tuple(x-tipmap.get(k,_ZM)[i][j] for j,x in enumerate(row)) for i,row in enumerate(strain))
                          for k,strain in enumerate(Ei))
    lam,shear=bank.lame_coefficients('upper' if depth.hi<=bank.elastic_layer_areal_depth_m else 'lower')
    point=NativeCommonColdContinuumStrainPoint(system.binding_sha256,original_parent_id,original_ray,depth,coords,
        cert.fixed_radial_scale,base,inverse,columns,D,E,Ei,Eij,tips,remainder,remainder_first,lam,shear,precision,'')
    point=replace(point,binding_sha256=_digest(_body(point))); _POINTS[point.binding_sha256]=_typed_guard(point)
    return point


@dataclass(frozen=True,slots=True)
class NativeContinuumGramInverseCertificate:
    gram_bounds_n_m: tuple
    inverse_gram_bounds_m_n: tuple
    interval_ldl_positive_pivots_n_m: tuple
    rational_midpoint_preconditioner_m_n: tuple
    preconditioner_residual_infinity_norm_upper: F
    inverse_enclosure_error_bound_m_n: F
    positive_definite_certified: bool = True
    global_eight_tip_overlap_retained: bool = True
    inverse_enclosure_method: str = "INTERVAL_LDL_AND_RATIONAL_NEUMANN_RESIDUAL"
    authentic_interior_gram_coercivity_lower_bounds_n_m: tuple = ()


def _rational_inverse(matrix):
    n=len(matrix)
    rows=[list(row)+[F(int(i==j)) for j in range(n)] for i,row in enumerate(matrix)]
    for k in range(n):
        pivot=next((i for i in range(k,n) if rows[i][k]),None)
        if pivot is None:
            raise ValueError("The actual global tip Gram midpoint is singular")
        rows[k],rows[pivot]=rows[pivot],rows[k]
        divisor=rows[k][k]; rows[k]=[v/divisor for v in rows[k]]
        for i in range(n):
            if i!=k:
                factor=rows[i][k]; rows[i]=[a-factor*b for a,b in zip(rows[i],rows[k])]
    return tuple(tuple(row[n:]) for row in rows)


def _verified_spd_inverse(matrix,precision=96):
    """Interval LDL positivity plus a rational-preconditioned Neumann proof."""
    _precision(precision)
    if (type(matrix) is not tuple or len(matrix)!=8 or any(type(row) is not tuple or len(row)!=8 for row in matrix)
            or any(type(x) is not I for row in matrix for x in row)
            or any(matrix[i][j]!=matrix[j][i] for i in range(8) for j in range(8))):
        raise ValueError("The global calibration requires all symmetric eight-by-eight actual Gram bounds")
    n=8; L=[[I(int(i==j)) for j in range(n)] for i in range(n)]; pivots=[]
    for j in range(n):
        pivot=_round(matrix[j][j]-sum((_square(L[j][k])*pivots[k] for k in range(j)),I(0)),precision)
        if pivot.lo<=0:
            raise ValueError("Refine the actual global Gram enclosure until all LDL pivots prove positive")
        pivots.append(pivot)
        for i in range(j+1,n):
            L[i][j]=_round((matrix[i][j]-sum((L[i][k]*L[j][k]*pivots[k] for k in range(j)),I(0)))/pivot,precision)
    midpoint=tuple(tuple(x.midpoint for x in row) for row in matrix)
    C=_rational_inverse(midpoint)
    residual=tuple(tuple(_round(I(int(i==j))-sum((C[i][k]*matrix[k][j] for k in range(n)),I(0)),precision)
                         for j in range(n)) for i in range(n))
    norm=max(sum((max(abs(x.lo),abs(x.hi)) for x in row),F(0)) for row in residual)
    if norm>=1:
        raise ValueError("Refine the actual Gram until the verified inverse residual contracts")
    Cnorm=max(sum((abs(x) for x in row),F(0)) for row in C)
    error=norm*Cnorm/(1-norm)
    inverse=tuple(tuple(_round(I(C[i][j])+I(-error,error),precision) for j in range(n)) for i in range(n))
    # A structural zero joining disconnected Gram components is exact.  This
    # keeps the two distinct owner volumes from acquiring invented coupling.
    adjacency=[{j for j in range(n) if matrix[i][j].lo or matrix[i][j].hi} for i in range(n)]
    for _ in range(n):
        adjacency=[set().union(*(adjacency[j] for j in row))|row for row in adjacency]
    inverse=tuple(tuple(inverse[i][j] if j in adjacency[i] else I(0) for j in range(n)) for i in range(n))
    return NativeContinuumGramInverseCertificate(matrix,inverse,tuple(pivots),C,norm,error)


def _inverse_gram(matrix,precision=96):
    return _verified_spd_inverse(matrix,precision).inverse_gram_bounds_m_n


def _verified_continuum_gram_inverse(system,raw,precision):
    try:
        return _verified_spd_inverse(raw.gram_bounds_n_m,precision)
    except ValueError:
        # The rectangular hull may contain indefinite matrices because it
        # forgets covariance between Gram entries.  The actual integral keeps
        # that covariance: on each original quarter interior only its own tip
        # is active, while every remaining volume contributes a PSD Gram.
        # Thus the genuine full M is >=diag(delta_j), with delta_j obtained
        # from the actual enclosed interior norm integrals, never a new floor.
        lower=raw.authentic_interior_gram_coercivity_lower_bounds_n_m
        if (len(lower)!=8 or any(type(x) is not F or x<=0 for x in lower)
                or not raw.quadrature_certificate.whole_domain_enclosure_certified
                or not raw.quadrature_certificate.authentic_original_volume_included_once):
            raise ValueError("Refine until the actual whole Gram or each authentic interior norm proves positive")
        M=raw.gram_bounds_n_m
        if (len(M)!=8 or any(len(row)!=8 for row in M)
                or any(type(x) is not I for row in M for x in row)
                or any(M[i][j]!=M[j][i] for i in range(8) for j in range(8))):
            raise ValueError("The complete actual symmetric global Gram is required")
        C=_rational_inverse(tuple(tuple(x.midpoint for x in row) for row in M))
        R=tuple(tuple(_round(I(int(i==j))-sum((M[i][k]*C[k][j] for k in range(8)),I(0)),precision)
                      for j in range(8)) for i in range(8))
        owners=tuple(sorted(set(b.owner for b in system.banks)))
        errors={}; norms=[]
        for owner in owners:
            indices=tuple(i for i,b in enumerate(system.banks) if b.owner==owner)
            if len(indices)!=4 or any(M[i][j].lo or M[i][j].hi for i in indices for j in range(8) if j not in indices):
                raise ValueError("The actual original source owners require two separate four-tip Gram blocks")
            delta=min(lower[i] for i in indices)
            norm=max(sum((max(abs(R[i][j].lo),abs(R[i][j].hi)) for j in indices),F(0)) for i in indices)
            # ||M^-1||infinity <=sqrt(4)||M^-1||2 <=2/delta.
            # M^-1-C=M^-1(I-M C); no contraction assumption is made.
            errors[owner]=2*norm/delta; norms.append(norm)
        entryerrors=[]
        for i in range(8):
            row=[]
            for j in range(8):
                indices=tuple(k for k,b in enumerate(system.banks) if b.owner==system.banks[i].owner)
                norm=sum((max(abs(R[k][j].lo),abs(R[k][j].hi))**2/lower[k] for k in indices),F(0))
                row.append(interval_sqrt(I(norm/lower[i]),precision).hi)
            entryerrors.append(row)
        inverse=[]
        for i in range(8):
            row=[]
            for j in range(8):
                if system.banks[i].owner!=system.banks[j].owner:
                    row.append(I(0)); continue
                error=min(errors[system.banks[i].owner],entryerrors[i][j],entryerrors[j][i])
                enclosed=_round(I(C[i][j])+I(-error,error),precision)
                cap=1/interval_sqrt(I(lower[i]*lower[j]),precision).lo
                # M>=D implies |M^-1_ij|<=1/sqrt(delta_i delta_j).
                # The diagonal additionally lies in [0,1/delta_i].
                lo=max(F(0) if i==j else -cap,enclosed.lo)
                hi=min(1/lower[i] if i==j else cap,enclosed.hi)
                row.append(I(lo,hi))
            inverse.append(tuple(row))
        inverse=tuple(inverse)
        return NativeContinuumGramInverseCertificate(M,inverse,(),C,max(norms),max(errors.values()),
            inverse_enclosure_method="ACTUAL_INTERIOR_GRAM_COERCIVITY_AND_RIGHT_RESIDUAL_NORM",
            authentic_interior_gram_coercivity_lower_bounds_n_m=lower)


@dataclass(frozen=True,slots=True)
class NativeCommonColdContinuumIntegral:
    system_binding_sha256: str
    coordinate_increments: tuple
    gram_bounds_n_m: tuple
    authentic_interior_gram_coercivity_lower_bounds_n_m: tuple
    projection_remainder_n: tuple
    projection_remainder_gradient_n_m: tuple
    projection_hessian_n_m2: tuple
    remainder_strain_inner_product_j: I
    remainder_local_gradient_n: tuple
    remainder_local_hessian_n_m: tuple
    quadrature_certificate: object
    precision: int
    binding_sha256: str


def _register_integral(value):
    if type(value) is not NativeCommonColdContinuumIntegral:
        raise ValueError("A full whole-material integral receipt is required")
    value=replace(value,binding_sha256=_digest(_body(value)))
    _INTEGRALS[value.binding_sha256]=_typed_guard(value)
    return value


def validate_native_common_cold_continuum_integral(value,*,system=None,coordinate_increments=None):
    if (type(value) is not NativeCommonColdContinuumIntegral or _digest(_body(value))!=value.binding_sha256
            or _INTEGRALS.get(value.binding_sha256)!=_typed_guard(value)):
        raise ValueError("An altered or resealed whole continuum integral lacks its actual computation")
    if system is not None:
        _registered(system)
        if value.system_binding_sha256!=system.binding_sha256 or value.coordinate_increments!=_coordinates(system,coordinate_increments):
            raise ValueError("A continuum integral belongs to another complete amplitude or authority context")
    if (len(value.projection_remainder_n)!=8 or len(value.projection_remainder_gradient_n_m)!=8
            or any(len(row)!=56 for row in value.projection_remainder_gradient_n_m)
            or len(value.projection_hessian_n_m2)!=8
            or any(len(block)!=56 or any(len(row)!=56 for row in block) for block in value.projection_hessian_n_m2)
            or len(value.remainder_local_gradient_n)!=56 or len(value.remainder_local_hessian_n_m)!=56
            or any(len(row)!=56 for row in value.remainder_local_hessian_n_m)):
        raise ValueError("Every whole continuum derivative channel must be present")
    return value


@dataclass(frozen=True,slots=True)
class NativeCommonColdContinuumEnergyResponse:
    system_binding_sha256: str
    coordinate_order: tuple
    coordinate_increments: tuple
    integrals: NativeCommonColdContinuumIntegral
    gram_inverse_certificate: NativeContinuumGramInverseCertificate
    old_beam_reference_energy_j: F
    old_beam_reference_tip_forces_n: tuple
    tip_projection_increment_m: tuple
    total_tip_projection_m: tuple
    beam_energy_j: I
    orthogonal_continuum_energy_j: I
    energy_j: I
    gradient_n: tuple
    hessian_n_m: tuple
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    accepted_global_eight_beam_reference_energy_preserved: bool = True
    accepted_global_eight_beam_reference_tip_forces_preserved: bool = True
    old_beam_stiffness_added_as_second_energy: bool = False
    full_fifty_six_cold_gradient_and_hessian_included: bool = True
    nonlinear_prestress_curvature_retained: bool = True
    seam_cross_bank_derivatives_retained: bool = True
    historical_full_three_dimensional_prestress_recovered: bool = False
    new_reference_preparation_work_certified: bool = False
    mixed_force_equilibrium_certified: bool = False
    hot_constitutive_law_defined: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    accepted_new_native_intervals: int = 0


def _assemble_response(system,coords,raw,precision):
    raw=validate_native_common_cold_continuum_integral(raw,system=system,coordinate_increments=coords)
    inversecert=_verified_continuum_gram_inverse(system,raw,precision)
    A=inversecert.inverse_gram_bounds_m_n; r=raw.projection_remainder_n
    pR=tuple(_round(x,precision) for x in _mv(A,r))
    p=tuple(I(coords[b.tip_coordinate_index])+x for b,x in zip(system.banks,pR))
    total=tuple(I(b.accepted_tip_displacement_m)+x for b,x in zip(system.banks,p))
    jacR=_mm(A,raw.projection_remainder_gradient_n_m)
    jac=tuple(tuple(_round(x+int(i==b.tip_coordinate_index),precision) for i,x in enumerate(row)) for b,row in zip(system.banks,jacR))
    forces=tuple(I(b.source_reference_force_n)+b.source_old_tip_stiffness_n_m*x for b,x in zip(system.banks,p))
    inverseforces=_mv(A,forces)
    coefficient=tuple(_round(x-y,precision) for x,y in zip(inverseforces,pR))
    beamenergy=_round(sum((I(b.source_reference_energy_j)+b.source_reference_force_n*x+
        b.source_old_tip_stiffness_n_m*_square(x)/2 for b,x in zip(system.banks,p)),I(0)),precision)
    residual=_round((raw.remainder_strain_inner_product_j-sum((x*y for x,y in zip(r,pR)),I(0)))/2,precision)
    if residual.hi<0:
        raise ValueError("The full integral contradicts the positive orthogonal strain complement")
    residual=I(max(F(0),residual.lo),residual.hi)
    gradient=tuple(_round(raw.remainder_local_gradient_n[i]+sum((forces[k]*jac[k][i]-pR[k]*raw.projection_remainder_gradient_n_m[k][i]
                        for k in range(8)),I(0)),precision) for i in range(56))
    hessian=[]
    for i in range(56):
        row=[]
        for j in range(i,56):
            value=raw.remainder_local_hessian_n_m[i][j]
            value+=sum((system.banks[k].source_old_tip_stiffness_n_m*jac[k][i]*jac[k][j]
                -raw.projection_remainder_gradient_n_m[k][i]*jacR[k][j]
                +coefficient[k]*raw.projection_hessian_n_m2[k][i][j] for k in range(8)),I(0))
            row.append(_round(value,precision))
        hessian.append(row)
    fullh=tuple(tuple(hessian[min(i,j)][abs(j-i)] for j in range(56)) for i in range(56))
    reference=sum((b.source_reference_energy_j for b in system.banks),F(0))
    refforces=tuple(b.source_reference_force_n for b in system.banks)
    energy=_round(beamenergy+residual,precision)
    if not any(coords):
        # R and its tip derivatives are identically zero at the prepared
        # reference.  These source identities do not divide interval copies.
        beamenergy=energy=I(reference); residual=I(0)
        mutable=list(gradient)
        for bank,force in zip(system.banks,refforces):
            mutable[bank.tip_coordinate_index]=I(force)
        gradient=tuple(mutable)
    result=NativeCommonColdContinuumEnergyResponse(system.binding_sha256,system.coordinate_order,coords,raw,inversecert,
        reference,refforces,p,total,beamenergy,residual,energy,gradient,fullh,precision,'')
    result=replace(result,binding_sha256=_digest(_body(result))); _RESPONSES[result.binding_sha256]=_typed_guard(result)
    return result


def evaluate_native_common_cold_continuum_energy(system,coordinate_increments=None,*,integrals=None,precision=96,**quadrature_options):
    _registered(system); _precision(precision); coords=_coordinates(system,coordinate_increments)
    if integrals is None:
        integrals=integrate_native_common_cold_continuum_response(system,coords,precision=precision,**quadrature_options)
    elif quadrature_options:
        raise ValueError("Quadrature options cannot alter an already computed integral context")
    return _assemble_response(system,coords,integrals,precision)


def native_common_cold_continuum_energy_to_dict(system):
    _registered(system)
    return dict(format=FORMAT,**_payload(system))


def native_common_cold_continuum_energy_from_dict(raw,*,material_transport_packet,source_work_packet,
        material_carrier_packet,cold_carrier,cold_tensor_packet):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get('format')!=FORMAT:
        raise ValueError("A complete source-bound continuum system DTO is required")
    rebuilt=bind_native_common_cold_continuum_energy(material_transport_packet,source_work_packet,material_carrier_packet,
        cold_carrier=cold_carrier,cold_tensor_packet=cold_tensor_packet,provenance=raw.get('provenance'),precision=raw.get('precision'))
    if _encoded(raw)!=_encoded(native_common_cold_continuum_energy_to_dict(rebuilt)):
        raise ValueError("The continuum restart changed complete material, source, reference or clock authorities")
    return rebuilt


def native_common_cold_continuum_strain_point_to_dict(point):
    if (type(point) is not NativeCommonColdContinuumStrainPoint or _digest(_body(point))!=point.binding_sha256
            or _POINTS.get(point.binding_sha256)!=_typed_guard(point)):
        raise ValueError("An altered or resealed continuum strain point lacks its actual label-field query")
    return dict(format=POINT_FORMAT,**_payload(point))


def native_common_cold_continuum_energy_response_to_dict(response):
    if (type(response) is not NativeCommonColdContinuumEnergyResponse or _digest(_body(response))!=response.binding_sha256
            or _RESPONSES.get(response.binding_sha256)!=_typed_guard(response)):
        raise ValueError("An altered or resealed whole continuum response lacks its actual full integration")
    validate_native_common_cold_continuum_integral(response.integrals)
    return dict(format=RESPONSE_FORMAT,**_payload(response))


def native_common_cold_continuum_energy_response_from_dict(raw,*,system,expected_coordinate_increments=None,
        precision=96,**quadrature_options):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get('format')!=RESPONSE_FORMAT:
        raise ValueError("A complete whole continuum response DTO is required")
    rebuilt=evaluate_native_common_cold_continuum_energy(system,expected_coordinate_increments,
        precision=precision,**quadrature_options)
    if _encoded(raw)!=_encoded(native_common_cold_continuum_energy_response_to_dict(rebuilt)):
        raise ValueError("A whole continuum response was altered or transplanted across full source authorities")
    return rebuilt


def _angular_outer(first,second):
    return tuple(tuple(a*b for b in second) for a in first)


def _angular_add(first,second):
    return tuple(tuple(a+b for a,b in zip(row,other)) for row,other in zip(first,second))


def _angular_scale(value,matrix):
    return tuple(tuple(x*value for x in row) for row in matrix)


def _angular_cross(first,second):
    return (first[1]*second[2]-first[2]*second[1],first[2]*second[0]-first[0]*second[2],first[0]*second[1]-first[1]*second[0])


def _piece_banks(cold,original_parent_id,tag):
    current=next(b for b in cold.banks if b.original_parent_id==original_parent_id)
    if tag=='interior':
        return (current,),current.original_quarter_start,current.original_quarter_end
    boundary=current.original_quarter_start if tag=='start_seam' else current.original_quarter_end
    neighbors=tuple(b for b in cold.banks if b.owner==current.owner and b.original_parent_id!=original_parent_id
                    and boundary in (b.original_quarter_start,b.original_quarter_end))
    if len(neighbors)!=1:
        raise ValueError("Each authentic quarter edge must have exactly one original same-owner neighbor")
    # Both seam pieces use a local angle in [0,delta].  At the upper edge
    # phi=pi/2-theta interchanges the two quarter axes.  This keeps tiny seam
    # polynomial moments well conditioned without approximating the endpoint.
    start,end=(current.original_quarter_start,current.original_quarter_end) if tag=='start_seam' else (current.original_quarter_end,current.original_quarter_start)
    return (current,neighbors[0]),start,end


def _prepared_modal_fields(system,cold,archive,original_parent_id,tag,beta,depth,angular,*,prepared=True):
    """895 seam fields in a finite theta algebra and a validated beta/depth jet."""
    sourcebanks,start,end=_piece_banks(cold,original_parent_id,tag)
    current=sourcebanks[0]; scalar=angular.scalar; R=current.radius_m
    sinb,cosb=angular.sin(beta),angular.cos(beta)
    st,ct=angular.sine_theta(),angular.cosine_theta()
    inward=current.original_quarter_inward
    q=tuple(inward[i]*sinb+start[i]*cosb*ct+end[i]*cosb*st for i in range(3))
    etheta=tuple(-start[i]*st+end[i]*ct for i in range(3))
    r0=angular.cbrt(R**3-3*R**2*depth)
    # Apply the receiver radial inverse before expanding angular polynomials:
    # (I-qqT)L^-1=(r0/rnew)(I-qqT), qT L^-1=(rnew/r0)^2 qT.
    # This exact projector identity avoids artificial degree-four terms and
    # their interval cancellation; the complete prepared derivative is kept.
    rchart=angular.cbrt(R**3+3*R**2*(current.areal_material_thickness_m-depth)) if prepared and current.owner==archive.receiver_owner else r0
    gb=tuple((inward[i]-sinb*q[i])/(rchart*cosb) for i in range(3))
    dtau=tuple(-rchart*rchart/R**2*x for x in q)
    identity=tuple(tuple(angular.constant(int(i==j)) for j in range(3)) for i in range(3))
    dq=_angular_scale(1/rchart,_angular_add(identity,_angular_scale(-1,_angular_outer(q,q))))
    chi=angular.constant((1-beta/current.beta_cut_radians)*(1-beta/current.beta_cut_radians)*(beta/current.beta_cut_radians+2)/2)
    slope=angular.constant(-F(3,2)*(1-beta/current.beta_cut_radians)*(1+beta/current.beta_cut_radians)/current.beta_cut_radians)
    if type(angular) is _BoxAlgebra:
        t=beta.to_interval()/current.beta_cut_radians
        t=I(max(F(0),t.lo),min(F(1),t.hi))
        shape=lambda x:(1-x)*(1-x)*(x+2)/2
        chi=angular.constant(I(shape(t.hi),shape(t.lo)))
        slope=angular.constant(I(-F(3,2)*(1-t.lo*t.lo)/current.beta_cut_radians,
                                  -F(3,2)*(1-t.hi*t.hi)/current.beta_cut_radians))
    delta=cold.whole_domain_certificate.seam_half_width_radians
    if tag=='interior':
        weights=((angular.constant(1),(angular.constant(0),)*3),)
    else:
        t=(angular.theta()+delta)/(2*delta)
        weight=6*t**5-15*t**4+10*t**3
        derivative=30*t*t*(t-1)*(t-1)/(2*delta)
        if type(angular) is _BoxAlgebra:
            bounds=(angular.theta().to_interval()+delta)/(2*delta)
            bounds=I(max(F(1,2),bounds.lo),min(F(1),bounds.hi))
            shape=lambda x:6*x**5-15*x**4+10*x**3
            derivative_shape=lambda x:30*x*x*(x-1)*(x-1)/(2*delta)
            weight=angular.constant(I(shape(bounds.lo),shape(bounds.hi)))
            derivative=angular.constant(I(derivative_shape(bounds.hi),derivative_shape(bounds.lo)))
        dw=tuple(derivative*x/(rchart*cosb) for x in etheta)
        weights=((weight,dw),(1-weight,tuple(-x for x in dw)))
    original_columns=[]; indices=[]; tips=[]
    for oldbank,(weight,dweight) in zip(sourcebanks,weights):
        top=R+oldbank.roof_radial_displacement_m
        rold=angular.cbrt(top**3-3*R**2*depth)
        tau=depth+(R**3-top**3)/(3*R**2)
        axis=tuple(_round(I(x),angular.precision+16) for x in oldbank.angular_rate_per_physical_volume_depth_m)
        norm=interval_sqrt(sum((x*x for x in axis),I(0)),angular.precision+16)
        axisnorm=tuple(x/norm for x in axis)
        K=((I(0),-axisnorm[2],axisnorm[1]),(axisnorm[2],I(0),-axisnorm[0]),(-axisnorm[1],axisnorm[0],I(0)))
        K2=_mm(K,K)
        spin=angular.sin(tau*norm); one_minus_cos=1-angular.cos(tau*norm)
        P=tuple(tuple(identity[i][j]+spin*K[i][j]+one_minus_cos*K2[i][j] for j in range(3)) for i in range(3))
        Q=oldbank.material_owner_pose_bounds
        pose=_mm(P,Q); e=_mv(pose,q)
        de=_angular_add(_mm(pose,dq),_angular_outer(_angular_cross(tuple(angular.constant(x) for x in axis),e),dtau))
        X=tuple(rold*x for x in e)
        original=_angular_add(_angular_outer(e,tuple(rchart*rchart/(rold*rold)*x for x in q)),_angular_scale(rold,de))
        modes=[tuple(chi*x for x in e)]
        gradients=[_angular_add(_angular_scale(chi,de),_angular_outer(tuple(slope*x for x in e),gb))]
        for E in AFFINE_SYMMETRIC_GENERATORS:
            EX=_mv(E,X); ED=_mm(E,original)
            modes.append(tuple(chi*x/R for x in EX))
            gradients.append(_angular_add(_angular_scale(chi/R,ED),_angular_outer(tuple(slope*x/R for x in EX),gb)))
        bank=next(b for b in system.banks if b.original_parent_id==oldbank.original_parent_id)
        tips.append(len(indices))
        for globalindex,psi,G in zip(bank.coordinate_indices,modes,gradients):
            indices.append(globalindex)
            original_columns.append(_angular_add(_angular_scale(weight,G),_angular_outer(psi,dweight)))
    if not prepared:
        return tuple(indices),tuple(tips),tuple(original_columns)
    cert=cold.whole_domain_certificate.donor if current.owner==archive.donor_owner else cold.whole_domain_certificate.receiver
    scale=cert.fixed_radial_scale
    if current.owner==archive.donor_owner:
        # The owner relative-motion angle is at most Gamma.  For beta>=2Gamma
        # the original donor never enters the receiver during this journal;
        # its smooth activation and gradient are exactly zero, hence A887=Qd.
        inverse=tuple(tuple(angular.constant(x/scale) for x in row) for row in _prepared_inverse_rotation(archive.donor_axis,angular.precision))
    else:
        poseinverse=_prepared_inverse_rotation(archive.receiver_axis,angular.precision)
        inverse=tuple(tuple(angular.constant(x/scale) for x in row) for row in poseinverse)
    return tuple(indices),tuple(tips),tuple(_mm(G,inverse) for G in original_columns)


def _prepared_raw_kernel(columns,coordinates,local_tips,lame_pa,shear_pa):
    """Correlated remainder channels; each local active pair is retained."""
    n=len(columns); pairs=tuple((i,j) for i in range(n) for j in range(i,n))
    D=tuple(tuple(sum((a*g[i][j] for a,g in zip(coordinates,columns)),0) for j in range(3)) for i in range(3))
    linear=tuple(_sym(g) for g in columns)
    quadratic=_mm(_tr(D),D)
    tipset=set(local_tips)
    remainder=tuple(tuple(sum((coordinates[k]*linear[k][i][j] for k in range(n) if k not in tipset),0)
        +quadratic[i][j]/2 for j in range(3)) for i in range(3))
    zero=columns[0][0][0]*0; zeromatrix=((zero,)*3,)*3
    first=tuple(_angular_add(zeromatrix if i in tipset else linear[i],_sym(_mm(_tr(D),g))) for i,g in enumerate(columns))
    second=tuple(_sym(_mm(_tr(columns[i]),columns[j])) for i,j in pairs)
    tipstrains=tuple(linear[i] for i in local_tips)
    inner=lambda a,b:_tensor_inner(a,b,lame_pa,shear_pa)
    result=[inner(a,b) for i,a in enumerate(tipstrains) for b in tipstrains[i:]]
    result.extend(inner(H,remainder) for H in tipstrains)
    result.extend(inner(H,variation) for H in tipstrains for variation in first)
    result.extend(inner(H,variation) for H in tipstrains for variation in second)
    result.append(inner(remainder,remainder))
    result.extend(inner(variation,remainder) for variation in first)
    result.extend(inner(first[i],first[j])+inner(second[k],remainder) for k,(i,j) in enumerate(pairs))
    return tuple(result)


@dataclass(frozen=True,slots=True)
class NativeCommonColdContinuumQuadratureCertificate:
    material_reference_measure: str
    complete_original_parent_ids: tuple
    nonempty_archive_domain_ids: tuple
    exact_layer_cuts_m: tuple
    source_radius_m: F
    donor_activation_collar_beta_upper_radians: F
    donor_carrier_global_gradient_deviation_bound: F
    whole_source_volume_bounds_m3: I
    active_cold_source_volume_bounds_m3: I
    cell_receipts: tuple
    taylor_order: int
    beta_cell_count: int
    depth_cell_count: int
    maximum_absolute_raw_enclosure_width: F
    precision: int
    integration_method: str
    theta_cell_count: int
    total_cell_count: int
    whole_domain_enclosure_certified: bool = True
    analytic_seam_theta_moments_certified: bool = True
    donor_collar_uniform_enclosure_certified: bool = True
    authentic_original_volume_included_once: bool = True
    zero_clamp_complement_retained: bool = True
    gauss_difference_used_as_error_certificate: bool = False
    historical_archived_prestress_or_preparation_work_certified: bool = False
    whole_interval_box_range_certified: bool = True
    directed_ieee_basic_arithmetic_certified: bool = True
    transcendental_endpoints_rationally_certified: bool = True
    cell_raw_bounds_archived: bool = False
    cell_raw_bounds_digest_algorithm: str = "SHA256_CANONICAL_EXACT_DTO_FULL_RAW_BOUNDS_AND_CELL_PROOF_PARAMETERS"


def integrate_native_common_cold_continuum_response(system,coordinate_increments=None,*,precision=96,
        taylor_order=2,beta_cell_count=1,depth_cell_count=1,max_cells=4096,integration_method='whole_interval',theta_cell_count=4):
    cold,archive=_registered(system); _precision(precision); coords=_coordinates(system,coordinate_increments)
    if type(taylor_order) is not int or not 0<=taylor_order<=24:
        raise ValueError("The whole Taylor enclosure order is a literal integer from 0 to 24")
    if any(type(n) is not int or n<1 for n in (beta_cell_count,depth_cell_count,max_cells)):
        raise ValueError("Whole certified integration subdivision and cell budgets are literal positive integers")
    needed=8*2*3*beta_cell_count*depth_cell_count+4*2*3*depth_cell_count
    if needed>max_cells:
        raise ValueError("The complete requested domain exceeds its declared certified cell budget")
    if type(integration_method) is not str or integration_method not in ('whole_interval','analytic_theta_taylor'):
        raise ValueError("Select a declared whole interval or analytic theta Taylor enclosure")
    if type(theta_cell_count) is not int or theta_cell_count<1:
        raise ValueError("The whole theta subdivision is a literal positive integer")
    key=(system.binding_sha256,coords,precision,taylor_order,beta_cell_count,depth_cell_count,max_cells,integration_method,theta_cell_count)
    if key in _INTEGRAL_CACHE:
        return validate_native_common_cold_continuum_integral(_INTEGRAL_CACHE[key],system=system,coordinate_increments=coords)
    rows=tuple(dict(parent_id=b.original_parent_id,radius_m=b.radius_m,beta_cut=next(x.beta_cut_radians for x in cold.banks if x.original_parent_id==b.original_parent_id),
        thickness_m=b.areal_material_thickness_m,elastic_depth_m=b.elastic_layer_areal_depth_m,donor=b.owner==archive.donor_owner) for b in system.banks)
    def kernel_factory(row,tag,lo,hi,layer):
        sourcebanks,_,_=_piece_banks(cold,row['parent_id'],tag)
        active=tuple(i for old in sourcebanks for i in next(b.coordinate_indices for b in system.banks if b.original_parent_id==old.original_parent_id))
        tips=tuple(range(0,len(active),7)); material=next(b for b in system.banks if b.original_parent_id==row['parent_id'])
        lam,shear=material.lame_coefficients(layer)
        def evaluator(beta,depth,angular):
            actual,localtips,columns=_prepared_modal_fields(system,cold,archive,row['parent_id'],tag,beta,depth,angular)
            if actual!=active or localtips!=tips:
                raise ValueError("The prepared field changed an authentic active seam bank")
            return _prepared_raw_kernel(columns,tuple(coords[i] for i in active),tips,lam,shear)
        return active,tips,evaluator
    def collar_factory(row,tag,lo,hi,layer,bb,zb,active,tips):
        scalar=TaylorAlgebra(0,precision)
        angular=SeamAngularAlgebra(scalar,theta_power_scale=cold.whole_domain_certificate.seam_half_width_radians if tag!='interior' else F(1))
        _,_,columns=_prepared_modal_fields(system,cold,archive,row['parent_id'],tag,scalar.constant(bb),scalar.constant(zb),angular,prepared=False)
        g=[]
        # End seam polynomials use phi=pi/2-theta; its actual source measure
        # and complete [0,delta] integration are unchanged by the substitution.
        alo,ahi=(I(0),I(cold.whole_domain_certificate.seam_half_width_radians)) if tag=='end_seam' else (lo,hi)
        for G in columns:
            cap=sum((max(abs(v.lo),abs(v.hi))**2 for rowG in G for entry in rowG
                for v in (seam_angular_range(entry,alo,ahi).constant_bounds,)),F(0))
            g.append(interval_sqrt(I(cap),precision+16).hi)
        material=next(b for b in system.banks if b.original_parent_id==row['parent_id']); lam,shear=material.lame_coefficients(layer)
        cert=cold.whole_domain_certificate.donor
        return uniform_donor_collar_channels(tuple(g),tuple(coords[i] for i in active),tips,lame_pa=lam,shear_pa=shear,
            fixed_scale=cert.fixed_radial_scale,carrier_deviation=cert.archive_carrier_gradient_deviation_bound,precision=precision)
    if integration_method=='analytic_theta_taylor':
        receipts=integrate_whole_cold_domain(rows,kernel_factory,collar_factory,gamma=archive.whole_domain_certificate.gamma,
            delta=cold.whole_domain_certificate.seam_half_width_radians,precision=precision,order=taylor_order,
            beta_cell_count=beta_cell_count,depth_cell_count=depth_cell_count,max_cells=max_cells)
        actual_beta,actual_depth=beta_cell_count,depth_cell_count
    else:
        actual_beta,actual_depth=max(8,beta_cell_count),max(2,depth_cell_count)
        receipts=_integrate_whole_interval_domain(system,cold,archive,coords,rows,precision,
            actual_beta,actual_depth,theta_cell_count,max_cells)
    M=[[I(0) for _ in range(8)] for _ in range(8)]; pr=[I(0)]*8; coercivity=[F(0)]*8
    dpr=[[I(0)]*56 for _ in range(8)]; hpr=[[[I(0)]*56 for _ in range(56)] for _ in range(8)]
    N=I(0); gN=[I(0)]*56; hN=[[I(0)]*56 for _ in range(56)]
    tipmap={b.tip_coordinate_index:i for i,b in enumerate(system.banks)}
    for receipt in receipts:
        active=receipt['active_indices']; tips=receipt['local_tip_indices']; globaltips=tuple(tipmap[active[t]] for t in tips)
        values=receipt['integral_bounds']; cursor=0; n=len(active); pairs=tuple((i,j) for i in range(n) for j in range(i,n))
        for ti,gi in enumerate(globaltips):
            for gj in globaltips[ti:]:
                loidx,hiidx=min(gi,gj),max(gi,gj)
                if gi==gj and receipt['theta_piece']=='interior' and receipt['kind']!='whole_donor_activation_collar':
                    coercivity[gi]+=max(F(0),values[cursor].lo)
                M[loidx][hiidx]=_round(M[loidx][hiidx]+values[cursor],precision); cursor+=1
        for gi in globaltips:
            pr[gi]=_round(pr[gi]+values[cursor],precision); cursor+=1
        for gi in globaltips:
            for gj in active:
                dpr[gi][gj]=_round(dpr[gi][gj]+values[cursor],precision); cursor+=1
        for gt in globaltips:
            for i,j in pairs:
                gi,gj=active[i],active[j]; loidx,hiidx=min(gi,gj),max(gi,gj)
                hpr[gt][loidx][hiidx]=_round(hpr[gt][loidx][hiidx]+values[cursor],precision); cursor+=1
        N=_round(N+values[cursor],precision); cursor+=1
        for gi in active:
            gN[gi]=_round(gN[gi]+values[cursor],precision); cursor+=1
        for i,j in pairs:
            gi,gj=active[i],active[j]; loidx,hiidx=min(gi,gj),max(gi,gj)
            hN[loidx][hiidx]=_round(hN[loidx][hiidx]+values[cursor],precision); cursor+=1
        if cursor!=len(values):
            raise ValueError("The complete local continuum raw channel topology changed")
    M=tuple(tuple(M[min(i,j)][max(i,j)] for j in range(8)) for i in range(8))
    hpr=tuple(tuple(tuple(block[min(i,j)][max(i,j)] for j in range(56)) for i in range(56)) for block in hpr)
    hN=tuple(tuple(hN[min(i,j)][max(i,j)] for j in range(56)) for i in range(56))
    pi=interval_pi(precision); volume=pi*sum((2*b.radius_m**2*b.areal_material_thickness_m for b in system.banks),F(0))/4
    activevolume=sum((pi/2*b.radius_m**2*b.areal_material_thickness_m*_sc_trig(I(row['beta_cut']),precision)[0] for b,row in zip(system.banks,rows)),I(0))
    widths=tuple(x.width for receipt in receipts for x in receipt['integral_bounds'])
    compact=[]
    for receipt in receipts:
        summary={k:v for k,v in receipt.items() if k!='integral_bounds'}
        if receipt['theta_piece']=='interior' and receipt['kind']!='whole_donor_activation_collar':
            summary['interior_tip_gram_bounds_n_m']=receipt['integral_bounds'][0]
            summary['interior_tip_gram_global_index']=tipmap[receipt['active_indices'][receipt['local_tip_indices'][0]]]
        summary['raw_channel_count']=len(receipt['integral_bounds'])
        summary['maximum_absolute_raw_enclosure_width']=max((x.width for x in receipt['integral_bounds']),default=F(0))
        summary['complete_cell_receipt_canonical_sha256']=_digest(receipt)
        compact.append(summary)
    certificate=NativeCommonColdContinuumQuadratureCertificate(MEASURE,tuple(b.original_parent_id for b in system.banks),
        cold.whole_domain_certificate.complete_archive_domain_ids,tuple((b.original_parent_id,b.elastic_layer_areal_depth_m,b.areal_material_thickness_m) for b in system.banks),
        archive.radius_m,2*archive.whole_domain_certificate.gamma,cold.whole_domain_certificate.donor.archive_carrier_gradient_deviation_bound,
        volume,activevolume,tuple(compact),taylor_order if integration_method=='analytic_theta_taylor' else 0,
        actual_beta,actual_depth,max(widths,default=F(0)),precision,integration_method,theta_cell_count,len(receipts),
        analytic_seam_theta_moments_certified=integration_method=='analytic_theta_taylor',
        whole_interval_box_range_certified=integration_method=='whole_interval',
        directed_ieee_basic_arithmetic_certified=integration_method=='whole_interval')
    raw=NativeCommonColdContinuumIntegral(system.binding_sha256,coords,M,tuple(coercivity),tuple(pr),tuple(tuple(row) for row in dpr),hpr,N,tuple(gN),hN,certificate,precision,'')
    raw=_register_integral(raw); _INTEGRAL_CACHE[key]=raw
    return raw


# Pure development fragment; append only to the authorized new continuum core.
from functools import lru_cache
from math import comb, factorial

from .geometric_native_pose import interval_pi, _rounded_interval
from .geometric_native_curved_certified_integration import (
    TaylorAlgebra, TaylorJet, validated_integrate_rectangle,
)


def _sc_zero(value):
    if type(value) is TaylorJet:
        return not value.coefficients
    return type(value) is I and value.lo == value.hi == 0


def _sc_interval_power(value,n):
    if type(value) is not I or type(n) is not int or n<0:
        raise ValueError('An interval power requires a nonnegative integer exponent')
    if n==0: return I(1)
    if n%2: return I(value.lo**n,value.hi**n)
    return I(F(0) if value.lo<=0<=value.hi else min(value.lo**n,value.hi**n),
             max(value.lo**n,value.hi**n))


class SeamAngularPolynomial:
    """Finite theta^m cos(theta)^i sin(theta)^j with Taylor scalar coefficients."""
    __slots__ = ('algebra', 'coefficients')

    def __init__(self, algebra, coefficients):
        if type(algebra) is not SeamAngularAlgebra or type(coefficients) is not dict:
            raise ValueError('Seam angular expressions require their declared scalar algebra')
        result = {}
        for key, value in coefficients.items():
            if (type(key) is not tuple or len(key) != 3
                    or any(type(k) is not int or k < 0 for k in key)):
                raise ValueError('Angular powers must be three nonnegative integers')
            m, i, j = key
            if m > algebra.maximum_theta_degree or i+j > algebra.maximum_trig_degree:
                raise ValueError('Angular expression exceeded its explicit finite degree')
            value = algebra.scalar.constant(value)
            if _sc_zero(value):
                continue
            for k in range(j//2+1):
                reduced = (m, i+2*k, j % 2)
                term = value*F((-1)**k*comb(j//2, k))
                result[reduced] = result[reduced]+term if reduced in result else term
        self.algebra = algebra
        self.coefficients = {k:v for k,v in result.items() if not _sc_zero(v)}

    def _other(self, value):
        return self.algebra.constant(value)

    def __add__(self, value):
        value = self._other(value)
        zero = self.algebra.scalar.constant(0)
        return SeamAngularPolynomial(self.algebra, {k:self.coefficients.get(k,zero)
            +value.coefficients.get(k,zero) for k in self.coefficients.keys()|value.coefficients.keys()})
    __radd__ = __add__

    def __neg__(self):
        return SeamAngularPolynomial(self.algebra,{k:-v for k,v in self.coefficients.items()})

    def __sub__(self,value):
        return self+-self._other(value)

    def __rsub__(self,value):
        return self._other(value)+-self

    def __mul__(self,value):
        value=self._other(value); result={}
        for (m,i,j),first in self.coefficients.items():
            for (n,k,l),second in value.coefficients.items():
                key=(m+n,i+k,j+l); term=first*second
                result[key]=result[key]+term if key in result else term
        return SeamAngularPolynomial(self.algebra,result)
    __rmul__=__mul__

    def _constant_scalar(self):
        if any(k!=(0,0,0) for k in self.coefficients):
            raise ValueError('A nonpolynomial operation cannot erase angular dependence')
        return self.coefficients.get((0,0,0),self.algebra.scalar.constant(0))

    def __truediv__(self,value):
        return self*self.algebra.constant(1/self._other(value)._constant_scalar())

    def __rtruediv__(self,value):
        return self._other(value)/self

    def __pow__(self,n):
        if type(n) is not int:
            raise ValueError('Angular exponent must be an integer')
        if n<0:
            return self.algebra.constant(self._constant_scalar()**n)
        result=self.algebra.constant(1); value=self
        while n:
            if n&1: result=result*value
            n//=2
            if n: value=value*value
        return result


class SeamAngularAlgebra:
    __slots__=('scalar','precision','maximum_theta_degree','maximum_trig_degree','theta_power_scale')
    def __init__(self,scalar,*,maximum_theta_degree=24,maximum_trig_degree=32,theta_power_scale=F(1)):
        if type(scalar) is not TaylorAlgebra:
            raise ValueError('Seam angular coefficients require validated Taylor scalars')
        self.scalar,self.precision=scalar,scalar.precision
        self.maximum_theta_degree,self.maximum_trig_degree=maximum_theta_degree,maximum_trig_degree
        if type(theta_power_scale) is not F or theta_power_scale<=0:
            raise ValueError('The normalized angular power scale must be positive and exact')
        self.theta_power_scale=theta_power_scale

    def constant(self,value):
        if type(value) is SeamAngularPolynomial:
            if value.algebra is not self:
                raise ValueError('Angular expressions cannot mix derivative domains')
            return value
        return SeamAngularPolynomial(self,{(0,0,0):value})

    def theta(self): return SeamAngularPolynomial(self,{(1,0,0):self.theta_power_scale})
    def sine_theta(self): return SeamAngularPolynomial(self,{(0,0,1):F(1)})
    def cosine_theta(self): return SeamAngularPolynomial(self,{(0,1,0):F(1)})
    def sin(self,value): return self.constant(self.scalar.sin(self.constant(value)._constant_scalar()))
    def cos(self,value): return self.constant(self.scalar.cos(self.constant(value)._constant_scalar()))
    def sqrt(self,value): return self.constant(self.scalar.sqrt(self.constant(value)._constant_scalar()))
    def cbrt(self,value): return self.constant(self.scalar.cbrt(self.constant(value)._constant_scalar()))


@lru_cache(maxsize=32768)
def _sc_trig(value,precision):
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    return _trig_ranges_signed(value.lo,value.hi,precision)


@lru_cache(maxsize=16384)
def _sc_fourier_moment(m,n,lo,hi,precision):
    """Integral theta^m times (cos(n theta), sin(n theta)); no sampled endpoint."""
    if n<0:
        c,s=_sc_fourier_moment(m,-n,lo,hi,precision)
        return c,-s
    if n==0:
        return (_sc_interval_power(hi,m+1)-_sc_interval_power(lo,m+1))/F(m+1),I(0)
    # Narrow seam moments must retain their theta^(m+1) factor until after
    # normalization. Fixed-grid rounding or repeated integration by parts
    # would destroy moments far smaller than 2^-precision.
    radius=max(abs(lo.lo),abs(lo.hi),abs(hi.lo),abs(hi.hi))
    if lo.lo>=0 and radius<=F(1,4):
        c,s=I(0),I(0); k=0
        while True:
            even,odd=2*k,2*k+1
            cp=F((-1)**k*n**even,factorial(even))
            sp=F((-1)**k*n**odd,factorial(odd))
            c=c+cp*(_sc_interval_power(hi,m+even+1)-_sc_interval_power(lo,m+even+1))/F(m+even+1)
            s=s+sp*(_sc_interval_power(hi,m+odd+1)-_sc_interval_power(lo,m+odd+1))/F(m+odd+1)
            ctail=F(n**(even+1),factorial(even+1))*radius**(even+1)
            stail=F(n**(odd+1),factorial(odd+1))*radius**(odd+1)
            if max(ctail,stail)<=F(1,2**(precision+16)):
                measure=(hi.hi-lo.lo)*radius**m
                return c+I(-ctail*measure,ctail*measure),s+I(-stail*measure,stail*measure)
            k+=1
    slo,clo=_sc_trig(n*lo,precision); shi,chi=_sc_trig(n*hi,precision)
    c=(_sc_interval_power(hi,m)*shi-_sc_interval_power(lo,m)*slo)/n
    s=(_sc_interval_power(lo,m)*clo-_sc_interval_power(hi,m)*chi)/n
    if m:
        previous_c,previous_s=_sc_fourier_moment(m-1,n,lo,hi,precision)
        c=c-F(m,n)*previous_s
        s=s+F(m,n)*previous_c
    return _rounded_interval(c,precision),_rounded_interval(s,precision)


@lru_cache(maxsize=32768)
def seam_angular_moment(m,i,j,lo,hi,precision=160,theta_power_scale=F(1)):
    """Enclose the complete literal moment, including irrational quarter endpoints."""
    if any(type(k) is not int or k<0 for k in (m,i,j)) or type(lo) is not I or type(hi) is not I:
        raise ValueError('Moments require explicit angular powers and exact endpoint enclosures')
    if lo.hi>hi.lo:
        raise ValueError('An angular integration interval must have ordered endpoints')
    if j>=2:
        return sum((F((-1)**k*comb(j//2,k))*seam_angular_moment(m,i+2*k,j%2,lo,hi,precision,theta_power_scale)
                    for k in range(j//2+1)),I(0))
    result=I(0)
    for k in range(i+1):
        n=i-2*k; factor=F(comb(i,k),2**i)
        if not j:
            term=_sc_fourier_moment(m,n,lo,hi,precision)[0]
        else:
            term=(_sc_fourier_moment(m,1+n,lo,hi,precision)[1]
                  +_sc_fourier_moment(m,1-n,lo,hi,precision)[1])/2
        result=result+factor*term
    return _rounded_interval(result/(theta_power_scale**m),precision)


def integrate_seam_angular(value,lo,hi):
    if type(value) is not SeamAngularPolynomial:
        raise ValueError('Analytic theta integration requires a finite angular expression')
    result=value.algebra.scalar.constant(0)
    for (m,i,j),coefficient in value.coefficients.items():
        result=result+coefficient*seam_angular_moment(m,i,j,lo,hi,value.algebra.precision,value.algebra.theta_power_scale)
    return result


def seam_angular_range(value,lo,hi):
    """Whole-piece enclosure; used only for the explicitly measured donor collar."""
    if type(value) is not SeamAngularPolynomial:
        raise ValueError('An angular range requires a finite angular expression')
    theta=I(lo.lo,hi.hi); sine,cosine=_sc_trig(theta,value.algebra.precision)
    normalized=theta/value.algebra.theta_power_scale
    result=value.algebra.scalar.constant(0)
    for (m,i,j),coefficient in value.coefficients.items():
        result=result+coefficient*_sc_interval_power(normalized,m)*_sc_interval_power(cosine,i)*_sc_interval_power(sine,j)
    return result


def seam_quarter_pieces(delta,precision=160):
    if type(delta) is not F or not 0<delta<F(1,4):
        raise ValueError('The source seam width must be a literal positive small angle')
    end=interval_pi(precision)/2
    return (('start_seam',I(0),I(delta)),('interior',I(delta),end-delta),
            ('end_seam',end-delta,end))


def integrate_prepared_cold_piece(evaluator,*,theta_lo,theta_hi,beta_bounds,depth_bounds,
                                 radius_m,order=2,precision=160):
    """New full-parent kernel; the old seven-mode tensor quadrature is never invoked."""
    if type(radius_m) is not F or radius_m<=0:
        raise ValueError('The source-volume measure requires the authentic radius')
    def rectangle(beta,depth,scalar):
        scale=theta_hi.lo if theta_lo==I(0) and theta_hi.width==0 and 0<theta_hi.lo<=F(1,4) else F(1)
        angular=SeamAngularAlgebra(scalar,theta_power_scale=scale)
        values=evaluator(beta,depth,angular)
        if type(values) is not tuple:
            raise ValueError('The prepared kernel must return its complete immutable raw channels')
        measure=radius_m**2*scalar.cos(beta)
        return tuple(integrate_seam_angular(angular.constant(v),theta_lo,theta_hi)*measure for v in values)
    return validated_integrate_rectangle(rectangle,beta_bounds,depth_bounds,order=order,precision=precision)


def uniform_donor_collar_channels(original_gradient_frobenius_upper,coordinates_m,local_tip_indices,
                                  *,lame_pa,shear_pa,fixed_scale,carrier_deviation,precision=160):
    """Full signed raw-channel enclosure without replacing the actual donor map.

    The prepared derivative is A0=s A887 and ||A887-Q||<=D<1 globally.
    Therefore every exact prepared modal gradient obeys ||G_i||F<=g_i/[s(1-D)].
    The collar is still integrated with its complete source-volume measure.
    """
    if (type(fixed_scale) is not F or fixed_scale<=0 or type(carrier_deviation) is not F
            or not 0<=carrier_deviation<1):
        raise ValueError('A collar needs its proved positive prepared-reference inverse bound')
    if len(original_gradient_frobenius_upper)!=len(coordinates_m):
        raise ValueError('Every active physical coordinate needs one whole-collar gradient bound')
    n=len(coordinates_m); tips=tuple(local_tip_indices); tipset=set(tips)
    g=tuple(x/(fixed_scale*(1-carrier_deviation)) for x in original_gradient_frobenius_upper)
    if any(type(x) is not F or x<0 for x in g):
        raise ValueError('Collar gradient upper bounds must be nonnegative exact Fractions')
    c=max(2*shear_pa,abs(3*lame_pa+2*shear_pa))
    L=sum((abs(a)*v for a,v in zip(coordinates_m,g)),F(0))
    r=sum((abs(coordinates_m[i])*g[i] for i in range(n) if i not in tipset),F(0))+L*L/2
    first=tuple((F(0) if i in tipset else g[i])+L*g[i] for i in range(n))
    pairs=tuple((i,j) for i in range(n) for j in range(i,n))
    def signed(cap): return _rounded_interval(I(-cap,cap),precision)
    values=[]
    for a,ta in enumerate(tips):
        for tb in tips[a:]:
            cap=c*g[ta]*g[tb]
            values.append(_rounded_interval(I(0,cap),precision) if ta==tb else signed(cap))
    values.extend(signed(c*g[t]*r) for t in tips)
    values.extend(signed(c*g[t]*v) for t in tips for v in first)
    values.extend(signed(c*g[t]*g[i]*g[j]) for t in tips for i,j in pairs)
    values.append(_rounded_interval(I(0,c*r*r),precision))
    values.extend(signed(c*r*v) for v in first)
    values.extend(signed(c*(first[i]*first[j]+r*g[i]*g[j])) for i,j in pairs)
    return tuple(values)


def integrate_uniform_collar(bounds,*,theta_lo,theta_hi,beta_bounds,depth_bounds,radius_m,precision=160):
    sinlo=_sc_trig(I(beta_bounds.lo),precision)[0]
    sinhi=_sc_trig(I(beta_bounds.hi),precision)[0]
    measure=radius_m**2*(theta_hi-theta_lo)*(sinhi-sinlo)*depth_bounds.width
    if measure.lo<0:
        raise ValueError('A full-source collar requires a proved nonnegative literal volume measure')
    return tuple(_rounded_interval(v*measure,precision) for v in bounds)


def integrate_whole_cold_domain(parent_rows,kernel_factory,collar_factory,*,gamma,
                                delta=F(1,100),precision=160,order=2,
                                beta_cell_count=1,depth_cell_count=1,max_cells=4096):
    """Finite complete domain traversal; factories own exact field/material bindings.

    parent_rows: immutable tuple of records with parent_id, radius_m, beta_cut,
    thickness_m, elastic_depth_m, donor. kernel_factory(row,tag,lo,hi,layer)
    returns (active_indices,local_tip_indices,evaluator). collar_factory receives
    those same inputs plus the exact beta/depth boxes and returns already proved
    per-original-volume raw interval bounds. Results retain every cell and its
    active global indices; the core assembles its full8/56 tensors from them.
    """
    if (type(parent_rows) is not tuple or len(parent_rows)!=8 or type(gamma) is not F
            or not 0<gamma<F(1,4)):
        raise ValueError('The full cold domain requires all eight authentic parents and literal Gamma')
    if any(type(n) is not int or n<1 for n in (beta_cell_count,depth_cell_count,max_cells)):
        raise ValueError('Certified integration cell budgets must be positive integers')
    pieces=seam_quarter_pieces(delta,precision); receipts=[]
    for row in parent_rows:
        H,ze,b=row['thickness_m'],row['elastic_depth_m'],row['beta_cut']
        if not 0<ze<H or (row['donor'] and not 2*gamma<b):
            raise ValueError('Source layer cuts and donor collar must lie in the complete parent')
        for layer,zlo,zhi in (('upper',F(0),ze),('lower',ze,H)):
            for tag,tlo,thi in pieces:
                # Reflect the end seam: phi=pi/2-theta, swapping its authentic
                # start/end axes in the field factory. This gives both seam
                # polynomial charts the stable local interval [0,delta].
                alo,ahi=(I(0),I(delta)) if tag=='end_seam' else (tlo,thi)
                active,tips,evaluator=kernel_factory(row,tag,alo,ahi,layer)
                for zi in range(depth_cell_count):
                    zb=I(zlo+(zhi-zlo)*zi/depth_cell_count,zlo+(zhi-zlo)*(zi+1)/depth_cell_count)
                    if row['donor']:
                        bb=I(0,2*gamma)
                        bounds=collar_factory(row,tag,alo,ahi,layer,bb,zb,active,tips)
                        values=integrate_uniform_collar(bounds,theta_lo=alo,theta_hi=ahi,
                            beta_bounds=bb,depth_bounds=zb,radius_m=row['radius_m'],precision=precision)
                        receipts.append(dict(kind='whole_donor_activation_collar',parent_id=row['parent_id'],
                            layer=layer,theta_piece=tag,theta_lo=tlo,theta_hi=thi,beta_bounds=bb,
                            integration_angle_lo=alo,integration_angle_hi=ahi,end_seam_axes_reflected=tag=='end_seam',
                            depth_bounds=zb,active_indices=active,local_tip_indices=tips,integral_bounds=values,
                            whole_piece_uniform_bound=True))
                    blo=2*gamma if row['donor'] else F(0)
                    for bi in range(beta_cell_count):
                        bb=I(blo+(b-blo)*bi/beta_cell_count,blo+(b-blo)*(bi+1)/beta_cell_count)
                        result=integrate_prepared_cold_piece(evaluator,theta_lo=alo,theta_hi=ahi,
                            beta_bounds=bb,depth_bounds=zb,radius_m=row['radius_m'],order=order,precision=precision)
                        receipts.append(dict(kind='analytic_theta_validated_beta_depth',parent_id=row['parent_id'],
                            layer=layer,theta_piece=tag,theta_lo=tlo,theta_hi=thi,beta_bounds=bb,
                            integration_angle_lo=alo,integration_angle_hi=ahi,end_seam_axes_reflected=tag=='end_seam',
                            depth_bounds=zb,active_indices=active,local_tip_indices=tips,
                            integral_bounds=result.integral_bounds,
                            polynomial_integral_bounds=result.polynomial_integral_bounds,
                            remainder_absolute_upper_bounds=result.remainder_absolute_upper_bounds))
                    if len(receipts)>max_cells:
                        raise ValueError('Declared full-source certified integration budget exceeded')
    return tuple(receipts)


class _BoxScalar:
    """Finite IEEE binary64 interval, outward rounded after every basic op.

    Transcendentals enter only through the repository's rational validated
    interval functions.  Each endpoint is exported as its literal Fraction.
    """
    __slots__=('lo','hi','algebra')

    def __init__(self,lo,hi,algebra):
        if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:
            raise ValueError("A certified box scalar must retain finite ordered binary endpoints")
        self.lo,self.hi,self.algebra=lo,hi,algebra

    def to_interval(self):
        return I(F.from_float(self.lo),F.from_float(self.hi))

    def _other(self,value):
        return self.algebra.constant(value)

    def _out(self,lo,hi):
        return _BoxScalar(math.nextafter(lo,-math.inf),math.nextafter(hi,math.inf),self.algebra)

    def __neg__(self):
        return _BoxScalar(-self.hi,-self.lo,self.algebra)

    def __add__(self,value):
        value=self._other(value)
        if value.lo==value.hi==0: return self
        if self.lo==self.hi==0: return value
        return self._out(self.lo+value.lo,self.hi+value.hi)
    __radd__=__add__

    def __sub__(self,value): return self+-self._other(value)
    def __rsub__(self,value): return self._other(value)+-self

    def __mul__(self,value):
        value=self._other(value)
        if self.lo==self.hi==0 or value.lo==value.hi==0:
            return _BoxScalar(0.0,0.0,self.algebra)
        products=(self.lo*value.lo,self.lo*value.hi,self.hi*value.lo,self.hi*value.hi)
        return self._out(min(products),max(products))
    __rmul__=__mul__

    def __truediv__(self,value):
        value=self._other(value)
        if value.lo<=0<=value.hi:
            raise ValueError("A whole material box divisor cannot include zero")
        reciprocal=value._out(1/value.hi,1/value.lo)
        return self*reciprocal
    def __rtruediv__(self,value): return self._other(value)/self

    def __pow__(self,n):
        if type(n) is not int:
            raise ValueError("A box power requires a literal integer")
        if n<0: return (1/self)**(-n)
        if n>0 and self.lo==self.hi==0: return self.algebra.constant(0)
        if n==2:
            lower=0.0 if self.lo<=0<=self.hi else min(self.lo*self.lo,self.hi*self.hi)
            upper=max(self.lo*self.lo,self.hi*self.hi)
            result=self._out(lower,upper)
            return _BoxScalar(max(0.0,result.lo),result.hi,self.algebra)
        result=self.algebra.constant(1); value=self
        while n:
            if n&1: result=result*value
            n//=2
            if n: value=value**2
        return result


class _BoxAlgebra:
    __slots__=('precision','scalar','angle_bounds','_constants','_sine_theta','_cosine_theta')

    def __init__(self,angle_bounds,precision):
        self.precision=precision; self.scalar=self; self.angle_bounds=angle_bounds
        self._constants={}; self._sine_theta=self._cosine_theta=None

    def constant(self,value):
        if type(value) is _BoxScalar:
            if value.algebra is not self:
                raise ValueError("Certified box arithmetic cannot mix query domains")
            return value
        if type(value) not in (I,F,int):
            raise ValueError("A box constant requires a literal rational interval")
        if value in self._constants:
            return self._constants[value]
        interval=value if type(value) is I else I(value)
        try:
            lo,hi=float(interval.lo),float(interval.hi)
        except OverflowError as exc:
            raise ValueError("An exact source scalar overflowed the declared binary enclosure") from exc
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("An exact source scalar overflowed the declared binary enclosure")
        if F.from_float(lo)>interval.lo: lo=math.nextafter(lo,-math.inf)
        if F.from_float(hi)<interval.hi: hi=math.nextafter(hi,math.inf)
        result=_BoxScalar(lo,hi,self); self._constants[value]=result
        return result

    def theta(self): return self.constant(self.angle_bounds)
    def sine_theta(self):
        if self._sine_theta is None:
            self._sine_theta=self.constant(_sc_trig(self.angle_bounds,self.precision)[0])
        return self._sine_theta
    def cosine_theta(self):
        if self._cosine_theta is None:
            self._cosine_theta=self.constant(_sc_trig(self.angle_bounds,self.precision)[1])
        return self._cosine_theta
    def sin(self,value): return self.constant(_sc_trig(self.constant(value).to_interval(),self.precision)[0])
    def cos(self,value): return self.constant(_sc_trig(self.constant(value).to_interval(),self.precision)[1])
    def sqrt(self,value): return self.constant(interval_sqrt(self.constant(value).to_interval(),self.precision))
    def cbrt(self,value): return self.constant(_box_cbrt(self.constant(value).to_interval(),self.precision))


@lru_cache(maxsize=16384)
def _box_cbrt(value,precision):
    return interval_cbrt(value,precision)


@lru_cache(maxsize=128)
def _prepared_inverse_rotation(axis,precision):
    return _transpose(_rotation(axis,F(1),precision))


def _integrate_whole_interval_domain(system,cold,archive,coords,rows,precision,beta_cells,depth_cells,theta_cells,max_cells):
    """Enclose every point of each complete literal material-volume box."""
    delta=cold.whole_domain_certificate.seam_half_width_radians; gamma=archive.whole_domain_certificate.gamma
    needed=8*2*3*theta_cells*depth_cells*beta_cells+24
    if needed>max_cells:
        raise ValueError("The actual complete whole-interval box traversal exceeds its declared cell budget")
    receipts=[]
    for row in rows:
        material=next(b for b in system.banks if b.original_parent_id==row['parent_id'])
        for layer,zlo,zhi in (('upper',F(0),row['elastic_depth_m']),('lower',row['elastic_depth_m'],row['thickness_m'])):
            lam,shear=material.lame_coefficients(layer)
            for tag,tlo,thi in seam_quarter_pieces(delta,precision):
                # A reflected upper-edge chart retains the exact full piece.
                alo,ahi=(I(0),I(delta)) if tag=='end_seam' else (tlo,thi)
                count=theta_cells
                sourcebanks,_,_=_piece_banks(cold,row['parent_id'],tag)
                active=tuple(i for old in sourcebanks for i in next(b.coordinate_indices for b in system.banks if b.original_parent_id==old.original_parent_id))
                tips=tuple(range(0,len(active),7)); localcoords=tuple(coords[i] for i in active)
                for ti in range(count):
                    ta=alo+(ahi-alo)*F(ti,count); tb=alo+(ahi-alo)*F(ti+1,count)
                    angle=I(ta.lo,tb.hi)
                    for zi in range(depth_cells):
                        zb=I(zlo+(zhi-zlo)*zi/depth_cells,zlo+(zhi-zlo)*(zi+1)/depth_cells)
                        segments=((True,I(0,2*gamma)),) if row['donor'] and ti==zi==0 else ()
                        blo=2*gamma if row['donor'] else F(0)
                        segments+=tuple((False,I(blo+(row['beta_cut']-blo)*bi/beta_cells,
                            blo+(row['beta_cut']-blo)*(bi+1)/beta_cells)) for bi in range(beta_cells))
                        for collar,bb in segments:
                            boxangle=I(alo.lo,ahi.hi) if collar else angle
                            boxdepth=I(zlo,zhi) if collar else zb
                            aa,ab=(alo,ahi) if collar else (ta,tb)
                            algebra=_BoxAlgebra(boxangle,precision)
                            indices,localtips,columns=_prepared_modal_fields(system,cold,archive,row['parent_id'],tag,
                                algebra.constant(bb),algebra.constant(boxdepth),algebra,prepared=not collar)
                            if indices!=active or localtips!=tips:
                                raise ValueError("A uniform material box changed a complete active bank")
                            if collar:
                                g=tuple(interval_sqrt(I(sum((max(abs(F.from_float(x.lo)),abs(F.from_float(x.hi)))**2 for rowG in G for x in rowG),F(0))),precision).hi for G in columns)
                                cert=cold.whole_domain_certificate.donor
                                values=uniform_donor_collar_channels(g,localcoords,tips,lame_pa=lam,shear_pa=shear,
                                    fixed_scale=cert.fixed_radial_scale,carrier_deviation=cert.archive_carrier_gradient_deviation_bound,precision=precision)
                            else:
                                values=tuple(v.to_interval() for v in _prepared_raw_kernel(columns,localcoords,tips,lam,shear))
                            sinlo=_sc_trig(I(bb.lo),precision)[0]; sinhi=_sc_trig(I(bb.hi),precision)[0]
                            volume=row['radius_m']**2*(ab-aa)*(sinhi-sinlo)*boxdepth.width
                            if volume.lo<0:
                                raise ValueError("The exact whole-source box measure must be positive")
                            integrated=tuple(_round(v*volume,precision) for v in values)
                            receipts.append(dict(kind='whole_donor_activation_collar' if collar else 'whole_interval_theta_beta_depth',
                                parent_id=row['parent_id'],layer=layer,theta_piece=tag,theta_lo=tlo,theta_hi=thi,
                                integration_angle_lo=aa,integration_angle_hi=ab,end_seam_axes_reflected=tag=='end_seam',
                                beta_bounds=bb,depth_bounds=boxdepth,active_indices=active,local_tip_indices=tips,
                                source_volume_bounds_m3=volume,integral_bounds=integrated,
                                whole_piece_uniform_bound=True,directed_ieee_basic_arithmetic=True,
                                transcendental_endpoints_rationally_certified=True))
    return tuple(receipts)

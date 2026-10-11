"""Whole original-volume cold potential for composed895 modes and T_lambda.

This additive component reuses the sealed899 material law and inverse Gram.
It integrates NEW mixed moments, including the modal complement and actual
887 donor activation collar. No source reference or material clock is reset.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_pi, interval_sqrt, _rounded_interval
from .geometric_native_common_cold_variational_work import bind_native_cold_variational_source, _source
from .geometric_native_common_cold_preparation_integrals import _bindings, _cell_volume, _owner_direction, _relative_twist_kernel
from .geometric_native_common_cold_continuum_energy import (
    _BoxAlgebra, _BoxScalar, _piece_banks, _prepared_modal_fields,
    _prepared_inverse_rotation, seam_quarter_pieces, _tensor_inner,
)
from .geometric_native_archive_material_carrier import interval_exp
from .geometric_native_affine_body import AFFINE_SYMMETRIC_GENERATORS
from .geometric_native_common_preparation_balance import _authenticate, AUTHORITY_PINS

FORMAT = 'native-common-coupled-cold-energy-system-1'
RESPONSE_FORMAT = 'native-common-coupled-cold-energy-response-1'
MEASURE = 'AUTHENTIC_ORIGINAL_AREAL_VOLUME_R_SQUARED_DOMEGA_DZ_ONCE'
_SYSTEMS, _CONTEXTS, _RESPONSES, _CACHE, _GEOMETRY, _POINTS = {}, {}, {}, {}, {}, {}


def _payload(x):
    if type(x) is F: return [x.numerator, x.denominator]
    if type(x) is I: return dict(lo=_payload(x.lo), hi=_payload(x.hi))
    if type(x) in (tuple, list): return [_payload(v) for v in x]
    if type(x) is dict: return {k:_payload(v) for k,v in x.items()}
    if hasattr(type(x), '__dataclass_fields__'): return {f.name:_payload(getattr(x,f.name)) for f in fields(x)}
    if type(x) in (str, int, bool) or x is None: return x
    raise ValueError('An undeclared coupled cold scalar or container was supplied')


def _json(x): return json.dumps(_payload(x),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def _seal(x): return hashlib.sha256(_json({f.name:getattr(x,f.name) for f in fields(x) if f.name!='binding_sha256'}).encode()).hexdigest()
def _range(x): return x.to_interval() if type(x) is _BoxScalar else x if type(x) is I else I(x)
def _rnd(x,p): return _rounded_interval(_range(x),p+16)
def _mm(a,b): return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(3)),0) for j in range(3)) for i in range(3))
def _mv(a,b): return tuple(sum((x*y for x,y in zip(row,b)),0) for row in a)
def _tr(a): return tuple(zip(*a))
def _sym(a): return tuple(tuple((a[i][j]+a[j][i])/2 for j in range(3)) for i in range(3))
def _add(a,b): return tuple(tuple(a[i][j]+b[i][j] for j in range(3)) for i in range(3))
def _scale(x,a): return tuple(tuple(x*v for v in row) for row in a)
def _outer(a,b): return tuple(tuple(x*y for y in b) for x in a)
def _cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def _dot(a,b): return sum((x*y for x,y in zip(a,b)),0)
def _matrix_range(a): return tuple(tuple(_range(x) for x in row) for row in a)
def _abs(x): return max(abs(x.lo),abs(x.hi))
def _frobenius(a,p): return interval_sqrt(I(sum((_abs(_range(v))**2 for row in a for v in row),F(0))),p+16).hi
def _vector_norm(a,p): return interval_sqrt(I(sum((_abs(_range(v))**2 for v in a),F(0))),p+16).hi


def _precision(p):
    if type(p) is not int or not 96<=p<=512: raise ValueError('Precision is a literal integer in[96,512]')
    return p


def _coordinates(system,coordinates,lam):
    coordinates=(F(0),)*56 if coordinates is None else coordinates
    if type(coordinates) is not tuple or len(coordinates)!=56 or any(type(v) is not F for v in coordinates):
        raise ValueError('All56 cold coordinates must be literal Fractions')
    if type(lam) is not F or not 0<=lam<=1: raise ValueError('Preparation lambda must be a literal Fraction in[0,1]')
    if any(abs(v)>b for v,b in zip(coordinates,system.coordinate_envelope_abs_m)):
        raise ValueError('The composed cold field exceeds the authentic895 whole-source coordinate envelope')
    return coordinates


@dataclass(frozen=True,slots=True)
class NativeCommonCoupledColdEnergy:
    source899_raw_sha256: str
    source899_packet_sha256: str
    source923_raw_sha256: str
    source923_packet_sha256: str
    source_binding_sha256: str
    cold_carrier_binding_sha256: str
    continuum_system_binding_sha256: str
    archive_carrier_model_sha256: str
    preparation_binding_sha256s: tuple
    coordinate_order: tuple
    coordinate_units: tuple
    coordinate_envelope_abs_m: tuple
    tip_coordinate_indices: tuple
    original_parent_ids: tuple
    nonempty_archive_domain_ids: tuple
    exact_layer_cuts_m: tuple
    source_clocks_s: tuple
    literal_reference_energies_j: tuple
    literal_reference_forces_n: tuple
    tip_stiffnesses_n_m: tuple
    provenance: str
    binding_sha256: str
    material_reference_measure: str = MEASURE
    original_gram_and_inverse_reused_without_quadrature: bool = True
    historical_archived_three_dimensional_prestress_recovered: bool = False
    general_physical_material_handover_certified: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    accepted_new_native_intervals: int = 0


def bind_native_common_coupled_cold_energy(source899,cold_carrier,continuum_system,preparations,*,source923,provenance):
    if type(provenance) is not str or not provenance.strip(): raise ValueError('The composed cold potential needs explicit provenance')
    source=bind_native_cold_variational_source(source899)
    cold,archive=_bindings(source,cold_carrier,continuum_system,preparations)
    _authenticate(source923,AUTHORITY_PINS[0])
    old=source923['integral']
    if (old['cold_system_binding_sha256']!=cold.binding_sha256 or old['continuum_system_binding_sha256']!=continuum_system.binding_sha256
            or tuple(old['preparation_binding_sha256s'])!=tuple(p.binding_sha256 for p in preparations)
            or old['archive_carrier_model_sha256']!=archive.model_sha256
            or old['material_reference_measure']!=MEASURE):
        raise ValueError('The frozen923 preparation and composed material/constitutive sources differ')
    result=NativeCommonCoupledColdEnergy(source.source_raw_sha256,source.source_packet_sha256,AUTHORITY_PINS[0][1],AUTHORITY_PINS[0][2],
        source.binding_sha256,cold.binding_sha256,continuum_system.binding_sha256,archive.model_sha256,tuple(p.binding_sha256 for p in preparations),
        source.coordinate_order+(("preparation","cold_depth_twist","lambda"),),("m",)*56+("1",),cold.coordinate_envelope_abs_m,
        source.tip_coordinate_indices,tuple(b.original_parent_id for b in continuum_system.banks),cold.whole_domain_certificate.complete_archive_domain_ids,
        tuple((b.original_parent_id,b.elastic_layer_areal_depth_m,b.areal_material_thickness_m) for b in continuum_system.banks),preparations[0].source_clocks_s,
        source.literal_reference_energies_j,source.literal_reference_forces_n,source.tip_stiffnesses_n_m,provenance,'')
    result=replace(result,binding_sha256=_seal(result))
    if result.binding_sha256 in _SYSTEMS: return validate_native_common_coupled_cold_energy(_SYSTEMS[result.binding_sha256])
    _SYSTEMS[result.binding_sha256]=result
    _CONTEXTS[result.binding_sha256]=dict(source=source,source899=source899,source923=source923,cold_carrier=cold,
        continuum_system=continuum_system,preparations=preparations,archive=archive)
    return result


def validate_native_common_coupled_cold_energy(system):
    if type(system) is not NativeCommonCoupledColdEnergy or _SYSTEMS.get(system.binding_sha256) is not system or _seal(system)!=system.binding_sha256:
        raise ValueError('An altered, resealed or unregistered composed cold system is not authorized')
    _source(_CONTEXTS[system.binding_sha256]['source'])
    return system


def native_common_coupled_cold_energy_context(system):
    validate_native_common_coupled_cold_energy(system)
    return dict(_CONTEXTS[system.binding_sha256])


def _material_fields(ctx,parent,tag,beta,depth,a,prepared=True):
    """Actual895 displacement fields plus authentic prepared-relative gradients."""
    cold,archive,continuum=ctx['cold_carrier'],ctx['archive'],ctx['continuum_system']
    indices,tips,G=_prepared_modal_fields(continuum,cold,archive,parent,tag,beta,depth,a,prepared=prepared)
    banks,start,end=_piece_banks(cold,parent,tag); current=banks[0]; R=current.radius_m
    sb,cb=a.sin(beta),a.cos(beta); st,ct=a.sine_theta(),a.cosine_theta()
    q=tuple(current.original_quarter_inward[i]*sb+start[i]*cb*ct+end[i]*cb*st for i in range(3))
    r0=a.cbrt(R**3-3*R**2*depth)
    rchart=a.cbrt(R**3+3*R**2*(current.areal_material_thickness_m-depth)) if current.owner==archive.receiver_owner else r0
    inv=_prepared_inverse_rotation(archive.donor_axis if current.owner==archive.donor_owner else archive.receiver_axis,a.precision)
    pose=tuple(tuple(a.constant(inv[j][i]) for j in range(3)) for i in range(3))
    cert=cold.whole_domain_certificate.donor if current.owner==archive.donor_owner else cold.whole_domain_certificate.receiver
    base=tuple(cert.fixed_radial_scale*rchart*v for v in _mv(pose,q))
    t=beta.to_interval()/current.beta_cut_radians; t=I(max(F(0),t.lo),min(F(1),t.hi))
    shape=lambda z:(1-z)*(1-z)*(z+2)/2
    chi=a.constant(I(shape(t.hi),shape(t.lo)))
    delta=cold.whole_domain_certificate.seam_half_width_radians
    if tag=='interior': weights=(a.constant(1),)
    else:
        t=(a.theta().to_interval()+delta)/(2*delta); t=I(max(F(1,2),t.lo),min(F(1),t.hi))
        f=lambda z:6*z**5-15*z**4+10*z**3
        w=a.constant(I(f(t.lo),f(t.hi))); weights=(w,1-w)
    displacements=[]
    identity=tuple(tuple(a.constant(int(i==j)) for j in range(3)) for i in range(3))
    for bank,weight in zip(banks,weights):
        top=R+bank.roof_radial_displacement_m; rold=a.cbrt(top**3-3*R**2*depth)
        tau=depth+(R**3-top**3)/(3*R**2)
        omega=bank.angular_rate_per_physical_volume_depth_m
        norm=interval_sqrt(I(sum((v*v for v in omega),F(0))),a.precision+16)
        n=tuple(I(v)/norm for v in omega); K=((I(0),-n[2],n[1]),(n[2],I(0),-n[0]),(-n[1],n[0],I(0)))
        K2=_mm(K,K); spin=a.sin(tau*a.constant(norm)); omc=1-a.cos(tau*a.constant(norm))
        P=tuple(tuple(identity[i][j]+spin*K[i][j]+omc*K2[i][j] for j in range(3)) for i in range(3))
        e=_mv(_mm(P,bank.material_owner_pose_bounds),q); X=tuple(rold*v for v in e)
        displacements.append(tuple(weight*chi*v for v in e))
        displacements.extend(tuple(weight*chi*v/R for v in _mv(gen,X)) for gen in AFFINE_SYMMETRIC_GENERATORS)
    return indices,tips,base,tuple(displacements),G,cert


def _mixed_green_twojet(base,psi,G,coords,lam,omega,scale,radius,a,*,radial_lower=None,radial_upper=None):
    """Pure analytic joint Green twojet, local modal columns followed by lambda."""
    n=len(G); zero=a.constant(0); identity=tuple(tuple(a.constant(int(i==j)) for j in range(3)) for i in range(3))
    y=tuple(base[i]+sum((c*v[i] for c,v in zip(coords,psi)),zero) for i in range(3))
    B=tuple(tuple(identity[i][j]+sum((c*g[i][j] for c,g in zip(coords,G)),zero) for j in range(3)) for i in range(3))
    rb=_range(sum((v**2 for v in y),zero))
    if radial_lower is not None: rb=I(max(rb.lo,radial_lower**2),rb.hi)
    if radial_upper is not None: rb=I(rb.lo,min(rb.hi,radial_upper**2))
    if rb.lo<=0 or rb.lo>rb.hi: raise ValueError('The complete material box lacks a positive current radial chart')
    r=a.sqrt(a.constant(rb)); w=_cross(omega,y); k=F(1)/(scale**3*radius**2)
    d=_scale(-k*r,_outer(w,y)); ri=tuple(_dot(y,p)/r for p in psi); wi=tuple(_cross(omega,p) for p in psi)
    di=tuple(_scale(-k,_add(_add(_scale(ri[i],_outer(w,y)),_scale(r,_outer(wi[i],y))),_scale(r,_outer(w,psi[i])))) for i in range(n))
    A=_add(identity,_scale(lam,d)); Ft=_mm(A,B)
    fi=tuple(_add(_mm(A,g),_scale(lam,_mm(v,B))) for g,v in zip(G,di))+(_mm(d,B),)
    D=_add(tuple(tuple(sum((c*g[i][j] for c,g in zip(coords,G)),zero) for j in range(3)) for i in range(3)),_scale(lam,_mm(d,B)))
    E=_add(_sym(D),_scale(F(1,2),_mm(_tr(D),D)))
    Ei=tuple(_sym(_mm(_tr(Ft),v)) for v in fi)
    second=[]
    for i in range(n+1):
        row=[]
        for j in range(i,n+1):
            if i==j==n: fij=_scale(zero,identity)
            elif j==n: fij=_add(_mm(di[i],B),_mm(d,G[i]))
            else:
                rij=_dot(psi[i],psi[j])/r-_dot(y,psi[i])*_dot(y,psi[j])/r**3
                dij=_scale(-k,_add(_add(_add(_scale(rij,_outer(w,y)),
                    _scale(ri[i],_add(_outer(wi[j],y),_outer(w,psi[j])))),
                    _scale(ri[j],_add(_outer(wi[i],y),_outer(w,psi[i])))),
                    _scale(r,_add(_outer(wi[i],psi[j]),_outer(wi[j],psi[i])))))
                fij=_scale(lam,_add(_add(_mm(dij,B),_mm(di[i],G[j])),_mm(di[j],G[i])))
            row.append(_sym(_add(_mm(_tr(fi[i]),fi[j]),_mm(_tr(Ft),fij))))
        second.append(row)
    Eij=tuple(tuple(second[min(i,j)][abs(j-i)] for j in range(n+1)) for i in range(n+1))
    return E,Ei,Eij


def _whole_cells(system,p,theta_count,beta_count,depth_count,max_cells):
    key=(system.binding_sha256,p,theta_count,beta_count,depth_count,max_cells)
    if key in _GEOMETRY: return _GEOMETRY[key]
    ctx=_CONTEXTS[system.binding_sha256];cold,archive,continuum=ctx['cold_carrier'],ctx['archive'],ctx['continuum_system']
    needed=8*2*3*theta_count*beta_count*depth_count*2+4*2*3*depth_count
    if needed>max_cells: raise ValueError('Complete original-source traversal exceeds the declared cell budget')
    delta=cold.whole_domain_certificate.seam_half_width_radians; gamma=archive.whole_domain_certificate.gamma
    pi2=interval_pi(p+16)/2; cells=[]; prepmap={v.owner:v for v in ctx['preparations']}
    tipmap={v:i for i,v in enumerate(system.tip_coordinate_indices)}
    for material in continuum.banks:
        old=next(v for v in cold.banks if v.original_parent_id==material.original_parent_id)
        donor=material.owner==archive.donor_owner; omega=prepmap[material.owner].declaration.angular_rate_per_physical_volume_depth_m
        for layer,zlo,zhi in (('upper',F(0),material.elastic_layer_areal_depth_m),('lower',material.elastic_layer_areal_depth_m,material.areal_material_thickness_m)):
            lame,shear=material.lame_coefficients(layer)
            for tag,tlo,thi in seam_quarter_pieces(delta,p+16):
                alo,ahi=(I(0),I(delta)) if tag=='end_seam' else (tlo,thi)
                for zi in range(depth_count):
                    zb=I(zlo+(zhi-zlo)*F(zi,depth_count),zlo+(zhi-zlo)*F(zi+1,depth_count))
                    boxes=[]
                    if donor: boxes.append(('whole_donor_activation_collar',alo,ahi,I(0),I(2*gamma),False))
                    for ti in range(theta_count):
                        ta,tb=alo+(ahi-alo)*F(ti,theta_count),alo+(ahi-alo)*F(ti+1,theta_count)
                        for kind,blo,bhi in (('active_modal_composed_preparation',I(2*gamma if donor else 0),I(old.beta_cut_radians)),
                                            ('zero_modal_complement_preparation',I(old.beta_cut_radians),pi2)):
                            for bi in range(beta_count):
                                ba,bb=blo+(bhi-blo)*F(bi,beta_count),blo+(bhi-blo)*F(bi+1,beta_count)
                                boxes.append((kind,ta,tb,ba,bb,kind=='zero_modal_complement_preparation' and bi==beta_count-1))
                    for kind,ta,tb,ba,bb,pole in boxes:
                        a=_BoxAlgebra(I(ta.lo,tb.hi),p); beta=I(ba.lo,bb.hi);indices=tips=base=psi=G=()
                        caps=(); refmin=refmax=F(0); complement=()
                        if kind=='zero_modal_complement_preparation':
                            _,n=_owner_direction(cold,archive,material.original_parent_id,tag,a.constant(beta),a,p)
                            radial=a.constant(material.radius_m-3*zb+(0 if donor else 3*material.areal_material_thickness_m))
                            _,_,S2,_,S4,_,_,_=_relative_twist_kernel(n,radial,omega,(),lame,shear,a)
                            complement=(S2,S4)
                        else:
                            indices,localtips,base,psi,G,cert=_material_fields(ctx,material.original_parent_id,tag,a.constant(beta),a.constant(zb),a,
                                prepared=kind!='whole_donor_activation_collar')
                            tips=tuple((tipmap[indices[k]],k) for k in localtips)
                            r0=a.cbrt(material.radius_m**3-3*material.radius_m**2*a.constant(zb)).to_interval()
                            if kind=='whole_donor_activation_collar':
                                caps=tuple(_frobenius(g,p)/(cert.fixed_radial_scale*(1-cert.archive_carrier_gradient_deviation_bound)) for g in G)
                                G=tuple(tuple(tuple(a.constant(I(-cap,cap)) for _ in range(3)) for _ in range(3)) for cap in caps)
                                refmax=cert.fixed_radial_scale*r0.hi
                                contraction=interval_exp(-archive.dip_tangent_bounds*archive.whole_domain_certificate.maximum_beta_bounds,p+16)
                                refmin=cert.fixed_radial_scale*r0.lo*contraction.lo
                                base=tuple(a.constant(I(-refmax,refmax)) for _ in range(3))
                            else:
                                radial=a.cbrt(material.radius_m**3+3*material.radius_m**2*(
                                    (0 if donor else material.areal_material_thickness_m)-a.constant(zb))).to_interval()
                                refmin,refmax=cert.fixed_radial_scale*radial.lo,cert.fixed_radial_scale*radial.hi
                        volume=_cell_volume(material.radius_m,ta,tb,ba,bb,zb,p,pole)
                        metadata=dict(original_parent_id=material.original_parent_id,component_id=material.component_id,original_material_id=material.original_material_id,
                            owner=material.owner,layer=layer,kind=kind,theta_piece=tag,integration_angle_lo=ta,integration_angle_hi=tb,
                            beta_lo=ba,beta_hi=bb,beta_bounds=beta,depth_bounds_m=zb,upper_beta_is_exact_pole=pole,end_seam_axes_reflected=tag=='end_seam',
                            active_coordinate_indices=indices,global_tip_local_indices=tips,source_volume_bounds_m3=volume,
                            base_position_bounds_m=tuple(_range(v) for v in base),mode_displacement_columns=tuple(tuple(_range(v) for v in row) for row in psi),
                            relative_gradient_columns_per_m=tuple(_matrix_range(g) for g in G),
                            donor_collar_relative_gradient_frobenius_upper_per_m=caps,reference_radius_lower_m=refmin,reference_radius_upper_m=refmax,
                            activation_history_replaced_by_zero=False,original_material_cartesian_gradient_pullback_retained=True,
                            inactive_complement_half_norm_coefficients_j_m3=complement)
                        cells.append(dict(metadata=metadata,algebra=a,base=base,psi=psi,G=G,indices=indices,tips=tips,omega=omega,
                            scale=prepmap[material.owner].declaration.fixed_radial_scale,radius=material.radius_m,lame=lame,shear=shear,
                            reference_min=refmin,reference_max=refmax,complement=complement,volume=volume))
    if len(cells)!=needed: raise ValueError('The complete parent/layer/seam/support/collar partition changed')
    _GEOMETRY[key]=tuple(cells);return _GEOMETRY[key]


def _local_moments(cell,coordinates,lam,p):
    a=cell['algebra'];indices=cell['indices']; n=len(indices); pairs=tuple((i,j) for i in range(n+1) for j in range(i,n+1))
    if cell['complement']:
        S2,S4=cell['complement'];S=lam**2*S2+lam**4*S4;dS=2*lam*S2+4*lam**3*S4;ddS=2*S2+12*lam**2*S4
        return dict(r=(),dr=(),ddr=(),S=S,dS=(dS,),ddS=(ddS,),local_indices=(56,),pairs=((0,0),),tips=(),radial_lower=None,radial_upper=None)
    local=tuple(coordinates[i] for i in indices)
    ucap=sum((abs(c)*_vector_norm(psi,p) for c,psi in zip(local,cell['psi'])),F(0))
    rlo=cell['reference_min']-ucap; rhi=cell['reference_max']+ucap
    if rlo<=0: raise ValueError('The actual modal box exceeds its positive whole-source radial chart')
    E,Ei,Eij=_mixed_green_twojet(cell['base'],cell['psi'],cell['G'],local,lam,cell['omega'],cell['scale'],cell['radius'],a,
        radial_lower=rlo,radial_upper=rhi)
    tips=cell['tips'];H=tuple(_sym(cell['G'][k]) for _,k in tips)
    reference=_scale(a.constant(0),E)
    for h,(_,k) in zip(H,tips): reference=_add(reference,_scale(local[k],h))
    R=_add(E,_scale(-1,reference));tipdict={k:h for h,(_,k) in zip(H,tips)}
    Ri=tuple(_add(v,_scale(-1,tipdict[i])) if i in tipdict else v for i,v in enumerate(Ei))
    if not any(local) and not lam:
        R=_scale(a.constant(0),E)
        Ri=tuple(_scale(a.constant(0),E) if i in tipdict else v for i,v in enumerate(Ri))
    inner=lambda x,y:_tensor_inner(x,y,cell['lame'],cell['shear'])
    r=tuple(_range(inner(h,R)) for h in H)
    dr=tuple(tuple(_range(inner(h,v)) for v in Ri) for h in H)
    ddr=tuple(tuple(_range(inner(h,Eij[i][j])) for i,j in pairs) for h in H)
    S=_range(inner(R,R)/2)
    if S.hi<0: raise ValueError('Whole-source squared strain density cannot be negative')
    S=I(max(F(0),S.lo),S.hi)
    dS=tuple(_range(inner(v,R)) for v in Ri)
    ddS=tuple(_range(inner(Ri[i],Ri[j])+inner(Eij[i][j],R)) for i,j in pairs)
    return dict(r=r,dr=dr,ddr=ddr,S=S,dS=dS,ddS=ddS,local_indices=indices+(56,),pairs=pairs,tips=tuple(v for v,_ in tips),radial_lower=rlo,radial_upper=rhi)


@dataclass(frozen=True,slots=True)
class NativeCommonCoupledColdEnergyResponse:
    system_binding_sha256: str
    coordinate_order: tuple
    coordinate_units: tuple
    coordinate_increments56: tuple
    preparation_fraction: F
    energy_j: I
    gradient57: tuple
    hessian57: tuple
    beam_energy_j: I
    nonlocal_orthogonal_energy_j: I
    tip_projection_m: tuple
    tip_projection_first: tuple
    tip_projection_second: tuple
    literal_reference_energies_j: tuple
    literal_reference_forces_n: tuple
    tip_stiffnesses_n_m: tuple
    remainder_projection_n: tuple
    remainder_projection_first: tuple
    remainder_projection_second: tuple
    half_remainder_norm_j: I
    half_remainder_norm_first: tuple
    half_remainder_norm_second: tuple
    whole_source_volume_bounds_m3: I
    covered_source_volume_bounds_m3: I
    cell_receipts: tuple
    integration_options: tuple
    precision: int
    binding_sha256: str
    scope: str = 'COMPOSED_COLD56_AND_PHYSICAL_PREPARATION_LAMBDA_WHOLE_ORIGINAL_VOLUME_DECLARED899_POTENTIAL'
    whole_original_integral_certified: bool = True
    full_cold56_preparation_mixed_derivatives_computed: bool = True
    source_beams_included_once: bool = True
    original_gram_and_inverse_reused_without_quadrature: bool = True
    authentic_original_volume_included_once: bool = True
    complete_live_and_archive_parent_material_retained: bool = True
    complete_modal_zero_clamp_complement_integrated: bool = True
    donor_collar_uniform_actual_map_enclosure_certified: bool = True
    mixed_physical_force_equilibrium_certified: bool = False
    historical_archived_three_dimensional_prestress_recovered: bool = False
    full_system_preparation_work_paid: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    clocks_advanced: bool = False
    accepted_new_native_intervals: int = 0


def _assemble(system,coordinates,lam,r,dr,ddr,S,dS,ddS,receipts,options,p):
    source=_CONTEXTS[system.binding_sha256]['source'];A=source.inverse_gram_bounds_m_n;n=57
    mv=lambda v:tuple(_rnd(sum((x*y for x,y in zip(row,v)),I(0)),p) for row in A)
    pr=mv(r);projection=tuple(_rnd(I(coordinates[k])+v,p) for k,v in zip(system.tip_coordinate_indices,pr))
    ar=tuple(tuple(_rnd(sum((A[k][l]*dr[l][i] for l in range(8)),I(0)),p) for i in range(n)) for k in range(8))
    dp=tuple(tuple(_rnd(ar[k][i]+int(i==tip),p) for i in range(n)) for k,tip in enumerate(system.tip_coordinate_indices))
    ddp=tuple(tuple(tuple(_rnd(sum((A[k][l]*ddr[l][i][j] for l in range(8)),I(0)),p) for j in range(n)) for i in range(n)) for k in range(8))
    force=tuple(_rnd(f+K*x,p) for f,K,x in zip(system.literal_reference_forces_n,system.tip_stiffnesses_n_m,projection))
    square=lambda v:I(F(0) if v.lo<=0<=v.hi else min(v.lo**2,v.hi**2),max(v.lo**2,v.hi**2))
    beam=_rnd(sum((I(E0)+f0*x+K*square(x)/2 for E0,f0,K,x in zip(system.literal_reference_energies_j,system.literal_reference_forces_n,system.tip_stiffnesses_n_m,projection)),I(0)),p)
    residual=_rnd(S-sum((x*y for x,y in zip(r,pr)),I(0))/2,p)
    if residual.hi<0: raise ValueError('Integrated moments contradict the orthogonal positive cold strain complement')
    residual=I(max(F(0),residual.lo),residual.hi)
    coeff=tuple(_rnd(v-w,p) for v,w in zip(mv(force),pr))
    gradient=tuple(_rnd(dS[i]+sum((force[k]*dp[k][i]-pr[k]*dr[k][i] for k in range(8)),I(0)),p) for i in range(n))
    H=[[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i,n):
            H[i][j]=H[j][i]=_rnd(ddS[i][j]+sum((system.tip_stiffnesses_n_m[k]*dp[k][i]*dp[k][j]
                -dr[k][i]*ar[k][j]+coeff[k]*ddr[k][i][j] for k in range(8)),I(0)),p)
    energy=_rnd(beam+residual,p)
    if not any(coordinates) and not lam:
        beam=energy=I(sum(system.literal_reference_energies_j,F(0)));residual=I(0)
        gradient=list(gradient)
        for index,f in zip(system.tip_coordinate_indices,system.literal_reference_forces_n): gradient[index]=I(f)
        gradient=tuple(gradient)
    volume=interval_pi(p+16)/2*sum((b.radius_m**2*b.areal_material_thickness_m for b in _CONTEXTS[system.binding_sha256]['continuum_system'].banks),F(0))
    covered=_rnd(sum((c['source_volume_bounds_m3'] for c in receipts),I(0)),p)
    if not covered.lo<=volume.lo<=volume.hi<=covered.hi: raise ValueError('The full source-volume partition is incomplete')
    response=NativeCommonCoupledColdEnergyResponse(system.binding_sha256,system.coordinate_order,system.coordinate_units,coordinates,lam,energy,gradient,tuple(tuple(v) for v in H),
        beam,residual,projection,dp,ddp,system.literal_reference_energies_j,system.literal_reference_forces_n,system.tip_stiffnesses_n_m,
        r,dr,ddr,S,dS,ddS,volume,covered,receipts,options,p,'')
    response=replace(response,binding_sha256=_seal(response));_RESPONSES[response.binding_sha256]=response
    return response


def evaluate_native_common_coupled_cold_energy(system,coordinate_increments56=None,preparation_fraction=F(0),*,precision=96,
        theta_cell_count=4,beta_cell_count=1,depth_cell_count=1,max_cells=4096):
    validate_native_common_coupled_cold_energy(system);p=_precision(precision);coordinates=_coordinates(system,coordinate_increments56,preparation_fraction);lam=preparation_fraction
    if any(type(v) is not int or v<1 for v in (theta_cell_count,beta_cell_count,depth_cell_count,max_cells)):
        raise ValueError('Whole-source subdivisions and budget must be positive literal integers')
    options=(theta_cell_count,beta_cell_count,depth_cell_count,max_cells);key=(system.binding_sha256,coordinates,lam,p,options)
    if key in _CACHE: return validate_native_common_coupled_cold_energy_response(_CACHE[key],system=system)
    zero=I(0);r=[zero]*8;dr=[[zero]*57 for _ in range(8)];ddr=[[[zero]*57 for _ in range(57)] for _ in range(8)]
    S=zero;dS=[zero]*57;ddS=[[zero]*57 for _ in range(57)];receipts=[]
    for cell in _whole_cells(system,p,*options):
        raw=_local_moments(cell,coordinates,lam,p);volume=cell['volume'];idx=raw['local_indices'];pairs=raw['pairs']
        rv=tuple(_rnd(v*volume,p) for v in raw['r']);rd=tuple(tuple(_rnd(v*volume,p) for v in row) for row in raw['dr']);rdd=tuple(tuple(_rnd(v*volume,p) for v in row) for row in raw['ddr'])
        sv=_rnd(raw['S']*volume,p);sd=tuple(_rnd(v*volume,p) for v in raw['dS']);sdd=tuple(_rnd(v*volume,p) for v in raw['ddS'])
        for k,v,d,h in zip(raw['tips'],rv,rd,rdd):
            r[k]=_rnd(r[k]+v,p)
            for i,x in zip(idx,d):dr[k][i]=_rnd(dr[k][i]+x,p)
            for (i,j),x in zip(pairs,h):
                gi,gj=idx[i],idx[j];ddr[k][gi][gj]=_rnd(ddr[k][gi][gj]+x,p);ddr[k][gj][gi]=ddr[k][gi][gj]
        S=_rnd(S+sv,p)
        for i,v in zip(idx,sd):dS[i]=_rnd(dS[i]+v,p)
        for (i,j),v in zip(pairs,sdd):
            gi,gj=idx[i],idx[j];ddS[gi][gj]=_rnd(ddS[gi][gj]+v,p);ddS[gj][gi]=ddS[gi][gj]
        receipt=dict(cell['metadata'],local_coordinate_indices=idx,local_symmetric_pairs=pairs,global_projection_tip_indices=raw['tips'],
            current_radius_lower_m=raw['radial_lower'],current_radius_upper_m=raw['radial_upper'],
            projection_bounds_n=rv,projection_first_bounds=rd,projection_second_bounds=rdd,
            half_norm_bounds_j=sv,half_norm_first_bounds=sd,half_norm_second_bounds=sdd,
            whole_material_box_enclosure_certified=True,directed_ieee_basic_arithmetic=True,transcendental_endpoints_rationally_certified=True)
        receipt['receipt_sha256']=hashlib.sha256(_json(receipt).encode()).hexdigest();receipts.append(receipt)
    response=_assemble(system,coordinates,lam,tuple(r),tuple(tuple(v) for v in dr),tuple(tuple(tuple(v) for v in block) for block in ddr),
        S,tuple(dS),tuple(tuple(v) for v in ddS),tuple(receipts),options,p)
    _CACHE[key]=response;return response


def validate_native_common_coupled_cold_energy_response(response,*,system=None):
    if type(response) is not NativeCommonCoupledColdEnergyResponse or _RESPONSES.get(response.binding_sha256) is not response or _seal(response)!=response.binding_sha256:
        raise ValueError('An altered, resealed or unregistered actual whole-source cold response is not authorized')
    if system is not None:
        validate_native_common_coupled_cold_energy(system)
        if response.system_binding_sha256!=system.binding_sha256: raise ValueError('The response belongs to another whole-source system')
    return response


def native_common_coupled_cold_energy_to_dict(system):
    validate_native_common_coupled_cold_energy(system);return dict(format=FORMAT,**_payload(system))


def native_common_coupled_cold_energy_from_dict(raw,source899,cold_carrier,continuum_system,preparations,*,source923,provenance):
    expected=bind_native_common_coupled_cold_energy(source899,cold_carrier,continuum_system,preparations,source923=source923,provenance=provenance)
    if type(raw) is not dict or _json(raw)!=_json(native_common_coupled_cold_energy_to_dict(expected)): raise ValueError('The full coupled cold restart changed a source, material, declaration or clock')
    return expected


def native_common_coupled_cold_energy_response_to_dict(response):
    validate_native_common_coupled_cold_energy_response(response);return dict(format=RESPONSE_FORMAT,**_payload(response))


def native_common_coupled_cold_energy_response_from_dict(raw,*,system,expected_coordinate_increments56,expected_preparation_fraction,
        precision=96,theta_cell_count=4,beta_cell_count=1,depth_cell_count=1,max_cells=4096):
    expected=evaluate_native_common_coupled_cold_energy(system,expected_coordinate_increments56,expected_preparation_fraction,precision=precision,
        theta_cell_count=theta_cell_count,beta_cell_count=beta_cell_count,depth_cell_count=depth_cell_count,max_cells=max_cells)
    if type(raw) is not dict or _json(raw)!=_json(native_common_coupled_cold_energy_response_to_dict(expected)): raise ValueError('The whole-source response restart changed a measure, moment, coordinate or result')
    return expected


def evaluate_native_common_coupled_cold_point(system,original_parent_id,theta_piece,angle,beta,material_depth_m,
        coordinate_increments56=None,preparation_fraction=F(0),*,precision=96,layer=None):
    """Actual material-label mixed jet away from the archive activation collar.

    End seam angles use the reflected original-quarter axes, as in every
    whole-source receipt. A point is diagnostic geometry, never volume mass.
    """
    validate_native_common_coupled_cold_energy(system);p=_precision(precision)
    coordinates=_coordinates(system,coordinate_increments56,preparation_fraction);lam=preparation_fraction
    if any(type(v) is not F for v in (angle,beta,material_depth_m)):
        raise ValueError('A diagnostic original-source label requires exact Fraction parameters')
    ctx=_CONTEXTS[system.binding_sha256];cold,archive,continuum=ctx['cold_carrier'],ctx['archive'],ctx['continuum_system']
    material=next((b for b in continuum.banks if b.original_parent_id==original_parent_id),None)
    if material is None: raise ValueError('Every cold diagnostic label needs an authentic full original parent')
    old=next(b for b in cold.banks if b.original_parent_id==original_parent_id)
    donor=material.owner==archive.donor_owner;delta=cold.whole_domain_certificate.seam_half_width_radians
    if theta_piece not in ('interior','start_seam','end_seam'): raise ValueError('A diagnostic label needs an authentic quarter/seam chart')
    if theta_piece=='interior':
        if not delta<=angle<=(interval_pi(p+16)/2-I(delta)).lo: raise ValueError('The diagnostic interior angle leaves its exact source chart')
    elif not 0<=angle<=delta: raise ValueError('The diagnostic seam angle leaves its exact reflected source chart')
    if not (2*archive.whole_domain_certificate.gamma if donor else F(0))<=beta<=old.beta_cut_radians:
        raise ValueError('Split diagnostic labels outside the genuine active source support or donor collar')
    if not 0<=material_depth_m<=material.areal_material_thickness_m: raise ValueError('The diagnostic label leaves its original material depth')
    actual_layer='upper' if material_depth_m<=material.elastic_layer_areal_depth_m else 'lower'
    if layer is not None and layer!=actual_layer: raise ValueError('The diagnostic layer differs from its authentic original cut')
    a=_BoxAlgebra(I(angle),p);indices,tips,base,psi,G,cert=_material_fields(ctx,original_parent_id,theta_piece,a.constant(beta),a.constant(material_depth_m),a)
    prep=next(v for v in ctx['preparations'] if v.owner==material.owner)
    r0=a.cbrt(material.radius_m**3-3*material.radius_m**2*a.constant(material_depth_m))
    rchart=a.cbrt(material.radius_m**3+3*material.radius_m**2*((0 if donor else material.areal_material_thickness_m)-a.constant(material_depth_m)))
    refmin,refmax=cert.fixed_radial_scale*rchart.to_interval().lo,cert.fixed_radial_scale*rchart.to_interval().hi
    local=tuple(coordinates[i] for i in indices);u=sum((abs(c)*_vector_norm(v,p) for c,v in zip(local,psi)),F(0))
    E,Ei,Eij=_mixed_green_twojet(base,psi,G,local,lam,prep.declaration.angular_rate_per_physical_volume_depth_m,cert.fixed_radial_scale,material.radius_m,a,
        radial_lower=refmin-u,radial_upper=refmax+u)
    banks,start,end=_piece_banks(cold,original_parent_id,theta_piece);sb,cb=a.sin(a.constant(beta)),a.cos(a.constant(beta));st,ct=a.sine_theta(),a.cosine_theta()
    q=tuple(old.original_quarter_inward[i]*sb+start[i]*cb*ct+end[i]*cb*st for i in range(3))
    inv=_prepared_inverse_rotation(archive.donor_axis if donor else archive.receiver_axis,p);pose=tuple(tuple(a.constant(inv[j][i]) for j in range(3)) for i in range(3))
    identity=tuple(tuple(a.constant(int(i==j)) for j in range(3)) for i in range(3));qq=_outer(q,q)
    localA=identity if donor else _add(_scale(rchart/r0,_add(identity,_scale(-1,qq))),_scale(r0*r0/(rchart*rchart),qq))
    Aref=_scale(cert.fixed_radial_scale,_mm(pose,localA)); originalG=tuple(_mm(g,Aref) for g in G)
    result=dict(system_binding_sha256=system.binding_sha256,original_parent_id=original_parent_id,component_id=material.component_id,
        original_material_id=material.original_material_id,owner=material.owner,layer=actual_layer,theta_piece=theta_piece,
        integration_angle=angle,beta=beta,material_depth_m=material_depth_m,coordinate_increments56=coordinates,preparation_fraction=lam,
        local_coordinate_indices=indices+(56,),local_tip_indices=tips,base_position_bounds_m=tuple(_range(v) for v in base),
        mode_displacement_columns=tuple(tuple(_range(v) for v in row) for row in psi),relative_gradient_columns_per_m=tuple(_matrix_range(g) for g in G),
        prepared_reference_original_gradient_bounds=_matrix_range(Aref),mode_original_gradient_columns_per_m=tuple(_matrix_range(g) for g in originalG),
        green_strain_bounds=_matrix_range(E),green_strain_first=tuple(_matrix_range(v) for v in Ei),
        green_strain_second=tuple(tuple(_matrix_range(v) for v in row) for row in Eij),
        original_measure_used_for_energy=True,diagnostic_point_used_as_volume_weight=False,
        donor_historical_allocation_zero_proved_by_whole_support_separation=donor,
        preparation_angular_rate_per_physical_volume_depth_m=prep.declaration.angular_rate_per_physical_volume_depth_m,
        fixed_radial_scale=cert.fixed_radial_scale,radius_m=material.radius_m,precision=p)
    result['binding_sha256']=hashlib.sha256(_json(result).encode()).hexdigest();_POINTS[result['binding_sha256']]=result
    return result


def native_common_coupled_cold_point_to_dict(point):
    if type(point) is not dict or _POINTS.get(point.get('binding_sha256')) is not point or hashlib.sha256(_json({k:v for k,v in point.items() if k!='binding_sha256'}).encode()).hexdigest()!=point['binding_sha256']:
        raise ValueError('An altered or undeclared source-bound diagnostic cold point is not authorized')
    return dict(format='native-common-coupled-cold-point-1',**_payload(point))


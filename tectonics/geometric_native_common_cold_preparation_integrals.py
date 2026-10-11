"""Whole original-material moments of the physical919 cold depth twist.

This integrates the NEW T_lambda motion on the frozen zero895 prepared map,
with the frozen899 isotropic law in its prepared Cartesian frame.  It does
not rerun old quadrature or redefine its Gram matrix.  The inactive modal
complement is included: the physical twist acts there too.  Archive cohorts
are part of their original parents and are never added as extra volumes.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F

from .geometric_native_pose import FractionInterval as I, interval_pi, interval_sqrt, _rounded_interval
from .geometric_native_common_cold_variational_work import (
    _source, _seal, _payload, _json,
)
from .geometric_native_common_cold_depth_preparation import _validate_bound
from .geometric_native_common_cold_material_carrier import _saved_json_kinds
from .geometric_native_common_cold_continuum_energy import (
    _registered, native_common_cold_continuum_energy_to_dict,
    _BoxAlgebra, _BoxScalar, _piece_banks, _prepared_modal_fields,
    _prepared_inverse_rotation, _sc_trig, seam_quarter_pieces,
)

FORMAT = 'native-common-cold-preparation-integral-1'
SCOPE = 'NEW_PHYSICAL_DEPTH_TWIST_WHOLE_COLD_ZERO_PREPARED_MAP_MOMENTS_ONLY'
MEASURE = 'AUTHENTIC_ORIGINAL_AREAL_VOLUME_R_SQUARED_DOMEGA_DZ_ONCE'
_REGISTRY, _CONTEXTS, _CACHE = {}, {}, {}


def _sum(values):
    return sum(values, I(0))


def _mm(a, b):
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(3)), 0)
                       for j in range(3)) for i in range(3))


def _mv(a, b):
    return tuple(sum((v*w for v, w in zip(row, b)), 0) for row in a)


def _sym(a):
    return tuple(tuple((a[i][j]+a[j][i])/2 for j in range(3)) for i in range(3))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _range(value):
    return value.to_interval() if type(value) is _BoxScalar else value


def _matrix_range(value):
    return tuple(tuple(_range(x) for x in row) for row in value)


def _positive(value):
    if value.hi < 0:
        raise ValueError('A physical squared preparation norm cannot be negative')
    return I(max(F(0), value.lo), value.hi)


def _symmetric_inner(a, b):
    return sum((a[i][i]*b[i][i] for i in range(3)), 0) + 2*sum(
        (a[i][j]*b[i][j] for i in range(3) for j in range(i+1, 3)), 0)


def _relative_twist_kernel(unit_direction, radius_cubed_over_R_squared,
                           omega, tip_strains, lame_pa, shear_pa, algebra):
    """Exact relative Green coefficients; unit-vector identities stay correlated.

    d=-rho^3/R^2 (Omega cross n) outer n.  Rigid Q_lambda drops from
    DT_lambda.T DT_lambda.  tr(L)=0 and L:C:Q=0 exactly.  The invariant
    a=|Omega|^2-(Omega dot n)^2 is nonnegative for genuine unit n; intersecting
    its enclosure with [0,|Omega|^2] retains that exact identity.
    """
    n = tuple(algebra.constant(x) for x in unit_direction)
    rate = tuple(algebra.constant(x) for x in omega)
    t = algebra.constant(radius_cubed_over_R_squared)
    w = _cross(rate, n)
    omega_squared = sum((x*x for x in omega), F(0))
    radial_projection = sum((a*b for a, b in zip(rate, n)), 0)
    invariant = (algebra.constant(omega_squared)-radial_projection**2).to_interval()
    cross_invariant = sum((v**2 for v in w), 0).to_interval()
    lo = max(F(0), invariant.lo, cross_invariant.lo)
    hi = min(omega_squared, invariant.hi, cross_invariant.hi)
    if lo > hi:
        raise ValueError('A preparation range contradicts its authentic unit-direction identity')
    a = algebra.constant(I(lo, hi))
    d = tuple(tuple(-t*w[i]*n[j] for j in range(3)) for i in range(3))
    L = _sym(d)
    Q = tuple(tuple(t**2*a*n[i]*n[j]/2 for j in range(3)) for i in range(3))
    r1 = tuple(2*shear_pa*_symmetric_inner(H, L) for H in tip_strains)
    trace_Q = t**2*a/2
    r2 = tuple(lame_pa*sum((H[i][i] for i in range(3)), 0)*trace_Q
               +2*shear_pa*_symmetric_inner(H, Q) for H in tip_strains)
    S2 = _positive((shear_pa*t**2*a/2).to_interval())
    S4 = _positive(((lame_pa+2*shear_pa)*t**4*a**2/8).to_interval())
    return (tuple(_range(x) for x in r1), tuple(_range(x) for x in r2),
            S2, I(0), S4, _matrix_range(L), _matrix_range(Q), I(lo, hi))


def _collar_channels(original_tip_gradient_caps, *, fixed_scale, carrier_deviation,
                     omega_norm_upper, original_radius_cubed_over_R_squared_upper,
                     lame_pa, shear_pa, precision):
    """Uniform whole activation collar, retaining the actual nonlinear887 map.

    Donor chi887 contracts the original radius, hence ||d||<=|Omega|r0^3/R^2.
    The proved ||A887-Qowner||<D gives ||H_i||F<=g_i/[s(1-D)].
    No activation fraction, midpoint or zero-history donor surrogate is used.
    """
    if not (type(fixed_scale) is F and fixed_scale > 0
            and type(carrier_deviation) is F and 0 <= carrier_deviation < 1
            and type(omega_norm_upper) is F and omega_norm_upper >= 0
            and type(original_radius_cubed_over_R_squared_upper) is F
            and original_radius_cubed_over_R_squared_upper > 0
            and type(original_tip_gradient_caps) is tuple
            and all(type(x) is F and x >= 0 for x in original_tip_gradient_caps)):
        raise ValueError('The uniform collar requires authentic positive radial and inverse-gradient bounds')
    hcaps = tuple(g/(fixed_scale*(1-carrier_deviation)) for g in original_tip_gradient_caps)
    dcap = omega_norm_upper*original_radius_cubed_over_R_squared_upper
    c = max(2*shear_pa, abs(3*lame_pa+2*shear_pa))
    lcap = dcap/interval_sqrt(I(2), precision+16).lo
    qcap = dcap*dcap/2
    signed = lambda cap: _rounded_interval(I(-cap, cap), precision)
    return (tuple(signed(c*h*lcap) for h in hcaps),
            tuple(signed(c*h*qcap) for h in hcaps),
            _rounded_interval(I(0, shear_pa*dcap**2/2), precision), I(0),
            _rounded_interval(I(0, (lame_pa+2*shear_pa)*dcap**4/8), precision),
            hcaps, dcap)


@dataclass(frozen=True, slots=True)
class NativeColdPreparationCell:
    original_parent_id: str
    component_id: str
    original_material_id: str
    owner: int
    layer: str
    kind: str
    theta_piece: str
    integration_angle_lo: I
    integration_angle_hi: I
    beta_lo: I
    beta_hi: I
    beta_bounds: I
    depth_bounds_m: I
    upper_beta_is_exact_pole: bool
    end_seam_axes_reflected: bool
    global_tip_indices: tuple
    tip_strain_bounds_per_m: tuple
    relative_green_linear_bounds: tuple
    relative_green_quadratic_bounds: tuple
    angular_invariant_bounds_per_m2: I | None
    collar_tip_strain_frobenius_upper_per_m: tuple
    collar_twist_gradient_frobenius_upper: F | None
    source_volume_bounds_m3: I
    density_bounds: tuple
    integral_bounds: tuple


@dataclass(frozen=True, slots=True)
class NativeColdPreparationIntegral:
    source_binding_sha256: str
    source_raw_sha256: str
    source_packet_sha256: str
    cold_system_binding_sha256: str
    continuum_system_binding_sha256: str
    archive_carrier_model_sha256: str
    preparation_binding_sha256s: tuple
    preparation_reference_snapshot_sha256: str
    original_parent_ids: tuple
    original_material_ids: tuple
    component_ids: tuple
    nonempty_archive_domain_ids: tuple
    exact_layer_cuts_m: tuple
    source_clocks_s: tuple
    zero_prepared_modal_coordinates_m: tuple
    material_reference_measure: str
    whole_source_volume_bounds_m3: I
    covered_source_volume_bounds_m3: I
    r1_bounds_n: tuple
    r2_bounds_n: tuple
    S2_bounds_j: I
    S3_bounds_j: I
    S4_bounds_j: I
    inactive_complement_S2_bounds_j: I
    inactive_complement_S4_bounds_j: I
    cell_receipts: tuple
    donor_activation_collar_beta_upper_radians: F
    donor_carrier_global_gradient_deviation_bound: F
    precision: int
    theta_cell_count: int
    beta_cell_count: int
    depth_cell_count: int
    max_cells: int
    provenance: str
    binding_sha256: str
    scope: str = SCOPE
    whole_original_integral_certified: bool = True
    authentic_original_volume_included_once: bool = True
    complete_modal_zero_clamp_complement_integrated: bool = True
    inactive_complement_tip_strain_identically_zero: bool = True
    complete_original_live_and_archive_material_retained: bool = True
    original_gram_matrix_and_inverse_reused_without_quadrature: bool = True
    relative_prepared_cartesian_green_strain_used: bool = True
    isotropic_radial_twist_cubic_moment_exactly_zero: bool = True
    donor_collar_uniform_enclosure_certified: bool = True
    whole_interval_box_range_certified: bool = True
    directed_ieee_basic_arithmetic_certified: bool = True
    transcendental_endpoints_rationally_certified: bool = True
    gauss_difference_used_as_error_certificate: bool = False
    force_equilibrium_precision_certified: bool = False
    historical_archived_three_dimensional_prestress_recovered: bool = False
    reference_preparation_work_paid: bool = False
    full_system_preparation_work_certified: bool = False
    physical_source_force_prestress_work_handover_certified: bool = False
    physical_material_admission_authorized: bool = False
    whole_clock_all_pair_admission_authorized: bool = False
    new_mixed_force_equilibrium_certified: bool = False
    physical_elapsed_time_added: bool = False
    accepted_original_native_intervals: int = 1
    accepted_new_native_intervals: int = 0


def _bindings(source, cold_system, continuum_system, preparations):
    _source(source)
    cold, archive = _registered(continuum_system)
    if cold is not cold_system:
        raise ValueError('The whole source continuum and supplied895 carrier must be the same registered field')
    raw = _json(native_common_cold_continuum_energy_to_dict(continuum_system))
    if raw != source.complete_source_system_json:
        raise ValueError('A different continuum declaration cannot inherit the frozen899 source law')
    if type(preparations) is not tuple or len(preparations) != 2:
        raise ValueError('Both authentic owner preparations in owner order are required')
    for prep in preparations:
        _validate_bound(prep)
    if tuple(x.owner for x in preparations) != tuple(sorted((archive.donor_owner, archive.receiver_owner))):
        raise ValueError('Both authentic owner preparations in owner order are required')
    snapshots = set()
    for prep in preparations:
        cert = (cold.whole_domain_certificate.donor if prep.owner == archive.donor_owner
                else cold.whole_domain_certificate.receiver)
        banks = tuple(b for b in cold.banks if b.owner == prep.owner)
        if (prep.system_binding_sha256 != cold.binding_sha256
                or prep.archive_carrier_model_sha256 != archive.model_sha256
                or prep.original_parent_ids != cert.parent_ids
                or prep.source_clocks_s != (cold.source_geometry_elapsed_time_s,
                    cold.contact_geometry_elapsed_time_s, cold.memory_elapsed_time_s,
                    cold.material_geometry_elapsed_time_s)
                or prep.declaration.fixed_radial_scale != cert.fixed_radial_scale
                or any(b.radius_m != prep.declaration.radius_m or
                    b.angular_rate_per_physical_volume_depth_m != prep.declaration.angular_rate_per_physical_volume_depth_m
                    for b in banks)):
            raise ValueError('A preparation changed the complete owner, source clocks, radius or rate')
        snapshots.add(prep.reference_snapshot_sha256)
    if len(snapshots) != 1:
        raise ValueError('Both preparations require the same authentic common source reference')
    if (tuple(b.original_parent_id for b in continuum_system.banks) != tuple('octant:'+str(i) for i in range(8))
            or len(cold.whole_domain_certificate.complete_archive_domain_ids) != 3
            or tuple(b.tip_coordinate_index for b in continuum_system.banks) != source.tip_coordinate_indices
            or not cold.whole_domain_certificate.complete_live_and_archive_material_retained):
        raise ValueError('All eight original parents, three archives and genuine tip coordinates are required')
    return cold, archive


def _owner_direction(cold, archive, parent, tag, beta, algebra, precision):
    banks, start, end = _piece_banks(cold, parent, tag)
    bank = banks[0]
    sb, cb = algebra.sin(beta), algebra.cos(beta)
    st, ct = algebra.sine_theta(), algebra.cosine_theta()
    q = tuple(bank.original_quarter_inward[i]*sb
              +start[i]*cb*ct+end[i]*cb*st for i in range(3))
    axis = archive.donor_axis if bank.owner == archive.donor_owner else archive.receiver_axis
    inverse = _prepared_inverse_rotation(axis, precision)
    pose = tuple(tuple(algebra.constant(inverse[j][i]) for j in range(3)) for i in range(3))
    return bank, _mv(pose, q)


def _sin_endpoint(value, precision, pole=False):
    return I(1) if pole else _sc_trig(value, precision)[0]


def _cell_volume(radius_m, angle_lo, angle_hi, beta_lo, beta_hi, depth_bounds,
                 precision, upper_beta_is_pole=False):
    value = radius_m**2*(angle_hi-angle_lo)*(
        _sin_endpoint(beta_hi, precision, upper_beta_is_pole)
        -_sin_endpoint(beta_lo, precision))*depth_bounds.width
    if value.lo < 0:
        raise ValueError('A whole original-material box must have positive source measure')
    return _rounded_interval(value, precision)


def integrate_native_common_cold_preparation_moments(source, cold_system, continuum_system,
        preparations, *, precision=96, theta_cell_count=4, beta_cell_count=1,
        depth_cell_count=1, max_cells=4096,
        provenance='Whole original-source physical919 depth-twist moments on frozen zero895 prepared map'):
    """Enclose NEW moments over all eight complete original parents once.

    This path starts at zero56 amplitudes of899.  It covers the actual three
    archive cohorts through their complete original parents, including the
    whole activated donor collar.  It does not integrate old forces, beams,
    Gram kernels or historical constitutive controls again.
    """
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError('Whole preparation precision must be a literal integer in [96,512]')
    if any(type(x) is not int or x < 1 for x in (theta_cell_count, beta_cell_count, depth_cell_count, max_cells)):
        raise ValueError('Whole preparation subdivision counts must be positive literal integers')
    if type(provenance) is not str or not provenance.strip():
        raise ValueError('Whole preparation integration requires declared source provenance')
    needed = 8*2*3*theta_cell_count*beta_cell_count*depth_cell_count*2 + 4*2*3*depth_cell_count
    if needed > max_cells:
        raise ValueError('The complete preparation traversal exceeds its declared cell budget')
    cold, archive = _bindings(source, cold_system, continuum_system, preparations)
    key = (source.binding_sha256, cold.binding_sha256, continuum_system.binding_sha256,
        tuple(p.binding_sha256 for p in preparations), precision, theta_cell_count,
        beta_cell_count, depth_cell_count, max_cells, provenance)
    if key in _CACHE:
        return validate_native_common_cold_preparation_integral(source, _CACHE[key])
    gamma = archive.whole_domain_certificate.gamma
    delta = cold.whole_domain_certificate.seam_half_width_radians
    pi2 = interval_pi(precision+16)/2
    tipmap = {b.tip_coordinate_index: i for i, b in enumerate(continuum_system.banks)}
    prepmap = {p.owner: p for p in preparations}
    receipts = []
    for material in continuum_system.banks:
        old = next(b for b in cold.banks if b.original_parent_id == material.original_parent_id)
        prep = prepmap[material.owner]
        omega = prep.declaration.angular_rate_per_physical_volume_depth_m
        omega_norm = interval_sqrt(I(sum((v*v for v in omega), F(0))), precision+16).hi
        donor = material.owner == archive.donor_owner
        bcut = old.beta_cut_radians
        if not 0 < bcut < pi2.lo or (donor and not 2*gamma < bcut):
            raise ValueError('The complete original parent has an unresolved real modal support or collar cut')
        for layer, zlo, zhi in (('upper', F(0), material.elastic_layer_areal_depth_m),
                ('lower', material.elastic_layer_areal_depth_m, material.areal_material_thickness_m)):
            lam, mu = material.lame_coefficients(layer)
            if mu <= 0 or lam+2*mu <= 0:
                raise ValueError('Preparation density requires the authentic positive layer tensor')
            for tag, tlo, thi in seam_quarter_pieces(delta, precision+16):
                alo, ahi = (I(0), I(delta)) if tag == 'end_seam' else (tlo, thi)
                for zi in range(depth_cell_count):
                    zb = I(zlo+(zhi-zlo)*F(zi, depth_cell_count),
                           zlo+(zhi-zlo)*F(zi+1, depth_cell_count))
                    boxes = []
                    if donor:
                        boxes.append(('whole_donor_activation_collar', alo, ahi, I(0), I(2*gamma), False))
                    for ti in range(theta_cell_count):
                        ta = alo+(ahi-alo)*F(ti, theta_cell_count)
                        tb = alo+(ahi-alo)*F(ti+1, theta_cell_count)
                        for kind, blo, bhi in (('active_modal_preparation', I(2*gamma if donor else 0), I(bcut)),
                                              ('zero_modal_complement_preparation', I(bcut), pi2)):
                            for bi in range(beta_cell_count):
                                ba = blo+(bhi-blo)*F(bi, beta_cell_count)
                                bb = blo+(bhi-blo)*F(bi+1, beta_cell_count)
                                pole = kind == 'zero_modal_complement_preparation' and bi == beta_cell_count-1
                                boxes.append((kind, ta, tb, ba, bb, pole))
                    for kind, ta, tb, ba, bb, pole in boxes:
                        beta_bounds = I(ba.lo, bb.hi)
                        algebra = _BoxAlgebra(I(ta.lo, tb.hi), precision)
                        local_r1, local_r2 = (), ()
                        tips = (); strains = (); L = (); Q = (); angular_invariant = None
                        hcaps = (); dcap = None
                        if kind == 'whole_donor_activation_collar':
                            active, local_tips, G = _prepared_modal_fields(continuum_system, cold, archive,
                                material.original_parent_id, tag, algebra.constant(beta_bounds),
                                algebra.constant(zb), algebra, prepared=False)
                            tips = tuple(tipmap[active[i]] for i in local_tips)
                            gcaps = tuple(interval_sqrt(I(sum((max(abs(v.lo), abs(v.hi))**2
                                for row in _matrix_range(G[i]) for v in row), F(0))), precision+16).hi
                                for i in local_tips)
                            cert = cold.whole_domain_certificate.donor
                            local_r1, local_r2, s2, s3, s4, hcaps, dcap = _collar_channels(gcaps,
                                fixed_scale=cert.fixed_radial_scale,
                                carrier_deviation=cert.archive_carrier_gradient_deviation_bound,
                                omega_norm_upper=omega_norm,
                                original_radius_cubed_over_R_squared_upper=material.radius_m-3*zb.lo,
                                lame_pa=lam, shear_pa=mu, precision=precision)
                        else:
                            oldbank, n = _owner_direction(cold, archive, material.original_parent_id, tag,
                                algebra.constant(beta_bounds), algebra, precision)
                            H = ()
                            if kind == 'active_modal_preparation':
                                active, local_tips, G = _prepared_modal_fields(continuum_system, cold, archive,
                                    material.original_parent_id, tag, algebra.constant(beta_bounds),
                                    algebra.constant(zb), algebra, prepared=True)
                                tips = tuple(tipmap[active[i]] for i in local_tips)
                                H = tuple(_sym(G[i]) for i in local_tips)
                                strains = tuple(_matrix_range(h) for h in H)
                            radial = algebra.constant(material.radius_m-3*zb)
                            if not donor:
                                radial = radial+3*material.areal_material_thickness_m
                            local_r1, local_r2, s2, s3, s4, L, Q, angular_invariant = _relative_twist_kernel(
                                n, radial, omega, H, lam, mu, algebra)
                        density = [I(0)]*19
                        for i, a, b in zip(tips, local_r1, local_r2):
                            density[i] = a; density[8+i] = b
                        density[16:] = (s2, s3, s4)
                        volume = _cell_volume(material.radius_m, ta, tb, ba, bb, zb, precision, pole)
                        integral = tuple(_rounded_interval(v*volume, precision) for v in density)
                        receipts.append(NativeColdPreparationCell(material.original_parent_id,
                            material.component_id, material.original_material_id, material.owner, layer, kind,
                            tag, ta, tb, ba, bb, beta_bounds, zb, pole, tag == 'end_seam', tips,
                            strains, L, Q, angular_invariant, hcaps, dcap, volume,
                            tuple(density), integral))
    if len(receipts) != needed:
        raise ValueError('The complete original parent/layer/seam/support/collar traversal changed')
    total = tuple(_rounded_interval(_sum(c.integral_bounds[i] for c in receipts), precision)
                  for i in range(19))
    volume = interval_pi(precision+16)/2*sum((b.radius_m**2*b.areal_material_thickness_m
                                            for b in continuum_system.banks), F(0))
    covered_volume = _rounded_interval(_sum(c.source_volume_bounds_m3 for c in receipts), precision)
    if not (covered_volume.lo <= volume.lo and covered_volume.hi >= volume.hi):
        raise ValueError('Complete cell source measures do not enclose the original full eight-parent volume')
    complement2 = _rounded_interval(_sum(c.integral_bounds[16] for c in receipts
        if c.kind == 'zero_modal_complement_preparation'), precision)
    complement4 = _rounded_interval(_sum(c.integral_bounds[18] for c in receipts
        if c.kind == 'zero_modal_complement_preparation'), precision)
    value = NativeColdPreparationIntegral(source.binding_sha256, source.source_raw_sha256,
        source.source_packet_sha256, cold.binding_sha256, continuum_system.binding_sha256,
        archive.model_sha256, tuple(p.binding_sha256 for p in preparations),
        preparations[0].reference_snapshot_sha256,
        tuple(b.original_parent_id for b in continuum_system.banks),
        tuple(b.original_material_id for b in continuum_system.banks),
        tuple(b.component_id for b in continuum_system.banks),
        cold.whole_domain_certificate.complete_archive_domain_ids,
        tuple((b.original_parent_id, b.elastic_layer_areal_depth_m, b.areal_material_thickness_m)
              for b in continuum_system.banks), preparations[0].source_clocks_s, (F(0),)*56,
        MEASURE, volume, covered_volume, total[:8], total[8:16], total[16], total[17], total[18],
        complement2, complement4,
        tuple(receipts), 2*gamma, cold.whole_domain_certificate.donor.archive_carrier_gradient_deviation_bound,
        precision, theta_cell_count, beta_cell_count, depth_cell_count, max_cells, provenance, '')
    value = replace(value, binding_sha256=_seal(value))
    _REGISTRY[value.binding_sha256] = value
    _CONTEXTS[value.binding_sha256] = (cold_system, continuum_system, preparations)
    _CACHE[key] = value
    return value


def validate_native_common_cold_preparation_integral(source, value):
    _source(source)
    if (type(value) is not NativeColdPreparationIntegral
            or _REGISTRY.get(value.binding_sha256) is not value
            or value.source_binding_sha256 != source.binding_sha256
            or _seal(value) != value.binding_sha256):
        raise ValueError('An altered, declared or unauthenticated preparation moment lacks whole-source integration authority')
    _bindings(source, *_CONTEXTS[value.binding_sha256])
    return value


def native_common_cold_preparation_integral_to_dict(source, value):
    return dict(format=FORMAT, **_payload(validate_native_common_cold_preparation_integral(source, value)))


def native_common_cold_preparation_integral_from_dict(raw, *, source, cold_system,
        continuum_system, preparations, precision=96, theta_cell_count=4,
        beta_cell_count=1, depth_cell_count=1, max_cells=4096,
        provenance='Whole original-source physical919 depth-twist moments on frozen zero895 prepared map'):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get('format') != FORMAT:
        raise ValueError('A complete immutable whole preparation integral DTO is required')
    value = integrate_native_common_cold_preparation_moments(source, cold_system, continuum_system,
        preparations, precision=precision, theta_cell_count=theta_cell_count,
        beta_cell_count=beta_cell_count, depth_cell_count=depth_cell_count,
        max_cells=max_cells, provenance=provenance)
    if _json(raw) != _json(native_common_cold_preparation_integral_to_dict(source, value)):
        raise ValueError('Whole preparation restart changed a source, material, cell, measure or moment')
    return value

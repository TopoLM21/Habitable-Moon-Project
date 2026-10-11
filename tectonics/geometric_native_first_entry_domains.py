"""Whole selected-owner first-entry domains for the original two hemispheres.

The full analytic relative rotation, not its endpoint projection, defines the
selected donor loss.  Three original octants have an endpoint or empty domain;
the fourth has a proved curved envelope and a certified whole excess measure.
No curved domain is promoted to a great-circle polygon.  Measures and immutable
first-entry histories are evidence for a separately selected material closure,
not physical direction, force, or constitutive acceptance.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from math import comb, factorial
import hashlib
import json

from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_buried_support import inverse_native_history
from .geometric_native_domain_journal import (
    NativeDomainJournal, native_domain_journal_from_dict,
    native_domain_journal_to_dict, validate_native_domain_journal,
)
from .geometric_native_donor_sweep import _normal_coefficients
from .geometric_native_point_chronology import (
    _Budget, _characteristic, _normalize_positive, _plane_chronology,
    _point, _right_germ,
)
from .geometric_native_polygons import (
    NativePlane, NativeRay, _dot, _side, native_clip,
    native_plane_to_dict, native_polygon_to_dict,
)
from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, certify_rotation_history,
    interval_sqrt,
)
from .geometric_native_swept_domain import certify_native_swept_domain

FORMAT = "native-whole-selected-owner-first-entry-domains-1"
MODEL = "complete_axis_hemisphere_endpoint_and_curved_owner_envelope"
POINT_FORMAT = "native-selected-owner-first-entry-point-1"
QUERY_FORMAT = "native-selected-owner-first-entry-point-query-1"


class UnresolvedNativeFirstEntryDomain(ValueError):
    """The whole selected owner domain was not proved in the declared epoch."""


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _r(value):
    return [value.numerator, value.denominator]


def _i(value):
    return None if value is None else dict(lo=_r(value.lo), hi=_r(value.hi))


def _read_r(raw):
    if (type(raw) is not list or len(raw) != 2 or any(type(x) is not int for x in raw)
            or raw[1] <= 0):
        raise ValueError("First-entry rational requires a typed canonical integer pair")
    result = Fraction(*raw)
    if _r(result) != raw:
        raise ValueError("First-entry rational is not canonical")
    return result


def _round(value, precision):
    return _rounded_interval(value, precision+16)


def _poly_derivative(values, degree):
    return tuple(value*Fraction(factorial(k), factorial(k-degree))
                 for k, value in enumerate(values) if k >= degree) or (Fraction(0),)


def _poly_bounds(values, domain, precision):
    value = I(0)
    for coefficient in reversed(values):
        value = _round(value*domain+coefficient, precision)
    return value


@dataclass(frozen=True, slots=True)
class _AnalyticPolynomial:
    coefficients: tuple
    tails: tuple
    precision: int
    order: int

    def __post_init__(self):
        # Disposable rational Taylor centres are outward authenticated against
        # every exact source coefficient.  Relative rounding keeps saved exact
        # integer pairs bounded without a process-wide decimal digit override.
        # Its full derivative error is added to the analytic remainder.
        centres, errors = [], []
        for value in self.coefficients:
            centre = _rounded_interval(I(value), self.precision+24).midpoint
            centres.append(centre)
            errors.append(abs(centre-value))
        tails = []
        for degree, tail in enumerate(self.tails):
            error = sum((value*Fraction(factorial(k), factorial(k-degree))
                         for k, value in enumerate(errors) if k >= degree), Fraction(0))
            tails.append(_rounded_interval(I(tail+error), self.precision+24).hi)
        object.__setattr__(self, "coefficients", tuple(centres))
        object.__setattr__(self, "tails", tuple(tails))

    def bounds(self, domain, degree=0):
        if type(domain) is not I or not 0 <= domain.lo <= domain.hi <= 1:
            raise ValueError("First-entry analytic query is outside the complete time interval")
        if type(degree) is not int or not 0 <= degree < len(self.tails):
            raise ValueError("First-entry analytic derivative exceeds its explicit budget")
        return _round(_poly_bounds(_poly_derivative(self.coefficients, degree),
                      domain, self.precision)+I(-self.tails[degree], self.tails[degree]), self.precision)

    def derivative(self):
        return _AnalyticPolynomial(_poly_derivative(self.coefficients, 1),
            self.tails[1:], self.precision, self.order)

    def __neg__(self):
        return _AnalyticPolynomial(tuple(-x for x in self.coefficients), self.tails,
                                  self.precision, self.order)

    def __sub__(self, other):
        n = max(len(self.coefficients), len(other.coefficients))
        values = tuple((self.coefficients[k] if k < len(self.coefficients) else 0)
                       -(other.coefficients[k] if k < len(other.coefficients) else 0)
                       for k in range(n))
        tails = tuple(x+y for x, y in zip(self.tails, other.tails))
        return _AnalyticPolynomial(values, tails, self.precision, self.order)

    def __mul__(self, other):
        values = [Fraction(0)]*(len(self.coefficients)+len(other.coefficients)-1)
        for i, x in enumerate(self.coefficients):
            for j, y in enumerate(other.coefficients):
                values[i+j] += x*y
        keep = tuple(values[:self.order+1])
        tails = []
        for degree in range(min(len(self.tails), len(other.tails))):
            tail = sum((abs(value)*Fraction(factorial(k), factorial(k-degree))
                        for k, value in enumerate(values) if k > self.order and k >= degree), Fraction(0))
            for i in range(degree+1):
                j = degree-i
                ap = sum(map(abs, _poly_derivative(self.coefficients, i)), Fraction(0))
                bp = sum(map(abs, _poly_derivative(other.coefficients, j)), Fraction(0))
                tail += comb(degree, i)*(self.tails[i]*bp+other.tails[j]*ap
                                         +self.tails[i]*other.tails[j])
            tails.append(tail)
        return _AnalyticPolynomial(keep, tuple(tails), self.precision, self.order)


def _normal_series(donor_omega, receiver_omega, dt_myr, order, precision):
    duration = Fraction.from_float(dt_myr)
    donor = tuple(Fraction.from_float(x)*duration for x in donor_omega)
    receiver = tuple(Fraction.from_float(x)*duration for x in receiver_omega)
    rate = sum(map(abs, donor), Fraction(0))+sum(map(abs, receiver), Fraction(0))
    coefficients = _normal_coefficients((-1, 0, 0), donor, receiver, order)
    tails = tuple(rate**(order+1)/factorial(order+1-degree)
                  /(1-rate/Fraction(order+2-degree)) for degree in range(7))
    if rate >= Fraction(1, 4):
        raise UnresolvedNativeFirstEntryDomain("Complete first-entry series needs a bounded small relative motion")
    c = _AnalyticPolynomial(tuple(-row[0] for row in coefficients), tails, precision, order)
    y = _AnalyticPolynomial(tuple(row[1] for row in coefficients), tails, precision, order)
    z = _AnalyticPolynomial(tuple(row[2] for row in coefficients), tails, precision, order)
    return c, y, z, rate


def _composites(c, y, z):
    ap = y.derivative()*c-y*c.derivative()
    bp = z.derivative()*c-z*c.derivative()
    curvature = bp.derivative()*ap-bp*ap.derivative()
    height = z*y.derivative()-y*z.derivative()
    return ap, bp, curvature, height


def _integral_bounds(series, domain):
    central = sum((value*(domain.hi**(k+1)-domain.lo**(k+1))/Fraction(k+1)
                   for k, value in enumerate(series.coefficients)), Fraction(0))
    tail = series.tails[0]*domain.width
    return _round(I(central-tail, central+tail), series.precision)


@dataclass(frozen=True, slots=True)
class NativeFirstEntryIntegrationCell:
    time_fraction: I
    integrand_bounds: I
    integral_bounds: I


@dataclass(frozen=True, slots=True)
class NativeFirstEntryDonorDomain:
    fragment_id: str
    source_geometry: object
    parent_area_bounds_sr: I
    endpoint_domain: object
    endpoint_area_bounds_sr: I
    extra_swept_area_bounds_sr: I
    swept_area_bounds_sr: I
    domain_case: str


@dataclass(frozen=True, slots=True)
class NativeFirstEntryDomainsCertificate:
    journal: NativeDomainJournal
    donor_owner: int
    receiver_owner: int
    precision: int
    order: int
    integration_cells: int
    donor_domains: tuple
    normal_rate_bound: Fraction
    analytic_polynomials: tuple
    proof_bounds: tuple
    maximum_ratio_time_fraction: I
    maximum_ratio_bounds: I
    terminal_ratio_bounds: I
    terminal_slope_bounds: I
    terminal_height_bounds: I
    quadrature: tuple
    uncertain_initial_integral_bounds: I
    curved_planar_integral_bounds: I
    endpoint_planar_triangle_bounds: I
    excess_planar_area_bounds: I
    excess_density_bounds: I
    extra_swept_area_bounds_sr: I
    extra_endpoint_fixed_parent_id: str
    extra_endpoint_parent_support_bounds: tuple
    swept_area_bounds_sr: I


def _axis_sources(journal, donor_owner, receiver_owner):
    if type(donor_owner) is not int or type(receiver_owner) is not int or (donor_owner, receiver_owner) != (1, 0):
        raise UnresolvedNativeFirstEntryDomain("Current whole-domain proof supports explicitly selected actual owner 1 into owner 0")
    if len(journal.source_surface.fragments) != 8 or set(dict(journal.omega_rad_per_myr)) != {0, 1}:
        raise UnresolvedNativeFirstEntryDomain("Whole first-entry proof requires all eight original hemisphere octants")
    sources = {}
    for fragment in journal.source_surface.fragments:
        poly = fragment.geometry
        if (poly.pose or len(poly.supports) != 3
                or any(p.rotations for p in poly.supports)
                or any(ray.kind != "seed" or ray.rotations for ray in poly.vertices)):
            raise UnresolvedNativeFirstEntryDomain("Whole first-entry proof requires an unposed original primitive axis epoch")
        normals = set(p.normal for p in poly.supports)
        signs = tuple(next((s for s in (-1, 1) if tuple(s if j == i else 0 for j in range(3)) in normals), None)
                      for i in range(3))
        if None in signs or signs in sources or fragment.parcel.plate != (1 if signs[0] == 1 else 0):
            raise UnresolvedNativeFirstEntryDomain("Original octants do not form the complete declared two hemisphere owners")
        sources[signs] = fragment
    if len(sources) != 8:
        raise UnresolvedNativeFirstEntryDomain("A whole original octant is absent")
    return sources


def certify_native_first_entry_domains(journal, *, donor_owner=1, receiver_owner=0,
        precision=256, order=24, integration_cells=256):
    journal = validate_native_domain_journal(journal)
    for value, lo, hi, name in ((precision, 96, 1024, "precision"), (order, 16, 48, "Taylor order"),
                                (integration_cells, 8, 8192, "integration cell count")):
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError("First-entry "+name+" must be an explicit bounded integer")
    _axis_sources(journal, donor_owner, receiver_owner)
    return _certify(journal, donor_owner, receiver_owner, precision, order, integration_cells)


_DOMAIN_CACHE = {}


def _certify(journal, donor_owner, receiver_owner, precision, order, integration_cells):
    key = (journal.journal_sha256, donor_owner, receiver_owner, precision, order, integration_cells)
    if key not in _DOMAIN_CACHE:
        if len(_DOMAIN_CACHE) >= 8:
            _DOMAIN_CACHE.clear()
        _DOMAIN_CACHE[key] = _build_certificate(journal, donor_owner, receiver_owner,
                                               precision, order, integration_cells)
    return _DOMAIN_CACHE[key]


def _build_certificate(journal, donor_owner, receiver_owner, precision, order, integration_cells):
    sources = _axis_sources(journal, donor_owner, receiver_owner)
    motions = dict(journal.omega_rad_per_myr)
    ds, rs = RotationIncrement(motions[donor_owner], journal.dt_myr), RotationIncrement(motions[receiver_owner], journal.dt_myr)
    complete = certify_native_swept_domain(sources[(1, 1, -1)].geometry,
        sources[(-1, 1, -1)].geometry, ds.omega_rad_per_myr, rs.omega_rad_per_myr,
        ds.dt_myr, precision=precision, order=order)
    root = complete.ratio_turning_fraction
    c, y, z, rate = _normal_series(ds.omega_rad_per_myr, rs.omega_rad_per_myr, ds.dt_myr, order, precision)
    ap, bp, curvature, height = _composites(c, y, z)
    area_numerator = -height*curvature
    whole = I(0, 1)
    proof = dict(c_positive=c.bounds(whole), a_derivative_numerator_positive=ap.bounds(whole),
        b_derivative_numerator_positive=bp.bounds(whole),
        curvature_derivative_negative=curvature.bounds(whole, 1),
        curvature_at_maximum_ratio_negative=curvature.bounds(I(root.lo)),
        curvature_after_maximum_ratio_negative=curvature.bounds(I(root.lo, 1)))
    proof.update(y_over_u_positive=complete.a_over_u_bounds,
        z_over_u_positive=-complete.b_over_u_bounds,
        maximum_ratio_turning_derivative_positive=complete.turning_derivative_bounds,
        maximum_ratio_initial_turning_negative=complete.initial_turning_bounds,
        maximum_ratio_terminal_turning_positive=complete.terminal_turning_bounds,
        initial_ratio_above_terminal=complete.initial_minus_terminal_q_bounds)
    if (proof["c_positive"].lo <= 0 or proof["a_derivative_numerator_positive"].lo <= 0
            or proof["b_derivative_numerator_positive"].lo <= 0
            or any(proof[key].hi >= 0 for key in ("curvature_derivative_negative",
                  "curvature_at_maximum_ratio_negative", "curvature_after_maximum_ratio_negative"))):
        raise UnresolvedNativeFirstEntryDomain("Complete envelope orientation and curvature were not certified")
    qm = z.bounds(root)/y.bounds(root)
    q1 = z.bounds(I(1))/y.bounds(I(1))
    k1 = bp.bounds(I(1))/ap.bounds(I(1))
    h1 = height.bounds(I(1))/ap.bounds(I(1))
    a1 = y.bounds(I(1))/c.bounds(I(1))
    if qm.lo <= q1.hi or q1.lo <= k1.hi or k1.lo <= 0 or h1.lo <= 0:
        raise UnresolvedNativeFirstEntryDomain("Positive complete curved-envelope branch order was not proved")
    cells = []
    integral = I(0)
    start = root.hi
    for index in range(integration_cells):
        domain = I(start+(1-start)*Fraction(index, integration_cells),
                   start+(1-start)*Fraction(index+1, integration_cells))
        h = height.bounds(domain)/ap.bounds(domain)
        minus_kprime = -curvature.bounds(domain)/(ap.bounds(domain)*ap.bounds(domain))
        # Analytic envelope proof gives nonnegative height at every u>=u_m;
        # outward cancellation in the first cell must not become a negative area.
        h = I(max(Fraction(0), h.lo), max(Fraction(0), h.hi))
        value = _round(h*minus_kprime, precision)
        # Integrate the authenticated Taylor numerator exactly.  Only the
        # strictly positive denominator A1(u)^3 needs interval enclosure;
        # this removes the large cancellation of a coarse rectangle rule.
        numerator_integral = _integral_bounds(area_numerator, domain)
        numerator_integral = I(max(Fraction(0), numerator_integral.lo),
                               max(Fraction(0), numerator_integral.hi))
        denominator = ap.bounds(domain)
        amount = _round(numerator_integral/(denominator*denominator*denominator), precision)
        cells.append(NativeFirstEntryIntegrationCell(domain, value, amount))
        integral = _round(integral+amount, precision)
    initial_domain = I(root.lo, root.hi)
    initial_h = height.bounds(initial_domain)/ap.bounds(initial_domain)
    initial_kprime = -curvature.bounds(initial_domain)/(ap.bounds(initial_domain)*ap.bounds(initial_domain))
    initial = I(0, max(Fraction(0), initial_h.hi)*max(Fraction(0), initial_kprime.hi)*root.width)
    total_integral = _round(integral+initial, precision)
    triangle = _round(a1*(q1-k1)*(q1-k1)/2, precision)
    excess = _round(total_integral-triangle, precision)
    if excess.lo <= 0:
        raise UnresolvedNativeFirstEntryDomain("Full curved excess area needs a sharper integration enclosure")
    density_base = I(1, 1+qm.hi**2+h1.hi**2)
    density = _round(1/(density_base*interval_sqrt(density_base, precision)), precision)
    extra = _round(excess*density, precision)
    endpoint_plane = NativePlane((-1, 0, 0)).rotated((rs,)+inverse_native_history((ds,)))
    donor_domains = []
    total = I(0)
    for signs in sorted(sign for sign in sources if sign[0] == 1):
        parent = sources[signs]
        endpoint = native_clip(parent.geometry, endpoint_plane)
        endpoint_area = I(0) if endpoint is None else endpoint.area_bounds()
        positive_extra = extra if signs == (1, -1, 1) else I(0)
        case = ("empty" if signs == (1, -1, -1) else "endpoint_plus_curved_swept_extra"
                if signs == (1, -1, 1) else "endpoint_equals_whole_swept")
        if (signs == (1, -1, -1) and endpoint is not None) or (signs != (1, -1, -1) and endpoint is None):
            raise UnresolvedNativeFirstEntryDomain("Actual full endpoint hemisphere disagrees with proved octant cases")
        area = _round(endpoint_area+positive_extra, precision)
        donor_domains.append(NativeFirstEntryDonorDomain(parent.fragment_id, parent.geometry,
            parent.geometry.area_bounds(), endpoint, endpoint_area, positive_extra, area, case))
        total = _round(total+area, precision)
    # Every curved-only point is in x∈[0,H(1)], y∈[-qmax,-k(1)], z=1.
    # Certify its complete endpoint image remains inside one fixed old footprint.
    chart = (I(0, h1.hi), I(-qm.hi, -k1.lo), I(1))
    pose = certify_rotation_history((ds,), precision).intervals
    moved = tuple(_round(sum((pose[i][j]*chart[j] for j in range(3)), I(0)), precision) for i in range(3))
    parent = sources[(1, -1, 1)]
    support_bounds = tuple((plane.normal, _dot(tuple(I(value) for value in plane.normal), moved))
                           for plane in parent.geometry.supports)
    if any(value.lo <= 0 for _, value in support_bounds):
        raise UnresolvedNativeFirstEntryDomain("Whole curved-only endpoint image is not inside its proved fixed original footprint")
    analytic = tuple((name, series.coefficients, series.tails) for name, series in (
        ("C", c), ("Y", y), ("Z", z), ("A1", ap), ("B1", bp), ("curvature_numerator", curvature),
        ("height_numerator", height), ("area_integrand_numerator", area_numerator)))
    return NativeFirstEntryDomainsCertificate(journal, donor_owner, receiver_owner,
        precision, order, integration_cells, tuple(donor_domains), rate, analytic,
        tuple(sorted(proof.items())), root, qm, q1, k1, h1, tuple(cells), initial,
        total_integral, triangle, excess, density, extra, parent.fragment_id,
        support_bounds, total)


def _body(certificate):
    return dict(format=FORMAT, model=MODEL, journal=native_domain_journal_to_dict(certificate.journal),
        donor_owner=certificate.donor_owner, receiver_owner=certificate.receiver_owner,
        precision=certificate.precision, order=certificate.order, integration_cells=certificate.integration_cells,
        normal_rate_bound=_r(certificate.normal_rate_bound),
        polynomial_enclosure_contract="relative_rational_centres_at_precision_plus_24_with_full_derivative_rounding_errors_in_outward_tails",
        analytic_polynomials=[dict(name=name, coefficients=[_r(value) for value in coefficients],
                                  derivative_tails=[_r(value) for value in tails])
                              for name, coefficients, tails in certificate.analytic_polynomials],
        proof_bounds={name:_i(value) for name, value in certificate.proof_bounds},
        envelope=dict(maximum_ratio_time_fraction=_i(certificate.maximum_ratio_time_fraction),
            maximum_ratio_bounds=_i(certificate.maximum_ratio_bounds), terminal_ratio_bounds=_i(certificate.terminal_ratio_bounds),
            terminal_slope_bounds=_i(certificate.terminal_slope_bounds), terminal_height_bounds=_i(certificate.terminal_height_bounds),
            implicit_domain="x>=0 and max_{0<u<=1}[-C(u)*x+Y(u)*y+Z(u)*z]>0",
            curved_boundary="(x,y,z)=(b(u)-a(u)*k(u),-k(u),1), u_m<=u<=1; a=Y/C,b=Z/C,k=b'/a'",
            early_branch="u<=u_m positive contacts dominated by u_m",
            negative_z_branch="whole swept equals endpoint", positive_y_positive_z_branch="whole swept equals endpoint",
            negative_y_negative_z_branch="empty", boundary_measure="zero_two_dimensional_native_area_retained_separately",
            represented_as_great_circle_polygon=False),
        quadrature=[dict(time_fraction=_i(cell.time_fraction), integrand_bounds=_i(cell.integrand_bounds),
                         integral_bounds=_i(cell.integral_bounds)) for cell in certificate.quadrature],
        integration=dict(uncertain_initial_integral_bounds=_i(certificate.uncertain_initial_integral_bounds),
            curved_planar_integral_bounds=_i(certificate.curved_planar_integral_bounds),
            endpoint_planar_triangle_bounds=_i(certificate.endpoint_planar_triangle_bounds),
            excess_planar_area_bounds=_i(certificate.excess_planar_area_bounds),
            excess_density_bounds=_i(certificate.excess_density_bounds), extra_swept_area_bounds_sr=_i(certificate.extra_swept_area_bounds_sr),
            formula="integral_{u_m}^1 (b-a*k)*(-k') du - a(1)*(q(1)-k(1))^2/2; spherical_density_bounds_times_planar_excess"),
        donor_domains=[dict(fragment_id=domain.fragment_id, source_geometry=native_polygon_to_dict(domain.source_geometry),
            parent_area_bounds_sr=_i(domain.parent_area_bounds_sr),
            endpoint_domain=None if domain.endpoint_domain is None else native_polygon_to_dict(domain.endpoint_domain),
            endpoint_area_bounds_sr=_i(domain.endpoint_area_bounds_sr), extra_swept_area_bounds_sr=_i(domain.extra_swept_area_bounds_sr),
            swept_area_bounds_sr=_i(domain.swept_area_bounds_sr), domain_case=domain.domain_case)
            for domain in certificate.donor_domains],
        extra_endpoint_fixed_parent_id=certificate.extra_endpoint_fixed_parent_id,
        extra_endpoint_parent_support_bounds=[dict(normal=list(normal), bounds=_i(value))
            for normal, value in certificate.extra_endpoint_parent_support_bounds],
        swept_area_bounds_sr=_i(certificate.swept_area_bounds_sr),
        scope="complete_original_selected_owner_1_into_owner_0_first_entry_domain_and_measures",
        physical_direction_authenticated=False, physical_archive_adopted=False,
        native_force_step_accepted=False, global_inventory_updated=False)


def validate_native_first_entry_domains(certificate):
    if type(certificate) is not NativeFirstEntryDomainsCertificate:
        raise ValueError("Expected the immutable whole first-entry domain certificate")
    replay = certify_native_first_entry_domains(certificate.journal, donor_owner=certificate.donor_owner,
        receiver_owner=certificate.receiver_owner, precision=certificate.precision, order=certificate.order,
        integration_cells=certificate.integration_cells)
    if _encoded(_body(replay)) != _encoded(_body(certificate)):
        raise ValueError("Whole first-entry domain differs from full typed analytic and geometric replay")
    return replay


def native_first_entry_domains_to_dict(certificate):
    body = _body(validate_native_first_entry_domains(certificate))
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def native_first_entry_domains_from_dict(raw, *, journal=None):
    if type(raw) is not dict or raw.get("format") != FORMAT or raw.get("model") != MODEL:
        raise ValueError("Invalid whole first-entry domain schema")
    try:
        restored_journal = native_domain_journal_from_dict(raw["journal"])
        if journal is not None and _encoded(native_domain_journal_to_dict(validate_native_domain_journal(journal))) != _encoded(native_domain_journal_to_dict(restored_journal)):
            raise ValueError("First-entry saved domain belongs to another authoritative source journal")
        replay = certify_native_first_entry_domains(restored_journal,
            donor_owner=raw["donor_owner"], receiver_owner=raw["receiver_owner"], precision=raw["precision"],
            order=raw["order"], integration_cells=raw["integration_cells"])
        if _encoded(raw) != _encoded(native_first_entry_domains_to_dict(replay)):
            raise ValueError("Saved whole first-entry domain differs from sealed full replay")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid whole first-entry domain fields") from error
    return replay


@dataclass(frozen=True, slots=True)
class NativeFirstEntryPointCertificate:
    domains: NativeFirstEntryDomainsCertificate
    donor_fragment_id: str
    original_point: tuple
    max_depth: int
    max_cells: int
    plane_chronology: object
    generator_characteristic_polynomial: tuple
    processed_cells: int
    first_entry_time_fraction: I | None
    swept_membership: str


@dataclass(frozen=True, slots=True)
class NativeFirstEntryPointQuery:
    certificate: NativeFirstEntryPointCertificate
    time_fraction: Fraction
    material_allocation: str
    first_entry_time_fraction: I | None
    contact_occupancy: str
    archive_debits: int


def certify_native_first_entry_point(certificate, donor_fragment_id, original_point, *,max_depth=192,max_cells=4096):
    domains = validate_native_first_entry_domains(certificate)
    if type(donor_fragment_id) is not str:
        raise ValueError("First-entry point must retain its typed original fragment identity")
    parent = next((domain for domain in domains.donor_domains if domain.fragment_id == donor_fragment_id), None)
    if parent is None:
        raise ValueError("First-entry point is not from a selected original donor")
    point = _point(original_point)
    if any(_side(NativeRay("seed", point), support) < 0 for support in parent.source_geometry.supports):
        raise ValueError("First-entry point is outside its original donor")
    if type(max_depth) is not int or not 1 <= max_depth <= 768 or type(max_cells) is not int or not 1 <= max_cells <= 65536:
        raise ValueError("First-entry point budgets must be explicit bounded integers")
    return _certify_point(domains, donor_fragment_id, point, max_depth, max_cells)


_POINT_CACHE = {}


def _certify_point(domains, donor_fragment_id, point, max_depth, max_cells):
    key = (domains.journal.journal_sha256, domains.donor_owner, domains.receiver_owner,
           domains.precision, domains.order, domains.integration_cells,
           donor_fragment_id, point, max_depth, max_cells)
    if key not in _POINT_CACHE:
        if len(_POINT_CACHE) >= 128:
            _POINT_CACHE.clear()
        _POINT_CACHE[key] = _build_point(domains, donor_fragment_id, point, max_depth, max_cells)
    return _POINT_CACHE[key]


def _build_point(domains, donor_fragment_id, point, max_depth, max_cells):
    dt = Fraction.from_float(domains.journal.dt_myr)
    motions = dict(domains.journal.omega_rad_per_myr)
    donor = tuple(Fraction.from_float(value)*dt for value in motions[domains.donor_owner])
    receiver = tuple(Fraction.from_float(value)*dt for value in motions[domains.receiver_owner])
    budget = _Budget(max_cells, max_depth)
    plane, _ = _plane_chronology(0, NativePlane((-1, 0, 0)), donor, receiver,
        _normalize_positive(point), domains.order, domains.precision, dt, budget)
    plane = replace(plane, events=tuple(replace(event, sequence=index)
        for index, event in enumerate(plane.events, 1)))
    entry = next((event for event in plane.events if event.kind == "entry"), None)
    # A positive no-root cell or any entry proves true open-hemisphere contact;
    # a coincident ray is a boundary, never positive two-dimensional material.
    positive = any(cell.sign == 1 for cell in plane.cells) or entry is not None
    membership = "boundary" if plane.coincident else "inside" if positive else "outside"
    return NativeFirstEntryPointCertificate(domains, donor_fragment_id, point, max_depth, max_cells,
        plane, _characteristic(donor, receiver), budget.count,
        None if entry is None else entry.time_fraction, membership)


def _point_body(c):
    plane = c.plane_chronology
    return dict(format=POINT_FORMAT, domains=native_first_entry_domains_to_dict(c.domains),
        donor_fragment_id=c.donor_fragment_id, original_point=[_r(value) for value in c.original_point],
        max_depth=c.max_depth, max_cells=c.max_cells,
        generator_characteristic_polynomial=[_r(value) for value in c.generator_characteristic_polynomial],
        processed_cells=c.processed_cells, first_entry_time_fraction=_i(c.first_entry_time_fraction),
        swept_membership=c.swept_membership,
        owner_plane=dict(support_index=plane.support_index, support=native_plane_to_dict(plane.support),
            initial_zero_order=plane.initial_zero_order, coincident=plane.coincident,
            cells=[dict(time_fraction=_i(cell.time_fraction), status=cell.status, sign=cell.sign,
                gap_bounds=_i(cell.gap_bounds), derivative_bounds=_i(cell.derivative_bounds),
                left_gap_bounds=_i(cell.left_gap_bounds), right_gap_bounds=_i(cell.right_gap_bounds),
                root_fraction=_i(cell.root_fraction)) for cell in plane.cells],
            events=[dict(sequence=event.sequence, kind=event.kind, support_index=event.support_index,
                         time_fraction=_i(event.time_fraction), time_myr=_i(event.time_myr)) for event in plane.events]),
        receiver_internal_interfaces_do_not_create_new_owner_debits=True,
        boundary_convention="positive_owner_hemisphere_contact_with_original_one_dimensional_interface_of_zero_two_dimensional_measure_retained",
        finite_material_measure_assigned_to_point=False, physical_adoption=False)


def validate_native_first_entry_point(certificate):
    if type(certificate) is not NativeFirstEntryPointCertificate:
        raise ValueError("Expected immutable owner first-entry point context")
    replay = certify_native_first_entry_point(certificate.domains, certificate.donor_fragment_id,
        certificate.original_point, max_depth=certificate.max_depth, max_cells=certificate.max_cells)
    if _encoded(_point_body(replay)) != _encoded(_point_body(certificate)):
        raise ValueError("First-entry point differs from complete typed owner-plane replay")
    return replay


def native_first_entry_point_to_dict(certificate):
    body = _point_body(validate_native_first_entry_point(certificate))
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def native_first_entry_point_from_dict(raw):
    if type(raw) is not dict or raw.get("format") != POINT_FORMAT:
        raise ValueError("Invalid selected owner first-entry point schema")
    try:
        replay = certify_native_first_entry_point(native_first_entry_domains_from_dict(raw["domains"]),
            raw["donor_fragment_id"], tuple(_read_r(value) for value in raw["original_point"]),
            max_depth=raw["max_depth"], max_cells=raw["max_cells"])
        if _encoded(raw) != _encoded(native_first_entry_point_to_dict(replay)):
            raise ValueError("Saved first-entry point differs from full sealed replay")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid selected owner point fields") from error
    return replay


def evaluate_native_first_entry_point(certificate, time_fraction):
    c = validate_native_first_entry_point(certificate)
    if type(time_fraction) not in (int, Fraction) or not 0 <= time_fraction <= 1:
        raise ValueError("First-entry query requires an exact fraction in the declared interval")
    time = Fraction(time_fraction)
    first = c.first_entry_time_fraction
    plane = c.plane_chronology
    if plane.coincident:
        occupancy = "boundary"
    elif any(event.time_fraction.lo <= time <= event.time_fraction.hi for event in plane.events):
        occupancy = "event_window"
    else:
        sign = _right_germ(plane)
        for event in plane.events:
            if event.time_fraction.hi < time:
                sign = 1 if event.kind == "entry" else -1
        occupancy = "inside" if sign > 0 else "outside"
    if c.swept_membership == "boundary":
        allocation, debits = "boundary", 0
    elif first is None or time < first.lo:
        allocation, debits = "live", 0
    elif time <= first.hi:
        allocation, debits = "event_window", 0
    else:
        allocation, debits = "archived", 1
    return NativeFirstEntryPointQuery(c, time, allocation, first, occupancy, debits)


def _query_body(query):
    return dict(format=QUERY_FORMAT, point_certificate_sha256=native_first_entry_point_to_dict(query.certificate)["payload_sha256"],
        time_fraction=_r(query.time_fraction), material_allocation=query.material_allocation,
        first_entry_time_fraction=_i(query.first_entry_time_fraction), contact_occupancy=query.contact_occupancy,
        archive_debits=query.archive_debits, physical_adoption=False)


def native_first_entry_point_query_to_dict(query):
    if type(query) is not NativeFirstEntryPointQuery:
        raise ValueError("Expected immutable first-entry query")
    replay = evaluate_native_first_entry_point(query.certificate, query.time_fraction)
    if _encoded(_query_body(replay)) != _encoded(_query_body(query)):
        raise ValueError("First-entry query differs from its authoritative point context")
    body = _query_body(replay)
    return dict(body, payload_sha256=hashlib.sha256(_encoded(body)).hexdigest())


def native_first_entry_point_query_from_dict(raw, *, certificate):
    if type(raw) is not dict or raw.get("format") != QUERY_FORMAT:
        raise ValueError("Invalid first-entry query schema")
    try:
        query = evaluate_native_first_entry_point(certificate, _read_r(raw["time_fraction"]))
        if _encoded(raw) != _encoded(native_first_entry_point_query_to_dict(query)):
            raise ValueError("Saved first-entry query differs from its authoritative context")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid first-entry query fields") from error
    return query

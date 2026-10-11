"""An additive NEW common reference for all thirteen genuine source masks.

Historical DTOs, owner clocks, five saved128 groups and their literal rates
remain unchanged.  A separate runtime embedding applies one physical-radius
flow to all source bodies.  Its inverse and determinant are exact identities;
actual interval point/face/clamp bounds are evaluated here.  The archived
initial partition theorem is source-bound and its complete spherical support
arrangement is replayed, rather than promoting a copied admission flag.

Mapped faces and transported MATERIAL clamps are geometric reference data.
Neither historical world-fixed clamps, a contact rest metric/gap, elapsed-time
movement, force equilibrium nor whole-clock material admission is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
from math import factorial
from pathlib import Path
import hashlib
import json

from .geometric_intrinsic_geometry import rotation_increment_from_dict
from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, certify_rotation_history,
    interval_pi, interval_sqrt,
)
from .geometric_native_spherical_solid import _arc_frame, interval_cbrt
from .geometric_native_joint_spherical_field import (
    JointSphericalLayerMap, bound_joint_layer_point, rotate_joint_physical_depth,
)
from .geometric_native_normal_support import (
    NativeNormalSupportSource, _encoded, _payload, _read_fraction, _read_interval,
    load_native_normal_support_source, validate_native_normal_support_source,
)


FORMAT = "native-common-reference-1"
ARCHIVED_INPUTS = (
    ("common_reference_candidate.json", "50af6b71758feaaed531651b231b50fe5e45228cfa4efa808b26c5a2070d86f4", "candidate_packet_sha256"),
    ("common_reference_source_partition.json", "038b774e15eba4d8dcdb26ef0d65778218cdea0090a71bc956f331637330a840", "proof_packet_sha256"),
    ("common_reference_partition_independent_audit.json", "3e32618b38c2a06f6ce8c6db74b66551d518406531e7a6cb004bc1f61e4b28f7", "audit_packet_sha256"),
    ("common_reference_curved_partition_peer.json", "d07f10dbf7c50a1d454db18d194ff4999889aadd56f27b7a178b30e2d679a259", None),
)
REFERENCE_PROVENANCE = (
    "NEW common physical-radius reference embedding of all genuine source masks; "
    "historical saved128 groups and physical clocks remain literal historical inputs"
)
_REFERENCES = {}
_BANKS = {}
_SOURCES = {}
_FACE_AUTHENTICATED = {}
_COMMON_BORN_FACE_CACHE = {}


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _map(M, x):
    return tuple(_dot(row, x) for row in M)


def _square(x):
    return I(0 if x.lo <= 0 <= x.hi else min(x.lo*x.lo, x.hi*x.hi),
             max(x.lo*x.lo, x.hi*x.hi))


def _rotate_common(vector, literal_rate, tau, precision, order):
    """Enclose the literal exp([A]tau), compressing disposable coefficients.

    The huge exact source rate remains the authoritative constant.  Outward
    interval scratch coefficients avoid million-bit denominators in its64th
    power.  No midpoint, saved alias, material geometry or basis is replaced.
    The induced l1 norm gives one uniform absolute exponential-series tail.
    """
    def rounded(x):
        return _rounded_interval(x if type(x) is I else I(x), precision+16)
    if not any(literal_rate) or tau == I(0):
        return vector
    A = tuple(_rounded_interval(I(x), precision+24) for x in literal_rate)
    rho = sum((max(abs(x.lo), abs(x.hi)) for x in A), F(0))
    time = max(abs(tau.lo), abs(tau.hi))
    argument = rounded(I(rho*time)).hi
    ratio = argument/F(order+2)
    if ratio >= 1:
        raise ValueError("The common interval exponential exceeds its uniform tail budget")
    scale = sum((max(abs(x.lo), abs(x.hi)) for x in vector), F(0))
    tail = rounded(I(scale*argument**(order+1)/factorial(order+1)/(1-ratio))).hi
    coefficients = [vector]
    for degree in range(1, order+1):
        coefficients.append(tuple(rounded(x/degree) for x in _cross(A, coefficients[-1])))
    result = []
    for axis in range(3):
        value = I(0)
        for row in reversed(coefficients):
            value = rounded(value*tau+row[axis])
        result.append(rounded(value+I(-tail, tail)))
    return tuple(result)


def _budget(precision, order=28):
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Common reference precision must be an integer from96 to512")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Common reference rotation order must be an integer from8 to64")


def _vector(x, *, unit=False):
    if type(x) is not tuple or len(x) != 3 or any(type(v) is not I for v in x):
        raise ValueError("Common geometry requires an immutable three-interval vector")
    norm = sum((_square(v) for v in x), I(0))
    if unit and not norm.lo <= 1 <= norm.hi:
        raise ValueError("A material direction enclosure must contain a unit vector")
    return x


def _typed_record(value):
    return tuple(getattr(value, f.name) for f in fields(value))


def _bank_seal(bank):
    return hashlib.sha256(_encoded({f.name: _payload(getattr(bank, f.name))
        for f in fields(bank) if f.name != "bank_snapshot_sha256"})).hexdigest()


def _reference_seal(reference):
    # Runtime bank IDs are derived from this reference ID; their source fields
    # are included directly, without a circular hash dependency.
    body = {f.name: _payload(getattr(reference, f.name)) for f in fields(reference)
            if f.name not in ("reference_snapshot_sha256", "banks")}
    body["banks"] = [{f.name: _payload(getattr(bank, f.name)) for f in fields(bank)
        if f.name not in ("bank_snapshot_sha256", "reference_snapshot_sha256")}
        for bank in reference.banks]
    return hashlib.sha256(_encoded(body)).hexdigest()


@lru_cache(maxsize=1)
def _archived():
    root = Path(__file__).resolve().parents[1]/"analysis/geometric_native_cold_body_validation"
    result = []
    for name, digest, seal_key in ARCHIVED_INPUTS:
        raw = (root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("The immutable common-reference proof dependency changed: "+name)
        packet = json.loads(raw)
        if seal_key:
            body = dict(packet)
            if body.pop(seal_key) != hashlib.sha256(_encoded(body)).hexdigest():
                raise ValueError("The common-reference dependency canonical seal changed: "+name)
        result.append(packet)
    return tuple(result)


def _history_word(history):
    result = []
    for step in history:
        axis = tuple(F.from_float(x) for x in step["omega_rad_per_myr"])
        duration = F.from_float(step["dt_myr"])
        if not any(axis) or not duration:
            continue
        if result and duration == result[-1][1] and axis == tuple(-x for x in result[-1][0]):
            result.pop()
        else:
            result.append((axis, duration))
    return tuple(result)


def _plane_predicate(poly, histories):
    planes = []
    for p in poly["supports"]:
        n = p["normal"]
        if sum(bool(x) for x in n) != 1 or max(map(abs, n)) != 1:
            raise ValueError("The authenticated initial arrangement lost its literal axis normal")
        axis = next(i for i, x in enumerate(n) if x)
        word = _history_word(p["rotations"]+poly["pose"])
        owner = next((o for o, h in histories.items() if word == _history_word(h)), None)
        if word and owner is None:
            raise ValueError("A source mask contains another ordered plane history")
        planes.append((axis if not word else 3+3*owner+axis, 1 if n[axis] > 0 else -1))
    return tuple(planes)


def _belongs(signs, planes):
    return all(signs[index] == sign for index, sign in planes)


@dataclass(frozen=True, slots=True)
class NativeCommonSourcePartitionReplay:
    strict_triple_count: int
    strict_feasible_cell_count: int
    source_component_count: int
    world_normal_bounds: tuple
    strict_triple_bounds: tuple
    whole_cell_unique_component_ids: tuple
    curved_extra_area_bounds_sr: I
    strict_curved_fixed_parent_support_bounds: tuple
    archived_dependency_raw_sha256: tuple

    @property
    def initial_all13_positive_volume_exclusion_certified(self):
        return (self.strict_triple_count == 84 and self.strict_feasible_cell_count == 74
                and self.source_component_count == 13 and self.curved_extra_area_bounds_sr.lo > 0
                and all(x.lo > 0 for x in self.strict_curved_fixed_parent_support_bounds))


def _replay_partition(source, candidate, proof, audit, curved, precision):
    from .geometric_native_first_entry_domains import _normal_series, _composites
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    epoch = before["snapshot"]["current_material_epoch"]
    material, journal = epoch["source_material_state"], epoch["source_material_state"]["journal"]
    if (candidate["complete_authenticated_source_dto_json"] != [list(x) for x in source.complete_source_dto_json]
            or _encoded(candidate["complete_historical_saved128_reference"]) != source.saved_field_json.encode()
            or proof["source_snapshot_sha256"] != source.source_snapshot_sha256
            or proof["source_epoch_sha256"] != source.source_epoch_sha256
            or audit["proof_raw_sha256"] != ARCHIVED_INPUTS[1][1]
            or audit["candidate_raw_sha256"] != ARCHIVED_INPUTS[0][1]):
        raise ValueError("Common reference dependencies do not bind the complete independently authenticated source")
    owners = dict(journal["omega_rad_per_myr"])
    if dict(epoch["additional_owner_histories"]) != {0: [], 1: []}:
        raise ValueError("The initial source partition cannot admit differential later transport")
    histories = {o: [dict(omega_rad_per_myr=owners[o], dt_myr=journal["dt_myr"])] for o in (0, 1)}
    if before["snapshot"]["owner_histories"] != [[o, histories[o]] for o in (0, 1)]:
        raise ValueError("Current owner histories changed the genuine initial world partition")
    pose = {o: certify_rotation_history(tuple(rotation_increment_from_dict(s) for s in h), precision).intervals
            for o, h in histories.items()}
    normals = tuple(tuple(I(int(i == axis)) for i in range(3)) for axis in range(3))
    normals += tuple(tuple(pose[o][i][axis] for i in range(3)) for o in (0, 1) for axis in range(3))
    triples = []
    for i, j, k in combinations(range(9), 3):
        determinant = _dot(_cross(normals[i], normals[j]), normals[k])
        if determinant.lo <= 0 <= determinant.hi:
            raise ValueError("A whole source arrangement triple sign is unresolved")
        triples.append(determinant)
    components = {c["component_id"]: c for c in epoch["components"]}
    parents = {p["fragment_id"]: p for p in journal["source_surface"]["fragments"]}
    polygons = {cid: tuple(_plane_predicate(g, histories) for g in c["proxy_geometries"])
                for cid, c in components.items()}
    moved = {p["fragment_id"]: _plane_predicate(p["geometry"], histories) for p in journal["moved_fragments"]}
    fixed = {pid: _plane_predicate(p["geometry"], histories) for pid, p in parents.items()}
    raw = {pid: tuple(_plane_predicate(v["geometry"], histories) for v in journal["raw_vacancies"]
                     if v["fixed_original_parent_id"] == pid) for pid in parents}
    seen, members = set(), []
    for row in proof["spherical_arrangement"]["all_74_whole_cell_membership_receipts"]:
        signs = tuple(row["signs"])
        witness = tuple(I(_read_fraction(x)) for x in row["independently_certified_strict_rational_witness"]["exact_rational_direction"])
        if signs in seen or len(signs) != 9 or any(sign not in (-1, 1) for sign in signs):
            raise ValueError("The arrangement profiles are not distinct literal strict signs")
        if any((sign*_dot(n, witness)).lo <= 0 for sign, n in zip(signs, normals)):
            raise ValueError("An archived feasible direction fails actual source normal replay")
        seen.add(signs)
        full = {pid: _belongs(signs, p) for pid, p in moved.items()}
        receiver = any(full[pid] for pid, p in parents.items() if p["parcel"]["plate"] == 0)
        live_count = 0
        found = []
        for cid, c in components.items():
            count = sum(_belongs(signs, p) for p in polygons[cid])
            if count not in (0, 1):
                raise ValueError("A source polygon union double counts a whole arrangement cell")
            expected = (full[c["original_parent_id"]] and (c["owner"] == 0 or not receiver)
                        if c["material_kind"] == "original_live" else
                        _belongs(signs, fixed[c["original_parent_id"]]) and not any(full.values()))
            if bool(count) != expected:
                raise ValueError("An actual source endpoint support differs from its whole set construction")
            if c["material_kind"] == "original_live":
                live_count += count
            if count:
                found.append(cid)
        if live_count > 1 or bool(live_count) != any(full.values()) or len(found) != 1:
            raise ValueError("The true endpoint mask arrangement fails partition replay")
        for pid in parents:
            expected = _belongs(signs, fixed[pid]) and not any(full.values())
            if sum(_belongs(signs, p) for p in raw[pid]) != int(expected):
                raise ValueError("The real raw gap differs from the complete moved-original complement")
        members.append(found[0])
    if len(seen) != 74 or len(components) != 13:
        raise ValueError("The whole nine-circle arrangement was not exhausted")
    # All74 strictly feasible profiles meet the universal nine-circle maximum.
    # Their Boolean conclusions apply to whole open cells, not sample points.
    sweep = material["sweep_certificate"]
    C, Y, Z, _ = _normal_series(owners[1], owners[0], journal["dt_myr"], sweep["order"], sweep["precision"])
    ap, bp, curvature, height = _composites(C, Y, Z)
    named = (("C", C), ("Y", Y), ("Z", Z), ("A1", ap), ("B1", bp),
             ("curvature_numerator", curvature), ("height_numerator", height),
             ("area_integrand_numerator", -height*curvature))
    for row, (name, poly) in zip(sweep["analytic_polynomials"], named):
        if (row["name"] != name or tuple(_read_fraction(x) for x in row["coefficients"]) != poly.coefficients
                or tuple(_read_fraction(x) for x in row["derivative_tails"]) != poly.tails):
            raise ValueError("The true curved-set analytic source polynomial failed replay")
    whole = I(0, 1)
    if (C.bounds(whole).lo <= 0 or ap.bounds(whole).lo <= 0 or bp.bounds(whole).lo <= 0
            or curvature.bounds(whole, 1).hi >= 0):
        raise ValueError("The archived curved-set theorem lost its strict analytic source hypotheses")
    corrections = epoch["paired_corrections"]
    if len(corrections) != 1:
        raise ValueError("The exact curved transfer must be a single source paired subset")
    correction = corrections[0]
    if (correction["negative_component_id"] != curved["negative_component_id"]
            or correction["positive_component_id"] != curved["positive_component_id"]
            or correction["owner"] != 1 or sweep["extra_endpoint_fixed_parent_id"] != "octant:5"):
        raise ValueError("The real source curved subset was transferred to another component")
    envelope = sweep["envelope"]
    h = _read_interval(envelope["terminal_height_bounds"])
    q = _read_interval(envelope["maximum_ratio_bounds"])
    k = _read_interval(envelope["terminal_slope_bounds"])
    chart = (I(0, h.hi), I(-q.hi, -k.lo), I(1))
    moved_chart = _map(pose[1], chart)
    supports = tuple(_dot(tuple(I(x) for x in plane["normal"]), moved_chart)
                     for plane in parents["octant:5"]["geometry"]["supports"])
    if any(value.lo <= 0 for value in supports):
        raise ValueError("The complete curved subset endpoint image is not in its true fixed parent")
    extra = _read_interval(correction["area_bounds_sr"])
    if extra.lo <= 0 or correction["area_bounds_sr"] != curved["true_curved_extra_area_bounds_sr"]:
        raise ValueError("Positive genuine curved material cannot be clipped or redeclared")
    return NativeCommonSourcePartitionReplay(len(triples), len(seen), len(components), normals,
        tuple(triples), tuple(members), extra, supports, tuple((n, d) for n, d, _ in ARCHIVED_INPUTS))


@dataclass(frozen=True, slots=True)
class NativeCommonReferenceBank:
    component_id: str
    material_id: str
    original_parent_id: str
    owner: int
    material_kind: str
    radius_m: F
    roof_radial_displacement_m: F
    areal_material_thickness_m: F
    angular_rate_per_physical_volume_depth_m: tuple
    historical_angular_rate_per_physical_volume_depth_m: tuple
    actual_area_bounds_sr: I
    reference_volume_bounds_m3: I
    complete_mask_source_json: str
    complete_parent_allocation_or_birth_json: str
    paired_corrections_source_json: tuple
    historical_group_json: str
    source_owner_history_json: str
    saved_field_json: str
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_packet_raw_sha256: str
    saved_field_declaration_precision: int
    reference_snapshot_sha256: str
    bank_snapshot_sha256: str

    @property
    def layer(self):
        return JointSphericalLayerMap(self.radius_m, self.roof_radial_displacement_m,
            self.areal_material_thickness_m, self.angular_rate_per_physical_volume_depth_m, REFERENCE_PROVENANCE)

    @property
    def top_physical_volume_depth_m(self):
        return self.layer.top_physical_volume_depth_m

    @property
    def complete_actual_material_mask_retained(self):
        return True

    @property
    def historical_saved_reference_rewritten(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCommonReference:
    banks: tuple
    radius_m: F
    angular_rate_per_physical_volume_depth_m: tuple
    complete_source_dto_json: tuple
    saved_field_json: str
    source_owner_histories_json: str
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_curved_packet_raw_sha256: str
    source_joint_packet_raw_sha256: str
    archived_dependency_raw_sha256: tuple
    source_partition_replay: NativeCommonSourcePartitionReplay
    geometry_enclosure_precision: int
    provenance: str
    reference_snapshot_sha256: str

    @property
    def initial_all13_positive_volume_exclusion_certified(self):
        return self.source_partition_replay.initial_all13_positive_volume_exclusion_certified

    @property
    def elapsed_physical_time_added(self):
        return False

    @property
    def whole_clock_all_pairs_admission_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


@lru_cache(maxsize=8)
def _bind(source, precision, provenance):
    candidate, proof, audit, curved = _archived()
    replay = _replay_partition(source, candidate, proof, audit, curved, max(precision, 160))
    families = tuple(source.families)
    if tuple(f.family_id for f in families) != ("born-shared-0", "born-shared-3"):
        raise ValueError("The common rate requires both authentic born source families")
    A = families[0].shared_depth_axis_per_m
    if not any(A) or any(f.shared_depth_axis_per_m != A for f in families):
        raise ValueError("The common reference needs one literal shared nonzero source rate")
    if tuple(_read_fraction(x) for x in candidate["new_reference"]["literal_common_A_per_m"]) != A:
        raise ValueError("The candidate common rate differs from its authentic source declaration")
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    epoch = before["snapshot"]["current_material_epoch"]
    material = epoch["source_material_state"]
    field = json.loads(source.saved_field_json)
    groups = {cid: group for group in field["groups"] for cid in group["component_ids"]}
    histories = dict(before["snapshot"]["owner_histories"])
    allocations = {p["live_domain_id"]: p for p in material["parent_allocations"]}
    births = {p["newborn"]["vacancy_domain_id"]: p for p in material["endpoint_vacancies"] if p["newborn"] is not None}
    banks = []
    for bank in sorted(source.banks, key=lambda b: b.component_id):
        group = groups[bank.component_id]
        historical = A if group["family_ids"] else (F(0),)*3
        if any(not _read_interval(v).lo <= a <= _read_interval(v).hi for a, v in
               zip(historical, group["angular_rate_per_physical_volume_depth_m"])):
            raise ValueError("A historical saved bank rate was changed or midpoint substituted")
        area = _read_interval(json.loads(bank.complete_mask_source_json)["area_bounds_sr"])
        if area.lo <= 0:
            raise ValueError("The common reference must preserve every positive actual source body")
        record = allocations[bank.component_id] if bank.material_kind == "original_live" else births[bank.component_id]
        corrections = tuple(_encoded(c).decode() for c in epoch["paired_corrections"]
                            if bank.component_id in (c["negative_component_id"], c["positive_component_id"]))
        row = NativeCommonReferenceBank(bank.component_id, bank.material_id, bank.original_parent_id,
            bank.owner, bank.material_kind, source.radius_m, bank.roof_radial_displacement_m,
            bank.areal_material_thickness_m, A, historical, area,
            area*source.radius_m**2*bank.areal_material_thickness_m, bank.complete_mask_source_json,
            _encoded(record).decode(), corrections, _encoded(group).decode(),
            _encoded(histories[bank.owner]).decode(), source.saved_field_json, source.source_snapshot_sha256,
            source.source_support_sha256, source.source_epoch_sha256, source.curved_packet_raw_sha256,
            source.saved_field_declaration_precision, "", "")
        row.layer
        banks.append(row)
    result = NativeCommonReference(tuple(banks), source.radius_m, A, source.complete_source_dto_json,
        source.saved_field_json, _encoded(before["snapshot"]["owner_histories"]).decode(),
        source.source_snapshot_sha256, source.source_support_sha256, source.source_epoch_sha256,
        source.curved_packet_raw_sha256, source.joint_packet_raw_sha256,
        tuple((n, d) for n, d, _ in ARCHIVED_INPUTS), replay, precision, provenance, "")
    ref_id = _reference_seal(result)
    final_banks = []
    for bank in result.banks:
        bank = replace(bank, reference_snapshot_sha256=ref_id)
        bank = replace(bank, bank_snapshot_sha256=_bank_seal(bank))
        _BANKS[bank.bank_snapshot_sha256] = _typed_record(bank)
        final_banks.append(bank)
    result = replace(result, banks=tuple(final_banks), reference_snapshot_sha256=ref_id)
    if _reference_seal(result) != ref_id:
        raise ValueError("Common runtime embedding hash was not stable")
    _REFERENCES[ref_id] = _typed_record(result)
    _SOURCES[ref_id] = source
    return result


def bind_native_common_reference(*, source=None, precision=160, provenance=REFERENCE_PROVENANCE):
    source = load_native_normal_support_source() if source is None else validate_native_normal_support_source(source)
    _budget(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("A new common reference needs an explicit reference interpretation")
    return _bind(source, precision, provenance)


def validate_native_common_reference(reference):
    if (type(reference) is not NativeCommonReference
            or _REFERENCES.get(reference.reference_snapshot_sha256) != _typed_record(reference)
            or _reference_seal(reference) != reference.reference_snapshot_sha256):
        raise ValueError("An altered or re-sealed common reference lacks complete source authentication")
    return reference


def bind_native_common_reference_bank(reference, component_id):
    reference = validate_native_common_reference(reference)
    matches = tuple(b for b in reference.banks if b.component_id == component_id)
    if type(component_id) is not str or len(matches) != 1:
        raise ValueError("The common reference requires one genuine source component identity")
    return validate_native_common_reference_bank(matches[0], reference=reference)


def validate_native_common_reference_bank(bank, *, reference=None):
    if (type(bank) is not NativeCommonReferenceBank
            or _BANKS.get(bank.bank_snapshot_sha256) != _typed_record(bank)
            or _bank_seal(bank) != bank.bank_snapshot_sha256):
        raise ValueError("An altered common bank cannot substitute source masks, rates, roofs or histories")
    if reference is not None:
        reference = validate_native_common_reference(reference)
        if bank.reference_snapshot_sha256 != reference.reference_snapshot_sha256 or bank not in reference.banks:
            raise ValueError("The runtime bank belongs to another authentic common reference")
    return bank


def native_common_reference_to_dict(reference):
    reference = validate_native_common_reference(reference)
    return dict(format=FORMAT, reference=_payload(reference),
        initial_source_positive_volume_exclusion_certified=reference.initial_all13_positive_volume_exclusion_certified,
        source_saved_reference_groups_rewritten=False, elapsed_physical_time_added=False,
        constitutive_rest_contact_reference_recovered=False, physical_material_admission_authorized=False,
        whole_clock_all_pairs_admission_certified=False)


def native_common_reference_from_dict(raw, *, source=None):
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Invalid complete common-reference serialization")
    value = raw.get("reference", {})
    try:
        replay = bind_native_common_reference(source=source, precision=value["geometry_enclosure_precision"],
                                             provenance=value["provenance"])
    except (KeyError, TypeError) as exc:
        raise ValueError("The complete source common reference is absent") from exc
    if _encoded(raw) != _encoded(native_common_reference_to_dict(replay)):
        raise ValueError("A saved common reference differs from full independent source replay")
    return replay


def native_common_reference_bank_to_dict(bank):
    return _payload(validate_native_common_reference_bank(bank))


@dataclass(frozen=True, slots=True)
class NativeCommonReferencePointBounds:
    reference_snapshot_sha256: str
    component_id: str
    input_position_bounds_m: tuple
    world_position_bounds_m: tuple
    reference_gradient_bounds: tuple
    rotation_matrix_bounds: tuple
    radius_bounds_m: I
    physical_volume_depth_bounds_m: I
    material_depth_bounds_m: I | None
    determinant_exact: F
    signed_material_chart_jacobian_exact: F | None
    inverse_map: bool

    @property
    def material_membership_authenticated(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def _point_bounds(ref_id, component_id, R, A, u, cubes, z, precision, order, inverse=False):
    r = interval_cbrt(cubes, precision)
    tau = (I(R**3)-cubes)/(3*R**2)
    rate = tuple(-x for x in A) if inverse else A
    if sum(map(abs, rate), F(0))*max(abs(tau.lo), abs(tau.hi)) >= F(1, 4):
        raise ValueError("This interval rotation query exceeds its explicit small-angle bound")
    rotated = _rotate_common(u, rate, tau, precision, order)
    point = tuple(_rounded_interval(r*x, precision) for x in rotated)
    columns = tuple(_rotate_common(tuple(I(int(i == j)) for i in range(3)),
        rate, tau, precision, order) for j in range(3))
    Q = tuple(tuple(columns[j][i] for j in range(3)) for i in range(3))
    shear = _cross(rate, u)
    rank_one = tuple(tuple(I(int(i == j))-cubes/R**2*shear[i]*u[j]
                          for j in range(3)) for i in range(3))
    gradient = tuple(tuple(_rounded_interval(sum((Q[i][k]*rank_one[k][j]
        for k in range(3)), I(0)), precision) for j in range(3)) for i in range(3))
    return NativeCommonReferencePointBounds(ref_id, component_id, tuple(r*x for x in u), point,
        gradient, Q, r, tau, z, F(1), -R**2 if z is not None else None, inverse)


def bound_native_common_reference_bank_point(bank, direction_bounds, material_depth_bounds_m,
        *, precision=160, order=28):
    """Full physical-depth map on a source ray; arbitrary rays prove no mask membership."""
    bank = validate_native_common_reference_bank(bank)
    _budget(precision, order)
    u = _vector(direction_bounds, unit=True)
    z = material_depth_bounds_m
    if type(z) is not I or not 0 <= z.lo <= z.hi <= bank.areal_material_thickness_m:
        raise ValueError("A common bank query must remain inside its full literal source depth")
    R = bank.radius_m
    cubes = I((R+bank.roof_radial_displacement_m)**3)-3*R**2*z
    return _point_bounds(bank.reference_snapshot_sha256, bank.component_id, R,
        bank.angular_rate_per_physical_volume_depth_m, u, cubes, z, precision, order)


def bound_native_common_reference_point(reference, point_bounds_m, *, precision=160, order=28):
    """Evaluate the NEW global reference map and Cartesian differential.

    The exact map/inverse are defined for every positive radius.  A numerical
    query must additionally fit this explicit bounded Taylor-rotation budget.
    The input box is a Cartesian extension query, not angular-mask authority.
    """
    reference = validate_native_common_reference(reference)
    return _bound_cartesian(reference, point_bounds_m, False, precision, order)


def bound_native_common_reference_inverse(reference, point_bounds_m, *, precision=160, order=28):
    reference = validate_native_common_reference(reference)
    return _bound_cartesian(reference, point_bounds_m, True, precision, order)


def _bound_cartesian(reference, point, inverse, precision, order):
    _budget(precision, order)
    point = _vector(point)
    radius = interval_sqrt(sum((_square(x) for x in point), I(0)), precision)
    if radius.lo <= 0:
        raise ValueError("A Cartesian interval inverse requires a positive-radius box")
    u = tuple(x/radius for x in point)
    cubes = radius*radius*radius
    return _point_bounds(reference.reference_snapshot_sha256, "", reference.radius_m,
        reference.angular_rate_per_physical_volume_depth_m, u, cubes, None, precision, order, inverse)


@dataclass(frozen=True, slots=True)
class NativeCommonReferenceFace:
    reference_snapshot_sha256: str
    family_id: str
    donor_component_id: str
    receiver_component_id: str
    start_unit_bounds: tuple
    tangent_unit_bounds: tuple
    arc_normal_bounds: tuple
    receiving_normal_orientation: int
    native_angle_end_bounds_radians: I
    complete_carrier_source_json: str
    genuine_shared_boundary_arc_json: str
    geometry_enclosure_precision: int
    face_snapshot_sha256: str

    @property
    def true_source_boundary_of_both_actual_masks_certified(self):
        return True

    @property
    def constitutive_rest_contact_reference_recovered(self):
        return False


def _face_seal(face):
    return hashlib.sha256(_encoded({f.name: _payload(getattr(face, f.name))
        for f in fields(face) if f.name != "face_snapshot_sha256"})).hexdigest()


def bind_native_common_reference_face(reference, family_id, *, precision=160):
    """Bind a true born material boundary, replaying both native polygon edges.

    The two supported born faces have no paired curved correction touching
    their banks.  Their native polygon boundaries therefore are boundaries
    of the complete TRUE source masks.  A new curved face needs its own proof.
    """
    from .geometric_native_polygons import (
        NativeArc, _ray_same, _same_plane, native_arc_to_dict,
        native_ray_from_dict, native_plane_from_dict, native_polygon_from_dict,
        native_shared_boundary_arcs, _plane_values,
    )
    reference = validate_native_common_reference(reference)
    _budget(precision)
    if type(family_id) is not str:
        raise ValueError("A common born face requires one actual source family")
    key = (reference.reference_snapshot_sha256, family_id, max(precision, 320))
    if key in _COMMON_BORN_FACE_CACHE:
        return validate_native_common_reference_face(_COMMON_BORN_FACE_CACHE[key], reference=reference)
    source = _SOURCES[reference.reference_snapshot_sha256]
    matches = tuple(f for f in source.families if f.family_id == family_id)
    if type(family_id) is not str or len(matches) != 1:
        raise ValueError("A common born face requires one actual source family")
    family = matches[0]
    donor = bind_native_common_reference_bank(reference, family.donor_component_id)
    receiver = bind_native_common_reference_bank(reference, family.receiver_component_id)
    if donor.paired_corrections_source_json or receiver.paired_corrections_source_json:
        raise ValueError("This exact arc proof cannot substitute a proxy for a true curved mask")
    carrier = json.loads(family.complete_carrier_source_json)
    arc_raw = carrier["exact_initial_shared_arc"]
    declared = NativeArc(native_ray_from_dict(arc_raw["start"]), native_ray_from_dict(arc_raw["end"]),
        native_plane_from_dict(arc_raw["plane"]), arc_raw["length_radians"])
    genuine = []
    for a in json.loads(donor.complete_mask_source_json)["proxy_geometries"]:
        for b in json.loads(receiver.complete_mask_source_json)["proxy_geometries"]:
            for arc in native_shared_boundary_arcs(native_polygon_from_dict(a), native_polygon_from_dict(b)):
                if (_same_plane(arc.plane, declared.plane) and _ray_same(arc.start, declared.start)
                        and _ray_same(arc.end, declared.end)):
                    genuine.append(arc)
    if len(genuine) != 1:
        raise ValueError("The declared source face is not a unique true shared bank boundary")
    # The source stores its full native angle at256 precision independently
    # of its128 angular-rate declaration.  Replay more sharply, then verify
    # inclusion in that complete endpoint interval without widening it.
    frame_precision = max(precision, 320)
    start, tangent, _, angle = _arc_frame(genuine[0], frame_precision)
    normal = _cross(start, tangent)
    inward = _plane_values(native_plane_from_dict(carrier["receiver_inward_plane"]), frame_precision)[1]
    orientation = _dot(normal, inward)
    if orientation.lo <= 0 <= orientation.hi:
        raise ValueError("A mapped source face lacks strict receiving orientation")
    sign = 1 if orientation.lo > 0 else -1
    end = family.native_angle_end_bounds_radians
    if not end.lo <= angle.lo <= angle.hi <= end.hi:
        raise ValueError("The complete native source angle is outside its authenticated enclosure")
    face = NativeCommonReferenceFace(reference.reference_snapshot_sha256, family_id,
        donor.component_id, receiver.component_id, start, tangent, normal, sign, end,
        family.complete_carrier_source_json, _encoded(native_arc_to_dict(genuine[0])).decode(), frame_precision, "")
    face = replace(face, face_snapshot_sha256=_face_seal(face))
    _FACE_AUTHENTICATED[face.face_snapshot_sha256] = _typed_record(face)
    _COMMON_BORN_FACE_CACHE[key] = face
    return face


def validate_native_common_reference_face(face, *, reference=None):
    if (type(face) is not NativeCommonReferenceFace
            or _FACE_AUTHENTICATED.get(face.face_snapshot_sha256) != _typed_record(face)
            or _face_seal(face) != face.face_snapshot_sha256):
        raise ValueError("An altered common source face lacks genuine boundary authentication")
    if reference is not None and validate_native_common_reference(reference).reference_snapshot_sha256 != face.reference_snapshot_sha256:
        raise ValueError("A common face belongs to another complete reference")
    return face


@dataclass(frozen=True, slots=True)
class NativeCommonFaceBounds:
    face: NativeCommonReferenceFace
    angle_bounds_radians: I
    native_angle_end_bounds_radians: I
    common_physical_radius_cubed_bounds_m3: I
    physical_radius_bounds_m: I
    common_physical_volume_depth_bounds_m: I
    donor_material_depth_bounds_m: I
    receiver_material_depth_bounds_m: I
    donor_world_position_bounds_m: tuple
    receiver_world_position_bounds_m: tuple
    angular_tangent_bounds_m: tuple
    material_depth_tangent_bounds: tuple
    receiver_inward_unit_normal_bounds: tuple
    area_density_bounds_m2_per_radian_per_material_m: I
    local_signed_cot_dip_bounds: I
    source_declared_dip_radians: F
    source_declared_birth_gap_m: F

    @property
    def world_position_bounds_m(self):
        return self.donor_world_position_bounds_m

    @property
    def reference_faces_match_at_every_common_physical_radius(self):
        return self.donor_world_position_bounds_m == self.receiver_world_position_bounds_m

    @property
    def whole_common_reference_face_enclosed(self):
        return self.angle_bounds_radians == I(0, self.native_angle_end_bounds_radians.hi)

    @property
    def source_constant_dip_reproduced_on_whole_face(self):
        return False

    @property
    def geometric_gluing_supplies_constitutive_rest_gap(self):
        return False

    @property
    def every_point_of_parameter_enclosure_is_material(self):
        # The true irrational endpoint lies in its retained complete interval.
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def _surface_geometry(R, A, S, T, sign, angle, cubes, precision, order):
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    sine, cosine = _trig_ranges_signed(angle.lo, angle.hi, precision)
    U = tuple(s*cosine+t*sine for s, t in zip(S, T))
    tangent = tuple(-s*sine+t*cosine for s, t in zip(S, T))
    N = _cross(S, T)
    r = interval_cbrt(cubes, precision)
    tau = (I(R**3)-cubes)/(3*R**2)
    P, B = R**2/r, r*r*_dot(A, tangent)
    area = interval_sqrt(_square(P)+_square(B), precision)
    if area.lo <= 0:
        raise ValueError("The whole mapped face has unresolved positive metric")
    def rotate(x):
        return _rotate_common(x, A, tau, precision, order)
    angular = tuple(_rounded_interval(r*x, precision) for x in rotate(tangent))
    depth = rotate(tuple(-R**2/(r*r)*u+r*v for u, v in zip(U, _cross(A, U))))
    nJ = rotate(tuple(sign*(P*n-B*u) for n, u in zip(N, U)))
    normal = tuple(_rounded_interval(x/area, precision) for x in nJ)
    cot = -sign*cubes/R**2*_dot(A, tangent)
    return U, r, tau, angular, depth, normal, area, cot


def bound_native_common_reference_face(reference, family_id, *, angle_bounds_radians=None,
        physical_radius_cubed_bounds_m3=None, precision=160, order=28):
    reference = validate_native_common_reference(reference)
    _budget(precision, order)
    face = (validate_native_common_reference_face(family_id, reference=reference)
            if type(family_id) is NativeCommonReferenceFace else
            bind_native_common_reference_face(reference, family_id, precision=precision))
    donor = bind_native_common_reference_bank(reference, face.donor_component_id)
    receiver = bind_native_common_reference_bank(reference, face.receiver_component_id)
    low = max(donor.layer.layer.bottom_radius_cubed_m3, receiver.layer.layer.bottom_radius_cubed_m3)
    high = min(donor.layer.layer.top_radius_m**3, receiver.layer.layer.top_radius_m**3)
    cubes = I(low, high) if physical_radius_cubed_bounds_m3 is None else physical_radius_cubed_bounds_m3
    if low >= high or type(cubes) is not I or not low <= cubes.lo <= cubes.hi <= high:
        raise ValueError("A born common face requires the genuine positive physical-radius intersection")
    angle = I(0, face.native_angle_end_bounds_radians.hi) if angle_bounds_radians is None else angle_bounds_radians
    if type(angle) is not I or not 0 <= angle.lo <= angle.hi <= face.native_angle_end_bounds_radians.hi:
        raise ValueError("The face angle enclosure lies outside its whole native source arc")
    R, A = reference.radius_m, reference.angular_rate_per_physical_volume_depth_m
    U, r, tau, angular, depth, normal, area, cot = _surface_geometry(R, A,
        face.start_unit_bounds, face.tangent_unit_bounds, face.receiving_normal_orientation,
        angle, cubes, precision, order)
    zd = (I((R+donor.roof_radial_displacement_m)**3)-cubes)/(3*R**2)
    zr = (I((R+receiver.roof_radial_displacement_m)**3)-cubes)/(3*R**2)
    pd = bound_native_common_reference_bank_point(donor, U, zd, precision=precision, order=order)
    pr = bound_native_common_reference_bank_point(receiver, U, zr, precision=precision, order=order)
    if pd.world_position_bounds_m != pr.world_position_bounds_m:
        raise ValueError("Common-radius reference points failed bankwise physical-depth gluing")
    declaration = json.loads(face.complete_carrier_source_json)["declaration"]
    return NativeCommonFaceBounds(face, angle, face.native_angle_end_bounds_radians, cubes, r, tau, zd, zr,
        pd.world_position_bounds_m, pr.world_position_bounds_m, angular, depth, normal, area, cot,
        F.from_float(declaration["dip_radians"]), F.from_float(declaration["birth_gap_free_m"]))


@dataclass(frozen=True, slots=True)
class NativeCommonClampBounds:
    bank: NativeCommonReferenceBank
    trace_id: str
    source_bank: str
    beta_cut_radians: F
    angle_bounds_radians: I
    material_depth_bounds_m: I
    common_physical_volume_depth_bounds_m: I
    mapped_world_position_bounds_m: tuple
    reference_gradient_bounds: tuple
    source_live_support_inequality_upper: F | None
    complete_source_clamp_row_json: str
    complete_original_parent_json: str
    complete_original_allocation_json: str
    mode_displacement_per_m: tuple
    mode_gradient_m_inv: tuple

    @property
    def genuine_live_material_clamp_support_certified(self):
        return self.source_bank == "over" or self.source_live_support_inequality_upper < 0

    @property
    def transported_source_material_clamp(self):
        return True

    @property
    def source_world_clamp_kept_fixed(self):
        return False

    @property
    def all_seven_modal_fields_and_gradients_zero(self):
        return all(not x for v in self.mode_displacement_per_m for x in v) and all(
            not x for M in self.mode_gradient_m_inv for row in M for x in row)

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_common_reference_clamp(reference, component_id, *, angle_bounds_radians=None,
        material_depth_bounds_m=None, precision=160, order=28):
    """Map the actual beta=b material clamp through the SAME common embedding.

    The complete true-live support is reproved using the literal all-time
    relative-rotation majorant.  No archive/newborn supplies this support.
    Chi(b)=chi'(b)=0, and the invertible reference chain rule consequently
    makes all seven displacement fields and spatial gradients exactly zero
    at every mapped source clamp point, across the entire material depth.
    """
    from .geometric_native_cold_body import bind_native_cold_body
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    reference = validate_native_common_reference(reference)
    _budget(precision, order)
    bank = bind_native_common_reference_bank(reference, component_id)
    source = _SOURCES[reference.reference_snapshot_sha256]
    if bank.material_kind != "original_live":
        raise ValueError("HOT material cannot replace a true original cold clamp")
    body = bind_native_cold_body(component_id, source=source, precision=precision)
    angle = I(0, interval_pi(precision).hi/2) if angle_bounds_radians is None else angle_bounds_radians
    z = I(0, bank.areal_material_thickness_m) if material_depth_bounds_m is None else material_depth_bounds_m
    if (type(angle) is not I or not 0 <= angle.lo <= angle.hi <= interval_pi(precision).hi/2
            or type(z) is not I or not 0 <= z.lo <= z.hi <= bank.areal_material_thickness_m):
        raise ValueError("A transported clamp requires its genuine full source quarter/depth")
    sine, cosine = _trig_ranges_signed(angle.lo, angle.hi, precision)
    sb, cb = _trig_ranges_signed(body.beta_cut_radians, body.beta_cut_radians, precision)
    U = tuple(s*cosine+t*sine for s, t in zip(body.current_quarter_start_bounds, body.current_quarter_tangent_bounds))
    p = tuple(u*cb+n*sb for u, n in zip(U, body.current_quarter_inward_bounds))
    upper = None
    if body.source_bank == "sub":
        before = json.loads(dict(source.complete_source_dto_json)["before"])
        journal = before["snapshot"]["current_material_epoch"]["source_material_state"]["journal"]
        rho = F.from_float(journal["dt_myr"])*sum((abs(F.from_float(x))
            for _, omega in journal["omega_rad_per_myr"] for x in omega), F(0))
        if not 0 <= rho < 1:
            raise ValueError("The source all-time clamp majorant is unresolved")
        error = rho/(1-rho)
        # The original donor x=sin(b).  Every quarter point has l1 norm<=3,
        # independently of theta and irrespective of the tiny curved subset.
        upper = -sb.lo+3*error
        if upper >= 0:
            raise ValueError("The actual whole original clamp is not proved outside the TRUE archive")
        clamp = json.loads(body.source_clamp_row_json)
        for vertex in clamp["enclosing_geometry"]["vertices"]:
            if vertex["kind"] != "seed" or vertex["rotations"]:
                raise ValueError("An original source clamp enclosure was geometrically substituted")
            q = tuple(F(x) for x in vertex["reference"])
            if -q[0]+error*sum(map(abs, q), F(0)) >= 0:
                raise ValueError("A source enclosure vertex is not true-live for the complete original interval")
    else:
        allocation = json.loads(body.complete_parent_allocation_json)
        if _read_interval(allocation["archived_area_bounds_sr"]) != I(0):
            raise ValueError("The original receiver clamp was placed into the selected archive")
    mapped = bound_native_common_reference_bank_point(bank, p, z, precision=precision, order=order)
    scaled = body.beta_cut_radians/(body.width_m/body.radius_m)
    chi = (1-scaled)**2*(scaled+2)/2
    chi_prime = -F(3, 2)*(1-scaled)*(1+scaled)/(body.width_m/body.radius_m)
    if chi or chi_prime:
        raise ValueError("The actual source clamp does not zero its displacement AND gradient factors")
    # Psi0=chi*e, Psia=chi*Ea*X/R.  Their spatial derivatives consist only
    # of chi times a smooth reference derivative and chi_prime times grad beta.
    # Both literal factors were evaluated above; detD_Phi=1 ensures the
    # material-to-spatial chain rule exists over the whole positive-radius set.
    modes = tuple((F(0),)*3 for _ in range(7))
    gradients = tuple(tuple((F(0),)*3 for _ in range(3)) for _ in range(7))
    return NativeCommonClampBounds(bank, body.trace_id, body.source_bank, body.beta_cut_radians,
        angle, z, mapped.physical_volume_depth_bounds_m, mapped.world_position_bounds_m,
        mapped.reference_gradient_bounds, upper, body.source_clamp_row_json, body.complete_parent_source_json,
        body.complete_parent_allocation_json, modes, gradients)


@dataclass(frozen=True, slots=True)
class NativeCommonCarriedOriginalFaceBounds:
    reference_snapshot_sha256: str
    trace_id: str
    source_bank: str
    component_id: str
    angle_bounds_radians: I
    material_depth_bounds_m: I
    common_physical_radius_cubed_bounds_m3: I
    physical_volume_depth_bounds_m: I
    world_position_bounds_m: tuple
    angular_tangent_bounds_m: tuple
    material_depth_tangent_bounds: tuple
    bank_inward_unit_normal_bounds: tuple
    area_density_bounds_m2_per_radian_per_material_m: I
    local_signed_cot_dip_bounds: I
    complete_source_trace_json: str
    complete_original_parent_allocation_json: str
    complete_original_mechanical_history_json: str

    @property
    def carried_original_material_sheet_enclosed(self):
        return True

    @property
    def whole_sheet_is_retained_live_contact_interface(self):
        # In particular the donor beta=0 source labels can already be archived.
        return False

    @property
    def donor_and_receiver_historical_sheets_glued(self):
        return False

    @property
    def old_sixteen_constitutive_histories_preserved(self):
        return True

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_common_reference_carried_original_face(reference, trace_id, source_bank, *,
        angle_bounds_radians=None, physical_radius_cubed_bounds_m3=None, precision=160, order=28):
    """Map ONE historical original trace sheet, preserving its real owner pose.

    Distinct R0(trace) and R1(trace) are distinct source sheets.  The common
    invertible map cannot glue them or reclassify archived donor labels as
    retained material.  The current true carried contact cut/rest reference
    requires a separate source declaration; all original histories remain.
    """
    from .geometric_native_cold_body import bind_native_cold_body
    reference = validate_native_common_reference(reference)
    _budget(precision, order)
    if type(trace_id) is not str or source_bank not in ("sub", "over"):
        raise ValueError("A carried original material sheet requires its literal trace and bank")
    source = _SOURCES[reference.reference_snapshot_sha256]
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    material = before["snapshot"]["current_material_epoch"]["source_material_state"]
    binding = tuple(b for b in material["trace_bindings"] if b["trace_id"] == trace_id)
    if len(binding) != 1:
        raise ValueError("A carried source trace lacks its genuine original bank pair")
    parent_id = binding[0]["donor_fragment_id" if source_bank == "sub" else "receiver_fragment_id"]
    bank = next(b for b in reference.banks if b.material_kind == "original_live" and b.original_parent_id == parent_id)
    body = bind_native_cold_body(bank.component_id, source=source, precision=precision)
    R = reference.radius_m
    low, high = bank.layer.layer.bottom_radius_cubed_m3, bank.layer.layer.top_radius_m**3
    cubes = I(low, high) if physical_radius_cubed_bounds_m3 is None else physical_radius_cubed_bounds_m3
    angle = I(0, interval_pi(precision).hi/2) if angle_bounds_radians is None else angle_bounds_radians
    if (type(cubes) is not I or not low <= cubes.lo <= cubes.hi <= high or type(angle) is not I
            or not 0 <= angle.lo <= angle.hi <= interval_pi(precision).hi/2):
        raise ValueError("The carried source sheet lies outside its declared whole physical layer/quarter")
    # Source arc N=S cross T; its inward bank normal is -N for donor,+N for receiver.
    sign = -1 if source_bank == "sub" else 1
    U, r, tau, angular, depth, normal, area, cot = _surface_geometry(R,
        reference.angular_rate_per_physical_volume_depth_m, body.current_quarter_start_bounds,
        body.current_quarter_tangent_bounds, sign, angle, cubes, precision, order)
    z = (I((R+bank.roof_radial_displacement_m)**3)-cubes)/(3*R**2)
    mapped = bound_native_common_reference_bank_point(bank, U, z, precision=precision, order=order)
    return NativeCommonCarriedOriginalFaceBounds(reference.reference_snapshot_sha256, trace_id,
        source_bank, bank.component_id, angle, z, cubes, tau, mapped.world_position_bounds_m,
        angular, depth, normal, area, cot, body.complete_source_trace_json,
        body.complete_parent_allocation_json, _encoded(before["original_physical_history"]).decode())


def native_common_reference_face_to_dict(face):
    return _payload(validate_native_common_reference_face(face))

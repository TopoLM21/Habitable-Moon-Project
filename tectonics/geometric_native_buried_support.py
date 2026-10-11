"""Native accepted-loss ancestry and complete original material support.

Every buried footprint is an actual accepted NativeLoss. Its donor pullback
retains the complete inverse SO(3) history at acceptance; display coordinates
never enter the proof. Chronological native endpoint partitions authenticate
all parents, fresh births, four reservoirs, and once-only footprints. This
receipt certifies original material domains. It does not certify a live clamp,
create mechanical pairs, or authorize a newly created physical fault.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral

from .fractional_surface import EXTENSIVE_FIELDS
from .geometric_intrinsic_geometry import (
    RotationIncrement, rotation_increment_from_dict, rotation_increment_to_dict,
)
from .geometric_native_polygons import (
    NativePolygon, native_intersection, native_polygon_from_dict,
    native_polygon_to_dict, native_subtract, validate_native_polygon,
)
from .geometric_native_surface import (
    NativeFragment, NativeLoss, NativeSurfaceState, NativeTransportResult,
    _parcel_from_dict, native_surface_from_dict, native_surface_to_dict,
)

FORMAT = "geometric-native-original-material-registry-1"
SUPPORT_MODEL = "native_complete_original_2D_material_domain"
_EPS = math.ulp(1.)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def native_surface_digest(state):
    return _hash(native_surface_to_dict(state))


def _fragment_to_dict(fragment):
    if type(fragment) is not NativeFragment:
        raise ValueError("Native accepted ancestry requires explicit native fragments")
    geometry = validate_native_polygon(fragment.geometry)
    parcel = _parcel_from_dict(asdict(fragment.parcel))
    return dict(fragment_id=fragment.fragment_id,
        geometry=native_polygon_to_dict(geometry), parcel=asdict(parcel),
        parent_fragment_id=fragment.parent_fragment_id)


def _fragment_from_dict(raw):
    if (not isinstance(raw, dict) or set(raw) != {
            "fragment_id", "geometry", "parcel", "parent_fragment_id"}):
        raise ValueError("Invalid native accepted fragment schema")
    return NativeFragment(raw["fragment_id"], native_polygon_from_dict(raw["geometry"]),
        _parcel_from_dict(raw["parcel"]), raw["parent_fragment_id"])


def native_loss_to_dict(loss):
    """The complete native material payload, including receiver and decisions."""
    if type(loss) is not NativeLoss:
        raise ValueError("Native registry accepts actual NativeLoss records only")
    if (not isinstance(loss.source_fragment_id, str) or not loss.source_fragment_id
            or not isinstance(loss.receiver_fragment_id, str) or not loss.receiver_fragment_id
            or isinstance(loss.receiver_plate, bool) or not isinstance(loss.receiver_plate, Integral)
            or loss.receiver_plate < 0 or type(loss.time_myr) not in (int, float)
            or not math.isfinite(loss.time_myr)
            or any(not isinstance(value, str) or not value for value in (
                loss.attachment_status, loss.polarity_basis))
            or any(not isinstance(values, tuple) or any(not isinstance(value, str) or not value
                for value in values) or len(set(values)) != len(values)
                for values in (loss.contact_ids, loss.polarity_evidence_ids))):
        raise ValueError("Invalid complete native accepted-loss payload")
    return dict(fragment=_fragment_to_dict(loss.fragment),
        source_fragment_id=loss.source_fragment_id,
        receiver_fragment_id=loss.receiver_fragment_id, receiver_plate=loss.receiver_plate,
        time_myr=loss.time_myr, contact_ids=list(loss.contact_ids),
        attachment_status=loss.attachment_status, polarity_basis=loss.polarity_basis,
        polarity_evidence_ids=list(loss.polarity_evidence_ids))


def native_loss_from_dict(raw):
    if (not isinstance(raw, dict) or set(raw) != {field.name for field in fields(NativeLoss)}
            or any(not isinstance(raw[key], list) for key in (
                "contact_ids", "polarity_evidence_ids"))):
        raise ValueError("Invalid native accepted-loss schema")
    result = NativeLoss(**{**raw, "fragment": _fragment_from_dict(raw["fragment"]),
        "contact_ids": tuple(raw["contact_ids"]),
        "polarity_evidence_ids": tuple(raw["polarity_evidence_ids"])})
    native_loss_to_dict(result)
    return result


def _owners(origin):
    return tuple(sorted({fragment.parcel.plate for fragment in origin.fragments}))


def _histories(origin, value):
    if isinstance(value, Mapping):
        rows = tuple(value.items())
        if any(isinstance(owner, bool) or not isinstance(owner, Integral) or owner < 0 for owner, _ in rows):
            raise ValueError("Native donor map owners must be explicit nonnegative integers")
        rows = tuple(sorted(rows))
    elif isinstance(value, tuple):
        rows = value
    else:
        raise ValueError("Native donor maps require an explicit owner/history mapping")
    if (any(not isinstance(row, tuple) or len(row) != 2 for row in rows)
            or any(isinstance(owner, bool) or not isinstance(owner, Integral) or owner < 0
                for owner, _ in rows)
            or tuple(owner for owner, _ in rows) != _owners(origin)):
        raise ValueError("Native donor maps must name every original owner exactly once")
    result = []
    for owner, history in rows:
        if not isinstance(history, tuple) or any(type(step) is not RotationIncrement for step in history):
            raise ValueError("Native donor maps require full immutable ordered increments")
        checked = tuple(RotationIncrement(step.omega_rad_per_myr, step.dt_myr) for step in history)
        result.append((int(owner), checked))
    return tuple(result)


def inverse_native_history(history):
    """A full inverse recipe; no matrix approximation or component projection."""
    if not isinstance(history, tuple) or any(type(step) is not RotationIncrement for step in history):
        raise ValueError("Native inverse requires full ordered physical increments")
    return tuple(RotationIncrement(tuple(-value for value in step.omega_rad_per_myr),
        step.dt_myr) for step in reversed(history))


def native_pullback(geometry, donor_history):
    if type(geometry) is not NativePolygon:
        raise ValueError("Native pullback requires its authoritative footprint")
    result = replace(geometry, pose=geometry.pose + inverse_native_history(donor_history))
    return validate_native_polygon(result)


@dataclass(frozen=True, slots=True)
class NativeBuriedMaterialReference:
    event_id: str
    payload_sha256: str
    loss: NativeLoss
    donor_history: tuple[RotationIncrement, ...]
    original_geometry: NativePolygon
    record_sha256: str


@dataclass(frozen=True, slots=True)
class NativeBurialTransaction:
    transaction_id: str
    before_surface_sha256: str
    endpoint: NativeSurfaceState
    increments: tuple[tuple[int, RotationIncrement], ...]
    event_ids: tuple[str, ...]
    births: tuple[NativeFragment, ...]
    transaction_sha256: str


@dataclass(frozen=True, slots=True)
class NativeBuriedRegistry:
    origin_surface_sha256: str
    current_surface_sha256: str
    owner_histories: tuple[tuple[int, tuple[RotationIncrement, ...]], ...]
    records: tuple[NativeBuriedMaterialReference, ...]
    transactions: tuple[NativeBurialTransaction, ...]
    registry_sha256: str


@dataclass(frozen=True, slots=True)
class NativeOriginalMaterialDomain:
    material_id: str
    owner: int
    original_fragment_ids: tuple[str, ...]
    live_fragment_ids: tuple[str, ...]
    buried_event_ids: tuple[str, ...]
    original_measure_sr: float
    conserved_reservoirs: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class NativeOriginalMaterialSupportReceipt:
    model: str
    origin_surface_sha256: str
    current_surface_sha256: str
    registry_sha256: str
    owner_histories: tuple[tuple[int, tuple[RotationIncrement, ...]], ...]
    original_material_domains: tuple[NativeOriginalMaterialDomain, ...]
    live_clamp_support: str
    receipt_sha256: str


def _record_body(record):
    return dict(event_id=record.event_id, payload_sha256=record.payload_sha256,
        loss=native_loss_to_dict(record.loss),
        donor_history=[rotation_increment_to_dict(step) for step in record.donor_history],
        original_geometry=native_polygon_to_dict(record.original_geometry))


def _transaction_body(transaction):
    return dict(transaction_id=transaction.transaction_id,
        before_surface_sha256=transaction.before_surface_sha256,
        endpoint=native_surface_to_dict(transaction.endpoint),
        increments=[[owner, rotation_increment_to_dict(step)] for owner, step in transaction.increments],
        event_ids=list(transaction.event_ids), births=[_fragment_to_dict(item) for item in transaction.births])


def _registry_body(registry):
    return dict(origin_surface_sha256=registry.origin_surface_sha256,
        current_surface_sha256=registry.current_surface_sha256,
        owner_histories=[[owner, [rotation_increment_to_dict(step) for step in history]]
            for owner, history in registry.owner_histories],
        records=[dict(**_record_body(item), record_sha256=item.record_sha256) for item in registry.records],
        transactions=[dict(**_transaction_body(item), transaction_sha256=item.transaction_sha256)
            for item in registry.transactions])


def _seal_record(record):
    return replace(record, record_sha256=_hash(_record_body(record)))


def _seal_transaction(transaction):
    return replace(transaction, transaction_sha256=_hash(_transaction_body(transaction)))


def _seal_registry(registry):
    return replace(registry, registry_sha256=_hash(_registry_body(registry)))


def initial_native_buried_registry(origin):
    """Declare a fresh native material origin, excluding all prior legacy burial."""
    digest = native_surface_digest(origin)
    return _seal_registry(NativeBuriedRegistry(digest, digest,
        tuple((owner, ()) for owner in _owners(origin)), (), (), ""))


def _subtract_all(geometry, covers):
    remaining = (geometry,)
    for cover in covers:
        remaining = tuple(piece for current in remaining for piece in native_subtract(current, cover))
        if not remaining:
            break
    return remaining


def _prove_partition(parent, geometries, label):
    geometries = tuple(geometries)
    if not geometries:
        raise ValueError(label + " has missing positive material support")
    for index, geometry in enumerate(geometries):
        if _subtract_all(geometry, (parent,)):
            raise ValueError(label + " extends outside its authoritative donor footprint")
        if any(native_intersection(geometry, other) is not None for other in geometries[index + 1:]):
            raise ValueError(label + " duplicates positive material support")
    if _subtract_all(parent, geometries):
        raise ValueError(label + " has missing positive material support")


def _same_material(child, parent, age):
    parcel = child.parcel
    wanted = parent.parcel
    if (parcel.material_id != wanted.material_id or parcel.plate != wanted.plate
            or parcel.cell != wanted.cell or parcel.material_fields != wanted.material_fields
            or parcel.specific_properties != wanted.specific_properties or parcel.age_myr != age):
        raise ValueError("Native accepted material intensive history or parent lineage changed")


def _check_transaction(before, endpoint, records, births, increments):
    if type(endpoint) is not NativeSurfaceState or endpoint.radius_km != before.radius_km:
        raise ValueError("Native burial requires an actual same-radius endpoint partition")
    endpoint = native_surface_from_dict(native_surface_to_dict(endpoint))
    owners = tuple(sorted({fragment.parcel.plate for fragment in before.fragments}))
    if (not isinstance(increments, tuple) or any(not isinstance(row, tuple) or len(row) != 2
            for row in increments) or tuple(owner for owner, _ in increments) != tuple(sorted(set(owner for owner, _ in increments)))
            or not set(owners).issubset(owner for owner, _ in increments)
            or any(isinstance(owner, bool) or not isinstance(owner, Integral) or owner < 0
                   or type(step) is not RotationIncrement for owner, step in increments)):
        raise ValueError("Native burial needs one full physical increment per current owner")
    motion = dict(increments)
    dt = increments[0][1].dt_myr
    if (any(step.dt_myr != dt for _, step in increments)
            or endpoint.time_myr != before.time_myr + dt or endpoint.time_myr <= before.time_myr):
        raise ValueError("Native burial transaction changed its single physical clock")
    originals = {fragment.fragment_id: fragment for fragment in before.fragments}
    current = {fragment.fragment_id: fragment for fragment in endpoint.fragments}
    born = {fragment.fragment_id: fragment for fragment in births}
    if (len(born) != len(births) or any(identity not in current or current[identity] != fragment
            for identity, fragment in born.items())):
        raise ValueError("Native births must be the actual endpoint fragment payloads")
    fresh = [fragment.parcel.material_id for fragment in births]
    if (len(set(fresh)) != len(fresh) or any(identity in before.known_material_ids for identity in fresh)
            or set(endpoint.known_material_ids) != set(before.known_material_ids) | set(fresh)):
        raise ValueError("Native burial changed its fresh historical material manifest")
    children = {identity: [] for identity in originals}
    losses = {identity: [] for identity in originals}
    for fragment in endpoint.fragments:
        if fragment.fragment_id in born:
            parent = originals.get(fragment.parent_fragment_id)
            if (parent is None or fragment.parcel.plate != parent.parcel.plate
                    or fragment.parcel.cell != parent.parcel.cell
                    or fragment.parcel.age_myr != 0. or fragment.parcel.cold_mantle_volume_km3 != 0.
                    or fragment.parcel.density_excess_mass_kg != 0.):
                raise ValueError("Native newborn payload does not belong to an actual local hot vacancy")
            continue
        parent_id = fragment.fragment_id if fragment.fragment_id in originals else fragment.parent_fragment_id
        if parent_id not in originals:
            raise ValueError("Native live child has no actual preceding parent")
        children[parent_id].append(fragment)
    for record in records:
        loss = record.loss
        native_loss_to_dict(loss)
        source = originals.get(loss.source_fragment_id)
        receiver = current.get(loss.receiver_fragment_id)
        if (source is None or receiver is None or loss.time_myr != endpoint.time_myr
                or loss.fragment.parent_fragment_id != source.fragment_id
                or receiver.parcel.plate != loss.receiver_plate
                or loss.receiver_plate == loss.fragment.parcel.plate
                or _subtract_all(loss.fragment.geometry, (receiver.geometry,))):
            raise ValueError("Native accepted loss lacks its actual donor or receiver payload")
        measured = loss.fragment.geometry.area_sr * before.radius_km**2
        if not math.isclose(measured, loss.fragment.parcel.area_km2, rel_tol=2e-10, abs_tol=0.):
            raise ValueError("Native accepted loss area disagrees with its exact footprint")
        losses[source.fragment_id].append(loss.fragment)
    for identity, parent in originals.items():
        carried = tuple(children[identity] + losses[identity])
        for fragment in carried:
            _same_material(fragment, parent, parent.parcel.age_myr + dt)
        moved = parent.geometry.rotated(motion[parent.parcel.plate].omega_rad_per_myr, dt)
        _prove_partition(moved, (fragment.geometry for fragment in carried), "Native parent " + identity)
        for field in EXTENSIVE_FIELDS:
            expected = getattr(parent.parcel, field)
            actual = math.fsum(getattr(fragment.parcel, field) for fragment in carried)
            if not math.isclose(actual, expected, rel_tol=64. * _EPS * (len(carried) + 1), abs_tol=0.):
                raise ValueError("Native parent lost its " + field + " reservoir")
    # The old native domain minus the retained endpoint geometry is the real
    # vacancy. Per departed parent it must be filled exactly by its own births.
    retained = tuple(fragment.geometry for identity, fragment in current.items() if identity not in born)
    # A fully audited endpoint containing only retained descendants already
    # covers the sphere. No local positive vacancy can then exist.
    vacancy_parents = originals.items() if births else ()
    for identity, parent in vacancy_parents:
        vacancies = _subtract_all(parent.geometry, retained)
        local_births = tuple(fragment.geometry for fragment in births if fragment.parent_fragment_id == identity)
        for geometry in local_births:
            if _subtract_all(geometry, vacancies):
                raise ValueError("Native birth extends outside its actual local vacancy")
        for index, geometry in enumerate(local_births):
            if any(native_intersection(geometry, other) is not None for other in local_births[index + 1:]):
                raise ValueError("Native births duplicate a positive local vacancy")
        for vacancy in vacancies:
            if _subtract_all(vacancy, local_births):
                raise ValueError("Native births leave a positive local vacancy")
    return endpoint


def validate_native_buried_registry(origin, registry, current):
    """Strictly replay all accepted native parent/receiver and material proofs."""
    if (type(registry) is not NativeBuriedRegistry or registry.origin_surface_sha256 != native_surface_digest(origin)
            or registry.current_surface_sha256 != native_surface_digest(current)
            or _seal_registry(registry) != registry
            or not isinstance(registry.records, tuple) or not isinstance(registry.transactions, tuple)
            or any(type(item) is not NativeBuriedMaterialReference for item in registry.records)
            or any(type(item) is not NativeBurialTransaction for item in registry.transactions)):
        raise ValueError("Native buried registry binding or integrity changed")
    expected_histories = tuple((owner, ()) for owner in _owners(origin))
    supplied_histories = _histories(origin, registry.owner_histories)
    records = {record.event_id: record for record in registry.records}
    if len(records) != len(registry.records):
        raise ValueError("Native accepted events must be unique")
    previous, accepted_ids, ordered = origin, set(), []
    for transaction in registry.transactions:
        if (transaction.before_surface_sha256 != native_surface_digest(previous)
                or _seal_transaction(transaction) != transaction
                or not isinstance(transaction.event_ids, tuple)
                or tuple(sorted(set(transaction.event_ids))) != transaction.event_ids
                or accepted_ids.intersection(transaction.event_ids)
                or any(identity not in records for identity in transaction.event_ids)):
            raise ValueError("Native acceptance transaction chronology or manifest changed")
        identity_body = _transaction_body(transaction)
        identity_body.pop("transaction_id")
        if transaction.transaction_id != "native-burial-transaction:" + _hash(identity_body):
            raise ValueError("Native acceptance transaction identity changed")
        if tuple(owner for owner, _ in transaction.increments) != _owners(origin):
            raise ValueError("Native acceptance motion omitted an original physical owner")
        step_histories = dict(expected_histories)
        for owner, step in transaction.increments:
            if owner not in step_histories:
                raise ValueError("Native accepted owner is outside its declared material origin")
            step_histories[owner] += (step,)
        expected_histories = tuple(sorted(step_histories.items()))
        selected = tuple(records[identity] for identity in transaction.event_ids)
        for record in selected:
            loss_digest = _hash(native_loss_to_dict(record.loss))
            if (record.payload_sha256 != loss_digest or record.event_id != "native-accepted-loss:" + loss_digest
                    or record.donor_history != step_histories[record.loss.fragment.parcel.plate]
                    or record.original_geometry != native_pullback(record.loss.fragment.geometry, record.donor_history)
                    or _seal_record(record) != record):
                raise ValueError("Native accepted loss payload or full donor map changed")
        previous = _check_transaction(previous, transaction.endpoint, selected,
                                      transaction.births, transaction.increments)
        accepted_ids.update(transaction.event_ids)
        ordered.extend(transaction.event_ids)
    if (accepted_ids != set(records) or tuple(ordered) != tuple(record.event_id for record in registry.records)
            or previous != current or expected_histories != supplied_histories):
        raise ValueError("Native registry omitted an accepted event, motion, or endpoint")
    return registry


def advance_native_buried_registry(origin, previous, before, result, owner_histories_at_endpoint):
    """Authenticate exactly one actual native transport interval, atomically."""
    validate_native_buried_registry(origin, previous, before)
    if type(result) is not NativeTransportResult:
        raise ValueError("Native acceptance requires its actual native transaction result")
    histories = _histories(origin, owner_histories_at_endpoint)
    old = dict(previous.owner_histories)
    increments = []
    for owner, history in histories:
        if len(history) != len(old[owner]) + 1 or history[:-1] != old[owner]:
            raise ValueError("Native acceptance maps must append one actual full increment")
        increments.append((owner, history[-1]))
    by_owner = dict(histories)
    records = []
    for loss in result.losses:
        digest = _hash(native_loss_to_dict(loss))
        owner = loss.fragment.parcel.plate
        if owner not in by_owner:
            raise ValueError("Native loss donor has no full acceptance map")
        records.append(_seal_record(NativeBuriedMaterialReference("native-accepted-loss:" + digest,
            digest, loss, by_owner[owner], native_pullback(loss.fragment.geometry, by_owner[owner]), "")))
    records.sort(key=lambda item: item.event_id)
    endpoint = _check_transaction(before, result.state, tuple(records), result.births, tuple(increments))
    transaction = NativeBurialTransaction("", native_surface_digest(before), endpoint,
        tuple(increments), tuple(record.event_id for record in records), result.births, "")
    identity_body = _transaction_body(transaction)
    identity_body.pop("transaction_id")
    transaction = _seal_transaction(replace(transaction,
        transaction_id="native-burial-transaction:" + _hash(identity_body)))
    registry = _seal_registry(NativeBuriedRegistry(previous.origin_surface_sha256,
        native_surface_digest(endpoint), histories, previous.records + tuple(records),
        previous.transactions + (transaction,), ""))
    validate_native_buried_registry(origin, registry, endpoint)
    return registry


def validate_native_original_material_support(origin, current, registry, current_owner_histories=None):
    """Prove the original native footprints are covered once by live and buried material."""
    validate_native_buried_registry(origin, registry, current)
    histories = registry.owner_histories if current_owner_histories is None else _histories(origin, current_owner_histories)
    if histories != registry.owner_histories:
        raise ValueError("Native support uses a different full current donor map")
    maps = dict(histories)
    originals = {}
    for fragment in origin.fragments:
        originals.setdefault(fragment.parcel.material_id, []).append(fragment)
    domains = []
    for identity, parents in sorted(originals.items()):
        parcel = parents[0].parcel
        if any((parent.parcel.plate, parent.parcel.material_fields, parent.parcel.specific_properties,
                parent.parcel.age_myr) != (parcel.plate, parcel.material_fields,
                parcel.specific_properties, parcel.age_myr) for parent in parents):
            raise ValueError("One native original material ID has heterogeneous properties")
        live = tuple(fragment for fragment in current.fragments if fragment.parcel.material_id == identity)
        buried = tuple(record for record in registry.records if record.loss.fragment.parcel.material_id == identity)
        pieces = tuple(native_pullback(fragment.geometry, maps[parcel.plate]) for fragment in live)
        pieces += tuple(record.original_geometry for record in buried)
        parent_shapes = tuple(parent.geometry for parent in parents)
        for index, geometry in enumerate(pieces):
            if _subtract_all(geometry, parent_shapes):
                raise ValueError("Native original material extends outside its 2D origin")
            if any(native_intersection(geometry, other) is not None for other in pieces[index + 1:]):
                raise ValueError("Native original live/buried domains overlap")
        for parent in parents:
            if _subtract_all(parent.geometry, pieces):
                raise ValueError("Native original material has missing 2D support")
        carried = tuple(fragment.parcel for fragment in live) + tuple(record.loss.fragment.parcel for record in buried)
        if any(material.plate != parcel.plate or material.material_fields != parcel.material_fields
               or material.specific_properties != parcel.specific_properties for material in carried):
            raise ValueError("Native original material changed owner or intensive history")
        reservoirs = []
        for field in EXTENSIVE_FIELDS:
            expected = math.fsum(getattr(parent.parcel, field) for parent in parents)
            actual = math.fsum(getattr(material, field) for material in carried)
            if not math.isclose(actual, expected, rel_tol=64. * _EPS * (len(carried) + 1), abs_tol=0.):
                raise ValueError("Native original material lost its " + field + " reservoir")
            reservoirs.append((field, expected))
        domains.append(NativeOriginalMaterialDomain(identity, parcel.plate,
            tuple(sorted(parent.fragment_id for parent in parents)),
            tuple(sorted(fragment.fragment_id for fragment in live)),
            tuple(sorted(record.event_id for record in buried)),
            math.fsum(parent.geometry.area_sr for parent in parents), tuple(reservoirs)))
    receipt = NativeOriginalMaterialSupportReceipt(SUPPORT_MODEL, registry.origin_surface_sha256,
        registry.current_surface_sha256, registry.registry_sha256, histories, tuple(domains),
        "unresolved_separate_live_continuous_clamp_certificate_required", "")
    body = asdict(receipt)
    body.pop("receipt_sha256")
    return replace(receipt, receipt_sha256=_hash(body))


def native_buried_registry_to_dict(registry):
    if type(registry) is not NativeBuriedRegistry or _seal_registry(registry) != registry:
        raise ValueError("Invalid native buried registry integrity")
    return dict(format=FORMAT, registry=dict(**_registry_body(registry), registry_sha256=registry.registry_sha256))


def native_buried_registry_from_dict(raw, origin, current):
    if (not isinstance(raw, dict) or set(raw) != {"format", "registry"} or raw["format"] != FORMAT):
        raise ValueError("Invalid native buried registry format")
    body = raw["registry"]
    if (not isinstance(body, dict) or set(body) != {field.name for field in fields(NativeBuriedRegistry)}
            or any(not isinstance(body[key], list) for key in ("owner_histories", "records", "transactions"))):
        raise ValueError("Invalid native buried registry fields")
    def rows(values, *, full):
        if not isinstance(values, list) or any(not isinstance(row, list) or len(row) != 2 for row in values):
            raise ValueError("Invalid native chronological owner map rows")
        result = []
        for owner, value in values:
            if full:
                if not isinstance(value, list):
                    raise ValueError("Invalid native full donor history")
                value = tuple(rotation_increment_from_dict(step) for step in value)
            else:
                value = rotation_increment_from_dict(value)
            result.append((owner, value))
        return tuple(result)
    records = []
    for row in body["records"]:
        if (not isinstance(row, dict) or set(row) != {field.name for field in fields(NativeBuriedMaterialReference)}
                or not isinstance(row["donor_history"], list)):
            raise ValueError("Invalid native buried material record fields")
        records.append(NativeBuriedMaterialReference(**{**row,
            "loss": native_loss_from_dict(row["loss"]),
            "donor_history": tuple(rotation_increment_from_dict(step) for step in row["donor_history"]),
            "original_geometry": native_polygon_from_dict(row["original_geometry"])}))
    transactions = []
    for row in body["transactions"]:
        if (not isinstance(row, dict) or set(row) != {field.name for field in fields(NativeBurialTransaction)}
                or any(not isinstance(row[key], list) for key in ("event_ids", "births"))):
            raise ValueError("Invalid native burial transaction fields")
        transactions.append(NativeBurialTransaction(**{**row,
            "endpoint": native_surface_from_dict(row["endpoint"]),
            "increments": rows(row["increments"], full=False),
            "event_ids": tuple(row["event_ids"]),
            "births": tuple(_fragment_from_dict(item) for item in row["births"])}))
    registry = NativeBuriedRegistry(**{**body,
        "owner_histories": rows(body["owner_histories"], full=True),
        "records": tuple(records), "transactions": tuple(transactions)})
    return validate_native_buried_registry(origin, registry, current)


@dataclass(frozen=True, slots=True)
class NativeLiveClampSupport:
    trace_id: str
    bank: str
    owner: int
    width_m: float
    exact_width_radians: tuple[int, int]
    enclosing_geometry: NativePolygon
    live_original_fragment_ids: tuple[str, ...]
    analytic_enclosure: str


@dataclass(frozen=True, slots=True)
class NativeBendingBankSupportReceipt:
    model: str
    origin_surface_sha256: str
    current_surface_sha256: str
    registry_sha256: str
    trace_sha256: str
    bending_source_sha256: str
    material_support_sha256: str
    owner_histories: tuple[tuple[int, tuple[RotationIncrement, ...]], ...]
    live_clamps: tuple[NativeLiveClampSupport, ...]
    receipt_sha256: str


def _quarter_clamp_envelope(trace, width_m, radius_km, sign, bank):
    """An exact rational enclosure of the whole physical small-circle clamp.

    For an exact orthonormal quarter frame S,E,H, the physical curve is
    proportional to S cos(theta)+E sin(theta)+sign H tan(w), 0<=theta<=pi/2.
    Its axial sum is in [1,sqrt(2)] and w<tan(w)<4w for 0<w<=1/4.
    The exact quad with inner coefficient w/4 and outer coefficient 4w
    therefore contains every curve point, including both endpoint seams.
    These inequalities are analytic; no trigonometric float sampling is used.
    """
    from fractions import Fraction
    from .geometric_native_polygons import NativePlane, NativeRay, _cross, _dot
    from .geometric_fault_motion_support import UnresolvedFaultMotionSupport

    start = tuple(Fraction.from_float(float(value)) for value in trace.start)
    end = tuple(Fraction.from_float(float(value)) for value in trace.end)
    if _dot(start, start) != 1 or _dot(end, end) != 1 or _dot(start, end) != 0:
        raise UnresolvedFaultMotionSupport("Native live clamp enclosure requires exact orthonormal quarter endpoints")
    horizontal = _cross(start, end)
    width = Fraction.from_float(float(width_m)) / (Fraction.from_float(float(radius_km)) * 1000)
    if not 0 < width <= Fraction(1, 4):
        raise UnresolvedFaultMotionSupport("Native live quarter-clamp proof requires 0 < width/radius <= 1/4")
    inner, outer = width / 4, 4 * width
    points = tuple(tuple(value + sign * offset * normal for value, normal in zip(point, horizontal))
                   for point, offset in ((start, inner), (end, inner), (end, outer), (start, outer)))
    rays = tuple(NativeRay("seed", point) for point in points)
    normals = tuple(_cross(a.reference, b.reference) for a, b in zip(rays, rays[1:] + rays[:1]))
    if any(_dot(normal, ray.reference) < 0 for normal in normals for ray in rays):
        rays = rays[::-1]
        normals = tuple(_cross(a.reference, b.reference) for a, b in zip(rays, rays[1:] + rays[:1]))
    planes = tuple(NativePlane(normal, label="native-clamp-envelope:" + trace.trace_id + ":" + bank)
                   for normal in normals)
    return validate_native_polygon(NativePolygon(rays, planes)), width


def validate_native_bending_bank_support(origin, initial_mechanics, current, registry,
                                        current_owner_histories=None, *, legacy_origin=None):
    """Certify complete retained material and LIVE whole quarter-arc clamps.

    The bounded enclosure is a stronger sufficient gate than curve membership.
    A trace outside the exact quarter-frame domain refuses. Every positive
    enclosure footprint must be covered by actual pulled-back live ORIGINAL
    material, across any number of fragments. Buried material and hot newborn
    material can never supply a live clamp.
    """
    from .geometric_bending_contact_dynamics import BendingContactState, _bending_signature
    from .geometric_boundary import surface_digest
    from .geometric_fault_motion_support import (
        UnresolvedFaultMotionSupport, validate_reference_bending_bank_support,
    )
    from .geometric_native_surface import enroll_native_surface
    from .geometric_fault_traces import validate_fault_trace_set

    if legacy_origin is None or enroll_native_surface(legacy_origin) != origin:
        raise ValueError("Native bending support requires its exact declared legacy-origin enrollment")
    if type(initial_mechanics) is not BendingContactState or initial_mechanics.surface_sha256 != surface_digest(legacy_origin):
        raise ValueError("Native bending support changed its actual original mechanical configuration")
    bounds = tuple(getattr(initial_mechanics, name) for name in (
        "max_rotation_radians", "max_tip_slope", "max_bending_strain", "max_deflection_radius_ratio"))
    if initial_mechanics.bending_source_sha256 != _bending_signature(legacy_origin,
            initial_mechanics.fault_traces, initial_mechanics.bending_parameters, bounds):
        raise ValueError("Native bending support changed original physical widths, traces, or bounds")
    validate_fault_trace_set(legacy_origin, initial_mechanics.fault_traces)
    validate_reference_bending_bank_support(legacy_origin, initial_mechanics.fault_traces,
                                          initial_mechanics.bending_parameters)
    material = validate_native_original_material_support(origin, current, registry, current_owner_histories)
    histories = dict(registry.owner_histories)
    original_ids = {fragment.parcel.material_id for fragment in origin.fragments}
    live = tuple((fragment.fragment_id, fragment.parcel.plate,
                  native_pullback(fragment.geometry, histories[fragment.parcel.plate]))
                 for fragment in current.fragments if fragment.parcel.material_id in original_ids)
    parameters = {item.trace_id: item for item in initial_mechanics.bending_parameters}
    clamps = []
    for trace in initial_mechanics.fault_traces.traces:
        parameter = parameters[trace.trace_id]
        for bank, owner, properties, sign in (("sub", trace.subducting_plate, parameter.sub, -1),
                                            ("over", trace.overriding_plate, parameter.over, 1)):
            envelope, width = _quarter_clamp_envelope(trace, properties.width_m, origin.radius_km, sign, bank)
            original_cover = tuple(fragment.geometry for fragment in origin.fragments if fragment.parcel.plate == owner)
            if _subtract_all(envelope, original_cover):
                raise UnresolvedFaultMotionSupport("Native original quarter-clamp enclosure crosses a foreign owner")
            current_cover = tuple(geometry for _, live_owner, geometry in live if live_owner == owner)
            if _subtract_all(envelope, current_cover):
                raise UnresolvedFaultMotionSupport("Native continuous clamp has missing LIVE original material support")
            clamps.append(NativeLiveClampSupport(trace.trace_id, bank, owner, properties.width_m,
                (width.numerator, width.denominator), envelope,
                tuple(sorted(identity for identity, live_owner, _ in live if live_owner == owner)),
                "exact quarter frame; 1<=cos(theta)+sin(theta)<2; w<tan(w)<4w; inner=w/4; outer=4w"))
    receipt = NativeBendingBankSupportReceipt("native_exact_quarter_live_clamp_enclosure_and_complete_original_2D_domain",
        registry.origin_surface_sha256, registry.current_surface_sha256, registry.registry_sha256,
        initial_mechanics.fault_traces.trace_sha256, initial_mechanics.bending_source_sha256,
        material.receipt_sha256, registry.owner_histories, tuple(clamps), "")
    body = asdict(receipt)
    body.pop("receipt_sha256")
    return replace(receipt, receipt_sha256=_hash(body))

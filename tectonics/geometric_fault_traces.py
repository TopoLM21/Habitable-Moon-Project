"""Declared physical dipping arcs and quadrature independent of material cuts.

Oriented endpoints explicitly define the horizontal normal from the sinking
bank toward the overriding bank. No fault is inferred from a fragment ID,
damage, age, density or velocity. A trace set is bound to one exact frozen
surface, while its separate physical signature excludes material cut IDs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np

from .genesis_contact_law import ContactLawParameters
from .geometric_boundary import surface_digest
from .geometric_contact_mechanics import ElasticColumnParameters
from .geometric_contacts import extract_contacts
from .geometric_polarity import PolarityEvidence, contact_polarity_report, intersect_arcs
from .geometric_surface import GeometricSurfaceState
from .spherical_polygons import _edge_normal, arc_length

FORMAT = "geometric-physical-fault-traces-1"
FRAME = "global_oriented_minor_arc_cross_start_end_sub_to_over"
EVOLUTION = "fixed_physical_reference_no_fault_advection"


class UnresolvedFaultTrace(ValueError):
    """Physical support or history does not uniquely define the network."""


def _number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be finite and real")
    value = float(value)
    if (positive and value <= 0.) or (nonnegative and value < 0.):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _digest_text(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid {name}")


def _point(value, name):
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3:
        raise ValueError(f"{name} must be a Cartesian unit vector")
    result = tuple(_number(v, name) for v in value)
    if abs(math.fsum(v*v for v in result)-1.) > 64*np.finfo(float).eps:
        raise ValueError(f"{name} must be a Cartesian unit vector")
    return result


def _arc_frame(start, end):
    a, b = np.asarray(start), np.asarray(end)
    length = arc_length(a, b)
    if not 0. < length < math.pi:
        raise ValueError("Physical fault support must be a positive minor great-circle arc")
    normal = _edge_normal(a, b)
    tangent = np.cross(normal, a)
    tangent /= np.sqrt(np.sum(tangent*tangent))
    return a, normal, tangent, length


@dataclass(frozen=True, slots=True)
class PhysicalFaultTrace:
    trace_id: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    subducting_plate: int
    overriding_plate: int
    dip_radians: float
    columns: ElasticColumnParameters
    law: ContactLawParameters
    contact_height_m: float
    water: float
    pore_pressure_pa: float
    initial_gap_m: float
    initial_jump_m: float
    provenance: str

    def __post_init__(self):
        _text(self.trace_id, "Physical trace ID")
        _text(self.provenance, "Physical trace provenance")
        for name in ("start", "end"):
            object.__setattr__(self, name, _point(getattr(self, name), name))
        _arc_frame(self.start, self.end)
        for name in ("subducting_plate", "overriding_plate"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
                raise ValueError("Physical fault banks require nonnegative actual owner IDs")
            object.__setattr__(self, name, int(value))
        if self.subducting_plate == self.overriding_plate:
            raise ValueError("Physical fault banks must differ")
        dip = _number(self.dip_radians, "dip_radians", positive=True)
        if dip >= math.pi/2.:
            raise ValueError("Physical fault dip must lie strictly between zero and pi/2")
        object.__setattr__(self, "dip_radians", dip)
        if not isinstance(self.columns, ElasticColumnParameters) or not isinstance(self.law, ContactLawParameters):
            raise ValueError("Physical traces require explicit existing column and contact parameters")
        self.law.validate()
        for name in ("contact_height_m", "water", "pore_pressure_pa", "initial_gap_m", "initial_jump_m"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.contact_height_m <= 0. or not 0. <= self.water <= 1.:
            raise ValueError("Physical contact height must be positive and water in [0,1]")
        if self.pore_pressure_pa != 0.:
            raise ValueError("Physical trace mechanics currently requires explicit zero pore pressure")
        if self.initial_gap_m > 0.:
            raise UnresolvedFaultTrace("Initial normal opening is unsupported")

    @property
    def normal_sub_to_over(self):
        return tuple(map(float, _arc_frame(self.start, self.end)[1]))


@dataclass(frozen=True, slots=True)
class TraceQuadrature:
    panel_count: int
    gauss_order: int

    def __post_init__(self):
        for name, minimum in (("panel_count", 1), ("gauss_order", 2)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"Explicit {name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True, slots=True)
class FaultTraceSet:
    surface_sha256: str
    time_myr: float
    radius_km: float
    traces: tuple[PhysicalFaultTrace, ...]
    quadrature: TraceQuadrature
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class TraceContactSupport:
    trace_id: str
    contact_id: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    trace_start_radians: float
    trace_end_radians: float
    contact_start_radians: float
    contact_end_radians: float


@dataclass(frozen=True, slots=True)
class FaultTraceCoverage:
    contact_trace_ids: tuple[tuple[str, tuple[str, ...]], ...]
    trace_contact_ids: tuple[tuple[str, tuple[str, ...]], ...]
    overlaps: tuple[TraceContactSupport, ...]


@dataclass(frozen=True, slots=True)
class FaultTracePoint:
    point_id: str
    trace_id: str
    panel_index: int
    node_index: int
    position: tuple[float, float, float]
    horizontal: tuple[float, float, float]
    normal: tuple[float, float, float]
    down_dip: tuple[float, float, float]
    line_weight_m: float
    area_m2: float


@dataclass(frozen=True, slots=True)
class TracePolarityEvidence:
    origin: PolarityEvidence
    trace_id: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    origin_surface_sha256: str

    def __post_init__(self):
        if not isinstance(self.origin, PolarityEvidence):
            raise ValueError("Trace history must retain its original explicit polarity evidence")
        _text(self.trace_id, "Historical physical trace ID")
        _digest_text(self.origin_surface_sha256, "historical origin surface checksum")
        for name in ("start", "end"):
            object.__setattr__(self, name, _point(getattr(self, name), name))
        _arc_frame(self.start, self.end)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _physical_signature(radius_km, traces, quadrature):
    return _hash(dict(format=FORMAT, coordinate_convention=FRAME, evolution=EVOLUTION,
        radius_km=radius_km, traces=[asdict(t) for t in traces], quadrature=asdict(quadrature)))


def _checked_set(trace_set):
    if not isinstance(trace_set, FaultTraceSet):
        raise ValueError("Expected an explicit FaultTraceSet")
    _digest_text(trace_set.surface_sha256, "trace surface checksum")
    _digest_text(trace_set.trace_sha256, "physical trace checksum")
    _number(trace_set.time_myr, "trace time", nonnegative=True)
    _number(trace_set.radius_km, "trace radius", positive=True)
    if not isinstance(trace_set.traces, tuple) or any(not isinstance(t, PhysicalFaultTrace) for t in trace_set.traces):
        raise ValueError("Physical traces must be an immutable tuple")
    if tuple(t.trace_id for t in trace_set.traces) != tuple(sorted({t.trace_id for t in trace_set.traces})):
        raise ValueError("Physical trace IDs must be unique and canonical")
    if not isinstance(trace_set.quadrature, TraceQuadrature):
        raise ValueError("Explicit physical quadrature is required")
    if _physical_signature(trace_set.radius_km, trace_set.traces, trace_set.quadrature) != trace_set.trace_sha256:
        raise ValueError("Physical trace geometry, parameters or quadrature changed")


def _interval(start, end, low, high):
    a, _, tangent, extent = _arc_frame(start, end)
    values = []
    for point in (low, high):
        # Saved endpoints are exact incidence evidence. Reconstructing their
        # zero coordinate as tangent.point introduces absolute cancellation
        # error, which can exceed a short arc's *relative* arithmetic budget.
        if np.array_equal(point, start):
            values.append(0.)
        elif np.array_equal(point, end):
            values.append(extent)
        else:
            origin, position, direction = (np.asarray(value, dtype=np.longdouble)
                for value in (a, point, tangent))
            values.append(float(np.arctan2(np.dot(direction, position-origin),
                                           np.dot(origin, position))))
    # Both clipped endpoints are already on this minor arc. Arithmetic error
    # can put a shared endpoint on either side of zero or extent. Snap only
    # coordinates within that floating-point budget, not a geographic length.
    tolerance = 64*np.finfo(float).eps*extent
    for index, value in enumerate(values):
        if abs(value) <= tolerance:
            values[index] = 0.
        elif abs(value-extent) <= tolerance:
            values[index] = extent
    return tuple(sorted(values))


def _cover_once(intervals, extent, name):
    if not intervals:
        raise UnresolvedFaultTrace(f"Missing physical support for {name}")
    values = sorted(intervals)
    end = 0.
    # Relative angle arithmetic only; do not assign a physical minimum length.
    tolerance = 128*np.finfo(float).eps*extent
    for low, high in values:
        if high <= low:
            raise UnresolvedFaultTrace(f"Nonpositive physical support for {name}")
        if low < end-tolerance:
            raise UnresolvedFaultTrace(f"Multiple physical supports overlap on {name}")
        if low > end+tolerance:
            raise UnresolvedFaultTrace(f"Missing physical support inside {name}")
        end = high
    if abs(end-extent) > tolerance:
        raise UnresolvedFaultTrace(f"Physical support does not cover the endpoint of {name}")


def validate_fault_trace_set(surface, trace_set, history=()):
    """Require exactly one physical plane on every positive contact segment.

    A physical arc may span many material-cut contacts, or declared adjacent
    physical arcs may partition one such contact. Endpoint touch is no support.
    Every positive part of every physical trace must also be a real contact.
    """
    _checked_set(trace_set)
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Physical traces require a resolved frozen surface")
    if (trace_set.surface_sha256 != surface_digest(surface) or trace_set.time_myr != surface.time_myr
            or trace_set.radius_km != surface.radius_km):
        raise ValueError("Physical trace set belongs to another exact surface or time")
    contacts = extract_contacts(surface.fragments, surface.radius_km)
    by_contact = {c.contact_id: [] for c in contacts}
    by_trace = {t.trace_id: [] for t in trace_set.traces}
    overlaps = []
    for trace in trace_set.traces:
        horizontal = np.asarray(trace.normal_sub_to_over)
        for contact in contacts:
            clipped = intersect_arcs(trace.start, trace.end, contact.start, contact.end)
            if clipped is None:
                continue
            ti = _interval(trace.start, trace.end, *clipped)
            ci = _interval(contact.start, contact.end, *clipped)
            # Endpoint touch can be reported as a rounded angular sliver by
            # atan2. After the endpoint arithmetic clamp it has zero measure;
            # it cannot grant support or create an apparent duplicate plane.
            if ti[1] <= ti[0] or ci[1] <= ci[0]:
                continue
            if {trace.subducting_plate, trace.overriding_plate} != {contact.plate_a, contact.plate_b}:
                raise UnresolvedFaultTrace("Physical trace support has incompatible material bank owners")
            contact_normal = np.asarray(contact.normal_a_to_b)*(1. if trace.overriding_plate == contact.plate_b else -1.)
            if float(np.dot(horizontal, contact_normal)) <= 0.:
                raise UnresolvedFaultTrace("Oriented trace endpoints oppose the declared sub-to-over direction")
            overlap = TraceContactSupport(trace.trace_id, contact.contact_id, *clipped, *ti, *ci)
            overlaps.append(overlap)
            by_trace[trace.trace_id].append(overlap)
            by_contact[contact.contact_id].append(overlap)
    for contact in contacts:
        _cover_once([(s.contact_start_radians, s.contact_end_radians) for s in by_contact[contact.contact_id]],
                    arc_length(contact.start, contact.end), f"contact {contact.contact_id}")
    for trace in trace_set.traces:
        _cover_once([(s.trace_start_radians, s.trace_end_radians) for s in by_trace[trace.trace_id]],
                    arc_length(trace.start, trace.end), f"trace {trace.trace_id}")
    reports = contact_polarity_report(contacts, tuple(history))
    traces = {t.trace_id: t for t in trace_set.traces}
    for report in reports:
        if report["status"] == "conflicting":
            raise UnresolvedFaultTrace("Conflicting local material history cannot choose a physical plane")
        direction = tuple(report["directions"][0])
        if any(direction != (traces[s.trace_id].subducting_plate, traces[s.trace_id].overriding_plate)
               for s in by_contact[report["contact_id"]]):
            raise UnresolvedFaultTrace("Physical trace direction opposes the retained material history")
    return FaultTraceCoverage(
        tuple((key, tuple(sorted({s.trace_id for s in values}))) for key, values in sorted(by_contact.items())),
        tuple((key, tuple(sorted({s.contact_id for s in values}))) for key, values in sorted(by_trace.items())),
        tuple(sorted(overlaps, key=lambda s: (s.trace_id, s.trace_start_radians, s.contact_id))))


def declare_fault_trace_set(surface, traces, quadrature):
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Physical traces require a resolved frozen surface")
    values = tuple(traces)
    if any(not isinstance(t, PhysicalFaultTrace) for t in values) or not isinstance(quadrature, TraceQuadrature):
        raise ValueError("Explicit physical trace records and quadrature are required")
    values = tuple(sorted(values, key=lambda t: t.trace_id))
    result = FaultTraceSet(surface_digest(surface), surface.time_myr, surface.radius_km,
        values, quadrature, _physical_signature(surface.radius_km, values, quadrature))
    validate_fault_trace_set(surface, result)
    return result


def fault_trace_points(surface, trace_set):
    """Generate nodes from physical arcs only, never from material intersections."""
    validate_fault_trace_set(surface, trace_set)
    nodes, weights = np.polynomial.legendre.leggauss(trace_set.quadrature.gauss_order)
    radius = trace_set.radius_km*1000.
    result = []
    for trace in trace_set.traces:
        start, horizontal, tangent, extent = _arc_frame(trace.start, trace.end)
        panel_width = extent/trace_set.quadrature.panel_count
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        for panel in range(trace_set.quadrature.panel_count):
            midpoint = (panel+.5)*panel_width
            for index, (node, weight) in enumerate(zip(nodes, weights)):
                angle = midpoint+.5*panel_width*float(node)
                point = start*math.cos(angle)+tangent*math.sin(angle)
                point /= np.sqrt(np.sum(point*point))
                line_weight = radius*.5*panel_width*float(weight)
                area = line_weight*trace.contact_height_m/sine
                if not math.isfinite(area) or area <= 0.:
                    raise ValueError("Physical fault quadrature area overflowed or is nonpositive")
                result.append(FaultTracePoint(
                    f"{trace.trace_id}:panel:{panel}:gauss:{index}", trace.trace_id, panel, index,
                    tuple(map(float, point)), tuple(map(float, horizontal)),
                    tuple(map(float, sine*horizontal+cosine*point)),
                    tuple(map(float, cosine*horizontal-sine*point)), line_weight, area))
    return tuple(result)


def fault_trace_set_to_dict(surface, trace_set):
    validate_fault_trace_set(surface, trace_set)
    return dict(format=FORMAT, coordinate_convention=FRAME, evolution=EVOLUTION, **asdict(trace_set))


def fault_trace_set_from_dict(raw, surface):
    expected = {f.name for f in fields(FaultTraceSet)} | {"format", "coordinate_convention", "evolution"}
    if (not isinstance(raw, dict) or set(raw) != expected or raw["format"] != FORMAT
            or raw["coordinate_convention"] != FRAME or raw["evolution"] != EVOLUTION):
        raise ValueError("Invalid physical fault trace checkpoint schema")
    try:
        if not isinstance(raw["traces"], list) or not isinstance(raw["quadrature"], dict):
            raise ValueError("Stored physical traces and quadrature have invalid types")
        traces = []
        for row in raw["traces"]:
            if not isinstance(row, dict) or set(row) != {f.name for f in fields(PhysicalFaultTrace)}:
                raise ValueError("Physical trace record must retain every explicit field")
            values = dict(row)
            if not isinstance(values["law"], dict) or set(values["law"]) != {f.name for f in fields(ContactLawParameters)}:
                raise ValueError("Physical trace must retain every existing contact law parameter")
            values["law"] = ContactLawParameters(**values["law"])
            values["columns"] = ElasticColumnParameters(**values["columns"])
            traces.append(PhysicalFaultTrace(**values))
        result = FaultTraceSet(raw["surface_sha256"], raw["time_myr"], raw["radius_km"],
            tuple(traces), TraceQuadrature(**raw["quadrature"]), raw["trace_sha256"])
        validate_fault_trace_set(surface, result)
        return result
    except (TypeError, KeyError) as error:
        raise ValueError("Malformed physical fault trace checkpoint record") from error


def bind_fault_trace_history(surface, trace_set, history):
    """Retain original contact evidence and its real physical overlap supports."""
    coverage = validate_fault_trace_set(surface, trace_set)
    contacts = extract_contacts(surface.fragments, surface.radius_km)
    history = tuple(history)
    # This also checks original contact/owner identities and duplicate events.
    contact_polarity_report(contacts, history)
    result = []
    for item in history:
        for support in coverage.overlaps:
            if support.contact_id == item.contact_id:
                result.append(TracePolarityEvidence(item, support.trace_id, support.start,
                                                   support.end, trace_set.surface_sha256))
    return tuple(sorted(result, key=_history_key))


def _history_key(item):
    return item.trace_id, item.origin.evidence_id, item.start, item.end


def validate_fault_trace_history(trace_set, records):
    """Check frozen trace support while retaining old material-cut provenance."""
    _checked_set(trace_set)
    records = tuple(records)
    if any(not isinstance(item, TracePolarityEvidence) for item in records):
        raise ValueError("Physical history requires TracePolarityEvidence records")
    keys = tuple(_history_key(item) for item in records)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("Physical history supports must be unique and canonical")
    traces = {trace.trace_id: trace for trace in trace_set.traces}
    by_trace = {}
    origins = {}
    for item in records:
        trace = traces.get(item.trace_id)
        if trace is None:
            raise ValueError("Historical support refers to an absent physical trace")
        clipped = intersect_arcs(trace.start, trace.end, item.start, item.end)
        if clipped is None:
            raise ValueError("Historical evidence has no positive physical trace support")
        if float(np.dot(_arc_frame(item.start, item.end)[1], np.asarray(trace.normal_sub_to_over))) <= 0.:
            raise ValueError("Historical support endpoints must follow their physical trace orientation")
        original_length, clipped_length = arc_length(item.start, item.end), arc_length(*clipped)
        if abs(original_length-clipped_length) > 128*np.finfo(float).eps*original_length:
            raise ValueError("Historical support extends beyond its physical trace")
        direction = item.origin.subducting_plate, item.origin.overriding_plate
        if set(direction) != {trace.subducting_plate, trace.overriding_plate}:
            raise ValueError("Historical physical support has incompatible owners")
        identity = item.origin.evidence_id
        payload = item.origin, item.origin_surface_sha256
        if identity in origins and origins[identity] != payload:
            raise ValueError("One historical event cannot retain contradictory origin records")
        origins[identity] = payload
        by_trace.setdefault(item.trace_id, []).append((item, direction))
    for trace_id, entries in by_trace.items():
        trace = traces[trace_id]
        expected = trace.subducting_plate, trace.overriding_plate
        # Opposed directions, even on separate retained intervals of a single
        # fixed-orientation trace, cannot be silently collapsed into one plane.
        if len({direction for _, direction in entries}) > 1:
            raise UnresolvedFaultTrace("Conflicting physical trace history remains unresolved")
        if entries[0][1] != expected:
            raise UnresolvedFaultTrace("Physical trace opposes its retained history")
    return records


def trace_history_to_dict(records):
    records = tuple(records)
    if any(not isinstance(item, TracePolarityEvidence) for item in records):
        raise ValueError("Physical history requires TracePolarityEvidence records")
    return [asdict(item) for item in records]


def trace_history_from_dict(raw, trace_set):
    if not isinstance(raw, list):
        raise ValueError("Physical trace history must be a list")
    result = []
    try:
        for row in raw:
            if not isinstance(row, dict) or set(row) != {f.name for f in fields(TracePolarityEvidence)}:
                raise ValueError("Physical history record must retain every origin and support field")
            values = dict(row)
            if not isinstance(values["origin"], dict) or set(values["origin"]) != {f.name for f in fields(PolarityEvidence)}:
                raise ValueError("Historical origin must retain every polarity evidence field")
            values["origin"] = PolarityEvidence(**values["origin"])
            result.append(TracePolarityEvidence(**values))
        return validate_fault_trace_history(trace_set, tuple(result))
    except (TypeError, KeyError) as error:
        raise ValueError("Malformed physical trace history record") from error

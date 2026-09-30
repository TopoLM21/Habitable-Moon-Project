"""Accepted-material boundary inventory for the opt-in young mechanics path.

Signed closure is geometry, not automatically subduction.  Only a conservative
material transaction may create attached slab inventory.  The raster producer
below reports material already removed from the surface; consuming its events
must never remove that material again.  A future fractional transport can use
the same event API, provided raster reconciliation does not emit it twice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
import math
from typing import Iterable

import numpy as np

from .kinematics import BoundaryRecord, BoundaryType, classify_boundaries

MODEL_VERSION = "young-boundary-material-2"


@dataclass(slots=True)
class SlabAcceptance:
    """One already-accepted parcel, with actual loss/receiver and SI buoyancy."""
    source_face: int
    target_face: int
    subducting_plate: int
    overriding_plate: int
    accepted_area_km2: float
    oceanic_volume_km3: float
    cold_mantle_volume_km3: float
    density_excess_mass_kg: float
    contact_key: str
    trench_length_km: float
    midpoint: list[float]
    torque_direction: list[float]
    attached: bool = True


@dataclass(slots=True)
class YoungContact:
    key: str
    face_a: int
    face_b: int
    plate_a: int
    plate_b: int
    trench_length_km: float
    midpoint: list[float]
    torque_direction_ab: list[float]
    normal_rate_km_per_myr: float = 0.
    cumulative_closure_area_km2: float = 0.
    cumulative_opening_area_km2: float = 0.
    present: bool = True


@dataclass(slots=True)
class SlabThermalCohort:
    acceptance_time_myr: float
    accepted_area_km2: float
    oceanic_volume_km3: float
    cold_mantle_volume_km3: float
    initial_density_excess_mass_kg: float
    initial_buoyancy_moment_kg: list[float]
    initial_thickness_km: float
    deep_transfer_fraction: float = 0.


@dataclass(slots=True)
class YoungSlabSegment:
    key: str
    contact_key: str
    subducting_plate: int
    overriding_plate: int
    source_anchor: list[float]
    receiver_anchor: list[float]
    midpoint: list[float]
    trench_length_km: float
    accepted_area_km2: float = 0.
    oceanic_volume_km3: float = 0.
    cold_mantle_volume_km3: float = 0.
    density_excess_mass_kg: float = 0.
    # Σ positive density excess * accepted volume * (r × towards_trench).
    # Its magnitude is retained; cancelling segments must not become unit axes.
    buoyancy_moment_kg: list[float] = field(default_factory=lambda: [0., 0., 0.])
    first_acceptance_time_myr: float = 0.
    last_acceptance_time_myr: float = 0.
    attached: bool = True
    thermal_cohorts: list[SlabThermalCohort] = field(default_factory=list)


@dataclass(slots=True)
class YoungBoundaryState:
    version: str = MODEL_VERSION
    contacts: dict[str, YoungContact] = field(default_factory=dict)
    segments: dict[str, YoungSlabSegment] = field(default_factory=dict)
    cumulative_accepted_area_km2: float = 0.
    cumulative_accepted_oceanic_volume_km3: float = 0.
    last_material_time_myr: float = -math.inf
    thermal_diffusivity_m2_s: float = 1e-6
    mantle_depth_km: float | None = None
    # Legacy inventories retain their original fixed-edge interpretation. New
    # mechanics must opt into local, conservative attachment transport.
    connectivity_model: str = "legacy_fixed_contacts"
    mechanical_detachments: list[dict] = field(default_factory=list)
    # Explicitly persisted: changing distal cohort removal is a model change.
    buoyancy_geometry_model: str = "uniform_thermal_mass_v1"


@dataclass(frozen=True, slots=True)
class SlabBuoyancyLayer:
    """A retained acceptance batch occupying one interval along the slab.

    Newest material starts at the trench. Equal-time parcels have no resolved
    order and share one layer; their thermal deficits are evaluated separately
    before aggregation. All extensive values exclude material transferred deep.
    """
    acceptance_time_myr: float
    age_myr: float
    arc_start_km: float
    arc_end_km: float
    accepted_area_km2: float
    cold_mantle_volume_km3: float
    density_excess_mass_kg: float
    area_thickness_cubed_km5: float


@dataclass(frozen=True, slots=True)
class SlabForceSection:
    """One currently connected trench section, aggregated before force assembly.

    Extensive slab properties include only material still above the physical
    mantle-depth transfer boundary. Thickness cubed is area integrated so a
    bending closure can use its physical moment without counting a resistance
    once per saved thermal cohort. All geometry is that of the current contact.
    """
    contact_key: str
    subducting_plate: int
    overriding_plate: int
    source_face: int
    receiver_face: int
    trench_length_km: float
    accepted_area_km2: float
    slab_length_km: float
    slab_depth_km: float
    dip_deg: float
    oceanic_volume_km3: float
    cold_mantle_volume_km3: float
    area_thickness_cubed_km5: float
    density_excess_mass_kg: float
    midpoint: tuple[float, float, float]
    torque_direction: tuple[float, float, float]
    buoyancy_layers: tuple[SlabBuoyancyLayer, ...] = ()


def _matching_contact(inventory, segment):
    contact = inventory.contacts.get(segment.contact_key)
    return contact if (contact is not None and contact.present and
        {contact.plate_a, contact.plate_b} ==
        {segment.subducting_plate, segment.overriding_plate}) else None


def _contact_dip(cohorts, time_myr, params):
    age = max(0., float(time_myr)-min(c.acceptance_time_myr for c in cohorts))
    return params.initial_dip_deg+(params.mature_dip_deg-params.initial_dip_deg)*(
        1.-math.exp(-age/max(params.dip_maturation_myr, 1e-9)))


def _contact_vertices(key):
    parts = key.split(':')
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return frozenset()
    return frozenset(map(int, parts))


def _transport_attachment_geometry(mesh, inventory):
    """Move attachment only across adjacent raster edges of the same plate pair.

    A grid-edge hop is not physical breakoff. A same-pair contact elsewhere on
    the sphere is not evidence of a connected slab either. The shared-vertex
    rule permits a local edge hop and otherwise retains material in the
    detached ledger. No force, mass, age, or thermal state is created here.
    """
    if inventory.connectivity_model == "legacy_fixed_contacts":
        return
    if inventory.connectivity_model != "local_edge_transfer_v1":
        raise ValueError("Unknown young slab connectivity model")
    by_pair = {}
    for contact in inventory.contacts.values():
        if contact.present:
            by_pair.setdefault(frozenset((contact.plate_a, contact.plate_b)), []).append(contact)
    for segment in inventory.segments.values():
        if not segment.attached:
            continue
        contact = _matching_contact(inventory, segment)
        if contact is None:
            vertices = _contact_vertices(segment.contact_key)
            candidates = [candidate for candidate in by_pair.get(
                frozenset((segment.subducting_plate, segment.overriding_plate)), [])
                if vertices & _contact_vertices(candidate.key)]
            if not candidates:
                segment.attached = False
                continue
            contact = min(candidates, key=lambda candidate: (
                -float(np.dot(segment.midpoint, candidate.midpoint)), candidate.key))
        segment.contact_key = contact.key
        segment.midpoint = list(contact.midpoint)
        segment.trench_length_km = contact.trench_length_km
        source, receiver = contact.face_a, contact.face_b
        if contact.plate_a != segment.subducting_plate:
            source, receiver = receiver, source
        segment.source_anchor = mesh.centroids[source].tolist()
        segment.receiver_anchor = mesh.centroids[receiver].tolist()


def slab_thermal_deficit_fraction(age_myr,thickness_km,diffusivity_m2_s):
    """Mean deficit in a finite sheet warmed from both faces by a fixed bath.

    κ is the existing material diffusivity. No fitted relaxation time is used.
    The exact short-time heat-content limit avoids a truncated-series jump at
    age=0. This is passive slab rheology; it adds no heat to global Genesis.
    """
    from .genesis import SECONDS_PER_MYR
    if thickness_km <= 0:
        return 0.
    if not math.isfinite(diffusivity_m2_s) or diffusivity_m2_s <= 0:
        raise ValueError("Slab thermal diffusivity must be positive and finite")
    fourier=diffusivity_m2_s*max(float(age_myr),0.)*SECONDS_PER_MYR/(thickness_km*1000.)**2
    if fourier <= .001:
        return max(1.-4.*math.sqrt(fourier/math.pi),0.)
    odd=np.arange(1,max(3,int(math.ceil(math.sqrt(40./(math.pi**2*fourier))))+2),2,dtype=float)
    return float(8./math.pi**2*np.sum(np.exp(-math.pi**2*fourier*odd*odd)/(odd*odd)))


def _cohort_current(cohort,inventory,time_myr):
    retained=1.-cohort.deep_transfer_fraction
    thermal=slab_thermal_deficit_fraction(time_myr-cohort.acceptance_time_myr,
        cohort.initial_thickness_km,inventory.thermal_diffusivity_m2_s)
    return retained,retained*thermal


def limit_attached_inventory(inventory,dip_deg,*,time_myr=None,params=None):
    """Move oldest material beyond physical mantle depth to a retained ledger.

    Depth comes from the physical interior model; the numerical slab-length
    cap is not used. Dip is the existing explicit geometric closure. No mass
    or cumulative accepted volume is deleted by this FIFO depth accounting.
    The nominal straight-slab depth is a conservative bound for a shallow bend;
    no incoming surface thickness is available here to reconstruct that bend.
    Ordered geometry removes equal-time parcels proportionally: their relative
    distal order is unresolved, so dictionary order cannot select which cools.
    """
    if inventory.mantle_depth_km is None:
        return
    if not math.isfinite(inventory.mantle_depth_km) or inventory.mantle_depth_km <= 0:
        raise ValueError("Physical slab mantle depth must be finite and positive")
    sine=math.sin(math.radians(float(dip_deg)))
    if sine <= 0:
        raise ValueError("Attached slab dip must be positive")
    groups={}
    for segment in inventory.segments.values():
        if segment.attached:
            key=(segment.contact_key,segment.subducting_plate,segment.overriding_plate)
            groups.setdefault(key,[]).append(segment)
    for segments in groups.values():
        width=max(s.trench_length_km for s in segments)
        cohorts=[c for s in segments for c in s.thermal_cohorts]
        if params is not None and cohorts:
            local_dip=_contact_dip(cohorts,time_myr,params)
            sine=math.sin(math.radians(local_dip))
        capacity=width*inventory.mantle_depth_km/sine
        if inventory.buoyancy_geometry_model == "ordered_thermal_cohorts_v1":
            excess=math.fsum(c.accepted_area_km2*(1.-c.deep_transfer_fraction)
                for c in cohorts)-capacity
            batches={}
            for cohort in cohorts:
                batches.setdefault(cohort.acceptance_time_myr,[]).append(cohort)
            for _,batch in sorted(batches.items()):
                if excess <= 0:
                    break
                current=math.fsum(c.accepted_area_km2*(1.-c.deep_transfer_fraction)
                    for c in batch)
                if current <= 0:
                    continue
                moved=min(current,excess)
                remaining=1.-moved/current
                for cohort in batch:
                    cohort.deep_transfer_fraction=1.-(1.-cohort.deep_transfer_fraction)*remaining
                excess-=moved
            continue
        if inventory.buoyancy_geometry_model != "uniform_thermal_mass_v1":
            raise ValueError("Unknown young slab buoyancy geometry model")
        excess=sum(c.accepted_area_km2*(1.-c.deep_transfer_fraction) for c in cohorts)-capacity
        for cohort in sorted(cohorts,key=lambda c:c.acceptance_time_myr):
            if excess <= 0:
                break
            current=cohort.accepted_area_km2*(1.-cohort.deep_transfer_fraction)
            moved=min(current,excess)
            cohort.deep_transfer_fraction=min(1.,cohort.deep_transfer_fraction+moved/cohort.accepted_area_km2)
            excess-=moved


def _geometry(mesh, b, radius_km):
    r = np.asarray(b.midpoint, dtype=float)
    d = mesh.centroids[b.face_b] - mesh.centroids[b.face_a]
    d = d-r*np.dot(d, r)
    norm = float(np.linalg.norm(d))
    if norm < 1e-14:
        raise ValueError("Degenerate physical contact")
    direction = np.cross(r, d/norm)
    length = float(radius_km)*math.acos(float(np.clip(
        mesh.vertices[b.vertex_u]@mesh.vertices[b.vertex_v], -1., 1.)))
    return length, r, direction


def contact_key(b):
    return f"{min(b.vertex_u,b.vertex_v)}:{max(b.vertex_u,b.vertex_v)}"


def advance_contact_geometry(mesh, boundaries: Iterable[BoundaryRecord], radius_km,
                             dt_myr, inventory: YoungBoundaryState):
    """Keep exact signed candidates independent of diagnostic classification.

    Cumulative closure/opening are kinematic area integrals only.  They never
    enter slab mass, energy, or length until an accepted material event exists.
    """
    if not math.isfinite(dt_myr) or dt_myr <= 0:
        raise ValueError("Contact geometry needs a positive finite timestep")
    for contact in inventory.contacts.values():
        contact.present = False
    for b in boundaries:
        key = contact_key(b)
        length, r, torque = _geometry(mesh, b, radius_km)
        previous = inventory.contacts.get(key)
        if previous is None or (previous.face_a, previous.face_b) != (b.face_a,b.face_b):
            previous = YoungContact(key,b.face_a,b.face_b,b.plate_a,b.plate_b,
                                    length,r.tolist(),torque.tolist())
            inventory.contacts[key] = previous
        previous.plate_a, previous.plate_b = int(b.plate_a),int(b.plate_b)
        previous.trench_length_km = length
        previous.midpoint, previous.torque_direction_ab = r.tolist(),torque.tolist()
        previous.normal_rate_km_per_myr = float(b.normal_rate_km_per_myr)
        previous.cumulative_closure_area_km2 += length*max(-b.normal_rate_km_per_myr,0.)*dt_myr
        previous.cumulative_opening_area_km2 += length*max(b.normal_rate_km_per_myr,0.)*dt_myr
        previous.present = True
    _transport_attachment_geometry(mesh, inventory)


def raster_acceptance_events(mesh, old_state, new_owner, material_source_index,
                             transport_map, system, radius_km):
    """Report every lost oceanic source exactly once from the accepted remap.

    Event polarity follows the donor actually removed and receiver actually
    retained, so its law is shared with material transport instead of another
    buoyancy/plate-ID decision in dynamics.  Physical contacts are matched by
    location and owner pair, independently of their display classification.
    """
    if old_state.oceanic_volume_km3 is None:
        raise ValueError("Young slab inventory requires tracked oceanic volume")
    area = mesh.physical_cell_areas_km2(radius_km)
    from .lithosphere import continental_material_fields
    fraction, _ = continental_material_fields(old_state, area)
    volumes = np.asarray(old_state.oceanic_volume_km3)
    survived = np.zeros(mesh.cell_count,dtype=bool)
    inherited = np.asarray(material_source_index)
    survived[inherited[inherited >= 0]] = True
    target_by_source = np.full(mesh.cell_count,-1,dtype=int)
    for pid, targets in enumerate(transport_map.source_to_target):
        sources = np.flatnonzero(old_state.cell_plate == pid)
        target_by_source[sources] = targets
    boundaries = classify_boundaries(mesh,system,radius_km,0.,0.)
    by_pair = {}
    for b in boundaries:
        by_pair.setdefault(frozenset((b.plate_a,b.plate_b)),[]).append(b)
    events = []
    for source in np.flatnonzero((~survived)&(volumes > 0.)):
        target = int(target_by_source[source])
        if target < 0:
            raise ValueError("Accepted ocean loss has no transport destination")
        sub, over = int(old_state.cell_plate[source]),int(new_owner[target])
        if sub == over:
            raise ValueError("Young slab acceptance cannot consume a same-plate parcel")
        candidates = by_pair.get(frozenset((sub,over)),[])
        if candidates:
            # Match the resolved collision to one local edge, not the whole pair.
            b = max(candidates,key=lambda item: float(mesh.centroids[target]@item.midpoint))
            length,r,torque = _geometry(mesh,b,radius_km)
            if b.plate_a != sub:
                torque = -torque
            edge_key = contact_key(b)
        else:
            # A parcel can disappear in an underresolved new contact. Retain its
            # accepted volume, but do not invent an attached trench or force.
            edge_key = f"unresolved:{target}:{source}"
            length,r,torque = 0.,mesh.centroids[target],np.zeros(3)
        accepted_area = float(area[source]*max(1.-fraction[source],0.))
        h = 0. if old_state.mantle_lithosphere_thickness_km is None else max(
            float(old_state.mantle_lithosphere_thickness_km[source]),0.)
        rho = 0. if old_state.mantle_lithosphere_density_anomaly_kg_m3 is None else max(
            float(old_state.mantle_lithosphere_density_anomaly_kg_m3[source]),0.)
        if accepted_area <= 0:
            raise ValueError("Oceanic volume has no physical source footprint")
        cold_volume = accepted_area*h
        events.append(SlabAcceptance(int(source),target,sub,over,accepted_area,
            float(volumes[source]),cold_volume,cold_volume*1e9*rho,edge_key,
            length,r.tolist(),torque.tolist(),bool(candidates)))
    expected = float(np.sum(volumes[~survived]))
    if not math.isclose(sum(e.oceanic_volume_km3 for e in events),expected,rel_tol=5e-13,abs_tol=1e-7):
        raise ValueError("Accepted slab events do not close the surface ocean-volume loss")
    return events


def young_ocean_overlap_order(mesh, state, source_faces, target_face):
    """Retain least negatively buoyant ocean, using material history for ties.

    Actual density/age asymmetry is preferred. Equal material keeps the local
    preexisting owner when possible, then the nearest source material. A final
    lexicographic material-position tie is a deterministic raster convention,
    not a claim to resolve perfectly symmetric physical initiation. Plate IDs
    are only compared for equality with the preexisting owner, never ordered.
    """
    source_faces = np.asarray(source_faces,dtype=int)
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        buoyancy = np.zeros(len(source_faces))
    else:
        buoyancy = np.maximum(state.mantle_lithosphere_thickness_km[source_faces],0.)*np.maximum(
            state.mantle_lithosphere_density_anomaly_kg_m3[source_faces],0.)
    x = mesh.centroids[source_faces]
    prior = state.cell_plate[source_faces] != state.cell_plate[target_face]
    distance = 1.-x@mesh.centroids[target_face]
    return np.lexsort((x[:,2],x[:,1],x[:,0],distance,prior,
                       state.crust_age_myr[source_faces],buoyancy))


def accept_slab_material(mesh, inventory: YoungBoundaryState,
                         events: Iterable[SlabAcceptance], time_myr: float):
    """Consume an already-conservative material transaction once, without sinks."""
    time_myr = float(time_myr)
    events = tuple(events)
    if not math.isfinite(time_myr) or time_myr <= inventory.last_material_time_myr:
        raise ValueError("Young slab material transaction time must increase; duplicate rejected")
    # Validate the complete transaction before mutating its persistent ledger.
    sources = set()
    for e in events:
        if e.source_face in sources:
            raise ValueError("Duplicate slab source in one material transaction")
        sources.add(e.source_face)
        values = (e.accepted_area_km2,e.oceanic_volume_km3,e.cold_mantle_volume_km3,
                  e.density_excess_mass_kg,e.trench_length_km)
        if any(not math.isfinite(x) or x < 0 for x in values) or (e.attached and e.trench_length_km <= 0):
            raise ValueError("Slab acceptance quantities must be finite and nonnegative")
        if e.subducting_plate == e.overriding_plate or e.accepted_area_km2 <= 0 or e.oceanic_volume_km3 <= 0:
            raise ValueError("Slab acceptance requires actual inter-plate ocean loss")
        if not np.isfinite(e.midpoint).all() or not np.isfinite(e.torque_direction).all():
            raise ValueError("Invalid slab geometry")
    for e in events:
        if inventory.connectivity_model != "legacy_fixed_contacts" and e.attached:
            contact = inventory.contacts.get(e.contact_key)
            local = (contact is not None and contact.present and
                {contact.plate_a, contact.plate_b} ==
                {e.subducting_plate, e.overriding_plate} and
                bool({contact.face_a, contact.face_b} & {e.source_face, e.target_face}))
            if not local:
                # The raster's nearest same-pair edge may be remote from an
                # underresolved collision. Keep its material, but no invented
                # mechanical connection to that remote trench.
                e = replace(e, contact_key=f"unresolved:locality:{e.target_face}:{e.source_face}",
                    attached=False, trench_length_km=0.,
                    midpoint=mesh.centroids[e.target_face].tolist(),
                    torque_direction=[0., 0., 0.])
        key = f"{e.contact_key}:{e.subducting_plate}:{e.overriding_plate}"
        segment = inventory.segments.get(key)
        if inventory.connectivity_model != "legacy_fixed_contacts" and segment is not None and (
                not segment.attached or segment.contact_key != e.contact_key or
                (segment.subducting_plate, segment.overriding_plate) !=
                (e.subducting_plate, e.overriding_plate)):
            # Old segment keys survive edge transfer and relabeling. Never
            # reconnect their detached historical cohorts through a new event.
            key = f"{key}:accepted:{time_myr.hex()}:{e.source_face}"
            segment = inventory.segments.get(key)
        if segment is None:
            segment = YoungSlabSegment(key,e.contact_key,e.subducting_plate,e.overriding_plate,
                mesh.centroids[e.source_face].tolist(),mesh.centroids[e.target_face].tolist(),
                list(e.midpoint),e.trench_length_km,first_acceptance_time_myr=time_myr,
                attached=e.attached)
            inventory.segments[key] = segment
        segment.accepted_area_km2 += e.accepted_area_km2
        segment.oceanic_volume_km3 += e.oceanic_volume_km3
        segment.cold_mantle_volume_km3 += e.cold_mantle_volume_km3
        segment.density_excess_mass_kg += e.density_excess_mass_kg
        segment.buoyancy_moment_kg = (np.asarray(segment.buoyancy_moment_kg)
            +e.density_excess_mass_kg*np.asarray(e.torque_direction)).tolist()
        segment.thermal_cohorts.append(SlabThermalCohort(time_myr,e.accepted_area_km2,
            e.oceanic_volume_km3,e.cold_mantle_volume_km3,e.density_excess_mass_kg,
            (e.density_excess_mass_kg*np.asarray(e.torque_direction)).tolist(),
            e.cold_mantle_volume_km3/e.accepted_area_km2))
        segment.last_acceptance_time_myr = time_myr
        inventory.cumulative_accepted_area_km2 += e.accepted_area_km2
        inventory.cumulative_accepted_oceanic_volume_km3 += e.oceanic_volume_km3
    inventory.last_material_time_myr = time_myr


def accepted_slab_buoyancy_torque(inventory: YoungBoundaryState | None,
                                 plate_count, radius_km, gravity_m_s2,*,time_myr=None):
    """Return R*g*ΣΔρV(r×towards) in N m; no numerical length normalization.

    This is the maximum fully transmitted tangential slab-buoyancy torque
    closure.  It is not a resolved bending/slab-viscous-stress calculation.
    Directional cancellation is retained, and zero accepted cold buoyancy gives
    exactly zero.  The caller owns the explicit traction-transmission model.
    """
    torque = np.zeros((int(plate_count),3),dtype=float)
    if inventory is None:
        return torque
    if time_myr is None:
        time_myr=inventory.last_material_time_myr
    for segment in inventory.segments.values():
        if segment.attached and 0 <= segment.subducting_plate < plate_count:
            for cohort in segment.thermal_cohorts:
                _,weight=_cohort_current(cohort,inventory,time_myr)
                torque[segment.subducting_plate] += weight*np.asarray(cohort.initial_buoyancy_moment_kg)
    return torque*(float(radius_km)*1000.*float(gravity_m_s2))


def accepted_slab_torques_nm(memory, plate_count, radius_km, gravity_m_s2):
    """Public dynamics adapter; result is a (plate_count,3) N m array."""
    inventory = None if memory is None else memory.young_boundary_state
    result = np.zeros((int(plate_count),3),dtype=float)
    if inventory is None:
        return result
    # Breakoff is applied between memory advance and force evaluation. Honour
    # it immediately without mutating memory from this read-only force query.
    for segment in inventory.segments.values():
        zone = memory.zones.get((segment.subducting_plate,segment.overriding_plate))
        if (segment.attached and 0 <= segment.subducting_plate < plate_count
                and not (zone is not None and zone.broken_off)):
            for cohort in segment.thermal_cohorts:
                _,weight=_cohort_current(cohort,inventory,memory.time_myr)
                result[segment.subducting_plate] += weight*np.asarray(cohort.initial_buoyancy_moment_kg)
    return result*(float(radius_km)*1000.*float(gravity_m_s2))


def accepted_slab_force_sections(memory, *, params=None,
                                 buoyancy_model="uniform_thermal_mass_v1"):
    """Return present, connected physical slabs for a dissipative force closure.

    This read-only query deliberately does not use the historical upper-bound
    torque API: current contact orientation defines the force application, and
    absent or mismatched contacts cannot transmit it. Grouping by trench and
    ordered pair prevents duplicated bending/drag terms after refinement,
    checkpoint subdivision, or multiple accepted thermal cohorts. Ordered
    buoyancy is explicit; the default retains the original uniform force input.
    """
    if buoyancy_model not in {"uniform_thermal_mass_v1", "ordered_thermal_cohorts_v1"}:
        raise ValueError("Unknown young slab buoyancy geometry model")
    inventory = None if memory is None else memory.young_boundary_state
    if inventory is None:
        return ()
    groups = {}
    for segment in inventory.segments.values():
        if not segment.attached:
            continue
        contact = _matching_contact(inventory, segment)
        zone = memory.zones.get((segment.subducting_plate, segment.overriding_plate))
        if contact is None or (zone is not None and zone.broken_off):
            continue
        key = (contact.key, segment.subducting_plate, segment.overriding_plate)
        groups.setdefault(key, []).extend(segment.thermal_cohorts)
    sections = []
    for (key, sub, over), cohorts in sorted(groups.items()):
        contact = inventory.contacts[key]
        width = float(contact.trench_length_km)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Connected slab force requires a positive trench width")
        weights = [(cohort, *_cohort_current(cohort, inventory, memory.time_myr))
                   for cohort in cohorts]
        area = sum(cohort.accepted_area_km2*retained for cohort, retained, _ in weights)
        if area <= 0:
            continue
        zone = memory.zones.get((sub, over))
        if params is not None:
            dip = float(_contact_dip(cohorts, memory.time_myr, params))
        elif zone is not None:
            dip = float(zone.dip_deg)
        else:
            raise ValueError("Connected slab force requires explicit dip parameters or a synchronized zone")
        if not math.isfinite(dip) or not 0 < dip <= 90:
            raise ValueError("Connected slab dip must lie in (0, 90] degrees")
        length = area/width
        depth = length*math.sin(math.radians(dip))
        if inventory.mantle_depth_km is not None and depth > inventory.mantle_depth_km*(1.+1e-12):
            raise ValueError("Slab force inventory must be depth-limited before force evaluation")
        direction = np.asarray(contact.torque_direction_ab, dtype=float)
        source, receiver = contact.face_a, contact.face_b
        if contact.plate_a != sub:
            direction = -direction
            source, receiver = receiver, source
        layers = ()
        if buoyancy_model == "ordered_thermal_cohorts_v1":
            batches = {}
            for cohort, retained, cold in weights:
                if retained > 0 and cohort.accepted_area_km2 > 0:
                    batches.setdefault(cohort.acceptance_time_myr, []).append((cohort, retained, cold))
            ordered = []
            batch_areas = []
            for accepted_time, batch in sorted(batches.items(), reverse=True):
                batch_area = math.fsum(c.accepted_area_km2*r for c, r, _ in batch)
                start = math.fsum(batch_areas)/width
                batch_areas.append(batch_area)
                end = math.fsum(batch_areas)/width
                ordered.append(SlabBuoyancyLayer(float(accepted_time),
                    max(0., float(memory.time_myr)-accepted_time), start, end, batch_area,
                    math.fsum(c.cold_mantle_volume_km3*r for c, r, _ in batch),
                    math.fsum(c.initial_density_excess_mass_kg*cold for c, _, cold in batch),
                    math.fsum(c.accepted_area_km2*r*c.initial_thickness_km**3 for c, r, _ in batch)))
            layers = tuple(ordered)
        sections.append(SlabForceSection(key, sub, over, source, receiver,
            width, area, length, depth, dip,
            sum(cohort.oceanic_volume_km3*retained for cohort, retained, _ in weights),
            sum(cohort.cold_mantle_volume_km3*retained for cohort, retained, _ in weights),
            sum(cohort.accepted_area_km2*retained*cohort.initial_thickness_km**3
                for cohort, retained, _ in weights),
            sum(cohort.initial_density_excess_mass_kg*cold for cohort, _, cold in weights),
            tuple(contact.midpoint), tuple(direction), layers))
    return tuple(sections)


def accepted_slab_inventory_diagnostics(memory):
    """Separate cumulative accepted volume, current attached material and coldness."""
    result=dict(cumulative_accepted_area_km2=0.,cumulative_accepted_oceanic_volume_km3=0.,
        attached_area_km2=0.,attached_oceanic_volume_km3=0.,deep_oceanic_volume_km3=0.,
        unresolved_or_detached_oceanic_volume_km3=0.,attached_mantle_volume_km3=0.,
        current_negative_buoyancy_mass_kg=0.,initial_negative_buoyancy_mass_kg=0.,
        segment_count=0,thermal_cohort_count=0,
        mechanical_detachment_count=0,
        unresolved_or_detached_fraction_of_accepted_volume=0.)
    inventory=None if memory is None else memory.young_boundary_state
    if inventory is None:
        return result
    result['cumulative_accepted_area_km2']=inventory.cumulative_accepted_area_km2
    result['cumulative_accepted_oceanic_volume_km3']=inventory.cumulative_accepted_oceanic_volume_km3
    result['segment_count']=len(inventory.segments)
    result['mechanical_detachment_count']=len(inventory.mechanical_detachments)
    for segment in inventory.segments.values():
        for c in segment.thermal_cohorts:
            result['thermal_cohort_count']+=1
            retained,cold=_cohort_current(c,inventory,memory.time_myr)
            result['initial_negative_buoyancy_mass_kg']+=c.initial_density_excess_mass_kg
            result['deep_oceanic_volume_km3']+=c.oceanic_volume_km3*c.deep_transfer_fraction
            if segment.attached:
                result['attached_area_km2']+=c.accepted_area_km2*retained
                result['attached_oceanic_volume_km3']+=c.oceanic_volume_km3*retained
                result['attached_mantle_volume_km3']+=c.cold_mantle_volume_km3*retained
                result['current_negative_buoyancy_mass_kg']+=c.initial_density_excess_mass_kg*cold
            else:
                result['unresolved_or_detached_oceanic_volume_km3']+=c.oceanic_volume_km3*retained
    total=result['cumulative_accepted_oceanic_volume_km3']
    if total > 0:
        result['unresolved_or_detached_fraction_of_accepted_volume']=(
            result['unresolved_or_detached_oceanic_volume_km3']/total)
    return result


def commit_slab_neck_failures(memory, failures, time_myr):
    """Commit a successful pure force solve's tensile failures exactly once.

    Material remains in the accepted ledger. This changes only mechanical
    attachment and adds a persistent event; future calls or a resumed run
    cannot detach the same cohorts again. The caller then refreshes zone views.
    """
    inventory = None if memory is None else memory.young_boundary_state
    if inventory is None:
        if failures:
            raise ValueError("Cannot commit slab neck failure without accepted inventory")
        return
    if not math.isfinite(time_myr):
        raise ValueError("Slab neck failure time must be finite")
    transactions = []
    seen = set()
    for failure in failures:
        key = (failure['contact_key'], int(failure['subducting_plate']), int(failure['overriding_plate']))
        if key in seen:
            continue
        seen.add(key)
        for name in ('tension_n', 'capacity_n'):
            if not math.isfinite(float(failure[name])) or float(failure[name]) < 0:
                raise ValueError("Slab neck failure forces must be finite and nonnegative")
        segments = [s for s in inventory.segments.values() if s.attached and
            (s.contact_key, s.subducting_plate, s.overriding_plate) == key]
        if not segments:
            continue
        retained_volume = sum(c.oceanic_volume_km3*(1.-c.deep_transfer_fraction)
            for s in segments for c in s.thermal_cohorts)
        row = {**deepcopy(failure), 'time_myr': float(time_myr),
            'reason': 'tensile_neck_failure',
            'retained_oceanic_volume_km3': float(retained_volume)}
        transactions.append((segments, row))
    for segments, row in transactions:
        for segment in segments:
            segment.attached = False
        inventory.mechanical_detachments.append(row)


def boundary_state_to_json(inventory):
    if inventory is None:
        return None
    result = dict(version=inventory.version,
        contacts=[asdict(value) for _,value in sorted(inventory.contacts.items())],
        segments=[asdict(value) for _,value in sorted(inventory.segments.items())],
        cumulative_accepted_area_km2=inventory.cumulative_accepted_area_km2,
        cumulative_accepted_oceanic_volume_km3=inventory.cumulative_accepted_oceanic_volume_km3,
        thermal_diffusivity_m2_s=inventory.thermal_diffusivity_m2_s,
        mantle_depth_km=inventory.mantle_depth_km,
        last_material_time_myr=(inventory.last_material_time_myr if math.isfinite(inventory.last_material_time_myr) else None))
    if inventory.connectivity_model != "legacy_fixed_contacts":
        result['connectivity_model'] = inventory.connectivity_model
    if inventory.connectivity_model != "legacy_fixed_contacts" or inventory.mechanical_detachments:
        result['mechanical_detachments'] = deepcopy(inventory.mechanical_detachments)
    if inventory.buoyancy_geometry_model != "uniform_thermal_mass_v1":
        result['buoyancy_geometry_model'] = inventory.buoyancy_geometry_model
    return result


def boundary_state_from_json(data):
    if data is None:
        return None
    if data.get("version") != MODEL_VERSION:
        raise ValueError("Unsupported young boundary material model version")
    result = YoungBoundaryState()
    result.contacts = {row['key']:YoungContact(**row) for row in data.get('contacts',[])}
    for row in data.get('segments',[]):
        row=dict(row)
        row['thermal_cohorts']=[SlabThermalCohort(**c) for c in row.get('thermal_cohorts',[])]
        result.segments[row['key']]=YoungSlabSegment(**row)
    result.cumulative_accepted_area_km2 = float(data.get('cumulative_accepted_area_km2',0.))
    result.cumulative_accepted_oceanic_volume_km3 = float(data.get('cumulative_accepted_oceanic_volume_km3',0.))
    result.last_material_time_myr = -math.inf if data.get('last_material_time_myr') is None else float(data['last_material_time_myr'])
    result.thermal_diffusivity_m2_s=float(data['thermal_diffusivity_m2_s'])
    result.mantle_depth_km=None if data.get('mantle_depth_km') is None else float(data['mantle_depth_km'])
    result.connectivity_model = data.get('connectivity_model', 'legacy_fixed_contacts')
    if result.connectivity_model not in {'legacy_fixed_contacts', 'local_edge_transfer_v1'}:
        raise ValueError("Unknown young slab connectivity model")
    result.mechanical_detachments = deepcopy(data.get('mechanical_detachments', []))
    result.buoyancy_geometry_model = data.get('buoyancy_geometry_model', 'uniform_thermal_mass_v1')
    if result.buoyancy_geometry_model not in {'uniform_thermal_mass_v1', 'ordered_thermal_cohorts_v1'}:
        raise ValueError("Unknown young slab buoyancy geometry model")
    return result


def remap_young_inventory(mesh, old_owner, new_owner, inventory):
    """Relabel attached parcels by their local material anchors, conserving mass."""
    old_owner,new_owner = np.asarray(old_owner),np.asarray(new_owner)
    def mapped(pid,anchor):
        cells = np.flatnonzero(old_owner == pid)
        if not len(cells):
            return -1
        face = int(cells[np.argmax(mesh.centroids[cells]@np.asarray(anchor))])
        return int(new_owner[face])
    for segment in inventory.segments.values():
        sub = mapped(segment.subducting_plate,segment.source_anchor)
        over = mapped(segment.overriding_plate,segment.receiver_anchor)
        if sub < 0 or over < 0 or sub == over:
            segment.attached = False
        segment.subducting_plate,segment.overriding_plate = sub,over
    for contact in inventory.contacts.values():
        contact.plate_a,contact.plate_b = int(new_owner[contact.face_a]),int(new_owner[contact.face_b])
        if contact.plate_a == contact.plate_b:
            contact.present = False
    _transport_attachment_geometry(mesh, inventory)


def remesh_young_inventory(old_mesh,new_mesh,old_owner,new_owner,ancestors,inventory,radius_km):
    """Refine candidate edges and accepted cohorts without changing their torque.

    Extensive quantities are partitioned by descendant edge lengths. Stored
    buoyancy moments are divided directly, never reoriented/renormalized by
    the new grid. Thus refinement cannot manufacture or lose accepted mass,
    coldness, or vector torque. Detached/historical contacts retain an anchor.
    """
    if inventory is None:
        return None
    result=deepcopy(inventory)
    result.contacts={};result.segments={}
    ancestors=np.asarray(ancestors,dtype=int)
    descendants={}
    for fa,fb,vu,vv in new_mesh.shared_edges:
        if new_owner[fa] == new_owner[fb] or ancestors[fa] == ancestors[fb]:
            continue
        r=new_mesh.vertices[vu]+new_mesh.vertices[vv]
        r=r/np.linalg.norm(r)
        b=BoundaryRecord(int(fa),int(fb),int(vu),int(vv),int(new_owner[fa]),int(new_owner[fb]),r,0.,0.,0.,BoundaryType.INACTIVE)
        descendants.setdefault(frozenset((int(ancestors[fa]),int(ancestors[fb]))),[]).append(b)
    contact_mapping={}
    for old_key,old in inventory.contacts.items():
        children=descendants.get(frozenset((old.face_a,old.face_b)),[])
        if not children:
            child=deepcopy(old)
            child.key='retired:'+old.key
            child.face_a=int(np.flatnonzero(ancestors==old.face_a)[0])
            child.face_b=int(np.flatnonzero(ancestors==old.face_b)[0])
            child.present=False
            result.contacts[child.key]=child
            contact_mapping[old_key]=[(child.key,1.,child)]
            continue
        lengths=np.asarray([_geometry(new_mesh,b,radius_km)[0] for b in children])
        weights=lengths/lengths.sum()
        weights[-1]=1.-float(weights[:-1].sum())
        contact_mapping[old_key]=[]
        for b,w in zip(children,weights):
            length,r,torque=_geometry(new_mesh,b,radius_km)
            key=contact_key(b)
            child=YoungContact(key,b.face_a,b.face_b,b.plate_a,b.plate_b,length,
                r.tolist(),torque.tolist(),old.normal_rate_km_per_myr,
                old.cumulative_closure_area_km2*float(w),
                old.cumulative_opening_area_km2*float(w),old.present)
            result.contacts[key]=child
            contact_mapping[old_key].append((key,float(w),child))
    for old_key,old in inventory.segments.items():
        children=contact_mapping.get(old.contact_key)
        if children is None:
            # Unresolved accepted loss has no force/edge to subdivide.
            result.segments[old_key]=deepcopy(old)
            continue
        for key,w,contact in children:
            child=deepcopy(old)
            child.key=f"refined:{old_key}:{key}"
            child.contact_key=key
            child.midpoint=list(contact.midpoint)
            for name in ('trench_length_km','accepted_area_km2','oceanic_volume_km3',
                         'cold_mantle_volume_km3','density_excess_mass_kg'):
                setattr(child,name,getattr(old,name)*w)
            child.buoyancy_moment_kg=(np.asarray(old.buoyancy_moment_kg)*w).tolist()
            for cohort in child.thermal_cohorts:
                for name in ('accepted_area_km2','oceanic_volume_km3','cold_mantle_volume_km3',
                             'initial_density_excess_mass_kg'):
                    setattr(cohort,name,getattr(cohort,name)*w)
                cohort.initial_buoyancy_moment_kg=(np.asarray(cohort.initial_buoyancy_moment_kg)*w).tolist()
            if inventory.connectivity_model != "legacy_fixed_contacts":
                child.attached = child.attached and contact.present and {
                    contact.plate_a, contact.plate_b} == {
                    child.subducting_plate, child.overriding_plate}
                child.trench_length_km = contact.trench_length_km
                source, receiver = contact.face_a, contact.face_b
                if contact.plate_a != child.subducting_plate:
                    source, receiver = receiver, source
                child.source_anchor = new_mesh.centroids[source].tolist()
                child.receiver_anchor = new_mesh.centroids[receiver].tolist()
            result.segments[child.key]=child
    return result


def update_young_zone_moments(memory):
    """Rebuild aggregate axes after topology without restoring unit magnitude."""
    inventory = memory.young_boundary_state
    for key,zone in memory.zones.items():
        segments = [s for s in inventory.segments.values() if s.attached and
                    (s.subducting_plate,s.overriding_plate) == key]
        weighted=[(cohort,_cohort_current(cohort,inventory,memory.time_myr)[1])
                  for s in segments for cohort in s.thermal_cohorts]
        total=sum(c.initial_density_excess_mass_kg*w for c,w in weighted)
        zone.torque_axis=sum((w*np.asarray(c.initial_buoyancy_moment_kg) for c,w in weighted),np.zeros(3))/max(total,1e-30)


def synchronize_young_zones(memory, state, params):
    """Compatibility view for arc/breakoff diagnostics; inventory stays primary."""
    from .subduction_memory import SlabZone
    inventory = memory.young_boundary_state
    limit_attached_inventory(inventory,params.initial_dip_deg,time_myr=state.time_myr,params=params)
    groups = {}
    for segment in inventory.segments.values():
        key = (segment.subducting_plate,segment.overriding_plate)
        previous = memory.zones.get(key)
        if previous is not None and previous.broken_off:
            segment.attached = False
        if segment.attached:
            groups.setdefault(key,[]).append(segment)
    if inventory.connectivity_model != "legacy_fixed_contacts":
        for key, zone in memory.zones.items():
            if key not in groups:
                zone.active = False
                zone.slab_length_km = zone.slab_depth_km = zone.trench_length_km = 0.
                zone.convergence_rate_km_per_myr = zone.buoyancy_factor = 0.
                zone.torque_axis = np.zeros(3)
    for key, segments in sorted(groups.items()):
        cumulative_area = sum(s.accepted_area_km2 for s in segments)
        cohort_weights=[(c,*_cohort_current(c,inventory,state.time_myr))
                        for s in segments for c in s.thermal_cohorts]
        area=sum(c.accepted_area_km2*w for c,w,_ in cohort_weights)
        edge_lengths = {}
        for segment in segments:
            edge_lengths[segment.contact_key] = max(edge_lengths.get(segment.contact_key,0.),segment.trench_length_km)
        length = sum(edge_lengths.values())
        zone = memory.zones.get(key)
        if zone is None:
            zone = SlabZone(*key)
            memory.zones[key] = zone
            memory.births += 1
        present = [c for c in inventory.contacts.values() if c.present and {c.plate_a,c.plate_b} == set(key)]
        zone.active = bool(present)
        zone.active_age_myr = max(0.,state.time_myr-min(s.first_acceptance_time_myr for s in segments))
        zone.dip_deg = params.initial_dip_deg+(params.mature_dip_deg-params.initial_dip_deg)*(1.-math.exp(-zone.active_age_myr/max(params.dip_maturation_myr,1e-9)))
        zone.slab_length_km = area/max(length,1e-30)
        zone.slab_depth_km = zone.slab_length_km*math.sin(math.radians(zone.dip_deg))
        if inventory.mantle_depth_km is not None:
            zone.slab_depth_km = min(inventory.mantle_depth_km,zone.slab_depth_km)
        zone.trench_length_km = length
        zone.cumulative_subducted_area_km2 = cumulative_area
        zone.trench_midpoint = sum((s.accepted_area_km2*np.asarray(s.midpoint) for s in segments),np.zeros(3))/max(cumulative_area,1e-30)
        norm = np.linalg.norm(zone.trench_midpoint)
        if norm > 0:
            zone.trench_midpoint /= norm
        excess = sum(c.initial_density_excess_mass_kg*w for c,_,w in cohort_weights)
        zone.torque_axis = sum((w*np.asarray(c.initial_buoyancy_moment_kg) for c,_,w in cohort_weights),np.zeros(3))/max(excess,1e-30)
        cold = sum(c.cold_mantle_volume_km3*w for c,w,_ in cohort_weights)
        zone.buoyancy_factor = max(excess/1e9/max(area,1e-30)/6000.,0.)**.75 if cold else 0.
        contact_length = sum(c.trench_length_km for c in present)
        zone.convergence_rate_km_per_myr = sum(c.trench_length_km*max(-c.normal_rate_km_per_myr,0.) for c in present)/max(contact_length,1e-30)
    memory.cumulative_subducted_area_km2 = inventory.cumulative_accepted_area_km2
    memory.time_myr = float(state.time_myr)


__all__ = ['MODEL_VERSION','SlabAcceptance','YoungBoundaryState','YoungSlabSegment','SlabThermalCohort',
    'advance_contact_geometry','raster_acceptance_events','young_ocean_overlap_order','accept_slab_material',
    'accepted_slab_buoyancy_torque','accepted_slab_torques_nm','boundary_state_to_json','boundary_state_from_json',
    'synchronize_young_zones','remap_young_inventory','remesh_young_inventory','update_young_zone_moments',
    'slab_thermal_deficit_fraction','limit_attached_inventory','accepted_slab_inventory_diagnostics',
    'SlabBuoyancyLayer','SlabForceSection','accepted_slab_force_sections','commit_slab_neck_failures']

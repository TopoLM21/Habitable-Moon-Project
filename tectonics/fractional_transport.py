"""Experimental conservative fractional ocean transport, independent of raster commits.

Every component follows its own rigid plate velocity. Exact great-circle edge
fluxes feed a positive first-order upwind finite-volume update with CFL
substeps. Ocean overlap is removed once and geometric vacancies receive new
ridge material. The surface component state, not a visible winner, is primary.

This is a transport foundation, not an activated mature force/topology solver.
Upwind diffusion broadens interfaces; exact material histories can increase the
number of components. No small component is silently discarded or averaged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .fractional_surface import (FractionalSurfaceState, IncomingPiece,
    SurfaceParcel, TransferPiece, commit_surface, remap_surface)


@dataclass(frozen=True, slots=True)
class FractionalLoss:
    parcel: SurfaceParcel
    source_cell: int
    source_fraction: float
    time_myr: float
    receiver_plate_fractions: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class FractionalTransportResult:
    state: FractionalSurfaceState
    losses: tuple[FractionalLoss, ...]
    births: tuple[SurfaceParcel, ...]
    diagnostics: dict


_EXTENSIVE = ('area_km2', 'oceanic_volume_km3', 'cold_mantle_volume_km3', 'density_excess_mass_kg')
_EPS = np.finfo(float).eps


def rigid_edge_area_fluxes(mesh, omega_rad_per_myr, radius_km):
    """Return signed (plate, shared-edge) area flux, km²/Myr, face_a to face_b.

    For unit edge endpoints u,v and n=sign*(u×v)/|u×v| pointing a→b,
    integral (omega×r)·n d(edge angle) = sign*omega·(u-v).
    Multiplying by R² gives physical area flux. Endpoint differences telescope
    around each triangle, preserving zero divergence of rigid rotation.
    """
    omega = np.asarray(omega_rad_per_myr, dtype=float)
    radius = float(radius_km)
    if (omega.ndim != 2 or omega.shape[1] != 3 or not len(omega)
            or not np.isfinite(omega).all() or not math.isfinite(radius) or radius <= 0.):
        raise ValueError('Fractional transport requires finite plate angular velocities and positive radius')
    edges = np.asarray(mesh.shared_edges, dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 4:
        raise ValueError('Fractional transport requires a closed triangular edge mesh')
    u, v = mesh.vertices[edges[:, 2]], mesh.vertices[edges[:, 3]]
    cross = np.cross(u, v)
    direction = mesh.centroids[edges[:, 1]]-mesh.centroids[edges[:, 0]]
    orientation = np.einsum('ij,ij->i', cross, direction)
    if np.any(orientation == 0.):
        raise ValueError('Degenerate edge has no outward transport orientation')
    vectors = np.sign(orientation)[:, None]*(u-v)
    return (omega@vectors.T)*(radius*radius)


def _split_piece(piece, fraction):
    parcel = replace(piece.parcel, **{name: getattr(piece.parcel, name)*fraction for name in _EXTENSIVE})
    return IncomingPiece(piece.source_index, piece.fraction*fraction, parcel)


def _totals(parcels):
    parcels = tuple(parcels)
    return {name: math.fsum(getattr(p, name) for p in parcels) for name in _EXTENSIVE}


def _occupancy(parcels):
    grouped = {}
    for p in parcels:
        grouped.setdefault((p.cell, p.plate), []).append(p.area_km2)
    return {key: math.fsum(values) for key, values in grouped.items()}


def _same_priority(a, b):
    # Only collapse arithmetic noise introduced by proportional donor splits;
    # this is not a tunable physical buoyancy/age threshold.
    return all(math.isclose(x, y, rel_tol=64.*_EPS, abs_tol=0.) for x, y in zip(a, b))


def _resolve(before, incoming, end_time, birth_factory, serial, *,
             event_time_quadrature=None, birth_times=None):
    by_cell = [[] for _ in before.cell_areas_km2]
    for piece in incoming:
        by_cell[piece.parcel.cell].append(piece)
    occupancy_before = [{} for _ in before.cell_areas_km2]
    for (cell, plate), area in _occupancy(before.parcels).items():
        occupancy_before[cell][plate] = area
    retained, lost, born, records = [], [], [], []
    roundoff = 0.
    for cell, pieces in enumerate(by_cell):
        capacity = before.cell_areas_km2[cell]
        arrived = math.fsum(p.parcel.area_km2 for p in pieces)
        tolerance = 128.*_EPS*max(capacity, arrived)
        excess = arrived-capacity
        local_retained, local_lost = [], []
        if excess > tolerance:
            def priority(piece):
                p = piece.parcel
                return (p.specific_properties[2], p.age_myr)
            ordered = sorted(pieces, key=priority, reverse=True)
            start = 0
            while start < len(ordered):
                if excess <= tolerance:
                    # Do not turn a subtraction-roundoff remainder into a new
                    # physical sink (possibly even against the only survivor).
                    local_retained.extend(ordered[start:])
                    roundoff = max(roundoff, abs(excess))
                    break
                stop = start+1
                score = priority(ordered[start])
                while stop < len(ordered) and _same_priority(score, priority(ordered[stop])):
                    stop += 1
                batch = ordered[start:stop]
                batch_area = math.fsum(p.parcel.area_km2 for p in batch)
                rejected = min(max(excess, 0.), batch_area)
                fraction = rejected/batch_area
                for piece in batch:
                    if fraction > 0.:
                        local_lost.append(_split_piece(piece, fraction))
                    if fraction < 1.:
                        local_retained.append(_split_piece(piece, 1.-fraction))
                excess -= rejected
                start = stop
        else:
            local_retained = pieces
            if abs(excess) <= tolerance:
                roundoff = max(roundoff, abs(excess))
        retained.extend(local_retained)
        lost.extend(local_lost)
        surviving = _occupancy(p.parcel for p in local_retained)
        for piece in local_lost:
            receivers = {plate: area for (target, plate), area in surviving.items()
                         if plate != piece.parcel.plate and area > 0.}
            total = math.fsum(receivers.values())
            if total <= 0.:
                raise ValueError('Rejected fractional ocean has no retained overriding plate')
            records.append(FractionalLoss(piece.parcel,
                before.parcels[piece.source_index].cell, piece.fraction, end_time,
                tuple((plate, area/total) for plate, area in sorted(receivers.items()))))
        deficit = capacity-arrived
        if deficit > tolerance:
            # The departing owners create this local vacancy. Dividing birth
            # between their occupancy losses is a symmetric first-order ridge
            # ownership closure, including cells with no surviving old parcel.
            arriving = _occupancy(p.parcel for p in pieces)
            departures = {plate: max(area-arriving.get((cell, plate), 0.), 0.)
                          for plate, area in occupancy_before[cell].items()}
            departures = {plate: area for plate, area in departures.items() if area > 0.}
            total = math.fsum(departures.values())
            if total <= 0.:
                raise ValueError('Fractional vacancy has no departing plate provenance')
            for plate, area in sorted(departures.items()):
                amount = deficit*area/total
                events = ((end_time, 1.),) if event_time_quadrature is None else tuple(
                    (before.time_myr+(end_time-before.time_myr)*node, weight)
                    for node, weight in event_time_quadrature)
                for event_time, weight in events:
                    cohort_area = amount*weight
                    parcel = birth_factory(cell, plate, cohort_area, event_time, serial)
                    serial += 1
                    if (parcel.cell != cell or parcel.plate != plate
                            or not math.isclose(parcel.area_km2, cohort_area, rel_tol=16.*_EPS)
                            or parcel.age_myr != 0. or parcel.cold_mantle_volume_km3 != 0.
                            or parcel.density_excess_mass_kg != 0.):
                        raise ValueError('Ridge birth must match its vacancy and have zero age/cold mantle')
                    if birth_times is not None:
                        if parcel.material_id in birth_times:
                            raise ValueError('Time cohorts require distinct newborn material IDs')
                        birth_times[parcel.material_id] = event_time
                    born.append(parcel)
    return retained, lost, born, records, roundoff, serial


def advance_fractional_transport(mesh, state, omega_rad_per_myr, radius_km, dt_myr, *, birth_factory,
                                event_time_quadrature=None):
    """Advance an ocean-only component state without calling the raster remap.

    ``birth_factory(cell, plate, area_km2, time_myr, serial_id)`` must return a
    newborn SurfaceParcel with explicit crust thickness and fresh material
    memory. Each returned loss carries actual donor identity/history and a
    symmetric distribution over retained overriding owners. Within-cell trench
    geometry is unresolved and is not invented by this transport experiment.

    Optional positive ``(time_fraction, weight)`` nodes integrate a constant
    birth/removal rate within each CFL interval. Births stay distinct material
    histories with their physical ages at the endpoint. ``births`` records are
    zero-age source transactions. Loss records carry their event ages/times,
    but their cold properties remain conservative transported quantities;
    the thermal caller must refresh those and account for the thermal change.
    Omitting the quadrature preserves the original endpoint scheme exactly.
    """
    dt = float(dt_myr)
    if not math.isfinite(dt) or dt <= 0. or not callable(birth_factory):
        raise ValueError('Fractional transport requires a positive timestep and birth factory')
    if state.material_model != 'oceanic_only_v1':
        raise ValueError('Experimental fractional transport supports oceanic material only')
    if event_time_quadrature is not None:
        quadrature = tuple((float(node), float(weight)) for node, weight in event_time_quadrature)
        if (not quadrature or any(not math.isfinite(node) or not 0. <= node <= 1.
                or not math.isfinite(weight) or weight <= 0. for node, weight in quadrature)
                or any(a[0] >= b[0] for a, b in zip(quadrature, quadrature[1:]))
                or not math.isclose(math.fsum(w for _, w in quadrature), 1., rel_tol=16.*_EPS, abs_tol=0.)):
            raise ValueError('Event time quadrature needs ordered nodes in [0,1] and positive unit-sum weights')
        event_time_quadrature = quadrature
    areas = np.asarray(state.cell_areas_km2, dtype=float)
    geometric = mesh.physical_cell_areas_km2(radius_km)
    if areas.shape != (mesh.cell_count,) or not np.allclose(areas, geometric, rtol=2e-13, atol=0.):
        raise ValueError('Fractional surface areas do not match the transport mesh/radius')
    flux = rigid_edge_area_fluxes(mesh, omega_rad_per_myr, radius_km)
    if any(p.plate < 0 or p.plate >= len(flux) for p in state.parcels):
        raise ValueError('Fractional component references an unknown moving plate')
    edges = np.asarray(mesh.shared_edges, dtype=int)
    outflows = [[[] for _ in areas] for _ in flux]
    divergence_error = 0.
    for plate, row in enumerate(flux):
        divergence = np.bincount(edges[:, 0], weights=row, minlength=len(areas))-np.bincount(
            edges[:, 1], weights=row, minlength=len(areas))
        denominator = max(float(np.max(np.abs(row), initial=0.)), np.finfo(float).tiny)
        divergence_error = max(divergence_error, float(np.max(np.abs(divergence), initial=0.))/denominator)
        for (a, b, _, _), amount in zip(edges, row):
            if amount > 0.:
                outflows[plate][a].append((int(b), float(amount)/areas[a]))
            elif amount < 0.:
                outflows[plate][b].append((int(a), float(-amount)/areas[b]))
    if divergence_error > 2e-13:
        raise ValueError('Rigid edge flux is not discretely divergence-free')
    rate = max((math.fsum(value for _, value in cell) for row in outflows for cell in row), default=0.)
    substeps = max(1, int(math.ceil(dt*rate)))
    sub_dt = dt/substeps
    original = state
    losses, births = [], []
    serial = 0
    roundoff = 0.
    max_outgoing = 0.
    for step in range(substeps):
        transfers = []
        for index, parcel in enumerate(state.parcels):
            weights = [(target, sub_dt*value) for target, value in outflows[parcel.plate][parcel.cell]]
            outgoing = math.fsum(value for _, value in weights)
            if outgoing > 1.+32.*_EPS:
                raise ValueError('Fractional transport CFL step overspent a donor')
            if outgoing > 1.:
                # Normalize only possible floating-point overshoot at CFL=1.
                weights = [(target, value/outgoing) for target, value in weights]
                outgoing = 1.
            max_outgoing = max(max_outgoing, outgoing)
            transfers.extend(TransferPiece(index, target, value) for target, value in weights if value > 0.)
            if outgoing < 1.:
                transfers.append(TransferPiece(index, parcel.cell, 1.-outgoing))
        incoming = remap_surface(state, transfers, sub_dt)
        step_start_time = state.time_myr
        end_time = (step_start_time+sub_dt if event_time_quadrature is not None else
                    original.time_myr+(step+1)*sub_dt)
        birth_times = {} if event_time_quadrature is not None else None
        retained, lost, born, records, error, serial = _resolve(state, incoming, end_time, birth_factory, serial,
            event_time_quadrature=event_time_quadrature, birth_times=birth_times)
        state = commit_surface(state, retained, lost, born, sub_dt)
        if event_time_quadrature is not None:
            state = replace(state, parcels=tuple(replace(p, age_myr=state.time_myr-birth_times[p.material_id])
                if p.material_id in birth_times else p for p in state.parcels))
            records = [replace(record,
                parcel=replace(record.parcel, age_myr=record.parcel.age_myr-sub_dt*(1.-node),
                    **{key: getattr(record.parcel, key)*weight for key in _EXTENSIVE}),
                source_fraction=record.source_fraction*weight,
                time_myr=step_start_time+(end_time-step_start_time)*node)
                for record in records for node, weight in event_time_quadrature]
        losses.extend(records)
        births.extend(born)
        roundoff = max(roundoff, error)
    initial_totals, final_totals = _totals(original.parcels), _totals(state.parcels)
    lost_totals, born_totals = _totals(x.parcel for x in losses), _totals(births)
    diagnostics = dict(substeps=substeps, max_outgoing_fraction=max_outgoing,
        rigid_divergence_relative_error=divergence_error, capacity_roundoff_max_km2=roundoff,
        initial_parcel_count=len(original.parcels), final_parcel_count=len(state.parcels),
        loss_piece_count=len(losses), birth_piece_count=len(births))
    if event_time_quadrature is not None:
        diagnostics['event_time_quadrature'] = [list(pair) for pair in event_time_quadrature]
    for key in _EXTENSIVE:
        diagnostics['initial_'+key] = initial_totals[key]
        diagnostics['final_'+key] = final_totals[key]
        diagnostics['subducted_'+key] = lost_totals[key]
        diagnostics['created_'+key] = born_totals[key]
        diagnostics['balance_residual_'+key] = final_totals[key]-initial_totals[key]+lost_totals[key]-born_totals[key]
    return FractionalTransportResult(state, tuple(losses), tuple(births), diagnostics)


__all__ = ['FractionalLoss', 'FractionalTransportResult', 'rigid_edge_area_fluxes', 'advance_fractional_transport']

"""Spatially connected continental collision contacts.

A pair of plate IDs can meet along several unrelated seams.  Their lengths
must not be pooled to qualify for a collision or weld.  The metrics here are
computed for each connected seam first, then the longest seam is selected.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum

import numpy as np

from .kinematics import BoundaryRecord, BoundaryType
from .mesh import SphereMesh


@dataclass(frozen=True, slots=True)
class ContinentalContact:
    pair: tuple[int, int]
    length_km: float
    mean_relative_speed_km_per_myr: float
    mean_normal_rate_km_per_myr: float
    mean_positive_divergence_km_per_myr: float
    convergent_length_fraction: float
    divergent_length_fraction: float
    edge_keys: tuple[tuple[int, int], ...]
    face_indices: tuple[int, ...]

    @property
    def faces(self) -> tuple[int, ...]:
        """Adjacent cells of the selected seam, for spatial memory matching."""
        return self.face_indices

    def can_initiate(
        self,
        minimum_length_km: float,
        max_relative_speed_km_per_myr: float,
        minimum_convergent_fraction: float = 0.5,
    ) -> bool:
        """Require an extensive, predominantly converging contact.

        One convergent edge cannot initiate collision of a mostly opening
        seam, even when its large compression makes the signed mean negative.
        """
        return (
            self.length_km >= float(minimum_length_km)
            and self.mean_relative_speed_km_per_myr
            <= float(max_relative_speed_km_per_myr)
            and self.convergent_length_fraction
            >= float(minimum_convergent_fraction)
            and self.mean_normal_rate_km_per_myr < 0.0
        )

    def can_maintain(
        self,
        minimum_length_km: float,
        max_divergence_km_per_myr: float,
    ) -> bool:
        """Allow a mature collision to become stationary without resetting.

        Use the length-weighted positive part of normal velocity.  Compression
        elsewhere on a contact must not cancel its opening motion.
        """
        return (
            self.length_km >= float(minimum_length_km)
            and self.mean_positive_divergence_km_per_myr
            <= float(max_divergence_km_per_myr)
        )

    def is_quiet(
        self,
        max_relative_speed_km_per_myr: float,
        max_divergence_km_per_myr: float,
    ) -> bool:
        """Kinematic weld criterion; the caller still applies maturity clocks."""
        return (
            self.mean_relative_speed_km_per_myr
            <= float(max_relative_speed_km_per_myr)
            and self.mean_positive_divergence_km_per_myr
            <= float(max_divergence_km_per_myr)
        )


def strongest_connected_continental_contacts(
    mesh: SphereMesh,
    boundaries: list[BoundaryRecord],
    continental_fraction: np.ndarray,
    radius_km: float,
    min_continental_fraction: float = 0.25,
) -> dict[tuple[int, int], ContinentalContact]:
    """Return the longest connected continental seam for each plate pair.

    Edges connect through a shared mesh vertex only within the same unordered
    plate pair.  Both adjacent cells must meet the continental threshold;
    effective length is arc length times their minimum continental fraction.
    Oceanic gaps therefore break a seam instead of joining distant contacts.

    Pair ordering, component traversal, summation, and ties are deterministic
    under boundary-list permutations and reversals of edge/pair orientation.
    ``edge_keys`` and ``face_indices`` identify the selected seam for callers
    that also track its spatial persistence across steps.
    """
    fraction = np.asarray(continental_fraction, dtype=float)
    by_pair: dict[tuple[int, int], list[tuple[BoundaryRecord, float]]] = defaultdict(list)
    for boundary in boundaries:
        pa, pb = int(boundary.plate_a), int(boundary.plate_b)
        if pa == pb:
            continue
        fa = float(fraction[boundary.face_a])
        fb = float(fraction[boundary.face_b])
        if not (fa >= min_continental_fraction and fb >= min_continental_fraction):
            continue
        u, v = mesh.vertices[boundary.vertex_u], mesh.vertices[boundary.vertex_v]
        length = float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))) * float(radius_km) * min(fa, fb)
        if length <= 0.0:
            continue
        by_pair[tuple(sorted((pa, pb)))].append((boundary, length))

    def edge_key(item: tuple[BoundaryRecord, float]) -> tuple[int, int]:
        boundary = item[0]
        return tuple(sorted((int(boundary.vertex_u), int(boundary.vertex_v))))

    strongest: dict[tuple[int, int], ContinentalContact] = {}
    for pair in sorted(by_pair):
        edges = sorted(by_pair[pair], key=edge_key)
        at_vertex: dict[int, list[int]] = defaultdict(list)
        for i, (boundary, _) in enumerate(edges):
            at_vertex[int(boundary.vertex_u)].append(i)
            at_vertex[int(boundary.vertex_v)].append(i)

        visited: set[int] = set()
        contacts: list[ContinentalContact] = []
        for start in range(len(edges)):
            if start in visited:
                continue
            pending = [start]
            visited.add(start)
            component: list[int] = []
            while pending:
                i = pending.pop()
                component.append(i)
                boundary = edges[i][0]
                for vertex in (boundary.vertex_u, boundary.vertex_v):
                    for j in at_vertex[int(vertex)]:
                        if j not in visited:
                            visited.add(j)
                            pending.append(j)
            segment = [edges[i] for i in sorted(component)]
            length = fsum(weight for _, weight in segment)
            contacts.append(ContinentalContact(
                pair=pair,
                length_km=length,
                mean_relative_speed_km_per_myr=fsum(weight * float(b.relative_speed_km_per_myr) for b, weight in segment) / length,
                mean_normal_rate_km_per_myr=fsum(weight * float(b.normal_rate_km_per_myr) for b, weight in segment) / length,
                mean_positive_divergence_km_per_myr=fsum(weight * max(0.0, float(b.normal_rate_km_per_myr)) for b, weight in segment) / length,
                convergent_length_fraction=fsum(weight for b, weight in segment if b.boundary_type == BoundaryType.CONVERGENT) / length,
                divergent_length_fraction=fsum(weight for b, weight in segment if b.boundary_type == BoundaryType.DIVERGENT) / length,
                edge_keys=tuple(edge_key(item) for item in segment),
                face_indices=tuple(sorted({int(face) for b, _ in segment for face in (b.face_a, b.face_b)})),
            ))
        strongest[pair] = min(contacts, key=lambda contact: (-contact.length_km, contact.edge_keys))
    return strongest

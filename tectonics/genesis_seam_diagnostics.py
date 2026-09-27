"""Read-only connectivity of cut material and remaining cohesive bonds.

A cut in the reference mesh does not by itself remove mechanical coupling.
The banks can retain shared vertex degrees of freedom or cohesive material.
Conversely, loss of all cohesive bonds does not remove compressive contact or
friction, and none of the counts here certifies an independently moving plate.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .genesis_contact_growth import CohortState
from .genesis_seams import SeamTopology


def cohort_bonded_traces(cohorts: CohortState, ntraces: int) -> np.ndarray:
    """Return whether each endpoint trace contains any remaining cohesive bond.

    The caller supplies represented material cohorts with positive reference
    areas. Every bonded cohort with damage strictly below one counts, however
    small its area or remaining stiffness. Area-averaged damage can round to
    one while a real cohort still has a bond and must not be used here.
    Constitutive history, work, and depth consistency are validated by the
    owning model; this helper only checks the fields it reads.
    """
    if (isinstance(ntraces, bool) or not isinstance(ntraces, Integral)
            or ntraces < 0):
        raise ValueError("Contact trace count must be a nonnegative integer")
    if not isinstance(cohorts, CohortState):
        raise ValueError("Contact cohorts must be a CohortState")
    index, bonded, damage = cohorts.trace_index, cohorts.bonded, cohorts.damage
    if (not isinstance(index, np.ndarray) or index.ndim != 1
            or index.dtype.kind not in "iu" or np.any(index < 0)
            or np.any(index >= ntraces)):
        raise ValueError("Cohort trace indices must be integers inside the contact geometry")
    if (not isinstance(bonded, np.ndarray) or bonded.dtype != bool
            or bonded.shape != index.shape):
        raise ValueError("Cohort bonded flags must be a boolean array matching trace indices")
    if (not isinstance(damage, np.ndarray) or damage.shape != index.shape
            or damage.dtype.kind not in "fiu" or not np.isfinite(damage).all()
            or np.any((damage < 0) | (damage > 1))):
        raise ValueError("Cohort damage must be finite values between zero and one")
    result = np.zeros(ntraces, dtype=bool)
    np.logical_or.at(result, index, bonded & (damage < 1.))
    return result


def seam_connectivity(topology: SeamTopology, bonded_trace_mask) -> dict[str, int | float]:
    """Classify a validated split topology without changing it or its histories.

    ``bonded_trace_mask`` has boolean shape ``(2 * seam_count,)``, ordered like
    the contact endpoint traces: two consecutive entries per cut edge. A true
    entry means at least one represented material cohort has a remaining
    cohesive bond. Either endpoint is sufficient to connect the two banks.

    Cut components use only intact shared edges. Cohesive components also
    connect every pair of faces sharing an actual split vertex, and the banks
    of cuts with a bonded endpoint. ``shared_vertex_link_count`` counts cut
    seams retaining at least one shared endpoint; ``shared_vertex_trace_count``
    counts those endpoint pairs. A fully decohered seam has no bonded endpoint,
    but may retain shared vertices and can still transmit compression/friction.
    Areas are the reference face areas stored in the supplied topology.
    """
    if not isinstance(topology, SeamTopology):
        raise ValueError("Connectivity requires a SeamTopology")
    bonded = np.asarray(bonded_trace_mask)
    nseams = topology.seam_count
    if bonded.dtype != bool or bonded.shape != (2*nseams,):
        raise ValueError("Bonded trace mask must be boolean with shape (2 * seam_count,)")
    faces = topology.mesh.faces
    nfaces = len(faces)
    intact = np.asarray(topology.intact_shared_edges, dtype=np.int64).reshape(-1, 4)
    graph = coo_matrix((np.ones(len(intact), dtype=bool), (intact[:, 0], intact[:, 1])),
                       shape=(nfaces, nfaces)).tocsr()
    cut_count, cut_labels = connected_components(graph, directed=False)
    sizes = np.bincount(cut_labels, minlength=cut_count)
    areas = np.bincount(cut_labels, weights=topology.mesh.areas_unit_sphere,
                        minlength=cut_count)

    # Use actual split IDs, never original parent IDs or coincident positions:
    # an opened or just-created cut can have distinct DOFs at the same point.
    first_face = np.full(topology.mesh.vertex_count, nfaces, dtype=np.int64)
    face_per_corner = np.repeat(np.arange(nfaces), 3)
    np.minimum.at(first_face, faces.ravel(), face_per_corner)
    vertex_sources = cut_labels[first_face[faces.ravel()]]
    vertex_targets = cut_labels[face_per_corner]
    bonded_seams = bonded.reshape(nseams, 2).any(axis=1)
    bonded_faces = topology.seam_faces[bonded_seams]
    sources = np.concatenate((vertex_sources, cut_labels[bonded_faces[:, 0]]))
    targets = np.concatenate((vertex_targets, cut_labels[bonded_faces[:, 1]]))
    links = sources != targets
    cohesive_graph = coo_matrix((np.ones(np.count_nonzero(links), dtype=bool),
                                (sources[links], targets[links])),
                               shape=(cut_count, cut_count)).tocsr()
    cohesive_count = connected_components(cohesive_graph, directed=False, return_labels=False)
    shared = topology.bank_vertices[:, 0, :] == topology.bank_vertices[:, 1, :]
    return {
        "cut_component_count": int(cut_count),
        "two_cell_component_count": int(np.count_nonzero(sizes == 2)),
        "largest_cut_component_area_fraction": float(areas.max()/areas.sum()),
        "cohesive_component_count": int(cohesive_count),
        "shared_vertex_link_count": int(np.count_nonzero(shared.any(axis=1))),
        "shared_vertex_trace_count": int(np.count_nonzero(shared)),
        "bonded_seam_count": int(np.count_nonzero(bonded_seams)),
        "fully_decohered_seam_count": int(nseams-np.count_nonzero(bonded_seams)),
    }

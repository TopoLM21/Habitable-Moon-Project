"""Nested mesh-aligned crack and smooth loading for shell energy controls.

One original icosahedron edge is a minor great-circle arc. Recursive spherical
bisection keeps this arc, its midpoint and its quarter points exactly: the
seed and two extension blocks consequently describe the same physical crack
at 320, 1280, 5120 and 20480 faces. The path is prescribed, not selected from
damage, and does not test orientation bias of a general crack insertion method.

The loading is an analytic, reference-position-dependent tangential surface
traction. It is a dead load during mechanical relaxation. Its continuum net
force and torque vanish on a complete sphere; a solver still must account for
its chosen discrete quadrature and displacement constraints. Neither the
fixture nor its parameterization supplies a fracture energy or a crack speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from tectonics.mesh import SphereMesh, build_icosphere


def _immutable(array, dtype=None):
    array = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def opening_traction(positions_xyz, normal_xyz, amplitude_pa: float = 1.):
    """Return ``T (n.r) [n - (n.r) r]`` in world coordinates, in pascals.

    Positive ``T`` pulls away from the great-circle plane ``n.r == 0`` on both
    sides. The field is smooth, tangent and odd under ``r -> -r``; it is the
    unit-sphere gradient of ``T (n.r)**2 / 2``. This spatial gradient property
    is distinct from the mechanical dead-load potential ``-integral(t.u dA)``.
    Zero amplitude is the unloaded control and negative amplitude reverses the
    load. Coordinates and normal are normalized without mutating their inputs.
    """
    positions = np.asarray(positions_xyz, dtype=float)
    normal = np.asarray(normal_xyz, dtype=float)
    if (positions.ndim < 1 or positions.shape[-1] != 3
            or not np.isfinite(positions).all()):
        raise ValueError("Positions must be finite vectors with shape (..., 3)")
    lengths = np.linalg.norm(positions, axis=-1)
    if np.any(lengths <= 0):
        raise ValueError("Position vectors must have positive length")
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or np.linalg.norm(normal) <= 0):
        raise ValueError("Normal must be a finite nonzero vector of shape (3,)")
    if not np.isfinite(amplitude_pa):
        raise ValueError("Traction amplitude must be finite")
    positions = positions / lengths[..., None]
    normal = normal / np.linalg.norm(normal)
    latitude = positions @ normal
    return float(amplitude_pa)*latitude[..., None]*(
        normal - latitude[..., None]*positions)


@dataclass(frozen=True)
class ShellReleaseFixture:
    """A fixed open path and loading direction on an actual ``SphereMesh``.

    ``vertex_path`` is ordered from original vertex 0 to original vertex 1.
    Seed length is half the path; an extension block is one quarter. Arrays
    created by the builder are read-only. Face and vertex IDs do not change
    under the optional rigid rotation.
    """

    mesh: SphereMesh
    radius_km: float
    subdivisions: int
    vertex_path: np.ndarray
    edge_lengths_m: np.ndarray
    normal_xyz: np.ndarray
    seed_edge_count: int
    extension_block_edge_count: int
    traction_pa: float = 1.
    depth_m: float = 10000.

    @property
    def radius_m(self):
        return self.radius_km * 1000.

    @property
    def path_vertices(self):
        return self.vertex_path

    @property
    def extension_edge_count(self):
        return self.extension_block_edge_count

    @property
    def traction_xyz_pa(self):
        """The prescribed load sampled at reference vertices, in pascals."""
        return self.traction(self.mesh.vertices)

    @property
    def edge_count(self):
        return len(self.vertex_path) - 1

    @property
    def seed_cuts(self):
        return self.cuts(self.seed_edge_count)

    @property
    def trial_cuts(self):
        return self.cuts(self.seed_edge_count + self.extension_edge_count)

    @property
    def single_edge_trial_cuts(self):
        return self.cuts(self.seed_edge_count + 1)

    def cuts(self, edge_count: int):
        """Existing source-edge IDs for a contiguous prefix of this path.

        Rows retain path order; endpoints within each row are canonicalized.
        Empty and one-edge prefixes are allowed as diagnostic controls, though
        a one-edge cut has no independently moving bank vertex in this mesh.
        """
        if (not isinstance(edge_count, Integral) or isinstance(edge_count, bool)
                or not 0 <= edge_count <= self.edge_count):
            raise ValueError("Edge count must be an integer within the path")
        pairs = np.column_stack((self.vertex_path[:edge_count],
                                 self.vertex_path[1:edge_count+1]))
        return np.sort(pairs, axis=1)

    def length_m(self, edge_count: int):
        """Physical spherical arclength of the selected prefix."""
        self.cuts(edge_count)  # Use the same prefix validation as the topology.
        return float(self.edge_lengths_m[:edge_count].sum())

    def traction(self, positions_xyz=None, amplitude_pa: float | None = None):
        """Sample the same analytic loading; default samples are face centers."""
        if positions_xyz is None:
            positions_xyz = self.mesh.centroids
        if amplitude_pa is None:
            amplitude_pa = self.traction_pa
        return opening_traction(positions_xyz, self.normal_xyz, amplitude_pa)


def build_shell_release_fixture(subdivisions: int = 2, *, radius_km: float = 5300.,
                                rotation=None, traction_pa: float = 1.,
                                depth_m: float = 10000.) -> ShellReleaseFixture:
    """Build the exact nested 0-to-1 edge arc, with a finite half-arc seed.

    Subdivisions 2, 3, 4 and 5 correspond to 320, 1280, 5120 and 20480 faces.
    No nearest-neighbour path search, mesh snapping, or damage threshold is
    involved. ``rotation`` rotates the entire mesh and load together, testing
    coordinate objectivity rather than mesh-orientation convergence.
    """
    if (not isinstance(subdivisions, Integral) or isinstance(subdivisions, bool)
            or subdivisions not in (2, 3, 4, 5)):
        raise ValueError("Fixture subdivisions must be 2, 3, 4 or 5")
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("Radius must be positive and finite")
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("Depth must be positive and finite")
    if not np.isfinite(traction_pa):
        raise ValueError("Traction amplitude must be finite")
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-12)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-12)):
        raise ValueError("Rotation must be a proper orthogonal 3 by 3 matrix")

    source = build_icosphere(int(subdivisions))
    first, last = source.vertices[[0, 1]]
    normal = np.cross(first, last)
    normal /= np.linalg.norm(normal)
    tangent = np.cross(normal, first)
    angle = np.arctan2(np.dot(last, tangent), np.dot(last, first))
    phase = np.arctan2(source.vertices @ tangent, source.vertices @ first)
    in_plane = np.abs(source.vertices @ normal) <= 2e-13
    inside_arc = (phase >= -2e-13) & (phase <= angle + 2e-13)
    path = np.flatnonzero(in_plane & inside_arc)
    path = path[np.argsort(phase[path])]
    expected_edges = 2**int(subdivisions)
    if len(path) != expected_edges + 1 or path[0] != 0 or path[-1] != 1:
        raise RuntimeError("Original edge no longer has its expected nested vertices")
    shared = {tuple(sorted((u, v))) for _, _, u, v in source.shared_edges}
    if any(tuple(sorted(pair)) not in shared for pair in zip(path[:-1], path[1:])):
        raise RuntimeError("Nested fixture path contains a non-mesh edge")

    starts, ends = source.vertices[path[:-1]], source.vertices[path[1:]]
    edge_angles = np.arctan2(np.linalg.norm(np.cross(starts, ends), axis=1),
                             np.einsum("ij,ij->i", starts, ends))
    if not np.allclose(edge_angles, angle / expected_edges, rtol=2e-13, atol=2e-15):
        raise RuntimeError("Nested fixture edges no longer bisect the reference arc")
    mesh = SphereMesh(
        vertices=_immutable(source.vertices @ rotation.T),
        faces=_immutable(source.faces),
        centroids=_immutable(source.centroids @ rotation.T),
        areas_unit_sphere=_immutable(source.areas_unit_sphere),
        neighbors=source.neighbors,
        shared_edges=source.shared_edges,
    )
    return ShellReleaseFixture(
        mesh=mesh, radius_km=float(radius_km), subdivisions=int(subdivisions),
        vertex_path=_immutable(path, np.int64),
        edge_lengths_m=_immutable(edge_angles * float(radius_km) * 1000.),
        normal_xyz=_immutable(rotation @ normal),
        seed_edge_count=expected_edges // 2,
        extension_block_edge_count=expected_edges // 4,
        traction_pa=float(traction_pa), depth_m=float(depth_m),
    )


def make_fixture(subdivisions: int = 2, rotation=None, traction_pa: float = 1., *,
                 radius_m: float = 5.3e6, depth_m: float = 10000.) -> ShellReleaseFixture:
    """SI-unit convenience entry point used by the shell energy control runner."""
    return build_shell_release_fixture(
        subdivisions, radius_km=radius_m / 1000., rotation=rotation,
        traction_pa=traction_pa, depth_m=depth_m)

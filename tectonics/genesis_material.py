"""Geometry and conserved columns for a moving material shell.

Faces retain their material identity and layer masses as their vertices move.
There is no interpolation, face exchange, remeshing, or mass source here.
Spherical areas determine column depth; chord triangles define the local
membrane kinematics, consistently with :class:`genesis_shell.Membrane`.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from .mesh import SphereMesh


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _same_topology(first, second):
    if first.vertex_count != second.vertex_count or not np.array_equal(first.faces, second.faces):
        raise ValueError("Material meshes must have identical face connectivity")


def rebuild_material_mesh(template: SphereMesh, vertices: np.ndarray) -> SphereMesh:
    """Recompute a closed, outward oriented mesh without changing topology.

    Unit vertices are required rather than silently normalizing checkpoint
    input. Edges reaching 90 degrees are rejected: the chord membrane needs
    local triangles, well before an antipodal edge becomes ambiguous.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (template.vertex_count, 3) or not np.all(np.isfinite(vertices)):
        raise ValueError("Material vertices must be finite with shape (vertex_count, 3)")
    if not np.allclose(np.linalg.norm(vertices, axis=1), 1., rtol=0., atol=2e-12):
        raise ValueError("Material vertices must be unit vectors")
    triangles = vertices[template.faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    dots = np.stack((np.einsum("fi,fi->f", a, b), np.einsum("fi,fi->f", b, c),
                     np.einsum("fi,fi->f", c, a)), axis=1)
    if np.any(dots <= 0.):
        raise ValueError("Material edge reaches or spans a hemisphere (90 degree limit)")
    oriented_volume = np.einsum("fi,fi->f", a, np.cross(b, c))
    if np.any(oriented_volume <= 1e-14):
        raise ValueError("Material face is inverted or collapsed")
    areas = 2. * np.arctan2(oriented_volume, 1. + dots.sum(axis=1))
    if not np.isclose(areas.sum(), 4. * np.pi, rtol=5e-11, atol=0.):
        raise ValueError("Material spherical areas do not cover 4 pi")
    centers = triangles.sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    return SphereMesh(vertices=vertices.copy(), faces=template.faces.copy(), centroids=centers,
                      areas_unit_sphere=areas, neighbors=template.neighbors,
                      shared_edges=template.shared_edges)


def face_frames(mesh: SphereMesh) -> np.ndarray:
    """Return (face, xyz, component) chord frames exactly matching Membrane."""
    xyz = mesh.vertices[mesh.faces]
    first = xyz[:, 1] - xyz[:, 0]
    normal = np.cross(first, xyz[:, 2] - xyz[:, 0])
    first_length, normal_length = np.linalg.norm(first, axis=1), np.linalg.norm(normal, axis=1)
    if np.any(first_length <= 0) or np.any(normal_length <= 0):
        raise ValueError("Cannot construct frames for a collapsed material face")
    first = first / first_length[:, None]
    normal = normal / normal_length[:, None]
    second = np.cross(normal, first)
    return np.stack((first, second), axis=2)


def face_deformation(oldmesh: SphereMesh, newmesh: SphereMesh,
                     old_radius_km: float, new_radius_km: float) -> np.ndarray:
    """Map old chord coordinates to new ones, including uniform radius change.

    Both local frames follow the first material edge, so a rigid 3D rotation
    produces the identity map. The returned polar rotation maps tensor
    components in the old local frame to those in the new local frame.
    """
    _same_topology(oldmesh, newmesh)
    old_radius = _positive(old_radius_km, "old_radius_km")
    new_radius = _positive(new_radius_km, "new_radius_km")

    def edges(mesh):
        xyz = mesh.vertices[mesh.faces]
        chord = np.stack((xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=2)
        return np.einsum("fji,fjk->fik", face_frames(mesh), chord)

    old, new = edges(oldmesh), edges(newmesh)
    deformation = (new_radius / old_radius) * (new @ np.linalg.inv(old))
    if not np.all(np.isfinite(deformation)) or np.any(np.linalg.det(deformation) <= 0):
        raise ValueError("Material deformation must preserve orientation")
    return deformation


def polar_increment(deformation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proper polar R and right Hencky log(U), as [xx, yy, 2xy] in old frame."""
    deformation = np.asarray(deformation, dtype=float)
    if deformation.shape[-2:] != (2, 2) or not np.all(np.isfinite(deformation)):
        raise ValueError("Deformation must be finite with trailing shape (2, 2)")
    if np.any(np.linalg.det(deformation) <= 0):
        raise ValueError("Deformation must preserve orientation")
    left, stretch, right_t = np.linalg.svd(deformation)
    if np.any(stretch[..., -1] <= 1e-12 * stretch[..., 0]):
        raise ValueError("Deformation is collapsed or ill conditioned")
    rotation = left @ right_t
    right = np.swapaxes(right_t, -1, -2)
    hencky = (right * np.log(stretch)[..., None, :]) @ right_t
    return rotation, np.stack((hencky[..., 0, 0], hencky[..., 1, 1], 2 * hencky[..., 0, 1]), axis=-1)


def rotate_tensor(values: np.ndarray, rotation: np.ndarray, engineering: bool = False) -> np.ndarray:
    """Rotate stress [xx, yy, xy], or strain [xx, yy, gamma] when requested."""
    values, rotation = np.asarray(values, dtype=float), np.asarray(rotation, dtype=float)
    if values.shape[-1:] != (3,) or rotation.shape != values.shape[:-1] + (2, 2):
        raise ValueError("Tensor values and rotations must have matching batch shapes")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(rotation)):
        raise ValueError("Tensor values and rotations must be finite")
    if not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), np.eye(2), rtol=0, atol=2e-10) or np.any(np.linalg.det(rotation) <= 0):
        raise ValueError("Tensor rotation must be a proper orthogonal matrix")
    shear_factor = 2. if engineering else 1.
    matrix = np.empty(values.shape[:-1] + (2, 2))
    matrix[..., 0, 0], matrix[..., 1, 1] = values[..., 0], values[..., 1]
    matrix[..., 0, 1] = matrix[..., 1, 0] = values[..., 2] / shear_factor
    result = rotation @ matrix @ np.swapaxes(rotation, -1, -2)
    return np.stack((result[..., 0, 0], result[..., 1, 1], shear_factor * result[..., 0, 1]), axis=-1)


def move_mesh(mesh: SphereMesh, vertex_basis: np.ndarray, delta_rad: np.ndarray) -> SphereMesh:
    """Apply normalized tangent increments, then validate the moving mesh."""
    basis, delta = np.asarray(vertex_basis, dtype=float), np.asarray(delta_rad, dtype=float)
    if basis.shape != (mesh.vertex_count, 3, 2) or delta.shape != (mesh.vertex_count, 2):
        raise ValueError("Vertex basis or material increment has the wrong shape")
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(delta)):
        raise ValueError("Vertex basis and material increment must be finite")
    if not np.allclose(np.einsum("vji,vjk->vik", basis, basis), np.eye(2), rtol=0, atol=2e-12):
        raise ValueError("Vertex basis must be orthonormal")
    if not np.allclose(np.einsum("vi,vij->vj", mesh.vertices, basis), 0., rtol=0, atol=2e-12):
        raise ValueError("Vertex basis must be tangent to the current material mesh")
    vertices = mesh.vertices + np.einsum("vij,vj->vi", basis, delta)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    return rebuild_material_mesh(mesh, vertices)


def material_layer_mass(mesh: SphereMesh, radius_km: float, density_kg_m3: float,
                        depth_km: float, layers: int) -> np.ndarray:
    """Initialize equal-mass layers within each thin, constant-density column."""
    radius, density, depth = (_positive(radius_km, "radius_km"),
                              _positive(density_kg_m3, "density_kg_m3"),
                              _positive(depth_km, "depth_km"))
    if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
        raise ValueError("layers must be a positive integer")
    per_layer = mesh.physical_cell_areas_km2(radius) * depth * 1e9 * density / layers
    return np.broadcast_to(per_layer[:, None], (mesh.cell_count, layers)).copy()


def material_column_depth(mesh: SphereMesh, radius_km: float, layer_mass: np.ndarray,
                          density_kg_m3: float) -> np.ndarray:
    """Current depths in km from fixed masses and current spherical face areas."""
    radius, density = _positive(radius_km, "radius_km"), _positive(density_kg_m3, "density_kg_m3")
    mass = np.asarray(layer_mass, dtype=float)
    if mass.ndim != 2 or mass.shape[0] != mesh.cell_count or mass.shape[1] < 1:
        raise ValueError("Layer masses must have shape (cell_count, layers)")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Layer masses must be positive and finite")
    return mass.sum(axis=1) / (mesh.physical_cell_areas_km2(radius) * density * 1e9)


def geometry_diagnostics(mesh: SphereMesh, reference_mesh: SphereMesh,
                         radius: float, reference_radius: float) -> dict:
    """Report chord triangle quality and physical area changes of material cells."""
    _same_topology(mesh, reference_mesh)
    radius, reference_radius = _positive(radius, "radius"), _positive(reference_radius, "reference_radius")
    xyz = mesh.vertices[mesh.faces]
    edge1, edge2, edge3 = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 1], xyz[:, 0] - xyz[:, 2]
    chord_area = .5 * np.linalg.norm(np.cross(edge1, -edge3), axis=1)
    denominator = (edge1**2 + edge2**2 + edge3**2).sum(axis=1)
    quality = 4 * np.sqrt(3.) * chord_area / denominator
    ratio = mesh.physical_cell_areas_km2(radius) / reference_mesh.physical_cell_areas_km2(reference_radius)
    return {"min_face_quality": float(np.min(quality)),
            "min_area_ratio": float(np.min(ratio)), "max_area_ratio": float(np.max(ratio)),
            "total_area_relative_residual": float((mesh.areas_unit_sphere.sum() - 4 * np.pi) / (4 * np.pi))}

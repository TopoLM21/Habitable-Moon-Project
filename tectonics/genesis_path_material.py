"""Conservative material subdivision for an explicitly inserted crack support.

Refining spherical triangles changes the chord finite-element approximation.
Conservation of material and isotropic stored energy does NOT imply force
equilibrium, stiffness nesting, or a valid fracture-energy comparison. Nothing
here advances time, resets elastic memory, or creates a production checkpoint.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from .genesis_material import face_frames, rotate_tensor
from .genesis_seams import split_mesh


def _owned(values):
    values = np.ascontiguousarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _normal_rotation(first, second):
    """Shortest proper 3D rotation between corresponding unit directions."""
    cross = np.cross(first, second)
    cosine = np.einsum("fi,fi->f", first, second)
    if np.any(cosine < 0):
        raise ValueError("Subdivision frame normals must share a hemisphere")
    skew = np.zeros((len(first), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -cross[:, 2], cross[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = cross[:, 2], -cross[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -cross[:, 1], cross[:, 0]
    return np.eye(3)+skew+skew@skew/(1+cosine)[:, None, None]


class ConservativeSubdivision:
    """Validated child ancestry, area shares and objective frame transport.

    Material is piecewise constant within each original face. Extensive fields
    use spherical-area shares normalized within the SAME parent. Specific
    enthalpy, damage, water and history are copied, without smoothing or mixing.
    Parent-to-child frame changes are a declared parallel-transport convention
    between chord normals, not a physical deformation or release of stress.
    """

    def __init__(self, parent_mesh, child_mesh, parent_face):
        # Reuse the strict closed oriented material-manifold validation, and
        # own geometry so future caller mutations cannot alter the projection.
        self.parent_mesh = split_mesh(parent_mesh, []).mesh
        self.mesh = split_mesh(child_mesh, []).mesh
        for mesh in (self.parent_mesh, self.mesh):
            for name in ("vertices", "faces", "centroids", "areas_unit_sphere"):
                setattr(mesh, name, _owned(getattr(mesh, name)))
        parent = np.asarray(parent_face)
        count = self.parent_mesh.cell_count
        if (parent.shape != (self.mesh.cell_count,) or parent.dtype.kind not in "iu"
                or np.any(parent < 0) or np.any(parent >= count)):
            raise ValueError("Child parent_face must contain valid original material face IDs")
        summed = np.bincount(parent, weights=self.mesh.areas_unit_sphere, minlength=count)
        old_area = self.parent_mesh.areas_unit_sphere
        if np.any(summed <= 0) or not np.allclose(summed, old_area, rtol=5e-11, atol=0):
            raise ValueError("Child areas must cover each parent exactly without material loss")
        # Area alone would accept a permutation of equal-area parent faces.
        # Each child vertex must also lie inside its claimed convex triangle.
        old_xyz = self.parent_mesh.vertices[self.parent_mesh.faces[parent]]
        child_xyz = self.mesh.vertices[self.mesh.faces]
        normals = np.cross(old_xyz, np.roll(old_xyz, -1, axis=1))
        normals /= np.linalg.norm(normals, axis=2)[:, :, None]
        if np.any(np.einsum("fij,fkj->fik", normals, child_xyz) < -2e-11):
            raise ValueError("Child geometry lies outside its claimed material parent")
        self.parent_face = _owned(parent.astype(np.int64))
        self.area_fraction = _owned(self.mesh.areas_unit_sphere/summed[parent])
        self.area_relative_error = float(np.max(np.abs(summed/old_area-1)))
        a, b = face_frames(self.parent_mesh)[parent], face_frames(self.mesh)
        na, nb = np.cross(a[:, :, 0], a[:, :, 1]), np.cross(b[:, :, 0], b[:, :, 1])
        world = _normal_rotation(na, nb)
        rotation = np.einsum("fji,fjk,fkl->fil", b, world, a)
        unchanged = np.all(old_xyz == child_xyz, axis=(1, 2))
        rotation[unchanged] = np.eye(2)
        world[unchanged] = np.eye(3)
        if not np.allclose(rotation@rotation.transpose(0, 2, 1), np.eye(2), atol=2e-12, rtol=0):
            raise ValueError("Child frame transport is not a proper orthogonal map")
        self.frame_rotation = _owned(rotation)
        self.world_rotation = _owned(world)
        # Diagnostics such as velocity live in centroid tangent planes, rather
        # than the slightly different chord frames used by elastic tensors.
        tangent_rotation = _normal_rotation(self.parent_mesh.centroids[parent], self.mesh.centroids)
        tangent_rotation[unchanged] = np.eye(3)
        self.tangent_rotation = _owned(tangent_rotation)
        largest = np.full(count, -1, dtype=np.int64)
        for child, old in enumerate(parent):
            if largest[old] < 0 or self.area_fraction[child] > self.area_fraction[largest[old]]:
                largest[old] = child
        self._largest_child = _owned(largest)

    def _field(self, value):
        value = np.asarray(value)
        if (value.ndim < 1 or len(value) != self.parent_mesh.cell_count
                or value.dtype.kind not in "biuf" or not np.isfinite(value).all()):
            raise ValueError("Material field must be finite numeric values per original face")
        return value

    def intensive(self, value):
        """Copy a scalar/array per parent; boolean and integer identity survives."""
        return self._field(value)[self.parent_face].copy()

    def extensive(self, value):
        """Partition a quantity by area with per-parent roundoff correction."""
        value = np.asarray(self._field(value), dtype=float)
        weights = self.area_fraction.reshape((-1,)+(1,)*(value.ndim-1))
        result = value[self.parent_face]*weights
        # Correct only floating summation error, using the largest child so a
        # tiny sliver never receives a parent-sized roundoff correction.
        summed = np.zeros_like(value)
        np.add.at(summed, self.parent_face, result)
        result[self._largest_child] += value-summed
        if not np.isfinite(result).all():
            raise ValueError("Material subdivision overflowed an extensive field")
        if np.any((value[self.parent_face] >= 0) & (result < 0)):
            raise ValueError("Subdivision roundoff would create a negative extensive quantity")
        return result

    def tensor(self, value, *, engineering=False):
        """Rotate stress or engineering strain without changing eigenvalues."""
        value = self._field(value)
        if value.shape != (self.parent_mesh.cell_count, 3):
            raise ValueError("Material tensor must have shape (parent_faces, 3)")
        return rotate_tensor(value[self.parent_face], self.frame_rotation, engineering=engineering)

    def plane_normals(self, value):
        value = self._field(value)
        if value.shape != (self.parent_mesh.cell_count, 2):
            raise ValueError("Material plane normal must have two face-frame components")
        norm = np.linalg.norm(value, axis=1)
        if not np.all((norm == 0) | np.isclose(norm, 1., rtol=0, atol=1e-10)):
            raise ValueError("Material plane normals must be unit or inactive zero vectors")
        return np.einsum("fij,fj->fi", self.frame_rotation, value[self.parent_face])

    def tangent_vectors(self, value):
        value = self._field(value)
        if value.shape != (self.parent_mesh.cell_count, 3):
            raise ValueError("World tangent vectors must have shape (parent_faces, 3)")
        scale = np.maximum(np.linalg.norm(value, axis=1), 1.)
        if np.any(np.abs(np.einsum("fi,fi->f", value, self.parent_mesh.centroids)) > 1e-10*scale):
            raise ValueError("World vectors must be tangent at the parent centroid")
        return np.einsum("fij,fj->fi", self.tangent_rotation, value[self.parent_face])

    def fault_fields(self, state):
        """Project every FaultState face array into an explicit material bundle.

        This returns fields only, NOT a FaultState/production checkpoint. Global
        clocks and ledgers belong to the source archive. The old equilibrium
        residual is not transferable to the new finite-element approximation.
        Unknown array fields fail rather than silently dropping future history.
        """
        if not np.array_equal(state.vertices, self.parent_mesh.vertices):
            raise ValueError("Fault material state does not match the projection's parent geometry")
        intensive = {"column_enthalpy", "damage", "water_access", "peak_tensile_pa",
            "weak_duration_myr", "path_length_km", "tidal_stress_mpa", "fault_active",
            "activation_time_myr", "fault_candidate_age_myr", "cumulative_shear",
            "signed_shear", "last_shear_increment", "shear_stress_pa", "normal_stress_pa",
            "shear_strength_pa"}
        extensive = {"layer_mass_kg", "friction_work_cell_j", "viscous_work_cell_j"}
        result = {}
        for item in fields(state):
            name, value = item.name, getattr(state, item.name)
            if not isinstance(value, np.ndarray) or name == "vertices":
                continue
            if name in intensive:
                result[name] = self.intensive(value)
            elif name in extensive:
                result[name] = self.extensive(value)
            elif name == "elastic_strain":
                result[name] = self.tensor(value, engineering=True)
            elif name == "plane_normal":
                result[name] = self.plane_normals(value)
            elif name == "velocity_km_myr":
                result[name] = self.tangent_vectors(value)
            else:
                raise ValueError(f"No explicit subdivision rule for material array {name}")
        return result

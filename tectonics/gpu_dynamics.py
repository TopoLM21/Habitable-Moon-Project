"""Isolated exact-order GPE torque candidates; not wired into the model.

The static geometry is prepared with the scalar NumPy operations used by
``update_plate_dynamics``. CUDA only multiplies the evolving thickness weights
and sums each plate's edges in the original cell/neighbour order. No atomics,
parallel reassociation, fast math, or lower precision are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lithosphere import CrustType
from .mesh import SphereMesh


@dataclass
class GpeGeometry:
    mesh: SphereMesh
    source: np.ndarray
    neighbors: np.ndarray
    cross: np.ndarray
    valid: np.ndarray

    @classmethod
    def prepare(cls, mesh: SphereMesh) -> "GpeGeometry":
        neighbors = np.asarray(mesh.neighbors, dtype=np.int32)
        if neighbors.shape != (mesh.cell_count, 3):
            raise ValueError("GPE probe requires three neighbours per cell")
        if np.any(neighbors < 0) or np.any(neighbors >= mesh.cell_count):
            raise ValueError("Neighbour index out of range")
        source = np.repeat(np.arange(mesh.cell_count, dtype=np.int32), 3)
        cross = np.zeros((mesh.cell_count * 3, 3), dtype=np.float64)
        valid = np.zeros(mesh.cell_count * 3, dtype=np.int8)
        for cell in range(mesh.cell_count):
            r0 = np.asarray(mesh.centroids[cell], dtype=np.float64)
            for slot, nb in enumerate(mesh.neighbors[cell]):
                tangent = np.asarray(mesh.centroids[int(nb)] - r0, dtype=np.float64)
                tangent -= r0 * float(np.dot(tangent, r0))
                tn = float(np.linalg.norm(tangent))
                if tn <= 1e-14:
                    continue
                tangent /= tn
                edge = 3 * cell + slot
                cross[edge] = np.cross(r0, tangent)
                valid[edge] = 1
        return cls(mesh, source, neighbors, cross, valid)

    @property
    def host_bytes(self) -> int:
        return sum(x.nbytes for x in (self.source, self.neighbors, self.cross, self.valid))


def _inputs(geometry, state, plate_count, href):
    count = int(plate_count)
    if count <= 0 or count != plate_count:
        raise ValueError("plate_count must be positive")
    n = geometry.mesh.cell_count
    owner = np.ascontiguousarray(state.cell_plate, dtype=np.int32)
    thickness = np.ascontiguousarray(state.crust_thickness_km, dtype=np.float64)
    continental = np.ascontiguousarray(
        np.asarray(state.crust_type) == int(CrustType.CONTINENTAL), dtype=np.int8
    )
    if any(x.shape != (n,) for x in (owner, thickness, continental)):
        raise ValueError("State fields must have one value per mesh cell")
    if np.any(owner < 0) or np.any(owner >= count):
        raise ValueError("Plate owner index out of range")
    if not np.all(np.isfinite(thickness)) or not np.isfinite(href):
        raise ValueError("Thickness and reference must be finite")
    return owner, thickness, continental


def gpe_reference(mesh, state, plate_count, href):
    """Literal original GPE loop including its per-plate normalisation."""
    drive = np.zeros((plate_count, 3), dtype=np.float64)
    weight = np.zeros(plate_count, dtype=np.float64)
    cont = np.asarray(state.crust_type) == int(CrustType.CONTINENTAL)
    for cell in np.flatnonzero(cont):
        pid = int(state.cell_plate[cell])
        h0 = float(state.crust_thickness_km[cell])
        excess = max(h0 - float(href), 0.0)
        if excess <= 0.0:
            continue
        r0 = np.asarray(mesh.centroids[cell], dtype=np.float64)
        for nb in mesh.neighbors[cell]:
            nb = int(nb)
            if int(state.cell_plate[nb]) != pid or not cont[nb]:
                continue
            dh = max(h0 - float(state.crust_thickness_km[nb]), 0.0)
            if dh <= 0.0:
                continue
            tangent = np.asarray(mesh.centroids[nb] - r0, dtype=np.float64)
            tangent -= r0 * float(np.dot(tangent, r0))
            tn = float(np.linalg.norm(tangent))
            if tn <= 1e-14:
                continue
            tangent /= tn
            w = excess * dh
            drive[pid] += w * np.cross(r0, tangent)
            weight[pid] += w
    nonzero = weight > 0.0
    drive[nonzero] /= weight[nonzero, None]
    return drive, weight


def gpe_prepared_cpu(geometry, state, plate_count, href):
    """Vectorised edge products with sequential NumPy ``add.at`` reduction."""
    owner, thickness, continental = _inputs(geometry, state, plate_count, href)
    source = geometry.source
    neighbor = geometry.neighbors.ravel()
    excess = np.maximum(thickness[source] - float(href), 0.0)
    dh = np.maximum(thickness[source] - thickness[neighbor], 0.0)
    valid = (
        (continental[source] != 0) & (continental[neighbor] != 0)
        & (owner[source] == owner[neighbor]) & (excess > 0.0)
        & (dh > 0.0) & (geometry.valid != 0)
    )
    ids = owner[source[valid]]
    values = excess[valid] * dh[valid]
    drive = np.zeros((plate_count, 3), dtype=np.float64)
    weight = np.zeros(plate_count, dtype=np.float64)
    np.add.at(drive, ids, values[:, None] * geometry.cross[valid])
    np.add.at(weight, ids, values)
    nonzero = weight > 0.0
    drive[nonzero] /= weight[nonzero, None]
    return drive, weight


CUDA_SOURCE = r'''
extern "C" __global__ void emit_gpe(
    const int* owner, const double* thickness, const signed char* continental,
    const int* neighbors, const double* cross, const signed char* valid,
    double* edge_value, int n, double href) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= 3 * n) return;
    int cell = edge / 3, nb = neighbors[edge];
    double excess = thickness[cell] - href;
    double dh = thickness[cell] - thickness[nb];
    double w = 0.0;
    if (continental[cell] && continental[nb] && owner[cell] == owner[nb]
            && excess > 0.0 && dh > 0.0 && valid[edge]) w = excess * dh;
    for (int axis = 0; axis < 3; ++axis)
        edge_value[4 * edge + axis] = w * cross[3 * edge + axis];
    edge_value[4 * edge + 3] = w;
}

extern "C" __global__ void reduce_gpe(
    const double* edge_value, const int* cells, const int* offsets,
    double* out, int plates) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 4 * plates) return;
    int plate = index / 4, axis = index % 4;
    double total = 0.0;
    for (int j = offsets[plate]; j < offsets[plate + 1]; ++j) {
        int cell = cells[j];
        for (int slot = 0; slot < 3; ++slot) {
            int edge = 3 * cell + slot;
            // The original loop skips zero-weight edges entirely.
            if (edge_value[4 * edge + 3] > 0.0)
                total += edge_value[4 * edge + axis];
        }
    }
    out[index] = total;
}
'''


class GpuGpeProbe:
    """One-mesh experimental workspace; host fields refreshed on every call."""
    def __init__(self, geometry: GpeGeometry, device: int = 0):
        import cupy as cp

        self.cp = cp
        self.geometry = geometry
        self.device = cp.cuda.Device(device)
        self.device.use()
        module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++17", "--fmad=false"))
        self.emit = module.get_function("emit_gpe")
        self.reduce = module.get_function("reduce_gpe")
        self.neighbors = cp.asarray(geometry.neighbors)
        self.cross = cp.asarray(geometry.cross)
        self.valid = cp.asarray(geometry.valid)
        n = geometry.mesh.cell_count
        self.owner = cp.empty(n, dtype=cp.int32)
        self.thickness = cp.empty(n, dtype=cp.float64)
        self.continental = cp.empty(n, dtype=cp.int8)
        self.cells = cp.empty(n, dtype=cp.int32)
        self.edge_value = cp.empty((3 * n, 4), dtype=cp.float64)
        self.offsets: Any = None
        self.out: Any = None
        cp.cuda.get_current_stream().synchronize()

    def calculate(self, state, plate_count, href):
        self.device.use()
        owner, thickness, continental = _inputs(self.geometry, state, plate_count, href)
        # Stable grouping retains increasing cell index within each plate.
        selected = np.flatnonzero((continental != 0) & (thickness > href))
        selected = selected[np.argsort(owner[selected], kind="stable")].astype(np.int32)
        counts = np.bincount(owner[selected], minlength=plate_count)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
        cp = self.cp
        if self.out is None or self.out.shape != (plate_count, 4):
            self.out = cp.empty((plate_count, 4), dtype=cp.float64)
            self.offsets = cp.empty(plate_count + 1, dtype=cp.int32)
        self.owner.set(owner)
        self.thickness.set(thickness)
        self.continental.set(continental)
        if len(selected):
            self.cells[:len(selected)].set(selected)
        self.offsets.set(offsets)
        n = self.geometry.mesh.cell_count
        self.emit(((3 * n + 255) // 256,), (256,), (
            self.owner, self.thickness, self.continental, self.neighbors,
            self.cross, self.valid, self.edge_value, np.int32(n), np.float64(href),
        ))
        self.reduce(((4 * plate_count + 127) // 128,), (128,), (
            self.edge_value, self.cells, self.offsets, self.out, np.int32(plate_count),
        ))
        out = self.out.get()
        drive, weight = out[:, :3].copy(), out[:, 3].copy()
        nonzero = weight > 0.0
        drive[nonzero] /= weight[nonzero, None]
        return drive, weight

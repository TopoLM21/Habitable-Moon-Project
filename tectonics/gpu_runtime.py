"""Optional deterministic CUDA execution for regular cell kernels.

CuPy is imported only when :class:`GpuExecution` is entered.  The stable and
CPU-optimized runners therefore keep working in environments without CUDA.
One execution context owns compiled kernels, a bounded cache of immutable
mesh connectivity, and an optional reusable workspace for surface physics.
The surface workspace refreshes its host inputs on every call.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .mesh import SphereMesh


CUDA_SOURCE = r'''
extern "C" __global__ void emit_flux(
    const double* z, const int* neighbors, const double* mobile,
    double* sediment, double* edges, int n, double sea, double land, double basin) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double v = mobile[i];
    double drops[3];
    double total = 0.0;
    bool downhill = false;
    for (int k = 0; k < 3; ++k) {
        edges[3*i+k] = 0.0;
        double d = z[i] - z[neighbors[3*i+k]];
        drops[k] = d > 1.0e-9 ? d : 0.0;
        total += drops[k];
        downhill = downhill || d > 1.0e-9;
    }
    if (!(v > 0.0)) return;
    if (!downhill) { sediment[i] += v; return; }
    double dep = z[i] <= sea ? basin : land;
    dep = dep < 0.0 ? 0.0 : (dep > 1.0 ? 1.0 : dep);
    sediment[i] += v * dep;
    double move = v * (1.0 - dep);
    double denominator = total > 1.0e-30 ? total : 1.0e-30;
    for (int k = 0; k < 3; ++k)
        if (drops[k] > 0.0) edges[3*i+k] = move * (drops[k] / denominator);
}

extern "C" __global__ void gather_flux(
    const double* edges, const int* incoming, double* next, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double total = 0.0;
    // Source IDs are sorted, preserving the CPU scatter-add order.
    for (int k = 0; k < 3; ++k) total += edges[incoming[3*i+k]];
    next[i] = total;
}

extern "C" __global__ void paint_arcs(
    const double* centroids, const int* owner,
    const double* centers, const int* plates, const double* amplitudes,
    const double* sigma, const double* minimum_dot,
    double* field, int cells, int tasks, double radius) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= cells) return;
    double x = centroids[3*cell];
    double y = centroids[3*cell+1];
    double z = centroids[3*cell+2];
    int plate = owner[cell];
    double best = 0.0;
    for (int task = 0; task < tasks; ++task) {
        if (plates[task] != plate || amplitudes[task] <= 0.0) continue;
        double dot = x*centers[3*task] + y*centers[3*task+1] + z*centers[3*task+2];
        if (dot < minimum_dot[task]) continue;
        dot = dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot);
        double distance = acos(dot) * radius;
        double scaled = distance / sigma[task];
        double value = amplitudes[task] * exp(-0.5 * scaled * scaled);
        best = best > value ? best : value;
    }
    field[cell] = best;
}
'''


class GpuUnavailableError(RuntimeError):
    """CUDA execution could not be initialized."""


@dataclass
class DeviceMesh:
    """Immutable connectivity resident on one CUDA device."""

    mesh: SphereMesh
    neighbors: Any
    incoming: Any
    static_bytes: int
    centroids: Any | None = None


_active: GpuExecution | None = None


class GpuExecution(AbstractContextManager):
    """Own one deterministic CUDA context and its bounded static-data cache."""

    def __init__(self, device: int = 0, *, arc_painting: bool = False,
                 surface_pipeline: bool = False) -> None:
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("GPU device must be a non-negative integer")
        self.device_id = device
        self.arc_painting = bool(arc_painting)
        self.surface_pipeline = bool(surface_pipeline)
        self.cp: Any | None = None
        self.device_name: str | None = None
        self.cupy_version: str | None = None
        self.cuda_runtime_version: int | None = None
        self.compute_capability: str | None = None
        self._module: Any | None = None
        self._emit: Any | None = None
        self._gather: Any | None = None
        self._paint_arcs: Any | None = None
        self._previous_device_id: int | None = None
        self._meshes: OrderedDict[int, DeviceMesh] = OrderedDict()
        self._surface_workspace: Any | None = None
        self.surface_report: dict[str, object] | None = None
        self.routing_calls = 0
        self.routing_cells = 0
        self.routing_seconds = 0.0
        self.dynamic_transfer_bytes = 0
        self.routing_scalar_transfer_bytes = 0
        self.routing_threshold_transfer_bytes = 0
        self.static_transfer_bytes = 0
        self.arc_calls = 0
        self.arc_tasks = 0
        self.arc_seconds = 0.0
        self.arc_dynamic_transfer_bytes = 0

    def __enter__(self) -> "GpuExecution":
        global _active
        if _active is not None:
            raise RuntimeError("A GPU execution context is already active")
        try:
            try:
                import cupy as cp
            except (ImportError, OSError) as exc:
                raise GpuUnavailableError(
                    "CuPy/CUDA is unavailable; install requirements-gpu.txt in this environment"
                ) from exc
            self.cp = cp
            try:
                count = int(cp.cuda.runtime.getDeviceCount())
                if self.device_id >= count:
                    raise GpuUnavailableError(
                        f"CUDA device {self.device_id} does not exist; detected {count} device(s)"
                    )
                self._previous_device_id = int(cp.cuda.runtime.getDevice())
                cp.cuda.Device(self.device_id).use()
                properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
                name = properties["name"]
                self.device_name = name.decode() if isinstance(name, bytes) else str(name)
                self.compute_capability = f"{int(properties['major'])}.{int(properties['minor'])}"
                self.cupy_version = str(cp.__version__)
                self.cuda_runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
                self._module = cp.RawModule(
                    code=CUDA_SOURCE,
                    options=("--std=c++17", "--fmad=false"),
                )
                self._emit = self._module.get_function("emit_flux")
                self._gather = self._module.get_function("gather_flux")
                self._paint_arcs = self._module.get_function("paint_arcs")
            except GpuUnavailableError:
                raise
            except Exception as exc:
                raise GpuUnavailableError(
                    f"CUDA device {self.device_id} could not be initialized: {exc}"
                ) from exc
            _active = self
            return self
        except BaseException:
            try:
                self._restore_device()
            finally:
                self._surface_workspace = None
                self._meshes.clear()
                self.cp = None
                self._module = self._emit = self._gather = self._paint_arcs = None
            raise

    def __exit__(self, *exc: object) -> None:
        global _active
        try:
            if self.cp is not None:
                self.cp.cuda.Device(self.device_id).use()
                self.cp.cuda.get_current_stream().synchronize()
        finally:
            try:
                if self._surface_workspace is not None:
                    self.surface_report = self._surface_workspace.report()
            finally:
                self._surface_workspace = None
                self._meshes.clear()
                self._module = self._emit = self._gather = self._paint_arcs = None
                try:
                    self._restore_device()
                finally:
                    self.cp = None
                    _active = None

    def _restore_device(self) -> None:
        try:
            if self.cp is not None and self._previous_device_id is not None:
                self.cp.cuda.Device(self._previous_device_id).use()
        finally:
            self._previous_device_id = None

    def _require_active(self) -> Any:
        if self.cp is None or _active is not self:
            raise RuntimeError("GPU execution context is not active")
        self.cp.cuda.Device(self.device_id).use()
        return self.cp

    def surface_workspace(self, mesh: SphereMesh) -> Any:
        """Return one reusable surface workspace, bound to this mesh identity."""
        self._require_active()
        if not self.surface_pipeline:
            raise RuntimeError("GPU surface pipeline is not enabled")
        if self._surface_workspace is None or self._surface_workspace.mesh is not mesh:
            from .gpu_surface import SurfaceWorkspace

            # Drop old device buffers before allocating a different mesh.
            if self._surface_workspace is not None:
                self.surface_report = self._surface_workspace.report()
            self._surface_workspace = None
            self._surface_workspace = SurfaceWorkspace(self, mesh)
        return self._surface_workspace

    def geometry(self, mesh: SphereMesh) -> DeviceMesh:
        """Upload fixed triangular connectivity once for this mesh identity."""
        cp = self._require_active()
        key = id(mesh)
        if key not in self._meshes:
            raw_neighbors = np.asarray(mesh.neighbors)
            if mesh.cell_count < 1 or raw_neighbors.shape != (mesh.cell_count, 3):
                raise ValueError("CUDA routing requires a closed triangular mesh")
            if not np.issubdtype(raw_neighbors.dtype, np.integer):
                raise ValueError("CUDA routing neighbour indices must be integers")
            if (mesh.cell_count > np.iinfo(np.int32).max // 3
                    or np.any(raw_neighbors < 0) or np.any(raw_neighbors >= mesh.cell_count)):
                raise ValueError("CUDA routing neighbour indices are out of range")
            neighbors = np.ascontiguousarray(raw_neighbors, dtype=np.int32)
            sources = np.sort(neighbors, axis=1)
            if (np.any(sources[:, 1:] == sources[:, :-1])
                    or np.any(neighbors == np.arange(mesh.cell_count)[:, None])):
                raise ValueError("CUDA routing requires unique non-self neighbours")
            matches = neighbors[sources] == np.arange(mesh.cell_count)[:, None, None]
            if not np.all(matches.sum(axis=2) == 1):
                raise ValueError("CUDA routing requires reciprocal, unique neighbours")
            incoming = (sources * 3 + np.argmax(matches, axis=2)).astype(np.int32)
            item = DeviceMesh(
                mesh=mesh,
                neighbors=cp.asarray(neighbors),
                incoming=cp.asarray(incoming),
                static_bytes=int(neighbors.nbytes + incoming.nbytes),
            )
            self.static_transfer_bytes += item.static_bytes
            self._meshes[key] = item
            if len(self._meshes) > 2:
                self._meshes.popitem(last=False)
        self._meshes.move_to_end(key)
        return self._meshes[key]

    def route_mobile(self, mesh: SphereMesh, elevation_m: np.ndarray,
                     stationary: np.ndarray, mobile: np.ndarray, params: object,
                     sea_level_m: float) -> np.ndarray:
        """Run exact-order FP64 sediment routing and return host-owned state."""
        cp = self._require_active()
        z_host = np.ascontiguousarray(elevation_m, dtype=np.float64)
        sed_host = np.ascontiguousarray(stationary, dtype=np.float64)
        mob_host = np.ascontiguousarray(mobile, dtype=np.float64)
        expected_shape = (mesh.cell_count,)
        if z_host.shape != expected_shape or sed_host.shape != expected_shape or mob_host.shape != expected_shape:
            raise ValueError("CUDA routing fields must match mesh cell count")

        started = perf_counter()
        previous_seconds = self.routing_seconds
        z = cp.asarray(z_host)
        sed = cp.asarray(sed_host)
        mob = cp.asarray(mob_host)
        self.dynamic_transfer_bytes += int(z_host.nbytes + sed_host.nbytes + mob_host.nbytes)
        result = self.route_mobile_device(mesh, z, sed, mob, params, sea_level_m).get()
        self.dynamic_transfer_bytes += int(result.nbytes)
        # The host wrapper includes all transfers and device completion.
        self.routing_seconds = previous_seconds + perf_counter() - started
        return result

    def route_mobile_device(self, mesh: SphereMesh, z: Any, sed: Any, mob: Any,
                            params: object, sea_level_m: float, *,
                            next_mob: Any | None = None, edges: Any | None = None) -> Any:
        """Route private FP64 device buffers and return combined sediment on-device.

        ``sed``, ``mob`` and optional scratch buffers may be overwritten. They
        must be contiguous, mutually non-overlapping arrays on this context's
        device. ``z`` is read-only and must not overlap them. The returned array
        is ``sed``; its final addition remains queued on the current stream.
        Sweep stopping downloads one scalar and, near the cutoff only, a full
        mobile field to preserve the CPU reduction's decision exactly.
        """
        cp = self._require_active()
        if self._emit is None or self._gather is None:
            raise RuntimeError("GPU execution context is not active")
        geometry = self.geometry(mesh)
        started = perf_counter()
        expected_shape = (mesh.cell_count,)
        arrays = [("z", z, expected_shape), ("sed", sed, expected_shape),
                  ("mob", mob, expected_shape)]
        if next_mob is not None:
            arrays.append(("next_mob", next_mob, expected_shape))
        if edges is not None:
            arrays.append(("edges", edges, (mesh.cell_count, 3)))
        spans = []
        for name, array, shape in arrays:
            if (not isinstance(array, cp.ndarray) or array.dtype != cp.float64
                    or array.shape != shape or not array.flags.c_contiguous):
                raise ValueError(f"CUDA routing {name} must be a contiguous float64 device array of shape {shape}")
            if array.device.id != self.device_id:
                raise ValueError(f"CUDA routing {name} is on the wrong device")
            start = int(array.data.ptr)
            end = start + int(array.nbytes)
            if any(start < other_end and other_start < end for other_start, other_end in spans):
                raise ValueError("CUDA routing device buffers must not overlap")
            spans.append((start, end))
        if next_mob is None:
            next_mob = cp.empty_like(mob)
        if edges is None:
            edges = cp.empty((mesh.cell_count, 3), dtype=cp.float64)
        grid = ((mesh.cell_count + 255) // 256,)
        block = (256,)
        for _ in range(max(int(getattr(params, "routing_sweeps")), 0)):
            self._emit(
                grid,
                block,
                (
                    z,
                    geometry.neighbors,
                    mob,
                    sed,
                    edges,
                    np.int32(mesh.cell_count),
                    np.float64(sea_level_m),
                    np.float64(getattr(params, "land_deposition_fraction_per_sweep")),
                    np.float64(getattr(params, "basin_deposition_fraction_per_sweep")),
                ),
            )
            self._gather(
                grid,
                block,
                (edges, geometry.incoming, next_mob, np.int32(mesh.cell_count)),
            )
            mob, next_mob = next_mob, mob
            total = float(cp.sum(mob).get())
            self.routing_scalar_transfer_bytes += np.dtype(np.float64).itemsize
            self.dynamic_transfer_bytes += np.dtype(np.float64).itemsize
            # Only the exact CPU reduction can decide a near-threshold case.
            if total <= 4.0e-12:
                threshold_host = mob.get()
                self.routing_threshold_transfer_bytes += int(threshold_host.nbytes)
                self.dynamic_transfer_bytes += int(threshold_host.nbytes)
                total = float(np.sum(threshold_host))
            if total <= 1.0e-12:
                break
        sed += mob
        self.routing_seconds += perf_counter() - started
        self.routing_calls += 1
        self.routing_cells += mesh.cell_count
        return sed

    def paint_volcanic_arcs(self, mesh: SphereMesh, cell_plate: np.ndarray,
                            radius_km: float, batches: list[dict[str, object]]) -> np.ndarray:
        """Paint independent arc fronts with one CUDA thread per target cell.

        Maxima are deterministic, but CUDA ``acos``/``exp`` are not byte-identical
        to NumPy.  Callers must opt into this tolerance-based kernel explicitly.
        """
        if not self.arc_painting:
            raise RuntimeError("GPU volcanic-arc painting is not enabled")
        cp = self._require_active()
        if self._paint_arcs is None:
            raise RuntimeError("GPU execution context is not active")
        started = perf_counter()
        geometry = self.geometry(mesh)
        if geometry.centroids is None:
            centroids = np.asarray(mesh.centroids, dtype=np.float64)
            geometry.centroids = cp.asarray(centroids)
            geometry.static_bytes += int(centroids.nbytes)
            self.static_transfer_bytes += int(centroids.nbytes)

        nonempty = [batch for batch in batches if len(batch["centers"])]
        if not nonempty:
            self.arc_calls += 1
            self.arc_seconds += perf_counter() - started
            return np.zeros(mesh.cell_count, dtype=np.float64)
        centers = np.concatenate([
            np.asarray(batch["centers"], dtype=np.float64) for batch in nonempty
        ])
        plates = np.concatenate([
            np.asarray(batch["plates"], dtype=np.int32) for batch in nonempty
        ])
        amplitudes = np.concatenate([
            np.asarray(batch["amplitudes"], dtype=np.float64) for batch in nonempty
        ])
        sigma = np.concatenate([
            np.full(len(batch["centers"]), float(batch["sigma_km"]), dtype=np.float64)
            for batch in nonempty
        ])
        minimum_dot_parts = []
        for batch in nonempty:
            theta = min(float(batch["outer_km"]) / max(float(radius_km), 1.0e-9), np.pi)
            chord = 2.0 * np.sin(0.5 * theta) + 1.0e-12
            minimum_dot_parts.append(
                np.full(len(batch["centers"]), 1.0 - 0.5 * chord * chord, dtype=np.float64)
            )
        minimum_dot = np.concatenate(minimum_dot_parts)
        owner = np.asarray(cell_plate, dtype=np.int32)
        if owner.shape != (mesh.cell_count,):
            raise ValueError("GPU volcanic-arc owner field must match mesh cell count")

        device_owner = cp.asarray(owner)
        device_centers = cp.asarray(centers)
        device_plates = cp.asarray(plates)
        device_amplitudes = cp.asarray(amplitudes)
        device_sigma = cp.asarray(sigma)
        device_minimum_dot = cp.asarray(minimum_dot)
        device_field = cp.empty(mesh.cell_count, dtype=cp.float64)
        grid = ((mesh.cell_count + 255) // 256,)
        block = (256,)
        self._paint_arcs(
            grid,
            block,
            (
                geometry.centroids,
                device_owner,
                device_centers,
                device_plates,
                device_amplitudes,
                device_sigma,
                device_minimum_dot,
                device_field,
                np.int32(mesh.cell_count),
                np.int32(len(centers)),
                np.float64(radius_km),
            ),
        )
        result = device_field.get()
        self.arc_calls += 1
        self.arc_tasks += len(centers)
        self.arc_seconds += perf_counter() - started
        self.arc_dynamic_transfer_bytes += int(
            owner.nbytes + centers.nbytes + plates.nbytes + amplitudes.nbytes
            + sigma.nbytes + minimum_dot.nbytes + result.nbytes
        )
        return result

    def report(self) -> dict[str, object]:
        if self._surface_workspace is not None:
            self.surface_report = self._surface_workspace.report()
        return {
            "backend": "cuda",
            "device_id": self.device_id,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "cupy_version": self.cupy_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "float_dtype": "float64",
            "deterministic_order": True,
            "fused_multiply_add": False,
            "arc_painting_enabled": self.arc_painting,
            "surface_pipeline_enabled": self.surface_pipeline,
            "byte_exact_kernels": ["sediment_routing"]
                                  + (["surface_pipeline"] if self.surface_pipeline else []),
            "tolerance_kernels": ["volcanic_arc_painting"] if self.arc_painting else [],
            "routing_calls": self.routing_calls,
            "routing_cells": self.routing_cells,
            "routing_seconds": self.routing_seconds,
            "routing_seconds_scope": "host wall time; resident final addition is asynchronous",
            "routing_scalar_transfer_bytes": self.routing_scalar_transfer_bytes,
            "routing_threshold_transfer_bytes": self.routing_threshold_transfer_bytes,
            "static_transfer_bytes": self.static_transfer_bytes,
            "dynamic_transfer_bytes": self.dynamic_transfer_bytes,
            "arc_calls": self.arc_calls,
            "arc_tasks": self.arc_tasks,
            "arc_seconds": self.arc_seconds,
            "arc_dynamic_transfer_bytes": self.arc_dynamic_transfer_bytes,
            "cached_meshes": len(self._meshes),
            "surface_pipeline": self.surface_report,
            "surface_pipeline_report_scope": "current or most recent mesh workspace; resets when mesh identity changes",
        }


def current_execution() -> GpuExecution | None:
    return _active


__all__ = ["GpuExecution", "GpuUnavailableError", "current_execution"]

"""Experimental CUDA conjugate-gradient solve for spherical flexure.

This module is a measured candidate, not production dispatch.  It deliberately
uses the same CPU-built CSR operator and material fields as ``flexure.py`` while
moving repeated matrix-vector products and CG vector updates to CUDA.  Keeping
it separate makes numerical acceptance an explicit decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .flexure import (
    FlexureDiagnostics,
    FlexureParameters,
    _geometry_operators,
    effective_elastic_thickness_km,
    flexural_parameter_km,
    flexural_rigidity_nm,
)
from .lithosphere import LithosphereState
from .mesh import SphereMesh


CUDA_SOURCE = r'''
extern "C" __global__ void csr_matvec(
    const int* indptr, const int* indices, const double* data,
    const double* x, double* out, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n) return;
    double total = 0.0;
    for (int j = indptr[row]; j < indptr[row + 1]; ++j)
        total += data[j] * x[indices[j]];
    out[row] = total;
}
'''


@dataclass
class GpuCgReport:
    iterations: int
    converged: bool
    solve_seconds: float
    relative_residual: float
    static_transfer_bytes: int
    dynamic_transfer_bytes: int


class GpuFlexureSolver:
    """Reusable CUDA operator for one immutable mesh and physical radius."""

    def __init__(self, execution: Any, mesh: SphereMesh, radius_km: float) -> None:
        if execution.cp is None:
            raise RuntimeError("GPU execution context is not active")
        self.execution = execution
        self.cp = execution.cp
        self.mesh = mesh
        self.radius_km = float(radius_km)
        matrix, areas = _geometry_operators(mesh, self.radius_km)
        matrix = matrix.tocsr(copy=True)
        matrix.sort_indices()
        self.host_matrix = matrix
        self.host_areas = np.asarray(areas, dtype=np.float64)
        self.indptr = self.cp.asarray(matrix.indptr, dtype=self.cp.int32)
        self.indices = self.cp.asarray(matrix.indices, dtype=self.cp.int32)
        self.data = self.cp.asarray(matrix.data, dtype=self.cp.float64)
        self.areas = self.cp.asarray(self.host_areas)
        self.module = self.cp.RawModule(
            code=CUDA_SOURCE,
            options=("--std=c++17", "--fmad=false"),
        )
        self.kernel = self.module.get_function("csr_matvec")
        self.grid = ((mesh.cell_count + 255) // 256,)
        self.block = (256,)
        self.static_transfer_bytes = int(
            matrix.indptr.nbytes + matrix.indices.nbytes + matrix.data.nbytes + self.host_areas.nbytes
        )
        self.last_report: GpuCgReport | None = None

    def matvec(self, vector: Any, q: Any, restoring: float,
               first: Any, second: Any, result: Any) -> Any:
        n = np.int32(self.mesh.cell_count)
        self.kernel(self.grid, self.block, (self.indptr, self.indices, self.data, vector, first, n))
        first *= q
        self.kernel(self.grid, self.block, (self.indptr, self.indices, self.data, first, second, n))
        result[...] = self.areas * vector + second / restoring
        return result

    def solve(self, rhs_source: np.ndarray, q_host: np.ndarray, restoring: float,
              diagonal_host: np.ndarray, rtol: float, maxiter: int) -> tuple[np.ndarray, GpuCgReport]:
        cp = self.cp
        h_host = np.asarray(rhs_source, dtype=np.float64)
        q_host = np.asarray(q_host, dtype=np.float64)
        diagonal_host = np.asarray(diagonal_host, dtype=np.float64)
        expected = (self.mesh.cell_count,)
        if h_host.shape != expected or q_host.shape != expected or diagonal_host.shape != expected:
            raise ValueError("GPU flexure vectors must match mesh cell count")

        started = perf_counter()
        h = cp.asarray(h_host)
        q = cp.asarray(q_host)
        diagonal = cp.asarray(diagonal_host)
        right = self.areas * h
        x = h.copy()
        first = cp.empty_like(h)
        second = cp.empty_like(h)
        product = cp.empty_like(h)
        self.matvec(x, q, restoring, first, second, product)
        residual = right - product
        z = residual / diagonal
        direction = z.copy()
        rho = cp.dot(residual, z)
        right_norm = float(cp.sqrt(cp.dot(right, right)).get())
        residual_norm = float(cp.sqrt(cp.dot(residual, residual)).get())
        threshold = max(float(rtol) * right_norm, 0.0)
        iterations = 0
        converged = residual_norm <= threshold
        while not converged and iterations < int(maxiter):
            self.matvec(direction, q, restoring, first, second, product)
            alpha = rho / cp.dot(direction, product)
            x += alpha * direction
            residual -= alpha * product
            iterations += 1
            residual_norm = float(cp.sqrt(cp.dot(residual, residual)).get())
            converged = residual_norm <= threshold
            if converged:
                break
            z = residual / diagonal
            next_rho = cp.dot(residual, z)
            direction *= next_rho / rho
            direction += z
            rho = next_rho
        result = x.get()
        elapsed = perf_counter() - started
        report = GpuCgReport(
            iterations=iterations,
            converged=converged,
            solve_seconds=elapsed,
            relative_residual=float(residual_norm / max(right_norm, 1.0e-300)),
            static_transfer_bytes=self.static_transfer_bytes,
            dynamic_transfer_bytes=int(
                h_host.nbytes + q_host.nbytes + diagonal_host.nbytes + result.nbytes
            ),
        )
        self.last_report = report
        return result, report


def solve_flexural_response_gpu(
    execution: Any,
    solver: GpuFlexureSolver,
    mesh: SphereMesh,
    state: LithosphereState,
    local_target_m: np.ndarray,
    radius_km: float,
    gravity_m_s2: float,
    params: FlexureParameters,
) -> tuple[np.ndarray, FlexureDiagnostics, np.ndarray, np.ndarray, GpuCgReport]:
    """GPU candidate with CPU-identical coefficient construction."""
    if solver.execution is not execution or solver.mesh is not mesh or solver.radius_km != float(radius_km):
        raise ValueError("GPU flexure solver does not match this execution, mesh or radius")
    h = np.asarray(local_target_m, dtype=np.float64)
    if h.shape != (mesh.cell_count,):
        raise ValueError("local_target_m must match mesh cell count")
    elastic = effective_elastic_thickness_km(state, params)
    flexural_parameter = flexural_parameter_km(elastic, gravity_m_s2, params)
    if not bool(params.enabled):
        report = GpuCgReport(0, True, 0.0, 0.0, solver.static_transfer_bytes, 0)
        diagnostics = FlexureDiagnostics(
            mean_elastic_thickness_km=float(np.mean(elastic)),
            min_elastic_thickness_km=float(np.min(elastic)),
            max_elastic_thickness_km=float(np.max(elastic)),
            mean_flexural_parameter_km=float(np.mean(flexural_parameter)),
            area_mean_source_m=float(np.mean(h)),
            area_mean_response_m=float(np.mean(h)),
        )
        return h.copy(), diagnostics, elastic, flexural_parameter, report

    matrix, areas = _geometry_operators(mesh, radius_km)
    rigidity = flexural_rigidity_nm(elastic, params)
    q = rigidity / np.maximum(areas, 1.0e-30)
    restoring = max(
        float(params.restoring_density_contrast_kg_m3) * float(gravity_m_s2),
        1.0e-12,
    )
    diagonal_b = np.asarray(matrix.multiply(matrix) @ q).ravel()
    diagonal = areas + diagonal_b / restoring
    response, report = solver.solve(
        h,
        q,
        restoring,
        diagonal,
        float(params.cg_rtol),
        int(params.cg_maxiter),
    )
    correction = response - h
    mean_source = float(np.sum(areas * h) / np.sum(areas))
    mean_response = float(np.sum(areas * response) / np.sum(areas))
    diagnostics = FlexureDiagnostics(
        mean_elastic_thickness_km=float(np.mean(elastic)),
        min_elastic_thickness_km=float(np.min(elastic)),
        max_elastic_thickness_km=float(np.max(elastic)),
        mean_flexural_parameter_km=float(np.mean(flexural_parameter)),
        max_abs_flexural_correction_m=float(np.max(np.abs(correction))),
        rms_flexural_correction_m=float(np.sqrt(np.mean(correction * correction))),
        cg_iterations=int(report.iterations),
        cg_converged=bool(report.converged),
        area_mean_source_m=mean_source,
        area_mean_response_m=mean_response,
    )
    return response, diagnostics, elastic, flexural_parameter, report


__all__ = ["GpuCgReport", "GpuFlexureSolver", "solve_flexural_response_gpu"]

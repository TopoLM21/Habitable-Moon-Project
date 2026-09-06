"""Resident FP64 surface-material pipeline with explicit host boundaries.

The CPU prepares conservative advection/material inputs. Erosion, reworking,
routing, basin spill and relief response share reusable device buffers. A single
packed download supplies new state and the original NumPy budget reductions.
No device result is committed to model state until every phase has completed.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np


CUDA_SOURCE = r'''
extern "C" __global__ void erode_rework(
    const double* input, const int* neighbors, double* output, double* work,
    int n, double erosion, double multiplier, double max_remove, double rework) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const double* z = input;
    double advected = input[n+i], area = input[2*n+i];
    double cf = input[3*n+i], cv = input[4*n+i], eros = input[5*n+i];
    double mean = (z[neighbors[3*i]] + z[neighbors[3*i+1]]) + z[neighbors[3*i+2]];
    mean /= 3.0;
    double excess = fmax(z[i] - mean, 0.0);
    double remove = fmin(((erosion * multiplier) * excess) * eros, max_remove);
    if (!(z[i] > 0.0 && cf > 1e-12)) remove = 0.0;
    double requested = (area * cf) * (remove / 1000.0);
    double removed = fmin(requested, cv);
    double new_cv = fmax(cv - removed, 0.0);
    double eff = cf > 1e-12 ? new_cv / fmax(area * cf, 1e-12) : 0.0;
    double slope = fmin(fmax(excess / 500.0, 0.0), 1.0);
    double fraction = fmin(fmax(rework * slope, 0.0), 0.65);
    double reworked = advected * fraction;
    output[i] = new_cv;
    output[n+i] = eff;
    output[2*n+i] = removed;
    output[3*n+i] = reworked;
    output[4*n+i] = advected - reworked; // stationary, then routed, then final
    work[i] = removed + reworked;       // mobile ping-pong buffer
    work[2*n+i] = requested;
    work[3*n+i] = remove;
}

extern "C" __global__ void spill_emit(
    const double* input, const double* output, const int* neighbors,
    double* edges, double* own, int n, double limit, double fraction) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const double* z = input;
    const double* area = input + 2*n;
    const double* sed = output + 4*n;
    own[i] = 0.0;
    for (int k=0; k<3; ++k) edges[3*i+k] = 0.0;
    double thickness = sed[i] / fmax(area[i], 1e-30);
    if (!(thickness > limit)) return;
    double excess = ((thickness - limit) * area[i]) * fraction;
    if (!(excess > 0.0)) return;
    double scores[3];
    double total = 0.0;
    for (int k=0; k<3; ++k) {
        int j = neighbors[3*i+k];
        double thj = sed[j] / fmax(area[j], 1e-30);
        double score = fmax(z[i]-z[j],0.0) + 250.0*fmax(thickness-thj,0.0);
        scores[k] = score > 1e-12 ? score : 0.0;
        total += scores[k];
    }
    if (!(total > 0.0)) return;
    own[i] = excess;
    for (int k=0; k<3; ++k)
        if (scores[k] > 0.0) edges[3*i+k] = excess * (scores[k]/fmax(total,1e-30));
}

extern "C" __global__ void spill_gather(
    const int* incoming, const double* edges, const double* own,
    double* output, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double transfer = 0.0;
    bool applied_own = false;
    // The CPU interleaves a source's own subtraction and its neighbor scatter.
    // Insert our subtraction between incoming sources smaller/larger than i.
    for (int k=0; k<3; ++k) {
        int edge = incoming[3*i+k];
        if (!applied_own && edge/3 > i) {
            if (own[i] > 0.0) transfer -= own[i];
            applied_own = true;
        }
        if (edges[edge] > 0.0) transfer += edges[edge];
    }
    if (!applied_own && own[i] > 0.0) transfer -= own[i];
    output[4*n+i] = fmax(output[4*n+i] + transfer, 0.0);
}

extern "C" __global__ void relief_response(
    const double* input, const double* work, double* output, int n, double net) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double requested = work[2*n+i], remove = work[3*n+i];
    double removed = output[2*n+i], cf = input[3*n+i];
    double mean_remove = requested > 1e-30 ? (remove * (removed/requested)) * cf : 0.0;
    double delta = (1000.0 * (output[4*n+i] - input[n+i])) / fmax(input[2*n+i],1e-30);
    output[5*n+i] = (input[i] - mean_remove) + net * delta;
}
'''


class SurfaceWorkspace:
    """One mesh's scratch storage; refreshed input values on every call."""

    def __init__(self, execution, mesh):
        self.execution, self.mesh = execution, mesh
        cp = execution.cp
        self.geometry = execution.geometry(mesh)
        self.input = cp.empty((6, mesh.cell_count), dtype=cp.float64)
        self.output = cp.empty_like(self.input)
        self.work = cp.empty((5, mesh.cell_count), dtype=cp.float64)
        self.edges = cp.empty((mesh.cell_count, 3), dtype=cp.float64)
        self.module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++17", "--fmad=false"))
        self.erode = self.module.get_function("erode_rework")
        self.spill_emit = self.module.get_function("spill_emit")
        self.spill_gather = self.module.get_function("spill_gather")
        self.relief = self.module.get_function("relief_response")
        self.calls = 0
        self.host_upload_bytes = self.host_download_bytes = 0
        self.pipeline_seconds = 0.0

    def calculate(self, z, advected, areas, cf, cv, dt, params, eros, sea):
        from .sediment import sediment_net_surface_factor

        self.execution._require_active()
        started = perf_counter()
        # All mutable inputs are refreshed, including after changes of plate IDs.
        packed = np.stack([np.asarray(x, dtype=np.float64)
                           for x in (z, advected, areas, cf, cv, eros)])
        if packed.shape != self.input.shape:
            raise ValueError("GPU surface fields must match mesh cell count")
        if not np.all(np.isfinite(packed)):
            raise ValueError("GPU surface fields must be finite")
        scalar_values = [dt, sea, *[getattr(params, name) for name in params.__dataclass_fields__]]
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("GPU surface parameters must be finite")
        self.input.set(packed)
        self.host_upload_bytes += packed.nbytes
        n = np.int32(self.mesh.cell_count)
        grid, block = ((int(n) + 255) // 256,), (256,)
        erosion = min(float(params.erosion_diffusion_per_myr)*float(dt),
                      float(params.max_erosion_fraction_per_step))
        self.erode(grid, block, (self.input, self.geometry.neighbors, self.output, self.work, n,
                   np.float64(erosion), np.float64(params.bedrock_volume_multiplier),
                   np.float64(float(params.max_bedrock_erosion_km_per_step)*1000.0),
                   np.float64(float(params.sediment_reworking_rate_per_myr)*float(dt))))
        # No full-array transfers between these dependent numerical phases.
        self.execution.route_mobile_device(
            self.mesh, self.input[0], self.output[4], self.work[0], params, sea,
            next_mob=self.work[1], edges=self.edges)
        limit = max(float(params.burial_soft_limit_km), 0.0)
        fraction = float(np.clip(params.burial_spill_fraction_per_step, 0.0, 1.0))
        if limit > 0.0 and fraction > 0.0:
            self.spill_emit(grid, block, (self.input, self.output, self.geometry.neighbors,
                            self.edges, self.work[4], n, np.float64(limit), np.float64(fraction)))
            self.spill_gather(grid, block, (self.geometry.incoming, self.edges, self.work[4], self.output, n))
        self.relief(grid, block, (self.input, self.work, self.output, n,
                                 np.float64(sediment_net_surface_factor(params))))
        # Detached host storage: later workspace reuse cannot mutate a checkpoint.
        result = self.output.get()
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("GPU surface calculation produced non-finite fields")
        self.host_download_bytes += result.nbytes
        self.calls += 1
        self.pipeline_seconds += perf_counter() - started
        return result

    def report(self):
        return {
            "calls": self.calls,
            "buffer_allocations": 1,
            "workspace_bytes": sum(x.nbytes for x in (self.input, self.output, self.work, self.edges)),
            "host_upload_bytes": self.host_upload_bytes,
            "host_download_bytes": self.host_download_bytes,
            "pipeline_seconds": self.pipeline_seconds,
            "phases": ["erosion", "reworking", "routing", "basin_spill", "relief_response"],
            "residence_scope": "surface block within one time step",
            "diagnostic_reductions": "NumPy on packed final outputs",
        }

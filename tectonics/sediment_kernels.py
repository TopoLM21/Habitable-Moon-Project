"""Array-based candidate kernels; not enabled by the production model by default."""
from __future__ import annotations

import numpy as np


def route_mobile_batched(mesh, elevation_m, stationary, mobile, params, sea_level_m):
    """Same ordered source-to-neighbour fluxes as sediment._route_mobile.

    Independent cell arithmetic is batched. Scatter additions still follow the
    original ascending-source / original-neighbour order, without parallel atomics
    or fast-math. Sweeps remain sequential because each consumes the previous one.
    """
    z = np.asarray(elevation_m, dtype=np.float64)
    sed = np.asarray(stationary, dtype=np.float64).copy()
    mob = np.asarray(mobile, dtype=np.float64).copy()
    neighbors = np.asarray(mesh.neighbors, dtype=np.int32)
    if neighbors.ndim != 2 or neighbors.shape[1] != 3:
        raise ValueError("Batched routing requires three neighbours per triangular cell")
    for _ in range(max(int(params.routing_sweeps), 0)):
        next_mob = np.zeros_like(mob)
        active = np.flatnonzero(mob > 0.0)
        nbs = neighbors[active]
        drops = z[active, None] - z[nbs]
        lower = drops > 1e-9
        has_downhill = np.any(lower, axis=1)
        sinks = active[~has_downhill]
        sed[sinks] += mob[sinks]
        flowing = active[has_downhill]
        ids = nbs[has_downhill]
        valid = lower[has_downhill]
        weights = np.where(valid, drops[has_downhill], 0.0)
        dep = np.where(z[flowing] <= float(sea_level_m),
                       float(params.basin_deposition_fraction_per_sweep),
                       float(params.land_deposition_fraction_per_sweep))
        dep = np.clip(dep, 0.0, 1.0)
        values = mob[flowing]
        sed[flowing] += values * dep
        move = values * (1.0 - dep)
        weights /= np.maximum(np.sum(weights, axis=1), 1e-30)[:, None]
        fluxes = move[:, None] * weights
        # Boolean indexing flattens in row-major order, matching the old loops.
        np.add.at(next_mob, ids[valid], fluxes[valid])
        mob = next_mob
        if float(np.sum(mob)) <= 1e-12:
            break
    sed += mob
    return sed

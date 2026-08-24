"""Filled global-map rasterisation helpers for triangular cell data."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _grid_unit_vectors(width: int = 480, height: int = 240):
    lon_edges = np.linspace(-np.pi, np.pi, int(width) + 1)
    lat_edges = np.linspace(-0.5*np.pi, 0.5*np.pi, int(height) + 1)
    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    llon, llat = np.meshgrid(lon, lat)
    clat = np.cos(llat)
    xyz = np.stack((clat*np.cos(llon), clat*np.sin(llon), np.sin(llat)), axis=-1).reshape(-1, 3)
    return lon_edges, lat_edges, xyz


def rasterize_cells(mesh, values, width: int = 480, height: int = 240):
    """Nearest-cell sample onto a regular lon/lat grid for filled Mollweide maps."""
    lon_edges, lat_edges, xyz = _grid_unit_vectors(width, height)
    tree = cKDTree(mesh.centroids)
    _, idx = tree.query(xyz, k=1, workers=-1)
    data = np.asarray(values)[np.asarray(idx, dtype=np.int32)].reshape(int(height), int(width))
    return lon_edges, lat_edges, data

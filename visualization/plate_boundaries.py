"""Plate ownership edges for geographic maps, including inactive contacts."""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import patheffects


def plate_boundary_segments(mesh, cell_plate):
    """Project shared spherical edges without drawing across the map seam."""
    edges = np.asarray(mesh.shared_edges, dtype=np.int64).reshape(-1, 4)
    owners = np.asarray(cell_plate)
    edges = edges[owners[edges[:, 0]] != owners[edges[:, 1]]]
    if not len(edges):
        return []
    ends = mesh.vertices[edges[:, 2:]]
    angles = np.arccos(np.clip(np.sum(ends[:, 0] * ends[:, 1], axis=1), -1, 1))
    count = max(2, int(np.ceil(angles.max() / np.deg2rad(2))) + 1)
    weights = np.linspace(0, 1, count)[None, :, None]
    points = ends[:, :1] * (1 - weights) + ends[:, 1:] * weights
    points /= np.linalg.norm(points, axis=2, keepdims=True)
    lon = np.arctan2(points[:, :, 1], points[:, :, 0])
    lat = np.arcsin(np.clip(points[:, :, 2], -1, 1))
    # Longitude at a pole is undefined; approach it along the edge's meridian.
    for endpoint, neighbor in ((0, 1), (-1, -2)):
        polar = np.linalg.norm(points[:, endpoint, :2], axis=1) < 1e-12
        lon[polar, endpoint] = lon[polar, neighbor]
    coords = np.stack((lon, lat), axis=2)
    segments = np.stack((coords[:, :-1], coords[:, 1:]), axis=2).reshape(-1, 2, 2)
    crossing = np.abs(segments[:, 1, 0] - segments[:, 0, 0]) > np.pi
    result = list(segments[~crossing])
    for first, last in segments[crossing]:
        unwrapped = last[0] - np.sign(last[0] - first[0]) * 2 * np.pi
        if abs(unwrapped - first[0]) < 1e-12:
            result.append(np.array([first, [first[0], last[1]]]))
            continue
        seam = np.copysign(np.pi, first[0])
        fraction = (seam - first[0]) / (unwrapped - first[0])
        seam_lat = first[1] + fraction * (last[1] - first[1])
        result.extend((np.array([first, [seam, seam_lat]]),
                       np.array([[-seam, seam_lat], last])))
    return result


def draw_plate_boundaries(ax, mesh, cell_plate):
    segments = plate_boundary_segments(mesh, cell_plate)
    if not segments:
        return
    lines = LineCollection(segments, colors="#ffedb3", linewidths=0.75, zorder=4)
    lines.set_path_effects([patheffects.Stroke(linewidth=1.5, foreground="#283b48"),
                           patheffects.Normal()])
    ax.add_collection(lines, autolim=False)

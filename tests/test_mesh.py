import numpy as np

from tectonics.mesh import build_icosphere


def test_icosphere_counts_and_area() -> None:
    subdivisions = 3
    mesh = build_icosphere(subdivisions)
    assert mesh.cell_count == 20 * (4 ** subdivisions)
    assert len(mesh.shared_edges) == 3 * mesh.cell_count // 2
    assert all(len(n) == 3 for n in mesh.neighbors)
    assert np.isclose(mesh.areas_unit_sphere.sum(), 4.0 * np.pi, rtol=0.0, atol=1e-12)


def test_centroids_are_unit_vectors() -> None:
    mesh = build_icosphere(2)
    assert np.allclose(np.linalg.norm(mesh.centroids, axis=1), 1.0, atol=1e-14)

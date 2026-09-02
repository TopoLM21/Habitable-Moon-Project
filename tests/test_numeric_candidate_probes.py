from types import SimpleNamespace

import numpy as np
import pytest

from analysis.probe_numeric_candidates import local_proxy_scalar, root_contrast_batch
from tectonics.plume_rifting import _neighbor_root_contrast
from tectonics.subduction_memory import _local_proxy


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_candidate_proxy_exact_for_signs_zero_and_missing_fields(dtype):
    state = SimpleNamespace(
        mantle_lithosphere_thickness_km=np.array([-0., 0., -1., 100., 0.1, 150.], dtype=dtype),
        mantle_lithosphere_density_anomaly_kg_m3=np.array([20., -0., -10., -5., 0.3, 80.], dtype=dtype),
        crust_age_myr=np.array([-0., -1., 0., 35., 45., 65.], dtype=dtype))
    for missing in (None, "mantle_lithosphere_thickness_km", "mantle_lithosphere_density_anomaly_kg_m3"):
        modified = SimpleNamespace(**vars(state))
        if missing:
            setattr(modified, missing, None)
        reference = np.asarray([_local_proxy(modified, face) for face in range(6)])
        candidate = np.asarray([local_proxy_scalar(modified, face) for face in range(6)])
        assert reference.tobytes() == candidate.tobytes()


@pytest.mark.parametrize("neighbors", [((1, 2, 3),) * 4, ((), (0, 2), (0, 1, 3), (2,))])
def test_candidate_contrast_exact_with_fallback_and_missing_roots(neighbors):
    mesh = SimpleNamespace(cell_count=4, neighbors=neighbors)
    for roots in (None, np.array([0., -2., 1.23, 500.], dtype=np.float64)):
        state = SimpleNamespace(mantle_lithosphere_thickness_km=roots)
        assert _neighbor_root_contrast(mesh, state).tobytes() == root_contrast_batch(mesh, state).tobytes()

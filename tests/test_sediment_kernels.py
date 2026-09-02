from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.sediment import SedimentParameters, _route_mobile
from tectonics.sediment_kernels import route_mobile_batched


@pytest.mark.parametrize("sweeps", [0, 1, 3, 8])
@pytest.mark.parametrize("deposition", [(0.1, 0.45), (-0.2, 1.2), (0.0, 0.0), (1.0, 1.0)])
def test_batched_sediments_are_byte_exact_and_preserve_inputs(sweeps, deposition):
    for seed in range(10):
        rng = np.random.default_rng(seed)
        count = 80
        mesh = SimpleNamespace(neighbors=rng.integers(0, count, (count, 3), dtype=np.int32))
        z = rng.normal(0, 3000, count)
        stationary = np.exp(rng.normal(0, 8, count))
        mobile = np.exp(rng.normal(0, 8, count))
        mobile[::5] = 0
        z[::3] = 0  # flat patches and multiple contributors to a destination
        if seed == 0:
            z[:] = 0
        if seed == 1:
            mobile[:] = 0
        if seed == 2:
            mobile *= 1e-25
        params = replace(SedimentParameters(), routing_sweeps=sweeps,
                         land_deposition_fraction_per_sweep=deposition[0],
                         basin_deposition_fraction_per_sweep=deposition[1])
        before = [value.tobytes() for value in (z, stationary, mobile)]
        expected = _route_mobile(mesh, z, stationary, mobile, params, 0.0)
        actual = route_mobile_batched(mesh, z, stationary, mobile, params, 0.0)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes(), f"seed={seed}, sweeps={sweeps}"
        assert before == [value.tobytes() for value in (z, stationary, mobile)]


def test_array_kernel_is_only_used_when_explicitly_enabled(monkeypatch):
    from tectonics.cpu_runtime import CpuExecution
    import tectonics.sediment_kernels as kernels
    mesh = SimpleNamespace(neighbors=np.zeros((1, 3), dtype=np.int32))
    def marker(*args):
        return np.array([123.0])
    monkeypatch.setattr(kernels, "route_mobile_batched", marker)
    inputs = mesh, np.zeros(1), np.ones(1), np.zeros(1), SedimentParameters(), 0.0
    with CpuExecution(1):
        assert _route_mobile(*inputs)[0] == 1.0
    with CpuExecution(1, cell_kernels=True):
        assert _route_mobile(*inputs)[0] == 123.0

"""Contract checks for the isolated stage4 probe, not a production backend."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.probe_gpu_boundary_forces import (
    BoundaryGeometry, GpuOrderedReduction, comparison, make_reference, prepared_cpu,
)
from tectonics.cpu_runtime import CpuExecution
from tectonics.dynamics import DynamicsParameters
from tectonics.kinematics import BoundaryRecord, BoundaryType
from tectonics.lithosphere import CrustType
from tectonics.mesh import build_icosphere


def fixture(seed=372):
    rng = np.random.default_rng(seed)
    mesh = build_icosphere(1)
    n, plates = mesh.cell_count, 5
    owner = rng.integers(0, plates, n, dtype=np.int32)
    state = SimpleNamespace(
        cell_plate=owner, tidal_damage=rng.uniform(-0.2, 2.0, n),
        crust_type=rng.choice([int(CrustType.OCEANIC), int(CrustType.CONTINENTAL)], n),
        crust_age_myr=rng.uniform(-10, 300, n),
        crust_thickness_km=rng.uniform(5, 80, n),
        continental_fraction=rng.uniform(-0.2, 1.2, n),
        mantle_lithosphere_thickness_km=rng.uniform(-10, 180, n),
        mantle_lithosphere_density_anomaly_kg_m3=rng.uniform(-10, 120, n),
        craton_strength=rng.uniform(-0.1, 1.1, n),
    )
    boundaries = []
    for i, (a, b, u, v) in enumerate(mesh.shared_edges):
        if owner[a] == owner[b]:
            continue
        midpoint = mesh.vertices[u] + mesh.vertices[v]
        midpoint /= np.linalg.norm(midpoint)
        boundaries.append(BoundaryRecord(int(a), int(b), int(u), int(v),
            int(owner[a]), int(owner[b]), midpoint, 0.0, 0.0, 0.0, BoundaryType(i % 4)))
    memory = SimpleNamespace(zones={(0, 1): SimpleNamespace(broken_off=True),
                                    (3, 4): SimpleNamespace(broken_off=False)})
    return mesh, state, boundaries, plates, memory


@pytest.mark.parametrize('case', ['mixed', 'legacy', 'discrete', 'no_mantle_flow',
                                  'empty', 'inactive', 'duplicates', 'degenerate', 'ties'])
def test_prepared_boundary_byte_exact(case):
    mesh, state, boundaries, plates, memory = fixture()
    kwargs = {'mantle_flow': object(), 'subduction_memory': memory,
              'thermal_lithosphere_thickness_km': 130.0}
    if case == 'legacy':
        state.mantle_lithosphere_thickness_km = None
        state.mantle_lithosphere_density_anomaly_kg_m3 = None
        state.continental_fraction = None
        state.craton_strength = None
    elif case == 'discrete':
        state.continental_fraction = None
    elif case == 'no_mantle_flow':
        kwargs['mantle_flow'] = None
        kwargs['subduction_memory'] = None
    elif case == 'empty':
        boundaries = []
    elif case == 'inactive':
        boundaries = [replace(b, boundary_type=BoundaryType.INACTIVE) for b in boundaries]
    elif case == 'duplicates':
        boundaries = list(reversed(boundaries)) + boundaries[::2] + boundaries[:5]
    elif case == 'degenerate':
        boundaries = [replace(b, vertex_v=b.vertex_u) if i % 3 == 0 else
                      replace(b, face_b=b.face_a) if i % 3 == 1 else b
                      for i, b in enumerate(boundaries)]
    elif case == 'ties':
        state.crust_type[:] = CrustType.OCEANIC
        state.mantle_lithosphere_thickness_km[:] = 100.0
        state.mantle_lithosphere_density_anomaly_kg_m3[:] = 50.0
    params = DynamicsParameters(slab_buoyancy_exponent=0.731, slab_thermal_exponent=0.273)
    inputs = (state, boundaries, 5287.0, plates, params)
    with CpuExecution(numeric_kernels=True):
        expected = make_reference()(mesh, *inputs, **kwargs)
        geometry = BoundaryGeometry(mesh)
        for _ in range(2):
            actual = prepared_cpu(geometry, *inputs, **kwargs)
            result = comparison(expected, actual)
            assert all(item['byte_exact'] for item in result.values()), result
        # Reusing buffers/geometry must still reload dynamic fields and radius.
        state.tidal_damage *= 0.81
        state.crust_thickness_km += 0.012
        revised = (state, list(reversed(boundaries)), 3200.0, plates, params)
        expected = make_reference()(mesh, *revised, **kwargs)
        result = comparison(expected, prepared_cpu(geometry, *revised, **kwargs))
        assert all(item['byte_exact'] for item in result.values()), result


def test_gpu_ordered_boundary_reduction_byte_exact():
    cp = pytest.importorskip('cupy')
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device unavailable')
    if count < 1:
        pytest.skip('CUDA device unavailable')
    mesh, state, boundaries, plates, memory = fixture(1267)
    inputs = (state, boundaries + list(reversed(boundaries)), 5287.0, plates, DynamicsParameters())
    kwargs = {'mantle_flow': object(), 'subduction_memory': memory}
    with CpuExecution(numeric_kernels=True):
        reference = make_reference()(mesh, *inputs, **kwargs)
        gpu = GpuOrderedReduction()
        actual = gpu.calculate(BoundaryGeometry(mesh), *inputs, **kwargs)
    result = comparison(reference, actual)
    assert all(item['byte_exact'] for item in result.values()), result


def test_geometry_cache_reloads_ownership_orientation_and_midpoint():
    mesh, state, boundaries, plates, memory = fixture(34345)
    geometry = BoundaryGeometry(mesh)
    reference = make_reference()
    params = DynamicsParameters()
    kwargs = {'mantle_flow': object(), 'subduction_memory': memory}
    with CpuExecution(numeric_kernels=True):
        prepared_cpu(geometry, state, boundaries, 5287.0, plates, params, **kwargs)
        # Topology renumbers ownership while all geometry remains unchanged.
        permutation = np.array([3, 1, 4, 0, 2], dtype=np.int32)
        state.cell_plate = permutation[state.cell_plate]
        changed = [replace(b, plate_a=int(permutation[b.plate_a]), plate_b=int(permutation[b.plate_b]))
                   for b in boundaries]
        # Opposite orientations need their own scalar geometry cache entries.
        changed = [replace(b, face_a=b.face_b, face_b=b.face_a,
                           plate_a=b.plate_b, plate_b=b.plate_a,
                           vertex_u=b.vertex_v, vertex_v=b.vertex_u)
                   if i % 2 else b for i, b in enumerate(changed)]
        # A custom midpoint must not hit an old same-index cache entry.
        midpoint = changed[0].midpoint + np.array([0.01, 0.02, -0.01])
        midpoint /= np.linalg.norm(midpoint)
        changed[0] = replace(changed[0], midpoint=midpoint)
        inputs = (state, changed, 3400.0, plates, params)
        result = comparison(reference(mesh, *inputs, **kwargs), prepared_cpu(geometry, *inputs, **kwargs))
        assert all(item['byte_exact'] for item in result.values()), result


@pytest.mark.parametrize('kind', ['float32', 'nonfinite', 'shape'])
def test_prepared_rejects_unsupported_dynamic_arrays(kind):
    mesh, state, boundaries, plates, _ = fixture()
    if kind == 'float32':
        state.tidal_damage = state.tidal_damage.astype(np.float32)
    elif kind == 'nonfinite':
        state.tidal_damage[0] = np.nan
    else:
        state.tidal_damage = state.tidal_damage[:-1]
    geometry = BoundaryGeometry(mesh)
    with pytest.raises(ValueError, match='tidal_damage.*FP64'):
        prepared_cpu(geometry, state, boundaries, 5287.0, plates, DynamicsParameters())
    assert not geometry.edges

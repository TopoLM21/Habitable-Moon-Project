"""Parity checks for isolated GPE candidates; no model dispatch is enabled."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.gpu_dynamics import GpeGeometry, GpuGpeProbe, gpe_prepared_cpu, gpe_reference
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def geometry():
    return GpeGeometry.prepare(build_icosphere(1))


def case(geometry, seed):
    rng = np.random.default_rng(seed)
    n = geometry.mesh.cell_count
    # Broad same-plate regions plus an absent plate exercise the reduction.
    return SimpleNamespace(
        cell_plate=(np.arange(n) // 20 % 3).astype(np.int32),
        crust_type=(rng.random(n) < 0.85).astype(np.int8),
        crust_thickness_km=rng.uniform(20.0, 80.0, n),
    )


@pytest.mark.parametrize("href", [0.0, 40.0, 100.0])
@pytest.mark.parametrize("seed", [17, 92])
def test_prepared_cpu_exact(geometry, href, seed):
    state = case(geometry, seed)
    expected = gpe_reference(geometry.mesh, state, 4, href)
    actual = gpe_prepared_cpu(geometry, state, 4, href)
    for a, b in zip(actual, expected):
        assert a.tobytes() == b.tobytes()


def test_bad_state_rejected(geometry):
    state = case(geometry, 17)
    state.cell_plate[0] = 4
    with pytest.raises(ValueError, match="owner"):
        gpe_prepared_cpu(geometry, state, 4, 40.0)
    state.cell_plate[0] = 0
    state.crust_thickness_km[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        gpe_prepared_cpu(geometry, state, 4, 40.0)


def test_gpe_thresholds_and_normalization_are_exact(geometry):
    state = case(geometry, 17)
    state.cell_plate[:] = 0
    state.crust_type[:] = 1
    state.crust_thickness_km[:] = 40.0
    state.crust_thickness_km[:6] = [
        np.nextafter(40.0, np.inf), np.nextafter(40.0, -np.inf),
        80.0, 60.0, 20.0, 40.0,
    ]
    expected = gpe_reference(geometry.mesh, state, 2, 40.0)
    actual = gpe_prepared_cpu(geometry, state, 2, 40.0)
    assert expected[1][0] > 0.0
    assert expected[1][1] == 0.0
    for left, right in zip(expected, actual):
        assert left.tobytes() == right.tobytes()


def test_measurement_rejects_any_warm_mismatch():
    from analysis.probe_gpu_dynamics import _measure
    reference = (np.ones((1, 3)), np.ones(1))
    outputs = iter([reference, (np.zeros((1, 3)), np.ones(1)), reference])
    result = _measure(lambda: next(outputs), reference, 3)
    assert not result["all_outputs_byte_exact"]
    assert len(result["wall_seconds"]) == 2


@pytest.mark.parametrize("bad_first,bad_warm", [(True, False), (False, True), (True, True)])
def test_speedup_suppressed_for_nonexact_gpu_candidate(bad_first, bad_warm):
    from analysis.probe_gpu_dynamics import _record_speedup
    report = {
        "reference": {"all_outputs_byte_exact": True, "median_seconds": 1.0},
        "gpu": {"all_outputs_byte_exact": not bad_warm, "median_seconds": 0.1},
        "gpu_first_comparison": {"drive": {"byte_exact": not bad_first}},
    }
    _record_speedup(report, "gpu", 5.0)
    assert not report["gpu"]["eligible_for_exact_integration"]
    assert report["gpu"]["loop_speedup"] is None
    assert report["gpu"]["estimated_calls_to_amortize_cold_cost"] is None


@pytest.mark.parametrize("failure", [None, "input_changed", "output_race"])
def test_cpu_probe_main_hashes_and_exclusive_output(geometry, monkeypatch, tmp_path, failure):
    import analysis.probe_gpu_dynamics as probe
    state = case(geometry, 17)
    state.time_myr = 700.0
    checkpoint = SimpleNamespace(state=state, system=SimpleNamespace(plates=(0, 1, 2, 3)))
    config = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "checkpoint"
    output = tmp_path / "report.json"
    hash_calls = []

    def hash_file(path):
        assert isinstance(path, Path)
        hash_calls.append(path)
        if failure == "input_changed" and path == config and hash_calls.count(path) > 1:
            return "changed"
        return "unchanged"

    def load_checkpoint(*_):
        if failure == "output_race":
            output.write_text("belongs to another run", encoding="utf-8")
        return checkpoint

    monkeypatch.setattr(probe, "_hash_file", hash_file)
    monkeypatch.setattr(probe, "load_config", lambda _: {
        "plate_dynamics": {"gpe_reference_thickness_km": 51.0},
    })
    monkeypatch.setattr(probe, "build_prototype", lambda _: SimpleNamespace(mesh=geometry.mesh))
    monkeypatch.setattr(probe, "load_checkpoint", load_checkpoint)
    args = ["--config", str(config), "--checkpoint", str(checkpoint_path),
            "--output", str(output), "--repeat", "1", "--cpu-only"]
    if failure == "output_race":
        with pytest.raises(FileExistsError):
            probe.main(args)
        assert output.read_text(encoding="utf-8") == "belongs to another run"
        return
    if failure == "input_changed":
        with pytest.raises(SystemExit, match="changed during probe"):
            probe.main(args)
    else:
        probe.main(args)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["reference_thickness_km"] == 51.0
    assert len(report["hashes_after"]) == 4
    assert report["inputs_unchanged"] is (failure is None)
    candidate = report["cases"][0]["prepared_cpu"]
    assert candidate["eligible_for_exact_integration"] is (failure is None)
    if failure is not None:
        assert candidate["loop_speedup"] is None


def test_gpu_exact_reused_workspace(geometry):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    gpu = GpuGpeProbe(geometry)
    kept = None
    for seed, href, plates in ((17, 40.0, 4), (92, 0.0, 4), (1, 100.0, 3)):
        state = case(geometry, seed)
        state.cell_plate = np.roll(state.cell_plate, seed)
        expected = gpe_reference(geometry.mesh, state, plates, href)
        actual = gpu.calculate(state, plates, href)
        for a, b in zip(actual, expected):
            assert a.tobytes() == b.tobytes()
        if kept is None:
            kept = actual, tuple(a.tobytes() for a in actual)
        assert tuple(a.tobytes() for a in kept[0]) == kept[1]

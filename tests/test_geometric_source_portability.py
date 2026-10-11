"""Canonical mesh reconstruction preserves the packaged Starter source identity."""
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pytest

from tectonics.genesis_starter_continuation import load_starter_source
from tectonics.mesh import build_icosphere


ROOT = Path(__file__).resolve().parents[1]


def test_subdivision_four_mesh_matches_saved_source_without_blas(monkeypatch):
    original_norm = np.linalg.norm

    def require_batched_norm(vector, *args, **kwargs):
        if np.asarray(vector).ndim == 1:
            raise AssertionError("Canonical midpoint normalization must not use BLAS dot")
        return original_norm(vector, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", require_batched_norm)
    mesh = build_icosphere(4)
    fingerprint = hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()
    assert fingerprint == "1993d4ce236f916bc615a318630eb5b5c6d06016ec654b0c8e9ca46c8face944"
    assert np.allclose(np.linalg.norm(mesh.vertices, axis=1), 1., rtol=0., atol=3e-16)


@pytest.mark.parametrize(("directory", "expected_time"), [
    ("results/genesis_runs/starter_20260928_192306_547551", 0.8781219482421875),
    ("analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050/young_context",
     50.87812194824219),
])
def test_packaged_starter_sources_load_with_original_fingerprint(tmp_path, directory, expected_time):
    # Read the checked-in byte-preserving archive, not ignored local run files.
    with zipfile.ZipFile(ROOT / "handoff/cloud_sources.zip") as packed:
        for name in ("parameters.json", "starter_checkpoint.npz"):
            (tmp_path / name).write_bytes(packed.read(f"{directory}/{name}"))
    model, state, _ = load_starter_source(tmp_path / "starter_checkpoint.npz")
    assert model.mesh.cell_count == 5120
    assert state.time_myr == expected_time

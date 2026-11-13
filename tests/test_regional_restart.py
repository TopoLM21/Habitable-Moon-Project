from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.lithosphere import LithosphereState
from tectonics.cratons import CratonParameters


def test_loader_preserves_saved_frozen_memory_when_surface_labels_changed(monkeypatch):
    import run_long_evolution_v124 as runner
    mesh=build_icosphere(1);n=mesh.cell_count
    state=LithosphereState(48.,np.zeros(n,int),np.zeros(n,np.int8),np.ones(n)*100,np.ones(n)*7,np.zeros(n),
        continental_fraction=np.zeros(n),continental_volume_km3=np.zeros(n),
        continental_lithosphere_age_myr=np.ones(n)*900,mantle_depletion_fraction=np.ones(n)*.4,craton_strength=np.ones(n)*.6)
    saved=deepcopy(state)
    cp=SimpleNamespace(state=state,craton_rows=[{'time_myr':48.}])
    monkeypatch.setattr(runner,'_original_load_checkpoint',lambda path,manager:cp)
    monkeypatch.setattr(runner,'_mesh',mesh);monkeypatch.setattr(runner,'_radius_km',5287.)
    monkeypatch.setattr(runner,'_params',CratonParameters(enabled=False))
    restored=runner._load_checkpoint_v124('synthetic',None).state
    for field in ('continental_lithosphere_age_myr','mantle_depletion_fraction','craton_strength'):
        assert np.array_equal(getattr(restored,field),getattr(saved,field))

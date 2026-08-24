import numpy as np
from tectonics.continental import _copy_lithosphere
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system


def test_continental_copy_preserves_late_material_memory():
    mesh=build_icosphere(2)
    system=random_plate_system(mesh,4,17,0.1,0.1,0.2)
    state=initialize_lithosphere(mesh,system,0.2,2)
    state.collision_seam_weakness[:]=np.linspace(0,1,mesh.cell_count)
    state.intraplate_stress[:]=0.37
    state.supercontinent_heat[:]=0.19
    copied=_copy_lithosphere(state)
    assert np.array_equal(copied.collision_seam_weakness,state.collision_seam_weakness)
    assert np.array_equal(copied.intraplate_stress,state.intraplate_stress)
    assert np.array_equal(copied.supercontinent_heat,state.supercontinent_heat)
    assert copied.intraplate_stress is not state.intraplate_stress

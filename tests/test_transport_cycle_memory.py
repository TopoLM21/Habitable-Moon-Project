import numpy as np

from tectonics.continental import ContinentalCycleParameters, ContinentalCycleState, advance_continental_cycle
from tectonics.lithosphere import advance_lithosphere, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.tides import constant_eccentricity
from tectonics.transport import SubgridTransportParameters, initialize_transport_state


def test_felsic_memory_uses_exact_transport_source_map():
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 5, 55, 0.15, 0.25, 0.5)
    state = initialize_lithosphere(mesh, system, 0.25, 3)
    ts = initialize_transport_state(len(system.plates))
    tp = SubgridTransportParameters(min_changed_fraction=0.0, min_p75_cell_spacing_fraction=0.0)
    nxt, _, _, ldiag = advance_lithosphere(
        mesh, system, state, 4.0, 5287.0, 7.12, 47.0, constant_eccentricity(0.00047),
        transport_state=ts, transport_parameters=tp,
    )
    base = np.linspace(0.0, 0.9, mesh.cell_count)
    cycle = ContinentalCycleState(time_myr=0.0, felsic_potential=base.copy())
    # With no boundaries and an enormous decay time, the only change should be
    # the exact source-map advection (continental cells are then zeroed by the
    # cycle as usual, so compare oceanic targets only).
    params = ContinentalCycleParameters(arc_potential_decay_myr=1e30)
    _, new_cycle, _ = advance_continental_cycle(
        mesh, nxt, [], cycle, 4.0, 5287.0, params,
        transport_source_index=ldiag.material_source_index,
    )
    src = ldiag.material_source_index
    ocean = nxt.crust_type == 0
    valid = ocean & (src >= 0)
    assert np.allclose(new_cycle.felsic_potential[valid], base[src[valid]], atol=1e-12)

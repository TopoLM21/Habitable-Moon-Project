"""Potential interfaces do not create independently detached plates at birth."""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_contact import ContactModel, ContactParameters
from tectonics.genesis_coupled import CoupledModel
from test_genesis_contact import _source_bytes


@pytest.mark.parametrize("coupled", [False, True])
def test_fully_cut_mesh_still_has_one_cohesively_connected_shell(tmp_path, coupled):
    source, _, _, _ = _source_bytes(tmp_path)
    parameters = ContactParameters(alignment_degrees=90.)
    model = (CoupledModel(source, contact_parameters=parameters) if coupled
             else ContactModel(source, parameters))
    state = model.initial()
    count = model.source_model.mesh.cell_count
    before = state.cohorts.damage.copy() if coupled else state.interface_damage.copy()
    report = (model.diagnostics(state, model.source.thermal_state, model.source.orbit)
              if coupled else model.diagnostics(state))
    assert report["cut_component_count"] == count
    assert report["component_count"] == count  # historical CSV alias
    assert report["cohesive_component_count"] == 1
    assert report["fully_decohered_seam_count"] == 0
    np.testing.assert_array_equal(state.cohorts.damage if coupled else state.interface_damage, before)
    if coupled:
        # Display aggregates are deliberately not the source of connectivity.
        display_only = replace(state, contact=replace(state.contact,
            interface_damage=np.ones_like(state.contact.interface_damage)))
        report = model.diagnostics(display_only, model.source.thermal_state, model.source.orbit)
        assert report["cohesive_component_count"] == 1


def test_legacy_results_do_not_get_invented_connectivity_counts():
    from visualization.genesis_connectivity import connectivity_text
    assert connectivity_text({"component_count": 108}) == ""
    note = connectivity_text({"cut_component_count": 108, "cohesive_component_count": 1,
                              "two_cell_component_count": 69})
    assert "108" in note and "с учётом сцепления: 1" in note and "69" in note
    assert "не является числом устойчивых плит" in note

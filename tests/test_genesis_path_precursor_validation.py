"""Material ancestry and thermal-context contracts of the onset experiment."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_path_precursor_validation import PrecursorCase, material_support
from tectonics.genesis_coupled import CoupledModel
from tectonics.genesis_path_birth import recover_tied_tractions
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.mesh import build_icosphere
from test_genesis_contact import _source_bytes
from test_genesis_path_dynamics import _same


def _points(mesh):
    value = np.array([[.6, .2, .2], [.2, .5, .3], [.25, .2, .55]])@mesh.vertices[mesh.faces[0]]
    return value/np.linalg.norm(value, axis=1)[:, None]


def test_material_support_is_identity_on_same_geometry():
    mesh = build_icosphere(1)
    points = _points(mesh)
    pulled, faces, weights = material_support(mesh, mesh, points)
    np.testing.assert_allclose(pulled, points, rtol=0, atol=4e-16)
    np.testing.assert_array_equal(faces, 0)
    np.testing.assert_allclose(weights, [[.6, .2, .2], [.2, .5, .3], [.25, .2, .55]], atol=1e-15)


def test_material_support_follows_known_finite_rotation():
    early = build_icosphere(1)
    rotation = Rotation.from_rotvec([.2, -.15, .08])
    late = rebuild_material_mesh(early, rotation.apply(early.vertices))
    original = _points(early)
    pulled, _, _ = material_support(late, early, rotation.apply(original))
    np.testing.assert_allclose(pulled, original, rtol=0, atol=4e-16)


@pytest.mark.parametrize("field,value", [("membrane_established", True),
    ("elastic_strain", np.array([[1e-20, 0, 0]])), ("damage", np.array([1e-20]))])
def test_gate_entry_cannot_erase_existing_stress_or_damage(field, value):
    values = dict(membrane_established=False, elastic_strain=np.zeros((1, 3)), damage=np.zeros(1))
    values[field] = value
    with pytest.raises(ValueError, match="unestablished, unstressed"):
        PrecursorCase(None, SimpleNamespace(**values), None, None, None)


@pytest.mark.parametrize("active,field", [(True, None), (False, "traction_pa"), (False, "fracture_work_j")])
def test_current_section_cannot_rewrite_active_or_loaded_contact(active, field):
    values = {name: np.zeros(1) for name in ("traction_pa", "damage", "max_opening_m",
        "plastic_slip_m", "cumulative_slip_m", "friction_work_j", "viscous_work_j",
        "fracture_work_j", "shear_remainder_j")}
    if field is not None:
        values[field][0] = 1e-20
    state = SimpleNamespace(active_interval=object() if active else None, cohorts=SimpleNamespace(**values))
    with pytest.raises(ValueError, match="entirely tied, zero-work"):
        PrecursorCase.current_section(None, state, None)


def test_thermal_adapter_keeps_parent_memory_and_physical_clocks(tmp_path):
    # Controlled cold geometry exercises the adapter, not a formation-date
    # prediction. The synthetic entry is explicitly stress/damage free.
    payload, source_model, (source, thermal, orbit), _ = _source_bytes(tmp_path, elastic=(0., 0., 0.), active=False)
    coupled = CoupledModel(payload)
    before = replace(source, membrane_established=False, damage=np.zeros_like(source.damage))
    gate = SimpleNamespace(depth_km=coupled.source.depth_m/1000,
        thermal_state=thermal, orbit=orbit, column_enthalpy=source.column_enthalpy,
        water_access=source.water_access, boundary_energy_j=source.boundary_energy_j)
    mesh = source_model.mesh_for(source)
    case = PrecursorCase(source_model, before, gate, mesh, _points(mesh))
    old = deepcopy(case.initial)
    context_before = deepcopy(case.context)
    after, stress, water, context, loading, thermal_loading = case.trial(case.initial, case.context, 1.)
    _same(case.initial, old)
    _same(case.context, context_before)
    np.testing.assert_allclose(after.elastic_strain,
        case.basis.subdivision.tensor(context.elastic_parent, engineering=True), atol=1e-16, rtol=1e-11)
    assert context.thermal.time_myr == context.orbit.time_myr
    assert context.thermal.time_myr == case.context.thermal.time_myr+1e-6
    assert after.elapsed_years == after.last_step_years == 1.
    assert after.accepted_steps == 1 and after.active_interval is None
    np.testing.assert_array_equal(after.cohorts.traction_pa, 0.)
    np.testing.assert_array_equal(stress, np.einsum("fij,fj->fi", loading.elasticity, after.elastic_strain))
    original_state = deepcopy(after)
    current, measured = case.current_section(after, loading)
    grown, grown_state = case.current_section(after, replace(loading, volume_m3=1.5*loading.volume_m3))
    _same(after, original_state)
    np.testing.assert_array_equal(measured.constraint_reaction_n, grown_state.constraint_reaction_n)
    np.testing.assert_array_equal(measured.displacement_m, grown_state.displacement_m)
    np.testing.assert_array_equal(measured.elastic_strain, grown_state.elastic_strain)
    assert measured.bulk_work_j == grown_state.bulk_work_j
    np.testing.assert_allclose(grown.geometry.interface_area_m2,
        1.5*current.geometry.interface_area_m2, rtol=2e-15)
    # With the same reaction and zero prior, growing the section must divide
    # its reconstructed traction by the same factor. This catches recovering
    # the growing shell's force against stale thermal-gate interface areas.
    zero_prior = np.zeros_like(stress)
    before_traction = recover_tied_tractions(current, measured, zero_prior).traction_pa
    after_traction = recover_tied_tractions(grown, grown_state, zero_prior).traction_pa
    np.testing.assert_allclose(after_traction, before_traction/1.5, rtol=1e-12, atol=1e-12)

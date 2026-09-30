"""Independent transactions: real geometry, exact donor ledgers and atomic ties."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_surface import (
    from_fractional_surface, load_geometric_checkpoint, rotate_surface, save_geometric_checkpoint,
)
from tectonics.geometric_transport import UnresolvedPolarityError, advance_geometric_surface
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import intersect_convex, polygon_area


def fixture(*, tied=False, three_owners=False):
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0]>0.).astype(int)
    if three_owners:
        owners[(mesh.centroids[:, 1]>.3) & (mesh.centroids[:, 0]<0.)] = 2
    parcels = tuple(SurfaceParcel(i, int(owner), f"origin:{i}", float(area), 2.*area,
        10.*area, (1e12 if tied else 1e12+i*1e10)*area,
        5. if tied else 5.+i, (("damage", i/20.),)) for i, (area, owner) in enumerate(zip(areas, owners)))
    fractional = FractionalSurfaceState(0., tuple(areas), parcels)
    return mesh, from_fractional_surface(mesh, fractional, 100.)


def birth(cell, plate, area, time, serial):
    return SurfaceParcel(cell, plate, f"birth:{time.hex()}:{serial}", area, 2.*area, 0., 0., 0.,
                         (("damage", 0.),))


def snapshot(state):
    return json.dumps(asdict(state), sort_keys=True, allow_nan=False)


def grouped(fragments, field, *, by_owner=False):
    out = {}
    for fragment in fragments:
        key = fragment.parcel.plate if by_owner else fragment.parcel.material_id
        out.setdefault(key, []).append(getattr(fragment.parcel, field))
    return {key: math.fsum(values) for key, values in out.items()}


def assert_budgets(initial, result):
    for field in EXTENSIVE_FIELDS:
        before = grouped(initial.fragments, field)
        after = grouped(result.state.fragments, field)
        born = grouped(result.births, field)
        lost = grouped((x.fragment for x in result.losses), field)
        for identity in before.keys() | after.keys() | born.keys() | lost.keys():
            expected = before.get(identity, 0.)+born.get(identity, 0.)
            actual = after.get(identity, 0.)+lost.get(identity, 0.)
            assert actual == pytest.approx(expected, rel=5e-12, abs=0.)


@pytest.mark.parametrize("omega", [np.array([[.021, -.033, .046]]*2), np.zeros((2, 3))])
def test_common_motion_has_no_source_or_sink_and_preserves_input(omega):
    mesh, initial = fixture()
    before = snapshot(initial)
    result = advance_geometric_surface(mesh, initial, omega, .25, birth_factory=birth)
    assert result.losses == result.births == ()
    assert snapshot(initial) == before
    assert snapshot(result.state) == snapshot(rotate_surface(initial, omega, .25))
    assert_budgets(initial, result)


def test_convergence_and_gap_birth_preserve_four_budgets_and_true_receivers():
    mesh, initial = fixture()
    omega = np.array([[0., 0., .01], [0., 0., -.01]])
    result = advance_geometric_surface(mesh, initial, omega, .25, birth_factory=birth)
    assert result.losses and result.births
    assert_budgets(initial, result)
    retained = {f.fragment_id: f for f in result.state.fragments}
    originals = {f.fragment_id: f for f in initial.fragments}
    old_ids = set(initial.known_material_ids)
    for loss in result.losses:
        receiver = retained[loss.receiver_fragment_id]
        assert loss.receiver_plate == receiver.parcel.plate != loss.fragment.parcel.plate
        intersection = intersect_convex(loss.fragment.polygon, receiver.polygon)
        assert polygon_area(intersection) == pytest.approx(polygon_area(loss.fragment.polygon), rel=2e-10)
        original = originals[loss.source_fragment_id]
        assert loss.fragment.parcel.material_id == original.parcel.material_id
        assert loss.fragment.parcel.material_fields == original.parcel.material_fields
        assert loss.time_myr == .25
    for newborn in result.births:
        assert newborn.parcel.material_id not in old_ids
        assert newborn.parcel.age_myr == newborn.parcel.cold_mantle_volume_km3 == newborn.parcel.density_excess_mass_kg == 0.
        departed = originals[newborn.parent_fragment_id]
        intersection = intersect_convex(newborn.polygon, departed.polygon)
        assert polygon_area(intersection) == pytest.approx(polygon_area(newborn.polygon), rel=2e-10)
        assert newborn.parcel.plate == departed.parcel.plate
    contacts = {c.contact_id: c for c in result.contacts}
    attached = [loss for loss in result.losses if loss.contact_ids]
    assert attached
    for loss in attached:
        assert loss.attachment_status == "geometric_contact"
        for identity in loss.contact_ids:
            contact = contacts[identity]
            assert {contact.plate_a, contact.plate_b} == {loss.receiver_plate, loss.fragment.parcel.plate}


def test_equal_buoyancy_and_age_cannot_choose_polarity_by_id():
    mesh, initial = fixture(tied=True)
    before = snapshot(initial)
    calls = []
    def forbidden_birth(*args):
        calls.append(args)
        raise AssertionError("Tie must be rejected before births")
    with pytest.raises(UnresolvedPolarityError) as failure:
        advance_geometric_surface(mesh, initial, np.array([[0., 0., .01], [0., 0., -.01]]),
                                  .25, birth_factory=forbidden_birth)
    assert failure.value.overlaps
    assert all(item.area_km2 > 0. for item in failure.value.overlaps)
    assert not calls
    assert snapshot(initial) == before


def test_one_ulp_buoyancy_difference_is_not_a_physical_polarity_choice():
    mesh, initial = fixture(tied=True)
    nominal = 1e12
    initial = replace(initial, fragments=tuple(replace(f, parcel=replace(f.parcel,
        density_excess_mass_kg=f.parcel.area_km2*(np.nextafter(nominal, math.inf)
            if f.parcel.plate == 0 else nominal), specific_properties=())) for f in initial.fragments))
    with pytest.raises(UnresolvedPolarityError):
        advance_geometric_surface(mesh, initial, np.array([[0., 0., .01], [0., 0., -.01]]),
                                  .25, birth_factory=birth)


def test_age_resolves_mass_roundoff_tie_instead_of_sinking_younger_material():
    mesh, initial = fixture(tied=True)
    nominal = 1e12
    initial = replace(initial, fragments=tuple(replace(f, parcel=replace(f.parcel,
        density_excess_mass_kg=f.parcel.area_km2*(np.nextafter(nominal, math.inf)
            if f.parcel.plate == 1 else nominal), age_myr=10. if f.parcel.plate == 0 else 5.,
        specific_properties=())) for f in initial.fragments))
    result = advance_geometric_surface(mesh, initial, np.array([[0., 0., .01], [0., 0., -.01]]),
                                      .25, birth_factory=birth)
    assert result.losses
    assert all(loss.fragment.parcel.plate == 0 and loss.receiver_plate == 1 for loss in result.losses)
    assert_budgets(initial, result)


def test_three_owner_receiver_is_actual_overlapping_fragment_not_cell_occupancy():
    mesh, initial = fixture(three_owners=True)
    # All retained source indices stay valid, but deliberately put every parcel
    # in the same integration bin. This cannot change a physical contact rule.
    same_cell = replace(initial, fragments=tuple(replace(f, parcel=replace(f.parcel, cell=0))
                                                for f in initial.fragments))
    omega = np.array([[0., 0., .01], [0., 0., -.01], [.005, -.007, .003]])
    result = advance_geometric_surface(mesh, same_cell, omega, .25, birth_factory=birth)
    assert result.losses
    assert_budgets(same_cell, result)
    accepted = {f.fragment_id: f for f in result.state.fragments}
    moved = rotate_surface(same_cell, omega, .25)
    original_moved = {f.fragment_id: f for f in moved.fragments}
    for loss in result.losses:
        receiver = accepted[loss.receiver_fragment_id]
        removed_area = polygon_area(loss.fragment.polygon)
        assert polygon_area(intersect_convex(loss.fragment.polygon, receiver.polygon)) == pytest.approx(removed_area, rel=2e-10)
        donor = original_moved[loss.source_fragment_id]
        assert polygon_area(intersect_convex(loss.fragment.polygon, donor.polygon)) == pytest.approx(removed_area, rel=2e-10)
        assert receiver.parcel.plate != donor.parcel.plate


def test_owner_relabeling_and_input_order_preserve_physical_budgets():
    mesh, initial = fixture()
    omega = np.array([[0., 0., .01], [0., 0., -.01]])
    actual = advance_geometric_surface(mesh, initial, omega, .25, birth_factory=birth)
    relabeled = replace(initial, fragments=tuple(replace(f, parcel=replace(f.parcel, plate=1-f.parcel.plate))
                                                for f in reversed(initial.fragments)))
    changed = advance_geometric_surface(mesh, relabeled, omega[::-1], .25, birth_factory=birth)
    for field in EXTENSIVE_FIELDS:
        for left, right in ((actual.state.fragments, changed.state.fragments),
                            ((x.fragment for x in actual.losses), (x.fragment for x in changed.losses)),
                            (actual.births, changed.births)):
            first, second = grouped(left, field, by_owner=True), grouped(right, field, by_owner=True)
            assert first.keys() == {1-key for key in second}
            for key in first:
                assert first[key] == pytest.approx(second[1-key], rel=2e-10, abs=0.)


def test_archive_and_geometry_checkpoint_continue_exactly(tmp_path):
    mesh, initial = fixture()
    omega = np.array([[0., 0., .01], [0., 0., -.01]])
    first = advance_geometric_surface(mesh, initial, omega, .25, birth_factory=birth)
    # The next segment uses common motion so no additional unresolved physical
    # policy is needed for equal-age new material. Existing archive is retained.
    common = np.array([[.002, -.003, .001]]*2)
    direct = advance_geometric_surface(mesh, first.state, common, .25, birth_factory=birth)
    path = tmp_path/"with-passive-archive.json"
    archive = json.loads(json.dumps([asdict(loss) for loss in first.losses]))
    save_geometric_checkpoint(path, first.state, provenance={"removed_material": archive})
    loaded, metadata = load_geometric_checkpoint(path)
    assert metadata["removed_material"] == archive
    resumed = advance_geometric_surface(mesh, loaded, common, .25, birth_factory=birth)
    assert snapshot(resumed.state) == snapshot(direct.state)
    assert resumed.contacts == direct.contacts


def test_four_differential_steps_preserve_archives_budgets_and_exact_restart(tmp_path):
    mesh, initial = fixture()
    initial = replace(initial, fragments=tuple(replace(f, parcel=replace(f.parcel,
        density_excess_mass_kg=f.parcel.area_km2*(1e12 if f.parcel.plate == 1 else 2e12),
        age_myr=5. if f.parcel.plate == 1 else 10., specific_properties=()))
        for f in initial.fragments))
    omega = np.array([[0., 0., 0.], [0., 0., .01]])
    state = initial
    losses, births = [], []
    midway = None
    for index in range(4):
        result = advance_geometric_surface(mesh, state, omega, .1, birth_factory=birth)
        assert_budgets(state, result)
        losses.extend(result.losses)
        births.extend(result.births)
        state = result.state
        if index == 1:
            midway = state
            archived = json.loads(json.dumps([asdict(loss) for loss in losses]))
    assert losses and births and midway is not None
    for field in EXTENSIVE_FIELDS:
        before = grouped(initial.fragments, field)
        after = grouped(state.fragments, field)
        removed = grouped((loss.fragment for loss in losses), field)
        created = grouped(births, field)
        for identity in before.keys() | after.keys() | removed.keys() | created.keys():
            assert after.get(identity, 0.)+removed.get(identity, 0.) == pytest.approx(
                before.get(identity, 0.)+created.get(identity, 0.), rel=5e-12, abs=0.)
    path = tmp_path/"after-two.json"
    save_geometric_checkpoint(path, midway, provenance={"removed_material": archived})
    resumed, metadata = load_geometric_checkpoint(path)
    assert metadata["removed_material"] == archived
    for _ in range(2):
        resumed = advance_geometric_surface(mesh, resumed, omega, .1, birth_factory=birth).state
    assert snapshot(resumed) == snapshot(state)

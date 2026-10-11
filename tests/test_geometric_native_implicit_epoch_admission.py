"""Fresh newborn positive area and immediate entry, without borrowed mechanics."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tectonics.geometric_native_first_entry_material import native_first_entry_material_from_dict
from tectonics.geometric_native_implicit_epoch_admission import (
    certify_native_implicit_epoch_incoming_geometry,
    native_implicit_epoch_incoming_geometry_from_dict,
    native_implicit_epoch_incoming_geometry_to_dict,
)
from tectonics.geometric_native_implicit_material_epoch import initialize_native_implicit_material_epoch
from tectonics.geometric_native_polygons import native_intersection
from tectonics.thermal import SECONDS_PER_MYR


ROOT = Path(__file__).resolve().parents[1]
ACTUAL_SECOND_OMEGA = (
    (0, (-1.7605770254242942e-21, 8.343362481217581e-22, 31.616288107981852)),
    (1, (1.7605762556828817e-21, 1.2036777150937242e-21, 37.810431892018144)),
)
ACTUAL_SECOND_DT_MYR = 1000000.0/SECONDS_PER_MYR


@lru_cache(maxsize=1)
def get_actual_newborn_geometry():
    directory = ROOT/"analysis/geometric_native_first_entry_validation/control_release_02"
    initial = json.loads((directory/"original_force/initial.json").read_bytes())
    material = native_first_entry_material_from_dict(
        json.loads((directory/"material_state.json").read_bytes()),
        expected_original_mechanics_payload=initial["native_underthrust_motion"]["initial_mechanics"])
    epoch = initialize_native_implicit_material_epoch(material)
    return certify_native_implicit_epoch_incoming_geometry(epoch, ACTUAL_SECOND_OMEGA,
        ACTUAL_SECOND_DT_MYR, precision=256)


@pytest.fixture(scope="module")
def actual_geometry():
    return get_actual_newborn_geometry()


def _replay(raw, geometry, **changes):
    arguments = dict(expected_epoch=geometry.source_epoch,
        expected_omega_rad_per_myr=geometry.omega_rad_per_myr,
        expected_dt_myr=geometry.dt_myr)
    arguments.update(changes)
    return native_implicit_epoch_incoming_geometry_from_dict(raw, **arguments)


def test_actual_newborn_receiving_patch_has_strict_positive_area(actual_geometry):
    witness = next(row for row in actual_geometry.positive_contacts
        if row.donor_original_parent_id == "octant:6" and row.donor_material_kind == "original_live"
        and row.receiver_original_parent_id == "octant:2" and row.receiver_material_kind == "hot_birth")
    assert native_intersection(witness.initial_donor_geometry,
        witness.initial_receiver_geometry) is None
    bounds = witness.endpoint_intersection_area_bounds_sr
    assert Fraction(18, 10**47) < bounds.lo < bounds.hi < Fraction(19, 10**47)
    assert bounds.width < bounds.lo/Fraction(10**65)
    negative = {row.negative_component_id for row in actual_geometry.source_epoch.paired_corrections}
    assert all(row.donor_component_id not in negative and row.receiver_component_id not in negative
        for row in actual_geometry.positive_contacts)
    assert actual_geometry.positive_newborn_contact_proved
    assert not actual_geometry.force_admission_authorized


def test_newborn_donor_and_receiver_contact_is_preserved_at_its_actual_birth_clock(actual_geometry):
    hot_pairs = tuple(row for row in actual_geometry.positive_contacts
        if row.donor_material_kind == row.receiver_material_kind == "hot_birth")
    assert len(hot_pairs) >= 2
    assert {row.donor_original_parent_id for row in hot_pairs} >= {"octant:4", "octant:6"}
    assert {row.receiver_original_parent_id for row in hot_pairs} >= {"octant:0", "octant:2"}
    assert all(row.endpoint_intersection_area_bounds_sr.lo > 0 for row in hot_pairs)
    assert all(row.donor_owner != row.receiver_owner for row in actual_geometry.positive_contacts)
    raw = native_implicit_epoch_incoming_geometry_to_dict(actual_geometry)
    assert raw["full_prior_ordered_SO3_and_actual_birth_clock_preserved"]
    assert raw["previous_interval_root_certificates_reused"] is False
    assert raw["complete_all_pairs_exclusion_proved"] is False


def test_initial_edge_flux_proves_no_positive_safe_duration_for_same_euler_vectors(actual_geometry):
    witness = next(row for row in actual_geometry.positive_contacts
        if row.donor_original_parent_id == "octant:6" and row.donor_material_kind == "original_live")
    assert len(witness.immediate_incoming_edges) == 1
    edge = witness.immediate_incoming_edges[0]
    assert edge.edge.length_radians > 0
    assert all(value.lo > 0 for value in edge.endpoint_incoming_flux_bounds_per_myr)
    assert all(value.lo > 0 for _, value in edge.donor_other_support_bounds+edge.receiver_other_support_bounds)
    assert edge.interior_incoming_flux_bounds_per_myr.lo > Fraction(2, 10**19)
    assert actual_geometry.immediate_unsupported_incoming_proved
    raw = native_implicit_epoch_incoming_geometry_to_dict(actual_geometry)
    edge_raw = next(row for row in raw["positive_contacts"]
        if row["donor_original_parent_id"] == "octant:6"
        and row["donor_material_kind"] == "original_live")["immediate_incoming_edges"][0]
    assert edge_raw["safe_positive_duration_with_same_euler_vectors_exists"] is False
    assert edge_raw["isolated_point_area_or_work_assigned"] is False


def test_geometry_replay_is_externally_bound_to_every_new_euler_component(actual_geometry):
    raw = native_implicit_epoch_incoming_geometry_to_dict(actual_geometry)
    assert _replay(raw, actual_geometry) == actual_geometry
    zeroed = tuple((owner, (0., vector[1], vector[2])) for owner, vector in ACTUAL_SECOND_OMEGA)
    with pytest.raises(ValueError, match="independent complete source"):
        _replay(raw, actual_geometry, expected_omega_rad_per_myr=zeroed)
    with pytest.raises(ValueError, match="independent complete source"):
        _replay(raw, actual_geometry, expected_dt_myr=actual_geometry.dt_myr/2.)


def test_geometry_typed_and_dto_mutations_cannot_reseal_authority(actual_geometry):
    raw = native_implicit_epoch_incoming_geometry_to_dict(actual_geometry)
    forged = deepcopy(raw)
    forged["positive_contacts"][0]["endpoint_intersection_area_bounds_sr"]["lo"] = [0, 1]
    with pytest.raises(ValueError, match="independent complete source"):
        _replay(forged, actual_geometry)
    with pytest.raises(ValueError, match="source replay"):
        native_implicit_epoch_incoming_geometry_to_dict(replace(actual_geometry,
            geometry_sha256="0"*64))
    with pytest.raises(ValueError, match="source replay"):
        native_implicit_epoch_incoming_geometry_to_dict(replace(actual_geometry,
            positive_contacts=()))
    forged = deepcopy(raw)
    forged["positive_contacts"][0]["receiver_owner"] = False
    with pytest.raises(ValueError):
        _replay(forged, actual_geometry)


def test_frozen_object_cache_corruption_is_detected_by_an_independent_saved_proof(actual_geometry):
    original = actual_geometry.geometry_sha256
    try:
        object.__setattr__(actual_geometry, "geometry_sha256", "0"*64)
        with pytest.raises(ValueError, match="independently saved complete proof"):
            native_implicit_epoch_incoming_geometry_to_dict(actual_geometry)
    finally:
        object.__setattr__(actual_geometry, "geometry_sha256", original)


@pytest.mark.parametrize("axes", [(), ((),), ((0, (1., 2., 3.)),),
    ((False, (1., 2., 3.)), (1, (1., 2., 3.))),
    ((0, (1, 2., 3.)), (1, (1., 2., 3.)))])
def test_incomplete_or_nonliteral_full_euler_records_are_rejected(actual_geometry, axes):
    with pytest.raises(ValueError, match="every saved literal full Euler"):
        certify_native_implicit_epoch_incoming_geometry(actual_geometry.source_epoch,
            axes, actual_geometry.dt_myr)


def test_complete_receipt_bytes_replay_in_a_fresh_python_process(actual_geometry, tmp_path):
    destination = tmp_path/"fresh_process_geometry.json"
    program = """import json,sys
from pathlib import Path
from tests.test_geometric_native_implicit_epoch_admission import get_actual_newborn_geometry
from tectonics.geometric_native_implicit_epoch_admission import native_implicit_epoch_incoming_geometry_to_dict
Path(sys.argv[1]).write_text(json.dumps(native_implicit_epoch_incoming_geometry_to_dict(get_actual_newborn_geometry()),sort_keys=True,separators=(',',':')))
"""
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-c", program, str(destination)], cwd=ROOT,
        env=environment, check=True, timeout=240)
    expected = json.dumps(native_implicit_epoch_incoming_geometry_to_dict(actual_geometry),
        sort_keys=True, separators=(",", ":")).encode()
    assert destination.read_bytes() == expected

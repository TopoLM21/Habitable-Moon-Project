"""Authentic full-source checks of the active original-contact preparation."""
import copy
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import json
from pathlib import Path
import runpy

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_preparation_contacts as api
from tectonics.geometric_native_common_carrier_source_work import _interval, _fraction

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def context():
    return runpy.run_path(str(ROOT/"analysis/geometric_native_common_preparation_balance_validation/native_control.txt"))["context"]()


@pytest.fixture(scope="module")
def zero(context):
    return api.evaluate_native_common_preparation_contacts(context["contacts_system"],F(0))


@pytest.fixture(scope="module")
def endpoint(context):
    return api.evaluate_native_common_preparation_contacts(context["contacts_system"],F(1))


def overlaps(a,b):
    return max(a.lo,b.lo) <= min(a.hi,b.hi)


def test_authentic_catalogue_has_fifteen_active_coordinates_without_full_mixed_claim(context,zero):
    system=context["contacts_system"]
    assert system.active_original100_indices == tuple(range(6))+tuple(range(92,100))
    assert len(system.original_coordinate_order)==100
    assert len(system.coordinate_order)==len(zero.gradient_bounds)==len(zero.hessian_bounds)==15
    assert all(len(row)==15 for row in zero.hessian_bounds)
    assert system.lambda_coordinate_index==14
    assert zero.full_general_state_mixed_operator_certified is False
    assert zero.new_mixed_force_equilibrium_certified is False
    assert zero.clocks_advanced is False
    assert zero.accepted_new_native_intervals==0


def test_all_original_live_archive_material_and_literal_energies_retained(context,zero):
    system=context["contacts_system"]
    assert (zero.original_atom_count,zero.original_material_point_count,
            zero.original_live_attachment_count,zero.original_archived_attachment_count)==(16,32,24,8)
    assert zero.energy_bounds_j==I(sum(system.literal_source_energies_j,F(0)))
    assert zero.recoverable_energy_bounds_j==zero.energy_bounds_j
    assert zero.fracture_work_bounds_j==I(0)
    for row in map(json.loads,zero.atoms_json):
        assert row["exact_zero_literal_energy_identity_preserved"] is True
        assert _interval(row["scalar_increment_twojet"]["value_bounds"])==I(0)
        assert _interval(row["energy_bounds_j"])==I(_fraction(row["literal_source_energy_j"]))
    assert zero.source_contact_relaxation_lower_bound_j<0


def test_new_zero_owner_internal_block_overlaps_authentic_saved907_atoms(context,zero):
    queries=context["source907"]["evaluation_queries"]
    old=next(q["response"] for q in queries if all(v==[0,1] for v in q["response"]["coordinate_increments"]))
    oldatoms={a["point_id"]:a for a in map(json.loads,old["contact_atoms_json"])}
    for new in map(json.loads,zero.atoms_json):
        prior=oldatoms[new["point_id"]]
        g=dict((i,_interval(v)) for i,v in new["energy_twojet"]["first_derivative_bounds"])
        h=dict(((i,j),_interval(v)) for i,j,v in new["energy_twojet"]["second_derivative_bounds"])
        oldindices=prior["active_coordinate_indices"]
        oldg={i:_interval(v) for i,v in zip(oldindices,prior["gradient_bounds"])}
        oldh={(i,j):_interval(prior["hessian_bounds"][a][b]) for a,i in enumerate(oldindices) for b,j in enumerate(oldindices)}
        for i,oi in enumerate(zero.active_original100_indices):
            assert overlaps(g.get(i,I(0)),oldg.get(oi,I(0)))
            for j,oj in enumerate(zero.active_original100_indices):
                if i<=j:
                    assert overlaps(h.get((i,j),I(0)),oldh.get((oi,oj),I(0)))


def test_actual_endpoint_certifies_twelve_failed_and_four_compressed_atoms(endpoint):
    rows=list(map(json.loads,endpoint.atoms_json))
    assert sum(v["loading"]["branch"]=="fully_failed_open" for v in rows)==12
    assert sum(v["loading"]["branch"]=="intact_or_compressed" for v in rows)==4
    assert endpoint.fracture_work_bounds_j.lo>0
    for row in rows:
        free=_interval(row["loading"]["free_gap_bounds_m"])
        if row["loading"]["branch"]=="fully_failed_open":
            assert free.lo>_fraction(row["loading"]["failure_free_gap_m"])
        else:
            assert free.hi<0
    assert endpoint.proposed_normal_damage_or_fracture_work_applied is False
    assert endpoint.historical_normal_fracture_memory_recovered is False


def test_preparation_orientation_and_internal_cross_terms_are_kept(zero,endpoint):
    assert any(v!=I(0) for v in zero.gradient_bounds[14:])
    assert any(zero.hessian_bounds[i][14]!=I(0) for i in range(6,14))
    assert endpoint.hessian_bounds==tuple(zip(*endpoint.hessian_bounds))
    assert endpoint.simultaneous_owner_rotation_directional_first_derivative_bounds==(I(0),)*3
    for row in map(json.loads,endpoint.atoms_json):
        assert all(_interval(v).lo<=0<=_interval(v).hi for v in row["global_moment_residual_bounds_nm"])
        for work in row["virtual_work"]:
            residual=_interval(work["residual_bounds"])
            assert residual.lo<=0<=residual.hi
        assert row["receiver_nanson_normal"]["second_preparation_original_gradient_curvature_included"] is True
        assert row["receiver_nanson_normal"]["current_area_energy_multiplier_added"] is False


def test_source_restart_requires_full_authorities_and_matching_provenance(context):
    system=context["contacts_system"]
    dto=api.native_common_preparation_contacts_to_dict(system)
    restored=api.native_common_preparation_contacts_from_dict(dto,context["source907"],context["source891"],context["preparations"],
        source895=context["source895"],provenance=system.provenance)
    assert api.native_common_preparation_contacts_to_dict(restored)==dto
    changed=copy.deepcopy(dto);changed["lambda_coordinate_index"]=13
    with pytest.raises(ValueError):
        api.native_common_preparation_contacts_from_dict(changed,context["source907"],context["source891"],context["preparations"],
            source895=context["source895"],provenance=system.provenance)
    with pytest.raises(ValueError):
        api.validate_native_common_preparation_contacts(replace(system,provenance="rebased"))


def test_whole_source_packet_tamper_and_owner_reordering_are_rejected(context):
    source=copy.deepcopy(context["source895"])
    source["cold_material_carrier_system"]["banks"][0]["roof_radial_displacement_m"][0]+=1
    args=(context["source907"],context["source891"],context["preparations"])
    with pytest.raises(ValueError):
        api.bind_native_common_preparation_contacts(*args,source895=source,provenance="tampered")
    with pytest.raises(ValueError):
        api.bind_native_common_preparation_contacts(context["source907"],context["source891"],context["preparations"][::-1],
            source895=context["source895"],provenance="reordered")


def test_complete_endpoint_restart_and_query_transplant_rejection(context,endpoint):
    dto=api.native_common_preparation_contacts_response_to_dict(endpoint)
    restored=api.native_common_preparation_contacts_response_from_dict(dto,system=context["contacts_system"],expected_preparation_fraction=F(1))
    assert api.native_common_preparation_contacts_response_to_dict(restored)==dto
    changed=copy.deepcopy(dto);changed["gradient_bounds"][14]["lo"][0]+=1
    with pytest.raises(ValueError):
        api.native_common_preparation_contacts_response_from_dict(changed,system=context["contacts_system"],expected_preparation_fraction=F(1))
    with pytest.raises(ValueError):
        api.native_common_preparation_contacts_response_from_dict(dto,system=context["contacts_system"],expected_preparation_fraction=F(0))


@pytest.mark.parametrize("parameter",[0.5,True,F(-1,10),F(11,10)])
def test_invalid_preparation_parameter_is_rejected(context,parameter):
    with pytest.raises(ValueError):
        api.evaluate_native_common_preparation_contacts(context["contacts_system"],parameter)


def test_point_state_api_refuses_broad_interval_across_changed_cohesive_branches(context,zero,endpoint):
    assert {row["loading"]["branch"] for row in map(json.loads,zero.atoms_json)}=={"intact_or_compressed"}
    assert "fully_failed_open" in {row["loading"]["branch"] for row in map(json.loads,endpoint.atoms_json)}
    with pytest.raises(ValueError,match="point-state"):
        api.evaluate_native_common_preparation_contacts(context["contacts_system"],I(F(0),F(1,4)))


def test_nonzero_internal_arm_is_convected_with_lambda_cross_curvature(context):
    internal=tuple(F((-1 if i%2 else 1)*(i+1),1000000) for i in range(8))
    response=api.evaluate_native_common_preparation_contacts(context["contacts_system"],F(1,4),internal)
    assert response.internal_coordinates==internal
    assert any(response.hessian_bounds[i][14]!=I(0) for i in range(6,14))
    assert all(row["source_original_area_included_once"] is True for row in map(json.loads,response.atoms_json))


def test_actual_small_preparation_exercises_all_three_source_cohesive_branches(context):
    response=api.evaluate_native_common_preparation_contacts(context["contacts_system"],F(1,100))
    branches={row["loading"]["branch"] for row in map(json.loads,response.atoms_json)}
    assert branches=={"intact_or_compressed","monotone_softening","fully_failed_open"}
    for row in map(json.loads,response.atoms_json):
        residual=_interval(row["loading"]["energy_partition_residual_bounds_j_m2"])
        assert residual.lo<=0<=residual.hi


def test_original_internal_arm_cannot_cross_zero_radius(context):
    with pytest.raises(ValueError,match="positive radius"):
        api.evaluate_native_common_preparation_contacts(context["contacts_system"],F(0),(F(-100000),)*8)


@pytest.mark.parametrize("precision,coefficient",[(96,F(3,2)),(512,F(1,10))])
def test_rigorous_rotation_kernel_closed_form_and_serializable_tail(precision,coefficient):
    # A nilpotent generator has exp(K)=I+K exactly, independently of Taylor
    # implementation. The first case also exercises power-of-two squaring;
    # the second verifies the supported precision512 certificate serialization.
    K=((I(0),I(coefficient),I(0)),(I(0),I(0),I(0)),(I(0),I(0),I(0)))
    Q,proof=api._preparation_matrix_exponential(K,precision=precision)
    for i in range(3):
        for j in range(3):
            exact=F(int(i==j))+(coefficient if (i,j)==(0,1) else F(0))
            assert Q[i][j].lo<=exact<=Q[i][j].hi
    assert proof["tail_ratio"]<1
    assert proof["tail_scaled_matrix_upper"]<=proof["tail_target_upper"]
    assert proof["guard_precision_decimal_digits"]==32
    assert proof["tail_bound_outward_compressed"] is True
    if precision==96:
        assert proof["scaling_squaring_count"]==2
    payload=api._json(api._payload(proof))
    assert json.loads(payload)["output_precision_decimal_digits"]==precision

"""Counterexamples for canonical seed archives versus opaque typed bindings.

These small descriptors exercise provenance guards without producing a seed,
initializing planetary material, restoring a checkpoint or replaying Genesis.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from analysis.genesis_volume_memory_validation import independent_audit_v2 as audit


def reseal_archive(inherited, *, surface=True):
    raw = inherited["finite_seed"]
    if surface:
        inner = raw["surface"]
        inner["embedded_surface_sha256"] = audit.digest(
            {k: v for k, v in inner.items() if k != "embedded_surface_sha256"})
    raw["finite_seed_sha256"] = audit.digest(
        {k: v for k, v in raw.items() if k != "finite_seed_sha256"})


@pytest.fixture
def descriptors():
    definition = dict(original_cohort_id=17, original_face_index=2,
        seed_center_world_m=[1., 2., 3.], seed_radius_m=.4,
        normal_xyz=[0., 0., 1.], offset_m=3.)
    parent = dict(source_checkpoint_sha256="a"*64,
        runtime_state=dict(geometry=dict(source_identity="original-material-source",
            time_myr=.25, vertices_m=[[0., 0., 0.]]*3,
            cohorts=[dict(id=17, face_index=2)])))
    archived = dict(selected_cohort_id=17, seed_center_world_m=[1., 2., 3.],
        seed_radius_m=.4, surface=dict(source_checkpoint_sha256="a"*64,
            source_identity="original-material-source", time_myr=.25,
            source_geometry_bound=True, declared_normal_xyz=[0., 0., 1.],
            declared_offset_m=3., quadrature_order=8, selected_cohort_ids=[17],
            selected_cohort_indices=[0], parent_ndof=18))
    inherited = dict(name="declared-material-seed", seed_definition=definition,
        finite_seed=archived)
    reseal_archive(inherited)
    case = dict(name=inherited["name"], seed_definition=copy.deepcopy(definition),
        finite_seed_binding_sha256="d"*64,
        accepted_interval_reaction=dict(finite_seed_binding_sha256="d"*64,
            embedded_partition_binding_sha256="e"*64),
        embedded_partition=dict(embedded_partition_sha256="e"*64))
    return case, inherited, parent


def test_valid_distinct_hash_domains_keep_all_numerical_functions_unchanged(descriptors):
    case, inherited, parent = descriptors
    assert case["finite_seed_binding_sha256"] != inherited["finite_seed"]["finite_seed_sha256"]
    report = audit.audit_finite_seed_provenance_link(case, inherited, parent)
    assert report["literal_constructor_and_parent_source_fields_exact"]
    assert report["typed_binding_scope"] == "opaque_integrity_link_only_not_independently_reconstructed"
    assert report["canonical_DTO_and_typed_binding_hash_domains_distinct"]
    old = Path(audit.__file__).with_name("independent_audit.py")
    before = {n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(old.read_text()).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(Path(audit.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    assert set(after)-set(before) == {"audit_finite_seed_provenance_link"}
    assert {name for name in before if before[name] != after[name]} == {"audit_fixture"}


def test_mutated_canonical_seed_and_nested_surface_seals_are_rejected(descriptors):
    for target in ("seed", "surface"):
        case, inherited, parent = copy.deepcopy(descriptors)
        if target == "seed":
            inherited["finite_seed"]["seed_radius_m"] = .5
            expected = "finite_seed_sha256 seal differs"
        else:
            inherited["finite_seed"]["surface"]["time_myr"] = .26
            reseal_archive(inherited, surface=False)
            expected = "embedded_surface_sha256 seal differs"
        with pytest.raises(ValueError, match=expected):
            audit.audit_finite_seed_provenance_link(case, inherited, parent)


def test_resealed_selected_material_id_substitution_is_rejected(descriptors):
    case, inherited, parent = descriptors
    inherited["finite_seed"]["selected_cohort_id"] = 18
    reseal_archive(inherited)
    with pytest.raises(ValueError, match="literal constructor selected_cohort_id"):
        audit.audit_finite_seed_provenance_link(case, inherited, parent)


def test_resealed_seed_center_substitution_is_rejected(descriptors):
    case, inherited, parent = descriptors
    inherited["finite_seed"]["seed_center_world_m"][0] = 1.1
    reseal_archive(inherited)
    with pytest.raises(ValueError, match="literal constructor seed_center_world_m"):
        audit.audit_finite_seed_provenance_link(case, inherited, parent)


def test_resealed_seed_radius_substitution_is_rejected(descriptors):
    case, inherited, parent = descriptors
    inherited["finite_seed"]["seed_radius_m"] = .5
    reseal_archive(inherited)
    with pytest.raises(ValueError, match="literal constructor seed_radius_m"):
        audit.audit_finite_seed_provenance_link(case, inherited, parent)


def test_opaque_typed_binding_requires_exact_lowercase_sha256_shape(descriptors):
    for invalid in (None, "", "d"*63, "D"*64, "g"*64):
        case, inherited, parent = copy.deepcopy(descriptors)
        case["finite_seed_binding_sha256"] = invalid
        case["accepted_interval_reaction"]["finite_seed_binding_sha256"] = invalid
        with pytest.raises(ValueError, match="not lowercase SHA256"):
            audit.audit_finite_seed_provenance_link(case, inherited, parent)


def test_reaction_seed_and_partition_link_substitutions_are_rejected(descriptors):
    for key in ("finite_seed_binding_sha256", "embedded_partition_binding_sha256"):
        case, inherited, parent = copy.deepcopy(descriptors)
        case["accepted_interval_reaction"][key] = "f"*64
        with pytest.raises(ValueError, match="integrity link"):
            audit.audit_finite_seed_provenance_link(case, inherited, parent)


def test_resealed_literal_source_and_preserved_definition_substitutions_are_rejected(descriptors):
    substitutions = dict(source_checkpoint_sha256="b"*64,
        source_identity="replacement-source", time_myr=.26,
        source_geometry_bound=False, declared_normal_xyz=[1., 0., 0.],
        declared_offset_m=3.1, quadrature_order=12, selected_cohort_ids=[18],
        selected_cohort_indices=[1], parent_ndof=24)
    for key, replacement in substitutions.items():
        case, inherited, parent = copy.deepcopy(descriptors)
        inherited["finite_seed"]["surface"][key] = replacement
        reseal_archive(inherited)
        with pytest.raises(ValueError, match="literal source "+key):
            audit.audit_finite_seed_provenance_link(case, inherited, parent)
    for target in ("definition", "name"):
        case, inherited, parent = copy.deepcopy(descriptors)
        if target == "definition":
            case["seed_definition"]["seed_radius_m"] = .5
        else:
            case["name"] = "replacement-seed"
        with pytest.raises(ValueError, match="preserved genuine finite-seed"):
            audit.audit_finite_seed_provenance_link(case, inherited, parent)

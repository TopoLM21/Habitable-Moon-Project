"""Independent own-scale work and projected-field forgery checks."""
from copy import deepcopy

import numpy as np
import pytest

from analysis.genesis_volume_clipped_validation.independent_contractions_audit import audit_order_contractions
from analysis.genesis_volume_clipped_validation.independent_bulk_audit import IndependentBulkRefusal
from tectonics.genesis_volume_clipped_bulk import (
    clipped_bulk_contractions, clipped_bulk_operator_to_dict, clipped_bulk_projected_blocks,
    prepare_clipped_bulk_operator,
)
from tectonics.genesis_volume_clipped_domain import prepare_cohort_clipped_domain
from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_operator import isotropic_elasticity


@pytest.fixture(scope="module")
def archive():
    X = np.array(((0., -5., 10.), (5., -5., 10.), (0., 5., 10.)))
    lo, hi, coefficient = .8, .95, .5
    volume = np.linalg.det(X)*(hi**3-lo**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, (1-hi**3)/coefficient, (1-lo**3)/coefficient,
        2*volume, volume, tuple(map(tuple, X)), hi, lo, (), "retained-owner", "retained-material")
    section = prepare_cohort_material_section(cohort, np.zeros((3, 6)), (0., 1., 0.), 0., conical_mass_coefficient=coefficient)
    domain = prepare_cohort_clipped_domain(section, (1., 0., 9.), .25, radial_order=3, angular_order=3, ray_order=3)
    e0, e1 = (np.arange(18).reshape(3, 6)-8)*1e-4, (np.arange(18).reshape(3, 6)[::-1]-7)*2e-4
    op = prepare_clipped_bulk_operator(domain, isotropic_elasticity(3e8, .27), np.eye(3), e0, e1)
    probes = [dict(name="complete_mean", parent_increment_m=(1e-4*np.sin(.37*(np.arange(18)+1))).tolist(),
        enrichment_increment_m=(2e-4*np.cos(.23*(np.arange(18)+1))).tolist(),
        mean_e0=(e0*.7).tolist(), mean_e1=(e1*1.1).tolist(), mean_z_m=(3e-4*np.sin(.19*(np.arange(18)+1))).tolist()),
        dict(name="exact_zero", parent_increment_m=np.zeros(18).tolist(), enrichment_increment_m=np.zeros(18).tolist())]
    basis = np.eye(18)[:, (0, 3, 7, 10)]
    projected = clipped_bulk_projected_blocks(op, basis)
    names = ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n")
    item = dict(bulk_operator=clipped_bulk_operator_to_dict(op, include_points=False),
        declared_probe_contractions=[dict(name=p["name"], contractions=clipped_bulk_contractions(op,
            p["parent_increment_m"], p["enrichment_increment_m"],
            **{k:p[k] for k in ("mean_e0", "mean_e1", "mean_z_m") if k in p})) for p in probes],
        declared_projected_blocks=dict(blocks={k:np.asarray(projected[k], float).tolist() for k in names},
            roundoff_component_bounds={k:np.asarray(projected[k+"_budget"], float).tolist() for k in names},
            added_basis=basis.tolist(), parent_basis=np.eye(18).tolist(),
            physical_contraction_certified=False, available_for_creation_funding=False,
            projection_before_Gram=True))
    return item, dict(section=domain.geometry_source), probes, basis


def test_complete_source_mean_work_and_forces_are_independently_recomputed(archive):
    item, source, probes, basis = archive
    result = audit_order_contractions(item, source, probes, basis)
    assert result["status"] == "PASS" and result["producer_imports"] is False
    assert result["declared_projected_blocks"]["coordinate_count"] == 4
    assert result["complete_signed_work_parts_verified"]
    assert result["physical_contraction_certified"] is False


@pytest.mark.parametrize("term", ["parent_work_j", "cross_parent_work_j", "cross_added_work_j", "self_work_j",
    "stiffness_cross_bilinear_j", "stiffness_self_bilinear_j", "mass_cross_bilinear_kg_m2", "mass_self_bilinear_kg_m2"])
def test_each_signed_work_part_refuses_its_own_corruption(archive, term):
    item, source, probes, basis = archive
    forged = deepcopy(item)
    value = forged["declared_probe_contractions"][0]["contractions"]["terms"][term]
    value["value"] *= 1.1
    with pytest.raises(IndependentBulkRefusal, match="own component differs"):
        audit_order_contractions(forged, source, probes, basis)


def test_positive_floor_cannot_hide_corruption_of_exact_zero_work(archive):
    item, source, probes, basis = archive
    forged = deepcopy(item)
    forged["declared_probe_contractions"][1]["contractions"]["terms"]["self_work_j"]["value"] = 1e-250
    with pytest.raises(IndependentBulkRefusal, match="self_work_j own component differs"):
        audit_order_contractions(forged, source, probes, basis)


def test_invented_roundoff_cannot_launder_finite_order_agreement(archive):
    item, source, probes, basis = archive
    forged = deepcopy(item)
    forged["declared_probe_contractions"][0]["contractions"]["terms"]["self_work_j"]["absolute_roundoff_budget"] = 1e20
    with pytest.raises(IndependentBulkRefusal, match="reported roundoff exceeds"):
        audit_order_contractions(forged, source, probes, basis)


@pytest.mark.parametrize("name", ["stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n"])
def test_source_operative_point_block_corruption_is_refused(archive, name):
    item, source, probes, basis = archive
    forged = deepcopy(item)
    value = np.asarray(forged["declared_projected_blocks"]["blocks"][name])
    index = np.unravel_index(np.abs(value).argmax(), value.shape)
    value[index] *= 1.01
    forged["declared_projected_blocks"]["blocks"][name] = value.tolist()
    with pytest.raises(IndependentBulkRefusal, match="projected .* own component differs"):
        audit_order_contractions(forged, source, probes, basis)


def test_prescribed_accepted_mean_memory_cannot_be_silently_replaced(archive):
    item, source, probes, basis = archive
    wrong_probes = deepcopy(probes)
    wrong_probes[0]["mean_e0"] = np.zeros((3, 6)).tolist()
    wrong_probes[0]["mean_e1"] = np.zeros((3, 6)).tolist()
    with pytest.raises(IndependentBulkRefusal, match="own component differs"):
        audit_order_contractions(item, source, wrong_probes, basis)


def test_false_physical_or_matrix_contraction_claim_is_refused(archive):
    item, source, probes, basis = archive
    forged = deepcopy(item)
    forged["declared_probe_contractions"][0]["contractions"]["projection_before_Gram"] = False
    with pytest.raises(IndependentBulkRefusal, match="before Gram"):
        audit_order_contractions(forged, source, probes, basis)


def test_source_operative_basis_binding_cannot_be_replaced(archive):
    item, source, probes, basis = archive
    forged = deepcopy(item)
    forged["declared_projected_blocks"]["added_basis"][0][0] *= 1.01
    with pytest.raises(IndependentBulkRefusal, match="projected basis differs"):
        audit_order_contractions(forged, source, probes, basis)

from fractions import Fraction as F
import numpy as np
import pytest
import json
from pathlib import Path
from analysis.genesis_volume_ray_enclosure_validation.experiment import EnclosureExperiment,Cell,fixture
from analysis.genesis_volume_ray_enclosure_validation.audit import audit_budget


@pytest.mark.parametrize('radius',[.25,2.**-12,2.**-100])
def test_root_uncertainty_is_positive_and_conserved_under_refinement(radius):
    experiment=EnclosureExperiment(fixture(radius))
    roots=[Cell(i) for i in range(3)]
    assert all(sum(experiment.root_upper(c)[s] for c in roots)==experiment.global_upper>0 for s in (0,1))
    for cell in roots:
        assert all(sum(experiment.root_upper(c)[s] for c in cell.children())==experiment.root_upper(cell)[s] for s in (0,1))


def test_integral_enclosure_width_is_paid_in_both_rules_without_quadrature_on_root_boxes():
    experiment=EnclosureExperiment(fixture());cell=Cell(0)
    cost=experiment.cost(cell);cell=experiment.compute(cell)
    assert experiment.nodes==cost and experiment.root_box_nodes==0
    assert experiment.planner.root_boxes>0
    for s in (0,1):
        # Coarse and fine half-widths add to at least the entire uncertainty U.
        assert F(*cell.source_error[s].as_integer_ratio())>=experiment.root_upper(cell)[s]>0


def test_exact_tangency_keeps_only_the_geometrically_nonempty_side_uncertain():
    experiment=EnclosureExperiment(fixture(offset=9.25))
    assert experiment.root_upper(Cell(0))[0]>0 and experiment.root_upper(Cell(0))[1]==0
    cell=experiment.compute(Cell(0))
    assert cell.fine[1]==cell.coarse[1]==cell.source_error[1]==0


def test_fixture_bound_refuses_deformation_instead_of_claiming_a_general_tensor_bound():
    raw=fixture();raw['section']['displacement_A_m']=[[.01,0.,0.],[0.,0.,0.],[0.,0.,0.]]
    with pytest.raises(ValueError):EnclosureExperiment(raw)


@pytest.mark.parametrize('corruption',['unpaid_uncertainty','smaller_integral_bound','allowance_credit'])
def test_independent_budget_audit_rejects_hidden_or_reduced_uncertainty(corruption):
    path=Path(__file__).resolve().parents[1]/'analysis/genesis_volume_ray_enclosure_validation/half_ball_report.json'
    report=json.loads(path.read_bytes())
    audit_budget(report)
    if corruption=='unpaid_uncertainty':report['trace'][-1]['source_error']=[0.,0.]
    elif corruption=='smaller_integral_bound':report['cell_records'][0]['root_integral_upper'][0]=['0','1']
    else:report['source_bounds_added_to_allowance']=True
    with pytest.raises(AssertionError):audit_budget(report)

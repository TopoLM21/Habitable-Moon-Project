from fractions import Fraction as F
import pytest
from tectonics.genesis_volume_sphere_event_mesh import parallel_cut,bracket_root,SphereEventRefiner
from tectonics.genesis_volume_event_partition import linear,source_events
from tectonics.genesis_volume_exact_event_mesh import event_mesh
from analysis.genesis_volume_banks_validation.partition_audit import check
from tests.test_genesis_volume_bank_bounds import fixture,simplex


@pytest.mark.parametrize('opposite',[0,1,2])
@pytest.mark.parametrize('t',[F(1,2),F(1,2**100),1-F(1,2**100)])
def test_parallel_cuts_keep_every_part_without_overlap(opposite,t):
    root=simplex();children=parallel_cut(root,opposite,t)
    check([tuple(n[1:] for n in root)],[tuple(n[1:] for n in c) for c in children])


def test_root_bracket_has_exact_opposite_signs_and_finite_relative_width():
    event=linear((F(-1),F(2),F()));root=simplex()
    t,cert=bracket_root(event,root[0],root[1])
    lo,hi=(F(*map(int,cert[k])) for k in ('lower','upper'))
    assert lo<F(1,3)<hi and lo<t<hi
    assert hi-lo<=min(lo,1-hi)/2**12


def test_hidden_spherical_cone_is_revealed_by_an_exact_stationary_point():
    raw=fixture(.25);root=simplex();refiner=SphereEventRefiner(raw)
    children,cert=refiner.split(root)
    assert cert['method']=='interior-conic-stationary-fan'
    point=tuple(F(*map(int,v)) for v in cert['point'])
    assert refiner.events['Delta'].evaluate(point)>0
    assert all(refiner.events['Delta'].evaluate(n)<0 for n in root)
    check([tuple(n[1:] for n in root)],[tuple(n[1:] for n in c) for c in children])


def test_curved_refinement_cannot_reintroduce_crossings_of_exact_affine_Gamma_events():
    raw=fixture(.5,True);raw['section']['normal_xyz']=[1.,0.,0.];raw['section']['offset_m']=1.
    roots,cert=event_mesh(raw,(simplex(),));assert cert['exact_event_alignment']
    refiner=SphereEventRefiner(raw);events,_=source_events(raw)
    for root in roots:
        children,_=refiner.split(root)
        check([tuple(n[1:] for n in root)],[tuple(n[1:] for n in c) for c in children])
        for child in children:
            for name in ('Gamma_cohort_lower','Gamma_cohort_upper'):
                values=[events[name].evaluate(n) for n in child]
                assert not min(values)<0<max(values)

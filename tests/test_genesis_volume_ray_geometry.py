from fractions import Fraction as F
import pytest
from analysis.genesis_volume_ray_validation.experiment import fixture,Experiment
from analysis.genesis_volume_banks_validation.independent_bank import fiber
from tectonics.genesis_volume_event_partition import source_events
from tectonics.genesis_volume_ray_roots import restrict_ray,isolate,evaluate
from analysis.genesis_volume_ray_validation.root_box_bound import fixture_root_box_mass_bound


@pytest.mark.parametrize('radius,center_z,roots',[
    (1.,9.,[F(1,3)]),  # Tangency on the upper cohort plane.
    (.5,10.,[F(1,6),F(1,2)]),
    (2.**-100,10.,[(1-F(1,2**100))/3,(1+F(1,2**100))/3]),
])
def test_actual_spherical_events_include_tangency_two_crossings_and_tiny_positive_chord(radius,center_z,roots):
    raw=fixture(radius);raw['seed_center_world_m']=[1.,1.,center_z]
    event=source_events(raw)[0]['sphere_cohort_upper']
    start=(F(3,4),F(),F(1,4));end=(F(),F(3,4),F(1,4))
    p=restrict_ray(event,start,end);result=isolate(p)
    assert result['distinct_roots']==len(roots)
    assert all(sum(a<=r<=b for a,b in result['roots'])==1 for r in roots)
    if len(roots)==1:
        assert evaluate(p,roots[0])==0
        assert evaluate(p,roots[0]-F(1,100))>0 and evaluate(p,roots[0]+F(1,100))>0
    else:assert evaluate(p,sum(roots)/2)<0


def test_all_skipped_ray_intervals_have_zero_independent_fiber_volume():
    experiment=Experiment(fixture())
    for index in range(3):
        for t in (F(1,5),F(1,2),F(4,5)):
            a,b,bands,certificate=experiment.planner.ray(index,t)
            assert sum(hi-lo for lo,hi,_ in certificate['intervals'])==1
            assert all(certificate['intervals'][i][1]==certificate['intervals'][i+1][0]
                       for i in range(len(certificate['intervals'])-1))
            for lo,hi,kind in certificate['intervals']:
                if (lo,hi,kind) in bands:continue
                assert kind=='root_free'
                for part in (F(1,4),F(1,2),F(3,4)):
                    rho=lo+(hi-lo)*part;N=tuple(x+rho*(y-x) for x,y in zip(a,b))
                    assert all(fiber(experiment.raw,N,side)==(F(),F()) for side in (-1,1))


@pytest.mark.parametrize('radius',[F(1,4),F(1,2**12),F(1,2**100)])
def test_uniform_root_box_mass_bound_scales_with_small_sphere_volume(radius):
    # pi>3 gives a strict lower bound 2*R^3 on each true half-ball volume.
    assert fixture_root_box_mass_bound(radius)/(2*radius**3)<F(1,10**9)
    assert sum(source_events(fixture(float(radius)))[0][k].degree for k in
        ('A','Delta','sphere_cohort_lower','sphere_cohort_upper','g',
         'Gamma_cohort_lower','Gamma_cohort_upper','Gamma_sphere_resultant'))<=17

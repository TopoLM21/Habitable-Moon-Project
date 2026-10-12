import math
import pytest
from tectonics.genesis_cutcell_initialization import partition_cell,mean_preserving_affine_envelope


def cell(anchors,energy=200.):
    return partition_cell(face=0,layer=0,q_lo=0.,q_hi=1.,mass_kg=10.,energy_j=energy,anchors=anchors,lower_h=0.,upper_h=40.)


def test_material_anchor_and_residual_preserve_cell_energy():
    rows=cell([dict(id=7,q_lo=0.,q_hi=.2,specific_h=10.)])
    assert [r['material_id'] for r in rows]==[7,None]
    assert rows[0]['energy_j']==20
    assert rows[1]['specific_h']==22.5
    assert math.fsum(r['energy_j'] for r in rows)==200
    assert math.fsum(r['mass_kg'] for r in rows)==10


def test_no_anchors_recovers_original_finite_volume_mean():
    rows=cell([]);assert len(rows)==1 and rows[0]['specific_h']==20.


def test_existing_boundaries_only_and_disjoint_partition():
    rows=cell([dict(id=7,q_lo=.1,q_hi=.2,specific_h=10.),dict(id=9,q_lo=.3,q_hi=.5,specific_h=15.)])
    assert [r['q_lo'] for r in rows]==[0.,.1,.2,.3,.5]
    assert math.fsum(r['energy_j'] for r in rows)==pytest.approx(200.)


def test_overlap_and_infeasible_energy_refused():
    with pytest.raises(ValueError,match='Overlapping'):cell([dict(id=1,q_lo=0.,q_hi=.5,specific_h=20.),dict(id=2,q_lo=.4,q_hi=.7,specific_h=20.)])
    with pytest.raises(ValueError,match='Residual'):cell([dict(id=1,q_lo=0.,q_hi=.9,specific_h=30.)],energy=100.)
    with pytest.raises(ValueError,match='Fully anchored'):cell([dict(id=1,q_lo=0.,q_hi=1.,specific_h=10.)])


def test_two_distinct_profiles_have_identical_mean_and_stay_bounded():
    c=dict(specific_h=20.);alt=mean_preserving_affine_envelope(c,10.,35.)
    assert alt['amplitude_h']==10
    assert (alt['left_h']+alt['right_h'])/2==20
    assert alt['left_h']>=10 and alt['right_h']<=35


def test_old_solid_material_not_forced_across_solidus():
    alt=mean_preserving_affine_envelope(dict(specific_h=19.),10.,40.,solidus_h=20.)
    assert alt['right_h']<=20
    assert (alt['left_h']+alt['right_h'])/2==19

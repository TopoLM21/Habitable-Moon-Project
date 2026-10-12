import numpy as np
import pytest
from analysis.genesis_kinetic_diagnostic.run import (
    bands, compare_velocities, direct_energy, fronts, kinetic, matrix, same_time)


def geometry(rows):
    return dict(vertices_m=(np.eye(3)*1e6).tolist(), faces=[[0,1,2]],
                column_mass_kg=[1e18],density_kg_m3=3000.,cohorts=rows)


def test_equal_energy_can_hide_opposite_velocity():
    a=np.array([1.,2.]);r=compare_velocities(np.diag([3.,4.]),7.,a,-a)
    assert r['signed_energy_difference_j']==0
    assert r['relative_rms_difference']==2
    assert r['mass_weighted_cosine']==pytest.approx(-1)
    assert r['difference_field_energy_j']==4*r['kinetic_a_j']


def test_cohort_partition_and_direct_physical_quadrature_agree():
    whole=bands([0],[.4]);split=bands([0],[.17])+bands([.17],[.4]);g=geometry(whole)
    v=np.linspace(-1e-8,2e-8,18)
    k=kinetic(matrix(g,whole),v)
    assert kinetic(matrix(g,split),v)==pytest.approx(k,rel=2e-14)
    assert direct_energy(g,whole,v)==pytest.approx(k,rel=2e-14)
    assert direct_energy(g,split,v,16)==pytest.approx(k,rel=2e-14)


def test_midpoint_work_and_energy_identity_includes_cross_term():
    M=np.array([[4.,1.],[1.,3.]])
    v0=np.array([.3,-2.]);v1=np.array([1.,.2]);mean=(v0+v1)/2;alt=(v1-v0)/2
    assert kinetic(M,v1)==pytest.approx(kinetic(M,mean)+kinetic(M,alt)+mean@M@alt)
    assert mean@M@(v1-v0)==pytest.approx(kinetic(M,v1)-kinetic(M,v0))


def test_clock_roundoff_is_allowed_but_different_time_is_not():
    same_time(.8684500000000005,.8684500000000006)
    with pytest.raises(ValueError):same_time(.86845,.868475)


@pytest.mark.parametrize('rows',[
    [dict(face_index=0,q_lo=0.,live_q_hi=.2),dict(face_index=0,q_lo=.3,live_q_hi=.4)],
    [dict(face_index=0,q_lo=0.,live_q_hi=.3),dict(face_index=0,q_lo=.2,live_q_hi=.4)]])
def test_material_gap_and_overlap_refused(rows):
    with pytest.raises(ValueError):fronts(geometry(rows))


def test_material_retreat_not_silently_treated_as_birth():
    with pytest.raises(ValueError):bands([.4],[.3])

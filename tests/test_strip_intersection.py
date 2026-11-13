from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from tectonics.strip_intersection import strip_intersection_integrals
from tectonics.continental import ContinentalCycleParameters
from test_regional_area import inventory,source,R


@pytest.mark.parametrize('rotation',[[0.,0.,0.],[.3,-.7,.8]])
def test_coarea_partial_meridional_wedges_against_independent_integral(rotation):
    rot=Rotation.from_rotvec(rotation);s=source();north=np.array([0.,0.,1.]);middle=np.array([math.cos(.37),math.sin(.37),0.])
    tri=np.array([[s.start,middle,north],[middle,s.end,north]])
    moved=replace(s,start=rot.apply(s.start),end=rot.apply(s.end),toward=rot.apply(s.toward))
    got=strip_intersection_integrals(rot.apply(tri.reshape(-1,3)).reshape(-1,3,3),moved,R,140.,320.,24)
    weighted=.7*R*quad(lambda d:math.cos(d/R)*math.exp(-.5*((d-105)/140)**2),0,425.,epsabs=1e-9)[0]
    area=R*R*math.sin(425./R)
    assert got==pytest.approx(np.array([.37,.63])[:,None]*np.array([weighted,area]),rel=3e-13,abs=1e-8)


def test_coarea_field_is_rotation_invariant_and_does_not_renormalize_error():
    inv=inventory(edge=125.);inv.area_params=replace(inv.area_params,strip_rule='coarea');s=source()
    inv.refine([s],[],R,140.,320.);f,d=inv.integrated_field([s],[],R,140.,320.)
    assert abs(d['source_deposition_residual_km2'])<1e-6
    assert d['coarea_order_difference_estimate_km2']<1e-6
    # Rotate the source relative to the tessellation: total measure is invariant,
    # but individual cell averages need not be. No normalization is permitted.
    rot=Rotation.from_rotvec([.3,-.7,.8]);s2=replace(s,start=rot.apply(s.start),end=rot.apply(s.end),toward=rot.apply(s.toward))
    inv.refine([s2],[],R,140.,320.);g,e=inv.integrated_field([s2],[],R,140.,320.)
    assert e['deposited_source_km2']==pytest.approx(d['deposited_source_km2'],rel=1e-8)
    assert e['source_supported_area_sum_km2']==pytest.approx(d['source_supported_area_sum_km2'],rel=1e-14)


def test_pure_felsic_refinement_cannot_generate_negative_ocean_area():
    inv=inventory(edge=125.);a=inv.leaves.arrays
    a['felsic_area']=a['area'].copy();a['felsic_volume']=a['area']*35;a['felsic_age_moment']=a['felsic_volume']*100
    a['basalt_volume']*=0;a['basalt_age_moment']*=0
    inv.refine([source()],[],R,140.,320.);inv.aggregate()
    assert np.array_equal(inv.coarse.arrays['area'],inv.coarse.arrays['felsic_area'])
    assert np.all(inv.coarse.arrays['potential']==0)

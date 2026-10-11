"""A whole-cell Gamma degeneracy is not an empty spatial source."""
from fractions import Fraction as F
import numpy as np
import pytest
from analysis.genesis_volume_mapped_validation.angular import CellGeometry,AngularControlRefusal


def planar_birth(center):
    X=np.eye(3);P=np.diag([0.,0.,-1.]);D=np.tile([0.,0.,1.],(3,1))
    return dict(section=dict(triangle_m=X,slope_P_m=P,displacement_A_m=P+D,
        normal_xyz=[0.,0.,1.],offset_m=1.,lambda_bounds=[.5,1.]),
        seed_center_world_m=center,seed_radius_m=.1)


def cell():
    return ((F(1,2),F(2,5),F(1,10)),(F(2,5),F(1,2),F(1,10)),
            (F(2,5),F(2,5),F(1,5)))


def test_whole_cell_gamma_remains_an_explicit_refusal():
    geometry=CellGeometry(planar_birth([.3,.3,1.]))
    with pytest.raises(AngularControlRefusal,match='Gamma'):
        geometry.bounds(cell())


def test_independently_certified_empty_cell_is_still_empty_on_gamma():
    geometry=CellGeometry(planar_birth([.3,.3,100.]))
    bounds=geometry.bounds(cell())
    assert all(hi==0 and lo==0 for lo,hi in bounds.values())

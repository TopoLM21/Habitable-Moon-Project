"""The explicitly selected stored W15 direction; no projection or new geometry."""
from fractions import Fraction as F
import numpy as np


def verify(candidate,B,raw):
    coords=list(map(F,candidate['coordinates_w15_m']));a=list(map(F,candidate['enrichment_increment_m']))
    assert coords==[F(1,8192)]+[F()]*14
    assert len(a)==18 and all(a[i]==sum(F(B[i][j])*coords[j] for j in range(15)) for i in range(18))
    section=raw['section'];X=np.asarray(section['triangle_m']);A=np.asarray(section['displacement_A_m']);P=np.asarray(section['slope_P_m']);n=np.asarray(section['normal_xyz'])
    za=(X+A)@n-section['offset_m'];zp=(X+P)@n;Z=np.zeros((18,3))
    for i in range(3):
        for j in range(3):Z[6*i+j,j]=za[i];Z[6*i+3+j,j]=zp[i]
    Z/=np.linalg.norm(Z,axis=0);v=np.asarray(candidate['enrichment_increment_m'])
    leak=float(np.linalg.norm(Z.T@v)/np.linalg.norm(v));threshold=256*np.finfo(float).eps
    assert leak<=threshold
    return dict(exact_literal_membership=True,coordinate_amplitude_m='1/8192',
                analytic_kernel_relative_leakage=leak,kernel_threshold=float(threshold),kernel_check_passed=True,
                physical_direction_selected=False,full_typed_release_validator_called=False)


def distance_to_zero(bounds):
    lo,hi=map(F,bounds)
    assert lo<=hi
    return lo if lo>0 else -hi if hi<0 else F()

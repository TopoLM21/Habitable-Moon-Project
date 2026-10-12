import numpy as np
from analysis.genesis_thermal_moment.sink_audit import quadratic_parts


def test_forcing_quadratic_cross_terms_are_counted_once():
    H=np.diag([2.,5.,7.]);v=dict(deformation=np.array([1.,2.,-1.]),thermal=np.array([-2.,-1.,.5]),relaxation=np.array([.1,-.3,.2]))
    terms=quadratic_parts(v,H);total=sum(v.values())
    np.testing.assert_allclose(sum(terms.values()),.5*total@H@total,rtol=1e-14)
    assert sum(terms[k] for k in v)!=sum(terms.values())


def test_orthogonal_projection_sink_identity_with_changed_damage_metric():
    # Two active coordinates and one unresolved coordinate; common SPD metric.
    Hf=np.diag([2.,3.,5.]);Ha=np.diag([1.8,2.7]);old=np.array([1.,2.,4.]);new=np.array([1.3,1.7,4.1]);ag=np.array([1.4,1.8])
    dp=new[:2]-old[:2];err=ag-new[:2];df=new-old;da=ag-old[:2]
    nf=.5*df@Hf@df;na=.5*da@Ha@da
    nr=.5*Hf[2,2]*df[2]**2;metric=.5*dp@(Ha-Hf[:2,:2])@dp
    np.testing.assert_allclose(na-nf,metric+dp@Ha@err+.5*err@Ha@err-nr,atol=1e-15)
    initial_reserve=.5*Hf[2,2]*old[2]**2
    assert initial_reserve>100*nr  # Stored projection reserve is not increment loss.

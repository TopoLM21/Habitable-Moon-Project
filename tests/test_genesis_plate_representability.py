import numpy as np
from analysis.genesis_plate_representability.measure import projection


def test_projection_preserves_represented_moments_and_splits_energy():
    rng=np.random.default_rng(93);q=rng.normal(size=(12,12));M=q.T@q+np.eye(12);R=rng.normal(size=(12,3));v=rng.normal(size=12)
    fitted,report=projection(M,v,R);res=v-fitted
    np.testing.assert_allclose(R.T@M@fitted,R.T@M@v,rtol=1e-12,atol=1e-12)
    assert abs(v@M@v-fitted@M@fitted-res@M@res)<1e-12
    assert 0<=report['relative_mass_rms']<=1


def test_pure_rigid_field_is_reproduced_without_amplitude_fit():
    rng=np.random.default_rng(15);R=rng.normal(size=(20,3));coeff=np.array([.2,-.1,.7]);v=R@coeff
    fitted,r=projection(np.eye(20),v,R)
    np.testing.assert_allclose(fitted,v,atol=1e-15)
    assert r['relative_mass_rms']<1e-14


def test_more_dofs_do_not_increase_best_fit_residual():
    rng=np.random.default_rng(52);R=rng.normal(size=(20,6));v=rng.normal(size=20)
    _,a=projection(np.eye(20),v,R[:,:3]);_,b=projection(np.eye(20),v,R)
    assert b['relative_mass_rms']<=a['relative_mass_rms']

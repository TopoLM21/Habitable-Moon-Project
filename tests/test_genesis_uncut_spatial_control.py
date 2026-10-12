import numpy as np
from analysis.genesis_uncut_spatial_control.run import run,cases,check_case


def test_complete_uncut_operator_matches_independent_physical_integral():
    result=run()
    assert result['status']=='PASS', result


def test_control_detects_centroid_underintegration(monkeypatch):
    import tectonics.genesis_volume_operator as production
    monkeypatch.setattr(production,'ANGULAR_BARYCENTRIC',np.full((3,3),1/3))
    result=check_case(*cases()[0])
    assert result['status']=='FAIL'
    assert result['errors']['stiffness_frobenius']>1e-4


def test_control_detects_missing_radial_variance(monkeypatch):
    import tectonics.genesis_volume_operator as production
    original=production.conical_radial_moments
    monkeypatch.setattr(production,'conical_radial_moments',lambda a,b:(original(a,b)[0],0.))
    result=check_case(*cases()[0])
    assert result['status']=='FAIL'
    assert result['errors']['stiffness_frobenius']>1e-4

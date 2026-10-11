"""Exact contraction arithmetic and certificate schema regression checks."""
from fractions import Fraction
from copy import deepcopy

import numpy as np
import pytest

from analysis.genesis_volume_operand_validation import independent_contractions as proof


def _packet(witness,label,value):
    value=np.asarray(value,dtype=object)
    pairs=[]
    for scalar in value.ravel():
        scalar=proof._fraction(scalar)
        pairs.append([str(scalar.numerator),str(scalar.denominator)])
    witness.setdefault("native_exact_binary_ratio",{})[label]={"shape":list(value.shape),
        "numerator_denominator":pairs}


def _fixture():
    N=np.array([1/3,1/3,1/3])
    raw={"section":{"triangle_m":np.eye(3).tolist(),"slope_P_m":np.zeros((3,3)).tolist(),
        "displacement_A_m":np.zeros((3,3)).tolist(),"normal_xyz":[1.,0.,0.],
        "offset_m":float(N[0]),"mass_kg":2.,"volume_m3":1.},
        "C":np.eye(6).tolist(),"Q":np.eye(3).tolist(),
        "e0":np.zeros((3,6)).tolist(),"e1":np.zeros((3,6)).tolist()}
    witness={"interval":{"gamma_root_xi":0.,"constant_side":None}}
    for key,value in {"lambda_mid":1,"lambda_halfwidth":Fraction(1,2),
                      "reference_determinant_m3":1,"gamma_endpoint_distance_xi":1}.items():
        _packet(witness,"interval."+key,value)
    _packet(witness,"interval.side",[0])
    _packet(witness,"interval.literal_phi_coefficients",[0,1])
    _packet(witness,"interval.barycentric",N)
    families={"lam":np.array([1,Fraction(1,2)],dtype=object),
        "Bhat":np.zeros((1,6,18),dtype=object),"Ghat_without_H":np.zeros((1,6,18),dtype=object),
        "T":np.zeros((1,3,18),dtype=object),"E_without_H":np.zeros((1,3,18),dtype=object),
        "memoryhat":np.zeros((1,6),dtype=object)}
    for key,scalar in (("Bhat",1),("Ghat_without_H",2),("T",1),("E_without_H",3)):
        families[key][0,0,0]=scalar
    families["memoryhat"][0,0]=5
    for key,value in families.items():_packet(witness,"coefficient."+key,value)
    return raw,N,witness,families


def test_polynomial_bilinear_matches_exact_hand_expansion():
    left=np.array([[[1],[2]],[[3],[4]]],dtype=object)
    right=np.array([[[5],[6]],[[7],[8]]],dtype=object)
    middle=np.array([[2,1],[1,3]],dtype=object)
    expected=[left[0].T@middle@right[0],
              left[0].T@middle@right[1]+left[1].T@middle@right[0],
              left[1].T@middle@right[1]]
    assert np.array_equal(proof.polynomial_bilinear(left,middle,right),expected)


@pytest.mark.parametrize("side_power",[0,1,2])
def test_exact_paired_moments_match_source_gamma_at_center(side_power):
    raw,N,witness,_=_fixture()
    native=proof.exact_moments(witness,16,side_power)
    source=proof.exact_moments(witness,16,side_power,gamma_mode="literal_source",raw=raw,N=N)
    assert native==source
    assert all(isinstance(v,Fraction) for v in native)
    if side_power==1:
        assert native[0]==0
        assert native[1]==Fraction(1,4)
    else:
        assert native[1]==0


def test_thin_native_endpoint_distance_survives_rounded_root():
    _,_,witness,_=_fixture()
    _packet(witness,"interval.side",[1])
    tiny=Fraction(1,2**100)
    _packet(witness,"interval.gamma_endpoint_distance_xi",tiny)
    moments=proof.exact_moments(witness,3,1)
    assert moments[1]==tiny/2
    assert moments[3]==tiny/2
    assert moments[0]==Fraction(-1,2)


def test_all_full_columns_before_projection_and_bilinear_probe():
    raw,N,witness,_=_fixture()
    W=np.zeros((18,2));W[0,0]=2;W[17,1]=1
    q=np.zeros(18);q[0]=4
    a=np.zeros(18);a[0]=5
    z=np.zeros(18);z[0]=6
    probes=[{"name":"declared","q":q.tolist(),"a":a.tolist(),"mean_z_m":z.tolist()}]
    result=proof.exact_line_contractions(raw,N,witness,added_basis=W,probes=probes)
    assert len(result)==1314+18*2*2+2*2*2+2+43
    assert result["full.stiffness_cross[0,0]"]==0
    assert result["full.stiffness_enriched[0,0]"]==1
    assert result["full.mass_enriched[0,0]"]==Fraction(39,8)
    assert result["projected.stiffness_enriched[0,0]"]==4
    assert result["projected.stiffness_enriched[1,1]"]==0
    assert result["probe.declared.stiffness_self_bilinear_j"]==25
    assert result["probe.declared.self_work_j"]==30
    assert result["probe.declared.mass_self_bilinear_kg_m2"]==Fraction(975,8)


def test_fractional_source_weights_do_not_silently_round_to_native_density():
    raw,N,witness,_=_fixture()
    raw["section"]["mass_kg"]=1.
    raw["section"]["volume_m3"]=3.
    native=proof.exact_line_contractions(raw,N,witness)
    exact=proof.exact_line_contractions(raw,N,witness,weight_mode="literal_source")
    assert exact["full.mass_enriched[0,0]"]==Fraction(13,16)
    assert native["full.mass_enriched[0,0]"]!=exact["full.mass_enriched[0,0]"]


def test_native_packet_noncanonical_or_nonbinary_is_rejected():
    _,_,witness,_=_fixture()
    witness["native_exact_binary_ratio"]["interval.side"]["numerator_denominator"]=[["0","2"]]
    with pytest.raises(proof.ContractionProofRefusal,match="canonical"):
        proof.decode_native_array(witness,"interval.side")
    witness["native_exact_binary_ratio"]["interval.side"]["numerator_denominator"]=[["1","3"]]
    with pytest.raises(proof.ContractionProofRefusal,match="binary power"):
        proof.decode_native_array(witness,"interval.side")


def test_source_gamma_uses_literal_N_without_normalization():
    raw,N,witness,_=_fixture()
    N=np.array([.25,.25,.5+2**-48])
    raw["section"]["normal_xyz"]=[0.,0.,1.]
    raw["section"]["offset_m"]=.5
    expected0,expected1=proof._literal_source_gamma(raw,N,Fraction(1),Fraction(1,2))
    assert expected0==Fraction(1,2**48)
    assert expected1==Fraction(*N[2].as_integer_ratio())/2


def test_high_degree_integral_and_arbitrary_rational_coefficients_are_exact():
    _,_,witness,_=_fixture()
    coefficients=np.zeros(17,dtype=object);coefficients[16]=Fraction(1,7)
    result=proof.exact_polynomial_integral(coefficients,proof.exact_moments(witness,16))
    assert result.item()==Fraction(1,119)


def test_whole_parent_work_is_not_in_line_schema():
    raw,N,witness,_=_fixture()
    result=proof.exact_line_contractions(raw,N,witness)
    assert len(result)==1314
    assert not any("parent_probe_work" in name or "funding" in name for name in result)


def _source_certificate_fixture():
    from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients
    N=(Fraction(1,4),Fraction(1,4),Fraction(1,2))
    raw={"section":{"triangle_m":[[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]],
        "displacement_A_m":np.zeros((3,3)).tolist(),"slope_P_m":np.zeros((3,3)).tolist(),
        "lambda_bounds":[.75,1.],"normal_xyz":[0.,0.,1.],"offset_m":1.75,
        "mass_kg":2.,"volume_m3":1.},"basis_xyz":np.eye(3).tolist(),
        "seed_center_world_m":[7/32,7/16,7/4],"seed_radius_m":1.,
        "memory0":np.arange(18.).reshape(3,6).tolist(),
        "memory1":(np.arange(18.).reshape(3,6)*-.125).tolist(),"elasticity_pa":np.eye(6).tolist()}
    lm,lh=Fraction(7,8),Fraction(1,64)
    witness={"interval":{"gamma_root_xi":0.,"constant_side":None}}
    for key,value in {"lambda_mid":lm,"lambda_halfwidth":lh,"reference_determinant_m3":2,
                      "gamma_endpoint_distance_xi":1}.items():_packet(witness,"interval."+key,value)
    _packet(witness,"interval.side",[0]);_packet(witness,"interval.literal_phi_coefficients",[0,2*lh])
    _packet(witness,"interval.barycentric",N)
    fields=exact_source_coefficients(raw,N,lm,lh)
    for key,value in fields.items():
        _packet(witness,"coefficient."+key,np.asarray([float(v) for v in value.ravel()]).reshape(value.shape))
    exact=proof.exact_line_contractions(raw,N,witness)
    witness["component_names"]=list(exact)
    _packet(witness,"values",list(map(float,exact.values())))
    _packet(witness,"budgets",np.ones(len(exact)))
    return raw,N,witness


def test_canonical_certificate_has_exact_scope_and_equality_verifier():
    raw,N,witness=_source_certificate_fixture()
    certificate=proof.build_contraction_certificate(raw,N,witness)
    assert certificate["checked_component_count"]==1314
    assert certificate["status"]=="PASS"
    assert certificate["historical_saved_bound_operand_authentication_certified"] is False
    assert certificate["source_endpoint_displacement_accuracy_certified"] is False
    assert certificate["selected_line_uniform_forward_bound_certified"] is False
    assert certificate["angular_node_uniform_error_certified"] is False
    assert proof.verify_contraction_certificate(raw,N,witness,certificate)["status"]=="PASS"


@pytest.mark.parametrize("field",["represented_polynomial_arithmetic_residual",
    "source_coefficient_contraction_difference","source_gamma_and_weights_contraction_difference",
    "exact_selected_interval_residual","exact_selected_interval_triangle_enclosure"])
def test_consistently_resealed_exact_error_inflation_is_rejected(field):
    raw,N,witness=_source_certificate_fixture()
    certificate=proof.build_contraction_certificate(raw,N,witness)
    component=next(iter(certificate["components"].values()))
    n,d=map(int,component[field]);component[field]=[str(n+d),str(d)]
    certificate.pop("certificate_sha256");certificate["certificate_sha256"]=proof._digest(certificate)
    with pytest.raises(proof.ContractionProofRefusal,match="Canonical"):
        proof.verify_contraction_certificate(raw,N,witness,certificate)


def test_altered_source_cannot_be_hidden_by_certificate_resealing():
    raw,N,witness=_source_certificate_fixture()
    certificate=proof.build_contraction_certificate(raw,N,witness)
    changed=deepcopy(raw);changed["memory0"][0][0]+=1
    with pytest.raises(proof.ContractionProofRefusal,match="Canonical"):
        proof.verify_contraction_certificate(changed,N,witness,certificate)


def test_certificate_refuses_other_material_N_even_without_normalization():
    raw,N,witness=_source_certificate_fixture()
    with pytest.raises(proof.ContractionProofRefusal,match="native material N"):
        proof.build_contraction_certificate(raw,(Fraction(1,3),)*3,witness)

"""Exact source-root goldens and adversarial native-chart authentication."""
from copy import deepcopy
from fractions import Fraction
import json

import numpy as np
import pytest

from analysis.genesis_volume_operand_validation.independent_geometry_bounds import (
    GeometryBoundRefusal, audit_geometry_operands, rational, round_binary,
    round_sqrt_binary, sqrt_enclosure,
)
from tectonics.genesis_volume_exact_lambda import prepare_exact_lambda_source, prepare_fixed_N_interval
from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section


LD=np.longdouble


def _section(*,normal=(1.,0.,0.),offset=2.,displacement=None,lower=.25,upper=.75):
    X=4*np.eye(3);coefficient=2.
    volume=np.linalg.det(X)*(upper**3-lower**3)/6
    cohort=ConicalCohortGeometry(17,0,2,(1-upper**3)/coefficient,(1-lower**3)/coefficient,
        2*volume,volume,tuple(map(tuple,X)),upper,lower,(),'original-owner','original-material')
    U=np.zeros((3,6)) if displacement is None else displacement
    return prepare_cohort_material_section(cohort,U,normal,offset,conical_mass_coefficient=coefficient)


def _native_record(value):
    array=np.asarray(value,LD)
    return dict(shape=list(array.shape),numerator_denominator=[
        [str(n),str(d)] for n,d in (x.as_integer_ratio() for x in array.reshape(-1))])


def _witness(interval):
    if interval is None:
        return dict(interval=None)
    names=('lambda_mid','lambda_halfwidth','lambda_bounds','birth_relative_coefficients',
        'literal_phi_coefficients','relative_lambda_mid_to_original_lower',
        'relative_lambda_mid_to_original_upper','gamma_root_arithmetic_estimate',
        'gamma_endpoint_distance_xi','gamma_left_width_xi','gamma_right_width_xi',
        'gamma_root_xi','constant_side')
    body={};records={}
    for name in names:
        value=getattr(interval,name)
        body[name]=None if value is None else np.asarray(value,float).tolist()
        if value is not None:
            records['interval.'+name]=_native_record(value)
    records['interval.barycentric']=_native_record(interval.barycentric)
    records['interval.side']=_native_record((interval.gamma_root_xi if interval.gamma_root_xi is not None
                                            else interval.constant_side,))
    return dict(interval=body,native_exact_binary_ratio=records)


def _case(*,center=(2.,0.,0.),radius=.5,N=(1.,0.,0.),**kwargs):
    section=_section(**kwargs)
    source=prepare_exact_lambda_source(section,center,radius)
    interval=prepare_fixed_N_interval(source,N)
    raw=dict(section={name:getattr(section,name) for name in ('triangle_m','displacement_A_m',
        'slope_P_m','lambda_bounds','normal_xyz','offset_m')},seed_center_world_m=center,seed_radius_m=radius)
    return raw,N,_witness(interval)


@pytest.mark.parametrize('value',(Fraction(2),Fraction(9,16),Fraction(2)**-20000,
                                   Fraction(3,7),Fraction(2)**2000))
def test_integer_sqrt_enclosure_contains_exact_root_without_precision_floor(value):
    low,high=sqrt_enclosure(value)
    assert low*low<=value<=high*high
    assert low>0
    if value==Fraction(9,16):
        assert low==high==Fraction(3,4)
    else:
        assert high-low<=high*Fraction(2)**-253


@pytest.mark.parametrize('bits',(3,53,64))
def test_nearest_even_binary_rounding_and_power_boundary(bits):
    quantum=Fraction(2)**(1-bits)
    assert round_binary(1+quantum/2,bits=bits)==1
    assert round_binary(1+3*quantum/2,bits=bits)==1+2*quantum
    assert round_binary(-1-quantum/2,bits=bits)==-1
    assert round_binary(2-quantum/2,bits=bits)==2


@pytest.mark.parametrize('value',(Fraction(2),Fraction(9,16),Fraction(7,11),Fraction(2)**-15000))
def test_exact_integer_sqrt_rounding_matches_native_sqrt(value):
    # Huge fractions can exceed LD numerator/denominator range before their
    # quotient; test the ordinary native range and an exact tiny power.
    native=LD(2)**-7500 if value==Fraction(2)**-15000 else np.sqrt(LD(value.numerator)/LD(value.denominator))
    assert round_sqrt_binary(value)==rational(native)


def test_exact_centered_ball_and_original_bound_roots_are_certified():
    raw,N,witness=_case(offset=2.125)
    report=audit_geometry_operands(raw,N,witness)
    assert report['status']=='PASS'
    assert report['source_geometry_error_bounds_certified']
    assert report['source_Gamma_crosses']
    for name in ('lambda_mid','lambda_halfwidth','represented_lower_endpoint','represented_upper_endpoint'):
        assert Fraction(report['component_error_enclosures'][name]['absolute_error_bound'])==0
    assert not report['operator_geometry_error_propagated']
    assert not report['source_geometry_accuracy_certified']
    assert not report['angular_accuracy_certified']
    assert not report['physical_creation_admitted']


@pytest.mark.parametrize('offset,side',((1.,'1/2'),(3.,'-1/2')))
def test_constant_Gamma_bank_sign_is_source_certified(offset,side):
    report=audit_geometry_operands(*_case(offset=offset))
    assert report['source_Gamma_bank_sign']==side
    assert not report['source_Gamma_crosses']


def test_original_cohort_caps_inside_large_sphere_have_zero_chart_shift():
    report=audit_geometry_operands(*_case(radius=2.))
    assert Fraction(report['component_error_enclosures']['lambda_halfwidth']['absolute_error_bound'])==0
    assert Fraction(report['component_error_enclosures']['lambda_mid']['absolute_error_bound'])==0


def test_near_sphere_tangency_retains_positive_margin_and_bounds_fallback():
    center=np.array((LD(2),LD(.5)-np.ldexp(LD(1),-60),LD(0)))
    report=audit_geometry_operands(*_case(center=center))
    assert report['native_sign_fallback_used']
    assert Fraction(report['exact_source_margin_m2'])>0
    assert report['support_classification_certified']


def test_positive_original_bound_sliver_is_not_widened():
    delta=np.ldexp(LD(1),-56)
    report=audit_geometry_operands(*_case(center=np.array((LD(3.5)-delta,LD(0),LD(0)))))
    half=report['component_error_enclosures']['lambda_halfwidth']
    assert Fraction(half['native_value'])==Fraction(1,2**59)
    assert Fraction(half['absolute_error_bound'])==0


def test_sub_ulp_Gamma_endpoint_sliver_has_positive_certified_bank_width():
    displacement=np.zeros((3,6));displacement[1,0]=1
    tiny=LD('1e-30')
    report=audit_geometry_operands(*_case(offset=3.,radius=2.,displacement=displacement,
                                            N=np.array((LD(1),tiny,LD(0)))))
    assert report['source_Gamma_crosses']
    width=report['component_error_enclosures']['gamma_right_width_xi']
    assert Fraction(width['native_value'])>0
    assert Fraction(width['source_value_enclosure']['lower'])>0
    root=report['component_error_enclosures']['gamma_root_xi']
    assert Fraction(root['native_value'])==1


def test_empty_line_requires_matching_source_and_native_support():
    report=audit_geometry_operands(*_case(center=(10.,0.,0.)))
    assert report['empty'] and report['status']=='PASS'


def test_exact_tangency_is_explicit_refusal_not_positive_error_certificate():
    with pytest.raises(GeometryBoundRefusal,match='Exact sphere tangency'):
        audit_geometry_operands(*_case(center=(2.,.5,0.)))


@pytest.mark.parametrize('label',('interval.lambda_mid','interval.gamma_left_width_xi',
                                'interval.birth_relative_coefficients'))
def test_consistently_resealed_native_operand_shift_is_refused(label):
    raw,N,witness=_case(offset=2.125)
    witness=deepcopy(witness)
    record=witness['native_exact_binary_ratio'][label]
    old=Fraction(*map(int,record['numerator_denominator'][0]))
    shifted=round_binary(old+Fraction(1,1024))
    record['numerator_denominator'][0]=[str(shifted.numerator),str(shifted.denominator)]
    key=label.split('.',1)[1]
    public=np.asarray(witness['interval'][key],float)
    public.reshape(-1)[0]=float(shifted)
    witness['interval'][key]=public.tolist()
    # The caller may reseal the enclosing archive.  This lane independently
    # determines the exact native result from source, so that cannot pass it.
    with pytest.raises(GeometryBoundRefusal,match='operation replay differs'):
        audit_geometry_operands(raw,N,witness)


def test_public_native_value_disagreement_is_refused():
    raw,N,witness=_case()
    witness['interval']['lambda_mid']=.51
    with pytest.raises(GeometryBoundRefusal,match='Public/native'):
        audit_geometry_operands(raw,N,witness)


def test_invalid_original_N_is_not_normalized():
    raw,N,witness=_case()
    with pytest.raises(GeometryBoundRefusal,match='leaves source simplex'):
        audit_geometry_operands(raw,(1.,.1,0.),witness)


def test_missing_native_logical_geometry_operand_is_refused():
    raw,N,witness=_case()
    del witness['native_exact_binary_ratio']['interval.lambda_halfwidth']
    with pytest.raises(GeometryBoundRefusal,match='Missing native'):
        audit_geometry_operands(raw,N,witness)


def test_rounding_created_Gamma_bank_beyond_true_sphere_is_refused():
    displacement=np.zeros((3,6));displacement[1,0]=1
    tiny=LD('1e-30')
    # The old radial kernel retains a bank in its represented chart, but its
    # true source sphere endpoint is just before Gamma (O(tiny**2)).
    # Source-bank admission cannot follow from native bank-width positivity.
    with pytest.raises(GeometryBoundRefusal,match='Source/native Gamma crossing'):
        audit_geometry_operands(*_case(offset=2.5,displacement=displacement,
                                      N=np.array((LD(1),tiny,LD(0)))))


# Literal minimal geometry witness from accepted Genesis cohort 1, actual03.
# No planet restore or new initialization; all source and native bits retained.
GENUINE_R4_GOLDEN=json.loads(r'''
{
  "N": [
    "12297829382473034411/36893488147419103232",
    "12297829382473034411/36893488147419103232",
    "12297829382473034411/36893488147419103232"
  ],
  "raw": {
    "section": {
      "displacement_A_m": [
        [
          0.917386949695752,
          -1.5080456939747349,
          -0.04374257440034894
        ],
        [
          0.6531170464550741,
          -0.3473096701875744,
          -0.4087710154277778
        ],
        [
          0.029878477215076627,
          -0.12691612315928352,
          -0.2873058353576092
        ]
      ],
      "lambda_bounds": [
        0.9999932803766464,
        1.0
      ],
      "normal_xyz": [
        0.2672612419124244,
        0.5345224838248488,
        0.8017837257372732
      ],
      "offset_m": 2403825.4588440103,
      "slope_P_m": [
        [
          90.89980337615366,
          -148.018601273836,
          -1.9079926572857926
        ],
        [
          141.4353788370711,
          -86.36862155791569,
          -57.48793851529024
        ],
        [
          53.28246077641166,
          -140.4433063729462,
          -90.77453442666696
        ]
      ],
      "triangle_m": [
        [
          -2779334.7908648686,
          4497293.439988639,
          445.91590265775136
        ],
        [
          -4276968.434620413,
          2642996.844734693,
          1634765.7483262795
        ],
        [
          -1633425.240690996,
          4276593.808315874,
          2644431.4598161923
        ]
      ]
    },
    "seed_center_world_m": [
      -2896565.8903007517,
      3805614.5844821963,
      1426542.6686451025
    ],
    "seed_radius_m": 4.024215510221975
  },
  "witness": {
    "interval": {
      "barycentric": [
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333
      ],
      "birth_relative_coefficients": [
        [
          2.1925077114149513e-12,
          2.7001086364861352e-11,
          -6.903465553182823e-11
        ],
        [
          -2.3355913238031847,
          3.0685856902940314,
          1.1502632317035517
        ]
      ],
      "constant_side": null,
      "gamma_endpoint_distance_xi": 0.9999999998197205,
      "gamma_left_width_xi": 1.0000000001802796,
      "gamma_right_width_xi": 0.9999999998197205,
      "gamma_root_arithmetic_estimate": 1.516276050275782e-30,
      "gamma_root_xi": 1.8027947923409779e-10,
      "geometry_operand_accuracy_certified": false,
      "lambda_bounds": [
        0.9999958338335206,
        0.9999974465431255
      ],
      "lambda_halfwidth": 8.06354802436182e-07,
      "lambda_mid": 0.999996640188323,
      "literal_phi_coefficients": [
        -3.49431630674271e-10,
        1.9382773466997014
      ],
      "reference_determinant_m3": 3.8855427409131e+19,
      "relative_lambda_mid_to_original_lower": 3.3598116766841773e-06,
      "relative_lambda_mid_to_original_upper": 3.359811676950674e-06
    },
    "native_exact_binary_ratio": {
      "interval.barycentric": {
        "numerator_denominator": [
          [
            "12297829382473034411",
            "36893488147419103232"
          ],
          [
            "12297829382473034411",
            "36893488147419103232"
          ],
          [
            "12297829382473034411",
            "36893488147419103232"
          ]
        ],
        "shape": [
          3
        ]
      },
      "interval.birth_relative_coefficients": {
        "numerator_denominator": [
          [
            "5558667432760368811",
            "2535301200456458802993406410752"
          ],
          [
            "2139246458576922119",
            "79228162514264337593543950336"
          ],
          [
            "-2734744453795972169",
            "39614081257132168796771975168"
          ],
          [
            "-5385506926371730179",
            "2305843009213693952"
          ],
          [
            "14151353724275339803",
            "4611686018427387904"
          ],
          [
            "5304652863158372367",
            "4611686018427387904"
          ]
        ],
        "shape": [
          2,
          3
        ]
      },
      "interval.gamma_endpoint_distance_xi": {
        "numerator_denominator": [
          [
            "18446744070383982201",
            "18446744073709551616"
          ]
        ],
        "shape": []
      },
      "interval.gamma_left_width_xi": {
        "numerator_denominator": [
          [
            "2305843009629390129",
            "2305843009213693952"
          ]
        ],
        "shape": []
      },
      "interval.gamma_right_width_xi": {
        "numerator_denominator": [
          [
            "18446744070383982201",
            "18446744073709551616"
          ]
        ],
        "shape": []
      },
      "interval.gamma_root_arithmetic_estimate": {
        "numerator_denominator": [
          [
            "554009982530922497",
            "365375409332725729550921208179070754913983135744"
          ]
        ],
        "shape": []
      },
      "interval.lambda_bounds": {
        "numerator_denominator": [
          [
            "18446667221502737899",
            "18446744073709551616"
          ],
          [
            "18446696970744084193",
            "18446744073709551616"
          ]
        ],
        "shape": [
          2
        ]
      },
      "interval.lambda_halfwidth": {
        "numerator_denominator": [
          [
            "7798585123482829869",
            "9671406556917033397649408"
          ]
        ],
        "shape": []
      },
      "interval.lambda_mid": {
        "numerator_denominator": [
          [
            "9223341048061705523",
            "9223372036854775808"
          ]
        ],
        "shape": []
      },
      "interval.literal_phi_coefficients": {
        "numerator_denominator": [
          [
            "-13842413011342767987",
            "39614081257132168796771975168"
          ],
          [
            "17877453079219096319",
            "9223372036854775808"
          ]
        ],
        "shape": [
          2
        ]
      },
      "interval.reference_determinant_m3": {
        "numerator_denominator": [
          [
            "38855427409131005980",
            "1"
          ]
        ],
        "shape": []
      },
      "interval.relative_lambda_mid_to_original_lower": {
        "numerator_denominator": [
          [
            "30988793067827",
            "9223372036854775808"
          ]
        ],
        "shape": []
      },
      "interval.relative_lambda_mid_to_original_upper": {
        "numerator_denominator": [
          [
            "30988793070285",
            "9223372036854775808"
          ]
        ],
        "shape": []
      },
      "interval.side": {
        "numerator_denominator": [
          [
            "14283211878746041513",
            "79228162514264337593543950336"
          ]
        ],
        "shape": [
          1
        ]
      }
    }
  }
}
''')


def test_genuine_birth_cohort_R4_preserves_own_absolute_center_error():
    golden=GENUINE_R4_GOLDEN
    report=audit_geometry_operands(golden['raw'],golden['N'],golden['witness'])
    errors=report['component_error_enclosures']
    bound=Fraction(errors['lambda_mid']['absolute_error_bound'])
    assert Fraction('8.33366e-20')<bound<Fraction('8.33368e-20')
    assert Fraction(errors['lambda_mid']['absolute_error_lower'])>Fraction('8.33366e-20')
    assert Fraction(errors['lambda_halfwidth']['absolute_error_bound'])<Fraction('3.498e-26')
    assert report['lambda_absolute_rounding_error_retained_in_own_geometry_operands']
    assert any(Fraction(q)>0 for q in report['represented_source_ball_endpoint_quadratic_m2'])
    assert report['represented_chart_endpoint_outside_source_ball_possible']
    assert not report['source_geometry_accuracy_certified']
    assert not report['operator_geometry_error_propagated']


def test_genuine_chart_shift_cannot_pass_by_resealing_numerical_budget():
    golden=deepcopy(GENUINE_R4_GOLDEN)
    record=golden['witness']['native_exact_binary_ratio']['interval.lambda_mid']
    native=Fraction(*map(int,record['numerator_denominator'][0]))
    shifted=round_binary(native+Fraction('1e-12'))
    record['numerator_denominator'][0]=[str(shifted.numerator),str(shifted.denominator)]
    golden['witness']['interval']['lambda_mid']=float(shifted)
    golden['witness']['geometry_operand_accuracy_certified']=True
    golden['witness']['roundoff_component_bounds']=[1e100]
    with pytest.raises(GeometryBoundRefusal,match='operation replay differs'):
        audit_geometry_operands(golden['raw'],golden['N'],golden['witness'])


def test_positive_Gamma_bank_below_public_binary64_range_retains_native_proof():
    displacement=np.zeros((3,6));displacement[1,0]=1
    tiny=LD('1e-400')
    raw,N,witness=_case(offset=3.,radius=2.,displacement=displacement,
                        N=np.array((LD(1),tiny,LD(0))))
    assert witness['interval']['gamma_right_width_xi']==0.
    report=audit_geometry_operands(raw,N,witness)
    assert report['source_Gamma_crosses']
    error=report['component_error_enclosures']['gamma_right_width_xi']
    assert Fraction(error['native_value'])>0
    assert Fraction(error['source_value_enclosure']['lower'])>0

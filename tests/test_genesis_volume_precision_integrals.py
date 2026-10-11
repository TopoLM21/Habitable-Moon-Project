"""Independent exact algebra and strict runtime precision regression checks."""
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_operator import isotropic_elasticity
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source, integrate_line, line_to_dict
from tectonics.genesis_volume_rational_fields import RationalFields, fraction
from tectonics.genesis_volume_precision_integrals import prepare_precision_integrals, MASS_REPAIR_TRIGGER
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients
from analysis.genesis_volume_integral_validation.independent_integrals import true_source_integral_enclosures


def fixture(radius=100):
    X = np.array(((0.,-5.,10.),(5.,-5.,10.),(0.,5.,10.)))
    volume=np.linalg.det(X)*(.95**3-.8**3)/6
    cohort=ConicalCohortGeometry(17,0,2,(1-.95**3)/.5,(1-.8**3)/.5,
        2*volume,volume,tuple(map(tuple,X)),.95,.8,(),'material','memory')
    dofs=(np.arange(18).reshape(3,6)-8)*1e-5
    section=prepare_cohort_material_section(cohort,dofs,(0.,0.,1.),9.03,conical_mass_coefficient=.5)
    raw=dict(section=asdict(section),elasticity_pa=isotropic_elasticity(3e8,.27),
        basis_xyz=np.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.))),
        memory0=(np.arange(18).reshape(3,6)-8)*1e-4,memory1=(np.arange(18).reshape(3,6)-7)*2e-4,
        seed_center_world_m=(1.,0.,9.),seed_radius_m=radius,source_identity='precision-test')
    W=np.eye(18)[:,:2];W[3,0]=-.125
    q=(np.arange(18)-8)*1e-6
    options=dict(added_basis=W,parent_basis=np.eye(18)[:,:3],
                 probes=[dict(name='saved-mean',q=q,a=q*.3,mean_z=q*.2,
                              mean_e0=raw['memory0']*.37,mean_e1=raw['memory1']*-.43)])
    N=np.array((.31,.2,.49),np.longdouble);N/=N.sum()
    return raw,prepare_source(raw),options,N


@pytest.mark.parametrize('radius',[100.,2.])
def test_runtime_coefficients_equal_separate_rational_oracle(radius):
    raw,source,options,N=fixture(radius)
    native=integrate_line(source,N,**options)
    assert native.interval is not None
    interval=native.interval
    actual=RationalFields(source).evaluate(interval)
    expected=exact_source_coefficients(raw,N,fraction(interval.lambda_mid),fraction(interval.lambda_halfwidth))
    for name in actual:
        assert np.array_equal(actual[name],expected[name]),name


@pytest.mark.parametrize('radius',[100.,2.])
def test_fresh_source_enclosures_and_errors_cover_independent_integrals(radius):
    raw,source,options,N=fixture(radius)
    native=integrate_line(source,N,**options)
    before=(source.parent_stiffness.tobytes(),source.parent_mass.tobytes())
    line=prepare_precision_integrals(source,raw,**options).evaluate(N)
    targets=true_source_integral_enclosures(raw,N,line_to_dict(native),**options)
    assert line.names==tuple(targets)
    for name,lo,hi,v,e in zip(line.names,line.enclosure.lower,line.enclosure.upper,line.values,line.errors):
        a,b=targets[name]
        assert fraction(lo)<=a<=b<=fraction(hi),name
        assert max(abs(fraction(v)-a),abs(fraction(v)-b))<=fraction(e),name
    assert before==(source.parent_stiffness.tobytes(),source.parent_mass.tobytes())
    assert not line.angular_accuracy_certified
    assert not line.physical_creation_admitted
    assert not line.available_for_creation_funding


def test_zero_budget_and_zero_scale_never_get_a_positive_floor():
    raw,source,options,N=fixture()
    prepared=prepare_precision_integrals(source,raw,**options,max_mass_fallback=0)
    assert prepared._failures([np.longdouble(0)],[np.longdouble('1e-40')])==[0]
    assert prepared._failures([np.longdouble(0)],[np.longdouble(0)])==[]
    exact_tol=Fraction(*float(2e-6).as_integer_ratio())
    assert prepared.tolerance==exact_tol


def test_refuse_relaxed_tolerance_and_unbounded_fallback():
    raw,source,options,N=fixture()
    with pytest.raises(ValueError,match='original'):
        prepare_precision_integrals(source,raw,**options,relative_tolerance=3e-6)
    with pytest.raises(ValueError,match='work cap'):
        prepare_precision_integrals(source,raw,**options,max_mass_fallback=325)


def test_internal_mass_repair_trigger_does_not_replace_acceptance_tolerance():
    raw,source,options,N=fixture()
    prepared=prepare_precision_integrals(source,raw,**options)
    values,errors=[np.longdouble(1)],[np.longdouble('1e-10')]
    assert prepared._failures(values,errors)==[]
    assert prepared._failures(values,errors,tolerance=MASS_REPAIR_TRIGGER)==[0]
    assert MASS_REPAIR_TRIGGER < prepared.tolerance


def test_literal_inputs_are_frozen_and_foreign_interval_is_refused():
    raw,source,options,N=fixture()
    prepared=prepare_precision_integrals(source,raw,**options)
    before=prepared.evaluate(N)
    raw['memory0'][:]=0; options['added_basis'][:]=0
    after=prepared.evaluate(N)
    assert np.array_equal(before.values,after.values)
    other=fixture()[1]
    with pytest.raises(ValueError,match='Foreign'):
        prepared.fields.evaluate(integrate_line(other,N).interval)

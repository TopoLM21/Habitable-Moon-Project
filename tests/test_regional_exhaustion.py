"""Complete consumption is a material transfer, not a numerical mass clamp."""
from copy import deepcopy
from decimal import Decimal,localcontext
import numpy as np
import pytest
from test_material_exchange import initialized,decimal_energy
from tectonics.material_exchange import MaterialExchangeBudget


@pytest.mark.parametrize('direction',[-np.inf,np.inf])
def test_empty_pool_moves_every_origin_and_sensible_enthalpy(direction):
    b,s,_,p,_=initialized()
    # Include both original basalt and newly supplied mantle material.
    vol=np.array(b.data['surface_volumes_km3'])+[0.,1234567.89]
    b.exchange([0.,1234567.89],[0.,0.],vol,p)
    before=deepcopy(b.data);energy=decimal_energy(b);tc=b.temperatures(p)[1]
    mass=b.pool_masses()[2]
    returned=float(np.nextafter(vol[1],direction))
    b.exchange([0.,0.],[0.,returned],[vol[0],0.],p)
    assert b.pool_masses()[2]==0.
    assert b.data['origin_hi'][2]==b.data['origin_lo'][2]==[0.,0.,0.]
    with localcontext() as ctx:
        ctx.prec=80
        assert abs(decimal_energy(b)-energy)<Decimal('1e10')
        q=sum(map(Decimal,b.data['advected_return_j']))-sum(map(Decimal,before['advected_return_j']))
        expected=Decimal(mass)*Decimal(p.mantle_heat_capacity_j_kg_k)*Decimal(tc)
        assert abs((q-expected)/expected)<Decimal('3e-16')
        for j in range(3):
            def total(data):
                return sum((Decimal(data[key][i][j]) for key in ('origin_hi','origin_lo') for i in range(3)),Decimal(0))
            assert abs(total(b.data)-total(before))<Decimal('1e-6')
    assert MaterialExchangeBudget.restore(b.checkpoint(),b.options).checkpoint()==b.checkpoint()


def test_empty_target_cannot_hide_an_unrequested_finite_transfer():
    b,s,_,p,_=initialized();before=b.checkpoint();vol=b.data['surface_volumes_km3']
    with pytest.raises(ValueError,match='Empty target'):
        b.exchange([0.,0.],[0.,vol[1]*(1-1e-6)],[vol[0],0.],p)
    assert b.checkpoint()==before


def test_exhausting_all_crust_exits_two_box_domain_atomically():
    b,s,_,p,_=initialized();before=b.checkpoint()
    with pytest.raises(ValueError,match='nonempty crust'):
        b.exchange([0.,0.],b.data['surface_volumes_km3'],[0.,0.],p)
    assert b.checkpoint()==before

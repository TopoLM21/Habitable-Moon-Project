"""70-digit check at an actual deformation/exchange endpoint, not a residual heat fix."""
from decimal import Decimal, localcontext
import json
import math
from verify import HERE, setup


def total_and_entropy(o):
    c=o.coupled;s=c.surface;p=s.params
    def D(x):return Decimal(float(x))
    with localcontext() as ctx:
        ctx.prec=70
        internal=sum(D(x) for k in ('energy_m_j','energy_c_j') for x in c.material.data[k])
        area=s.geometry(s.vertices)[0];R=D(p.radius_m);U=Decimal(0)
        for k,rho,frac in [('felsic',p.felsic_density_kg_m3,s.arrays['felsic_fraction']),
                           ('basalt',p.basalt_density_kg_m3,1-s.arrays['felsic_fraction'])]:
            for A,V in zip(area*frac,s.arrays[k+'_volume_m3']):
                if V==0:continue
                A=D(A);V=D(V);h=R*((1+3*V/(A*R))**(Decimal(1)/3)-1)
                U+=D(rho)*D(p.gravity_m_s2)*A*h*h*(Decimal('.5')+h/R*(Decimal(2)/3+h/R/4))
        net=sum(D(x) for x in o.data['net_volume_m3'])
        pressure=D(o.params.initial_pressure_pa)-net/D(o.params.compliance_m3_pa)
        E=D(o.params.compliance_m3_pa)*pressure**2/2
        masses=c.material.pool_masses();temperatures=c.material.temperatures(c.thermal_params)
        S=D(p.heat_capacity_j_kg_k)*(D(masses[0])*D(temperatures[0]).ln()+
                                   sum(D(x) for x in masses[1:])*D(temperatures[1]).ln())
        return internal+U+E,S


results=[]
for scenario,pressure in [('native',0.),('smooth',5e8)]:
    o=setup(scenario=scenario,sub=3,pressure=pressure)
    before_area=o.coupled.surface.geometry(o.coupled.surface.vertices)[0]
    o.coupled.step(1.)
    before,S0=total_and_entropy(o)
    o._exchange(before_area);o._validate()
    after,S1=total_and_entropy(o)
    with localcontext() as ctx:
        ctx.prec=70;delta=float(after-before);entropy=float(S1-S0)
    e=o.data['history'][-1]
    scale=max(abs(e['source_work_j']),abs(e['column_work_j']),e['heat_j'],1.)
    assert abs(delta)/scale<1e-12
    assert entropy>=0
    results.append(dict(scenario=scenario,pressure_initial_pa=pressure,
        independent_energy_delta_j=delta,relative_to_exchange=delta/scale,
        entropy_increment_j_k=entropy,transfer=e,
        scope='actual one-Myr geometry followed by isolated transfer; no simultaneous cooling in this entropy check'))
with (HERE/'independent_work.json').open('x') as f:json.dump(results,f,indent=2)
print(json.dumps(results),flush=True)

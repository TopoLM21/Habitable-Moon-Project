"""Conservative P0 reconstruction in material mass coordinate.

Cell means determine a representative phase, not a resolved subcell front.
Boundary temperatures are flux data and never replace cell-integrated energy.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class EnthalpyColumn:
    mass_kg:tuple
    energy_j:tuple
    density_kg_m3:float

    def __post_init__(self):
        if len(self.mass_kg)!=len(self.energy_j) or not self.mass_kg:raise ValueError('Cell coverage')
        if not math.isfinite(self.density_kg_m3) or self.density_kg_m3<=0:raise ValueError('Density')
        if any(not math.isfinite(x) or x<=0 for x in self.mass_kg):raise ValueError('Positive mass')
        if any(not math.isfinite(x) or x<0 for x in self.energy_j):raise ValueError('Nonnegative energy')
        if any(b<=a for a,b in zip(self.bounds[:-1],self.bounds[1:])):raise ValueError('Unresolved mass coordinate')

    @property
    def bounds(self):
        M=math.fsum(self.mass_kg)
        return (0.,)+tuple(math.fsum(self.mass_kg[:i])/M for i in range(1,len(self.mass_kg)))+(1.,)

    @property
    def specific_h(self):return tuple(e/m for m,e in zip(self.mass_kg,self.energy_j))

    def integrate(self,lo=0.,hi=1.):
        if not 0<=lo<=hi<=1:raise ValueError('Material interval')
        portions=[max(0.,min(hi,b)-max(lo,a))/(b-a) for a,b in zip(self.bounds[:-1],self.bounds[1:])]
        mass=math.fsum(f*m for f,m in zip(portions,self.mass_kg))
        return dict(mass_kg=mass,volume_m3=mass/self.density_kg_m3,enthalpy_j=math.fsum(f*e for f,e in zip(portions,self.energy_j)))

    def representative_phase(self,p):
        cp,L,ts,tl=p.silicate_heat_capacity_j_kg_k,p.silicate_latent_heat_j_kg,p.solidus_k,p.liquidus_k
        h=np.asarray(self.specific_h)
        T=np.where(h<=cp*ts,h/cp,np.where(h>=cp*tl+L,(h-L)/cp,(h+L*ts/(tl-ts))/(cp+L/(tl-ts))))
        phi=np.clip((T-ts)/(tl-ts),0,1)
        return dict(temperature_k=T.tolist(),melt_fraction=phi.tolist(),
          interpretation='P0 representative phase; unresolved subcell phase is not uniquely determined')

    def connected_front_bracket(self,surface_temperature,p):
        """Outer enclosure from means only; no claim of subcell continuity.

        A cell whose mean exceeds solidus enthalpy cannot be entirely solid.
        Means below it do not exclude unresolved hot pockets. Thus lower=0.
        """
        if not math.isfinite(surface_temperature) or surface_temperature<0:raise ValueError('Surface temperature')
        if surface_temperature>p.solidus_k:return dict(q_min=0.,q_max=0.,kind='surface_disconnected_not_material_melting')
        upper=next((self.bounds[i+1] for i,h in enumerate(self.specific_h) if h>p.silicate_heat_capacity_j_kg_k*p.solidus_k),1.)
        return dict(q_min=0.,q_max=upper,kind='unresolved_subcell_connectivity')


def require_resolved_mechanical_input(packet):
    # This implementation supplies no resolved-front certificate, even if an
    # external caller changes a readiness label in the diagnostic packet.
    raise ValueError('STOP: conservative means do not resolve inherited subcell material front')

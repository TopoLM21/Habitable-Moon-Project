"""Artificial solidus crossing for phase integration/restart tests."""
from copy import deepcopy
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from .synthetic import synthetic

def freezing_model():
    base=synthetic();cfg=deepcopy(base.config);s=deepcopy(base.state);i=len(cfg['mass'])-2;m=3e12;T=1410.;cp=1200.;latent=4e5;h=cp*T+latent*(T-1400)/600
    for name,value in [('mass',m),('cp',cp),('latent',latent),('offset',0.),('synthetic_power_w',-m*(h-cp*1390)/(25*365.25*86400))]:cfg[name].insert(i,value)
    cfg['edges']=[(a+(a>=i),b+(b>=i),G) for a,b,G in cfg['edges']]
    s['energy'].insert(i,m*h);s['memory']=np.zeros((base.nf,2,6)).tolist();s['omega']=[[0.,0.,1e-14]]
    cfg['phase_cells']=[dict(id='test-material-interval',node=i,face=0,q_lo=.1,q_hi=.2,mass_kg=m,volume_m3=m/3000,inertia=(np.eye(3)*3e25).tolist())]
    return SimplifiedGenesis(cfg,s),i


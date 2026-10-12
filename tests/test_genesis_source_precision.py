import json,math
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from tectonics.genesis import GenesisParameters,SECONDS_PER_MYR,outgoing_longwave_w_m2
from tectonics.genesis_tides import TidalParameters,TidalOrbitState
from tectonics.genesis_thermal_ports import surface_endpoint,apply_ports
from analysis.genesis_forced_coupled_validation.audit import audit_ports
from analysis.genesis_adaptive_control.backend import PARENT


def test_saved100_source_passes_unchanged_independent_audit():
    root=json.loads(PARENT.read_bytes());p=GenesisParameters(**root['source_packet']['source']['parameters']['thermal']);tp=TidalParameters(**root['source_packet']['source']['parameters']['tides'])
    packet=json.loads(Path('/workspace/genesis-exploratory-100-01/initial.json').read_text())['packet'];before=packet['thermal'];orbit=TidalOrbitState(**packet['orbit_after'])
    after,_,e=apply_ports(before,p,tp,orbit,before['time_myr']+100/1e6)
    assert audit_ports(before,after,e,p,tp)['status']=='PASS'
    assert before==packet['thermal']
    assert e['source_surface_endpoint_k']>=647.096


def test_saved_constant_olr_branch_against_70_digit_solution():
    root=json.loads(PARENT.read_bytes());p=GenesisParameters(**root['source_packet']['source']['parameters']['thermal'])
    th=json.loads(Path('/workspace/genesis-exploratory-100-01/initial.json').read_text())['packet']['thermal']
    E=th['energy_j'][-1];cap=th['mass_kg'][-1]*p.water_heat_capacity_j_kg_k;latent=th['mass_kg'][-1]*p.water_latent_heat_j_kg
    dt=((th['time_myr']+1e-4)-th['time_myr'])*SECONDS_PER_MYR;area_dt=dt*p.area_m2
    star=p.absorbed_stellar_flux_w_m2*p.area_m2*dt
    T=surface_endpoint(E,cap,latent,star,0.,area_dt,p)
    assert outgoing_longwave_w_m2(T,p)==282.
    with localcontext() as ctx:
        ctx.prec=70;D=Decimal.from_float
        exact=(D(E)+D(star)-D(area_dt)*D(282.)-D(latent))/D(cap)
        assert abs(D(T)-exact)<=D(math.ulp(T))


def test_manufactured_constant_branch_across_reservoir_scales():
    p=GenesisParameters()
    for cap in (1e10,1e20,1e25):
        latent=cap*500.;E=cap*1300+latent;dt_area=cap*.01;star=cap*2.
        T=surface_endpoint(E,cap,latent,star,0.,dt_area,p)
        with localcontext() as ctx:
            ctx.prec=70;D=Decimal.from_float
            exact=(D(E)+D(star)-D(dt_area)*D(282.)-D(latent))/D(cap)
            assert abs(D(T)-exact)<=2*D(math.ulp(T))
        residue=math.fsum((cap*T,latent,-E,-star,dt_area*282))
        assert abs(residue)<=8*(math.ulp(E)+math.ulp(cap*T+latent))

def test_nonlinear_hot_window_against_independent_decimal_bisection():
    from tectonics.genesis import SIGMA
    p=GenesisParameters();cap=1e20;latent=cap*500.;E=cap*2100.+latent;star=cap*2.;area_dt=cap*.05
    T=surface_endpoint(E,cap,latent,star,0.,area_dt,p)
    assert T>p.hot_window_temperature_k
    with localcontext() as ctx:
        ctx.prec=70;D=Decimal.from_float
        def f(x):return D(cap)*x+D(latent)-D(E)-D(star)+D(area_dt)*(D(282.)+D(SIGMA)*(x-D(1600.))**4)
        lo,hi=D(1600.),D(3000.)
        for _ in range(240):
            mid=(lo+hi)/2
            if f(mid)>0:hi=mid
            else:lo=mid
        assert abs(D(T)-(lo+hi)/2)<=2*D(math.ulp(T))
    residual=math.fsum((cap*T,latent,-E,-star,area_dt*outgoing_longwave_w_m2(T,p)))
    assert abs(residual)<=8*(math.ulp(E)+math.ulp(cap*T+latent))

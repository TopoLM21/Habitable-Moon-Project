"""Opt-in conservative thermal reservoirs. No hook into legacy evolution.

Global thermal energy is authoritative at migration; measured column profiles
and surface water are retained. Deep mantle is the explicitly remapped residual.
"""
from dataclasses import dataclass,asdict,replace
import hashlib,json,math
from pathlib import Path
from scipy.optimize import brentq
from .genesis import GenesisParameters,ENERGY_SCALE,temperatures,silicate_specific_enthalpy,vapor_column_kg_m2

FORMAT='genesis-coupled-thermal-0.1'


@dataclass(frozen=True)
class Reservoir:
    identity:str
    material:str
    mass_kg:float
    energy_j:float


@dataclass(frozen=True)
class CoupledThermalState:
    time_myr:float
    cells:tuple
    parameters_json:str
    origin_json:str
    reference_total_j:float
    numerical_sink_j:float=0.
    historical_opening_balance:object=None
    ledger_scope:str='since_reference_state_only'
    format:str=FORMAT


def total(s):return math.fsum(c.energy_j for c in s.cells)


def validate(s):
    if s.format!=FORMAT or s.historical_opening_balance is not None or s.ledger_scope!='since_reference_state_only':raise ValueError('Version/history')
    if not math.isfinite(s.time_myr) or s.time_myr<0:raise ValueError('Clock')
    if len({c.identity for c in s.cells})!=len(s.cells):raise ValueError('Overlapping reservoir identities')
    p=GenesisParameters(**json.loads(s.parameters_json));p.validate()
    for c in s.cells:
        if c.material not in ('silicate','water','elastic','kinetic'):raise ValueError('Material')
        if not all(math.isfinite(v) for v in (c.mass_kg,c.energy_j)) or c.energy_j<0:raise ValueError('Energy/mass')
        if (c.material in ('elastic','kinetic') and c.mass_kg!=0) or (c.material in ('silicate','water') and c.mass_kg<=0):raise ValueError('Mass ownership')
    origin=json.loads(s.origin_json)
    if 'silicate_mass_kg' in origin and not math.isclose(math.fsum(c.mass_kg for c in s.cells if c.material=='silicate'),origin['silicate_mass_kg'],rel_tol=2e-13):raise ValueError('Silicate mass inventory')
    if 'water_mass_kg' in origin and not math.isclose(math.fsum(c.mass_kg for c in s.cells if c.material=='water'),origin['water_mass_kg'],rel_tol=2e-13):raise ValueError('Water mass inventory')
    for c in s.cells:
        if c.material=='water' and temperature(c,p)<273.16:raise ValueError('Ice is outside inherited water closure')
    if not math.isfinite(s.reference_total_j) or s.reference_total_j<=0 or not math.isfinite(s.numerical_sink_j) or s.numerical_sink_j<0:raise ValueError('Ledger')
    if abs(total(s)+s.numerical_sink_j-s.reference_total_j)>2e-12*s.reference_total_j:raise ValueError('Energy conservation')
    return s


def specific_h(T,material,p):
    if material=='silicate':return silicate_specific_enthalpy(T,p)
    if material=='water':
        # Water terms of the existing surface law, without subtractive cancellation.
        return p.water_heat_capacity_j_kg_k*T+p.water_latent_heat_j_kg*vapor_column_kg_m2(T,p)/p.water_column_kg_m2
    raise ValueError('Mechanical energy has no temperature')


def temperature(c,p):
    h=c.energy_j/c.mass_kg
    if h==0:return 0.
    high=max(3000.,h/min(p.silicate_heat_capacity_j_kg_k,p.water_heat_capacity_j_kg_k)+1.)
    return brentq(lambda T:specific_h(T,c.material,p)-h,0.,high,xtol=1e-9)


def conduct(s,edges,dt_seconds):
    """Explicit FV exchange G*(Ta-Tb), G W/K; CFL uses minimum sensible cp.

    Edges must be unique unordered pairs. No imposed-temperature reservoirs.
    Latent heat is already in h(T). Mechanical reservoirs cannot conduct.
    """
    validate(s);p=GenesisParameters(**json.loads(s.parameters_json))
    if not math.isfinite(dt_seconds) or dt_seconds<=0:raise ValueError('Positive dt')
    cells={c.identity:c for c in s.cells};rates={k:0. for k in cells};seen=set()
    for a,b,G in edges:
        key=frozenset((a,b))
        if a==b or key in seen or not math.isfinite(G) or G<=0:raise ValueError('Duplicate/invalid edge')
        seen.add(key)
        if cells[a].material not in ('silicate','water') or cells[b].material not in ('silicate','water'):raise ValueError('Nonthermal edge')
        rates[a]+=G;rates[b]+=G
    for k,c in cells.items():
        cp=p.water_heat_capacity_j_kg_k if c.material=='water' else p.silicate_heat_capacity_j_kg_k
        if dt_seconds*rates[k]>c.mass_kg*cp:raise ValueError('Explicit conduction CFL')
    energy={k:c.energy_j for k,c in cells.items()};ledger=[]
    for a,b,G in edges:
        q=dt_seconds*G*(temperature(cells[a],p)-temperature(cells[b],p))
        energy[a]-=q;energy[b]+=q
        ledger.append(dict(donor=a,recipient=b,energy_j=q,donor_entry_j=-q,recipient_entry_j=q))
    from .genesis import SECONDS_PER_MYR
    result=replace(s,time_myr=s.time_myr+dt_seconds/SECONDS_PER_MYR,
                   cells=tuple(replace(c,energy_j=energy[c.identity]) for c in s.cells))
    return validate(result),ledger


def mechanical_exchange(s,*,kind,mechanical_id,recipient=None,energy_j):
    """Explicit located Maxwell conversion, or separate numerical loss.

    Caller must supply the co-located column for Maxwell. No drag/damage policy.
    This is a transfer primitive, not a new mechanical constitutive solver.
    """
    validate(s)
    if not math.isfinite(energy_j) or energy_j<0:raise ValueError('Transfer amount')
    cells={c.identity:c for c in s.cells};c=cells[mechanical_id]
    if kind=='maxwell':
        if c.material!='elastic' or recipient is None or mechanical_id!='elastic:'+recipient or not recipient.startswith('column:') or cells[recipient].material!='silicate':raise ValueError('Located Maxwell recipient required')
        cells[recipient]=replace(cells[recipient],energy_j=cells[recipient].energy_j+energy_j)
    elif kind=='numerical':
        if recipient is not None or c.material not in ('elastic','kinetic'):raise ValueError('Numerical loss cannot heat')
    else:raise ValueError('Drag/damage/thermal-work coupling policy not implemented')
    cells[mechanical_id]=replace(c,energy_j=c.energy_j-energy_j)
    return validate(replace(s,cells=tuple(cells[c.identity] for c in s.cells),
                            numerical_sink_j=s.numerical_sink_j+(energy_j if kind=='numerical' else 0.)))


def migrate(raw,expected_sha):
    """Conservative projection, not reconstruction of unknown past exchanges."""
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=expected_sha:raise ValueError('Legacy raw SHA')
    root=json.loads(raw);r=root['runtime_state'];t=r['thermal']
    params=root['source_packet']['source']['parameters']['thermal'];p=GenesisParameters(**params)
    y=t['global_state']['energy'];tm,ts=temperatures(y,p)
    water_mass=p.water_column_kg_m2*p.area_m2
    water_energy=water_mass*specific_h(ts,'water',p)
    mass=t['layer_mass_kg'];enthalpy=t['column_enthalpy']
    if len(mass)!=len(enthalpy) or any(len(a)!=len(b) for a,b in zip(mass,enthalpy)):raise ValueError('Column coverage')
    columns=[Reservoir(f'column:{i}:{j}','silicate',m,m*enthalpy[i][j]) for i,row in enumerate(mass) for j,m in enumerate(row)]
    column_mass=math.fsum(c.mass_kg for c in columns);column_energy=math.fsum(c.energy_j for c in columns)
    deep_mass=p.silicate_column_kg_m2*p.area_m2-column_mass
    thermal_total=math.fsum(y[:2])*ENERGY_SCALE*p.area_m2
    deep_energy=thermal_total-water_energy-column_energy
    if deep_mass<=0 or deep_energy<=0:raise ValueError('Conservative projection has no positive deep reservoir')
    cells=tuple([Reservoir('deep_mantle','silicate',deep_mass,deep_energy),*columns,Reservoir('surface_water','water',water_mass,water_energy)])
    origin=dict(legacy_raw_sha256=sha,legacy_format=root['format'],policy='global_total_column_profile_surface_water_preserved_deep_residual',
      legacy_thermal_total_j=thermal_total,column_mass_kg=column_mass,water_mass_kg=water_mass,
      silicate_mass_kg=p.silicate_column_kg_m2*p.area_m2,
      legacy_mantle_temperature_k=tm,new_deep_temperature_k=temperature(cells[0],p),
      deep_vs_legacy_mantle_energy_j=deep_energy-y[0]*ENERGY_SCALE*p.area_m2,
      representation_change_not_physical_flux=True,legacy_history_retained_by_sha=True,
      mechanical_attachment='not_attached',physical_column_geometry_mapping='pending',historical_opening_balance=None)
    return validate(CoupledThermalState(t['time_myr'],cells,json.dumps(params,sort_keys=True),json.dumps(origin,sort_keys=True),thermal_total))


def save(s,path):
    validate(s);body=asdict(s);encoded=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)
    packet=dict(state=body,sha256=hashlib.sha256(encoded.encode()).hexdigest())
    with Path(path).open('x') as f:json.dump(packet,f,sort_keys=True,indent=2);f.write('\n')


def load(path,*,legacy_path=None):
    packet=json.loads(Path(path).read_bytes());body=packet['state']
    if hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=packet['sha256']:raise ValueError('Seal')
    state=validate(CoupledThermalState(**{**body,'cells':tuple(Reservoir(**c) for c in body['cells'])}))
    origin=json.loads(state.origin_json)
    if 'legacy_raw_sha256' in origin:
        if legacy_path is None or hashlib.sha256(Path(legacy_path).read_bytes()).hexdigest()!=origin['legacy_raw_sha256']:raise ValueError('Authenticated legacy history required')
    return state

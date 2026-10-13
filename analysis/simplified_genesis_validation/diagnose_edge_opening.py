"""Read-only linear contact diagnosis from saved artificial edge state.

Capture the actual production assembly before its refusal. Solve its unilateral
QP separately. Never commit coordinates, heat, time or material to the model.
"""
from pathlib import Path
import json,sys,time
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_edges import advance as edge_advance
from tectonics.simplified_genesis_contact import unilateral_projection
from tectonics.genesis_transfer_bridge import encoded,digest
from tectonics.spherical_polygons import intersect_convex,polygon_area


class CapturedAssembly(Exception):pass


def capture(model,years):
    captured={}
    def trace(frame,event,arg):
        if frame.f_code is edge_advance.__code__ and event=='line' and 'contact_error' in frame.f_locals:
            captured.update(frame.f_locals)
            raise CapturedAssembly()
        return trace
    before=encoded(model.state);prior=sys.gettrace()
    try:
        sys.settrace(trace)
        edge_advance(model,years)
    except CapturedAssembly:pass
    finally:sys.settrace(prior)
    assert captured and encoded(model.state)==before
    return captured


def linear_balance(p,q,multipliers):
    model=p['model'];dt=p['dt'];mass=p['mass'];old=p['old'];C=p['C']
    drive=np.zeros_like(old);drive[:,0,:]=(p['B']@q).reshape(model.nf,6)
    new=p['A']@(old+drive);strain=np.einsum('fni,fij->fnj',model.P,new)
    stress=np.einsum('fij,fnj->fni',C,strain)
    heat=float(dt*np.sum(model.weights*p['rates']*np.sum(strain*stress,axis=2)))
    energy=lambda e:float(.5*np.einsum('fai,fab,fij,fbj->',e.astype(np.longdouble),model.G.astype(np.longdouble),C.astype(np.longdouble),e.astype(np.longdouble)))
    vel=p['R']*(p['P']@q).reshape(model.nf,3,3)/dt
    K=lambda v:float(.5*np.sum(mass[:,None,None]/3*v**2))
    work=float(p['external']@q);drag=float(q@(p['D']@q)/dt)
    contact=float(multipliers@(p['J']@q));dU=energy(new)-energy(old)
    residual=K(vel)-K(p['oldv'])+dU+heat+drag+energy(new-old)+K(vel-p['oldv'])-work-contact
    scale=max(abs(work),abs(dU),heat,1.)
    return dict(residual_j=residual,scale_j=scale,relative=abs(residual)/scale,
                maxwell_heat_j=heat,basal_heat_j=drag,contact_work_j=contact,
                full_thermal_damage_balance_checked=False)


def main():
    start=time.perf_counter();path=Path('/workspace/simplified-genesis-edges-01/edge_step_1.json')
    pin=digest(path.read_bytes());model=SimplifiedGenesis.load(path)
    state_pin=digest(encoded(model.state));original_pressure=model.config['edge_prescribed_compressive_pressure_pa']
    model.config['edge_prescribed_compressive_pressure_pa']=0.
    p=capture(model,.01);H=p['H'].toarray();J=p['J'];rhs=p['rhs'];scale=p['scale']
    free=np.linalg.solve(H/scale,rhs/scale);amplitude=max(float(np.linalg.norm(free,np.inf)),np.finfo(float).tiny)
    qscaled,lam_scaled=unilateral_projection(H/scale,rhs/(scale*amplitude),J)
    q=qscaled*amplitude;lam=lam_scaled*scale*amplitude;g=J@q
    stationarity=H@q-rhs-J.T@lam
    stat_scale=max(np.linalg.norm(rhs),np.linalg.norm(H@q),np.linalg.norm(J.T@lam),1.)
    gap_tolerance=1e-10*max(np.linalg.norm(q),amplitude)
    open_mask=g>gap_tolerance
    assert np.min(g)>=-gap_tolerance and np.min(lam)>=0
    assert np.linalg.norm(stationarity)/stat_scale<1e-10
    complementarity=float(np.max(abs(lam*g))/max(np.linalg.norm(lam)*np.linalg.norm(q),1.))
    assert complementarity<1e-10
    face_delta=(p['P']@q).reshape(model.nf,3,3)
    points=p['mesh'].vertices[p['mesh'].faces]+face_delta
    points/=np.linalg.norm(points,axis=2)[:,:,None]
    a,b,c=points[:,0],points[:,1],points[:,2]
    areas=2*np.arctan2(np.sum(a*np.cross(b,c),axis=1),1+np.sum(a*b+b*c+c*a,axis=1))
    worst=0.;positive=0
    for fa,fb,_,_ in p['mesh'].shared_edges:
        if model.state['owner'][fa]==model.state['owner'][fb]:continue
        poly=intersect_convex(points[fa],points[fb]);area=polygon_area(poly)*model.config['radius_km']**2 if len(poly) else 0.
        worst=max(worst,area);positive+=int(area>0)
    assert digest(path.read_bytes())==pin and digest(encoded(model.state))==state_pin
    result=dict(status='LINEAR_UNILATERAL_OPENING_DIAGNOSED_NOT_A_FINITE_STEP',source=str(path),source_sha256=pin,
        original_artificial_pressure_pa=original_pressure,diagnostic_pressure_pa=0.,new_planetary_steps=0,committed_steps=0,
        equality_solver=dict(enforced_condition='J q = 0 on every contact',normal_rows=len(J),normal_rank=p['rank'],
            nonnegative_reaction_fit_relative=p['contact_error'],failure_before_geometry_and_energy=True),
        unilateral_solver=dict(enforced_condition='J q >= 0; lambda >= 0; lambda*(J q) = 0',
            stationarity_relative=float(np.linalg.norm(stationarity)/stat_scale),complementarity_relative=complementarity,
            minimum_multiplier_j=float(lam.min()),opening_rows=int(open_mask.sum()),closed_rows=int((~open_mask).sum()),
            maximum_first_order_opening_m=float(g.max()*p['R']),minimum_first_order_gap_m=float(g.min()*p['R']),
            relative_difference_from_closed_candidate=float(np.linalg.norm(q-p['q'])/np.linalg.norm(q))),
        reduced_model_work=linear_balance(p,q,lam),
        finite_geometry_unaccepted=dict(sum_triangle_area_deficit_km2=float((4*np.pi-areas.sum())*model.config['radius_km']**2),
            maximum_neighbor_overlap_km2=worst,positive_neighbor_overlaps=positive,
            swept_nopenetration_proved=False,independent_material_vertices_required=True,
            void_filling_required_for_this_diagnostic=False),
        active_physics=dict(alpha=0.,damage_frozen=True,thermal_ports=False,conduction=False,phase_birth=False,
            material_sink=False,temperature_dependent_relaxation_implemented=True,
            activation_energy_j_mol=model.shell['activation_energy_j_mol'],mechanical_maxwell_and_drag_heat=True),
        seconds=time.perf_counter()-start)
    with Path(__file__).with_name('edge_opening_diagnosis.json').open('xb') as f:f.write(encoded(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Bounded linked controls; unique names and retained restart evidence."""
from dataclasses import asdict, replace
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tectonics.conforming_surface import ConformingSurface, SurfaceParameters, SurfaceControl
from tectonics.regional_deformation import DeformingSurface, DilationParameters
from tectonics.deformation_exchange import DeformationExchange, CoupledParameters
from tectonics.material_exchange import MaterialExchangeBudget
from tectonics.lithosphere import LithosphereState
from tectonics.thermal import ThermalParameters, initialize_thermal_state
from tectonics.tides import constant_eccentricity
from tectonics.strain_exchange import StrainExchange, PressureParameters, boundary_zone_fraction

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('previous_coupled_controls', ROOT/'analysis/regional_deformation_20261113/verify.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)


def setup(scenario='native', sub=3, dt=1., pressure=0., compliance=1e9, width=1e6, efficiency=1., quadrature=2):
    if scenario == 'native':
        c = previous.setup('native', sub, 1e6, dt)
    else:
        p = SurfaceParameters(); control = replace(SurfaceControl(), max_step_myr=dt)
        base = previous.prior.synthetic(scenario, sub, p, control)
        mesh = base.mesh(); area = mesh.physical_cell_areas_km2(p.radius_m/1000)
        state = LithosphereState(0., base.arrays['owner'].astype(np.int32), np.zeros(len(area), np.int8),
            np.ones(len(area)), base.arrays['basalt_volume_m3']/1e9/area, np.zeros(len(area)))
        tp = ThermalParameters(); thermal = initialize_thermal_state(.5, p.radius_m/1000, p.gravity_m_s2, tp)
        b, state, thermal = MaterialExchangeBudget.initialize(mesh, state, thermal, .5, p.radius_m/1000, p.gravity_m_s2, tp)
        arrays = deepcopy(base.arrays)
        arrays['basalt_volume_m3'] = state.oceanic_volume_km3*1e9
        arrays['basalt_age_moment_m3_myr'] = arrays['basalt_volume_m3']*40
        base = ConformingSurface(mesh, arrays, p, control, b.data['energy_c_j'], base.data['provenance'])
        s = DeformingSurface.from_surface(base, DilationParameters(1e6))
        c = DeformationExchange(s, b, thermal, tp, constant_eccentricity(0.), CoupledParameters(step_myr=dt))
    zone = boundary_zone_fraction(c.surface, width, quadrature)
    params = PressureParameters(compliance_m3_pa=compliance, initial_pressure_pa=pressure, exchange_fraction=efficiency)
    return StrainExchange(c, zone, params, dict(scenario=scenario, band_width_m=width,
        band_quadrature_refinement=quadrature, scope='new pressure initialization and material band; prior states unchanged'))


def run(name, scenario='native', sub=3, dt=1., pressure=0., compliance=1e9,
        width=1e6, efficiency=1., quadrature=2, end=20., resume=None):
    path = HERE/f'{name}.json'; saved = HERE/'states'/name
    if path.exists() or saved.exists(): raise FileExistsError(name)
    started = time.perf_counter()
    o = StrainExchange.load(HERE/'states'/resume) if resume else setup(scenario, sub, dt, pressure, compliance, width, efficiency, quadrature)
    initial = o.diagnostics(); initial_state = o.checkpoint(); history = [initial]
    then = time.perf_counter(); status = 'complete'; failure = None
    try:
        while o.coupled.thermal.time_myr < end:
            o.step(min(o.coupled.options.step_myr, end-o.coupled.thermal.time_myr))
            history.append(o.diagnostics())
    except (ValueError, RuntimeError) as exc:
        status = 'validity_or_numerical_limit'; failure = str(exc)
    elapsed = time.perf_counter()-then
    o.save(saved)
    assert StrainExchange.load(saved).checkpoint() == o.checkpoint()
    s = o.coupled.surface; final = o.diagnostics()
    coverage = previous.prior.sampled_coverage(s)[0]
    for k in ('owner', 'felsic_fraction', 'felsic_volume_m3', 'potential'):
        assert s.arrays[k].tolist() == initial_state['coupled']['surface']['arrays'][k]
    report = dict(case=name, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'tectonics/strain_exchange.py','tectonics/deformation_exchange.py','tectonics/regional_deformation.py',
            'tectonics/conforming_surface.py','tectonics/material_exchange.py')},
        status=status, failure=failure, scenario=scenario, cells=len(s.faces), dt_myr=dt,
        requested_end_myr=end, initial=initial, final=final, history=history,
        initial_strain=previous.strain_summary(DeformationExchange.restore(initial_state['coupled']).surface),
        final_strain=previous.strain_summary(s), coverage=coverage, exact_save_load=True,
        pressure_parameters=asdict(o.params), source_provenance=o.data['provenance'],
        integration_wall_s=elapsed, wall_s=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        snapshot_bytes=sum(p.stat().st_size for p in saved.iterdir()),
        max_work_identity_residual_j=max((abs(e['work_identity_residual_j']) for e in o.data['history']),default=0.),
        scope='conditional strain-gated basalt supply/return; no melting law, fracture or new continents')
    with path.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('case','status','failure','wall_s')}),flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('name')
    parser.add_argument('--scenario',choices=['native','smooth','oblique'],default='native')
    parser.add_argument('--sub',type=int,default=3);parser.add_argument('--dt',type=float,default=1.)
    parser.add_argument('--pressure',type=float,default=0.);parser.add_argument('--compliance',type=float,default=1e9)
    parser.add_argument('--width',type=float,default=1e6);parser.add_argument('--efficiency',type=float,default=1.)
    parser.add_argument('--quadrature',type=int,default=2);parser.add_argument('--end',type=float,default=20.)
    parser.add_argument('--resume');args=parser.parse_args();run(**vars(args))

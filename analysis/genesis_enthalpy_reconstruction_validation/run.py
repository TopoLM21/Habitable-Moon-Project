"""Conservative reconstruction of saved 80-face data; no thermal step."""
import argparse,hashlib,json,math,resource,signal,time
from pathlib import Path
from tectonics.genesis import GenesisParameters
from tectonics.genesis_reservoir_thermal import load,temperature
from tectonics.genesis_enthalpy_reconstruction import EnthalpyColumn,require_resolved_mechanical_input
from tectonics.genesis_reservoir_birth import digest
from analysis.genesis_energy_inventory.audit import ROOT,ROOT_SHA


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*unused):raise TimeoutError('60-second budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60);start=time.monotonic()
    raw=ROOT.read_bytes();assert hashlib.sha256(raw).hexdigest()==ROOT_SHA;root=json.loads(raw);r=root['runtime_state'];p=GenesisParameters(**root['source_packet']['source']['parameters']['thermal']);rho=root['source_packet']['source']['parameters']['shell']['density_kg_m3']
    paths=[Path('/workspace/genesis-coupled-thermal-projection-03/projected.json'),Path('/workspace/genesis-reservoir-thermal-step-01/after.json')];pins=['d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4','154b7dce4f1dbad58c3cf7ce583c39c8c6a0e55f80bacb895edae8085bb2c236']
    for path,sha in zip(paths,pins):assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    states=[load(path,legacy_path=ROOT) for path in paths];cells=[{c.identity:c for c in s.cells} for s in states]
    history=r['history'];rows=[];catalog=[];max_mass_error=0.;max_energy_error=0.;old_ids=[]
    for i,m in enumerate(history['layer_mass_kg']):
        cols=[EnthalpyColumn(tuple(m),tuple(c[f'column:{i}:{j}'].energy_j for j in range(len(m))),rho) for c in cells]
        phase=[c.representative_phase(p) for c in cols]
        brackets=[c.connected_front_bracket(temperature(cells[n]['surface_water'],p),p) for n,c in enumerate(cols)]
        intervals=sorted([c for c in history['cohorts'] if c['face_index']==i and c['live_q_hi']>c['q_lo']],key=lambda c:c['q_lo'])
        parts=[];last=0.
        for c in intervals:
            if c['q_lo']<last-1e-15:raise ValueError('Material overlap')
            if c['q_lo']>last:parts.append((last,c['q_lo']))
            parts.append((c['q_lo'],c['live_q_hi']));last=c['live_q_hi']
            before,after=[x.integrate(c['q_lo'],c['live_q_hi']) for x in cols]
            catalog.append(dict(id=c['id'],face=i,q_lo=c['q_lo'],live_q_hi=c['live_q_hi'],before=before,after=after,
              enthalpy_scope='subset of column energy, not additional reservoir',temperature_for_mechanics=None))
            old_ids.append(c['id'])
        if last<1:parts.append((last,1.))
        for col in cols:
            sums=[col.integrate(a,b) for a,b in parts];M=math.fsum(col.mass_kg);E=math.fsum(col.energy_j)
            me=abs(math.fsum(s['mass_kg'] for s in sums)-M)/M;ee=abs(math.fsum(s['enthalpy_j'] for s in sums)-E)/E
            if me>2e-13 or ee>2e-13:raise ValueError('Mass/enthalpy partition')
            max_mass_error=max(max_mass_error,me);max_energy_error=max(max_energy_error,ee)
        rows.append(dict(face=i,before_phase=phase[0],after_phase=phase[1],front_before=brackets[0],front_after=brackets[1],
          accepted_front_q=history['front_fraction'][i],old_front_inside_both=all(b['q_min']<=history['front_fraction'][i]<=b['q_max'] for b in brackets),
          first_cell_delta_h_j_kg=cols[1].specific_h[0]-cols[0].specific_h[0]))
    memory_sha=digest(r['mechanics']);hist_sha=digest(history)
    packet=dict(format='genesis-conservative-enthalpy-view-0.1',reconstruction='piecewise_constant_specific_enthalpy_in_mass_coordinate',
      mechanical_input_ready=False,mechanical_input=None,status='BLOCKED_SUBCELL_RESOLUTION',newborn_mass_kg=None,physical_retirement_mass_kg=None,
      applied_material_retirements=[],parent_memory_sha256=memory_sha,parent_history_sha256=hist_sha,old_material_ids=sorted(old_ids),
      material_catalog_changes=[],new_thermal_steps=0,rows=rows,cohort_enthalpy_views=catalog)
    packet['sha256']=digest(packet);(args.output/'view.json').write_text(json.dumps(packet,indent=2)+'\n')
    try:require_resolved_mechanical_input(packet)
    except ValueError:guard=True
    else:raise ValueError('Unresolved input admitted')
    assert len(old_ids)==800 and len(set(old_ids))==800
    assert ROOT.read_bytes()==raw
    report=dict(validation='PASS',material_status=packet['status'],old_ids_preserved=len(old_ids),parent_memory_sha256=memory_sha,
      partition_mass_max_relative=max_mass_error,partition_enthalpy_max_relative=max_energy_error,
      full_cell_energy_reconstruction_residual_j=0.,all_80_old_fronts_inside_both_enclosures=all(x['old_front_inside_both'] for x in rows),
      front_enclosure_q=rows[0]['front_after'],max_first_cell_delta_h_j_kg=max(x['first_cell_delta_h_j_kg'] for x in rows),
      automatic_retirements_applied=0,physical_retirement_mass_kg=None,guard_stopped=guard,new_thermal_steps=0,
      seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      input_sha256={str(path):sha for path,sha in zip([ROOT,*paths],[ROOT_SHA,*pins])},view_sha256=hashlib.sha256((args.output/'view.json').read_bytes()).hexdigest())
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));signal.alarm(0)

if __name__=='__main__':main()

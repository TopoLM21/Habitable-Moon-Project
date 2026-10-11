"""Replay the saved physical parent without evaluating any radial source."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import json
from analysis.genesis_volume_directional_validation.experiment import Cell,cell_key
from analysis.genesis_volume_ray_validation.experiment import gauss
from analysis.genesis_volume_residual_diagnosis.target_check import diagnose


def test_saved_physical_parent_selects_better_direction_without_new_calls():
    path=Path(__file__).resolve().parents[1]/'analysis/genesis_volume_residual_validation/target_nodes.json'
    saved=json.loads(path.read_bytes());c=Cell(0,F(1,2),F(1),F(1,2),F(1),2)
    plans={(cell_key(c),(n,n)):[] for n in (3,5)}
    for r in saved['node_requests']:
        plans[cell_key(c),(r['order'],r['order'])].append((tuple(map(F,r['N'])),F(r['weight']),r['band']))
    e=SimpleNamespace(nodes=33,requests=34,planner=SimpleNamespace(cache={}),plans=plans,
                      rules={n:gauss(n) for n in (3,5)},samples={
                          tuple(map(F,r['N'])):SimpleNamespace(lower=list(map(F,r['lower'])),upper=list(map(F,r['upper'])))
                          for r in saved['samples']})
    d,_,_=diagnose(e,c)
    assert (e.nodes,e.requests,len(e.planner.cache))==(33,34,0)
    assert len(e.samples)==33
    assert d['t'][0]>9*d['s'][0]>0
    assert d['t'][1]==d['s'][1]==0
    assert all(d[a]==list(map(F,saved['directions'][a])) for a in ('s','t'))

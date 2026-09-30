"""Compact results from the accepted default, separate from upper-bound trials."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    paths = {age: HERE / ('final' if age <= 50 else 'final_after_seed_fix')
             / 'corrected_mechanics' / f'elapsed_{age:04d}' for age in (5,50,100,200,400)}
    cases = []
    for age,path in paths.items():
        report = read(path/'continuation.json')
        assert report['status'] == 'completed' and all(report['checks'].values())
        assert report['young_slab_force_model'] == 'disabled_pending_closure' if age >= 100 else True
        cases.append(dict(elapsed_myr=age,path=str(path),
            mean_speed_mm_yr=report['final_mean_surface_speed_km_myr'],
            max_speed_mm_yr=report['final_max_surface_speed_km_myr'],
            plates=report['final_plate_count'],transport_commits=report['transport_commits'],
            checks=report['checks'],thermal_residual=report['history'][-1]['thermal_energy_relative_residual'],
            material_residual=report['material_ledger']['relative_volume_residual'],
            accepted_slab_inventory=report.get('accepted_slab_inventory')))
    final = read(paths[400]/'continuation.json')
    baseline_path = ROOT/'analysis/plate_velocity_validation/experiments/velocity_least_squares/elapsed_0400'
    baseline = read(baseline_path/'continuation.json')
    original = ROOT/'results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz'
    trace_path = paths[400].with_suffix('.dynamics.jsonl')
    last = json.loads(trace_path.read_text(encoding='utf-8').splitlines()[-1])['trace']
    total = sum(np.array(last[k]) for k in ('basal_driving_torque_nm','ridge_torque_nm','slab_torque_nm'))
    residual = np.array(last['target_torque_residual_nm'])
    scale = max(float(np.linalg.norm(total)),1.)
    result = dict(scope='Default prescribed basal SI + passive ridge; slab force disabled pending closure.',
        cases=cases,baseline_400_mean_speed_mm_yr=baseline['final_mean_surface_speed_km_myr'],
        original_starter_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        final_target_torque_relative_residual=float(np.linalg.norm(residual)/scale),
        final_power_balance_residual_w=last['total_source_power_w']-last['basal_drag_dissipation_w']-last['transient_net_power_w'],
        final_baseline_mantle_temperature_difference_k=final['history'][-1]['mantle_temperature_k']-baseline['history'][-1]['mantle_temperature_k'],
        temporal_and_grid_sensitivity=read(HERE/'convergence/comparison.json'),
        limitations=final['limitations'])
    (HERE/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for report,label,color in ((baseline,'Прежняя LS-механика','#687384'),
        (final,'SI: базальная тяга и хребты; slab pull выключен','#1b708c')):
        history=report['history']
        t=np.array([r['time_myr']-report['import']['origin_time_myr'] for r in history])
        axes[0].plot(t,[r['mean_surface_speed_km_myr'] for r in history],label=label,color=color,lw=2)
        axes[1].plot(t,[r['transport_commits'] for r in history],color=color,lw=2)
    axes[0].set_ylabel('Средняя скорость, мм/год')
    axes[1].set_ylabel('Число растровых переносов')
    for ax in axes:
        ax.set_xlabel('Время после первого разделения, млн лет')
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8,loc='best')
    fig.suptitle('Проверка реализации; физическая модель ещё неполна',fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE/'speed_and_transport.png',dpi=170)
    print(json.dumps({k:result[k] for k in ('baseline_400_mean_speed_mm_yr','final_target_torque_relative_residual','final_baseline_mantle_temperature_difference_k')}))
    print(json.dumps(cases,ensure_ascii=False))


if __name__ == '__main__':
    main()

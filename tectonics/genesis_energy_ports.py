"""Explicit open-system accounting. No temperature or mechanical-state updates."""
import math


def boundary_budget(ledger, thermal_increment_j, *, interface_in_control_volume=False):
    """All entries are joules. Positive port energy enters the chosen domain.

    Drag is either exported to an unresolved interface or included as its
    nonthermal accounting store, never both. Damage release is an unresolved
    nonthermal store, not a claimed free-energy potential or a source for healing.
    Maxwell and thermal work are internal transfers and must NOT be added here.
    """
    keys=('external_work_j','physical_drag_j','damage_energy_change_j',
          'numerical_kinetic_j','numerical_memory_j','kinetic_after_j',
          'kinetic_before_j','elastic_after_j','elastic_before_j')
    if any(not math.isfinite(ledger[k]) for k in keys) or not math.isfinite(thermal_increment_j):
        raise ValueError('Finite joule budget required')
    drag=ledger['physical_drag_j'];damage=-ledger['damage_energy_change_j']
    numerical=math.fsum(ledger[k] for k in ('numerical_kinetic_j','numerical_memory_j'))
    if min(drag,damage,ledger['numerical_kinetic_j'],ledger['numerical_memory_j'])<0:
        raise ValueError('Unfunded healing or negative dissipation; constitutive source required')
    if any(ledger.get(k,0)!=0 for k in ('birth_kinetic_inflow_j','remelt_kinetic_outflow_j','remelt_elastic_outflow_j')):
        raise ValueError('Material energy port not declared')
    ports=[dict(name='prescribed_mechanical_driver',energy_into_domain_j=ledger['external_work_j'],kind='external_work')]
    if not interface_in_control_volume:
        ports.append(dict(name='unresolved_basal_interface',energy_into_domain_j=-drag,kind='mechanical_energy_export'))
    stores=dict(damage_unresolved_j=damage,interface_unresolved_j=drag if interface_in_control_volume else 0.)
    delta_terms=[thermal_increment_j,ledger['kinetic_after_j'],-ledger['kinetic_before_j'],ledger['elastic_after_j'],-ledger['elastic_before_j'],*stores.values()]
    residual=math.fsum([*delta_terms,numerical,*[-p['energy_into_domain_j'] for p in ports]])
    return dict(unit='J',ports=ports,storage_increments=stores,numerical_sink_j=numerical,
                control_volume_energy_change_j=math.fsum(delta_terms),net_boundary_input_j=math.fsum(p['energy_into_domain_j'] for p in ports),
                balance_residual_j=residual,thermal_feedback_from_unresolved_j=0.,
                hypothesis='unresolved channels return no heat during this bounded interval; not a universal constitutive law')

"""Opt-in synthetic geological continuation; no claim of SI physical equivalence.

Uses native mature dynamics and material transport. Thermal, topography,
continental evolution, plumes and topology changes are deliberately inactive.
The complete source is frozen; archival copies are never active inventories.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import save_checkpoint, load_checkpoint
from .dynamics import DynamicsParameters, update_plate_dynamics, angular_velocity_vectors
from .genesis import SECONDS_PER_MYR
from .genesis_mature import build_experimental_genesis_mature_import
from .genesis_starter_ledger import continuation_material_ledger
from .genesis_transfer_bridge import encoded
from .lithosphere import advance_lithosphere
from .mesh import build_icosphere
from .plates import Plate, PlateSystem
from .thermal import ThermalState
from .tides import EccentricityHistory
from .topology import PlateTopologyManager, PlateTopologyParameters

VERSION = 'synthetic-geological-handoff-1'
DENSITY = 3000.0


def sha(data):
    return hashlib.sha256(data).hexdigest()


def kinematics(cp, mesh, radius):
    """Reference basalt shell proxy, NOT mature model's conserved SI momentum."""
    mass = cp.state.oceanic_volume_km3 * DENSITY * 1e9
    r = mesh.centroids * radius * 1000
    omega = angular_velocity_vectors(cp.system)[cp.state.cell_plate] / SECONDS_PER_MYR
    velocity = np.cross(omega, r)
    return dict(angular_momentum_kg_m2_s=np.sum(np.cross(r, mass[:, None]*velocity), axis=0).tolist(),
                linear_momentum_kg_m_s=np.sum(mass[:, None]*velocity, axis=0).tolist(),
                kinetic_energy_j=float(.5*np.sum(mass*np.sum(velocity**2, axis=1))))


class GeologicalContinuation:
    FORMAT = VERSION

    def __init__(self, checkpoint, record, source):
        self.checkpoint = checkpoint
        self.record = deepcopy(record)
        self.source = deepcopy(source)
        self.mesh = build_icosphere(record['subdivisions'])

    @classmethod
    def from_synthetic(cls, model, *, composition, mantle_omega, water_volume_km3):
        """Explicit manufactured basalt scenario only; real-state readiness is absent."""
        c, s = model.config, model.state
        if c['source_mode'] != 'synthetic_prescribed_power':
            raise ValueError('REAL_HANDOFF_NOT_READY: chemistry, remapping and thermal closure required')
        if composition != 'all_solid_nodes_basalt_reference_density_3000':
            raise ValueError('Explicit synthetic basalt composition required')
        if len(np.unique(s['owner'])) < 2:
            raise ValueError('Selector has not produced multiple domains')
        if s.get('born_material') or c.get('edge_strips') or c.get('phase_cells'):
            raise ValueError('Additional material/edge mapping not implemented')
        if not np.allclose(s['rotation'], np.eye(3)[None], rtol=0, atol=1e-14):
            raise ValueError('Rotated geometry requires conservative remapping')
        owner = np.array(s['owner'], dtype=np.int32)
        plates = []
        for p, w in enumerate(np.array(s['omega']) * SECONDS_PER_MYR):
            speed = float(np.linalg.norm(w))
            plates.append(Plate(p, int(np.flatnonzero(owner == p)[0]), w/speed if speed else np.array([0., 0., 1.]), speed))
        nodes = model.solid_nodes
        if len(np.unique(nodes)) != nodes.size or set(nodes.ravel()) != set(range(len(c['mass'])-2)):
            raise ValueError('Only solid nodes plus explicit deep/surface reservoirs supported')
        mass = np.array(c['mass']); energy = np.array(s['energy'])
        face_mass = mass[nodes].sum(axis=1)
        if not np.allclose(face_mass, c['face_mass_kg'], rtol=1e-14, atol=0):
            raise ValueError('Solid mass mapping mismatch')
        T = model.temperature(energy); radius = c['radius_km']; n = model.mesh.cell_count
        thermal = ThermalState(s['time_myr'], s['time_myr'], float(T[-2]), .12, 1., 35.)
        bundle = build_experimental_genesis_mature_import(model.mesh, radius_km=radius,
            system=PlateSystem(owner, tuple(plates)), crust_age_myr=np.full(n, s['time_myr']),
            crust_thickness_km=face_mass/(DENSITY*1e9*model.mesh.physical_cell_areas_km2(radius)),
            tidal_damage=np.array(s['damage']), mantle_lithosphere_thickness_km=np.zeros(n),
            mantle_lithosphere_density_anomaly_kg_m3=np.zeros(n), thermal=thermal,
            elevation_m=np.zeros(n), water_volume_km3=water_volume_km3,
            mantle_cell_omega_rad_per_myr=mantle_omega, source_mass_kg=face_mass,
            source_enthalpy_j=energy[nodes].sum(axis=1), next_plume_birth_time_myr=s['time_myr']+1e6,
            source_metadata={'scenario': VERSION, 'composition': composition})
        source = dict(config=deepcopy(c), state=deepcopy(s))
        elastic = float(np.sum(model.constitutive(T, s['damage'], np.array(s['memory']), T, 0)[4]))
        w = np.array(s['omega'])[owner]; I = model.material_inertia()
        source_L = np.einsum('fij,fj->i', I, w)
        source_K = float(.5*np.einsum('fi,fij,fj->', w, I, w))
        record = dict(format=VERSION, subdivisions=bundle.report['mesh_subdivisions'], radius_km=radius,
            dynamics_parameters=asdict(DynamicsParameters()), gravity_m_s2=9., period_hours=24., eccentricity=.00047,
            interior_reference_mass_kg=float(mass[-2]), inactive_surface_mass_kg=float(mass[-1]),
            initial_total_reference_mass_kg=float(mass.sum()), source_sha256=sha(encoded(source)), steps=[],
            mapping=dict(mass={'active_basalt_kg':float(face_mass.sum()), 'active_interior_kg':float(mass[-2]),
                              'inactive_surface_kg':float(mass[-1]), 'density_kg_m3':DENSITY,
                              'archive_is_duplicate_not_inventory':True},
                composition={'active':'explicit manufactured basalt, reference density; no elemental chemistry',
                             'real_composition_inferred':False,
                             'damage_mapping':'source scalar D reused as mature tidal weakening; constitutive meaning changes'},
                H={'archived_total_j':float(energy.sum()), 'active':'scalar deep temperature, frozen in this kernel subset',
                   'material_exchange_enthalpy_closed':False, 'heat_from_retired_memory_j':0.},
                momentum={'source_angular_kg_m2_s':source_L.tolist(), 'omega_mapping':'rad/s to rad/Myr exactly',
                          'mature_shell_proxy':kinematics(bundle.checkpoint,model.mesh,radius),
                          'SI_momentum_conserved':False},
                energy={'source_kinetic_j':source_K,'retired_elastic_j':elastic,
                        'archived_passive_reserve_j':s['passive_reserve_j'], 'coupled_energy_closed':False},
                Maxwellmemory={'status':'complete frozen source archive; no active constitutive consumer',
                               'released_as_heat_j':0.}),
            inactive_modules=['thermal evolution','topography','continental cycle','plumes','topology changes'],
            import_report=bundle.report)
        return cls(bundle.checkpoint, record, source)

    def step(self, dt_myr):
        if not np.isfinite(dt_myr) or dt_myr <= 0:
            raise ValueError('Positive finite dt required')
        # Transactional trial: failed material accounting cannot mutate the accepted state.
        old = self.checkpoint; new = deepcopy(old); r = self.record['radius_km']
        before = kinematics(old, self.mesh, r)
        new.system, dynamics, _, _ = update_plate_dynamics(self.mesh, new.state, new.system,
            new.baseline, r, dt_myr, 5., 1., DynamicsParameters(**self.record['dynamics_parameters']),
            mantle_flow=new.mantle_flow,**self._dynamics_options(new))
        history = EccentricityHistory(np.array([0., 1e9]), np.full(2, self.record['eccentricity']))
        new.state, _, _, diag = self._advance_lithosphere(new,dt_myr,r,history)
        new.system.cell_plate = new.state.cell_plate.copy()
        row = asdict(diag); source_index = row.pop('material_source_index')
        new.lithosphere_rows.append(row)
        self._after_lithosphere(old,new,source_index,dt_myr)
        budget = continuation_material_ledger(old,new,DENSITY)
        interior = self.record['interior_reference_mass_kg'] + budget['mantle_reference_mass_change_kg']
        if not budget['balanced'] or not np.isfinite(interior) or interior < 0:
            raise ValueError('Material budget failed or interior exhausted')
        omega = angular_velocity_vectors(new.system)
        relative = float(np.max(np.linalg.norm(omega[:,None]-omega[None,:], axis=2)))
        moved = (source_index >= 0) & (source_index != np.arange(self.mesh.cell_count))
        after = kinematics(new, self.mesh, r)
        audit = dict(time_myr=new.state.time_myr, dt_myr=dt_myr, relative_omega_rad_per_myr=relative,
            transported_winning_cells=int(moved.sum()), material_source_sha256=sha(source_index.tobytes()),
            transported_oceanic_volume_km3=float(new.state.oceanic_volume_km3[moved].sum()),
            transport_commits=diag.conservative_transport_commits, material=budget,
            interior_reference_mass_kg=interior, kinematics_before=before, kinematics_after=after,
            unclosed_kinetic_change_j=after['kinetic_energy_j']-before['kinetic_energy_j'],
            unclosed_angular_momentum_change_kg_m2_s=(np.array(after['angular_momentum_kg_m2_s'])-before['angular_momentum_kg_m2_s']).tolist(),
            unclosed_linear_momentum_change_kg_m_s=(np.array(after['linear_momentum_kg_m_s'])-before['linear_momentum_kg_m_s']).tolist(),
            SI_force_work_available=False, coupled_enthalpy_closed=False, heat_from_retired_memory_j=0.,
            dynamics=asdict(dynamics))
        self._after_material_step(old, new, source_index, audit)
        # Inactive native substates retain their own timestamps, visibly frozen.
        self.checkpoint = new
        self.record['interior_reference_mass_kg'] = interior
        self.record['steps'].append(audit)
        return deepcopy(audit)

    def _advance_lithosphere(self,new,dt_myr,r,history):
        return advance_lithosphere(self.mesh,new.system,new.state,dt_myr,r,
            self.record['gravity_m_s2'],self.record['period_hours'],history,transport_state=new.transport_state,
            primary_mass_jupiter=self.record.get('primary_mass_jupiter', 1.),
            love_h2=self.record.get('love_h2', .6),**self._lithosphere_options())

    def _after_material_step(self, old, new, source_index, audit):
        """Optional transactional extension; the legacy experiment keeps this empty."""

    def _lithosphere_options(self):
        return {}

    def _dynamics_options(self,checkpoint):
        return {}

    def _after_lithosphere(self,old,new,source_index,dt_myr):
        pass

    def save(self, path):
        root = Path(path); root.mkdir(parents=True, exist_ok=False)
        save_checkpoint(root/'mature', self.checkpoint)
        (root/'source.json').write_bytes(encoded(self.source))
        (root/'record.json').write_bytes(encoded(self.record))
        names = ['mature/meta.json','mature/state.npz','source.json','record.json']
        (root/'checksums.json').write_bytes(encoded({n:sha((root/n).read_bytes()) for n in names}))
        return root

    @classmethod
    def load(cls, path):
        root = Path(path); sums = json.loads((root/'checksums.json').read_bytes())
        names = {'mature/meta.json','mature/state.npz','source.json','record.json'}
        if set(sums) != names or any(sha((root/n).read_bytes()) != sums[n] for n in names):
            raise ValueError('Geological restart checksum mismatch')
        record = json.loads((root/'record.json').read_bytes()); source = json.loads((root/'source.json').read_bytes())
        if record['format'] != cls.FORMAT or record['source_sha256'] != sha(encoded(source)):
            raise ValueError('Geological source/version mismatch')
        cp = load_checkpoint(root/'mature', PlateTopologyManager(PlateTopologyParameters()))
        return cls(cp,record,source)

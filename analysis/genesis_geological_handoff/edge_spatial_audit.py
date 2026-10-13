"""Bounded attribution audit; no planetary advancement or law changes."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .edge_weakening_controls import GridControl
from .driven_shell import CONFIG
from tectonics.experimental_edge_weakening import face_damage
from .nodal_retirement_check import SOURCE

ROOT = Path(__file__).parent
SNAPSHOTS = (.5, 1., 2., 5., 10., 20.)
FIELDS = Path('/workspace/genesis-edge-spatial-audit-01/fields.npz')


def design(x):
    return np.column_stack((np.ones(len(x)), x, x[:, 0]*x[:, 1],
        x[:, 0]*x[:, 2], x[:, 1]*x[:, 2], x[:, 0]**2-x[:, 1]**2,
        3*x[:, 2]**2-1))


def world_stress(g, stress):
    sigma = stress.reshape(-1, 2, 6).mean(axis=1)
    tensor = np.zeros((g.a.nf, 3, 3))
    tensor[:, range(3), range(3)] = sigma[:, :3]
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
        tensor[:, i, j] = tensor[:, j, i] = sigma[:, k]
    return np.einsum('fai,fij,fbj->fab', g.frames, tensor, g.frames)


class AuditGrid(GridControl):
    def __init__(self, level, feedback, coeff):
        super().__init__(level)
        self.feedback = feedback
        old = np.clip(design(self.mesh.centroids)@coeff, 0, 1)
        newscale = .02+.98*(1-old)**2
        self.a.C *= np.repeat(newscale/self.base_scale, 2)[:, None, None]
        self.old, self.base_scale = old, newscale

    def ratio(self, edge):
        return super().ratio(edge) if self.feedback else np.ones(self.a.nf)

    def shear(self, stress):
        self.last_stress = world_stress(self, stress)
        self.last_shear = super().shear(stress)
        return self.last_shear


def restrict(values, fine_area):
    shape = (len(fine_area)//4, 4)+values.shape[1:]
    wshape = (len(fine_area)//4, 4)+(1,)*(values.ndim-1)
    return (values.reshape(shape)*fine_area.reshape(wshape)).sum(axis=1)/fine_area.reshape(wshape).sum(axis=1)


def metric(coarse, fine, fine_area):
    projected = restrict(fine, fine_area)
    weights = fine_area.reshape(-1, 4).sum(axis=1)
    weights /= weights.sum()
    axes = tuple(range(1, projected.ndim))
    sq = lambda x: np.sum(x*x, axis=axes) if axes else x*x
    numerator = float(np.sqrt(weights@sq(projected-coarse)))
    denominator = float(np.sqrt(weights@sq(projected)))
    return dict(absolute_rms=numerator, reference_rms=denominator,
                relative_l2=numerator/denominator if denominator else None)


def geometry(g):
    mesh = g.mesh
    edges = np.array(mesh.shared_edges)
    a, b = mesh.vertices[edges[:, 2]], mesh.vertices[edges[:, 3]]
    midpoint = (a+b)/np.linalg.norm(a+b, axis=1)[:, None]
    length = np.arccos(np.clip(np.sum(a*b, axis=1), -1, 1))*CONFIG['radius_km']
    # Area of the two weakened faces / edge arc length: equivalent band width.
    area = mesh.physical_cell_areas_km2(CONFIG['radius_km'])
    width = (area[edges[:, 0]]+area[edges[:, 1]])/length
    fixed_damage = lambda x: .2+.1*x[:, 0]*x[:, 1]
    projected = face_damage(mesh, fixed_damage(midpoint))
    direct = fixed_damage(mesh.centroids)
    return dict(edge_length_km=[float(length.min()), float(np.median(length)), float(length.max())],
                two_face_band_width_km=[float(width.min()), float(np.median(width)), float(width.max())],
                fixed_smooth_projection_bias=float(g.a.area@(projected-direct)),
                fixed_smooth_projection_rms=float(np.sqrt(g.a.area@(projected-direct)**2)),
                fixed_smooth_projected=projected.tolist())


def save_result(result):
    """Keep large fields local, with a checksum and reproducible compact report."""
    FIELDS.parent.mkdir(exist_ok=False)
    arrays = {}
    for key, run in result['runs'].items():
        for t, snapshot in run['snapshots'].items():
            for field in ('damage', 'stress', 'shear'):
                arrays[f'{key}_{t}_{field}'] = np.array(snapshot.pop(field))
    for level, geometry_result in result['geometry'].items():
        arrays[f'geometry_{level}'] = np.array(geometry_result.pop('fixed_smooth_projected'))
    np.savez_compressed(FIELDS, **arrays)
    result['fields'] = dict(path=str(FIELDS), sha256=hashlib.sha256(FIELDS.read_bytes()).hexdigest(),
                           bytes=FIELDS.stat().st_size)
    result['source_sha256'] = hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()
    result['code_base'] = 'b87aef4b87071e6524a3e1e5eeab309374953817'
    (ROOT/'edge_spatial_audit.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    if FIELDS.parent.exists():
        raise FileExistsError(f'Preserve existing audit: {FIELDS.parent}')
    start = time.monotonic()
    prior = json.loads((ROOT/'edge_spatial_control.json').read_text())
    assert hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest() == '94fbbb3cb88a15147d33ab04c8f791470dd92e3d73cf8cba4afc0f361cd583fc'
    coeff = np.array(prior['coefficients'])
    grids, runs, geometries = {}, {}, {}
    for feedback in (True, False):
        for level in (2, 3):
            key = f'{level}_{int(feedback)}'
            g = AuditGrid(level, feedback, coeff)
            grids[key] = g
            if feedback:
                geometries[str(level)] = geometry(g)
            state = g.initial()
            snapshots, errors = {}, []
            for step in range(40):
                if time.monotonic()-start > 120:
                    raise RuntimeError('Predeclared 120 second audit cap')
                state, row = g.step(state, .5)
                errors.append(row['energy_relative'])
                if state['time'] in SNAPSHOTS:
                    snapshots[str(state['time'])] = dict(
                        damage=face_damage(g.mesh, state['d']).tolist(),
                        stress=g.last_stress.tolist(), shear=g.last_shear.tolist(),
                        max_edge_damage=max(state['d']), reserve_j=state['reserve'])
            runs[key] = dict(snapshots=snapshots, energy_relative=max(errors))
            if feedback:
                expected = prior['runs'][level-2]['spatial_profile']
                np.testing.assert_allclose(snapshots['20.0']['damage'], expected, rtol=1e-12, atol=1e-13)
            print(json.dumps(dict(run=key, seconds=time.monotonic()-start)), flush=True)
    comparisons = {}
    for feedback in (True, False):
        coarse, fine = runs[f'2_{int(feedback)}'], runs[f'3_{int(feedback)}']
        area = grids[f'3_{int(feedback)}'].mesh.areas_unit_sphere
        comparisons[str(feedback)] = {
            t: {field: metric(np.array(coarse['snapshots'][t][field]),
                             np.array(fine['snapshots'][t][field]), area)
                for field in ('damage', 'stress')}
            for t in coarse['snapshots']}
    g2, g3 = grids['2_1'], grids['3_1']
    # Each coarse edge consists of two fine edges with the SAME great-circle conormal.
    fine_edges = {tuple(sorted((i, j))): k for k, (_, _, i, j) in enumerate(g3.mesh.shared_edges)}
    paired, angles = [], []
    for _, _, i, j in g2.mesh.shared_edges:
        mid = g2.mesh.vertices[i]+g2.mesh.vertices[j]; mid /= np.linalg.norm(mid)
        m = int(np.argmin(np.linalg.norm(g3.mesh.vertices-mid, axis=1)))
        assert np.linalg.norm(g3.mesh.vertices[m]-mid) < 1e-12
        paired.append([fine_edges[tuple(sorted((i, m)))], fine_edges[tuple(sorted((m, j)))]])
        n = np.cross(g2.mesh.vertices[i], g2.mesh.vertices[j]); n /= np.linalg.norm(n)
        nn = np.cross(g3.mesh.vertices[i], g3.mesh.vertices[m]); nn /= np.linalg.norm(nn)
        angles.append(float(np.linalg.norm(n-nn)))
    paired = np.array(paired)
    edge_comparisons = {}
    for feedback in (True, False):
        edge_comparisons[str(feedback)] = {}
        for t in runs['2_1']['snapshots']:
            c = np.array(runs[f'2_{int(feedback)}']['snapshots'][t]['shear'])
            f = np.array(runs[f'3_{int(feedback)}']['snapshots'][t]['shear'])[paired].mean(axis=1)
            edge_comparisons[str(feedback)][t] = float(np.linalg.norm(c-f)/np.linalg.norm(f))
    result = dict(scope='Four 20year manufactured runs, dt=.5; passive damage retains identical law but no stiffness feedback',
                  comparisons=comparisons, collinear_edge_shear_relative_l2=edge_comparisons,
                  conormal_max_vector_difference=max(angles), geometry=geometries,
                  fixed_projection_grid_metric=metric(np.array(geometries['2']['fixed_smooth_projected']),
                    np.array(geometries['3']['fixed_smooth_projected']), g3.mesh.areas_unit_sphere),
                  runs=runs, seconds=time.monotonic()-start)
    save_result(result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs', 'geometry')}, indent=2))


if __name__ == '__main__':
    main()

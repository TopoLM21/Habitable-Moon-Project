"""Static geometry checks; no advance and no changes to physical forcing."""
import json
import numpy as np
from .edge_spatial_audit import AuditGrid, ROOT, metric
from .driven_shell import CONFIG
from tectonics.experimental_edge_weakening import face_damage


def local_stress(g, tensor):
    world = np.broadcast_to(tensor, (g.a.nf, 3, 3))
    local = np.einsum('fai,fab,fbj->fij', g.frames, world, g.frames)
    voigt = np.column_stack([local[:, i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
    return np.repeat(voigt, 2, axis=0)


def main():
    prior = json.loads((ROOT/'edge_spatial_control.json').read_text())
    tensor = 1e6*np.array([[2., 1., .5], [1., -1., .3], [.5, .3, -1.]])
    rows, profiles, grids = [], [], []
    for level in (2, 3):
        g = AuditGrid(level, False, np.array(prior['coefficients'])); grids.append(g)
        shear = g.shear(local_stress(g, tensor))
        profiles.append(face_damage(g.mesh, .5/100*np.maximum(shear/1e6-1, 0)**2))
        hydro = g.shear(local_stress(g, np.eye(3)*1e6))
        r = CONFIG['radius_km']*1000; inner = 1-CONFIG['thickness_km']/CONFIG['radius_km']
        exact_volume = 4*np.pi*r**3*(1-inner**3)/3
        rows.append(dict(level=level, spherical_area_relative_error=abs(g.mesh.areas_unit_sphere.sum()-4*np.pi)/(4*np.pi),
            faceted_volume_to_spherical_volume=float(g.a.vol.sum()/exact_volume),
            radial_weight_pair_max_difference=float(abs(g.a.w[::2]-g.a.w[1::2]).max()),
            hydrostatic_shear_max_pa=float(hydro.max()),
            constant_tensor_max_damage=float(profiles[-1].max())))
    np.testing.assert_allclose(grids[1].mesh.areas_unit_sphere.reshape(-1,4).sum(axis=1), grids[0].mesh.areas_unit_sphere, rtol=1e-13)
    result = dict(scope='Static normalization and orientation probes, prescribed tensor is diagnostic only',
        constant_world_stress_pa=tensor.tolist(), checks=rows,
        constant_world_tensor_damage_grid_metric=metric(profiles[0], profiles[1], grids[1].mesh.areas_unit_sphere),
        child_areas_sum_to_parent=True)
    (ROOT/'edge_spatial_geometry_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

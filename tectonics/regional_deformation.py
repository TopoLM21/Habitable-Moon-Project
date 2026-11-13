"""Coarse, welded spherical deformation with a physical dilation resistance.

This is a dissipative surface-layer approximation, not elastic plates or
sliding contact. It reuses the exact area/potential and step controller of the
conforming control. The physical length is independent of the material grid.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .conforming_surface import ConformingSurface


@dataclass(frozen=True)
class DilationParameters:
    length_m: float = 1e6

    def __post_init__(self):
        if not math.isfinite(self.length_m) or self.length_m < 0:
            raise ValueError('Invalid physical dilatational length')


def tangent_basis(vertices):
    """Two orthonormal tangent directions, without a preferred physical axis."""
    axis = np.eye(3)[np.argmin(abs(vertices), axis=1)]
    first = np.cross(vertices, axis)
    first /= np.linalg.norm(first, axis=1)[:, None]
    return np.stack([first, np.cross(vertices, first)], axis=2)


class DeformingSurface(ConformingSurface):
    VERSION = 'regional_dilation_1'

    def __init__(self, *args, dilation=DilationParameters(), **kwargs):
        self.dilation = dilation
        super().__init__(*args, **kwargs)

    @classmethod
    def from_surface(cls, surface, dilation=DilationParameters()):
        """Explicit new constitutive choice, with unchanged geometry and energy."""
        cp = surface.checkpoint()
        if cp['data']['version'] != ConformingSurface.VERSION:
            raise ValueError('Expected the conforming gravity/drag reference')
        cp['data']['version'] = cls.VERSION
        cp['data']['material_created_m3'] = [[0., 0.], [0., 0.]]
        cp['data']['material_returned_m3'] = [[0., 0.], [0., 0.]]
        cp['dilation'] = asdict(dilation)
        cp['data']['provenance']['dilation_initialization'] = {
            'time_myr': cp['data']['time_myr'],
            'length_m': dilation.length_m,
            'scope': 'new viscous resistance; no added stored energy or restored stress',
        }
        return cls.restore(cp)

    def rate_operator(self, vertices):
        area, gradient = self.geometry(vertices)
        basis = tangent_basis(vertices)
        # d(A/R²)/d(two angular tangent coordinates) for each material face.
        values = np.einsum('fkj,fkjd->fkd', gradient, basis[self.faces]) / self.params.radius_m**2
        columns = (2*self.faces[:, :, None] + np.arange(2)).reshape(-1)
        rows = np.repeat(np.arange(len(self.faces)), 6)
        J = coo_matrix((values.ravel(), (rows, columns)),
                       shape=(len(self.faces), 2*len(vertices))).tocsr()
        return area, basis, J

    def evaluate(self, vertices):
        result = super().evaluate(vertices)
        area, basis, J = self.rate_operator(vertices)
        p = self.params
        s = area/p.radius_m**2
        weights = result['weights_m2']/p.radius_m**2
        length2 = (self.dilation.length_m/p.radius_m)**2
        K = diags(np.repeat(weights, 2)) + length2*(J.T @ diags(1/s) @ J)
        scale = p.basal_drag_pa_s_m*p.radius_m**4  # J s
        force = np.einsum('vjd,vj->vd', basis, result['gradient_j']).ravel()
        if self.dilation.length_m == 0:
            velocity = -force/(scale*np.repeat(weights, 2))
        else:
            velocity = spsolve(K.tocsc(), -force/scale)
        residual = K @ velocity + force/scale
        relative_residual = np.linalg.norm(residual)/max(np.linalg.norm(force/scale), 1e-300)
        if not np.all(np.isfinite(velocity)) or relative_residual > 2e-11:
            raise ValueError('Unresolved regional dilation force balance')
        area_rate = J @ velocity  # unit-sphere area per second
        basal = scale*math.fsum(np.repeat(weights, 2)*velocity**2)
        dilational = scale*length2*math.fsum(area_rate**2/s)
        power = -math.fsum(force*velocity)
        total = math.fsum([basal, dilational])
        result.update(
            velocity_per_s=np.einsum('vjd,vd->vj', basis, velocity.reshape(-1, 2)),
            dissipation_w=total, basal_dissipation_w=basal,
            dilation_dissipation_w=dilational,
            area_rate_m2_s=area_rate*p.radius_m**2,
            force_power_w=power, power_residual_w=power-total,
            linear_relative_residual=relative_residual,
        )
        return result

    def checkpoint(self):
        cp = super().checkpoint()
        cp['dilation'] = asdict(self.dilation)
        return cp

    @classmethod
    def restore(cls, checkpoint):
        cp = deepcopy(checkpoint)
        if cp['data']['version'] != cls.VERSION or 'dilation' not in cp:
            raise ValueError('Incomplete regional dilation checkpoint')
        dilation = DilationParameters(**cp.pop('dilation'))
        original = cp['data']['initial_volume_m3']
        current = [math.fsum(cp['arrays'][k+'_volume_m3']) for k in ('felsic', 'basalt')]
        for i in range(2):
            born = cp['data'].get('material_created_m3', [[0., 0.], [0., 0.]])[i]
            returned = cp['data'].get('material_returned_m3', [[0., 0.], [0., 0.]])[i]
            expected = math.fsum([original[i], *born, *(-x for x in returned)])
            if abs(expected-current[i]) > 64*math.ulp(max(abs(expected), abs(current[i]), 1.)):
                raise ValueError('Changed material volume without transfer history')
        # The reference loader checks fixed mass. Validate the explicitly
        # transferred total above, then retain the original initial inventory.
        cp['data']['initial_volume_m3'] = current
        cp['data']['version'] = ConformingSurface.VERSION
        base = ConformingSurface.restore(cp)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.dilation = dilation
        obj.data['version'] = cls.VERSION
        obj.data['initial_volume_m3'] = original
        obj.evaluate(obj.vertices)
        return obj

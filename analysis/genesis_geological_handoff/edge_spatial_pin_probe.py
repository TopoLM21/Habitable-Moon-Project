"""Four first-step solves to check the diagnostic's six rigid-mode pins."""
import json
import numpy as np
from .edge_spatial_audit import AuditGrid, ROOT


def main():
    prior = json.loads((ROOT/'edge_spatial_control.json').read_text())
    out = []
    for level in (2, 3):
        g = AuditGrid(level, False, np.array(prior['coefficients']))
        state = g.initial()
        g.step(state, .5)
        stress, shear = g.last_stress.copy(), g.last_shear.copy()
        fixed = g.a.fixed.tolist()
        g.a.free = np.arange(g.a.nd)
        try:
            _, row = g.step(state, .5)
            out.append(dict(level=level, fixed=fixed,
                stress_relative=float(np.linalg.norm(g.last_stress-stress)/np.linalg.norm(g.last_stress)),
                shear_relative=float(np.linalg.norm(g.last_shear-shear)/np.linalg.norm(g.last_shear)), checks=row))
        except (AssertionError, ValueError) as exc:
            out.append(dict(level=level, error=str(exc)))
    (ROOT/'edge_spatial_pin_probe.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

"""Plot an actual moved source contact and its original integration-cell edge."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.geometric_contacts import extract_contacts
from tectonics.fractional_surface import SurfaceParcel
from tectonics.geometric_surface import GeometricFragment, GeometricSurfaceState, load_geometric_checkpoint
from tectonics.mesh import build_icosphere


def plot(checkpoint, output):
    report_path = checkpoint.with_name("report.json")
    if report_path.is_file():
        # The validation run already audited this exact file on restart. Check
        # its recorded file hash before avoiding a duplicate full-sphere audit
        # just to draw two triangles. This path never changes simulation state.
        report = json.loads(report_path.read_text(encoding="utf-8"))
        encoded = checkpoint.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != report["checkpoint_sha256"]:
            raise ValueError("Plot checkpoint differs from the audited validation file")
        payload = json.loads(encoded)
        data = payload["state"]
        fragments = tuple(GeometricFragment(f["fragment_id"], tuple(map(tuple, f["polygon"])),
            SurfaceParcel(**f["parcel"]), f.get("parent_fragment_id")) for f in data.pop("fragments"))
        state = GeometricSurfaceState(fragments=fragments, **data)
        provenance = payload["provenance"]
    else:
        state, provenance = load_geometric_checkpoint(checkpoint)
    mesh = build_icosphere(provenance["subdivisions"])
    fragments = {f.fragment_id: f for f in state.fragments}
    edge_by_cells = {frozenset((a, b)): (u, v) for a, b, u, v in mesh.shared_edges}
    contacts = extract_contacts(state.fragments, state.radius_km)
    candidates = []
    for contact in contacts:
        a, b = fragments[contact.fragment_a], fragments[contact.fragment_b]
        edge = edge_by_cells.get(frozenset((a.parcel.cell, b.parcel.cell)))
        if edge is None:
            continue
        p, q = mesh.vertices[list(edge)]
        normal = np.cross(p, q)
        normal /= np.linalg.norm(normal)
        separation = state.radius_km*abs(np.arcsin(np.clip(np.dot(contact.midpoint, normal), -1., 1.)))
        candidates.append((separation, contact))
    separation, contact = max(candidates, key=lambda item: item[0])
    a, b = fragments[contact.fragment_a], fragments[contact.fragment_b]
    center = np.asarray(contact.midpoint)
    basis = np.stack((contact.normal_a_to_b, contact.tangent), axis=1)
    def project(points):
        points = np.asarray(points)
        return state.radius_km*(points@basis)/(points@center)[..., None]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax in axes:
        for fragment, color in ((a, "#78ACD5"), (b, "#F1B06C")):
            xy = project(fragment.polygon)
            ax.fill(*xy.T, color=color, alpha=.86)
            old = project(mesh.vertices[mesh.faces[fragment.parcel.cell]])
            closed = np.vstack((old, old[0]))
            ax.plot(*closed.T, color="#777777", ls="--", lw=1.4)
        arc = project((contact.start, contact.end))
        ax.plot(*arc.T, color="#191919", lw=2.1)
        ax.set_aspect("equal")
        ax.set_xlabel("Поперёк контакта, км")
        ax.set_ylabel("Вдоль контакта, км")
        ax.grid(alpha=.13)
    whole = np.vstack((project(a.polygon), project(b.polygon)))
    axes[0].set_xlim(whole[:, 0].min()*1.1, whole[:, 0].max()*1.1)
    axes[0].set_ylim(whole[:, 1].min()*1.1, whole[:, 1].max()*1.1)
    axes[0].set_title("Сохраняемые фрагменты материала")
    axes[1].set_xlim(-max(2., separation*2.7), max(2., separation*2.7))
    axes[1].set_ylim(-max(2., separation*2.7), max(2., separation*2.7))
    axes[1].set_title(f"Увеличение: смещение границы {separation:.3f} км")
    axes[1].text(.03, .04, "В ячейке появилась полоса\nматериала соседней плиты", transform=axes[1].transAxes,
                 fontsize=9, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    figure.suptitle("Настоящий контакт пересекает расчётную ячейку", fontsize=15)
    figure.legend(handles=[Line2D([0], [0], color="#191919", lw=2, label="Граница плит"),
                           Line2D([0], [0], color="#777777", ls="--", label="Граница ячеек сетки")],
                  loc="outside lower center", ncol=2, frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}; actual contact {contact.contact_id}; displacement {separation:.12g} km")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.checkpoint, args.output)

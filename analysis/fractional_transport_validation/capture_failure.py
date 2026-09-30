"""Save the exact unmodified refined input and offending resolver cell."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_fractional_transport_probe import load_probe_source,make_birth_factory,digest
from tectonics.fractional_surface_io import surface_from_lithosphere,save_fractional_checkpoint,refine_fractional_surface
from tectonics.fractional_transport import advance_fractional_transport
from tectonics.mesh import build_icosphere
import numpy as np


def main():
    output=Path(__file__).resolve().parent/"failure_refined_dt1"
    output.mkdir(exist_ok=False)
    mesh,cp,fracture,model,provenance=load_probe_source(ROOT/"analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050")
    surface=surface_from_lithosphere(mesh,cp.state,provenance["radius_km"],fracture_memory=fracture.memory)
    fine=build_icosphere(model.shell.subdivisions+1)
    surface=refine_fractional_surface(surface,mesh,fine,provenance["radius_km"])
    mesh=fine
    model=SimpleNamespace(shell=model.shell,strength_factor=np.repeat(model.strength_factor,4))
    omega=np.array([p.euler_axis*p.angular_speed_rad_per_myr for p in cp.system.plates])
    factory=make_birth_factory(surface,model,provenance)
    metadata=dict(source=provenance,omega_rad_per_myr=omega.tolist(),step_myr=1.,subdivisions=5,
        production_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/"tectonics").glob("fractional_*.py")})
    for index in range(1):
        try:
            result=advance_fractional_transport(mesh,surface,omega,provenance["radius_km"],1.,birth_factory=factory)
        except Exception as error:
            save_fractional_checkpoint(output/"fractional_checkpoint.json",mesh,surface,provenance["radius_km"],provenance=provenance)
            metadata.update(failed_step_index=index,input_time_myr=surface.time_myr,traceback=traceback.format_exc())
            frame=error.__traceback__
            while frame is not None:
                if frame.tb_frame.f_code.co_name=="_resolve":
                    local=frame.tb_frame.f_locals
                    cell_data={name:local[name] for name in ("cell","capacity","arrived","excess","deficit","tolerance","end_time") if name in local}
                    for name in ("pieces","local_lost","local_retained"):
                        cell_data[name]=[asdict(piece) for piece in local[name]]
                    cell_data["failing_piece"]=asdict(local["piece"])
                    cell_data["source_parcels"]=[dict(source_index=i,parcel=asdict(local["before"].parcels[i]))
                        for i in sorted({piece.source_index for piece in local["pieces"]})]
                    cell_data["surviving"]=[dict(cell=k[0],plate=k[1],area=v) for k,v in local["surviving"].items()]
                    (output/"resolver_cell.json").write_text(json.dumps(cell_data,indent=2)+"\n",encoding="utf-8")
                frame=frame.tb_next
            (output/"reproduction.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
            print(json.dumps(dict(output=str(output),failed_step_index=index,input_time_myr=surface.time_myr)),flush=True)
            return
        surface=result.state
    raise RuntimeError("Failure did not reproduce with current loaded kernel")


if __name__=="__main__":
    main()

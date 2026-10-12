"""Independent stage equation and RK work identity, no producer RHS calls."""
import numpy as np
from tectonics.genesis_implicit_enthalpy import invert
from .core import A,B,C,ENERGY_MATRIX


def audit(model,before,result):
    h=result['dt'];Y=result['stages'];K=result['rates'];t=result['time']
    n=model.nt;nv=model.nv
    external=heat=drag=maxwell=thermal=0.;impulse=np.zeros(nv);node_flux=np.zeros(n)
    rate_residuals=[];stage_defects=[];local_heat=np.zeros(n)
    for j in range(2):
        E=Y[j,:n];v=Y[j,n:n+nv];mem=[Y[j,s] for s in model.slices]
        T=invert(E/model.mass,model.cp,model.latent,model.ts,model.tl,model.offset)
        cap=model.mass*(model.cp+model.latent/(model.tl-model.ts)*((T>model.ts)&(T<model.tl)))
        for i,r in enumerate(model.regions):
            if r=='mush' and T[i] in (model.ts[i],model.tl[i]):cap[i]=model.mass[i]*(model.cp[i]+model.latent[i]/(model.tl[i]-model.ts[i]))
        Tdot=K[j,:n]/cap
        q,f=model.ports(t+C[j]*h,T,v,mem);q=np.array(q,float);force=np.array(f,float)-model.D@v
        external+=h*B[j]*float(v@f);heat+=h*B[j]*float(np.sum(q));drag+=h*B[j]*float(v@model.D@v)
        for a,b,g in model.edges:
            flux=g*(T[a]-T[b]);q[a]-=flux;q[b]+=flux
            node_flux[a]-=h*B[j]*flux;node_flux[b]+=h*B[j]*flux
        defects=[]
        for m,e,s in zip(model.blocks,mem,model.slices):
            Q=m.relaxation*float(e@m.H@e);W=-float(e@m.H@m.expansion)*Tdot[m.node]
            maxwell+=h*B[j]*Q;thermal+=h*B[j]*W
            q[m.node]+=Q-W;local_heat[m.node]+=h*B[j]*(Q-W)
            force-=m.B.T@m.H@e
            defects.extend(K[j,s]-(m.B@v-m.expansion*Tdot[m.node]-m.relaxation*e))
        defects=np.concatenate((K[j,:n]-q,model.M@K[j,n:n+nv]-force,defects))
        rate_residuals.append(defects)
        impulse+=h*B[j]*force
        stage_defects.append(Y[j]-before-h*np.sum(A[j,:,None]*K,axis=0))
    numerical_kinetic=numerical_memory=0.
    for i in range(2):
        for j in range(2):
            w=.5*h*h*ENERGY_MATRIX[i,j]
            numerical_kinetic+=w*float(K[i,n:n+nv]@model.M@K[j,n:n+nv])
            numerical_memory+=w*sum(float(K[i,s]@m.H@K[j,s]) for m,s in zip(model.blocks,model.slices))
    numerical=numerical_kinetic+numerical_memory
    after=result['after'];delta=model.energy(after)-model.energy(before)
    impulse_residual=model.M@(after[n:n+nv]-before[n:n+nv])-impulse
    # No conversion of signed RK defect into heat. Drag remains external storage.
    return dict(energy_change_j=delta,external_work_j=external,source_heat_j=heat,
        physical_drag_j=drag,physical_maxwell_j=maxwell,thermal_work_to_mechanics_j=thermal,
        thermal_feedback_j=local_heat.tolist(),conduction_node_j=node_flux.tolist(),
        numerical_kinetic_j=numerical_kinetic,numerical_memory_j=numerical_memory,
        signed_numerical_j=numerical,balance_residual_j=delta-external-heat+drag+numerical,
        impulse_residual=impulse_residual.tolist(),stage_equation_defects=np.array(stage_defects).tolist(),
        independent_rate_defects=np.array(rate_residuals).tolist())

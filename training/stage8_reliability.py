"""C1: numerical-reliability weights from the SAME three FD stencils.
No extra queries, parameters, return-based tuning, or changed evaluation masks.
"""
import numpy as np
from training.rollout import physical_return

def labels_with_reliability(env,starts,thetas,horizon=8,eps=.05):
 labels=[];weighted=[];steps=0
 for st,theta in zip(starts,thetas):
  th=np.asarray(theta);vals=[]
  for scale in [.5,1.,2.]:
   g=[]
   for j in range(28):
    p=th.copy();m=th.copy();p[j]=min(1,th[j]+eps*scale);m[j]=max(-1,th[j]-eps*scale)
    g.append((physical_return(env,p,st,horizon)[0]-physical_return(env,m,st,horizon)[0])/(p[j]-m[j]));steps+=2*horizon
   vals.append(g)
  vals=np.array(vals);g=vals[1];spread=np.ptp(vals,axis=0);mask=spread<=np.maximum(1e-9,.1*abs(g));s0=float(np.median(abs(g[mask]))) if mask.any() else 1.
  weights=np.clip(1/(1+(spread/np.maximum(1e-9,.1*abs(g)))**2),.25,1.)
  lab=dict(start=st,theta=th,g=g,mask=mask.astype(float),s0=s0 or 1.,control_steps=1344,fd_scales=vals,weights=weights)
  labels.append(lab);weighted.append({**lab,'mask':mask.astype(float)*weights})
 return labels,weighted,steps

"""Predeclared numerical stability; never chooses epsilon by return."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np,torch
from scripts.run_stage7c1_diagnosis import make_env
from training.stage8_1_geometry import Geometry
from training.stage8_2_dynamics import Kernel,Budget,N,TARGET
from training.rollout import actor_raw_numpy
from training.eval_metrics import gradient_quality
from analysis.stage8_2_run import physical,matrix
P=ROOT/'runs/stage8_2a_single_step';O=ROOT/'runs/stage8_2_numerical'
def dump(n,o):(O/n).write_text(json.dumps(o,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def main():
 O.mkdir(exist_ok=False);t0=time.time();env=make_env();geo=Geometry(env.model);budget=Budget(15000);data=torch.load(ROOT/'runs/stage7c2_true_ee_current_sites_seed20/data.pt',weights_only=False);base=json.loads((P/'references.json').read_text());single=json.loads((P/'single_step.json').read_text());multi=json.loads((P/'multistep.json').read_text());local=[];fds=[];quality=[]
 for i,s in enumerate(data['test_starts']):
  k=Kernel(env,s['snapshot'],budget);z=k.initial.astype(np.float32);a=np.clip(actor_raw_numpy(np.zeros(28),k.initial[:28],k.initial[TARGET],s['refs'][0]),-.03,.03).astype(np.float32);jp=np.array(single[i]['J_physical'])
  for scale in [.5,2.]:
   j=k.jacobian(z,a,0,scale);local.append(dict(start=i,scale=scale,full=matrix(j,jp),A=matrix(j[:28,:28],jp[:28,:28]),B=matrix(j[:28,N:],jp[:28,N:]),hidden=matrix(j[:28,70:N],jp[:28,70:N])))
  gs=[np.array(base[i]['gradients'])]
  for eps in [.005,.02]:
   g=np.zeros((8,28))
   for d in range(28):
    p=np.zeros(28);m=p.copy();p[d]=eps;m[d]=-eps;g[:,d]=(physical(k,p,s,geo)[1]-physical(k,m,s,geo)[1])/(2*eps)
   gs.append(g);fds.append(dict(start=i,eps=eps,g=g))
  gs=np.stack(gs);mask=np.ptp(gs,axis=0)<=np.maximum(1e-9,.1*np.abs(gs[0]))
  for r in multi:
   if r['start']!=i:continue
   h=r['horizon'];met=gradient_quality(r['g'],r['g_true'],mask[h-1]);quality.append(dict(start=i,condition=r['condition'],horizon=h,metrics=met,stable_mask=mask[h-1].astype(int),boundary_mask=[0]*28,fd_mode='central',epsilons=[.005,.01,.02],source_reference='stage8_2a_single_step/references.json'))
 dump('local_epsilon.json',local);dump('policy_epsilon.json',fds);dump('validated_metrics.json',quality);dump('budget.json',dict(control_steps=budget.steps,wall_seconds=time.time()-t0,geometry_calls=geo.calls));print('DONE',budget.steps,flush=True)
if __name__=='__main__':main()

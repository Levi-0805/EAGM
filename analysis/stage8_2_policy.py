"""One matched frozen-model Adam update; oracle costs capped at 20k steps."""
import sys,json,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np,torch
from scripts.run_stage7c1_diagnosis import make_env
from models.eagm import EAGMBase
from tasks.reach import FKSurrogate
from training.stage8_1_geometry import Geometry
from training.stage8_2_dynamics import Kernel,Budget,rollout
from analysis.stage8_2_run import physical
from training.eval_metrics import cosine
P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20';OUT=ROOT/'runs/stage8_2d_frozen_policy'
def dump(n,o):(OUT/n).write_text(json.dumps(o,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def main():
 assert json.loads((ROOT/'runs/stage8_2a_single_step/gate.json').read_text())['pass_D']
 OUT.mkdir(exist_ok=False);t0=time.time();budget=Budget(20000);data=torch.load(P/'data.pt',weights_only=False);env=make_env();geo=Geometry(env.model);b=torch.load(P/'dir_random.pt',weights_only=False);model=EAGMBase(b['stats']);model.load_state_dict(b['state']);model.eval();fk=FKSurrogate();fk.load_state_dict(torch.load(P/'fk.pt',weights_only=True));fk.eval()
 for p in list(model.parameters())+list(fk.parameters()):p.requires_grad_(False)
 starts=data['train_starts'][:2];kernels=[Kernel(env,s['snapshot'],budget) for s in starts];policies={}
 for c in 'ABCD':
  th=torch.zeros(28,requires_grad=True);opt=torch.optim.Adam([th],lr=.03);loss=-torch.stack([rollout(model,th,s,c,fk,geo,k) for s,k in zip(starts,kernels)]).mean();loss.backward();g=-th.grad.detach().numpy().copy();torch.nn.utils.clip_grad_norm_([th],1.);opt.step()
  with torch.no_grad():th.clamp_(-1,1)
  policies[c]=dict(theta=th.detach().numpy(),training_gradient=g,training_forward=float(-loss.detach()),theta_saturation=float((th.detach().abs()>=1-1e-8).float().mean()),rho_proj=1.,updates=1,train_episode_ids=[s['episode_id'] for s in starts]);dump('policies.json',policies)
 print('training steps',budget.steps,flush=True);train_steps=budget.steps;tests=data['test_starts'];evalkernels=[Kernel(env,s['snapshot'],budget) for s in tests];r0=[physical(k,np.zeros(28),s,geo,H=16)[1][-1] for k,s in zip(evalkernels,tests)];rows=[];finalgrads=[]
 for c in 'ABCD':
  th=policies[c]['theta'].astype(float)
  for i,(s,k) in enumerate(zip(tests,evalkernels)):
   ret=physical(k,th,s,geo,H=16)[1][-1];g=np.zeros(28)
   for j in range(28):
    p=th.copy();m=th.copy();p[j]+=.01;m[j]-=.01;g[j]=(physical(k,p,s,geo)[1][-1]-physical(k,m,s,geo)[1][-1])/.02
   rows.append(dict(condition=c,start=i,episode=s['episode_id'],theta0_return=r0[i],true_return=ret,delta=ret-r0[i],failed=int(ret<r0[i]-1e-6),update_vs_final_true_gradient_cosine=cosine(th,g),final_true_gradient_norm=float(np.linalg.norm(g))))
   finalgrads.append(dict(condition=c,start=i,g_true_final=g,epsilon=.01,fd_mode='central',stable_mask=None,stability_status='single-scale; not estimated',boundary_mask=[0]*28))
  print(c,'steps',budget.steps,flush=True);dump('per_start.json',rows);dump('final_true_gradients.json',finalgrads)
 with (OUT/'per_start.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={c:dict(true_return=float(np.mean([r['true_return'] for r in rows if r['condition']==c])),delta=float(np.mean([r['delta'] for r in rows if r['condition']==c])),failrate=float(np.mean([r['failed'] for r in rows if r['condition']==c])),success_rate=None,theta_saturation=policies[c]['theta_saturation']) for c in 'ABCD'}
 dump('summary.json',summary);dump('budget.json',dict(control_steps=budget.steps,training_control_steps=train_steps,evaluation_and_final_fd_steps=budget.steps-train_steps,limit=20000,wall_seconds=time.time()-t0,geometry_calls=geo.calls,device='cpu',cache='shared exact state/action queries, all methods same stencils and updates',success_rate_status='no preregistered task threshold; not fabricated',final_gradient_status='direct true FD, single-scale; initial full/stable cosines in ABC'))
 print('DONE',summary,'steps',budget.steps,flush=True)
if __name__=='__main__':main()

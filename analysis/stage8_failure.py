"""Low-cost numerical/geometry diagnosis; no model fitting or policy tuning."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np
import torch
from tasks.reach import FKSurrogate,ee_from_q_mujoco
from scripts.run_stage7c1_diagnosis import make_env,model_grad
from training.gradients import sim_grad_true_ee_masked_detailed
from training.rollout import set_reach_fk_surrogate,physical_return_true_ee,physical_return_surrogate
from models.eagm import EAGMBase,McWorld
from training.eval_metrics import gradient_quality
P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20';OUT=ROOT/'runs/stage8_failure'
def main():
 OUT.mkdir(exist_ok=False);t=time.time();d=torch.load(P/'data.pt',weights_only=False);fk=FKSurrogate();fk.load_state_dict(torch.load(P/'fk.pt',weights_only=True));set_reach_fk_surrogate(fk);env=make_env()
 pol=json.loads((P/'policies.json').read_text());audit=[];steps=0
 for i,st in enumerate(d['test_starts']):
  for tag,th in [('zero',np.zeros(28)),('dir_final',np.array(pol['dir_random']['theta']))]:
   for ep in [.001,.003,.01,.03]:
    g,m,n,meta=sim_grad_true_ee_masked_detailed(env,st,th,eps=ep,horizon=8);steps+=n
    audit.append(dict(start=i,theta=tag,eps=ep,g=g.tolist(),mask=m.tolist(),meta=meta))
  print('fd start',i,flush=True)
 def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
 dump('fd.json',audit)
 # deterministic replay noise, reference eps for comparison never selected on return
 g1,m,n,_=sim_grad_true_ee_masked_detailed(env,d['test_starts'][0],np.zeros(28),eps=.01,horizon=8);steps+=n
 repeat=max(abs(g1-np.array(next(x['g'] for x in audit if x['start']==0 and x['theta']=='zero' and x['eps']==.01))))
 geom=[];rng=np.random.default_rng(808)
 for split,ids in [('train',d['split'].train_eps),('validation',d['split'].val_eps),('test',d['split'].test_eps)]:
  rows=[r for e in d['episodes'] if e['episode_id'] in ids for r in e['rows']]
  for idx in rng.choice(len(rows),min(200,len(rows)),replace=False):
   q=rows[idx]['x'][:14];l,r=ee_from_q_mujoco(env,q);true=np.r_[l,r];pred=fk(torch.tensor(q,dtype=torch.float32)).detach().numpy();geom.append(dict(split=split,q=np.array(q).tolist(),true=true.tolist(),pred=pred.tolist(),l2=float(np.linalg.norm(pred-true))))
 dump('geometry.json',geom)
 path=[]
 for name in ['mc','base','dir_random','aux']:
  for index in [0,9,24,49]:
   curve=pol[name]['curve'][index];th=np.array(curve['theta'])
   for i,st in enumerate(d['test_starts']):
    rt=physical_return_true_ee(env,th,st,16)[0];rs=physical_return_surrogate(env,th,st,16)[0];steps+=32
    path.append(dict(model=name,update=index+1,start=i,true=rt,surrogate=rs,imagined_train=curve['imagined_return'],sat=curve['theta_param_sat_frac'],rho=curve['rho_proj']))
 dump('path.json',path)
 # Command and state one/multi-step errors on fixed held-out observed actions.
 dynamics=[]
 from training.rollout import c_obs_step,c_full_step
 for name,cls in [('mc',McWorld),('base',EAGMBase),('dir_random',EAGMBase)]:
  blob=torch.load(P/(name+'.pt'),weights_only=False);model=cls(blob['stats']);model.load_state_dict(blob['state']);model.eval()
  for e in d['episodes']:
   if e['episode_id'] not in d['split'].test_eps:continue
   rr=e['rows'];x=torch.tensor(rr[0]['x'],dtype=torch.float32);c=torch.tensor(rr[0]['c' if name=='mc' else 'c_obs'],dtype=torch.float32)
   with torch.no_grad():
    for k in range(16):
     a=torch.tensor(rr[k]['a'],dtype=torch.float32)
     if name=='mc':y=model(x,c,a);u=None;c=c_full_step(c,a,model.lower,model.upper)
     else:y,u=model(x[None],c[None],a[None]);y=y[0];u=u[0];c=c_obs_step(c,a,model.lower,model.upper,float(model.alpha))
     dynamics.append(dict(model=name,episode=e['episode_id'],horizon=k+1,nmse=float((((y-torch.tensor(rr[k]['y']))/model.ystd)**2).mean()),command_mae=None if u is None else float(abs(u-torch.tensor(rr[k]['u_exec'])).mean())))
     x=y
 dump('multistep.json',dynamics)
 dump('budget.json',dict(control_steps=steps,wall_seconds=time.time()-t,device='cpu',fd_queries=(len(audit)+1)*168,geometry_queries=len(geom),repeat_max_abs=repeat,seed=808))
 print('DONE',steps,repeat,flush=True)
if __name__=='__main__':main()

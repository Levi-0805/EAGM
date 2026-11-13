"""Geometry/Jacobian causal audit. Hard 20k control-step cap, no WM training."""
import sys,json,time,csv,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np
import torch
import mujoco
from scripts.run_stage7c1_diagnosis import make_env
from models.eagm import EAGMBase,McWorld
from tasks.reach import FKSurrogate
from training.stage8_1_geometry import Geometry,rollout
from training.gradients import sim_grad_true_ee_masked_detailed
from training.rollout import physical_return_true_ee
from training.eval_metrics import gradient_quality,cosine
P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20';OUT=ROOT/'runs/stage8_1_geometry'
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def grad(model,st,th,condition,fk,geo,is_mc=False):
 theta=torch.tensor(th,dtype=torch.float32,requires_grad=True);j=rollout(model,theta,st,condition,fk,geo,is_mc=is_mc);g=torch.autograd.grad(j,theta)[0].detach().numpy();return float(j.detach()),g

def main():
 OUT.mkdir(exist_ok=False);t0=time.time();cfg=json.loads((ROOT/'configs/stage8_1_protocol.json').read_text());dump('config.json',cfg);torch.manual_seed(8101);rng=np.random.default_rng(8101)
 env=make_env();geo=Geometry(env.model);steps=0;phase='static';counts={}
 original=env.step
 def counted(a):
  nonlocal steps
  if steps>=20000:raise RuntimeError('20,000 physical control-step cap reached')
  steps+=1;counts[phase]=counts.get(phase,0)+1;return original(a)
 env.step=counted
 data=torch.load(P/'data.pt',weights_only=False);starts=data['test_starts'];train=data['train_starts'];oldpol=json.loads((P/'policies.json').read_text());fk=FKSurrogate();fk.load_state_dict(torch.load(P/'fk.pt',weights_only=True));fk.eval()
 for p in fk.parameters():p.requires_grad_(False)
 models={}
 for name,cls in [('mc',McWorld),('base',EAGMBase),('dir_random',EAGMBase)]:
  b=torch.load(P/(name+'.pt'),weights_only=False);m=cls(b['stats']);m.load_state_dict(b['state']);m.eval()
  for p in m.parameters():p.requires_grad_(False)
  models[name]=m
 samples=[]
 for split,ids in [('train',data['split'].train_eps),('test',data['split'].test_eps)]:
  rr=[r for e in data['episodes'] if e['episode_id'] in ids for r in e['rows']]
  for idx in rng.choice(len(rr),100,replace=False):
   r=rr[idx];samples.append(dict(group=split,episode=r['episode_id'],frame=r['frame'],q=np.asarray(r['x'][:14]),goals=r['ee_goals']))
 for i in range(100):samples.append(dict(group='random_legal',episode=None,frame=i,q=rng.uniform(env.lower,env.upper),goals=starts[i%len(starts)]['ee_goals']))
 for tag,th in [('theta0_imagined',np.zeros(28)),('old_final_imagined',np.array(oldpol['dir_random']['theta']))]:
  for st in starts:
   _,ss=rollout(models['dir_random'],torch.tensor(th,dtype=torch.float32),st,'A',fk,geo,return_states=True)
   for k,x in enumerate(ss):samples.append(dict(group=tag,episode=st['episode_id'],frame=k,q=x[:14],goals=st['ee_goals']))
 static=[]
 for s in samples:
  q=np.array(s['q']);pt,jt=geo.evaluate(q);qt=torch.tensor(q,dtype=torch.float32,requires_grad=True);ps=fk(qt).detach().numpy();js=torch.autograd.functional.jacobian(fk,qt).detach().numpy();goal=np.asarray(s['goals']);gt=-jt.T@(pt-goal)/3;gs=-js.T@(ps-goal)/3
  # Factorial decomposition distinguishes geometric J error from residual-position error.
  only_j=-jt.T@(ps-goal)/3;only_p=-js.T@(pt-goal)/3
  static.append({**s,'p_true':pt,'p_fk':ps,'J_true':jt,'J_fk':js,'g_true':gt,'g_fk':gs,'g_trueJ_fkPosition':only_j,'g_fkJ_truePosition':only_p,'position_rmse':float(np.sqrt(np.mean((pt-ps)**2))),'position_mae':float(np.mean(abs(pt-ps))),'jacobian_rmse':float(np.sqrt(np.mean((jt-js)**2))),'jacobian_relative':float(np.linalg.norm(jt-js)/(np.linalg.norm(jt)+1e-12)),'jacobian_cosine':cosine(jt,js),'reward_gradient_cosine':cosine(gt,gs),'reward_gradient_relative':float(np.linalg.norm(gt-gs)/(np.linalg.norm(gt)+1e-12)),'position_value_true':float(-np.mean((pt-goal)**2)),'position_value_fk':float(-np.mean((ps-goal)**2)),'out_of_limits':bool(np.any(q<env.lower)|np.any(q>env.upper))})
 dump('static.json',static)
 checks=[]
 for s in samples[200:205]:
  q=np.asarray(s['q']);_,j=geo.evaluate(q)
  for ep in [1e-5,1e-4,1e-3]:
   fd=np.column_stack([(geo.evaluate(q+np.eye(14)[i]*ep)[0]-geo.evaluate(q-np.eye(14)[i]*ep)[0])/(2*ep) for i in range(14)])
   checks.append(dict(eps=ep,max_abs=float(np.max(abs(fd-j))),rmse=float(np.sqrt(np.mean((fd-j)**2)))))
 dump('geometry_fd_checks.json',checks);dump('geometry_metadata.json',geo.metadata())
 tr=[s for s in static if s['group']=='train'];gate=(np.mean([s['reward_gradient_cosine'] for s in tr])<.99 or np.mean([s['jacobian_relative'] for s in tr])>.05) and max(x['max_abs'] for x in checks)<1e-5
 dump('gate.json',dict(passed=bool(gate),train_reward_cosine=float(np.mean([s['reward_gradient_cosine'] for s in tr])),control_steps=steps));assert steps==0
 print('STATIC gate',gate,flush=True)
 # State-level reward derivative checks for all terms: native qdot penalty and action penalty analytic.
 from training.stage8_1_geometry import reward
 state_checks=[]
 for st in starts:
  x=torch.tensor(st['x'],dtype=torch.float64,requires_grad=True);a=torch.tensor(st['prev'],dtype=torch.float64,requires_grad=True);prev=a.detach()+.01
  # true position custom function supports float64; surrogate is float32 so use a double copy
  import copy
  fk64=copy.deepcopy(fk).double();r=reward('B',x,a,prev,st['ee_goals'],fk64,geo);gx,ga=torch.autograd.grad(r,(x,a));pt,jt=geo.evaluate(x[:14].detach().numpy());expected=-jt.T@(pt-np.asarray(st['ee_goals']))/3
  state_checks.append(dict(q_grad_max_error=float(np.max(abs(gx[:14].numpy()-expected))),qdot_grad_max_error=float(np.max(abs(gx[14:].numpy()+.001*x[14:].detach().numpy()/14))),action_grad_max_error=float(np.max(abs(ga.numpy()+.1*(a-prev).detach().numpy()/14)))))
 dump('reward_derivative_checks.json',state_checks)
 if not gate:
  dump('decision.json',dict(case='C',stop=True));return
 phase='theta0_fd';references=[]
 for i,st in enumerate(starts):
  g,m,n,meta=sim_grad_true_ee_masked_detailed(env,st,np.zeros(28),eps=.01,horizon=8,scales=(.5,1,2));references.append(dict(start=i,g=g,mask=m,meta=meta))
 dump('theta0_references.json',references)
 intervention=[]
 for name,m in models.items():
  for i,st in enumerate(starts):
   for condition in 'ABCD':
    value,g=grad(m,st,np.zeros(28),condition,fk,geo,is_mc=name=='mc');ref=references[i];metrics=gradient_quality(g,ref['g'],ref['mask']);metrics['relative_error']=float(np.linalg.norm(g-ref['g'])/(np.linalg.norm(ref['g'])+1e-12));intervention.append(dict(model=name,start=i,condition=condition,value=value,g=g,metrics=metrics))
 dump('intervention.json',intervention);print('INTERVENTION complete steps',steps,flush=True)
 # Keep identical model/actor/starts/Adam and projected update rule; ONLY reward intervention differs.
 policies={};m=models['dir_random']
 for condition in 'ABCD':
  th=torch.zeros(28,requires_grad=True);opt=torch.optim.Adam([th],lr=.03);curve=[]
  for u in range(50):
   before=th.detach().numpy().copy();loss=-torch.stack([rollout(m,th,st,condition,fk,geo) for st in train]).mean();opt.zero_grad();loss.backward();g=-th.grad.detach().numpy().copy();torch.nn.utils.clip_grad_norm_([th],1.);opt.step()
   with torch.no_grad():raw=th.detach().numpy().copy();th.clamp_(-1,1);after=th.detach().numpy().copy()
   curve.append(dict(update=u+1,theta=after,gradient=g,return_before=float(-loss.detach()),sat=float(np.mean(abs(after)>=1-1e-8)),rho=float(np.linalg.norm(raw-after)/(np.linalg.norm(raw-before)+1e-12)),update_gradient_cosine=cosine(after-before,g)))
  policies[condition]=dict(theta=th.detach().numpy().copy(),curve=curve);dump('policies.json',policies);print('policy',condition,flush=True)
 assert np.array_equal(policies['A']['theta'],policies['D']['theta'])
 assert np.array_equal(policies['B']['theta'],policies['C']['theta'])
 phase='physical_evaluation';per=[];r0=[]
 for st in starts:r0.append(physical_return_true_ee(env,np.zeros(28),st,horizon=16)[0])
 for c in 'ABCD':
  for i,st in enumerate(starts):
   r=physical_return_true_ee(env,policies[c]['theta'],st,horizon=16)[0];per.append(dict(condition=c,start=i,episode=st['episode_id'],true_return=r,theta0_return=r0[i],delta=r-r0[i]))
 with (OUT/'per_start.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
 phase='final_fd';final=[]
 for representative,conditions in [('A','AD'),('B','BC')]:
  th=policies[representative]['theta']
  for i,st in enumerate(starts):
   gt,mask,n,meta=sim_grad_true_ee_masked_detailed(env,st,th,eps=.01,horizon=8,scales=(.5,1.))
   for c in conditions:
    _,gh=grad(m,st,th,c,fk,geo);metrics=gradient_quality(gh,gt,mask);metrics['relative_error']=float(np.linalg.norm(gh-gt)/(np.linalg.norm(gt)+1e-12));final.append(dict(condition=c,start=i,g=gh,g_true=gt,mask=mask,meta=meta,metrics=metrics))
 dump('final_gradients.json',final)
 assert steps==19296
 summary={}
 for c in 'ABCD':
  rr=[r for r in per if r['condition']==c];ii=[r['metrics'] for r in intervention if r['condition']==c and r['model']=='dir_random'];ff=[r['metrics'] for r in final if r['condition']==c]
  summary[c]=dict(true_return=float(np.mean([r['true_return'] for r in rr])),delta=float(np.mean([r['delta'] for r in rr])),fail_rate=float(np.mean([r['delta']<0 for r in rr])),theta0_metrics={k:float(np.nanmean([r[k] for r in ii])) for k in ii[0]},final_metrics={k:float(np.nanmean([r[k] for r in ff])) for k in ff[0]},theta_sat=policies[c]['curve'][-1]['sat'])
 dump('summary.json',summary);dump('budget.json',dict(physical_steps=steps,by_phase=counts,limit=20000,geometry=geo.metadata(),wall_seconds=time.time()-t0,torch=torch.__version__,mujoco=mujoco.__version__,cpu=platform.processor()))
 dump('manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(OUT.glob('*'))+[P/'dir_random.pt',P/'base.pt',P/'mc.pt',P/'fk.pt',P/'data.pt',ROOT/'training/stage8_1_geometry.py',Path(__file__)] if p.is_file()})
 print('DONE',summary,flush=True)
if __name__=='__main__':main()

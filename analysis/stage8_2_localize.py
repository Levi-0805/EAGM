"""Supplemental factor localization and optimizer/horizon checks; no training."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np,torch,mujoco
from scripts.run_stage7c1_diagnosis import make_env
from models.eagm import EAGMBase
from training.stage8_1_geometry import Geometry
from training.stage8_2_dynamics import Kernel,Budget,N,TARGET
from training.rollout import actor_raw_numpy
from analysis.stage8_2_run import physical,matrix
P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20';O=ROOT/'runs/stage8_2_localization_v2'
def dump(n,o):(O/n).write_text(json.dumps(o,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def main():
 O.mkdir(exist_ok=False);t0=time.time();env=make_env();geo=Geometry(env.model);data=torch.load(P/'data.pt',weights_only=False);b=torch.load(P/'dir_random.pt',weights_only=False);model=EAGMBase(b['stats']);model.load_state_dict(b['state']);model.eval();ab=Budget(40000-21777);rows=[]
 for i,s in enumerate(data['test_starts']):
  k=Kernel(env,s['snapshot'],ab);z=k.initial;a=np.clip(actor_raw_numpy(np.zeros(28),z[:28],z[TARGET],s['refs'][0]),-.03,.03);out=k.forward(z,a,0);u=env.last_true_command.copy();v=np.r_[z[:28],u]
  def phys(v):
   env.restore(s['snapshot']);env.data.qpos[:]=v[:14];env.data.qvel[:]=v[14:28];env.data.ctrl[:]=v[28:];ab.tick();mujoco.mj_step(env.model,env.data,nstep=env.substeps);return env.observe().copy()
  def learn(v):
   x=v[:28];u=v[28:];return x+model.net(torch.cat([(x-model.xmean)/model.xstd,(u-model.umean)/model.ustd]))*model.ystd+model.ymean
  yp=phys(v);assert np.max(abs(yp-out[:28]))<1e-10
  jm=torch.autograd.functional.jacobian(learn,torch.tensor(v,dtype=torch.float32)).detach().numpy();jp=np.empty((28,42))
  for j in range(42):
   h=1e-4 if j<28 else 1e-5;p=v.copy();m=v.copy();p[j]+=h;m[j]-=h;jp[:,j]=(phys(p)-phys(m))/(2*h)
  y=learn(torch.tensor(v,dtype=torch.float32)).detach().numpy();rows.append(dict(start=i,A=matrix(jm[:,:28],jp[:,:28]),B_executed_command=matrix(jm[:,28:],jp[:,28:]),J_model=jm,J_physical=jp,true_command=u,forward_delta_normalized_mse=float(np.mean(((y-yp)/model.ystd.numpy())**2)),position_state=z[:14]))
 dump('direct_actuator_dynamics.json',rows);abcsteps=ab.steps
 # Independent optimization diagnostic: same saved policies, no additional updates.
 prior=json.loads((ROOT/'runs/stage8_2d_frozen_policy/budget.json').read_text());db=Budget(20000-prior['control_steps']);pol=json.loads((ROOT/'runs/stage8_2d_frozen_policy/policies.json').read_text());evals=[];grads=[]
 for group,starts in [('train',data['train_starts'][:2]),('test',data['test_starts'])]:
  for i,s in enumerate(starts):
   k=Kernel(env,s['snapshot'],db)
   for c in ['theta0','A','B','C','D']:
    th=np.zeros(28) if c=='theta0' else np.array(pol[c]['theta']);ret=physical(k,th,s,geo,H=16)[1];evals.append(dict(group=group,start=i,episode=s['episode_id'],condition=c,return_h8=ret[7],return_h16=ret[15]))
   if group=='train':
    g=np.zeros(28)
    for j in range(28):
     p=np.zeros(28);m=p.copy();p[j]=.01;m[j]=-.01;g[j]=(physical(k,p,s,geo)[1][-1]-physical(k,m,s,geo)[1][-1])/.02
    grads.append(dict(start=i,g_true_h8_theta0=g))
 dump('optimizer_horizon.json',evals);dump('train_true_gradients.json',grads);dump('budget.json',dict(ABC_extra_steps=abcsteps,D_extra_steps=db.steps,D_total_steps=prior['control_steps']+db.steps,wall_seconds=time.time()-t0,geometry_calls=geo.calls))
 print('DONE',abcsteps,db.steps,prior['control_steps']+db.steps,flush=True)
if __name__=='__main__':main()

"""A-C diagnostic, independent scratch data, complete execution memory."""
import sys,json,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import torch,numpy as np
from scripts.run_stage7c1_diagnosis import make_env
from models.eagm import EAGMBase
from tasks.reach import FKSurrogate
from training.stage8_1_geometry import Geometry
from training.stage8_2_dynamics import Kernel,Budget,transition,rollout,N,TARGET
from training.rollout import actor_raw_numpy
from training.eval_metrics import cosine,gradient_quality
P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20';OUT=ROOT/'runs/stage8_2a_single_step'
def dump(n,o):(OUT/n).write_text(json.dumps(o,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def matrix(a,b):
 den=np.linalg.norm(b)
 return dict(relative=None if den<1e-12 else float(np.linalg.norm(a-b)/den),cosine=None if den<1e-12 or np.linalg.norm(a)<1e-12 else cosine(a,b),true_norm=float(den),model_norm=float(np.linalg.norm(a)),singular_values_model=np.linalg.svd(a,compute_uv=False),singular_values_true=np.linalg.svd(b,compute_uv=False),absolute_error=float(np.linalg.norm(a-b)))
def physical(k,th,st,geo,H=8):
 z=k.initial.copy();prev=np.asarray(st['prev']);states=[];rs=[];total=0.
 for t in range(H):
  a=np.clip(actor_raw_numpy(th,z[:28],z[TARGET],st['refs'][t]),-.03,.03);z=k.forward(z,a,t);p,_=geo.evaluate(z[:14]);r=-np.mean((p-st['ee_goals'])**2)-.0005*np.mean(z[14:28]**2)-.05*np.mean((a-prev)**2);total+=.99**t*r;states.append(z[:28].copy());rs.append(total);prev=a
 return np.array(states),np.array(rs)
def main():
 OUT.mkdir(exist_ok=False);t0=time.time();budget=Budget(40000);data=torch.load(P/'data.pt',weights_only=False);env=make_env();geo=Geometry(env.model);b=torch.load(P/'dir_random.pt',weights_only=False);model=EAGMBase(b['stats']);model.load_state_dict(b['state']);model.eval();fk=FKSurrogate();fk.load_state_dict(torch.load(P/'fk.pt',weights_only=True));fk.eval()
 for p in list(model.parameters())+list(fk.parameters()):p.requires_grad_(False)
 singles=[];multis=[];inter=[];refs=[];numerical=[];kernels=[]
 for i,st in enumerate(data['test_starts']):
  k=Kernel(env,st['snapshot'],budget);kernels.append(k);z=torch.tensor(k.initial,dtype=torch.float32);th=torch.zeros(28,requires_grad=True);a=torch.tensor(np.clip(actor_raw_numpy(np.zeros(28),k.initial[:28],k.initial[TARGET],st['refs'][0]),-.03,.03),dtype=torch.float32)
  jp=k.jacobian(z.numpy(),a.numpy(),0);jm=torch.autograd.functional.jacobian(lambda v:transition(model,v[:N],v[N:],0,k,False),torch.cat([z,a])).numpy()
  ym=transition(model,z,a,0,k,False).detach().numpy();yp=k.forward(z.numpy(),a.numpy(),0)
  singles.append(dict(start=i,episode=st['episode_id'],A=matrix(jm[:28,:28],jp[:28,:28]),B_action=matrix(jm[:28,N:],jp[:28,N:]),C_observed=matrix(jm[:28,42:70],jp[:28,42:70]),hidden=matrix(jm[:28,70:N],jp[:28,70:N]),warm=matrix(jm[:28,28:42],jp[:28,28:42]),forward_nmse=float(np.mean(((ym[:28]-yp[:28])/model.ystd.numpy())**2)),J_model=jm,J_physical=jp))
  if i==0:
   for scale in [.5,2.]:numerical.append(dict(type='local_epsilon',scale=scale,metrics=matrix(k.jacobian(z.numpy(),a.numpy(),0,scale),jp)))
   # Model-only central FD through SAME forward values: local directional derivative.
   rng=np.random.default_rng(82);v=rng.normal(size=N+14);v/=np.linalg.norm(v);h=.003;inp=np.r_[z.numpy(),a.numpy()]
   plus=transition(model,torch.tensor((inp+h*v)[:N],dtype=torch.float32),torch.tensor((inp+h*v)[N:],dtype=torch.float32),0,k,False).detach().numpy()
   minus=transition(model,torch.tensor((inp-h*v)[:N],dtype=torch.float32),torch.tensor((inp-h*v)[N:],dtype=torch.float32),0,k,False).detach().numpy()
   # Hidden detached outputs are not differentiable native inputs; check x only.
   numerical.append(dict(type='model_local_direction',relative=float(np.linalg.norm((plus[:28]-minus[:28])/(2*h)-jm[:28]@v)/(np.linalg.norm(jm[:28]@v)+1e-12))))
  true_states,true_ret=physical(k,np.zeros(28),st,geo);sfd=np.zeros((8,28,28));gfd=np.zeros((8,28))
  for j in range(28):
   p=np.zeros(28);m=p.copy();p[j]=.01;m[j]=-.01;sp,rp=physical(k,p,st,geo);sm,rm=physical(k,m,st,geo);sfd[:,:,j]=(sp-sm)/.02;gfd[:,j]=(rp-rm)/.02
  refs.append(dict(start=i,gradients=gfd,state_sensitivity=sfd,true_states=true_states,true_returns=true_ret))
  for cond in 'ABCD':
   th=torch.zeros(28,requires_grad=True);j,states,returns=rollout(model,th,st,cond,fk,geo,k,details=True)
   values=[]
   for h in [1,2,4,8]:
    gh=torch.autograd.grad(returns[h-1],th,retain_graph=True)[0].numpy();sens=np.stack([torch.autograd.grad(states[h-1][d],th,retain_graph=True)[0].numpy() for d in range(28)])
    q=gradient_quality(gh,gfd[h-1],np.ones(28));q['relative_error']=float(np.linalg.norm(gh-gfd[h-1])/(np.linalg.norm(gfd[h-1])+1e-12))
    multis.append(dict(start=i,condition=cond,horizon=h,metrics=q,g=gh,g_true=gfd[h-1],sensitivity_relative=float(np.linalg.norm(sens-sfd[h-1])/(np.linalg.norm(sfd[h-1])+1e-12)),sensitivity=sens,state_error=float(np.linalg.norm(states[h-1][:28].detach().numpy()-true_states[h-1])),state=states[h-1][:28].detach().numpy(),forward_value=float(returns[h-1].detach())))
   inter.append(multis[-1]);values.append(float(j.detach()))
  # Strict equality of forward values/states, independent of replacements.
  same=[r for r in multis if r['start']==i and r['horizon']==8];assert all(np.array_equal(same[0]['state'],r['state']) for r in same);assert max(r['forward_value'] for r in same)-min(r['forward_value'] for r in same)<1e-10
  print('start',i,'steps',budget.steps,flush=True);dump('single_step.json',singles);dump('multistep.json',multis);dump('references.json',refs)
 # Actual-forward oracle chain check at predefined first start; separates linearization from actor bugs.
 k=kernels[0];st=data['test_starts'][0];th=torch.zeros(28,dtype=torch.float32,requires_grad=True);j=rollout(model,th,st,'D',fk,geo,k,physical_forward=True);gh=torch.autograd.grad(j,th)[0].numpy();gt=np.array(refs[0]['gradients'][7]);q=gradient_quality(gh,gt,np.ones(28));q['relative_error']=float(np.linalg.norm(gh-gt)/(np.linalg.norm(gt)+1e-12));numerical.append(dict(type='physical_chain_actor_check',g=gh,g_true=gt,metrics=q))
 gains=[next(r['metrics']['cosine_full'] for r in inter if r['start']==i and r['condition']=='D')-next(r['metrics']['cosine_full'] for r in inter if r['start']==i and r['condition']=='B') for i in range(6)]
 gate=q['cosine_full']>.99 and np.mean(gains)>.1 and sum(v>0 for v in gains)>=4
 dump('numerical_checks.json',numerical);dump('gate.json',dict(pass_D=bool(gate),D_minus_B=gains,chain_cosine=q['cosine_full'],epsilon_selection='fixed a priori; no return tuning'))
 dump('budget.json',dict(control_steps=budget.steps,limit=40000,wall_seconds=time.time()-t0,geometry_calls=geo.calls,device='cpu',state_dim=N,discrete_context='delay/B parameters/queue occupancy fixed from snapshot'))
 print('DONE gate',gate,'gains',gains,'chain',q,'steps',budget.steps,flush=True)
if __name__=='__main__':main()

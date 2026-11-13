"""Privileged full execution-state FD oracle, diagnostic only."""
import copy
import mujoco
import numpy as np
import torch
from training.rollout import actor,c_obs_step
from training.stage8_1_geometry import TrueFK
from training.reward import DISCOUNT,W_VEL,W_DA
N=154
X=slice(0,28);WARM=slice(28,42);TARGET=slice(42,56);FILT=slice(56,70);QUEUE=slice(70,98);PREV=slice(98,112);SMOOTH=slice(112,126);EXTRA=slice(126,154)

class Kernel:
 def __init__(self,env,snapshot,budget):
  self.env=env;self.snap=copy.deepcopy(snapshot);self.budget=budget;self.cache={};self.jcache={}
  assert env.model.na==0 and env.model.nq==env.model.nv==14
  self.lag=int(snapshot['controller_b']['active']['extra_lag']);assert self.lag<=2
  self.initial_len=len(snapshot['controller_b']['extra_queue']);env.restore(snapshot)
  self.time=float(env.data.time);self.initial=self.pack()
 def pack(self):
  e=self.env;extra=np.zeros((2,14))
  for i,v in enumerate(e._extra_queue):extra[i]=v
  return np.r_[e.observe(),e.data.qacc_warmstart,e.qtarget,e.filtered,e.queue.ravel(),e._u_prev,e._u_smooth,extra.ravel()]
 def forward(self,z,a,k):
  z=np.asarray(z,dtype=np.float64);a=np.asarray(a,dtype=np.float64);key=(k,z.tobytes(),a.tobytes())
  if key in self.cache:return self.cache[key].copy()
  e=self.env;e.restore(self.snap);e.data.qpos[:]=z[:14];e.data.qvel[:]=z[14:28];e.data.qacc_warmstart[:]=z[WARM];e.data.time=self.time+k*e.control_dt
  e.qtarget=z[TARGET].copy();e.filtered=z[FILT].copy();e.queue=z[QUEUE].reshape(2,14).copy();e._u_prev=z[PREV].copy();e._u_smooth=z[SMOOTH].copy();e._extra_queue=[v.copy() for v in z[EXTRA].reshape(2,14)[:min(self.lag,self.initial_len+k)]]
  self.budget.tick();e.step(a);out=self.pack();self.cache[key]=out.copy();return out
 def jacobian(self,z,a,k,scale=1.):
  z=np.asarray(z,dtype=np.float64);a=np.asarray(a,dtype=np.float64);key=(k,z.tobytes(),a.tobytes(),scale)
  if key in self.jcache:return self.jcache[key]
  v=np.r_[z,a];eps=np.r_[np.full(28,1e-4),np.full(14,1e-3),np.full(112,1e-5),np.full(14,1e-5)]*scale
  j=np.empty((N,N+14))
  for i,h in enumerate(eps):
   p=v.copy();m=v.copy();p[i]+=h;m[i]-=h
   j[:,i]=(self.forward(p[:N],p[N:],k)-self.forward(m[:N],m[N:],k))/(2*h)
  self.jcache[key]=j;return j

class ReplaceJacobian(torch.autograd.Function):
 @staticmethod
 def forward(ctx,z,a,value,j):ctx.save_for_backward(j);return value.clone()
 @staticmethod
 def backward(ctx,g):
  j,=ctx.saved_tensors;v=j.T@g;return v[:N],v[N:],None,None

def transition(model,z,a,k,kernel,oracle=False):
 raw=kernel.forward(z.detach().numpy(),a.detach().numpy(),k);v=torch.as_tensor(raw,dtype=z.dtype)
 c=torch.cat([z[TARGET],z[FILT],z.new_tensor([kernel.snap['delay']])]);y,_=model(z[X][None],c[None],a[None]);cn=c_obs_step(c,a,model.lower,model.upper,float(model.alpha))
 # Hidden values common to all variants; native model cannot use hidden state.
 tc=cn[:28]+(v[42:70]-cn[:28]).detach()
 out=torch.cat([y[0],v[WARM],tc,v[70:]])
 if oracle:
  j=torch.tensor(kernel.jacobian(z.detach().numpy(),a.detach().numpy(),k),dtype=z.dtype);out=ReplaceJacobian.apply(z,a,out.detach(),j)
 return out

def rollout(model,theta,start,condition,fk,geo,kernel,horizon=8,physical_forward=False,details=False):
 z=torch.tensor(kernel.initial,dtype=theta.dtype);prev=torch.tensor(start['prev'],dtype=theta.dtype);total=theta.sum()*0;states=[];returns=[]
 for k in range(horizon):
  a=actor(theta,z[X],z[TARGET],torch.tensor(start['refs'][k],dtype=theta.dtype))
  if physical_forward:
   raw=kernel.forward(z.detach().numpy(),a.detach().numpy(),k);j=kernel.jacobian(z.detach().numpy(),a.detach().numpy(),k)
   z=ReplaceJacobian.apply(z,a,torch.tensor(raw,dtype=theta.dtype),torch.tensor(j,dtype=theta.dtype))
  else:z=transition(model,z,a,k,kernel,condition in 'CD')
  goal=torch.tensor(start['ee_goals'],dtype=theta.dtype);rs=-((fk(z[:14])-goal)**2).mean();rt=-((TrueFK.apply(z[:14],geo)-goal)**2).mean()
  pos=rs if condition in 'AC' else rs.detach()+(rt-rt.detach())
  total=total+DISCOUNT**k*(pos-W_VEL*z[14:28].square().mean()-W_DA*(a-prev).square().mean());prev=a;states.append(z);returns.append(total)
 return (total,states,returns) if details else total

class Budget:
 def __init__(self,limit):self.limit=limit;self.steps=0
 def tick(self):
  if self.steps>=self.limit:raise RuntimeError('control-step budget exceeded')
  self.steps+=1

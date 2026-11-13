"""Frozen first-order MuJoCo geometry and explicit reward-gradient interventions.
No live simulation state mutation and no new learned parameters.
"""
import time
import mujoco
import numpy as np
import torch
from training.reward import W_VEL,W_DA,DISCOUNT
from training.rollout import actor,c_obs_step,c_full_step

class Geometry:
 def __init__(self,model):
  self.model=model;self.data=mujoco.MjData(model);self.calls=0;self.seconds=0.
  assert model.nq==model.nv==model.njnt==14
  assert np.all(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
  assert np.array_equal(model.jnt_qposadr,np.arange(14))
  assert np.array_equal(model.jnt_dofadr,np.arange(14))
  self.sites=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,n) for n in ['left_ee_control_point','right_ee_control_point']]
  assert min(self.sites)>=0
 def evaluate(self,q):
  t=time.perf_counter();self.data.qpos[:]=np.asarray(q,dtype=np.float64);mujoco.mj_forward(self.model,self.data)
  ps=[];js=[]
  for site in self.sites:
   jp=np.zeros((3,14));jr=np.zeros((3,14));mujoco.mj_jacSite(self.model,self.data,jp,jr,site);ps.append(self.data.site_xpos[site].copy());js.append(jp)
  self.calls+=1;self.seconds+=time.perf_counter()-t
  return np.concatenate(ps),np.concatenate(js)
 def metadata(self):
  return dict(nq=self.model.nq,nv=self.model.nv,njnt=self.model.njnt,sites=self.sites,joints=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(14)],qpos_addresses=self.model.jnt_qposadr.tolist(),dof_addresses=self.model.jnt_dofadr.tolist(),units='positions: world-frame metres; hinge q: radians; Jacobian m/rad',calls=self.calls,seconds=self.seconds)

class TrueFK(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,geometry):
  p,j=geometry.evaluate(q.detach().cpu().numpy());ctx.save_for_backward(torch.as_tensor(j,dtype=q.dtype,device=q.device));return torch.as_tensor(p,dtype=q.dtype,device=q.device)
 @staticmethod
 def backward(ctx,grad):
  j,=ctx.saved_tensors
  return j.T@grad,None

def reward(condition,x,a,prev,goals,fk,geometry):
 goals=torch.as_tensor(goals,dtype=x.dtype,device=x.device)
 rs=-((fk(x[:14])-goals)**2).mean()
 rt=-((TrueFK.apply(x[:14],geometry)-goals)**2).mean()
 if condition=='A':pos=rs
 elif condition=='B':pos=rt
 elif condition=='C':pos=rs.detach()+(rt-rt.detach())
 elif condition=='D':pos=rt.detach()+(rs-rs.detach())
 else:raise ValueError(condition)
 return pos-W_VEL*x[14:].square().mean()-W_DA*(a-prev).square().mean()

def rollout(model,theta,start,condition,fk,geometry,horizon=8,is_mc=False,return_states=False):
 x=torch.as_tensor(start['x'],dtype=theta.dtype);c=torch.as_tensor(start['c' if is_mc else 'c_obs'],dtype=theta.dtype);prev=torch.as_tensor(start['prev'],dtype=theta.dtype);total=theta.sum()*0;states=[]
 for k in range(horizon):
  a=actor(theta,x,c[:14],torch.as_tensor(start['refs'][k],dtype=theta.dtype))
  if is_mc:y=model(x,c,a);c=c_full_step(c,a,model.lower,model.upper)
  else:y,_=model(x[None],c[None],a[None]);y=y[0];c=c_obs_step(c,a,model.lower,model.upper,float(model.alpha))
  total=total+DISCOUNT**k*reward(condition,y,a,prev,start['ee_goals'],fk,geometry)
  states.append(y.detach().numpy());x=y;prev=a
 return (total,states) if return_states else total

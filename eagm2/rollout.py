"""Explicit causal imagined/physical objectives with query accounting."""
import time
import numpy as np
import torch
from training.rollout import actor,actor_raw_numpy
from training.reward import reward_torch,reward_numpy
from eagm2.prior import observable_step

def tensor(v,model):return torch.as_tensor(v,dtype=next(model.parameters()).dtype,device=next(model.parameters()).device)
def imagined(model,theta,start,horizon,*,reward_fn=None,vision_policy=None):
    x=tensor(start['x'],model);c=tensor(start['c'],model);fp=tensor(start['fprev'],model);prev=tensor(start['prev'],model)
    refs=tensor(start['refs'][:horizon],model)
    h=model.burnin(tensor(start['xhist'],model).unsqueeze(0),tensor(start['chist'],model).unsqueeze(0),tensor(start['ahist'],model).unsqueeze(0)).squeeze(0) if model.memory!='none' else None
    total=theta.sum()*0
    for k in range(horizon):
        a=actor(theta,x,c[:14],refs[k]) if vision_policy is None else vision_policy.action_from_latent(x,c,theta)
        y,u,h=model.step(x,c,a,fp,h)
        r=reward_torch(y,a,refs[k],prev) if reward_fn is None else reward_fn(y,a,prev)
        total=total+.99**k*r
        c,fp=observable_step(c,a,model.lower,model.upper,float(model.alpha))
        x=y;prev=a
    return total

def physical(env,theta,start,horizon,budget,kind,*,task='tracking',vision_policy=None,return_details=False):
    begin=time.perf_counter();steps_completed=0;env.restore(start['snapshot']);prev=np.asarray(start['prev']).copy();total=0.;reward_list=[]
    if vision_policy is not None:
        dev=next(vision_policy.parameters()).device;th=torch.as_tensor(theta,dtype=torch.float32,device=dev)
    try:
        for k in range(horizon):
            x=env.observe();c=env.observed_control_state()
            if vision_policy is None:a=np.clip(actor_raw_numpy(np.asarray(theta),x,c[:14],start['refs'][k]),-.03,.03)
            else:
                rgb=env.render_rgb()
                with torch.no_grad():a=vision_policy(torch.as_tensor(rgb,device=dev),torch.as_tensor(c,dtype=torch.float32,device=dev).unsqueeze(0),th).squeeze(0).cpu().numpy()
            y=env.step(a);steps_completed+=1
            r=reward_numpy(y,a,start['refs'][k],prev) if task=='tracking' else env.reward(a,prev)
            total+=.99**k*r;reward_list.append(float(r));prev=a
    finally:
        budget.add(kind,steps_completed,time.perf_counter()-begin)
    if return_details:
        return dict(return_=total,rewards=reward_list,**({} if task=='tracking' else env.metrics()))
    return float(total)

def gradient(model,start,theta,horizon,**kwargs):
    th=tensor(theta,model).clone().requires_grad_(True)
    return torch.autograd.grad(imagined(model,th,start,horizon,**kwargs),th)[0].detach().cpu().numpy()

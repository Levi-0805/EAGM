"""New research models. No simulator snapshot or hidden B state in deployable API."""
from __future__ import annotations
import torch
from torch import nn
from .prior import nominal_command, observable_step


def _mlp(inp, width, out):
    return nn.Sequential(nn.Linear(inp,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU(), nn.Linear(width,out))


class ExecutionWorldModel(nn.Module):
    """Command bottleneck with optional causal execution memory.

    memory='none': feedforward; 'recurrent': posterior Q / open-loop prior G;
    'window': corrected finite-window GRU, recomputed with predicted observations.
    x is deployable observation (physical state or visual latent), never snapshot.
    Recurrent actor inputs are outside this module; model memory is not actor input.
    """
    def __init__(self, stats, obs_dim=28, width=160, memory_dim=32,
                 memory='none', prior='original', residual=True, history_len=8,
                 max_increment=0.03):
        super().__init__()
        if memory not in ('none','recurrent','window') or prior not in ('original','delay','free'):
            raise ValueError('unknown model variant')
        self.obs_dim,self.width,self.memory_dim = obs_dim,width,memory_dim
        self.memory,self.prior,self.residual = memory,prior,residual
        self.history_len,self.max_increment=history_len,max_increment
        self.is_privileged=False
        defaults={'xmean':torch.zeros(obs_dim),'xstd':torch.ones(obs_dim),
                  'ymean':torch.zeros(obs_dim),'ystd':torch.ones(obs_dim),
                  'umean':torch.zeros(14),'ustd':torch.ones(14),
                  'lower':torch.full((14,),-3.),'upper':torch.full((14,),3.),'alpha':torch.tensor(.5)}
        for key, default in defaults.items():
            self.register_buffer(key,torch.as_tensor(stats.get(key,default),dtype=torch.float32).clone())
        for key in ('xmean','xstd','ymean','ystd'):
            if getattr(self,key).shape != (obs_dim,): raise ValueError(f'{key} observation dimension mismatch')
        if (self.xstd<=0).any() or (self.ystd<=0).any() or (self.ustd<=0).any():
            raise ValueError('normalization scales must be positive')
        mem = memory_dim if memory!='none' else 0
        if memory!='none':
            # Cells, rather than fused cuDNN GRU, support second derivatives.
            self.posterior = nn.GRUCell(obs_dim+29+14,memory_dim)
            if memory=='recurrent': self.transition = nn.GRUCell(29+14+14,memory_dim)
        self.command_net = nn.Sequential(nn.Linear(29+14+mem,64),nn.SiLU(),nn.Linear(64,14))
        if prior!='free':
            nn.init.zeros_(self.command_net[-1].weight);nn.init.zeros_(self.command_net[-1].bias)
        self.dynamics = _mlp(obs_dim+14,width,obs_dim)
        if not residual and prior!='free':
            for p in self.command_net.parameters():p.requires_grad_(False)

    def normalized_control(self,c):
        scale=(self.upper-self.lower).clamp_min(.1)
        center=(self.upper+self.lower)/2
        return torch.cat(((c[...,:14]-center)/scale,(c[...,14:28]-center)/scale,c[...,28:29]/2),-1)

    def record(self,x,c,prev_a):
        return torch.cat(((x-self.xmean)/self.xstd,self.normalized_control(c),prev_a/self.max_increment),-1)

    def initial_state(self,x):
        if self.memory=='none':return None
        if self.memory=='window':return x.new_zeros((*x.shape[:-1],self.history_len,self.obs_dim+43))
        return x.new_zeros((*x.shape[:-1],self.memory_dim))

    def observe(self,h_prior,x,c,prev_a):
        if self.memory=='none':return None
        if h_prior is None:h_prior=self.initial_state(x)
        e=self.record(x,c,prev_a)
        if self.memory=='window':return torch.cat((h_prior[...,1:,:],e.unsqueeze(-2)),-2)
        shape=e.shape[:-1]
        return self.posterior(e.reshape(-1,e.shape[-1]),h_prior.reshape(-1,self.memory_dim)).reshape(*shape,self.memory_dim)

    def burnin(self,xhist,chist,ahist,detach=True):
        """Aligned histories [...,T,D], a[i] is action preceding x[i]/c[i]."""
        if xhist.shape[-2] != chist.shape[-2] or xhist.shape[-2] != ahist.shape[-2]:
            raise ValueError('history lengths must match')
        if xhist.shape[-2]==0:raise ValueError('at least one public observation required')
        h=self.initial_state(xhist[...,0,:])
        for t in range(xhist.shape[-2]):
            # Propagate with the preceding issued action, then condition on the
            # current observable record. The first record starts a declared
            # reset/burn-in window; no hidden simulator state initializes h.
            if t and self.memory=='recurrent':
                prev_filter=chist[...,max(0,t-2),14:28]
                _,_,h=self.step(xhist[...,t-1,:],chist[...,t-1,:],ahist[...,t,:],prev_filter,h)
            h=self.observe(h,xhist[...,t,:],chist[...,t,:],ahist[...,t,:])
        return h.detach() if detach and h is not None else h

    def _encoded_memory(self,h,x):
        if self.memory=='none':return None
        if h is None:raise ValueError('memory model needs causal public burn-in')
        if self.memory=='recurrent':return h
        hidden=x.new_zeros((*x.shape[:-1],self.memory_dim))
        shape=x.shape[:-1]
        for t in range(h.shape[-2]):
            e=h[...,t,:]
            hidden=self.posterior(e.reshape(-1,e.shape[-1]),hidden.reshape(-1,self.memory_dim)).reshape(*shape,self.memory_dim)
        return hidden

    def command(self,x,c,a,f_prev,h=None):
        mem=self._encoded_memory(h,x)
        features=[self.normalized_control(c),a/self.max_increment]
        if mem is not None:features.append(mem)
        residual=self.command_net(torch.cat(features,-1))*self.ustd
        if self.prior=='free':return residual+self.umean
        if self.prior=='delay':u0=nominal_command(c,a,f_prev,self.lower,self.upper,self.alpha,self.max_increment)
        else:u0=observable_step(c,a,self.lower,self.upper,self.alpha,self.max_increment)[0][...,14:28]
        return u0+residual if self.residual else u0

    def step(self,x,c,a,f_prev,h=None):
        u=self.command(x,c,a,f_prev,h)
        inputs=torch.cat(((x-self.xmean)/self.xstd,(u-self.umean)/self.ustd),-1)
        newx=x+self.dynamics(inputs)*self.ystd+self.ymean
        next_h=None
        if self.memory=='recurrent':
            e=torch.cat((self.normalized_control(c),a/self.max_increment,(u-self.umean)/self.ustd),-1)
            shape=x.shape[:-1]
            next_h=self.transition(e.reshape(-1,e.shape[-1]),h.reshape(-1,self.memory_dim)).reshape(*shape,self.memory_dim)
        elif self.memory=='window':
            next_c,_=observable_step(c,a,self.lower,self.upper,self.alpha,self.max_increment)
            next_h=self.observe(h,newx,next_c,a)
        return newx,u,next_h

    forward=step

    def parameter_counts(self):
        return {'total':sum(p.numel() for p in self.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}


class PrivilegedExecutionWorldModel(ExecutionWorldModel):
    """Explicit extra-information reference; no deployable method accepts it.

    Does not assert an optimal performance upper bound or exact hidden dynamics.
    A caller must obtain/train with privileged_state separately; not a rollout
    drop-in substitute because future privileged state cannot be observed.
    """
    def __init__(self,stats,obs_dim=28,width=160,privileged_dim=74):
        super().__init__(stats,obs_dim=obs_dim,width=width,memory='none',prior='delay')
        self.is_privileged=True;self.privileged_dim=privileged_dim
        self.privileged_command=_mlp(29+14+privileged_dim,64,14)
        for p in self.command_net.parameters():p.requires_grad_(False)

    def step_privileged(self,x,c,a,f_prev,*,privileged_state):
        if privileged_state.shape[-1]!=self.privileged_dim:raise ValueError('privileged dimension')
        inp=torch.cat((self.normalized_control(c),a/self.max_increment,privileged_state),-1)
        u=nominal_command(c,a,f_prev,self.lower,self.upper,self.alpha)+self.privileged_command(inp)*self.ustd
        newx=x+self.dynamics(torch.cat(((x-self.xmean)/self.xstd,(u-self.umean)/self.ustd),-1))*self.ystd+self.ymean
        return newx,u

    def step(self,*args,**kwargs):
        raise TypeError('privileged reference requires explicit step_privileged; no deployable rollout')
    forward=step


def capacity_matched_width(stats,target_count,*,obs_dim=28,memory='none',prior='delay',memory_dim=32,tolerance=.05):
    """Choose width by parameter count only; no performance tuning/RNG mutation."""
    choices=[]
    with torch.random.fork_rng(devices=[]):
        for width in range(8,385):
            model=ExecutionWorldModel(stats,obs_dim=obs_dim,width=width,memory=memory,prior=prior,memory_dim=memory_dim)
            count=model.parameter_counts()['trainable']
            choices.append((abs(count-target_count),width,count))
    _,width,count=min(choices)
    if abs(count-target_count)/target_count>tolerance:
        raise ValueError('no capacity match within tolerance; choose explicit different memory size')
    return width,count
